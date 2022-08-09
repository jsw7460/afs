import time
import os
import pickle
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.noise import ActionNoise

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.jax_layers import FlattenExtractor
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import (
    GymEnv,
    Schedule
)
from offline_baselines_jax.common.utils import get_basic_rngs
from .buffers.prtr_buffer import VideoDataset
from .core import encoder_update
from .networks import ImgVariationalAutoEncoder
from .networks import (
    LSTMSubTrajectoryLastObsBasedSkillGenerator,
    MLPSkillPrior
)
from .policies import LowLevelSkillPolicy, SkillBasedComposedPolicy
from .sopt_skill_prior import SOPTSkillEmpowered
from .utils import clock

EPS = 1E-5


@jax.jit
def encode_images(rng: jnp.ndarray, encoder: Model, images: jnp.ndarray):
    rng, _ = jax.random.split(rng)
    mean, log_std = encoder.apply_fn(
        {"params": encoder.params},
        images,
        method=ImgVariationalAutoEncoder.get_latent_params
    )
    return rng, mean


class ImgBasedSkillPrior(SOPTSkillEmpowered):

    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str] = SkillBasedComposedPolicy,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: int = 0,
        _init_setup_model: bool = True,
        without_exploration: bool = False,
        ###
        dropout: float = 0.0,
        n_frames: int = 3,
        subseq_len: int = 10,
        batch_dim: int = 256,
        hidden_dim: int = 128,
        skill_dim: int = 5,

        model_archs: Optional[Dict[str, List]] = {},
        bc_reg_coef: float = 0.5,

        expert_dataset_load_interval: int = 500_000,
    ):
        super(ImgBasedSkillPrior, self).__init__(
            env=env,
            policy=policy,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            _init_setup_model=_init_setup_model,
            without_exploration=without_exploration,
            dropout=dropout,
            n_frames=n_frames,
            subseq_len=subseq_len,
            batch_dim=batch_dim,
            hidden_dim=hidden_dim,
            skill_dim=skill_dim,
            model_archs=model_archs,
            bc_reg_coef=bc_reg_coef,
            expert_dataset_load_interval=expert_dataset_load_interval
        )

        self.encoder: Model = None      # Image is preprocessed by the encoder
        self.img_embed_dim: int = -1
        self.encoder_applied_observation_space: gym.spaces.Space = None     # Encoder applied된 후에 state를 정의
        self.iterable_expert_buffer = None

    def build_hrl_models(self, hrl_config: Dict) -> Tuple[int, int]:
        raise NotImplementedError()

    def set_image_expert_buffer(self, path: str, cfg: Dict):
        from spirl.utils.general_utils import AttrDict
        """For image dataset, we use pytorch dataloader for saving the memory."""
        dataset_specs = AttrDict(**(cfg["dataset_spec"]), device="cpu")
        self.expert_buffer = VideoDataset(
            path, dataset_specs, resolution=32, n_frames=self.n_frames, **cfg["kwargs"]
        ).get_data_loader(**cfg["loader_kwargs"])
        self.iterable_expert_buffer = iter(self.expert_buffer)

    def load_next_expert_buffer(self):
        pass

    @property
    def skill_prior_model_save_interval(self):
        return 10_000

    @contextmanager
    def encoder_learning_phase(self):
        _train = self._train
        _offline_train = self._offline_train
        _without_exploration = self.without_exploration
        self._train = None
        self._offline_train = self.encoder_train
        self.without_exploration = True
        yield

        self._train = _train
        self._offline_train = _offline_train
        self.without_exploration = _without_exploration

    def build_lowlevel_policy(self) -> Dict:
        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor_class = FlattenExtractor

        init_obs = self.encoder_applied_observation_space.sample()[np.newaxis, ...]
        obs_dim = init_obs.shape[-1]

        features_extractor = features_extractor_class(_observation_space=self.encoder_applied_observation_space)
        lowlevel_policy_kwargs = {
            "features_extractor": features_extractor,
            "observation_space": self.encoder_applied_observation_space,
            # lowlevel action dim = obs dim. We have no info about action even dimension.
            "lowlevel_action_dim": obs_dim,
            # "lowlevel_action_dim": self.action_space.shape[-1],         # For real action skill extraction debug
            "net_arch": self.model_archs["lowlevel_policy"],
            "dropout": 0.1
        }
        lowlevel_policy_def = LowLevelSkillPolicy(**lowlevel_policy_kwargs)
        init_skill = jnp.zeros((1, self.skill_dim))

        self.lowlevel_policy = Model.create(
            lowlevel_policy_def,
            inputs=[rngs, init_obs, init_skill],
            tx=optax.adam(1e-4)
        )

        return lowlevel_policy_kwargs

    def build_skill_generator(self) -> Dict:
        features_extractor_class = FlattenExtractor
        init_obs = self.encoder_applied_observation_space.sample()[np.newaxis, ...]
        obs_dim = init_obs.shape[-1]

        self.rng, rngs = get_basic_rngs(self.rng)

        init_key, _ = jax.random.split(self.rng)
        rngs.update({"init": init_key})
        features_extractor = features_extractor_class(_observation_space=self.encoder_applied_observation_space)
        self.batch_dim = self.batch_size
        skill_generator_kwargs = {
            "features_extractor": features_extractor,
            "observation_space": self.encoder_applied_observation_space,
            "subseq_len": self.subseq_len,
            "batch_dim": self.batch_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.skill_dim,
            "dropout": self.dropout,
            "net_arch": self.model_archs["skill_generator"]
        }
        skill_generator_def = LSTMSubTrajectoryLastObsBasedSkillGenerator(**skill_generator_kwargs)
        # init_observations = jax.random.normal(init_key, shape=(1, self.subseq_len, obs_dim))
        init_observations = np.random.normal(size=(1, self.subseq_len, obs_dim))
        init_actions = np.random.normal(size=(1, self.subseq_len, obs_dim))

        self.skill_generator = Model.create(
            skill_generator_def,
            inputs=[rngs, init_observations, init_actions, init_obs],
            tx=optax.adam(self.learning_rate)
        )
        return skill_generator_kwargs

    def build_skill_prior(self) -> Dict:
        features_extractor_class = FlattenExtractor
        init_obs = self.encoder_applied_observation_space.sample()[np.newaxis, ...]

        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = features_extractor_class(_observation_space=self.encoder_applied_observation_space)
        skill_prior_kwargs = {
            "features_extractor": features_extractor,
            "observation_space": self.encoder_applied_observation_space,
            "latent_dim": 64,
            "skill_dim": self.skill_dim,
            "dropout": self.dropout,
            "net_arch": self.model_archs["skill_prior"]
        }
        skill_prior_def = MLPSkillPrior(**skill_prior_kwargs)

        self.skill_prior = Model.create(
            skill_prior_def,
            inputs=[rngs, init_obs],
            tx=optax.adam(learning_rate=self.learning_rate)
        )
        return skill_prior_kwargs

    def build_image_encoder(self, cfg: Dict):
        try:
            with open(cfg["config_dir"] + "encoder_config", "rb") as f:
                encoder_kwargs = pickle.load(f)
            observation_space = encoder_kwargs["observation_space"]
            encoder_def = ImgVariationalAutoEncoder(**encoder_kwargs)
            require_encoder_train = False

        except FileNotFoundError:
            observation_space = gym.spaces.Box(
                -float('inf'),
                float('inf'),
                shape=self.observation_space["image"].shape,
                dtype=np.float32,
            )
            encoder_kwargs = {
                "observation_space": observation_space,
                "feature_dim": cfg["sqrt_feature_dim"] ** 2,
            }
            encoder_def = ImgVariationalAutoEncoder(**encoder_kwargs)
            require_encoder_train = True

        init_image = observation_space.sample()[np.newaxis, ...]
        self.rng, rngs = get_basic_rngs(self.rng)
        rngs.update({"latent_sampling": self.rng})
        encoder = Model.create(encoder_def, inputs=[rngs, init_image], tx=optax.adam(1e-4))

        if not require_encoder_train:
            try:
                encoder_path = cfg["model_dir"] + "image_encoder_100000"
                encoder = encoder.load_dict(encoder_path)
                print(f"Encoder is loaded from {encoder_path}\n" * 10)
            except FileNotFoundError:
                require_encoder_train = True

        self.encoder = encoder
        self.encoder_applied_observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(cfg["sqrt_feature_dim"] ** 2, ),
            dtype=np.float32
        )
        self.img_embed_dim = cfg["sqrt_feature_dim"] ** 2
        return require_encoder_train, encoder_kwargs

    def build_skill_prior_models(self, skill_prior_config: Dict) -> Tuple[bool, int]:
        if not skill_prior_config["build"]: return 0        # Return required training step

        # Define: A dictionary which saves a parameters of pretrained model. This will be saved with model together.
        pretrained_kwargs = {}

        """Image based model requires pretrained encoder model"""
        # Define: Image encoder model
        require_encoder_train, encoder_kwargs = self.build_image_encoder(skill_prior_config["encoder_config"])

        if require_encoder_train:
            # Save encoder kwargs
            os.makedirs(skill_prior_config["encoder_config"]["config_dir"], exist_ok=True)
            encoder_config_path = skill_prior_config["encoder_config"]["config_dir"] + "encoder_config"
            with open(encoder_config_path, "wb") as f:
                pickle.dump(encoder_kwargs, f)

            print(f"Encoder config is saved in {encoder_config_path}\n" * 10)

        # Define: LowLevel Skill Policy (Learning with pseudo actions (e.g., delta sequence))
        lowlevel_policy_kwargs = self.build_lowlevel_policy()
        # Define: skill prior
        skill_generator_kwargs = self.build_skill_generator()
        # Save kwargs of pretrained models
        skill_prior_kwargs = self.build_skill_prior()

        pretrained_kwargs.update({
            "lowlevel_policy": lowlevel_policy_kwargs,
            "skill_generator": skill_generator_kwargs,
            "skill_prior": skill_prior_kwargs,
            "normalizing_max": None,
            "normalizing_min": None
        })

        # Save skill prior kwargs
        os.makedirs(skill_prior_config["config_save_dir"], exist_ok=True)
        config_path = skill_prior_config["config_save_dir"] + "config"
        with open(config_path, "wb") as f:
            pickle.dump(pretrained_kwargs, f)
        print(f"Config saved in {config_path}\n" * 10)

        self.skill_prior_model_save_dir = skill_prior_config["model_save_dir"]

        return require_encoder_train, skill_prior_config["total_timesteps"]

    def get_encoder_training_input(self, replay_data: Dict):
        images = replay_data["images"]
        stacked_image = np.concatenate(
            [images[:, t:t+replay_data["actions"].shape[1]] for t in range(self.n_frames)],
            axis=-1,
        ) / 255 * 2 - 1
        img_shape = stacked_image.shape[-3: ]
        stacked_image = stacked_image.reshape(-1, *img_shape)
        return stacked_image

    # @clock(fmt="[{name}: {elapsed: 0.8f}s]")
    def get_skill_prior_training_input(self):
        """
        :return:
            observations: [batch_size, subseq_len, dim]
            actions: [batch_size, subseq_len, dim]
            last_observations: [batch_size, dim]
        """
        assert self.encoder is not None

        # Define observations
        try:
            replay_data = next(self.iterable_expert_buffer)
        except StopIteration:
            self.iterable_expert_buffer = iter(self.expert_buffer)
            replay_data = next(self.iterable_expert_buffer)

        images = np.array(replay_data["images"])
        stacked_image = np.concatenate(
            [images[:, t:t + replay_data["actions"].shape[1]] for t in range(self.n_frames)],
            axis=-1,
        ) / 255 * 2 - 1

        img_shape = stacked_image.shape[-3:]
        stacked_image = stacked_image.reshape(-1, *img_shape)    # Applied to batch: [batch * (subseq_len+1), img_shape]
        self.rng, observations = encode_images(self.rng, self.encoder, stacked_image)
        observations = np.array(observations)

        observations = observations.reshape(self.batch_size, -1, self.img_embed_dim)  # [batch, subseq_len'+1', embed_dim]

        # Define pseudo actions
        pseudo_actions = (observations[:, 1:, ...] - observations[:, :-1, ...])

        # Define last observations
        last_images = np.array(replay_data["last_images"])
        stacked_last_image = np.concatenate(
            [last_images[:, t: t+1] for t in range(self.n_frames)],
            axis=-1
        ) / 255 * 2 - 1
        stacked_last_image = np.squeeze(stacked_last_image, axis=1)
        self.rng, last_observations = encode_images(self.rng, self.encoder, stacked_last_image)

        # Fix and check the length
        observations = observations[:, :self.subseq_len, ...]

        # Note: We must normalize the pseudo actions along the batch.
        print("?>?", pseudo_actions.shape)
        exit()
        reshaped_actions = pseudo_actions.reshape(pseudo_actions.shape[0] * pseudo_actions.shape[1], -1)
        max_act = reshaped_actions.max(axis=0, keepdims=True)
        min_act = reshaped_actions.min(axis=0, keepdims=True)

        reshaped_actions = 2 * ((reshaped_actions - min_act) / (max_act - min_act + EPS)) - 1
        pseudo_actions = reshaped_actions.reshape(pseudo_actions.shape[0], pseudo_actions.shape[1], -1)
        pseudo_actions = np.clip(pseudo_actions, -1+1E-4, 1-1E-4)

        assert pseudo_actions.shape[1] == self.subseq_len
        return observations, pseudo_actions, last_observations

    def encoder_train(self, *_, **__):
        image_encoder_losses = []
        recon_losses, kl_losses = [], []
        means, log_stds = [], []

        timestep = 0
        for epoch in range(100):
            for replay_data in self.expert_buffer:
                images = self.get_encoder_training_input(replay_data)
                self.rng, self.encoder, infos = encoder_update(self.rng, self.encoder, images, 0.001)
                timestep += 1

                image_encoder_losses.append(infos["encoder_loss"])
                recon_losses.append(infos["recon_loss"])
                kl_losses.append(infos["kl_loss"])
                means.append(infos["mean"])
                log_stds.append(infos["log_std"])

                if timestep % 50 == 0:

                    self.logger.record_mean("encoder/loss", np.mean(image_encoder_losses))
                    self.logger.record_mean("encoder/recon_loss", np.mean(recon_losses))
                    self.logger.record_mean("encoder/kl_loss", np.mean(kl_losses))
                    self.logger.record_mean("encoder/mu", np.mean(means))
                    self.logger.record_mean("encoder/log_std", np.mean(log_stds))

                    image_encoder_losses = []
                    recon_losses, kl_losses = [], []
                    means, log_stds = [], []

                if timestep % 100 == 0:
                    self.logger.record("current_step", timestep)
                    self.logger.dump(timestep)

                if timestep % 10000 == 0:
                    self.encoder.save_dict(self.skill_prior_model_save_dir + f"image_encoder_{timestep}")
