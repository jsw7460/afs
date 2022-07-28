import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import jax
import jax.numpy as jnp
import numpy as np
from stable_baselines3.common.noise import ActionNoise

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import (
    GymEnv,
    Schedule
)
from offline_baselines_jax.common.utils import get_basic_rngs
from .networks import ImgVariationalAutoEncoder
from .policies import SkillBasedComposedPolicy
from .sopt_skill_prior_hrl import SkillBasedHRLAgent

EPS = 1E-8


@jax.jit
def _get_representation(encoder: Model, image: jnp.ndarray):
    image = (image / 255.) * 2 - 1
    mean, log_std = encoder.apply_fn({"params": encoder.params}, image, method=ImgVariationalAutoEncoder.get_latent_params)
    return mean


class ImgBasedHRLAgent(SkillBasedHRLAgent):
    """
    Image based HRL Agent.
    """
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
        use_hiro_relabel: bool = True,

        model_archs: Optional[Dict[str, List]] = {},
        bc_reg_coef: float = 0.5,

        higher_ent_coef: Union[str, float] = "auto",
        lower_ent_coef: Union[str, float] = "auto",
        higher_target_entropy: Union[str, float] = "auto",
        lower_target_entropy: Union[str, float] = "auto",
    ):
        super(ImgBasedHRLAgent, self).__init__(
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
            use_hiro_relabel=use_hiro_relabel,
            model_archs=model_archs,
            bc_reg_coef=bc_reg_coef,
            higher_ent_coef=higher_ent_coef,
            lower_ent_coef=lower_ent_coef,
            higher_target_entropy=higher_target_entropy,
            lower_target_entropy=lower_target_entropy
        )
        self.representation_dim: int = 0        # Lower policy's image representation space

    @property
    def higher_observation_space(self) -> gym.spaces.Space:     # Concatenation of representation & state
        return gym.spaces.Box(-float('inf'), float('inf'), (self.representation_dim,))

    @property  # For lower policy
    def lower_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-float('inf'), float('inf'), (self.representation_dim,))

    @property
    def higher_buffer_space(self) -> gym.spaces.Space:
        assert self.observation_space["image"].dtype == np.uint8
        return self.observation_space["image"]

    @property
    def lower_buffer_space(self) -> gym.spaces.Space:      # For lower buffer: Image
        assert self.observation_space["image"].dtype == np.uint8
        return self.observation_space["image"]

    def get_higher_observation(self, obs):
        return self.get_representation(obs["image"])

    def get_lower_observation(self, obs):
        return self.get_representation(obs["image"])

    def get_higher_buffer_observation(self, obs, **kwargs):
        return obs["image"]

    def get_lower_buffer_observation(self, obs, **kwargs):
        return obs["image"]

    def get_representation(self, image):
        if isinstance(image, Dict):
            image = image["image"]
        return np.array(_get_representation(self.encoder, image))

    def build_hrl_models(self, hrl_config: Dict) -> Tuple[int, int]:
        """Load image encoder"""
        with open(hrl_config["config_dir"] + "encoder_config", "rb") as f:
            encoder_kwargs = pickle.load(f)

        self.representation_dim = encoder_kwargs["feature_dim"]

        encoder_def = ImgVariationalAutoEncoder(**encoder_kwargs)
        init_image = self.lower_buffer_space.sample()[np.newaxis, ...]
        self.rng, rngs = get_basic_rngs(self.rng)
        rngs.update({"latent_sampling": self.rng})
        encoder = Model.create(encoder_def, inputs=[rngs, init_image])      # Do not optimize this in RL

        encoder_file_name = hrl_config["model_dir"] + "image_encoder_100000"
        self.encoder = encoder.load_dict(encoder_file_name)
        print(f"Encoder params are loaded from {encoder_file_name}\n" * 10)
        return super(ImgBasedHRLAgent, self).build_hrl_models(hrl_config)
