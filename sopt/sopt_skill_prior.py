import os
import pickle
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.noise import VectorizedActionNoise
from stable_baselines3.common.vec_env import VecEnv

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.jax_layers import FlattenExtractor
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    Params
)
from offline_baselines_jax.common.type_aliases import (
    TrainFreq,
    TrainFrequencyUnit,
    RolloutReturn
)
from offline_baselines_jax.common.utils import get_basic_rngs
from offline_baselines_jax.common.utils import should_collect_more_steps
from .buffer import SoptFiSensorBasedExpertBuffer
from .core import (
    skill_prior_update,
    skill_regularized_sac_update,
)
from .networks import (
    LSTMSubTrajectoryLastObsBasedSkillGenerator,
    MLPSkillPrior
)
from .policies import LowLevelSkillPolicy, SkillBasedComposedPolicy
from .utils import clock


def intr_reward_postpreocessor(thresh: float):

    return lambda x: x if x < thresh else 1.0


def frozen_partial(func, **kwargs):
    frozen = kwargs

    def wrapper(*args, **_kwargs):
        _kwargs.update(frozen)
        return func(*args, **_kwargs)
    return wrapper


def static_vars_deco(**kwargs):

    def decorate(func: Callable):
        for key in kwargs:
            setattr(func, key, kwargs[key])
        return func
    return decorate


# @clock(fmt="[{elapsed: 0.8f}s {name}]")
def check_nested_dict_numpy_equation(x: Dict, y: Dict):
    assert x.keys() == y.keys(), "Different keys cannot be compared"

    results = []
    for xval, yval in zip(x.values(), y.values()):
        tmp = []
        if isinstance(xval, Dict):
            tmp.extend(check_nested_dict_numpy_equation(xval, yval))
        else:
            tmp.append(np.all(xval == yval))
        results.extend(tmp)
    return results


class LogEntropyCoef(nn.Module):
    init_value: float = 0.1

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_temp


class SOPTSkillEmpowered(OffPolicyAlgorithm):
    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str] = SkillBasedComposedPolicy,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,                   # 1e6
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
        relu_slope: float = 0.2,
        dropout: float = 0.0,
        n_frames: int = 3,
        subseq_len: int = 10,
        batch_dim: int = 256,
        hidden_dim: int = 128,
        skill_dim: int = 5,

        model_archs: Optional[Dict[str, List]] = {},
        bc_reg_coef: float = 0.5,
        pa_logstd_coef: float = 3.0,        # pseudo action logstd coef
        sp_logstd_coef: float = 2.0,        # Skill prior logstd coef
        pseudo_action_dim: Union[int] = None,           # Explicit value is given for Kitchen or ...
        skill_generator_cond_dim: Union[int] = None,    # Explicit value is given for Kitchen or ...

        expert_dataset_load_interval: int = 500_000,
    ):
        super(SOPTSkillEmpowered, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box, gym.spaces.Discrete),
            support_multi_env=True,
            without_exploration=without_exploration,
            dropout=dropout
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.entropy_update = True

        # Define model architectures name and type hint

        self.policy: SkillBasedComposedPolicy
        self.lowlevel_policy = None     # type: Model
        self.discriminator = None       # type: Model
        self.skill_generator = None     # type: Model
        self.skill_prior = None         # type: Model

        self.prerequisite_timestep = 0
        self.higher_replay_buffer = None    # type: ReplayBuffer

        self.use_real_actions: bool = False      # True for SBMTRL reproducing
        self.expert_dataset_list = None
        self.expert_buffer = None    # type: Union[SoptFiSensorBasedExpertBuffer]
        self.current_dataset_pos = 0

        self.expert_buffer_class = None
        self.expert_data_load_interval = expert_dataset_load_interval

        self.model_archs = model_archs
        self.bc_reg_coef = bc_reg_coef
        self.pa_logstd_coef = pa_logstd_coef
        self.sp_logstd_coef = sp_logstd_coef
        self.pseudo_action_dim = pseudo_action_dim
        self.skill_generator_cond_dim = skill_generator_cond_dim

        self.relu_slope = relu_slope
        self.n_frames = n_frames
        self.subseq_len = subseq_len
        self.batch_dim = batch_dim,
        self.hidden_dim = hidden_dim
        self.skill_dim = skill_dim

        if _init_setup_model:
            self._setup_model()

        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval

        self.dropout = dropout

        self.prerequisite_ft = None

        self._train = None   # type: Callable
        self._offline_train = None      # type: Callable

    @property
    def skill_prior_model_save_interval(self):
        return 500_000

    @property
    def skill_generator_class(self):
        return LSTMSubTrajectoryLastObsBasedSkillGenerator

    @property
    def skill_prior_class(self):
        return MLPSkillPrior

    @property
    def skill_decoder_class(self):
        return LowLevelSkillPolicy

    @contextmanager
    def skill_prior_learning_phase(self):
        _train = self._train
        _offline_train = self._offline_train
        _without_exploration = self.without_exploration
        self._train = None
        self._offline_train = self.skill_prior_train
        self.without_exploration = True
        yield

        self._train = _train
        self._offline_train = _offline_train
        self.without_exploration = _without_exploration

    @contextmanager
    def rl_phase(self):
        _train = self._train
        _without_exploration = self.without_exploration
        self._train = self.rl_train
        self.without_exploration = False
        yield
        self._train = _train
        self.without_exploration = _without_exploration

    @contextmanager
    def use_real_action_sequence(self):
        use_real_actions = self.use_real_actions
        self.use_real_actions = True
        yield
        self.use_real_actions = use_real_actions

    def set_expert_buffer(
        self,
        buffer_class,
        path: str,      # Directory. Not a file.
        n_frames: int,
        subseq_len: int,
        max_traj_len: int = 1_000_000
    ):
        self.expert_buffer_class = buffer_class
        self.expert_dataset_dir = path
        self.expert_dataset_list = sorted([f for f in os.listdir(path)])

        print("Expert dataset list:", self.expert_dataset_list)

        current_dataset_path = path + self.expert_dataset_list[self.current_dataset_pos]
        self.expert_buffer = buffer_class(
            data_path=current_dataset_path,
            n_frames=n_frames,
            subseq_len=subseq_len,
            max_traj_len=max_traj_len
        )
        self.expert_buffer.relabel_action_by_obs_difference()
        print(f"Expert data is load from {current_dataset_path}\n" * 10)
        print(f"\t Num trajectories: {len(self.expert_buffer)}")
        print(f"\t Num transitions: {self.expert_buffer.n_transitions}")

    def load_next_expert_buffer(self):
        self.current_dataset_pos += 1
        self.current_dataset_pos %= len(self.expert_dataset_list)
        current_dataset_path = self.expert_dataset_dir + self.expert_dataset_list[self.current_dataset_pos]
        print(f"Expert data is load from {current_dataset_path}")
        self.expert_buffer = self.expert_buffer_class(
            data_path=current_dataset_path,
            n_frames=self.n_frames,
            subseq_len=self.subseq_len
        )
        self.expert_buffer.relabel_action_by_obs_difference()

    def _setup_model(self) -> None:
        super(SOPTSkillEmpowered, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32) / 2
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)
        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            log_ent_coef_def = LogEntropyCoef(init_value)
            self.rng, temp_key = jax.random.split(self.rng, 2)
            self.log_ent_coef = Model.create(
                log_ent_coef_def, inputs=[temp_key],
                tx=optax.adam(learning_rate=self.lr_schedule(1))
            )

        else:
            # Force conversion to float
            # this will throw an erwror if a malformed string (different from 'auto')
            # is passed
            log_ent_coef_def = LogEntropyCoef(self.ent_coef)
            self.rng, temp_key = jax.random.split(self.rng, 2)
            self.log_ent_coef = Model.create(log_ent_coef_def, inputs=[temp_key])
            self.entropy_update = False

        self.target_alpha = self.target_entropy
        self.log_alpha = self.log_ent_coef
        self.alpha_update = self.entropy_update

    def build_lowlevel_policy(self) -> Dict:        # == pseudo action decoder
        self.rng, rngs = get_basic_rngs(self.rng)

        features_extractor_class = FlattenExtractor
        init_obs = self.observation_space.sample()[np.newaxis, ...]

        obs_dim = init_obs.shape[-1]
        lowlevel_action_dim = self.pseudo_action_dim if self.pseudo_action_dim is not None else obs_dim

        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        lowlevel_policy_kwargs = {
            "features_extractor": features_extractor,
            "observation_space": self.observation_space,
            # lowlevel action dim = obs dim. We have no info about action even dimension.
            "lowlevel_action_dim": lowlevel_action_dim,
            "net_arch": self.model_archs["lowlevel_policy"],
            "dropout": 0.0,
            "log_std_coef": self.pa_logstd_coef
        }
        lowlevel_policy_def = self.skill_decoder_class(**lowlevel_policy_kwargs)
        init_skill = jnp.zeros((1, self.skill_dim))

        self.lowlevel_policy = Model.create(
            lowlevel_policy_def,
            inputs=[rngs, init_obs, init_skill],
            tx=optax.adam(5e-4)
        )

        return lowlevel_policy_kwargs

    def build_skill_generator(self) -> Dict:
        features_extractor_class = FlattenExtractor

        skill_generator_cond_dim = self.skill_generator_cond_dim or self.observation_space.shape[-1]
        init_obs = np.random.uniform(-1, 1, size=(1, skill_generator_cond_dim))

        obs_dim = init_obs.shape[-1]
        pseudo_action_dim = self.pseudo_action_dim or obs_dim

        self.rng, rngs = get_basic_rngs(self.rng)

        init_key, _ = jax.random.split(self.rng)
        rngs.update({"init": init_key})
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        self.batch_dim = self.batch_size
        skill_generator_kwargs = {
            "features_extractor": features_extractor,
            "observation_space": self.observation_space,
            "subseq_len": self.subseq_len,
            "batch_dim": self.batch_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.skill_dim,
            "dropout": self.dropout,
            "net_arch": self.model_archs["skill_generator"]
        }
        skill_generator_def = self.skill_generator_class(**skill_generator_kwargs)
        init_observations = self.observation_space.sample()[np.newaxis, np.newaxis, ...]
        init_observations = np.repeat(init_observations, repeats=self.subseq_len, axis=1)
        init_actions = np.random.normal(size=(1, self.subseq_len, pseudo_action_dim))

        self.skill_generator = Model.create(
            skill_generator_def,
            inputs=[rngs, init_observations, init_actions, init_obs],
            tx=optax.adam(5e-4)
        )
        return skill_generator_kwargs

    def build_skill_prior(self) -> Dict:
        features_extractor_class = FlattenExtractor
        init_obs = self.observation_space.sample()[np.newaxis, ...]

        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        skill_prior_kwargs = {
            "features_extractor": features_extractor,
            "observation_space": self.observation_space,
            "latent_dim": 64,
            "skill_dim": self.skill_dim,
            "dropout": self.dropout,
            "net_arch": self.model_archs["skill_prior"],
            "log_std_coef": self.sp_logstd_coef,
            "relu_slope": self.relu_slope
        }
        skill_prior_def = self.skill_prior_class(**skill_prior_kwargs)

        self.skill_prior = Model.create(
            skill_prior_def,
            inputs=[rngs, init_obs],
            tx=optax.adam(learning_rate=5e-4)
        )
        return skill_prior_kwargs

    def build_skill_prior_models(self, skill_prior_config: Dict) -> int:
        if not skill_prior_config["build"]: return 0        # Return required training step

        # Define: A dictionary which saves a parameters of pretrained model. This will be saved with model together.
        pretrained_kwargs = {}

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
            "normalizing_max": self.expert_buffer.normalizing_max,
            "normalizing_min": self.expert_buffer.normalizing_min    # required to finetuning
        })

        os.makedirs(skill_prior_config["config_save_dir"], exist_ok=True)
        config_path = skill_prior_config["config_save_dir"] + "config"
        with open(config_path, "wb") as f:
            pickle.dump(pretrained_kwargs, f)
        print(f"Config is saved in {config_path}")

        self.skill_prior_model_save_dir = skill_prior_config["model_save_dir"]
        return skill_prior_config["total_timesteps"]

    def build_rl_models(self, rl_config: Dict) -> Tuple[int, int]:
        self.policy.build_policy(rl_config["policy_build_config"])
        self._create_rl_aliases()
        return rl_config["warmup_total_timesteps"], rl_config["total_timesteps"]

    def build_hrl_models(self, hrl_config: Dict) -> Tuple[int, int]:
        raise NotImplementedError()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _create_rl_aliases(self):
        self.policy: SkillBasedComposedPolicy
        self._create_aliases()
        self.skill_prior = self.policy.skill_prior

    def _update_rl_aliases(self, new_models: Dict[str, Any]):
        # We don't update the skill prior in RL phase ~!
        for model, val in new_models.items():
            assert hasattr(self, model)
            if model == "actor":
                self.actor = val
                self.policy.actor = val
            elif model == "critic":
                self.critic = val
                self.policy.critic = val
            elif model == "critic_target":
                self.critic_target = val
                self.policy.critic_target = val
            else:
                setattr(self, model, val)

    def train(self, gradient_steps: int, batch_size: int = 64):
        # Define train using context managing
        return self._train(gradient_steps, batch_size)

    def get_skill_prior_training_input(self):
        """
        :return:
            observations: [batch_size, subseq_len, dim]
            actions: [batch_size, subseq_len, dim]      # (Pseudo action)
            last_observations: [batch_size, dim]
        """
        replay_data = self.expert_buffer.sample_skill_prior_training_data(
            self.batch_size,
            real_actions=self.use_real_actions
        )
        observations, last_observations = replay_data.observations, replay_data.last_observations
        actions = np.array(replay_data.actions, dtype=np.float)
        if self.pseudo_action_dim is not None:
            actions = actions[..., :self.pseudo_action_dim]
        if self.skill_generator_cond_dim is not None:
            last_observations = observations[:, -1, self.skill_generator_cond_dim:]
        actions = jnp.clip(actions, -1 + 1E-6, 1 - 1E-6)
        return observations, actions, last_observations

    # @clock(fmt="[{name}: {elapsed: 0.8f}s]")
    def skill_prior_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        skill_generator_kl_losses = []
        lowerlevel_policy_losses = []
        skill_prior_losses = []

        skill_means, skill_log_stds, skill_self_lls = [], [], []
        skill_prior_means, skill_prior_log_stds, skill_prior_self_lls = [], [], []
        lowlevel_means, lowlevel_log_stds, lowlevel_self_lls = [], [], []

        for gradient_step in range(gradient_steps):

            # Train using pseudo action, not real one
            observations, actions, last_observations = self.get_skill_prior_training_input()

            self.rng, new_models, infos = skill_prior_update(
                rng=self.rng,

                lowlevel_policy=self.lowlevel_policy,
                skill_generator=self.skill_generator,
                skill_prior=self.skill_prior,

                observations=observations,
                actions=actions,
                last_observations=last_observations
            )

            self.apply_new_models(new_models)
            self.num_timesteps += 1

            # Log
            skill_generator_kl_losses.append(infos["skill_generator_kl_loss"])
            lowerlevel_policy_losses.append(infos["lowlevel_policy_loss"])
            skill_prior_losses.append(infos["skill_prior_loss"])

            skill_means.append(infos["skill_mean"])
            skill_log_stds.append(infos["skill_log_std"])
            skill_self_lls.append(infos["skill_generator_self_ll"])

            skill_prior_means.append(infos["skill_prior_mean"])
            skill_prior_log_stds.append(infos["skill_prior_log_std"])
            skill_prior_self_lls.append(infos["skill_prior_self_ll"])

            lowlevel_means.append(infos["lowlevel_policy_mean"])
            lowlevel_log_stds.append(infos["lowlevel_log_std"])
            lowlevel_self_lls.append(infos["lowlevel_self_ll"])

        if (self.num_timesteps % 50) == 0:
            self.logger.record("config/cur_dataset_pos", self.current_dataset_pos)
            self.logger.record_mean("train/skill_gen_kl_loss", np.mean(skill_generator_kl_losses))
            self.logger.record_mean("train/lowlevel_bc_ll", -np.mean(lowerlevel_policy_losses))
            self.logger.record_mean("train/skill_prior_loss", np.mean(skill_prior_losses))

            self.logger.record_mean("skill_gen/mean(g)", np.mean(skill_means))
            self.logger.record_mean("skill_gen/std(g)", np.mean(np.exp(skill_log_stds)))
            self.logger.record_mean("skill_gen/log_std(g)", np.mean(skill_log_stds))
            self.logger.record_mean("skill_gen/self_ll(g)", np.mean(skill_self_lls))

            self.logger.record_mean("skill_prior/mean(p)", np.mean(skill_prior_means))
            self.logger.record_mean("skill_prior/std(p)", np.mean(np.exp(skill_prior_log_stds)))
            self.logger.record_mean("skill_prior/log_std(p)", np.mean(skill_prior_log_stds))
            self.logger.record_mean("skill_prior/self_ll(p)", np.mean(skill_prior_self_lls))

            self.logger.record_mean("lowlevel_policy/mean(l)", np.mean(lowlevel_means))
            self.logger.record_mean("lowlevel_policy/log_std(l)", np.mean(lowlevel_log_stds))
            self.logger.record_mean("lowlevel_policy/self_ll(l)", np.mean(lowlevel_self_lls))

        if (self.num_timesteps % 500) == 0:
            print("=" * 30)
            print("Generated action", infos["pseudoaction_mean"][0])
            print("Real action", actions[0][0])
            self.logger.record_mean("timestep", self.num_timesteps)
            self.logger.dump(self.num_timesteps)

        if (self.num_timesteps % self.skill_prior_model_save_interval) == 0:
            print("*" * 30, f"model saved in {self.skill_prior_model_save_dir}")
            self.skill_prior.save_dict(self.skill_prior_model_save_dir + f"skill_prior_{self.num_timesteps}")
            self.skill_prior.save_batch_stats(self.skill_prior_model_save_dir + f"skill_prior_batch_stats_{self.num_timesteps}")

            self.lowlevel_policy.save_dict(self.skill_prior_model_save_dir + f"lowlevel_policy_{self.num_timesteps}")
            self.skill_generator.save_dict(self.skill_prior_model_save_dir + f"skill_generator_{self.num_timesteps}")

            self.lowlevel_policy.save_batch_stats(self.skill_prior_model_save_dir + f"lowlevel_policy_batch_stats_{self.num_timesteps}")
            self.skill_generator.save_batch_stats(self.skill_prior_model_save_dir + f"skill_generator_batch_stats_{self.num_timesteps}")

        if (self.num_timesteps % self.expert_data_load_interval) == 0:
            self.load_next_expert_buffer()

    def load_test(self, cfg):
        model_dir = cfg["model_save_dir"]
        self.lowlevel_policy = self.lowlevel_policy.load_dict(model_dir + "lowlevel_policy_100000")
        self.lowlevel_policy = self.lowlevel_policy.load_batch_stats(model_dir + "lowlevel_policy_batch_stats_100000")

        self.skill_prior = self.skill_prior.load_dict(model_dir + "skill_prior_100000")
        self.skill_prior = self.skill_prior.load_batch_stats(model_dir + "skill_prior_batch_stats_100000")

        self.skill_generator = self.skill_generator.load_dict(model_dir + "skill_generator_100000")
        self.skill_generator = self.skill_generator.load_batch_stats(model_dir + "skill_prior_batch_stats_100000")

        print("TEST TEST TEST \n" * 50)

    def rl_train(self, gradient_steps: int, batch_size: int = 64) -> None:

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size=batch_size)

            self.rng, key = jax.random.split(self.rng, 2)
            target_update_cond = (gradient_step % self.target_update_interval == 0)

            self.rng, new_models, info = skill_regularized_sac_update(
                rng=self.rng,
                skill_prior=self.skill_prior,
                actor=self.actor,
                critic=self.critic,
                critic_target=self.critic_target,
                log_alpha=self.log_alpha,

                observations=replay_data.observations,
                actions=replay_data.actions,
                rewards=replay_data.rewards + 1,
                next_observations=replay_data.next_observations,
                dones=replay_data.dones,

                gamma=self.gamma,
                target_alpha=self.target_alpha,
                bc_reg_coef=0.1,
                tau=self.tau,
                target_update_cond=target_update_cond,
                alpha_update=self.alpha_update
            )
            self._update_rl_aliases(new_models)

            if self.num_timesteps % 100 == 0:
                self.logger.record_mean("sac/actor_loss", info["actor_loss"].mean())
                self.logger.record_mean("sac/alpha_loss", info["alpha_loss"].mean())
                self.logger.record_mean("sac/critic_loss", info["critic_loss"].mean())
                self.logger.record_mean("sac/alpha", np.exp(info["log_alpha"]).mean())
                self.logger.record_mean("sac/min_q_val", info["min_qf_pi"].mean())

                self.logger.record_mean("adapt/prior_div", info["prior_divergence"].mean())

    def offline_train(self, gradient_steps: int, batch_size: int) -> None:
        return self._offline_train(gradient_steps, batch_size)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        num_collected_steps, num_collected_episodes = 0, 0
        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        assert not self.without_exploration

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    # self.logger.record(
                    #     "n_completed_task",
                    #     sum([v == 1 for k, v in env.get_attr("solved_subtasks")[0].items()])
                    # )

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> jnp.ndarray:

        self.policy: SkillBasedComposedPolicy

        if self.num_timesteps % self.policy.highlevel_update_interval == 0:
            z = self.policy.sample_higher_action(self._last_obs, deterministic=False, training=False)
        else:
            z = self.policy.z

        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter

            with self.policy.given_skill(z):
                unscaled_action = self.policy.predict(self._last_obs.copy(), deterministic=False)
                if np.random.uniform(0, 1) < 0.001:
                    print("Z", z)
                    print("Action", unscaled_action)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)

        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SOPTSkillEmpowered",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        if total_timesteps == 0: return None

        return super(SOPTSkillEmpowered, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(SOPTSkillEmpowered, self)._excluded_save_params() \
               + ["actor", "critic", "critic_target", "log_ent_coef"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = dict()
        params_dict['actor'] = self.actor.params
        params_dict['critic'] = self.critic.params
        params_dict['critic_target'] = self.critic_target.params
        params_dict['log_ent_coef'] = self.log_ent_coef.params
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ['actor', 'critic', 'critic_target', 'log_ent_coef']
