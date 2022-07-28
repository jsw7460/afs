import functools
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.noise import ActionNoise

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.jax_layers import (
    FlattenExtractor,
)
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    RolloutReturn,
    Params
)
from offline_baselines_jax.common.utils import should_collect_more_steps
from offline_baselines_jax.sac.policies import SACPolicy, MultiInputPolicy
from .buffer import SoptFiSensorBasedExpertBuffer
from .core import (
    adversarial_imitation_learning_update
)
from .networks import (
    DeterministicLSTMSensorBasedForwardDynamics,
    SensorBasedDoubleStateLastConditionedDiscriminator,
    LSTMSubTrajectoryLastObsBasedSkillGenerator,
    MLPSkillPrior
)


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


@functools.partial(jax.jit, static_argnames=('actor_apply_fn',))
def sample_actions(
    rng: int,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: jnp.ndarray
) -> Tuple[int, jnp.ndarray]:
    dist = actor_apply_fn({'params': actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


@functools.partial(jax.jit, static_argnames=("disc_apply_fn", ))        # Forward path of observations to discriminator
def sample_intrinsic_rewards(
    disc_apply_fn: Callable[..., Any],
    disc_params: Params,
    observation: jnp.ndarray,
    next_observation: jnp.ndarray,
    last_observation: jnp.ndarray
):
    disc_score = disc_apply_fn({"params": disc_params}, observation, next_observation, last_observation)
    return jnp.mean(disc_score)


@functools.partial(jax.jit, static_argnames=("inv_dyna_apply_fn", ))
def inv_dyna_forward_path(
    inv_dyna_apply_fn: Callable[..., Any],
    inv_dyna_params: Params,
    observations: jnp.ndarray,
    next_observations: jnp.ndarray
):
    predicted_actions = inv_dyna_apply_fn({"params": inv_dyna_params}, observations, next_observations)
    return predicted_actions


class LogEntropyCoef(nn.Module):
    init_value: float = 0.1

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_temp


class SOPTFI(OffPolicyAlgorithm):
    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, Type[SACPolicy]] = MultiInputPolicy,
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

        n_frames: int = 3,
        subseq_len: int = 10,
        batch_dim: int = 256,
        hidden_dim: int = 128,
        skill_dim: int = 32,

        dropout: float = 0.0,
        intrinsic_rew_coef: float = 0.0,
        intrinsic_rew_thresh: float = 1.0,
        original_rew_coef: float = 1.0,
        n_stack: int = 0,
        model_archs: Optional[Dict[str, List]] = {},
        record_interval: int = 1_000,
        static_intrinsic_rewards: bool = True,
        discriminator_update_interval: int = 3,
    ):
        super(SOPTFI, self).__init__(
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
            supported_action_spaces=(gym.spaces.Box),
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

        ### Define model architectures name
        self.discriminator = None       # type: Model
        self.skill_generator = None     # type: Model
        self.skill_prior = None         # type: Model

        self.prerequisite_timestep = 0
        self.behavior_cloner = None
        self.inverse_dynamics = None
        self.n_stack = n_stack
        self.expert_buffer = None    # type: SensorBasedExpertBuffer
        self.model_archs = model_archs
        self.record_interval = record_interval

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
        self._original_rew_coef = original_rew_coef
        self.original_rew_coef = None       # type: float
        self._intrinsic_rew_coef = intrinsic_rew_coef
        self.intrinsic_rew_coef = None      # type: float

        self.intrinsic_rew_thresh = intrinsic_rew_thresh
        self.intr_rew_postprocessor = intr_reward_postpreocessor(intrinsic_rew_thresh)
        self.static_intrinsic_rewards = static_intrinsic_rewards
        self.discriminator_update_interval = discriminator_update_interval

        self.collect_rollout_caller = None      # type: Callable

        self.prerequisite_ft = None

        self._train = None   # type: Callable
        self.warmup_ft = None   # type: Callable
        self.intrinsic_rew_list = []

        self.__temporal_buffer = []

    @property
    def temporal_buffer(self):
        return self.__temporal_buffer

    @temporal_buffer.setter
    def temporal_buffer(self, val):
        self.__temporal_buffer = val

    @contextmanager
    def adversarial_imitation_learning_phase(self):
        # Convert learning starts
        _learning_starts = self.learning_starts
        self.learning_starts = 10000

        # Convert reward coefficients
        self.original_rew_coef = 0.0
        self.intrinsic_rew_coef = 1.0

        # Convert rollout collecting function and training function
        _grad_steps = self.gradient_steps
        self.gradient_steps = -1
        self.collect_rollout_caller \
            = frozen_partial(self._collect_rollouts_discriminator_based_rewards, train_freq=self.train_freq)
        self._train = self.adversarial_imitation_train
        yield
        self.learning_starts = _learning_starts
        self.gradient_steps = _grad_steps
        self.train_freq = (1, "step")
        self._convert_train_freq()
        self.original_rew_coef = 0.0
        self.intrinsic_rew_coef = 1.0
        self.collect_rollout_caller = self._collect_rollouts

    @contextmanager
    def warmup_phase(self):
        self.original_rew_coef = 0.0
        self.intrinsic_rew_coef = 1.0
        self._train = self.warmup_train
        self.train_freq = (1, "step")
        self._convert_train_freq()
        yield
        self.original_rew_coef = None
        self.intrinsic_rew_coef = None
        self._train = None

    @contextmanager
    def task_specific_phase(self):
        self.original_rew_coef = self._original_rew_coef
        self.intrinsic_rew_coef = self._intrinsic_rew_coef
        self._train = self.rl_train
        self.train_freq = (1, "step")
        self._convert_train_freq()
        yield
        self.original_rew_coef = None
        self.intrinsic_rew_coef = None
        self._train = None

    def set_prerequisite_component(self, prerequisite_cfg: Dict):
        for component in prerequisite_cfg:
            prerequisite = getattr(self, component)     # type: Model
            if prerequisite_cfg[component]["load"]:
                path = prerequisite_cfg[component]["path"]
                print(f"Load {component} from {path}.")
                prerequisite = prerequisite.load_dict(path)
            setattr(self, component, prerequisite)
        exit()

    def set_expert_buffer(
        self,
        path: str,
        n_frames: int,
        subseq_len: int,
        max_traj_len: int = 1_000_000
    ):
        self.expert_buffer = SoptFiSensorBasedExpertBuffer(
            data_path=path,
            n_frames=n_frames,
            subseq_len=subseq_len,
            max_traj_len=max_traj_len
        )

    def _setup_model(self) -> None:
        super(SOPTFI, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
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

        ### SOPTFI Models
        features_extractor_class = FlattenExtractor
        init_obs = self.observation_space.sample()[np.newaxis, ...]
        obs_dim = self.observation_space.shape[0]

        # Define: Forward dynamics model. Maybe determinisitic or stochastic.
        param_key, dropout_key, init_key = jax.random.split(self.rng, 3)
        rngs = {"params": param_key, "dropout": dropout_key, "init": init_key}
        forward_dynamics_kwargs = {
            "features_extractor": features_extractor_class(_observation_space=self.observation_space),
            "observation_space": self.observation_space,

            "n_frames": self.n_frames,
            "batch_dim": self.batch_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": int(obs_dim / self.n_frames),
            "dropout": self.dropout
        }
        forward_dynamics_def = DeterministicLSTMSensorBasedForwardDynamics(**forward_dynamics_kwargs)
        init_lstm_obs = init_obs.reshape(-1, self.n_frames, int(obs_dim / self.n_frames))
        init_act = np.repeat(self.env.action_space.sample()[None, None, ...], repeats=self.n_frames, axis=1)
        self.forward_dynamics = Model.create(
            forward_dynamics_def,
            inputs=[rngs, init_lstm_obs, init_act],
            tx=optax.radam(self.learning_rate)
        )

        # Define: discriminator
        param_key, dropout_key = jax.random.split(param_key)
        rngs = {"params": param_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        discriminator_def = SensorBasedDoubleStateLastConditionedDiscriminator(
            features_extractor=features_extractor,
            dropout=self.dropout,
            net_arch=self.model_archs["discriminator"]
        )
        self.discriminator = Model.create(
            discriminator_def,
            inputs=[rngs, init_obs, init_obs, init_obs],
            tx=optax.adam(3e-4)
        )

        # Define: skill generator (output latent vector z)
        param_key, dropout_key, init_key = jax.random.split(param_key, 3)
        rngs = {"params": param_key, "dropout": dropout_key, "init": init_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        skill_generator_def = LSTMSubTrajectoryLastObsBasedSkillGenerator(
            features_extractor=features_extractor,
            observation_space=self.observation_space,

            subseq_len=self.subseq_len,
            batch_dim=self.batch_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.skill_dim,
            dropout=self.dropout,
            net_arch=self.model_archs["skill_generator"]
        )
        obs_dim = init_obs.shape[-1]
        init_trajectory=jax.random.normal(init_key, shape=(1, self.subseq_len, obs_dim))
        self.skill_generator = Model.create(
            skill_generator_def,
            inputs=[rngs, init_trajectory, init_obs],
        )

        # Define: skill prior
        param_key, dropout_key = jax.random.split(param_key)
        rngs = {"params": param_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        skill_prior_def = MLPSkillPrior(
            features_extractor=features_extractor,
            observation_space=self.observation_space,

            latent_dim=64,
            skill_dim=self.skill_dim,
            dropout=self.dropout,
            net_arch=self.model_archs["skill_prior"]
        )
        self.skill_prior = Model.create(
            skill_prior_def,
            inputs=[rngs, init_obs]
        )

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64):
        return self._train(gradient_steps, batch_size)

    def adversarial_imitation_train(self, gradient_steps, batch_size: int = 64):
        actor_losses = []
        ent_coef_losses = []
        ent_coefs = []

        critic_losses = []

        disc_losses, disc_expert_scores, disc_policy_scores = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            expert_data = self.expert_buffer.sample(batch_size)

            expert_observations = expert_data.observations[:, -2, ...]
            expert_next_observations = expert_data.observations[:, -1, ...]
            expert_last_observations = expert_data.last_observations

            observations = replay_data.observations
            actions = replay_data.actions
            next_observations = replay_data.next_observations
            dones = replay_data.dones
            last_observations = replay_data.last_observations

            if self.static_intrinsic_rewards:
                rewards = replay_data.rewards
            else:
                """
                Reward dynamically change during discriminator is trained.
                """
                rewards = sample_intrinsic_rewards(
                    self.discriminator.apply_fn,
                    self.discriminator.params,
                    observations,
                    next_observations,
                    last_observations
                )
                rewards = self.intr_rew_postprocessor(rewards)

            target_update_cond = (self.num_timesteps % self.target_update_interval == 0)
            discriminator_update_cond = (self.num_timesteps % self.discriminator_update_interval == 0)

            self.rng, new_models, infos = adversarial_imitation_learning_update(
                rng=self.rng,

                discriminator=self.discriminator,
                actor=self.actor,
                critic=self.critic,
                critic_target=self.critic_target,
                log_ent_coef=self.log_ent_coef,

                expert_observations=expert_observations,
                expert_next_observations=expert_next_observations,
                expert_last_observations=expert_last_observations,

                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones,
                last_observations=last_observations,

                gamma=self.gamma,
                target_entropy=self.target_entropy,
                tau=self.tau,
                target_update_cond=target_update_cond,
                discriminator_update_cond=discriminator_update_cond,
                entropy_update=self.entropy_update
            )
            self.apply_new_models(new_models)

            actor_losses.append(infos["actor_loss"])
            critic_losses.append(infos["critic_loss"])
            ent_coef_losses.append(infos["ent_coef_loss"])
            ent_coefs.append(infos["ent_coef"])

            disc_losses.append(infos["discriminator_loss"])
            disc_expert_scores.append(infos["expert_disc_score"])
            disc_policy_scores.append(infos["policy_disc_score"])

        self.logger.record_mean("train/actor_loss", np.mean(actor_losses))
        self.logger.record_mean("train/critic_loss", np.mean(critic_losses))
        self.logger.record_mean("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record_mean("train/ent_coef", np.mean(ent_coefs))

        self.logger.record_mean("train/disc_loss", np.mean(disc_losses) * self.discriminator_update_interval)
        self.logger.record_mean("train/disc_expert_score", np.mean(disc_expert_scores)* self.discriminator_update_interval)
        self.logger.record_mean("train/disc_policy_score", np.mean(disc_policy_scores)* self.discriminator_update_interval)

    def warmup_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        pass

    def rl_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        actor_losses = []

        ent_coef_losses = []
        ent_coefs = []

        critic_losses = []

        single_disc_losses, single_expert_disc_scores, single_policy_disc_scores = [], [], []
        double_disc_losses, double_expert_disc_scores, double_policy_disc_scores = [], [], []

        inverse_dynamics_losses = []
        action_matcher_losses = []

        training_info = None

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size=batch_size)
            expert_observation, _, expert_next_observation = self.expert_buffer.sample(batch_size=batch_size)
            target_update_cond = (gradient_step % self.target_update_interval == 0)
            self.rng, new_models, training_info = self.update_ft(
                rng=self.rng,
                log_ent_coef=self.log_ent_coef,
                actor=self.actor,
                critic=self.critic,
                critic_target=self.critic_target,
                behavior_cloner=self.behavior_cloner,
                single_state_discriminator=self.single_state_discriminator,
                double_state_discriminator=self.double_state_discriminator,
                action_matcher=self.action_matcher,
                inverse_dynamics=self.inverse_dynamics,

                observations=replay_data.observations,
                actions=replay_data.actions,
                rewards=replay_data.rewards,
                next_observations=replay_data.next_observations,
                dones=replay_data.dones,

                expert_observation=expert_observation,
                expert_next_observation=expert_next_observation,

                target_update_cond=target_update_cond,
                entropy_update=self.entropy_update,
                target_entropy=self.target_entropy,
                gamma=self.gamma,
                tau=self.tau
            )
            self.apply_new_models(new_models)

        if self.num_timesteps % self.record_interval == 0:
            actor_losses.append(training_info["actor_loss"])
            critic_losses.append(training_info["critic_loss"])
            ent_coef_losses.append(training_info["ent_coef_loss"])
            ent_coefs.append(training_info["ent_coef"])

            self.logger.record("config/phase", "rl", exclude="tensorboard")

            self.logger.record_mean("train/ent_coef", np.mean(np.array(ent_coefs)))
            self.logger.record_mean("train/actor_loss", np.mean(np.array(actor_losses)))
            self.logger.record_mean("train/critic_loss", np.mean(np.array(critic_losses)))
            self.logger.record_mean("train/ent_coef_loss", np.mean(np.array(ent_coef_losses)))

            action_matcher_losses.append(training_info["action_matcher_loss"])

            single_disc_losses.append(training_info["single_discriminator_loss"])
            single_expert_disc_scores.append(training_info["single_expert_disc_score"])
            single_policy_disc_scores.append(training_info["single_policy_disc_score"])

            double_disc_losses.append(training_info["double_discriminator_loss"])
            double_expert_disc_scores.append(training_info["double_expert_disc_score"])
            double_policy_disc_scores.append(training_info["double_policy_disc_score"])

            inverse_dynamics_losses.append(training_info["inverse_dynamics_loss"])

            self.logger.record_mean("train/action_matcher_loss", np.mean(np.array(action_matcher_losses)))

            self.logger.record_mean("train/single_disc_loss", np.mean(np.array(single_disc_losses)))
            self.logger.record_mean("train/single_expert_disc_score", np.mean(np.array(single_expert_disc_scores)))
            self.logger.record_mean("train/single_policy_disc_score", np.mean(np.array(single_policy_disc_scores)))

            self.logger.record_mean("train/double_disc_loss", np.mean(np.array(double_disc_losses)))
            self.logger.record_mean("train/double_expert_disc_score", np.mean(np.array(double_expert_disc_scores)))
            self.logger.record_mean("train/double_policy_disc_score", np.mean(np.array(double_policy_disc_scores)))

            self.logger.record_mean("train/inv_dyna_loss", np.mean(np.array(inverse_dynamics_losses)))

    def _store_transition_with_discriminator_hindsight(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        flush_flag = False  # If true, store into the buffer. If false, do not store.
        last_observation = None

        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    last_observation = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

                flush_flag = True

        # NOTE: Do not store until episode is done
        self.temporal_buffer.append(
            (self._last_original_obs, next_obs, buffer_action, reward_, dones, infos)
        )

        # if self.replay_buffer.pos > 3000:
        #     rep_buf_sample = self.replay_buffer.sample(128)
        #     exp_buf_sample = self.expert_buffer.sample(128)

            # print("rep obs\t", rep_buf_sample.observations.shape)
            # print("rep next obs\t", rep_buf_sample.next_observations.shape)
            # print("rep last obs\t", rep_buf_sample.last_observations.shape)
            #
            # print("exp obs\t", exp_buf_sample.observations[:, -2, ...].shape)
            # print("exp next obs\t", exp_buf_sample.observations[:, -1, ...].shape)
            # print("exp last obs\t", exp_buf_sample.last_observations.shape)

        ### Check 했다

        if flush_flag:      # Store episodic transitions
            assert last_observation is not None, "last observation should be given to define an intrinsic reward"

            # Define: Episodic rewards using discriminator
            for (observation, next_observation, action, original_reward, done, info) in self.temporal_buffer:
                if last_observation.ndim == 1:
                    last_observation = last_observation[np.newaxis, ...]

                intrinsic_reward = sample_intrinsic_rewards(
                    self.discriminator.apply_fn,
                    self.discriminator.params,
                    observation,
                    next_observation,
                    last_observation
                )
                # intrinsic_reward = intr_reward_postprocess(intrinsic_reward)
                intrinsic_reward = self.intr_rew_postprocessor(intrinsic_reward)
                replay_buffer.add_with_last_observation(
                    observation, next_observation, action, intrinsic_reward, done, info, last_observation
                )

            # Initialize temporal buffer
            self.temporal_buffer = []

        self._last_obs = new_obs.copy()
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(self, *args, **kwargs) -> RolloutReturn:
        return self.collect_rollout_caller(*args, **kwargs)

    def _collect_rollouts_discriminator_based_rewards(
        self,
        env,
        callback,
        train_freq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Rewards are defined 'after' episodes are done. This needs the modification of codes.
        :param env:
        :param callback:
        :param train_freq:
        :param replay_buffer:
        :param action_noise:
        :param learning_starts:
        :param log_interval:
        :return:
        """
        num_collected_steps, num_collected_episodes = 0, 0

        callback.on_rollout_start()
        continue_training = True

        intrinsic_rewards = 0

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            new_obs, original_rewards, dones, infos = env.step(actions)

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

            self._store_transition_with_discriminator_hindsight(
                replay_buffer,
                buffer_actions,
                new_obs,
                original_rewards,
                dones,
                infos
            )

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    current_ep_len = infos[0].get("episode", None)["l"]
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _collect_rollouts(
        self,
        env,
        callback,
        train_freq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 100,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        num_collected_steps, num_collected_episodes = 0, 0

        callback.on_rollout_start()
        continue_training = True

        intrinsic_rewards = 0
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            new_obs, original_rewards, dones, infos = env.step(actions)
            intrinsic_rewards = sample_intrinsic_rewards(
                disc_apply_fn=self.double_state_discriminator.apply_fn,
                disc_params=self.double_state_discriminator.params,
                observations=self._last_obs.copy(),
                next_observations=new_obs
            )
            self.intrinsic_rew_list.append(intrinsic_rewards)
            rewards = (self.original_rew_coef * original_rewards) + (self.intrinsic_rew_coef * intrinsic_rewards)
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
                    current_ep_len = infos[0].get("episode", None)["l"]
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:

                        if intrinsic_rewards != 0:
                            self.logger.record("config/intr_rew_coef", self.intrinsic_rew_coef, exclude="tensorboard")
                            self.logger.record("rollout/intr_rew_mean", np.mean(np.array(self.intrinsic_rew_list)))
                            self.intrinsic_rew_list = []

                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs.copy()
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> jnp.ndarray:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.policy.predict(self._last_obs, deterministic=False)

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

    def offline_train(self, gradient_steps: int, batch_size: int) -> None:
        raise NotImplementedError()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SOPTFI",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SOPTFI, self).learn(
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
        return super(SOPTFI, self)._excluded_save_params() \
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
