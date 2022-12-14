import functools
from collections import deque
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

from offline_baselines_jax.common.buffers import ReplayBuffer, DictReplayBuffer
from offline_baselines_jax.common.jax_layers import (
    CombinedExtractor,
    FlattenExtractor,
)
from offline_baselines_jax.common.off_policy_algorithm import OffPolicyAlgorithm
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    Params,
    RolloutReturn
)
from offline_baselines_jax.common.utils import should_collect_more_steps
from offline_baselines_jax.sac.policies import SACPolicy, MultiInputPolicy
from . import core, utils
from .buffer import SensorBasedExpertBuffer, MultiTaskSensorBasedExpertBuffer
from .networks import (
    SensorBasedSingleStateActionMatcherFromHighToLow,
    SensorBasedSingleStateDiscriminator,
    SensorBasedDoubleStateDiscriminator,
    SensorBasedInverseDynamics,
    NaiveSensorBasedBehaviorCloner
)

ENV_MAX_LEN = 2000
ITEMGETTER = utils.MultiItemGetter("domain_name", "episode", "is_success")


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
    observations: jnp.ndarray,
    next_observations: jnp.ndarray
):
    disc_score = disc_apply_fn({"params": disc_params}, observations, next_observations)
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
    init_value: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_temp


class SensorBasedActionMatcherSoptSAC(OffPolicyAlgorithm):
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

        dropout: float = 0.0,
        intrinsic_rew_coef: float = 0.0,
        original_rew_coef: float = 1.0,
        n_stack: int = 0,
        update_ft_str: str = None,
        model_archs: Optional[Dict[str, List]] = {},
        record_interval: int = 1_000
    ):

        super(SensorBasedActionMatcherSoptSAC, self).__init__(
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

        self.prerequisite_timestep = 0
        self.behavior_cloner = None
        self.inverse_dynamics = None
        self.n_stack = n_stack
        self.expert_buffer = None    # type: SensorBasedExpertBuffer
        self.model_archs = model_archs
        self.record_interval = record_interval
        self.mt_logger = None       # type: Union[utils.MultiTaskLogHelper] # Logging helper for multitask environment.

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

        self.prerequisite_ft = None

        self._train = None   # type: Callable
        self.update_ft_str = update_ft_str
        self.update_ft = getattr(core, update_ft_str)
        self.warmup_ft = None   # type: Callable
        self.intrinsic_rew_list = []

    @contextmanager
    def warmup_phase(self):
        self.original_rew_coef = 0.0
        self.intrinsic_rew_coef = 1.0
        self._train = self.warmup_train
        yield
        self.original_rew_coef = None
        self.intrinsic_rew_coef = None
        self._train = None

    @contextmanager
    def task_specific_phase(self):
        self.original_rew_coef = self._original_rew_coef
        self.intrinsic_rew_coef = self._intrinsic_rew_coef
        self._train = self.rl_train
        yield
        self.original_rew_coef = None
        self.intrinsic_rew_coef = None
        self._train = None

    def set_expert_buffer(
        self,
        path: str,
        n_frames: int,
        multitask: bool = False,
        max_traj_len: int = 1_000_000
    ):
        """
        :param path:
        :param n_frames:
        :param multitask: True for metaworld environment
        :param max_traj_len:
        :return:
        """
        expert_buffer_class = MultiTaskSensorBasedExpertBuffer if multitask else SensorBasedExpertBuffer
        domain_name2idx = None
        obs_dim_without_onehot = None
        if multitask:
            domain_name2idx = self.env.get_attr("domain_name2idx")[0]
            obs_dim_without_onehot = self.env.get_attr("obs_dim_without_onehot")[0]
            self.mt_logger = utils.MultiTaskLogHelper(domain_name2idx)

        self.expert_buffer = expert_buffer_class(
            path,
            n_frames,
            domain_name2idx=domain_name2idx,
            obs_dim_without_onehot=obs_dim_without_onehot,
            max_traj_len=max_traj_len
        )

        self.expert_buffer.relabel_action_by_obs_difference()

    def _setup_model(self) -> None:
        super(SensorBasedActionMatcherSoptSAC, self)._setup_model()
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

        # Define init observation and lower action dimension
        features_extractor_class = None
        init_obs = None
        low_action_dim = None
        highaction_dim = self.action_space.shape[0]
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            features_extractor_class = CombinedExtractor
            low_action_dim = self.observation_space.sample()["observations"].shape[-1]
            init_obs = self.observation_space.sample()
            for k, v in init_obs.items():
                v = v[np.newaxis, ...]
                init_obs.update({k: v})

        elif isinstance(self.env.observation_space, gym.spaces.Box):
            features_extractor_class = FlattenExtractor
            low_action_dim = self.observation_space.sample().shape[-1]
            init_obs = self.observation_space.sample()[np.newaxis, ...]

        # If metaworld domain, fix the low action dimension
        if "MultiTask" in str(self.replay_buffer):
            low_action_dim = 39

        # NOTE: Action matcher
        self.rng, params_key, dropout_key = jax.random.split(self.rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        action_matcher_def = SensorBasedSingleStateActionMatcherFromHighToLow(
            features_extractor=features_extractor,
            dropout=self.dropout,
            highaction_dim=highaction_dim,
            squash_output=True,
            net_arch=self.model_archs.get("action_matcher", None)
        )
        self.action_matcher = Model.create(
            action_matcher_def,
            inputs=[rngs, init_obs, np.zeros((1, low_action_dim))],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

        # NOTE: Single state based discriminator
        self.rng, params_key, dropout_key = jax.random.split(self.rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        discriminator_def = SensorBasedSingleStateDiscriminator(
            features_extractor=features_extractor,
            dropout=self.dropout,
            net_arch=self.model_archs.get("single_state_discriminator", None)
        )
        self.single_state_discriminator = Model.create(
            discriminator_def,
            inputs=[rngs, init_obs],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

        # NOTE: Double state based discriminator
        self.rng, params_key, dropout_key = jax.random.split(self.rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        discriminator_def = SensorBasedDoubleStateDiscriminator(
            features_extractor=features_extractor,
            dropout=self.dropout,
            net_arch=self.model_archs.get("double_state_discriminator", None)
        )
        self.double_state_discriminator = Model.create(
            discriminator_def,
            inputs=[rngs, init_obs, init_obs],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

        # NOTE: Behavior cloner
        self.rng, params_key, dropout_key = jax.random.split(self.rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        bc_def = NaiveSensorBasedBehaviorCloner(
            features_extractor=features_extractor,
            lowaction_dim=low_action_dim,                        # Test: = Sensor data dimension
            dropout=self.dropout,
            net_arch=self.model_archs.get("behavior_cloner", None)
        )
        self.behavior_cloner = Model.create(
            bc_def,
            inputs=[rngs, init_obs],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

        # NOTE: Inverse dynamics
        self.rng, params_key, dropout_key = jax.random.split(self.rng, 3)
        rngs = {"params": params_key, "dropout": dropout_key}
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        inv_dyna_def = SensorBasedInverseDynamics(
            features_extractor=features_extractor,
            dropout=self.dropout,
            highaction_dim=highaction_dim,
            squash_output=True
        )

        self.inverse_dynamics = Model.create(
            inv_dyna_def,
            inputs=[rngs, init_obs, init_obs],
            tx=optax.adam(learning_rate=self.learning_rate)
        )

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def set_prerequisite_component(self, path: str) -> Tuple[bool, bool]:
        warmup_is_required = False
        if self.update_ft_str == "sensor_based_single_state_amsopt_sac_update":
            self.prerequisite_ft = getattr(self, "train_bc")
            self.warmup_ft = getattr(core, "warmup_bc_types")
            prerequisite_component_str = "behavior_cloner"
            path += "-bc500000"
            warmup_is_required = True

        elif self.update_ft_str == "sensor_based_double_state_amsopt_sac_update":
            self.prerequisite_ft = getattr(self, "train_bc")
            self.warmup_ft = getattr(core, "warmup_bc_types")
            prerequisite_component_str = "behavior_cloner"
            path += "-bc500000"
            warmup_is_required = True

        elif self.update_ft_str == "sensor_based_inverse_dynamics_without_disc_amsopt_sac_update":
            self.prerequisite_ft = getattr(self, "train_inv_dyna")
            # No warmup.
            prerequisite_component_str = "inverse_dynamics"
            path += "-inv_dyna1000000"

        elif self.update_ft_str == "sensor_based_single_state_amsopt_sac_update_without_action_matching_update":
            # No prerequisite component and warmup.
            return True, warmup_is_required

        else:
            prerequisite_component_str = None

        prerequisite_component = getattr(self, prerequisite_component_str, None)

        # try:
        print("PATH" * 77, path)
        prerequisite_component.load_dict(path)
        print("RUN")
        exit()
        # with open(path, "rb") as f:
        #     params = f.read()
        #     prerequisite_component.load_dict(params)
        #     setattr(self, prerequisite_component_str, prerequisite_component.load_dict(params))
        return True, warmup_is_required

        # except:
        #     return False, warmup_is_required

    def train_bc(self, num_timesteps: int = 500_000, batch_size: int = 64, path: str = None):
        bc_losses = deque(maxlen=5000)
        while self.prerequisite_timestep < num_timesteps:
            rng, _ = jax.random.split(self.rng)
            observations, actions, _ = self.expert_buffer.sample(batch_size, relabled_action=True, return_action=True)

            behavior_cloner, bc_info = core.prerequisite_behavior_cloner_update(
                rng=rng,
                behavior_cloner=self.behavior_cloner,
                expert_observations=observations,
                expert_actions=actions
            )
            self.behavior_cloner = behavior_cloner
            bc_losses.append(np.mean(bc_info["bc_loss"]))

            self.prerequisite_timestep += 1
            if self.prerequisite_timestep % 5000 == 0:
                print("*" * 10 + "BC" + "*" * 10)
                print("Timestep", self.prerequisite_timestep)
                print("Loss", np.mean(bc_losses))
                save_path = path + f"-bc{self.prerequisite_timestep}"
                self.behavior_cloner.save_dict(save_path)

    def train_inv_dyna_with_expert_data(
        self,
        num_timesteps: int = 500_000,
        batch_size: int = 64,
        path: str = None
    ):
        inv_dyna_losses = deque(maxlen=5000)
        while self.prerequisite_timestep < num_timesteps:
            observations, actions, next_observations = self.expert_buffer.sample(
                batch_size=batch_size,
                relabled_action=False
            )
            inverse_dynamics, inv_dyna_info = core.prerequisite_inverse_dynamics_update(
                rng=self.rng,
                inverse_dynamics=self.inverse_dynamics,
                observations=observations,
                actions=actions,
                next_observations=next_observations
            )
            self.rng, _ = jax.random.split(self.rng)
            self.inverse_dynamics = inverse_dynamics
            inv_dyna_losses.append(np.mean(inv_dyna_info["inverse_dynamics_loss"]))

            self.prerequisite_timestep += 1
            if self.prerequisite_timestep % 5000 == 0:
                print("*" * 10 + "Inverse Dynamics with EXPERT DATASET" + "*" * 10)
                print("Timestep", self.prerequisite_timestep)
                print("Loss", np.mean(inv_dyna_losses))
                expert_eval_loss = self.test_inv_dyna_for_expert_data()
                print("IS THIS REDUCED?", expert_eval_loss)

    def train_inv_dyna(
        self,
        num_timesteps: int = 1_000_000_000,
        batch_size: int = 4096,
        path: str = None
    ):
        self.learn(1)       # For setup logger
        inv_dyna_losses = deque(maxlen=5000)
        replay_buffer_class = DictReplayBuffer if isinstance(self.observation_space, gym.spaces.Dict) else ReplayBuffer
        inv_dyna_replay_buffer = replay_buffer_class(
            buffer_size=10_000,
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=True
        )
        # Copy the environment to collect the random transitions
        from copy import deepcopy
        inv_dyna_training_env = deepcopy(self.env)

        observation = None
        while self.prerequisite_timestep < num_timesteps:
            # Obtain random transitions
            while not inv_dyna_replay_buffer.full:
                if observation is None:
                    observation = inv_dyna_training_env.reset()
                action = inv_dyna_training_env.action_space.sample()
                next_observation, reward, done, info = inv_dyna_training_env.step(action)
                # Store the transition into the buffer
                inv_dyna_replay_buffer.add(observation, next_observation, action, reward, done, info)
                observation = next_observation
                if done:
                    observation = None

            replay_data = inv_dyna_replay_buffer.sample(batch_size)
            # Train inverse dynamics

            inverse_dynamics, inv_dyna_info = core.prerequisite_inverse_dynamics_update(
                rng=self.rng,
                inverse_dynamics=self.inverse_dynamics,
                observations=replay_data.observations,
                actions=replay_data.actions,
                next_observations=replay_data.next_observations
            )
            self.rng, _ = jax.random.split(self.rng)
            self.inverse_dynamics = inverse_dynamics
            inv_dyna_losses.append(np.mean(inv_dyna_info["inverse_dynamics_loss"]))

            self.prerequisite_timestep += 1
            if self.prerequisite_timestep % 5_000 == 0:
                expert_eval_loss = self.test_inv_dyna_for_expert_data()
                print("*" * 10 + "Inverse Dynamics" + "*" * 10)
                print("Timestep", self.prerequisite_timestep)
                print("Train Loss", np.mean(inv_dyna_losses))
                print("Expert data eval loss", expert_eval_loss)

                self.logger.record("prerequisite/train_loss", np.mean(np.array(inv_dyna_losses)))
                self.logger.record("prerequisite/eval_loss", np.mean(np.array(expert_eval_loss)))
                self.logger.dump(step=self.prerequisite_timestep)

                save_path = path + f"-inv_dyna{self.prerequisite_timestep}"
                self.inverse_dynamics.save_dict(save_path)

            # Collect new random transitions
            if self.prerequisite_timestep % 100_000 == 0:
                inv_dyna_replay_buffer.reset()

    def test_inv_dyna_for_expert_data(self) -> jnp.ndarray:
        observations, actions, next_observations = self.expert_buffer.sample(10_000, relabled_action=False)
        predicted_actions = inv_dyna_forward_path(
            self.inverse_dynamics.apply_fn,
            self.inverse_dynamics.params,
            observations,
            next_observations
        )
        inv_dyna_expert_loss = jnp.mean((predicted_actions - actions) ** 2)

        return inv_dyna_expert_loss

    def train(self, gradient_steps: int, batch_size: int = 64):
        return self._train(gradient_steps, batch_size)

    def warmup_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        actor_losses = []

        ent_coef_losses = []
        ent_coefs = []

        critic_losses = []

        single_disc_losses, single_expert_disc_scores, single_policy_disc_scores = [], [], []
        double_disc_losses, double_expert_disc_scores, double_policy_disc_scores = [], [], []

        inverse_dynamics_losses = []

        action_matcher_losses = []

        training_info = None
        for train_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size=batch_size)
            expert_observation, _, expert_next_observation = self.expert_buffer.sample(batch_size=batch_size)

            self.rng, new_models, training_info = self.warmup_ft(
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

            self.logger.record("config/phase", "warmup", exclude="tensorboard")

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

    def collect_rollouts(
        self,
        env,
        callback,
        train_freq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
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

                    if self.mt_logger is not None:
                        domain_name, episode, success = ITEMGETTER(infos[0])
                        print("GOGO?")
                        # episode["r"] = episodic return
                        self.mt_logger[domain_name].add_episodic_info(episode["r"], success)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:

                        if intrinsic_rewards != 0:
                            self.logger.record("config/intr_rew_coef", self.intrinsic_rew_coef, exclude="tensorboard")
                            self.logger.record("rollout/intr_rew_mean", np.mean(np.array(self.intrinsic_rew_list)))
                            self.intrinsic_rew_list = []

                        if self.mt_logger is not None:
                            for domain_name, rewards_mean, success_mean in self.mt_logger.current_domain_informations:
                                self.logger.record(f"multitask_reward/{domain_name}_r", np.mean(np.array([rewards_mean])))
                                self.logger.record(f"multitask_success/{domain_name}_s", np.mean(np.array([success_mean])))
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
        tb_log_name: str = "SensorBasedActionMatcherSoptSAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SensorBasedActionMatcherSoptSAC, self).learn(
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
        return super(SensorBasedActionMatcherSoptSAC, self)._excluded_save_params() \
               + ["actor", "critic", "critic_target", "log_ent_coef"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        params_dict['actor'] = self.actor.params
        params_dict['critic'] = self.critic.params
        params_dict['critic_target'] = self.critic_target.params
        params_dict['log_ent_coef'] = self.log_ent_coef.params
        return params_dict

    def _get_jax_load_params(self) -> List[str]:
        return ['actor', 'critic', 'critic_target', 'log_ent_coef']
