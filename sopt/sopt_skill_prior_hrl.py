import pickle
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

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
from offline_baselines_jax.common.jax_layers import (
    FlattenExtractor,
)
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
from .buffer import HigherReplayBuffer, LowerReplayBuffer
from .core import hrl_higher_policy_update, hrl_lower_policy_update
from .hrl_policies import ConditionedActor
from .hrl_policies import HigherPolicy, LowerPolicy
from .networks import (
    MLPSkillPrior,
    PseudoActionPolicy,
    DeterministicPseudoActionPolicy,
)
from .policies import SkillBasedComposedPolicy
from .sopt_skill_prior import SOPTSkillEmpowered
from .utils import clock

EPS = 1E-8


def check_nested_dict_numpy_equation(x: Dict, y: Dict):
    assert x.keys() == y.keys()

    results = []
    for xval, yval in zip(x.values(), y.values()):
        tmp = []
        if isinstance(xval, Dict):
            tmp.extend(check_nested_dict_numpy_equation(xval, yval))
        else:
            tmp.append(np.all(xval == yval))
        results.extend(tmp)
    return results


def intr_reward_postpreocessor(thresh: float):

    return lambda x: x if x < thresh else 1.0


def frozen_partial(func, **kwargs):
    frozen = kwargs

    def wrapper(*args, **_kwargs):
        _kwargs.update(frozen)
        return func(*args, **_kwargs)
    return wrapper


@jax.jit
def forward_skill_prior(rng: jnp.ndarray, skill_prior: Model, observations: jnp.ndarray):
    z_dist, *_ = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        observations,
        training=False,
        deterministic=True
    )
    sampled_z = z_dist.sample(seed=rng)
    return sampled_z



@partial(jax.jit, static_argnames=("batch_size", "n_rand", "skill_dim"))
def sample_normal(rng: jnp.ndarray, *shape):
    return jax.random.normal(rng, shape=shape)


@partial(jax.jit)
def get_maxprob_indices(
    rng: jnp.ndarray,
    lower_policy: Model,
    observations: jnp.ndarray,
    skills: jnp.ndarray,
    lower_actions: jnp.ndarray,
    deterministic: bool = False
):
    rng, dropout_key = jax.random.split(rng)
    max_indices = lower_policy.apply_fn(
        {"params": lower_policy.params},
        observations,
        skills,
        lower_actions,
        deterministic=deterministic,
        rngs={"dropout": dropout_key},
        method=ConditionedActor.offpolicy_correction_max_indices
    )
    return max_indices


@clock(fmt="[{name}: {elapsed: 0.8f}s]")
def offpolicy_correction(
    rng: jnp.ndarray,
    lower_policy: LowerPolicy,

    observations: jnp.ndarray,      # [batch, subseq_len, obs_dim]
    lower_actions: jnp.ndarray,           # [batch, subseq_len, act_dim]
    higher_actions: jnp.ndarray,                 # [batch_size, skill_dim]

    skill_perturbing_std: Union[float, jnp.ndarray],        # Normal or from higher generator
    batch_size: int,
    skill_dim: int,
    n_rand: int,
):
    rng, sampling_key, dropout_key = jax.random.split(rng, 3)

    _higher_actions = higher_actions[:, jnp.newaxis, ...]
    _higher_actions = jnp.repeat(_higher_actions, repeats=n_rand-1, axis=1)
    perturbed_skills \
        = _higher_actions + skill_perturbing_std * sample_normal(sampling_key, (batch_size, n_rand-1, skill_dim))

    # Include original higher action
    perturbed_skills = jnp.concatenate((perturbed_skills, higher_actions[:, jnp.newaxis, ...]), axis=1)      # [batch, n_rand, skill_dim]
    max_indices = get_maxprob_indices(rng, lower_policy.actor, observations, perturbed_skills, lower_actions, deterministic=False)
    relabeled_skills = perturbed_skills[jnp.arange(batch_size), max_indices, ...]
    return relabeled_skills      # [batch, skill_dim]


@partial(jax.jit, static_argnames=("deterministic", "training", "pseudo_action_dim"))
def get_similarity_intrinsic_reward(
    rng: jnp.ndarray,
    representation: jnp.ndarray,
    next_representation: jnp.ndarray,
    normalizing_max: jnp.ndarray,
    normalizing_min: jnp.ndarray,
    z: jnp.ndarray,
    # z_var: jnp.ndarray,
    higher_actor: Model,
    pseudo_action_policy: Model,
    deterministic: bool = True,
    pseudo_action_dim: int = None,
    **_,
):
    rng, dropout_key = jax.random.split(rng)
    pseudo_action = pseudo_action_policy.apply_fn(
        {"params": pseudo_action_policy.params, "batch_stats": pseudo_action_policy.batch_stats},
        representation,
        z,
        rngs={"dropout": dropout_key},
        deterministic=deterministic,
        training=False,
        method=PseudoActionPolicy.get_deterministic_actions
    )
    # pseudo_action = pseudo_action_dist.sample(seed=rng)
    renormalized_pseudo_action = (normalizing_max - normalizing_min + EPS) * (pseudo_action + 1) / 2 + normalizing_min
    renormalized_pseudo_action = renormalized_pseudo_action[:, :pseudo_action_dim]
    pred = (next_representation - representation)[:, :pseudo_action_dim]
    # Compute cosine similarity

    similarity = jnp.dot(renormalized_pseudo_action, pred.T) \
                 / (jnp.linalg.norm(renormalized_pseudo_action) * jnp.linalg.norm(pred))

    # _, _, cur_z_logstd = higher_actor.apply_fn(
    #     {"params": higher_actor.params, "batch_stats": higher_actor.batch_stats},
    #     representation,
    #     deterministic=True
    # )
    # cur_z_var = jnp.exp(cur_z_logstd) ** 2
    _, _, next_z_logstd = higher_actor.apply_fn(
        {"params": higher_actor.params, "batch_stats": higher_actor.batch_stats},
        next_representation,
        deterministic=True,
        training=False
    )
    next_z_var = jnp.exp(next_z_logstd) ** 2
    # return 0.01 * (3 * similarity.mean() - z_var.mean()), (renormalized_pseudo_action, next_representation - representation), 0.0
    return 0.01 * (3 * similarity.mean() - next_z_var.mean())


class LogEntropyCoef(nn.Module):
    init_value: float = 0.1

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return log_temp


class SkillBasedHRLAgent(SOPTSkillEmpowered):
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
        dropout: float = 0.0,
        n_frames: int = 3,
        subseq_len: int = 10,
        batch_dim: int = 256,
        hidden_dim: int = 128,
        skill_dim: int = 5,
        use_hiro_relabel: bool = False,

        model_archs: Optional[Dict[str, List]] = {},
        bc_reg_coef: float = 0.5,

        pseudo_action_dim: Union[int] = None,  # Explicit value is given for Kitchen
        skill_generator_cond_dim: Union[int] = None,  # Explicit value is given for Kitchen

        higher_ent_coef: Union[str, float] = "auto",
        lower_ent_coef: Union[str, float] = "auto",
        higher_target_entropy: Union[str, float] = "auto",
        lower_target_entropy: Union[str, float] = "auto",

        intrinsic_reward_scaler: float = 0.01
    ):

        self.higher_ent_coef = higher_ent_coef
        self.lower_ent_coef = lower_ent_coef
        self.higher_target_entropy = higher_target_entropy
        self.lower_target_entropy = lower_target_entropy

        super(SkillBasedHRLAgent, self).__init__(
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

            pseudo_action_dim=pseudo_action_dim,
            skill_generator_cond_dim=skill_generator_cond_dim
        )
        self.higher_policy: HigherPolicy = None
        self.lower_policy: LowerPolicy = None
        self.skill_prior: Model = None
        self.pseudo_action_policy: Model = None
        self.use_hiro_relabel = use_hiro_relabel

        self.current_episode_timestep = 0
        self.higher_actor_rewards = 0.0

        self.higher_replay_buffer: HigherReplayBuffer = None
        self.lower_replay_buffer: LowerReplayBuffer = None

        self.normalizing_max = 1.0
        self.normalizing_min = -1.0

        self.intrinsic_rewards = []

        # Image encoder. If we use sensor-data, it is identity function.
        self.encoder = lambda x: x

        self.intrinsic_reward_scaler = intrinsic_reward_scaler

    def _create_entropy_coefs(
        self,
        ent_coef :Union[str, float],
        target_entropy: Union[str, float],
        init_value: float,
    ):
        # Target entropy is used when learning the entropy coefficient
        if target_entropy == "auto":
            # automatically set target entropy if needed
            target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32) / 2
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            target_entropy = float(target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            if "_" in ent_coef:
                init_value = float(ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            log_ent_coef_def = LogEntropyCoef(init_value)
            self.rng, temp_key = jax.random.split(self.rng, 2)
            log_ent_coef = Model.create(
                log_ent_coef_def,
                inputs=[temp_key],
                tx=optax.adam(learning_rate=3e-4)
            )
            entropy_update = True

        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            log_ent_coef_def = LogEntropyCoef(ent_coef)
            self.rng, temp_key = jax.random.split(self.rng, 2)
            log_ent_coef = Model.create(log_ent_coef_def, inputs=[temp_key])
            entropy_update = False

        return log_ent_coef, target_entropy, entropy_update

    def _setup_model(self) -> None:
        super(SkillBasedHRLAgent, self)._setup_model()
        self._create_aliases()

        self.higher_log_ent_coef, self.higher_target_entropy, self.higher_entropy_update = \
                self._create_entropy_coefs(self.higher_ent_coef, self.skill_dim / 2, 1.0)

        self.lower_log_ent_coef, self.lower_target_entropy, self.lower_entropy_update = \
                self._create_entropy_coefs(self.lower_ent_coef, self.lower_target_entropy, 1.0)

    @contextmanager
    def hrl_phase(self):
        _train = self._train
        _without_exploration = self.without_exploration
        self._train = self.hrl_train
        self.without_exploration = False
        yield
        self._train = _train
        self.without_exploration = _without_exploration

    @property
    def save_lowlevel_episodes(self) -> bool:
        return True

    @property
    def last_obs(self) -> np.ndarray:
        return self._last_obs.copy()

    @property
    def higher_observation_space(self) -> gym.spaces.Space:
        return self.observation_space

    @property
    def lower_observation_space(self) -> gym.spaces.Space:
        return self.observation_space

    @property
    def higher_buffer_space(self) -> gym.spaces.Space:
        return self.observation_space

    @property
    def lower_buffer_space(self) -> gym.spaces.Space:
        return self.observation_space

    @property
    def lower_action_space(self) -> gym.spaces.Space:
        return self.action_space

    @property
    def higher_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-99, 99, (self.skill_dim,), dtype=np.float32)

    @property
    def lower_feature_extractor(self):
        return FlattenExtractor

    def get_higher_observation(self, obs):
        return obs

    def get_lower_observation(self, obs):
        return obs

    def get_higher_buffer_observation(self, obs, **kwargs):
        return obs

    def get_lower_buffer_observation(self, obs, **kwargs):
        return obs

    def get_representation(self, obs):
        return obs

    def build_replay_buffers(self, hrl_config: Dict) -> None:
        self.higher_replay_buffer = HigherReplayBuffer(
            buffer_size=hrl_config["higher_buffer_size"],
            observation_space=self.higher_buffer_space,
            higher_action_space=self.higher_action_space,
            lower_action_space=self.lower_action_space,
            subseq_len=self.subseq_len,
            n_envs=1
        )

        self.lower_replay_buffer = LowerReplayBuffer(
            buffer_size=hrl_config["lower_buffer_size"],
            observation_space=self.lower_buffer_space,
            higher_action_space=self.higher_action_space,
            lower_action_space=self.lower_action_space,
        )

    def build_higher_policy(self, higher_policy_config: Dict) -> None:
        # NOTE: Inside the build, higher policy is initialized by the pretrained skill prior
        self.higher_policy = HigherPolicy(
            rng=self.rng,
            observation_space=self.higher_observation_space,
            action_space=self.higher_action_space,
            lr_schedule=self.lr_schedule,
            net_arch=higher_policy_config["net_arch"],
            features_extractor_class=FlattenExtractor,
            n_critics=higher_policy_config["n_critics"],
            dropout=self.dropout
        )
        self.higher_policy.build_model(higher_policy_config)

    def build_lower_policy(self, lower_policy_config: Dict) -> None:
        self.lower_policy = LowerPolicy(
            rng=self.rng,
            observation_space=self.lower_observation_space,
            action_space=self.action_space,  # Target environment's action space
            conditioned_dim=self.skill_dim,
            lr_schedule=self.lr_schedule,
            net_arch=lower_policy_config["net_arch"],
            features_extractor_class=FlattenExtractor,
            n_critics=lower_policy_config["n_critics"],
            dropout=self.dropout
        )
        self.lower_policy.build_model(lower_policy_config)

    def load_skill_prior(self, pretrained_kwargs: Dict, load_dir: str, load_epoch: int):
        skill_prior_def = self.skill_prior_class(**pretrained_kwargs["skill_prior"])

        self.rng, rngs = get_basic_rngs(self.rng)
        init_obs = self.higher_observation_space.sample()[np.newaxis, ...]
        skill_prior = Model.create(skill_prior_def, inputs=[rngs, init_obs])  # Skill prior is not trained
        skill_prior = skill_prior.load_dict(load_dir + f"skill_prior_{load_epoch}")  # Load param
        skill_prior = skill_prior.load_batch_stats(
            load_dir + f"skill_prior_batch_stats_{load_epoch}")  # Load batch stats (for batch normalization)
        self.skill_prior = skill_prior
        print(f"Skill prior params are loaded from {load_dir}\n" * 10)

    def load_pseudo_action_policy(self, pretrained_kwargs: Dict, load_dir: str, load_epoch: int):
        init_obs = self.higher_observation_space.sample()[np.newaxis, ...]
        init_z = jax.random.normal(self.rng, shape=(1, self.skill_dim))
        pseudo_action_policy_def = PseudoActionPolicy(**pretrained_kwargs["lowlevel_policy"])
        self.rng, rngs = get_basic_rngs(self.rng)
        pseudo_action_policy = Model.create(pseudo_action_policy_def, inputs=[rngs, init_obs, init_z])  # Not optimized
        pseudo_action_policy = pseudo_action_policy.load_dict(load_dir + f"lowlevel_policy_{load_epoch}")
        pseudo_action_policy = pseudo_action_policy.load_batch_stats(
            load_dir + f"lowlevel_policy_batch_stats_{load_epoch}"
        )
        self.pseudo_action_policy = pseudo_action_policy
        print(f"Pseudo action params are loaded from {load_dir}\n" * 10)

    def build_hrl_models(self, hrl_config: Dict) -> int:
        # Avoid the mistake
        del self.policy, self.critic, self.critic_target

        # >>> Define replay buffer
        self.build_replay_buffers(hrl_config)

        # >>> Define higher policy
        self.rng, _ = jax.random.split(self.rng)
        higher_policy_config = hrl_config["higher_policy_config"]
        self.build_higher_policy(higher_policy_config)

        # >>> Define lower policy       # == action transfer
        self.rng, _ = jax.random.split(self.rng)
        lower_policy_config = hrl_config["lower_policy_config"]
        self.build_lower_policy(lower_policy_config)

        with open(hrl_config["config_dir"] + "config", "rb") as f:
            pretrained_kwargs = pickle.load(f)      # Pretrained networks architecture, batch_stats, etc., are saved
        self.normalizing_max = pretrained_kwargs["normalizing_max"]     # Normalizing factor, used to define pseudo actions.
        self.normalizing_min = pretrained_kwargs["normalizing_min"]

        if self.pseudo_action_dim is not None:
            self.normalizing_max = self.normalizing_max[:, :self.pseudo_action_dim]
            self.normalizing_min = self.normalizing_min[:, :self.pseudo_action_dim]

        if self.normalizing_max is None:
            self.normalizing_max = 1.0

        if self.normalizing_min is None:
            self.normalizing_min = -1.0

        # >>> Define skill prior (load from pretrained model)
        self.load_skill_prior(pretrained_kwargs, hrl_config["model_dir"], hrl_config["pretrained_model_load_epoch"])

        # >>> Define pseudo action policy
        # WARNING: Term 'lowlevel_policy' here means the pseudo action policy in skill prior training... # skill prior 학습할 땐 이름을 그렇게 붙였다.
        self.load_pseudo_action_policy(pretrained_kwargs, hrl_config["model_dir"], hrl_config["pretrained_model_load_epoch"])

        return hrl_config["total_timesteps"]

    # @clock(fmt="[{name}: {elapsed: 0.8f}s]")
    def hrl_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        higher_alpha_losses, higher_alphas = [], []
        higher_actor_losses, higher_critic_losses = [], []
        higher_prior_divergences = []

        lower_alpha_losses, lower_alphas = [], []
        lower_actor_losses, lower_critic_losses = [], []

        for gradient_step in range(gradient_steps):

            higher_replay_data = self.higher_replay_buffer.sample(batch_size)
            lower_replay_data = self.lower_replay_buffer.sample(batch_size)

            target_update_cond = ((self.num_timesteps % self.target_update_interval) == 0)
            if self.use_hiro_relabel:
                z = offpolicy_correction(
                    rng=self.rng,
                    lower_policy=self.lower_policy,
                    observations=higher_replay_data.observations,
                    lower_actions=higher_replay_data.lower_actions,
                    higher_actions=higher_replay_data.higher_actions,
                    skill_perturbing_std=0.1,
                    batch_size=batch_size,
                    skill_dim=self.skill_dim,
                    n_rand=10,
                )
                z = np.array(z)

            else: z = higher_replay_data.higher_actions

            self.rng, higher_new_models, higher_infos = hrl_higher_policy_update(
                rng=self.rng,

                actor=self.higher_policy.actor,
                critic=self.higher_policy.critic,
                critic_target=self.higher_policy.critic_target,
                log_alpha=self.higher_log_ent_coef,
                skill_prior=self.skill_prior,

                observations=self.get_representation(higher_replay_data.observations[:, 0, ...]),
                actions=z,
                rewards=higher_replay_data.rewards,
                next_observations=self.get_representation(higher_replay_data.next_observations),
                dones=higher_replay_data.dones,
                gamma=self.gamma,
                target_alpha=self.higher_target_entropy,
                tau=self.tau,
                target_update_cond=target_update_cond,
                alpha_update=self.higher_entropy_update,
            )
            self.higher_policy.actor = higher_new_models["higher_actor"]
            self.higher_policy.critic = higher_new_models["higher_critic"]
            self.higher_policy.critic_target = higher_new_models["higher_critic_target"]
            self.higher_log_ent_coef = higher_new_models["higher_log_alpha"]

            for _ in range(4):
                self.rng, lower_new_models, lower_infos = hrl_lower_policy_update(
                    rng=self.rng,
                    actor=self.lower_policy.actor,
                    critic=self.lower_policy.critic,
                    critic_target=self.lower_policy.critic_target,
                    log_alpha=self.lower_log_ent_coef,

                    observations=self.get_representation(lower_replay_data.observations),
                    actions=lower_replay_data.actions,
                    rewards=lower_replay_data.rewards,
                    next_observations=self.get_representation(lower_replay_data.next_observations),
                    dones=lower_replay_data.dones,
                    conditions=lower_replay_data.higher_actions,
                    next_conditions=lower_replay_data.next_higher_actions,

                    gamma=self.gamma,
                    target_alpha=self.lower_target_entropy,
                    tau=self.tau,
                    target_update_cond=target_update_cond,
                    alpha_update=self.lower_entropy_update
                )

                self.lower_policy.actor = lower_new_models["lower_actor"]
                self.lower_policy.critic = lower_new_models["lower_critic"]
                self.lower_policy.critic_target = lower_new_models["lower_critic_target"]
                self.lower_log_ent_coef = lower_new_models["lower_log_alpha"]


            higher_actor_losses.append(higher_infos["higher_actor_loss"].mean())
            higher_prior_divergences.append(higher_infos["prior_divergence"].mean())
            higher_critic_losses.append(higher_infos["higher_critic_loss"].mean())
            higher_alphas.append(higher_infos["higher_alpha"].mean())
            higher_alpha_losses.append(higher_infos["higher_alpha_loss"].mean())

            lower_actor_losses.append(lower_infos["lower_actor_loss"].mean())
            lower_critic_losses.append(lower_infos["lower_critic_loss"].mean())
            lower_alphas.append(lower_infos["lower_alpha"].mean())
            lower_alpha_losses.append(lower_infos["lower_alpha_loss"].mean())

        if self.num_timesteps % 100 == 0:
            self.logger.record_mean("higher/actor_loss(h)", np.mean(higher_actor_losses))
            self.logger.record_mean("higher/prior_div(h)", np.mean(higher_prior_divergences))
            self.logger.record_mean("higher/critic_loss(h)", np.mean(higher_critic_losses))
            self.logger.record_mean("higher/alpha_loss(h)", np.mean(higher_alpha_losses))
            self.logger.record_mean("higher/alpha(h)", np.mean(higher_alphas))

            self.logger.record_mean("lower/actor_loss(l)", np.mean(lower_actor_losses))
            self.logger.record_mean("lower/critic_loss(l)", np.mean(lower_critic_losses))
            self.logger.record_mean("lower/alpha_loss(l)", np.mean(lower_alpha_losses))
            self.logger.record_mean("lower/alpha(l)", np.mean(lower_alphas))

    def offline_train(self, gradient_steps: int, batch_size: int) -> None:
        return self._offline_train(gradient_steps, batch_size)

    def calculate_intrinsic_reward(self, **kwargs):
        return get_similarity_intrinsic_reward(**kwargs)

    # @clock(fmt="[{name}: {elapsed: 0.8f}s]")
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
        assert env.num_envs == 1

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Select action randomly or according to policy
            # NOTE: new_higher_actions: If timestep % subseq_len == 0, then it is same with 'z'. Otherwise, None.
            #    This triggers the higher replay buffer to save the transitions.
            actions, buffer_actions, z, new_higher_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, extrinsic_rewards, dones, infos = env.step(actions)
            extrinsic_rewards = extrinsic_rewards

            self.higher_actor_rewards += extrinsic_rewards

            # Sample intrinsic reward to train the lower level policy
            intrinsic_rewards = self.calculate_intrinsic_reward(
                    rng=self.rng,
                    representation=self.get_representation(self.last_obs.copy()),
                    next_representation=self.get_representation(new_obs),
                    normalizing_max=self.normalizing_max,
                    normalizing_min=self.normalizing_min,
                    z=self.higher_policy.last_z,
                    # z_var=self.higher_policy.last_z_var,
                    higher_actor=self.higher_policy.actor,
                    pseudo_action_policy=self.pseudo_action_policy,
                    deterministic=True,
                    training=False,
                    pseudo_action_dim=self.pseudo_action_dim
                )
            intrinsic_rewards = self.intrinsic_reward_scaler * intrinsic_rewards
            self.intrinsic_rewards.append(intrinsic_rewards)

            self.current_episode_timestep += env.num_envs
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            if self.current_episode_timestep % self.subseq_len == 0:
                new_higher_actions = self.higher_policy.last_z
                self.higher_policy.last_z = None        # Should sample a new higher action

            # Store data in replay buffer (normalized action and unnormalized observation)
            self.higher_replay_buffer.add(
                obs=self.get_higher_buffer_observation(self.last_obs.copy()),
                next_obs=self.get_higher_buffer_observation(new_obs.copy()),
                lower_action=actions.copy(),
                reward=extrinsic_rewards.copy(),
                done=dones.copy(),
                infos=infos.copy(),
                higher_action=new_higher_actions,
                cur_time=self.current_episode_timestep      # Debug
            )

            next_higher_action, _ = self.sample_higher_action(self.get_higher_observation(new_obs.copy()))

            # store_lower_transition 안에서 last_obs, 등이 update된다.
            self.store_lower_transition(
                buffer_actions,
                new_obs,
                intrinsic_rewards,
                dones,
                z,
                next_higher_action,
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
                    self.postprocess_episode_done()

                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1
                    # >>> Reset
                    self.current_episode_timestep = 0
                    self.higher_replay_buffer.temporal_reset()
                    self.higher_policy.last_z = None

                    self.logger.record_mean("train/intrinsic_rew", np.sum(self.intrinsic_rewards))
                    self.intrinsic_rewards = []

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def postprocess_episode_done(self):
        """Postprocess episode done. May be used to update higher policy if it is trained by on-policy."""
        pass

    def sample_higher_action(self, observation: jnp.ndarray):
        if self.higher_policy.last_z is None:
            z = self.higher_policy.predict(
                observation,
                deterministic=False,
                training=False,
                new_sampled=True,
                timestep=self.num_timesteps
            )
            new_higher_action = z.copy()
            self.higher_policy.last_z = new_higher_action

        else:
            z = self.higher_policy.last_z
            new_higher_action = None
        return z, new_higher_action

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[jnp.ndarray, ...]:
        z, new_higher_action = self.sample_higher_action(self.get_higher_observation(self.last_obs))

        # # Select action randomly or according to policy
        # if self.num_timesteps < learning_starts:
        #     # unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        #     self.rng, _ = jax.random.split(self.rng)
        #     z = jax.random.uniform(self.rng, z.shape, minval=-2.0, maxval=2.0)
        #
        # else:
        #     # Note: when using continuous actions,
        #     # we assume that the policy uses tanh to scale the action
        #     # We use non-deterministic action in the case of SAC, for TD3, it does not matter
        #     pass

        unscaled_action = self.lower_policy.predict(
            self.get_lower_observation(self.last_obs),
            deterministic=False,
            conditions=z
        )

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.lower_policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.lower_policy.unscale_action(scaled_action)

        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action

        return action, buffer_action, z, new_higher_action

    def store_lower_transition(
        self,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        higher_action: np.ndarray,              # added
        next_higher_action: np.ndarray,         # added
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        NOTE: This adds the transition into the lower buffer.
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        :param higher_action:
        :param next_higher_action:
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

        self.lower_replay_buffer.add(
            self.get_lower_buffer_observation(self._last_original_obs),
            self.get_lower_buffer_observation(next_obs),
            buffer_action,

            higher_action,
            next_higher_action,

            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs.copy()
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SkillBasedHRLAgent",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        if total_timesteps == 0: return None

        return super(SkillBasedHRLAgent, self).learn(
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
        return super(SkillBasedHRLAgent, self)._excluded_save_params() \
               + ["actor", "critic", "critic_target", "log_ent_coef"]

    def _get_jax_save_params(self) -> Dict[str, Params]:
        """..."""

    def _get_jax_load_params(self) -> List[str]:
        return ['actor', 'critic', 'critic_target', 'log_ent_coef']
