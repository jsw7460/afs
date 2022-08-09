import time
import pickle
from abc import abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Callable, Union

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

from offline_baselines_jax.common.jax_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)
from offline_baselines_jax.common.jax_layers import FlattenExtractor, polyak_update
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.preprocessing import preprocess_obs, get_action_dim
from offline_baselines_jax.common.type_aliases import Schedule
from offline_baselines_jax.common.utils import get_basic_rngs
from .networks import MLPSkillPrior, SquashedMLPSkillPrior, LowLevelSkillPolicy
from .utils import clock


from .ra_networks import RASquashedMLPSkillPrior, RALowLevelSkillPolicy, RAMLPSkillPrior

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MAX = 2
LOG_STD_MIN = -10


class LeakyReLu:
    def __init__(self, negative_slope: float = 1e-2):
        self.negative_slope = negative_slope

    def __call__(self, *args, **kwargs):
        return nn.leaky_relu(*args, **kwargs, negative_slope=self.negative_slope)


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


@partial(jax.jit, static_argnames=("deterministic", ))
def _sample_higher_actions(
    rng: jnp.ndarray,
    higher_actor: Model,
    observations: jnp.ndarray,
    deterministic: bool = True,
):
    rng, sampling_key, dropout_key = jax.random.split(rng, 3)
    action_dist, mu, log_std = higher_actor.apply_fn(
        {"params": higher_actor.params, "batch_stats": higher_actor.batch_stats},
        observations,
        deterministic=deterministic,
        training=False,
        rngs={"dropout": dropout_key, "sampling": sampling_key},
    )
    z = action_dist.sample(seed=sampling_key)
    return rng, z, (mu, log_std)


@partial(jax.jit, static_argnames=("deterministic", ))
def _sample_lower_actions(
    rng: jnp.ndarray,
    lower_actor: Model,
    observations: jnp.ndarray,
    conditions: jnp.ndarray,
    deterministic: bool = True,
):
    rng, sampling_key, dropout_key = jax.random.split(rng, 3)
    action_dist, mu, log_std = lower_actor.apply_fn(
        {"params": lower_actor.params},
        observations,
        conditions,
        deterministic=deterministic,
        rngs={"dropout": dropout_key}
    )
    actions = action_dist.sample(seed=rng)
    return rng, (actions, mu, log_std)


@partial(jax.jit, static_argnames=("deterministic", ))
def batchnorm_sample_lower_actions(
    rng: jnp.ndarray,
    lower_actor: Model,
    observations: jnp.ndarray,
    conditions: jnp.ndarray,
    deterministic: bool = True,
):
    rng, sampling_key, dropout_key = jax.random.split(rng, 3)
    action_dist, actor_mu, actor_log_std = lower_actor.apply_fn(
        {"params": lower_actor.params, "batch_stats": lower_actor.batch_stats},
        observations,
        conditions,
        deterministic=deterministic,
        rngs={"dropout": dropout_key},
        training=False
    )
    actions = action_dist.sample(seed=rng)
    return rng, (actions, jnp.tanh(actor_mu), actor_log_std)


@partial(jax.jit, static_argnames=("deterministic", ))
def _sample_lower_action_params(
    rng: jnp.ndarray,
    lower_actor: Model,
    observations: jnp.ndarray,
    conditions: jnp.ndarray,
    deterministic: bool = True,
):
    rng, sampling_key, dropout_key = jax.random.split(rng, 3)
    action_dist, *_ = lower_actor.apply_fn(
        {"params": lower_actor.params},
        observations,
        conditions,
        deterministic=deterministic,
        rngs={"dropout": dropout_key}
    )
    return rng, action_dist


@partial(jax.jit)
def _sample_lower_actions_from_qfunc(
    rng: jnp.ndarray,
    q_func: Model,
    observations: jnp.ndarray,
    conditions: jnp.ndarray,
    deterministic: bool = True
):
    rng, sampling_key, dropout_key = jax.random.split(rng, 3)
    q_values = q_func.apply_fn(
        {"params": q_func.params},
        observations,
        conditions,
        deterministic=True,
        training=True
    )
    action = jnp.argmax(q_values, axis=1).reshape(-1)
    return rng, action

@jax.jit
def _log_probs(action_dist, actions: jnp.ndarray):
    return action_dist.log_prob(actions)


def zero_grad_optimizer(*args):

    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)


class ConditionedActor(nn.Module):      # Lowlevel Policy
    """Conditions(?): goal, latent z, ..."""
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    net_arch: List[int]
    activation_fn: Type[nn.Module] = nn.relu
    dropout: float = 0.0

    latent_pi = None
    mu = None
    log_std = None

    def setup(self):
        self.latent_pi = create_mlp(-1, self.net_arch, self.activation_fn, self.dropout, kernel_init=nn.initializers.xavier_normal())
        action_dim = get_action_dim(self.action_space)
        self.mu = create_mlp(action_dim, self.net_arch, self.activation_fn, self.dropout, kernel_init=nn.initializers.xavier_normal())
        self.log_std = create_mlp(action_dim, self.net_arch, self.activation_fn, self.dropout, kernel_init=nn.initializers.xavier_normal())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        conditions: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> Tuple[jnp.ndarray, ...]:
        mean_actions, log_stds = self.get_action_dist_params(
            observations,
            conditions,
            deterministic=deterministic,
            **kwargs
        )
        return self.actions_from_params(mean_actions, log_stds), mean_actions, log_stds

    def get_action_dist_params(
        self,
        observations: jnp.ndarray,
        conditions: jnp.ndarray,
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        observations = preprocess_obs(observations, self.observation_space)
        features = self.features_extractor(observations, **kwargs)
        features = jnp.concatenate((features, conditions), axis=-1)

        latent_pi = self.latent_pi(features, deterministic=deterministic)
        mean_actions = self.mu(latent_pi, deterministic=deterministic)
        log_stds = self.log_std(latent_pi, deterministic=deterministic)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        return mean_actions, log_stds

    def actions_from_params(self, mean: jnp.ndarray, log_std: jnp.ndarray):
        base_dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        sampling_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return sampling_dist

    def offpolicy_correction_max_indices(
        self,
        observations: jnp.ndarray,
        conditions: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ):
        """
        :param observations: [batch_size, subseq_len, obs_dim]
        :param conditions: [batch_size, n_skills, skill_dim]
        :param actions: [batch_size, subseq_len, action_dim]
        :param deterministic:
        :param kwargs:
        :return:
        먼저 [b , subseq_len, n_skills, obs+skill dim]를 만듦
         --> [b, n_skills, subseq_len] 개의 distributions
         --> 여기에 action의 logprob 구하면: [b, n_skills, subseq_len] 만큼의 실수 나옴
         마지막 dimension으로 더하면 log_prob 곱해진 것
         --> skill dimension으로 max 때리면 끝 ~!
        """
        subseq_len = observations.shape[1]
        n_skills = conditions.shape[1]

        observations = preprocess_obs(observations, self.observation_space)
        observations = jnp.repeat(observations[:, :, jnp.newaxis, ...], repeats=n_skills, axis=2)
        conditions = jnp.repeat(conditions[:, jnp.newaxis, ...], repeats=subseq_len, axis=1)
        actions = jnp.repeat(actions[:, :, jnp.newaxis, ...], repeats=n_skills, axis=2)

        features = jnp.concatenate((observations, conditions), axis=3)      # [batch, subseq_len, n_skills, obs+skill dim]
        latent_pi = self.latent_pi(features, deterministic=deterministic)
        mean_actions = self.mu(latent_pi, deterministic=deterministic)
        log_stds = self.log_std(latent_pi, deterministic=deterministic)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        base_dist = tfd.MultivariateNormalDiag(loc=mean_actions, scale_diag=jnp.exp(log_stds))

        log_probs = base_dist.log_prob(actions)     # [batch, subseq_len, n_skills]
        log_probs = jnp.sum(log_probs, axis=1)      # [batch, n_skills]

        return jnp.argmax(log_probs, axis=1)


HigherActor = MLPSkillPrior     # Same architecture !


class SingleCritic(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.relu

    q_net = None

    def setup(self):
        self.q_net = create_mlp(
            output_dim=1,
            net_arch=self.net_arch,
            dropout=self.dropout,
            batchnorm=False
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = False,
        training: bool = True,
        **kwargs,
    ):
        observations = preprocess_obs(observations, self.observation_space)

        features = self.features_extractor(observations)
        q_input = jnp.concatenate((features, actions), axis=1)
        return self.q_net(q_input, deterministic=deterministic, training=training, **kwargs)


class DiscreteSingleConditionedCritic(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Discrete
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.relu

    q_net = None

    def setup(self):
        action_dim = self.action_space.n
        self.q_net = create_mlp(
            output_dim=action_dim,
            net_arch=self.net_arch,
            dropout=self.dropout,
            batchnorm=False,
            kernel_init=nn.initializers.xavier_normal()
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        conditions: jnp.ndarray,
        deterministic: bool = False,
        training: bool = True,
        **kwargs,
    ):
        observations = preprocess_obs(observations, self.observation_space)
        features = self.features_extractor(observations)
        q_input = jnp.concatenate((features, conditions), axis=-1)
        return self.q_net(q_input, deterministic=deterministic, training=training, **kwargs)


class SingleConditionedCritic(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    net_arch: List[int]
    dropout: float
    activation_fn: Type[nn.Module] = nn.relu

    q_net = None

    def setup(self):
        self.q_net = create_mlp(
            output_dim=1,
            net_arch=self.net_arch,
            dropout=self.dropout,
            batchnorm=False,
            kernel_init=nn.initializers.xavier_normal()
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        conditions: jnp.ndarray,
        deterministic: bool = False,
        training: bool = True,
        **kwargs,
    ):
        observations = preprocess_obs(observations, self.observation_space)
        features = self.features_extractor(observations)
        features = jnp.concatenate((features, conditions), axis=-1)
        q_input = jnp.concatenate((features, actions), axis=1)
        return self.q_net(q_input, deterministic=deterministic, training=training, **kwargs)


class Critic(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    net_arch: List[int]
    dropout: float = 0.0
    activation_fn: Type[nn.Module] = nn.relu
    n_critics: int = 2

    q_networks = None

    def setup(self):
        batch_qs = nn.vmap(
            SingleCritic,
            in_axes=None,
            out_axes=1,                                             # 1
            variable_axes={"params": 1, "batch_stats": 1},          # 1
            split_rngs={"params": True, "dropout": True},
            axis_size=self.n_critics
        )
        self.q_networks = batch_qs(
            self.features_extractor,
            self.observation_space,
            self.net_arch,
            self.dropout,
            self.activation_fn
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        deterministic: bool = False,
        training: bool = True,
        **kwargs
    ):
        return self.q_networks(observations, actions, deterministic, training, **kwargs)


class DiscreteConditionedCritic(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    net_arch: List[int]
    dropout: float = 0.0
    activation_fn: Type[nn.Module] = nn.relu
    n_critics: int = 2

    q_networks = None

    def setup(self):
        batch_qs = nn.vmap(
            DiscreteSingleConditionedCritic,
            in_axes=None,
            out_axes=1,  # 1
            variable_axes={"params": 1, "batch_stats": 1},  # 1
            split_rngs={"params": True, "dropout": True},
            axis_size=self.n_critics
        )
        self.q_networks = batch_qs(
            self.features_extractor,
            self.observation_space,
            self.action_space,
            self.net_arch,
            self.dropout,
            self.activation_fn
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
            self,
            observations: jnp.ndarray,
            conditions: jnp.ndarray,
            deterministic: bool = False,
            training: bool = True,
            **kwargs
    ):
        return self.q_networks(observations, conditions, deterministic, training, **kwargs)


class ConditionedCritic(nn.Module):
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    net_arch: List[int]
    dropout: float = 0.0
    activation_fn: Type[nn.Module] = nn.relu
    n_critics: int = 2

    q_networks = None

    def setup(self):
        batch_qs = nn.vmap(
            SingleConditionedCritic,
            in_axes=None,
            out_axes=1,                         # 1
            variable_axes={"params": 1, "batch_stats": 1},        # 1
            split_rngs={"params": True, "dropout": True},
            axis_size=self.n_critics
        )
        self.q_networks = batch_qs(
            self.features_extractor,
            self.observation_space,
            self.net_arch,
            self.dropout,
            self.activation_fn
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        conditions: jnp.ndarray,
        deterministic: bool = False,
        training: bool = True,
        **kwargs
    ):
        return self.q_networks(observations, actions, conditions, deterministic, training, **kwargs)


class BasePolicy:
    def __init__(
        self,
        rng: jnp.ndarray,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: nn.Module = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0,
        **kwargs
    ):
        self.rng = rng
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        self.n_critics = n_critics
        self.dropout = dropout

    @abstractmethod
    def build_model(self, configs: Dict):
        """build components: actor, critic, ..."""

    @abstractmethod
    def _predict(self, observations: jnp.ndarray, deterministic: bool = True, **kwargs):
        """Predict action given observations. This method should call the jit-compiled function."""

    def predict(self, observations: jnp.ndarray, deterministic: bool = True, **kwargs) -> jnp.ndarray:
        actions = self._predict(observations, deterministic, **kwargs)
        if isinstance(self.action_space, gym.spaces.Box):
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return np.array(actions)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


class HigherPolicy(BasePolicy):
    def __init__(
        self,
        rng: jnp.ndarray,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: nn.Module = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0,
    ):
        super(HigherPolicy, self).__init__(
            rng=rng,
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            n_critics=n_critics,
            dropout=dropout
        )
        self.actor: Model = None
        self.critic: Model = None
        self.critic_target: Model = None
        self.__last_z = None            # Store last latent vector
        self.__last_z_var = None        # Store last variance when predicting the last_z

    @property
    def last_z(self):
        return self.__last_z

    @property
    def last_z_var(self):
        return self.__last_z_var

    @last_z.setter
    def last_z(self, val):
        self.__last_z = val

    @last_z_var.setter
    def last_z_var(self, val):
        self.__last_z_var = val

    def build_model(self, configs: Dict):
        with open(configs["config_dir"] + "/config", "rb") as f:
            pretrained_kwargs = pickle.load(f)
        init_obs = self.observation_space.sample()[np.newaxis, ...]
        init_act = self.action_space.sample()[np.newaxis, ...]

        # Define and load: Skill prior (s -> z). Higher actor will be regularized to this.
        # We 'squash' the skill prior model to [-2, 2] as in SBMTRL
        # This doesn't make serious problem since tanh and identity are similar around the origin

        higher_policy_kwargs = pretrained_kwargs["skill_prior"]
        higher_policy_kwargs.update({"dropout": 0.0})           # We don't use dropout for higher policy.
        skill_prior_def = SquashedMLPSkillPrior(**higher_policy_kwargs)
        self.rng, rngs = get_basic_rngs(self.rng)

        # Note: Higher policy actor is initialized by skill prior network
        skill_prior = Model.create(skill_prior_def, inputs=[rngs, init_obs], tx=optax.adam(1e-4))
        # skill_prior = Model.create(skill_prior_def, inputs=[rngs, init_obs], tx=optax.adam(0))

        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("Higher actor params load from:", configs["model_dir"] + "skill_prior")

        load_epoch = configs["skill_prior_load_epoch"]
        skill_prior = skill_prior.load_dict(configs["model_dir"] + f"skill_prior_{load_epoch}")
        self.actor = skill_prior.load_batch_stats(configs["model_dir"] + f"skill_prior_batch_stats_{load_epoch}")
        assert check_nested_dict_numpy_equation(self.actor.params, skill_prior.params)
        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        higher_critic_def = Critic(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            net_arch=self.net_arch,
            dropout=self.dropout,
            activation_fn=LeakyReLu(0.2),
            n_critics=self.n_critics
        )
        self.critic = Model.create(higher_critic_def, inputs=[rngs, init_obs, init_act], tx=optax.adam(1e-5))
        # self.critic = Model.create(higher_critic_def, inputs=[rngs, init_obs, init_act], tx=optax.adam(0))
        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        higher_critic_target_def = Critic(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            net_arch=self.net_arch,
            dropout=self.dropout,
            activation_fn=LeakyReLu(0.2),
            n_critics=self.n_critics
        )
        self.critic_target = Model.create(
            higher_critic_target_def,
            inputs=[rngs, init_obs, init_act],
        )
        self.critic_target = polyak_update(self.critic, self.critic_target, tau=1.0)

    def _predict(self, observations: jnp.ndarray, deterministic: bool = True, **kwargs):
        register = kwargs.get("register", True)     # By default, save last z

        # Register is possible only for newly sampled z. The variable register is false only when evaluation is running.
        if register:
            assert kwargs["new_sampled"], "Casting undesired behavior"
        without_sampling = kwargs.get("without_sampling", False)

        rng, actions, (mu, log_std) = _sample_higher_actions(
            self.rng,
            self.actor,
            observations,
            deterministic
        )
        self.rng = rng
        if register:
            self.last_z = actions
            self.last_z_var = jnp.exp(log_std) ** 2

        if without_sampling:
            return np.array(2.0 * jnp.tanh(mu))

        return np.array(actions)


class LowerPolicy(BasePolicy):
    def __init__(
        self,
        rng: jnp.ndarray,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        conditioned_dim: int,       #  == goal dim, latent_z dim, ...
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: nn.Module = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0,
    ):
        super(LowerPolicy, self).__init__(
            rng=rng,
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            n_critics=n_critics,
            dropout=dropout
        )
        self.actor: Model = None
        self.critic: Model = None
        self.critic_target: Model = None
        self.conditioned_dim = conditioned_dim

    def build_model(self, configs: Dict = None):
        init_obs = self.observation_space.sample()[np.newaxis, ...]
        init_cond = jax.random.normal(self.rng, shape=(1, self.conditioned_dim))
        init_act = self.action_space.sample()[np.newaxis, ...]

        actor_lr = configs.get("actor_lr", 1e-3)
        critic_lr = configs.get("critic_lr", 5e-4)

        # actor_lr = 0.0
        # critic_lr = 0.0

        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        lower_actor_def = ConditionedActor(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=self.net_arch,
            activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
        )
        self.actor = Model.create(
            lower_actor_def,
            inputs=[rngs, init_obs, init_cond],
            tx=optax.adam(actor_lr)
        )

        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        lower_critic_def = ConditionedCritic(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            net_arch=self.net_arch,
            dropout=self.dropout,
            activation_fn=LeakyReLu(0.2),
            n_critics=self.n_critics
        )

        self.critic = Model.create(
            lower_critic_def,
            inputs=[rngs, init_obs, init_act, init_cond],
            tx=optax.adam(critic_lr)
        )

        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        lower_critic_target_def = ConditionedCritic(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            net_arch=self.net_arch,
            dropout=self.dropout,
            activation_fn=LeakyReLu(0.2),
            n_critics=self.n_critics
        )

        self.critic_target = Model.create(
            lower_critic_target_def,
            inputs=[rngs, init_obs, init_act, init_cond],
        )

        self.critic_target = polyak_update(self.critic, self.critic_target, 1.0)

    # @clock(fmt="[{name}: {elapsed: 0.8f}s]")
    def _predict(self, observations: jnp.ndarray, deterministic: bool = True, **kwargs):
        rng, (actions, mu, log_std) = _sample_lower_actions(
            self.rng,
            self.actor,
            observations,
            kwargs["conditions"],
            deterministic
        )
        self.rng = rng

        without_sampling = kwargs.get("without_sampling", False)
        if without_sampling:
            return np.array(jnp.tanh(mu))

        return np.array(actions)


class RAHigherPolicy(BasePolicy):
    def __init__(
        self,
        rng: jnp.ndarray,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: nn.Module = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0,
    ):
        super(RAHigherPolicy, self).__init__(
            rng=rng,
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            n_critics=n_critics,
            dropout=dropout
        )
        self.actor: Model = None
        self.critic: Model = None
        self.critic_target: Model = None
        self.__last_z = None            # Store last latent vector
        self.__last_z_var = None        # Store last variance when predicting the last_z

    @property
    def last_z(self):
        return self.__last_z

    @property
    def last_z_var(self):
        return self.__last_z_var

    @last_z.setter
    def last_z(self, val):
        self.__last_z = val

    @last_z_var.setter
    def last_z_var(self, val):
        self.__last_z_var = val

    def build_model(self, configs: Dict):
        with open(configs["config_dir"] + "config", "rb") as f:
            pretrained_kwargs = pickle.load(f)
        init_obs = self.observation_space.sample()[np.newaxis, ...]
        init_act = self.action_space.sample()[np.newaxis, ...]

        # Define and load: Skill prior (s -> z). Higher actor will be regularized to this.

        # NOTE: We use a 'squashed' version.
        # 즉, skill prior와 mu - log_std는 kl을 걸어주지만, action 자체는 squash해서 뽑아야 하기 때문에 이것을 사용한다.
        skill_prior_def = RASquashedMLPSkillPrior(**pretrained_kwargs["skill_prior"])
        self.rng, rngs = get_basic_rngs(self.rng)

        # Note: Higher policy actor is initialized by skill prior network
        skill_prior = Model.create(skill_prior_def, inputs=[rngs, init_obs], tx=optax.adam(1e-4))
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")
        print("RA: Higher actor params load from:", configs["model_dir"] + "skill_prior")

        load_epoch = configs["skill_prior_load_epoch"]
        skill_prior = skill_prior.load_dict(configs["model_dir"] + f"skill_prior_{load_epoch}")
        self.actor = skill_prior.load_batch_stats(configs["model_dir"] + f"skill_prior_batch_stats_{load_epoch}")
        assert check_nested_dict_numpy_equation(self.actor.params, skill_prior.params)
        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        higher_critic_def = Critic(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            net_arch=self.net_arch,
            dropout=self.dropout,
            activation_fn=LeakyReLu(0.2),
            n_critics=self.n_critics
        )
        self.critic = Model.create(higher_critic_def, inputs=[rngs, init_obs, init_act], tx=optax.adam(1e-4))
        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        higher_critic_target_def = Critic(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            net_arch=self.net_arch,
            dropout=self.dropout,
            activation_fn=LeakyReLu(0.2),
            n_critics=self.n_critics
        )
        self.critic_target = Model.create(
            higher_critic_target_def,
            inputs=[rngs, init_obs, init_act],
        )
        self.critic_target = polyak_update(self.critic, self.critic_target, 1.0)

    def _predict(self, observations: jnp.ndarray, deterministic: bool = True, **kwargs):
        register = kwargs.get("register", True)

        # Register is possible only for newly sampled z. The variable register is false only while evaluations.
        if register:
            assert kwargs["new_sampled"], "Casting undesired behavior"
        without_sampling = kwargs.get("without_sampling", False)

        rng, actions, (mu, log_std) = _sample_higher_actions(
            self.rng,
            self.actor,
            observations,
            deterministic
        )
        self.rng = rng
        if register:
            self.last_z = np.array(actions).copy()
            self.last_z_var = np.array(jnp.exp(log_std) ** 2).copy()

        if without_sampling:
            return np.array(2.0 * jnp.tanh(mu))
        return actions


class RALowerPolicy(BasePolicy):
    def __init__(
        self,
        rng: jnp.ndarray,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        conditioned_dim: int,           # == goal dim, latent_z dim, ...
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: nn.Module = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0,
    ):
        super(RALowerPolicy, self).__init__(
            rng=rng,
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            n_critics=n_critics,
            dropout=dropout
        )
        self.conditioned_dim = conditioned_dim
        assert self.conditioned_dim > 0, "Invalid conditional dimension"
        self.actor: Model = None

    def build_model(self, configs: Dict):
        # Load from pretrained dir
        with open(configs["config_dir"] + "config", "rb") as f:
            pretrained_kwargs = pickle.load(f)

        actor_def = RALowLevelSkillPolicy(**pretrained_kwargs["lowlevel_policy"])
        init_obs = self.observation_space.sample()[np.newaxis, ...]
        init_z = jax.random.normal(self.rng, shape=(1, self.conditioned_dim))
        self.rng, rngs = get_basic_rngs(self.rng)
        actor = Model.create(actor_def, inputs=[rngs, init_obs, init_z])        # No training
        actor = actor.load_dict(configs["model_dir"] + f"lowlevel_policy_{configs['lowlevel_policy_load_epoch']}")
        actor = actor.load_batch_stats(
            configs["model_dir"] + f"lowlevel_policy_batch_stats_{configs['lowlevel_policy_load_epoch']}"
        )
        self.actor = actor

    def _predict(self, observations: jnp.ndarray, deterministic: bool = True, **kwargs):
        rng, (actions, mu, log_std) = batchnorm_sample_lower_actions(
            rng=self.rng,
            lower_actor=self.actor,
            observations=observations,
            conditions=kwargs["conditions"],
            deterministic=deterministic
        )
        self.rng = rng
        return np.array(jnp.tanh(mu))


class DiscreteLowerPolicy(BasePolicy):
    def __init__(
        self,
        rng: jnp.ndarray,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        conditioned_dim: int,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.relu,
        features_extractor_class: nn.Module = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0,
    ):
        super(DiscreteLowerPolicy, self).__init__(
            rng=rng,
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            n_critics=n_critics,
            dropout=dropout
        )
        self.critic: Model = None
        self.critic_target: Model = None
        self.conditioned_dim = conditioned_dim

    def build_model(self, configs: Dict = None):
        init_obs = self.observation_space.sample()[np.newaxis, ...]
        init_cond = jax.random.normal(self.rng, shape=(1, self.conditioned_dim))

        # init_act = self.action_space.sample()[np.newaxis, ...]
        init_act = np.array(self.action_space.sample())[np.newaxis, np.newaxis, ...]

        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        lower_critic_def = DiscreteSingleConditionedCritic(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=self.net_arch,
            dropout=self.dropout,
            activation_fn=LeakyReLu(0.2),
        )

        self.critic = Model.create(
            lower_critic_def,
            inputs=[rngs, init_obs, init_cond],
            tx=optax.adam(2e-5)
        )

        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        lower_critic_target_def = DiscreteSingleConditionedCritic(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=self.net_arch,
            dropout=self.dropout,
            activation_fn=LeakyReLu(0.2),
        )
        self.critic_target = Model.create(
            lower_critic_target_def,
            inputs=[rngs, init_obs, init_cond],
        )

        self.critic_target = polyak_update(self.critic, self.critic_target, 1.0)

    def _predict(self, observations: jnp.ndarray, deterministic: bool = True, **kwargs):
        """
        DQN sample actions directly from q-function
        """
        rng, actions = _sample_lower_actions_from_qfunc(
            rng=self.rng,
            q_func=self.critic,
            observations=observations,
            conditions=kwargs["conditions"],
            deterministic=deterministic
        )
        self.rng = rng
        return np.array(actions)
