import pickle
from abc import abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Callable

import flax.core
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
from offline_baselines_jax.common.jax_layers import (
    FlattenExtractor,
)
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.preprocessing import preprocess_obs
from offline_baselines_jax.common.type_aliases import Schedule
from offline_baselines_jax.common.utils import get_basic_rngs
from .networks import MLPSkillPrior, LowLevelSkillPolicy, ProbabilisticActionTransferLayer

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def dict_top_key_getter(fn: Callable):

    def map_fn(_dict):
        return {k: fn(k, v) for k, v in _dict.items()}
    return map_fn


def zero_grad_optimizer(*args):

    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)


@partial(jax.jit, static_argnames=("deterministic", "training"))
def _sample_actions(
    rng: jnp.ndarray,
    actor: Model,
    observations: jnp.ndarray,
    z: jnp.ndarray,
    deterministic: bool = True,
    training: bool = False,
):
    dropout_key, forwarding_key = jax.random.split(rng)
    dist, *_ = actor.apply_fn(
        {"params": actor.params, "batch_stats": actor.batch_stats},
        observations,
        z,
        deterministic=deterministic,
        training=training,
        rngs={"dropout": dropout_key, "forwarding_given_z": forwarding_key},
        method=ThreeComposedActor.forward_given_z
    )
    return dropout_key, dist.sample(seed=rng)


@partial(jax.jit, static_argnames=("deterministic", "training"))
def _sample_higher_actions(
    rng: jnp.ndarray,
    actor: Model,
    observations: jnp.ndarray,
    deterministic: bool = True,
    training: bool = False,
):
    rng, sampling_key, dropout_key = jax.random.split(rng, 3)
    z, _ = actor.apply_fn(
        {"params": actor.params, "batch_stats": actor.batch_stats},
        observations,
        deterministic=deterministic,
        training=training,
        rngs={"dropout": dropout_key, "sampling": sampling_key},
        method=ThreeComposedActor.sample_higher_action
    )
    return rng, z, actor.params


class CNNExtractor(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    feature_dim: int = 512

    @nn.compact
    def __call__(self, observations: jnp.array) -> jnp.array:
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(observations)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1)) # flatten

        x = nn.Dense(features=self.feature_dim)(x)
        x = nn.relu(x)
        return x


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
        deterministic: bool = False,
        training: bool = True,
        **kwargs
    ):
        return self.q_networks(observations, actions, deterministic, training, **kwargs)


class ThreeComposedActor(nn.Module):
    """
    Compose three networks:
        1. Higher actor: state -> z
        2. Low level policy: (state, z) -> pseudo action (e.g., delta)
        3. Action transfer layer: (state, pseudo action) -> environment specific action

    Thus,
        input = state
        output = environment action

    The second low level policy is predefined and parameters are fixed during training.
    """

    higher_actor: nn.Module
    lowlevel_policy: nn.Module
    action_transfer: nn.Module

    def setup(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, deterministic: bool = False, training: bool = True, **kwargs):
        rng = self.make_rng("forwarding")

        # pa: pseudo action
        z_sampling_key, pa_sampling_key = jax.random.split(rng)

        higher_action_dist, higher_mean, higher_log_std = self.higher_actor(
            observations=observations,
            deterministic=deterministic,
            training=training,
            **kwargs
        )
        # NOTE: 여기 sampling으로 하면 rl objective maximization이 잘 안되려나 ???????
        # z = higher_action_dist.sample(seed=z_sampling_key)
        z = higher_mean + jnp.exp(higher_log_std) * jax.random.normal(z_sampling_key, shape=higher_log_std.shape)

        pseudo_action_dist, pseudo_mean, pseudo_log_std = self.lowlevel_policy(
            observations=observations,
            z=z,
            deterministic=deterministic,
            training=training,
            **kwargs
        )

        pseudo_actions = pseudo_action_dist.sample(seed=pa_sampling_key)
        # pseudo_actions = pseudo_mean \
        #                  + jnp.exp(pseudo_log_std) \
        #                  * jax.random.normal(z_sampling_key, shape=pseudo_mean.shape)
        # pseudo_actions = tfb.Tanh(pseudo_actions)

        action_dist, action_mean, action_log_std = self.action_transfer(
            observations=observations,
            pseudo_actions=pseudo_actions,
            deterministic=deterministic,
            training=training,
            **kwargs
        )

        return action_dist, higher_action_dist

    def forward_given_z(
        self,
        observations: jnp.ndarray,
        z: jnp.ndarray,
        deterministic: bool = True,
        training: bool = False,
        **kwargs
    ):
        rng = self.make_rng("forwarding_given_z")
        pa_sampling_key, _ = jax.random.split(rng)
        pseudo_action_dist, pseudo_mean, pseudo_log_std = self.lowlevel_policy(
            observations=observations,
            z=z,
            deterministic=deterministic,
            training=training,
            **kwargs
        )

        pseudo_actions = pseudo_action_dist.sample(seed=pa_sampling_key)

        action_dist, action_mean, action_log_std = self.action_transfer(
            observations=observations,
            pseudo_actions=pseudo_actions,
            deterministic=deterministic,
            training=training,
            **kwargs
        )

        return action_dist, (action_mean, action_log_std)

    def forward_given_z_with_pseudo_params(
            self,
            observations: jnp.ndarray,
            z: jnp.ndarray,
            deterministic: bool = True,
            training: bool = False,
            **kwargs
    ):
        rng = self.make_rng("forwarding_given_z")
        pa_sampling_key, _ = jax.random.split(rng)
        pseudo_action_dist, pseudo_mean, pseudo_log_std = self.lowlevel_policy(
            observations=observations,
            z=z,
            deterministic=deterministic,
            training=training,
            **kwargs
        )

        pseudo_actions = pseudo_action_dist.sample(seed=pa_sampling_key)

        action_dist, action_mean, action_log_std = self.action_transfer(
            observations=observations,
            pseudo_actions=pseudo_actions,
            deterministic=deterministic,
            training=training,
            **kwargs
        )

        return action_dist, (action_mean, action_log_std), (pseudo_mean, pseudo_log_std)

    def sample_higher_action(self, observations: jnp.ndarray, deterministic: bool = False, training = False, **kwargs):
        rng = self.make_rng("sampling")
        higher_action_dist, higher_action_mean, higher_action_log_std = self.higher_actor(
            observations=observations,
            deterministic=deterministic,
            training=training,
            **kwargs
        )
        sampled_z = higher_action_mean \
                    + jnp.exp(higher_action_log_std) \
                    * jax.random.normal(key=rng, shape=higher_action_log_std.shape)

        return sampled_z, higher_action_dist

    def sample_higher_action_deterministically(
        self,
        observations: jnp.ndarray,
        deterministic: bool = False,
        training = False,
        **kwargs
    ):
        higher_action_dist, mu, log_stds = self.higher_actor(
            observations=observations,
            deterministic=deterministic,
            training=training,
            **kwargs
        )
        return mu       # No sampling

    def sample_higher_action_with_params(
        self,
        observations: jnp.ndarray,
        deterministic: bool = False,
        training: bool = True,
        **kwargs
    ):
        return self.higher_actor(observations=observations, deterministic=deterministic, training=training, **kwargs)


class SkillBasedComposedPolicy:
    """
    This class contains three components:
        1. Skill generator ( == 'Higher policy'): state -> z
        2. Low level policy: (state, z) -> pseudo action (e.g., delta). This is pretrained and will be loaded
        3. Action transfer layer: (state, pseudo action) -> environment specific action
    Compositing 1, 2, and 3, we construct a policy body

    NOTE 1: Middle level policy is pretrained using offline dataset
    NOTE 2: Skill generator (== 'Higher policy') will be regularized by skill prior
    """
    def __init__(
        self,
        rng,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        higher_action_dim: int,         # == skill dim
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        dropout: float = 0.0, *args, **kwargs
    ):
        self.rng = rng
        self.observation_space = observation_space
        self.action_space = action_space
        self.higher_action_dim = higher_action_dim          # == skill(z) dimension
        # self.pseudo_action_dim = self.observation_space.shape[-1]
        self.pseudo_action_dim = 999
        self.lr_schedule = lr_schedule
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        self.n_critics = n_critics
        self.dropout = dropout

        self.actor = None               # type: Model
        self.critic = None              # type: Model
        self.critic_target = None       # type: Model

        # Define pretrained components
        self.skill_prior = None         # type: Model
        self.lowlevel_policy = None     # type: Model

        self.z = None                   # Interacting with environments, z is fixed for update interval
        self.highlevel_update_interval = -1

    def build_policy(self, policy_build_config: Dict) -> None:
        """
        Build composed policy
            1. Higher actor
            2. Pseudo action low level policy
            3. Action transfer
        """
        pretrained_kwargs = self.load_pretrained_models(policy_build_config["pretrained_model_config"])

        self.highlevel_update_interval = policy_build_config["highlevel_update_interval"]

        init_obs = self.observation_space.sample()[np.newaxis, ...]
        init_act = self.action_space.sample()[np.newaxis, ...]

        # Define: Critic and critic target
        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        critic_def = Critic(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            net_arch=policy_build_config["critic_arch"],
            activation_fn=nn.leaky_relu,
            dropout=self.dropout,
            n_critics=policy_build_config["n_critics"]
        )
        self.critic = Model.create(critic_def, inputs=[rngs, init_obs, init_act], tx=optax.adam(1e-3))
        self.rng, rngs = get_basic_rngs(self.rng)
        self.critic_target = Model.create(critic_def, inputs=[rngs, init_obs, init_act])        # No optimize

        # Define: higher actor. If we implement the higher actor by TD3 style, then implement actor target in subclass
        # Note: We also initialize the higher actors parameter by skill prior ones. So they have a same structure
        self.rng, rngs = get_basic_rngs(self.rng)
        higher_actor = HigherActor(**pretrained_kwargs["skill_prior"])

        # Define: lowlevel skill policy.
        lowlevel_policy = LowLevelSkillPolicy(**pretrained_kwargs["lowlevel_policy"])

        # Define: transfer layer
        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = self.features_extractor_class(_observation_space=self.observation_space)
        action_transfer = ProbabilisticActionTransferLayer(
            features_extractor=features_extractor,
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=policy_build_config["action_transfer_arch"],
            dropout=self.dropout
        )

        # Define: three composed actor and load pretrained lowlevel policy's parameter
        f"""
        Note: 
            Although pretrained skill prior is instantiated by {Model} class, lowlevel policy is not instantiated.
            It only loaded in forms of parameter ({flax.core.FrozenDict} type). 
            See load_pretrained_models method.
        """
        action_transfer_learning_rate = 1e-4
        # NOTE: 이거는 debug용이라서 나중에 돌아가면 안된다 !!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@#!!@############################################
        pretr_transfer_layer_path = policy_build_config["pretrained_model_config"].get("bc_trained_transfer_dir", None)
        if pretr_transfer_layer_path:
            action_transfer_learning_rate = 0.0
        # NOTE: 여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!

        # Use multi transform to freeze the lowlevel policy parameters
        label_fn = dict_top_key_getter(lambda k, _: k)
        transforms = {
            "higher_actor": optax.adam(3e-4),
            "lowlevel_policy": zero_grad_optimizer(),
            "action_transfer": optax.adam(1e-4)
        }
        optimizer = optax.multi_transform(transforms, label_fn)

        self.rng, rngs = get_basic_rngs(self.rng)
        rngs.update({"forwarding": self.rng})
        actor_def = ThreeComposedActor(higher_actor, lowlevel_policy, action_transfer)
        self.actor = Model.create(actor_def, inputs=[rngs, init_obs], tx=optimizer)

        # >>> Copy the parameters from skill prior to highlevel policy
        source_params = self.skill_prior.params
        target_params = self.actor.params
        target_params["higher_actor"] = source_params
        source_batch_stats = self.skill_prior.batch_stats
        # Ugly !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@
        target_batch_stats = self.actor.batch_stats
        target_batch_stats = target_batch_stats.unfreeze()
        target_batch_stats["higher_actor"] = source_batch_stats.unfreeze()
        target_batch_stats = flax.core.frozen_dict.freeze(target_batch_stats)
        self.actor = self.actor.replace(params=target_params, batch_stats=target_batch_stats)

        # >>> 1. Copy the parameter of lowlevel policy
        # NOTE: We don't store lowlevel policy as a separated model. Just copy the parameters and
        # NOTE: include it as a part of an actor
        from flax.serialization import from_bytes
        source_params = dict()
        prtr_lowlevel_policy_path = policy_build_config["pretrained_model_config"]["model_dir"]
        print(f"Load lowlevel policy from {prtr_lowlevel_policy_path}")
        with open(prtr_lowlevel_policy_path + "lowlevel_policy", "rb") as f:
            lowlevel_policy_params = from_bytes(self.actor.params["lowlevel_policy"], f.read())
            source_params.update({"lowlevel_policy": lowlevel_policy_params})
        # loaded_params = self.actor.params.copy(source_params)
        loaded_params = source_params
        params = self.actor.params
        params["lowlevel_policy"] = loaded_params["lowlevel_policy"]
        self.actor = self.actor.replace(params=params)

        # >>> 2. Copy the batch stats of lowlevel policy
        source_batch_stats = dict()
        with open(prtr_lowlevel_policy_path + "lowlevel_policy_batch_stats", "rb") as f:
            lowlevel_policy_batch_stats = from_bytes(self.actor.batch_stats["lowlevel_policy"], f.read())
            source_batch_stats.update({"lowlevel_policy": lowlevel_policy_batch_stats})
        loaded_batch_stats = source_batch_stats
        batch_stats = self.actor.batch_stats.unfreeze()
        batch_stats["lowlevel_policy"] = loaded_batch_stats["lowlevel_policy"].unfreeze()
        batch_stats = flax.core.frozen_dict.freeze(batch_stats)
        self.actor = self.actor.replace(batch_stats=batch_stats)

        return      # NOTE: bc 학습할 때는 return 해줘야 이전에 했던 bc model을 load 안한다

        # NOTE: 이거는 debug용이라서 나중에 돌아가면 안된다 !!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@#!!@########################
        source_params = dict()
        if pretr_transfer_layer_path:
            # Debug 용
            print(f"We use bc-pretrained transfer layer from . This should be debug!\n" * 100)
            with open(pretr_transfer_layer_path + "transfer_layer", "rb") as f:
                action_transfer_params = from_bytes(self.actor.params, f.read())
                source_params.update({"action_transfer": action_transfer_params["action_transfer"]})
            loaded_params = source_params
            params = self.actor.params
            params["action_transfer"] = loaded_params["action_transfer"]
            self.actor = self.actor.replace(params=params)
        # NOTE: 여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!여기까지 돌아가면 안된다!

    def load_pretrained_models(self, pretrained_model_config: Dict) -> Dict:

        print("PRETRAINED load" * 999, pretrained_model_config["config_dir"] + "config")
        with open(pretrained_model_config["config_dir"] + "config", "rb") as f:
            pretrained_kwargs = pickle.load(f)

        # Check whether there exist desired parameters.
        assert "skill_prior" in pretrained_kwargs, "Skill prior must be loaded before rl training"
        assert "lowlevel_policy" in pretrained_kwargs, "Lowlevel policy must be loaded before rl training"

        init_obs = self.observation_space.sample()[np.newaxis, ...]

        # Define and load: Skill prior (s -> z). Higher actor will be regularized to this.
        skill_prior_def = MLPSkillPrior(**pretrained_kwargs["skill_prior"])
        self.rng, rngs = get_basic_rngs(self.rng)

        # Prior is not trained
        skill_prior = Model.create(skill_prior_def, inputs=[rngs, init_obs])

        skill_prior = skill_prior.load_dict(pretrained_model_config["model_dir"] + "skill_prior")
        self.skill_prior = skill_prior.load_batch_stats(pretrained_model_config["model_dir"] + "skill_prior_batch_stats")
        return pretrained_kwargs

    @contextmanager
    def given_skill(self, z: jnp.ndarray):
        self.z = z
        yield

    def sample_higher_action(
        self,
        observations: jnp.ndarray,
        deterministic: bool = False,
        training: bool = False
    ) -> jnp.ndarray:
        self.rng, higher_actions, actor_params = _sample_higher_actions(
            self.rng,
            self.actor,
            observations,
            deterministic=deterministic,
            training=training
        )
        return higher_actions

    def _predict(
        self,
        observations: jnp.ndarray,
        deterministic: bool = False,
        training: bool = False,
    ) -> jnp.ndarray:
        assert self.z is not None, "Higher skill should be fixed"
        rng, actions = _sample_actions(
            self.rng,
            self.actor,
            observations,
            self.z,
            deterministic=deterministic,
            training=training
        )
        self.rng = rng
        return np.asarray(actions)

    def predict(self, observation: jnp.ndarray, deterministic: bool = True) -> np.ndarray:
        actions = self._predict(observation, deterministic)
        if isinstance(self.action_space, gym.spaces.Box):
            # Actions could be on arbitrary scale, so clip the actions to avoid
            # out of bound error (e.g. if sampling from a Gaussian distribution)
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions

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
