from typing import Callable
from typing import Tuple, List, Any

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
from jax import dtypes
from jax import random
from tensorflow_probability.substrates import jax as tfp

from offline_baselines_jax.common.jax_layers import (
    Sequential,
    FlattenExtractor
)
from offline_baselines_jax.common.jax_layers import (
    create_mlp,
    BaseFeaturesExtractor
)
from offline_baselines_jax.common.preprocessing import preprocess_obs

DType = Any
LOG_STD_MAX = 2
LOG_STD_MIN = -10

MU_SCALING = 3.0
LOG_STD_SCALING = 2.0

tfd = tfp.distributions
tfb = tfp.bijectors


def uniform(scale=1e-2, dtype: DType = jnp.float_) -> Callable:
  """Builds an initializer that returns real uniformly-distributed random arrays.

  Args:
    scale: optional; the upper bound of the random distribution.
    dtype: optional; the initializer's default dtype.

  Returns:
    An initializer that returns arrays whose values are uniformly distributed in
    the range ``[0, scale)``.

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.uniform(10.0)
  >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)  # doctest: +SKIP
  DeviceArray([[7.298188 , 8.691938 , 8.7230015],
               [2.0818567, 1.8662417, 5.5022564]], dtype=float32)
  """
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return (random.uniform(key, shape, dtype) * 2 - 1) * scale
  return init


class LeakyReLu:
    def __init__(self, negative_slope: float = 1e-2):
        self.negative_slope = negative_slope

    def __call__(self, *args, **kwargs):
        return nn.leaky_relu(*args, **kwargs, negative_slope=self.negative_slope)


class ImgExtractor(nn.Module):
    img_shape: int = 32
    n_channel: int = 3
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        return x.reshape((x.shape[0], self.img_shape, self.img_shape, self.n_channel))


class CondVaeGoalGenerator(nn.Module):
    recon_dim: int          # One side length of image shape
    latent_dim: int
    dropout: float
    kernel_size: int
    strides: int
    features: list

    enc_conv = None
    dec_conv = None
    encoder = None
    decoder = None
    mu = None
    log_std = None

    def setup(self):
        modules = []
        for i in range(len(self.features)):
            modules.append(
                nn.Conv(
                    features=self.features[i],
                    kernel_size=[self.kernel_size, self.kernel_size],
                    strides=self.strides
                )
            )
            modules.append(nn.relu)
        modules.append(FlattenExtractor())      # Note Fix here

        self.enc_conv = Sequential(modules)
        self.encoder = create_mlp(
            output_dim=self.latent_dim,
            net_arch=[256, 128],
            dropout=self.dropout
        )
        self.mu = create_mlp(
            output_dim=self.latent_dim,
            net_arch=[64, 32],
            dropout=self.dropout
        )
        self.log_std = create_mlp(
            output_dim=self.latent_dim,
            net_arch=[64, 32],
            dropout=self.dropout
        )
        self.decoder = Sequential([
            create_mlp(
                output_dim=3 * self.recon_dim * self.recon_dim,
                net_arch=[128, 256, 512, 1024],
                dropout=self.dropout
            ),
            ImgExtractor(img_shape=self.recon_dim, n_channel=3)
        ])


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,              # This is not an image but (x, y) pos.
        target_future_hop: jnp.ndarray,
        deterministic: bool = False,
    ) -> [jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, ...]]:

        mu, log_std = self.encode(state, goal, target_future_hop, deterministic)  # Use deterministic encoder. No sampling
        latent = self.get_latent_vector(mu, log_std)
        recon = self.decode(state, goal, target_future_hop, latent=latent, deterministic=deterministic)

        return recon, latent, (mu, log_std)

    def encode(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,
        target_future_hop: jnp.ndarray,
        deterministic: bool = False
    ):
        """
        NOTE: Input history should be preprocessed before here, inside forward function.
        state: image based state.
        """
        proj = self.enc_conv(state)

        encoder_input = jnp.concatenate((proj, goal, target_future_hop), axis=1)
        emb = self.encoder(encoder_input, deterministic)
        mu = self.mu(emb, deterministic=deterministic)
        log_std = self.log_std(emb, deterministic=deterministic)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mu, log_std

    def decode(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,
        target_future_hop: jnp.ndarray,
        deterministic: bool,
        latent: np.ndarray = None
    ) -> jnp.ndarray:
        """
        This is conditional VAE. So we conditionally input a true goal state.
        Here, true goal state means the final observation of given trajectory.
        """
        if latent is None:
            mu, log_std = self.encode(state, goal, target_future_hop, deterministic)
            latent = self.get_latent_vector(mu, log_std)

        decoder_input = jnp.concatenate((latent, goal, target_future_hop), axis=1)
        recon = self.decoder(decoder_input)
        return recon

    def deterministic_sampling(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,
        target_future_hop: jnp.ndarray,
        deterministic: bool = True
    ):
        mu, log_std = self.encode(state, goal, target_future_hop, deterministic=True)
        recon = self.decode(state, goal, target_future_hop, deterministic=True, latent=mu)
        return recon, mu, (mu, log_std)

    def get_latent_vector(self, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        rng = self.make_rng("sampling")
        std = jnp.exp(log_std)
        latent = mu + std * jax.random.normal(rng, shape=mu.shape)
        return latent


class SensorBasedSingleStateDiscriminator(nn.Module):
    features_extractor: BaseFeaturesExtractor
    dropout: float
    net_arch: list = None

    latent_pi = None

    def setup(self):
        net_arch = self.net_arch if self.net_arch is not None else [32] * 2
        self.latent_pi = create_mlp(
            output_dim=1,
            net_arch=net_arch,
            activation_fn=nn.leaky_relu
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observation: jnp.ndarray, deterministic: bool = False):
        x = self.features_extractor(observation)
        y = self.latent_pi(x, deterministic=deterministic)
        y = jnp.clip(y, -10.0, 10.0)
        y = nn.sigmoid(y)
        return y


class SensorBasedDoubleStateDiscriminator(nn.Module):
    features_extractor: BaseFeaturesExtractor
    dropout: float
    net_arch: list = None

    latent_pi = None

    def setup(self):
        net_arch = self.net_arch if self.net_arch is not None else [256] * 3
        self.latent_pi = create_mlp(
            output_dim=1,
            net_arch=net_arch,
            activation_fn=nn.leaky_relu
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observation: jnp.ndarray, next_observation: jnp.ndarray, deterministic: bool = False):
        x1 = self.features_extractor(observation)
        x2 = self.features_extractor(next_observation)
        x = jnp.concatenate((x1, x2), axis=1)
        y = self.latent_pi(x, deterministic=deterministic)
        y = jnp.clip(y, -10.0, 10.0)
        y = nn.sigmoid(y)
        return y


class SensorBasedDoubleStateLastConditionedDiscriminator(nn.Module):
    features_extractor: BaseFeaturesExtractor = None
    dropout: float = 0.0
    net_arch: list = None

    latent_pi = None

    def setup(self):
        assert self.net_arch is not None
        net_arch = self.net_arch
        self.latent_pi = create_mlp(
            output_dim=1,
            net_arch=net_arch,
            activation_fn=nn.relu
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observation: jnp.ndarray,
        next_observation: jnp.ndarray,
        last_observation: jnp.ndarray,
        deterministic: bool = False
    ):
        x1 = self.features_extractor(observation)
        x2 = self.features_extractor(next_observation)
        x3 = self.features_extractor(last_observation)
        x = jnp.concatenate((x1, x2, x3), axis=1)
        y = self.latent_pi(x, deterministic=deterministic)
        y = jnp.clip(y, -10.0, 10.0)
        y = nn.sigmoid(y)
        return y


class SensorBasedSingleStateActionMatcherFromHighToLow(nn.Module):      # g: S x A_h x S --> A_l (A_h: Env action, A_l: Relabeled action)
    features_extractor: BaseFeaturesExtractor
    dropout: float
    highaction_dim: int
    squash_output: bool
    net_arch: list = None

    latent_pi = None

    def setup(self):
        net_arch = self.net_arch if self.net_arch is not None else [64] * 2
        self.latent_pi = create_mlp(
            output_dim=self.highaction_dim,
            net_arch=net_arch,
            activation_fn=nn.leaky_relu,
            dropout=self.dropout,
            squash_output=self.squash_output
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        high_obs: jnp.ndarray,
        low_act: jnp.ndarray,
        deterministic: bool = False,
    ):
        high_obs = self.features_extractor(high_obs)

        x = jnp.concatenate((high_obs, low_act), axis=-1)
        return self.latent_pi(x, deterministic=deterministic)


class NaiveSensorBasedBehaviorCloner(nn.Module):
    features_extractor: BaseFeaturesExtractor
    dropout: float
    lowaction_dim: int
    net_arch: list = None

    latent_pi = None

    def setup(self):
        net_arch = self.net_arch if self.net_arch is not None else [256] * 3
        self.latent_pi = create_mlp(
            output_dim=self.lowaction_dim,
            net_arch=net_arch,
            activation_fn=nn.leaky_relu,
            dropout=self.dropout,
            squash_output=True
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        deterministic: bool = False
    ):
        features = self.features_extractor(observations)
        return self.latent_pi(features, deterministic=deterministic)


class SensorBasedForwardDynamics(nn.Module):
    features_extractor: BaseFeaturesExtractor
    observation_dim: int
    dropout: float = 0.0
    squash_output: bool = False

    latent_pi = None

    def setup(self) -> None:
        self.latent_pi = create_mlp(
            output_dim=self.observation_dim,
            net_arch=[256 * 10] * 5,
            activation_fn=nn.leaky_relu,
            dropout=self.dropout,
            squash_output=self.squash_output
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, actions: np.ndarray, deterministic: bool = False):
        observations = self.features_extractor(observations)
        x = jnp.concatenate((observations, actions), axis=-1)
        return self.latent_pi(x, deterministic=deterministic)


class SensorBasedInverseDynamics(nn.Module):
    features_extractor: BaseFeaturesExtractor
    dropout: float
    highaction_dim: int
    squash_output: bool

    latent_pi = None

    def setup(self) -> None:
        self.latent_pi = create_mlp(
            output_dim=self.highaction_dim,
            net_arch=[256, 256, 256, 256, 256, 256],
            activation_fn=nn.leaky_relu,
            dropout=self.dropout,
            squash_output=self.squash_output
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, next_observations: jnp.ndarray, deterministic: bool = False):
        observations = self.features_extractor(observations)
        next_observations = self.features_extractor(next_observations)
        x = jnp.concatenate((observations, next_observations), axis=1)
        return self.latent_pi(x, deterministic=deterministic)


class LSTMSensorBasedForwardDynamics(nn.Module):
    """
    --- LSTM architecture
    """
    features_extractor: BaseFeaturesExtractor
    observation_space: gym.Space

    n_frames: int     # Length of history for the observation space
    batch_dim: int
    hidden_dim: int     # lstm hidden dimension
    output_dim: int     # output state dimension
    dropout: float = 0.0
    net_arch: List = None

    lstmcell = None
    mu = None
    log_std = None

    def setup(self) -> None:
        self.lstmcell = nn.LSTMCell()

        # Compute Gaussian mean-log stds
        self.mu = create_mlp(
            output_dim=self.output_dim,
            net_arch=[64, 64],
            dropout=self.dropout
        )
        self.log_std = create_mlp(
            output_dim=self.output_dim,
            net_arch=[32, 32],
            dropout=self.dropout
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def assert_init(self, rng: jnp.ndarray):
        rng, _ = jax.random.split(rng)
        carry = self.lstmcell.initialize_carry(rng, self.batch_dim, self.hidden_dim)
        return carry

    def forward(self, observations: jnp.ndarray, actions: jnp.ndarray, deterministic: bool = False):
        rng = self.make_rng("init")
        carry = self.assert_init(rng)
        output = None

        for i in range(self.n_frames):
            current_obs = preprocess_obs(observations[:, i, ...], self.observation_space, normalize_images=True)
            current_obs = self.features_extractor(current_obs)
            lstm_input = jnp.concatenate((current_obs, actions[:, i, ...]), axis=1)
            carry, output = self.lstmcell(carry, lstm_input)

        mu = self.mu(output, deterministic=deterministic)
        log_stds = self.log_std(output, deterministic=deterministic)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        return self.skills_from_params(mu, log_stds)

    def skills_from_params(
        self,
        mean_skills: jnp.ndarray,
        log_std: jnp.ndarray
    ):
        base_dist = tfd.MultivariateNormalDiag(loc=mean_skills, scale_diag=jnp.exp(log_std))
        sampling_dist = base_dist
        # sampling_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return sampling_dist


class DeterministicLSTMSensorBasedForwardDynamics(nn.Module):
    """
    --- LSTM architecture
    """
    features_extractor: BaseFeaturesExtractor
    observation_space: gym.Space

    n_frames: int     # NOTE: Length of history for the observation space. This should be fixed in runtime.
    batch_dim: Tuple
    hidden_dim: int     # lstm hidden dimension
    output_dim: int     # output state dimension
    dropout: float = 0.0
    net_arch: List = None

    lstmcell = None
    latent_pi = None

    def setup(self) -> None:
        self.lstmcell = nn.LSTMCell(activation_fn=nn.silu)

        # Compute Gaussian mean-log stds
        net_arch = self.net_arch or [256] * 5
        self.latent_pi = create_mlp(
            output_dim=self.output_dim,
            net_arch=net_arch,
            dropout=self.dropout,
            activation_fn=nn.silu
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def assert_init(self, rng: jnp.ndarray):
        rng, _ = jax.random.split(rng)
        carry = self.lstmcell.initialize_carry(rng, self.batch_dim, self.hidden_dim)
        return carry

    def forward(self, trajectory: jnp.ndarray, actions: jnp.ndarray, deterministic: bool = False):
        rng = self.make_rng("init")
        carry = self.assert_init(rng)
        output = None

        for i in range(self.n_frames):
            current_obs = preprocess_obs(trajectory[:, i, ...], self.observation_space, normalize_images=True)
            current_obs = self.features_extractor(current_obs)
            lstm_input = jnp.concatenate((current_obs, actions[:, i, ...]), axis=1)
            carry, output = self.lstmcell(carry, lstm_input)

        pred = self.latent_pi(output, deterministic=deterministic)

        return pred


class LSTMSubTrajectoryLastObsBasedSkillGenerator(nn.Module):
    """
    --- LSTM architecture
    NOTE: Input: Subtrajectory (Subseq) of expert demonstration + corresponding trajectory's last observation.
    NOTE: A skill is generated from this.
    """
    features_extractor: BaseFeaturesExtractor
    observation_space: gym.Space

    subseq_len: int
    batch_dim: int
    hidden_dim: int     # lstm hidden dimension
    output_dim: int     # skill dimension
    dropout: float = 0.0
    net_arch: List = None
    relu_slope: float = 0.2

    embed = None
    lstmcell = None
    lstm_batchnorm = None
    mu = None
    log_std = None

    def setup(self) -> None:
        self.embed = create_mlp(
            output_dim=128,
            net_arch=[],
            activation_fn=LeakyReLu(self.relu_slope),
            dropout=self.dropout,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )
        self.lstmcell = nn.LSTMCell(kernel_init=uniform(jnp.sqrt(1 / self.hidden_dim)))

        # Compute Gaussian mean and log stds
        net_arch = []
        self.mu = create_mlp(
            output_dim=self.output_dim * 2,
            net_arch=net_arch,
            activation_fn=LeakyReLu(self.relu_slope),
            dropout=self.dropout,
            batchnorm=False,
            squash_output=False,
            kernel_init=nn.initializers.xavier_normal()
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def assert_init(self, rng: jnp.ndarray):
        carry = self.lstmcell.initialize_carry(rng, (self.batch_dim,), self.hidden_dim)
        return carry

    def forward(
        self,
        observations: jnp.ndarray,      # subseq-len observation trajectory
        actions: jnp.ndarray,           # subseq-len action trajectory
        last_observation: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ):
        rng = self.make_rng("init")
        carry = self.assert_init(rng)
        last_obs = preprocess_obs(last_observation, self.observation_space, normalize_images=True)
        last_obs = self.features_extractor(last_obs)
        # last_obs = self.last_obs_compress(last_obs)       # Fixme

        output = None
        for t in range(self.subseq_len):
            current_obs = preprocess_obs(observations[:, t], self.observation_space, normalize_images=True)
            current_obs = self.features_extractor(current_obs)
            current_act = actions[:, t]
            features = jnp.concatenate((current_obs, current_act, last_obs), axis=-1)
            lstm_input = self.embed(features)
            carry, output = self.lstmcell(carry, lstm_input)

        _mu = self.mu(output, deterministic=deterministic, **kwargs)

        mu, log_stds = jnp.array_split(_mu, 2, axis=-1)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        return self.skills_from_params(mu, log_stds)

    def skills_from_params(
        self,
        mean_skills: jnp.ndarray,
        log_std: jnp.ndarray
    ):
        dist = tfd.MultivariateNormalDiag(loc=mean_skills, scale_diag=jnp.exp(log_std))
        return dist, mean_skills, log_std


class MLPSkillPrior(nn.Module):
    """
    Approximate the skill generator by inputting the current state
    To use KL-constraint, we output a distribution
    MLP type
    """
    features_extractor: BaseFeaturesExtractor
    observation_space: gym.Space

    latent_dim: int     # Output dim of hidden layer
    skill_dim: int
    log_std_coef: float = 2.0
    dropout: float = 0.0
    net_arch: List = None
    relu_slope: float = 0.2

    latent_pi = None
    mu = None
    log_std = None

    def setup(self) -> None:
        net_arch = self.net_arch
        self.latent_pi = create_mlp(
            output_dim=self.latent_dim,
            net_arch=net_arch,
            activation_fn=LeakyReLu(self.relu_slope),
            dropout=self.dropout,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )
        self.mu = create_mlp(
            output_dim=self.skill_dim,
            net_arch=[128],
            activation_fn=LeakyReLu(self.relu_slope),
            dropout=self.dropout,
            squash_output=False,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )
        self.log_std = create_mlp(
            output_dim=self.skill_dim,
            net_arch=[128],
            activation_fn=LeakyReLu(self.relu_slope),
            dropout=self.dropout,
            squash_output=False,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, deterministic: bool = False, **kwargs):
        # NOTE: observations: first timestep observation of given trajectory. This is not a trajectory type.
        observations = preprocess_obs(observations, self.observation_space, normalize_images=True)
        features = self.features_extractor(observations)
        latent_pi = self.latent_pi(features, **kwargs)

        mu = self.mu(latent_pi, deterministic=deterministic, **kwargs)

        log_stds = self.log_std(latent_pi, deterministic=deterministic, **kwargs)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        return self.skill_priors_from_params(mu, log_stds), mu, log_stds

    def skill_priors_from_params(self, mean_prior: jnp.ndarray, log_std: jnp.ndarray):
        dist = tfd.MultivariateNormalDiag(loc=mean_prior, scale_diag=jnp.exp(log_std))
        return dist


class EnsembleMLPSkillPrior(nn.Module):
    features_extractor: BaseFeaturesExtractor
    observation_space: gym.Space

    latent_dim: int  # Output dim of hidden layer
    skill_dim: int
    log_std_coef: float = 2.0
    dropout: float = 0.0
    net_arch: List = None
    relu_slope: float = 0.2
    n_skill_prior: int = 3

    skill_priors = None

    def setup(self) -> None:
        batch_priors = nn.vmap(
            MLPSkillPrior,
            in_axes=None,
            out_axes=1,
            variable_axes={"params":1, "batch_stats": 1},
            split_rngs={"params": True, "dropout": True},
            axis_size=self.n_skill_prior
        )
        self.skill_priors = batch_priors(
            self.features_extractor,
            self.observation_space,
            self.latent_dim,
            self.skill_dim,
            self.log_std_coef,
            self.dropout,
            self.net_arch,
            self.relu_slope
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, deterministic: bool = False, **kwargs):
        return self.skill_priors(observations, deterministic, **kwargs)


class SquashedMLPSkillPrior(nn.Module):
    """
    Approximate the skill generator by inputting the current state
    To use KL-constraint, we output a distribution
    MLP type
    """
    features_extractor: BaseFeaturesExtractor
    observation_space: gym.Space

    latent_dim: int     # Output dim of hidden layer
    skill_dim: int
    log_std_coef: float = 2.0
    dropout: float = 0.0
    net_arch: List = None
    relu_slope: float = 0.2

    latent_pi = None
    mu = None
    log_std = None

    def setup(self) -> None:
        net_arch = self.net_arch
        self.latent_pi = create_mlp(
            output_dim=self.latent_dim,
            net_arch=net_arch,
            activation_fn=LeakyReLu(self.relu_slope),
            # activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )

        self.mu = create_mlp(
            output_dim=self.skill_dim,
            net_arch=[128] * 1,
            activation_fn=LeakyReLu(self.relu_slope),
            # activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            squash_output=False,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )

        self.log_std = create_mlp(
            output_dim=self.skill_dim,
            net_arch=[128] * 1,
            activation_fn=LeakyReLu(self.relu_slope),
            # activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            squash_output=False,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, deterministic: bool = False, **kwargs):
        # NOTE: observations: first timestep observation of given trajectory. This is not a trajectory type.
        observations = preprocess_obs(observations, self.observation_space, normalize_images=True)
        features = self.features_extractor(observations)
        latent_pi = self.latent_pi(features, **kwargs)

        mu = self.mu(latent_pi, deterministic=deterministic, **kwargs)

        log_stds = self.log_std(latent_pi, deterministic=deterministic, **kwargs)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        return self.skill_priors_from_params(mu, log_stds), mu, log_stds

    def skill_priors_from_params(self, mean_prior: jnp.ndarray, log_std: jnp.ndarray):
        dist = tfd.MultivariateNormalDiag(loc=mean_prior, scale_diag=jnp.exp(log_std))
        dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Chain([
            tfb.Scale(scale=2.),
            tfb.Tanh()
        ]))
        return dist


class LowLevelSkillPolicy(nn.Module):
    # NOTE: == this is called the skill decoder in SpiRL (We, say 'pseudo action decoder')
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    lowlevel_action_dim: int
    net_arch: List[int]
    dropout: float = 0.0
    log_std_coef: float = None
    relu_slope: float = 0.2

    latent_pi = None
    mu = None
    log_std = None

    def setup(self) -> None:
        self.latent_pi = create_mlp(
            output_dim=64,
            net_arch=self.net_arch,
            activation_fn=LeakyReLu(self.relu_slope),
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )

        self.mu = create_mlp(
            output_dim=self.lowlevel_action_dim,
            net_arch=[128] * 2,
            batchnorm=True,
            activation_fn=LeakyReLu(self.relu_slope),
            kernel_init=nn.initializers.xavier_normal()
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, z: jnp.ndarray, deterministic: bool = False, **kwargs):
        # NOTE: observations are 'not' a trajectory. Just a size of [batch_size, observation_dim]
        assert observations.ndim == 2
        mean_actions, log_stds = self.get_action_dist_params(observations, z, deterministic=deterministic, **kwargs)
        return self.actions_from_params(mean_actions, log_stds), mean_actions, log_stds

    def get_action_dist_params(
        self,
        observations: jnp.ndarray,
        z: jnp.ndarray,
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        observations = preprocess_obs(observations, self.observation_space)
        features = self.features_extractor(observations)
        # Skill latent z is always conditioned
        features = jnp.concatenate((features, z), axis=1)
        latent_pi = self.latent_pi(features, deterministic=deterministic, **kwargs)

        mean_actions = self.mu(latent_pi, deterministic=deterministic, **kwargs)
        return mean_actions, jnp.zeros_like(mean_actions)

    def get_deterministic_actions(self, observations: jnp.ndarray, z: jnp.ndarray, deterministic: bool = False, **kwargs):
        _, mean_actions, _ = self.forward(observations, z, deterministic, **kwargs)
        return jnp.tanh(mean_actions)

    def actions_from_params(self, mean: jnp.ndarray, log_std: jnp.ndarray):
        base_dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        sampling_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return sampling_dist


class DeterministicLowLevelSkillPolicy(nn.Module):
    # NOTE: == this is called the skill decoder in SpiRL (We, say 'pseudo action decoder')
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    lowlevel_action_dim: int
    net_arch: List[int]
    dropout: float = 0.1

    latent_pi = None

    def setup(self) -> None:
        self.latent_pi = create_mlp(
            output_dim=self.lowlevel_action_dim,
            net_arch=[800] * 10,
            activation_fn=LeakyReLu(0.2),
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal(),
            dropout=self.dropout
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, observations: jnp.ndarray, z: jnp.ndarray, deterministic: bool = False, **kwargs):
        # NOTE: observations are 'not' a trajectory. Just a size of [batch_size, observation_dim]
        assert observations.ndim == 2
        observations = preprocess_obs(observations, self.observation_space)
        features = self.features_extractor(observations)
        # Skill latent z is conditioned
        features = jnp.concatenate((features, z), axis=1)

        return self.latent_pi(features, deterministic=deterministic, **kwargs)

    def repeated_dim_forward(
            self,
            observations: jnp.ndarray,
            z: jnp.ndarray,
            n_repeats: int,
            deterministic: bool = False,
            **kwargs
    ):
        # Note: The input observation has the shape [batch_size, repeated_size, observation_dim]
        # Note: Thus, z also must have the dimension 3
        observations = preprocess_obs(observations, self.observation_space)
        observations = jnp.repeat(observations[:, jnp.newaxis, ...], repeats=n_repeats, axis=1)
        z = jnp.repeat(z[:, jnp.newaxis, ...], repeats=n_repeats, axis=1)
        assert observations.ndim == 3
        # No apply the (flatten) features extractor.
        features = jnp.concatenate((observations, z), axis=2)
        return self.latent_pi(features, deterministic=deterministic, **kwargs)

PseudoActionPolicy = LowLevelSkillPolicy
DeterministicPseudoActionPolicy = DeterministicLowLevelSkillPolicy


class ProbabilisticActionTransferLayer(nn.Module):
    r"""
    This module maps pseudo actions to environment specific action
    i.e., pseudo action -> 'real' actions.
    """
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    net_arch: List[int]
    dropout: float = 0.0

    latent_pi = None
    mu = None
    log_std = None

    def setup(self) -> None:
        net_arch = self.net_arch if self.net_arch is not None else [256] * 2
        self.latent_pi = create_mlp(
            output_dim=32,
            net_arch=net_arch,
            activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            squash_output=False,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )

        action_dim = self.action_space.shape[-1]
        self.mu = create_mlp(
            output_dim=action_dim,
            net_arch=[256, 256],
            activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            squash_output=False,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )
        self.log_std = create_mlp(
            output_dim=action_dim,
            net_arch=[256, 256],
            activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            squash_output=False,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        observations: jnp.ndarray,
        pseudo_actions: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ):
        mean_actions, log_stds = self.get_action_dist_params(
            observations,
            pseudo_actions,
            deterministic=deterministic,
            **kwargs
        )
        return self.actions_from_params(mean_actions, log_stds), mean_actions, log_stds

    def get_action_dist_params(
        self,
        observations: jnp.ndarray,
        pseudo_actions: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        observations = preprocess_obs(observations, self.observation_space)
        features = self.features_extractor(observations)
        # Skill latent z is always conditioned
        features = jnp.concatenate((features, pseudo_actions), axis=1)
        latent_pi = self.latent_pi(features, deterministic=deterministic, **kwargs)

        mean_actions = self.mu(latent_pi, deterministic=deterministic, **kwargs)
        log_stds = self.log_std(latent_pi, deterministic=deterministic, **kwargs)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        return mean_actions, log_stds

    def actions_from_params(self, mean: jnp.ndarray, log_std: jnp.ndarray):
        base_dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        sampling_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return sampling_dist


class ImgVariationalAutoEncoder(nn.Module):
    observation_space: gym.spaces.Space
    feature_dim: int

    n_channel = None
    img_shape = None
    encoder = None
    mu = None
    log_std = None
    fc = None
    decoder = None

    def setup(self) -> None:
        self.img_shape = tuple(self.observation_space.shape)
        self.n_channel = self.img_shape[-1]
        self.encoder = nn.Sequential([
            nn.Conv(features=8, kernel_size=(8, 8), strides=(2, 2), padding="same"),
            nn.relu,
            nn.Conv(features=16, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            nn.relu,
            nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            nn.relu,
            FlattenExtractor(_observation_space=None)
        ])

        self.mu = create_mlp(self.feature_dim, net_arch=[], activation_fn=nn.relu)
        self.log_std = create_mlp(self.feature_dim, net_arch=[], activation_fn=nn.relu)

        self.fc = create_mlp(32 * 32 * 4, net_arch=[], activation_fn=nn.relu)
        self.decoder = nn.Sequential([
            nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(1, 1), padding=1),
            nn.relu,
            nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(1, 1), padding=1),
            nn.relu,
            nn.ConvTranspose(features=self.n_channel, kernel_size=(3, 3), strides=(1, 1), padding=1),
            nn.tanh
        ])

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, images: jnp.ndarray):
        """
        :param images: [batch_size, ..., channel]. It has range[-1, 1].
        """
        mean, log_std = self.get_latent_params(images)
        z = self.get_latent_vector(mean, log_std)
        z = self.fc(z)
        z = z.reshape(z.shape[0], 32, 32, 4)
        decoded_img = self.decode(z)
        return decoded_img, (mean, log_std)

    def encode(self, images: jnp.ndarray):
        features = self.encoder(images)
        return features

    def decode(self, z: jnp.ndarray):
        return self.decoder(z)

    def get_latent_params(self, images: jnp.ndarray):
        features = self.encode(images)
        mean = self.mu(features)
        log_std = self.log_std(features)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def get_latent_vector(self, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        rng = self.make_rng("latent_sampling")
        std = jnp.exp(log_std)
        latent = mu + std * jax.random.normal(rng, shape=mu.shape)
        return latent
