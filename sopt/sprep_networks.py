# sprep: Skill Prior REProducing networks.

from typing import Tuple, List

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np

from tensorflow_probability.substrates import jax as tfp
from offline_baselines_jax.common.preprocessing import preprocess_obs
from offline_baselines_jax.common.jax_layers import (
    create_mlp,
    Sequential,
    FlattenExtractor,
    BaseFeaturesExtractor
)

LOG_STD_MAX = 2
LOG_STD_MIN = -10

MU_SCALING = 3.0
LOG_STD_SCALING = 2.0

tfd = tfp.distributions
tfb = tfp.bijectors


class LeakyReLu:
    def __init__(self, negative_slope: float = 1e-2):
        self.negative_slope = negative_slope

    def __call__(self, *args, **kwargs):
        return nn.leaky_relu(*args, **kwargs, negative_slope=self.negative_slope)


class LSTMSubTrajectoryBasedSkillGenerator(nn.Module):
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

    embed = None
    lstmcell = None
    lstm_batchnorm = None
    mu = None
    log_std = None

    def setup(self) -> None:
        self.embed = create_mlp(
            output_dim=self.output_dim,
            net_arch=[],
            activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )
        self.lstmcell = nn.LSTMCell()

        # Compute Gaussian mean and log stds
        net_arch = []
        self.mu = create_mlp(
            output_dim=self.output_dim,
            net_arch=net_arch,
            activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            batchnorm=False,
            squash_output=False,
            kernel_init=nn.initializers.xavier_normal()
        )

        self.log_std = create_mlp(
            output_dim=self.output_dim,
            net_arch=net_arch,
            activation_fn=LeakyReLu(0.2),
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
        deterministic: bool = False,
        **kwargs
    ):
        rng = self.make_rng("init")
        carry = self.assert_init(rng)
        # last_obs = self.last_obs_compress(last_obs)       # Fixme

        output = None
        for t in range(self.subseq_len):
            current_obs = preprocess_obs(observations[:, t, ...], self.observation_space, normalize_images=True)
            current_obs = self.features_extractor(current_obs)
            current_act = actions[:, t, ...]
            features = jnp.concatenate((current_obs, current_act), axis=-1)
            lstm_input = self.embed(features)
            carry, output = self.lstmcell(carry, lstm_input)

        mu = self.mu(output, deterministic=deterministic, **kwargs)

        log_stds = self.log_std(output, deterministic=deterministic, **kwargs)
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

    latent_pi = None
    mu = None
    log_std = None

    def setup(self) -> None:
        net_arch = self.net_arch
        self.latent_pi = create_mlp(
            output_dim=self.latent_dim,
            net_arch=net_arch,
            activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )
        self.mu = create_mlp(
            output_dim=self.skill_dim,
            net_arch=[256] * 2,
            activation_fn=LeakyReLu(0.2),
            dropout=self.dropout,
            squash_output=False,
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )
        self.log_std = create_mlp(
            output_dim=self.skill_dim,
            net_arch=[256] * 2,
            activation_fn=LeakyReLu(0.2),
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


class LowLevelSkillPolicy(nn.Module):
    # NOTE: == this is called the skill decoder in SpiRL (We, say 'pseudo action decoder')
    features_extractor: nn.Module
    observation_space: gym.spaces.Space
    lowlevel_action_dim: int
    net_arch: List[int]
    dropout: float = 0.0

    latent_pi = None
    mu = None
    log_std = None

    def setup(self) -> None:
        self.latent_pi = create_mlp(
            output_dim=64,
            net_arch=self.net_arch,
            activation_fn=LeakyReLu(0.2),
            batchnorm=True,
            kernel_init=nn.initializers.xavier_normal()
        )

        self.mu = create_mlp(
            output_dim=self.lowlevel_action_dim,
            net_arch=[128] * 2,
            batchnorm=True,
            activation_fn=LeakyReLu(0.2),
            kernel_init=nn.initializers.xavier_normal()
        )

        # NOTE: In spirl, log_std is fixed to zero. So log_std layer is not used.
        self.log_std = create_mlp(
            output_dim=self.lowlevel_action_dim,
            net_arch=[128] * 2,
            batchnorm=True,
            activation_fn=LeakyReLu(0.2),
            squash_output=True,
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
        log_stds = self.log_std(latent_pi, deterministic=deterministic, **kwargs)

        log_stds = log_stds
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        return mean_actions, log_stds

    def actions_from_params(self, mean: jnp.ndarray, log_std: jnp.ndarray):
        base_dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        sampling_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return sampling_dist
