from functools import partial
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import Params, InfoDict
from .policies import ThreeComposedActor

tfd = tfp.distributions
tfb = tfp.bijectors

EPS = 1e-6
ALPHA = 2.5
POSITIVE_TARGET = 1
NEGATIVE_TARGET = 0
DISCRIMINATOR_BATCH_SIZE = 16

STD_MEAN = 0
STD_DEV = 1


def log_ent_coef_update(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    observations: jnp.ndarray,
    target_entropy: float,
) -> Tuple[Model, InfoDict]:

    def temperature_loss_fn(ent_params: Params):
        dist = actor(observations)
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = log_ent_coef.apply_fn({'params': ent_params})
        ent_coef_loss = -(ent_coef * (target_entropy + log_prob)).mean()

        return ent_coef_loss, {'ent_coef': ent_coef, 'ent_coef_loss': ent_coef_loss}

    new_ent_coef, info = log_ent_coef.apply_gradient(temperature_loss_fn)
    return new_ent_coef, info


def sac_actor_update(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
):
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({'params': actor_params}, observations)
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        ent_coef = jnp.exp(log_ent_coef())

        q_values_pi = critic(observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        return actor_loss, {'actor_loss': actor_loss, 'entropy': -log_prob}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def sac_critic_update(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float
):
    dist = actor(next_observations)
    next_actions = dist.sample(seed=rng)
    next_log_prob = dist.log_prob(next_actions)

    # Compute the next Q values: min over all critics targets
    next_q_values = critic_target(next_observations, next_actions)
    next_q_values = jnp.min(next_q_values, axis=1)

    ent_coef = jnp.exp(log_ent_coef())
    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network using action from the replay buffer
        q_values= critic.apply_fn({'params': critic_params}, observations, actions)

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs

        return critic_loss, {'critic_loss': critic_loss, 'current_q': q_values.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def sensor_based_single_state_discriminator_update(
    rng: jnp.ndarray,
    discriminator: Model,
    expert_observation: jnp.ndarray,
    observation: jnp.ndarray,
):
    dropout_key1, dropout_key2 = jax.random.split(rng)

    def discriminator_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:

        expert_score = discriminator.apply_fn(
            {"params": params},
            expert_observation,
            deterministic=False,
            rngs={"dropout": dropout_key1}
        )

        policy_score = discriminator.apply_fn(
            {"params": params},
            observation,
            deterministic=False,
            rngs={"dropout": dropout_key2}
        )

        # Bernoulli: Expert --> 1, Policy --> 0
        loss = - jnp.mean(jnp.log(expert_score) + jnp.log(1 - policy_score))

        return loss, {"single_discriminator_loss": loss, "single_expert_disc_score": expert_score, "single_policy_disc_score": policy_score}

    discriminator, info = discriminator.apply_gradient(discriminator_loss_fn)
    return discriminator, info


def sensor_based_double_state_discriminator_update(
    rng: jnp.ndarray,
    discriminator: Model,
    expert_observation: jnp.ndarray,
    expert_next_observation: jnp.ndarray,
    observation: jnp.ndarray,
    next_observation: jnp.ndarray,
):
    dropout_key1, dropout_key2 = jax.random.split(rng)
    def discriminator_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        expert_score = discriminator.apply_fn(
            {"params": params},
            expert_observation,
            expert_next_observation,
            deterministic=False,
            rngs={"dropout": dropout_key1}
        )

        policy_score = discriminator.apply_fn(
            {"params": params},
            observation,
            next_observation,
            deterministic=False,
            rngs={"dropout": dropout_key2}
        )

        # Bernoulli: Expert --> 1, Policy --> 0
        loss = - jnp.mean(jnp.log(expert_score) + jnp.log(1 - policy_score))

        return loss, {"double_discriminator_loss": loss, "double_expert_disc_score": expert_score, "double_policy_disc_score": policy_score}

    discriminator, info = discriminator.apply_gradient(discriminator_loss_fn)
    return discriminator, info


def sensor_based_double_state_last_conditioned_discriminator_update(
    rng: jnp.ndarray,
    discriminator: Model,
    expert_observation: jnp.ndarray,
    expert_next_observation: jnp.ndarray,
    expert_last_observation: jnp.ndarray,

    observation: jnp.ndarray,
    next_observation: jnp.ndarray,
    last_observation: jnp.ndarray,
):
    dropout_key1, dropout_key2 = jax.random.split(rng)
    def discriminator_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        expert_score = discriminator.apply_fn(
            {"params": params},
            expert_observation,
            expert_next_observation,
            expert_last_observation,
            deterministic=False,
            rngs={"dropout": dropout_key1}
        )[:DISCRIMINATOR_BATCH_SIZE, ...]

        policy_score = discriminator.apply_fn(
            {"params": params},
            observation,
            next_observation,
            last_observation,
            deterministic=False,
            rngs={"dropout": dropout_key2}
        )[:DISCRIMINATOR_BATCH_SIZE, ...]

        # Bernoulli: Expert --> 1, Policy --> 0
        loss = - jnp.mean(jnp.log(expert_score) + jnp.log(1 - policy_score))
        return loss, {"discriminator_loss": loss,
                      "expert_disc_score": expert_score.mean(),
                      "policy_disc_score": policy_score.mean()}

    discriminator, info = discriminator.apply_gradient(discriminator_loss_fn)
    return discriminator, info


# Critic update는 weight를 주지 않기로 결심하였다.
def disc_weighted_critic_update(
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float,
):
    next_dist = actor(next_observations)
    next_actions = next_dist.sample(seed=rng)
    next_log_prob = next_dist.log_prob(next_actions)

    # Compute the next Q values
    next_q_values = critic_target(next_observations, next_actions)
    next_q_values = jnp.min(next_q_values, axis=1)

    ent_coef = jnp.exp(log_ent_coef())
    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        q_values = critic.apply_fn({'params': params}, observations, actions)

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs

        return critic_loss, {'critic_loss': critic_loss, 'current_q': q_values.mean()}

    critic, info = critic.apply_gradient(critic_loss_fn)
    return critic, info


def disc_weighted_actor_update(
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    log_ent_coef: Model,
    behavior_cloner: Model,
    action_matcher: Model,

    policy_disc_score: Union[jnp.ndarray, float],
    observations: jnp.ndarray,
):
    low_actions = behavior_cloner(observations, deterministic=True)

    # 현재 observation에서, expert라면 했어야 할 action을 predict
    pred_high_actions = action_matcher(observations, low_actions)
    pred_high_actions = jnp.clip(pred_high_actions, -1 + EPS, 1 - EPS)

    ent_coef = jnp.exp(log_ent_coef())

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn({"params": params}, observations)
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)

        q_values_pi = critic(observations, actions_pi)
        min_qf_pi = jnp.min(q_values_pi, axis=1)

        sac_loss = jnp.mean(ent_coef * log_prob - min_qf_pi)
        supervised_loss = - jnp.mean(policy_disc_score * dist.log_prob(pred_high_actions))

        actor_loss = sac_loss + supervised_loss
        return actor_loss, {"actor_loss": actor_loss, "sac_loss": sac_loss, "supervised_loss": supervised_loss}

    actor, info = actor.apply_gradient(actor_loss_fn)
    return actor, info


def skill_generator_update(
    rng: jnp.ndarray,

    lowlevel_policy: Model,
    skill_generator: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    last_observations: jnp.ndarray,

):
    rng, dropout_key, init_key, batch_key, skill_sampling_key = jax.random.split(rng, 5)
    subseq_len = actions.shape[1]

    # Skill generator is trained by accelerating the behavior cloning of primitive action policy
    def skill_generator_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:

        z_dist, skill_mu, skill_log_std = skill_generator.apply_fn(
            {"params": params},
            observations,
            actions,
            last_observation=last_observations,
            rngs={"dropout": dropout_key, "init": init_key},
        )
        skill_std = jnp.exp(skill_log_std)

        # Compute kl-divergence loss (to N(0, 1))
        kl_loss = (jnp.log(STD_DEV) - skill_log_std) \
                  + skill_std ** 2 + (skill_mu - STD_MEAN) ** 2 \
                  / (2 * STD_DEV ** 2) - 0.5
        kl_loss = jnp.mean(kl_loss)

        # Sample skill making latent vector z (reparameterize)
        # z = skill_mu + skill_std * jax.random.normal(skill_sampling_key, shape=skill_mu.shape)
        nongrad_z = z_dist.sample(seed=skill_sampling_key)
        skill_generator_self_ll = z_dist.log_prob(nongrad_z)

        # Compute bc loss
        log_prob = 0
        batch_stats = lowlevel_policy.batch_stats
        for t in range(subseq_len):
            (dist, mean_actions, log_stds), updated_state = lowlevel_policy.apply_fn(
                {"params": lowlevel_policy.params, "batch_stats": batch_stats},
                observations[:, t, ...],
                nongrad_z,
                rngs={"dropout": dropout_key},
                training=True,
                mutable=["batch_stats"]
            )
            log_prob += dist.log_prob(actions[:, t, ...])
            batch_stats = updated_state["batch_stats"]

        nll_loss = - jnp.mean(log_prob) / subseq_len

        skill_generator_loss = 0.0005 * kl_loss + nll_loss
        _infos = {
            "skill_generator_loss": skill_generator_loss,
            "skill_generator_kl_loss": kl_loss,
            "skill_generator_nll_loss": nll_loss,
            "z": nongrad_z,   # This will be conditioned on skill decoder(lowlevel policy) and training target of skill prior
            "lowlevel_policy_update_state": updated_state,
            "skill_mean": skill_mu.mean(),
            "skill_log_std": skill_log_std.mean(),
            "skill_generator_self_ll": skill_generator_self_ll.mean()
        }
        return skill_generator_loss, _infos

    skill_generator, infos = skill_generator.apply_gradient(skill_generator_loss_fn)
    lowlevel_policy = lowlevel_policy.replace(batch_stats=infos["lowlevel_policy_update_state"]["batch_stats"])
    return skill_generator, lowlevel_policy, infos


def batch_skill_generator_update(
    rng: jnp.ndarray,

    lowlevel_policy: Model,
    skill_generator: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    last_observations: jnp.ndarray,

):
    obs_dim = observations.shape[-1]
    act_dim = actions.shape[-1]

    rng, dropout_key, init_key, batch_key, skill_sampling_key = jax.random.split(rng, 5)
    subseq_len = actions.shape[1]

    # Skill generator is trained by accelerating the behavior cloning of primitive action policy
    def skill_generator_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:

        z_dist, skill_mu, skill_log_std = skill_generator.apply_fn(
            {"params": params},
            observations,
            actions,
            last_observation=last_observations,
            rngs={"dropout": dropout_key, "init": init_key},
        )
        skill_std = jnp.exp(skill_log_std)

        # Compute kl-divergence loss (to N(0, 1))
        kl_loss = (jnp.log(STD_DEV) - skill_log_std) \
                  + skill_std ** 2 + (skill_mu - STD_MEAN) ** 2 \
                  / (2 * STD_DEV ** 2) - 0.5
        kl_loss = jnp.mean(kl_loss)

        # Sample skill making latent vector z (reparameterize)
        # z = skill_mu + skill_std * jax.random.normal(skill_sampling_key, shape=skill_mu.shape)

        nongrad_z = z_dist.sample(seed=skill_sampling_key)

        batch_observations = observations.reshape(-1, obs_dim)
        batch_actions = actions.reshape(-1, act_dim)
        batch_nongrad_z = jnp.repeat(nongrad_z, repeats=subseq_len, axis=0)

        skill_generator_self_ll = z_dist.log_prob(nongrad_z)

        # Compute bc loss
        (dist, mean_actions, log_stds), updated_state = lowlevel_policy.apply_fn(
            {"params": lowlevel_policy.params, "batch_stats": lowlevel_policy.batch_stats},
            batch_observations,
            batch_nongrad_z,
            rngs={"dropout": dropout_key},
            training=True,
            mutable=["batch_stats"]
        )
        nll_loss = jnp.mean((mean_actions - batch_actions) ** 2)

        # log_prob = dist.log_prob(batch_actions)
        # nll_loss = - jnp.mean(log_prob)

        skill_generator_loss = 0.0005 * kl_loss + nll_loss
        _infos = {
            "skill_generator_loss": skill_generator_loss,
            "skill_generator_kl_loss": kl_loss,
            "skill_generator_nll_loss": nll_loss,
            "z": nongrad_z,   # This will be conditioned on skill decoder(lowlevel policy) and training target of skill prior
            "lowlevel_policy_update_state": updated_state,
            "skill_mean": skill_mu.mean(),
            "skill_log_std": skill_log_std.mean(),
            "skill_generator_self_ll": skill_generator_self_ll.mean()
        }
        return skill_generator_loss, _infos

    skill_generator, infos = skill_generator.apply_gradient(skill_generator_loss_fn)
    lowlevel_policy = lowlevel_policy.replace(batch_stats=infos["lowlevel_policy_update_state"]["batch_stats"])
    return skill_generator, lowlevel_policy, infos


def skill_generator_det_low_update(     # Skill generator with deterministic lowlevel policy
    rng: jnp.ndarray,

    lowlevel_policy: Model,
    skill_generator: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    last_observations: jnp.ndarray,
):
    rng, dropout_key, init_key, batch_key, skill_sampling_key = jax.random.split(rng, 5)
    subseq_len = actions.shape[1]

    # Skill generator is trained by accelerating the behavior cloning of primitive action policy
    def skill_generator_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        z_dist, skill_mu, skill_log_std = skill_generator.apply_fn(
            {"params": params},
            observations,
            actions,
            last_observations,
            rngs={"dropout": dropout_key, "init": init_key},
        )
        skill_std = jnp.exp(skill_log_std)

        # Compute kl-divergence loss (to N(0, 1))
        kl_loss = (jnp.log(STD_DEV) - skill_log_std) \
                  + skill_std ** 2 + (skill_mu - STD_MEAN) ** 2 \
                  / (2 * STD_DEV ** 2) - 0.5
        kl_loss = jnp.mean(kl_loss)

        # Sample skill making latent vector z (reparameterize)
        z = skill_mu + skill_std * jax.random.normal(skill_sampling_key, shape=skill_mu.shape)

        nongrad_z = z_dist.sample(seed=skill_sampling_key)
        skill_generator_self_ll = z_dist.log_prob(nongrad_z)

        # Compute bc loss
        bc_loss = 0
        batch_stats = lowlevel_policy.batch_stats
        for t in range(subseq_len):
            pred_actions, updated_state = lowlevel_policy.apply_fn(
                {"params": lowlevel_policy.params, "batch_stats": batch_stats},
                observations[:, t, ...],
                z,
                rngs={"dropout": dropout_key},
                training=True,
                mutable=["batch_stats"]
            )
            bc_loss += jnp.mean((pred_actions - actions[:, t, ...]) ** 2)
            batch_stats = updated_state["batch_stats"]

        bc_loss = bc_loss / subseq_len

        skill_generator_loss = 0.1 * kl_loss + bc_loss
        _infos = {
            "skill_generator_loss": skill_generator_loss,
            "skill_generator_kl_loss": kl_loss,
            "skill_generator_bc_loss": bc_loss,
            "z": nongrad_z,
            # This will be conditioned on skill decoder(lowlevel policy) and training target of skill prior
            "lowlevel_policy_update_state": updated_state,
            "skill_mean": skill_mu.mean(),
            "skill_log_std": skill_log_std.mean(),
            "skill_generator_self_ll": skill_generator_self_ll.mean()
        }
        return skill_generator_loss, _infos

    skill_generator, infos = skill_generator.apply_gradient(skill_generator_loss_fn)
    lowlevel_policy = lowlevel_policy.replace(batch_stats=infos["lowlevel_policy_update_state"]["batch_stats"])
    return skill_generator, lowlevel_policy, infos


def lowlevel_policy_update(
    rng: jnp.ndarray,

    lowlevel_policy: Model,
    skill_generator: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    last_observations: jnp.ndarray,

):
    subseq_len = actions.shape[1]

    rng, z_dropout_key, z_init_key = jax.random.split(rng, 3)
    z_dist, *_ = skill_generator.apply_fn(
        {"params": skill_generator.params},
        observations,
        actions,
        last_observation=last_observations,
        rngs={"dropout": z_dropout_key, "init": z_init_key}
    )
    z = z_dist.sample(seed=rng)

    # Do bc using pseudo actions
    def lowlevel_policy_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        log_prob = 0
        batch_stats = lowlevel_policy.batch_stats

        for t in range(subseq_len):
            dropout_key, _ = jax.random.split(rng)
            (dist, mean_actions, log_stds), updated_state = lowlevel_policy.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                observations[:, t, ...],
                z,
                rngs={"dropout": dropout_key},
                training=True,
                mutable=["batch_stats"]
            )
            log_prob += dist.log_prob(actions[:, t, ...])
            batch_stats = updated_state["batch_stats"]

        lowlevel_policy_loss = - jnp.mean(log_prob) / subseq_len

        test_sample = dist.sample(seed=rng)
        lowlevel_self_ll = dist.log_prob(test_sample)

        _info = {
            "lowlevel_policy_loss": lowlevel_policy_loss,
            "updated_state": updated_state,
            "lowlevel_policy_mean": mean_actions.mean(),
            "lowlevel_log_std": log_stds.mean(),
            "lowlevel_self_ll": lowlevel_self_ll,
            "lowlevel_sample": test_sample,
            "pseudoaction_mean": mean_actions,
            "pseudoaction_log_stds": log_stds,
            "pseudoaction_sample": test_sample
        }
        return lowlevel_policy_loss, _info

    lowlevel_policy, infos = lowlevel_policy.apply_gradient(lowlevel_policy_loss_fn)
    lowlevel_policy = lowlevel_policy.replace(batch_stats=infos["updated_state"]["batch_stats"])
    return lowlevel_policy, infos


def batch_lowlevel_policy_update(
    rng: jnp.ndarray,

    lowlevel_policy: Model,
    skill_generator: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    last_observations: jnp.ndarray,

):
    subseq_len = actions.shape[1]

    rng, z_dropout_key, z_init_key = jax.random.split(rng, 3)
    z_dist, *_ = skill_generator.apply_fn(
        {"params": skill_generator.params},
        observations,
        actions,
        last_observation=last_observations,
        rngs={"dropout": z_dropout_key, "init": z_init_key}
    )
    z = z_dist.sample(seed=rng)

    # Do bc using pseudo actions
    def lowlevel_policy_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dropout_key, _ = jax.random.split(rng)
        batch_stats = lowlevel_policy.batch_stats

        obs_dim = observations.shape[-1]
        act_dim = actions.shape[-1]

        batch_observations = observations.reshape(-1, obs_dim)
        batch_actions = actions.reshape(-1, act_dim)
        batch_z = jnp.repeat(z, repeats=subseq_len, axis=0)

        (dist, mean_actions, log_stds), updated_state = lowlevel_policy.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            batch_observations,
            batch_z,
            rngs={"dropout": dropout_key},
            training=True,
            mutable=["batch_stats"]
        )

        # log_prob = dist.log_prob(batch_actions)
        # lowlevel_policy_loss = - jnp.mean(log_prob)

        lowlevel_policy_loss = jnp.mean((mean_actions - batch_actions) ** 2)

        # samples = dist.sample(seed=rng).reshape(-1, act_dim)
        # lowlevel_policy_loss = jnp.mean((samples - batch_actions) ** 2)

        test_sample = dist.sample(seed=rng)
        lowlevel_self_ll = dist.log_prob(test_sample)

        _info = {
            "lowlevel_policy_loss": lowlevel_policy_loss,
            "updated_state": updated_state,
            "lowlevel_policy_mean": mean_actions.mean(),
            "lowlevel_log_std": log_stds.mean(),
            "lowlevel_self_ll": lowlevel_self_ll,
            "lowlevel_sample": test_sample,
            "pseudoaction_mean": mean_actions,
            "pseudoaction_log_stds": log_stds,
            "pseudoaction_sample": test_sample,
        }
        return lowlevel_policy_loss, _info

    lowlevel_policy, infos = lowlevel_policy.apply_gradient(lowlevel_policy_loss_fn)
    lowlevel_policy = lowlevel_policy.replace(batch_stats=infos["updated_state"]["batch_stats"])
    return lowlevel_policy, infos


def deterministic_lowlevel_policy_update(
    rng: jnp.ndarray,

    lowlevel_policy: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    z: jnp.ndarray
):
    subseq_len = actions.shape[1]

    # Do bc
    def lowlevel_policy_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        loss = 0.0
        batch_stats = lowlevel_policy.batch_stats

        for t in range(subseq_len):
            dropout_key, _ = jax.random.split(rng)
            pred_actions, updated_state = lowlevel_policy.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                observations[:, t, ...],
                z,
                rngs={"dropout": dropout_key},
                training=True,
                mutable=["batch_stats"]
            )
            loss += jnp.mean((pred_actions - actions[:, t, ...]) ** 2)
            batch_stats = updated_state["batch_stats"]

        lowlevel_policy_loss = loss / subseq_len

        _info = {
            "lowlevel_policy_loss": lowlevel_policy_loss,
            "lowlevel_sample": pred_actions,
            "updated_state": updated_state
        }
        return lowlevel_policy_loss, _info

    lowlevel_policy, infos = lowlevel_policy.apply_gradient(lowlevel_policy_loss_fn)
    lowlevel_policy = lowlevel_policy.replace(batch_stats=infos["updated_state"]["batch_stats"])

    return lowlevel_policy, infos


def learned_skill_prior_update(
    rng: jnp.ndarray,

    skill_generator: Model,
    skill_prior: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    last_observations: jnp.ndarray,
):
    """Skill prior: approximate skill generator"""

    rng, z_dropout_key, z_init_key = jax.random.split(rng, 3)
    z_dist, *_ = skill_generator.apply_fn(
        {"params": skill_generator.params},
        observations,
        actions,
        last_observation=last_observations,
        rngs={"dropout": z_dropout_key, "init": z_init_key}
    )
    z = z_dist.sample(seed=rng)

    _, dropout_key, batch_key = jax.random.split(rng, 3)
    _, prior_sampling = jax.random.split(rng)

    # Training = False for skill prior, e.g., running mean = True for batch normalization
    prior_z_dist, prior_mu, prior_log_std = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        observations[:, 0, ...],
        deterministic=False,
        training=False
    )
    prior_z = prior_z_dist.sample(seed=prior_sampling)
    prior_z_self_ll = prior_z_dist.log_prob(prior_z)

    def skill_prior_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # NOTE: Observation = first timestep's observation of given trajectory
        (dist, mean, log_stds), updated_state = skill_prior.apply_fn(
            {"params": params, "batch_stats": skill_prior.batch_stats},
            observations[:, 0, ...],
            deterministic=False,
            training=True,
            rngs={"dropout": dropout_key},
            mutable=["batch_stats"]
        )
        skill_prior_loss = - jnp.mean(dist.log_prob(z))

        _infos = {
            "skill_prior_loss": skill_prior_loss,
            "updated_state": updated_state,
            "skill_prior_mean": mean.mean(),
            "skill_prior_log_std": log_stds.mean(),
            "testtt": dist.log_prob(z),
            "skill_prior_self_ll": prior_z_self_ll,
            "prior_log_std": prior_log_std
        }
        return skill_prior_loss, _infos

    skill_prior, infos = skill_prior.apply_gradient(skill_prior_loss_fn)
    skill_prior = skill_prior.replace(batch_stats=infos["updated_state"]["batch_stats"])
    return skill_prior, infos


def three_composed_actor_update(
    rng: jnp.ndarray,

    skill_prior: Model,
    three_composed_actor: Model,
    critic: Model,
    log_alpha: Model,

    observations: jnp.ndarray,
):
    """
    Actor loss
        = RL objective
        + Skill-prior regularization of higher policy
        + Real environment behavior cloning of transfer layer
    """
    alpha = jnp.exp(log_alpha())
    rng, forwarding_key, dropout_key, prior_sampling = jax.random.split(rng, 4)

    # Training = False for skill prior, e.g., running mean = True for batch normalization
    prior_z_dist, prior_mu, prior_log_std = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        observations,
        deterministic=False,
        training=False
    )
    prior_z = prior_z_dist.sample(seed=prior_sampling)
    self_ll = prior_z_dist.log_prob(prior_z)[:, jnp.newaxis]

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        action_dist, _ = three_composed_actor.apply_fn(
            {"params": params, "batch_stats": three_composed_actor.batch_stats},
            observations,
            deterministic=False,
            training=False,
            rngs={"dropout": dropout_key, "forwarding": forwarding_key},
        )
        actions_pi = action_dist.sample(seed=rng)

        # NOTE: Stop the gradient for transfer layer by using predefined forward ft when compute the prior divergence.
        _, higher_action_dist = three_composed_actor.apply_fn(
            {"params": params, "batch_stats": three_composed_actor.batch_stats},
            observations,
            deterministic=False,
            training=False,
            rngs={"dropout": dropout_key, "sampling": rng},
            method=ThreeComposedActor.sample_higher_action
        )
        # Recall that kl-divergence is just empirical ll
        prior_divergence = - higher_action_dist.log_prob(prior_z)[:, jnp.newaxis]

        # >>> RL Maximization & Skill prior regularization. Second objective is included by SAC-like objective.
        q_values_pi = critic.apply_fn(
            {"params": critic.params},
            observations,
            actions_pi,
            rngs={"dropout": dropout_key},
        )
        min_qf_pi = jnp.min(q_values_pi, axis=1)
        skill_reg_rl_loss = (alpha * prior_divergence - min_qf_pi).mean()

        actor_loss = skill_reg_rl_loss

        _infos = {
            "actor_loss": actor_loss,
            "prior_divergence": prior_divergence.mean(),
            "higher_action_dist": higher_action_dist,
            "prior_z_dist": prior_z_dist,
            "self_ll": self_ll,
            "actions_pi": actions_pi,
            "reg_term": skill_reg_rl_loss,
            "alpha*prior": alpha * prior_divergence,
            "min_qf_pi": min_qf_pi.mean()
        }
        return actor_loss, _infos

    three_composed_actor, infos = three_composed_actor.apply_gradient(actor_loss_fn)

    return three_composed_actor, critic, infos

### HRL
from .utils import clock
# @clock(fmt="[{name}: {elapsed: 0.8f}s]")
def hrl_skill_prior_regularized_actor_update(
    rng: jnp.ndarray,

    skill_prior: Model,
    actor: Model,       # Higher actor
    critic: Model,      # Higher critic
    log_alpha: Model,

    observations: jnp.ndarray,
):
    """Calculate the prior divergence analytically"""
    alpha = jnp.exp(log_alpha())
    rng, forwarding_key, dropout_key, prior_sampling = jax.random.split(rng, 4)

    # Training = False for skill prior, e.g., running mean = True for batch normalization
    prior_z_dist, prior_mu, prior_log_std = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        observations,
        deterministic=True,
        training=False
    )
    prior_z = prior_z_dist.sample(seed=prior_sampling)
    self_ll = prior_z_dist.log_prob(prior_z)[:, jnp.newaxis]

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        action_dist, *_ = actor.apply_fn(
            {"params": params, "batch_stats": actor.batch_stats},
            observations,
            deterministic=True,
            training=False
        )
        actions_pi = action_dist.sample(seed=prior_sampling)

        prior_divergence = tfd.kl_divergence(action_dist, prior_z_dist, allow_nan_stats=False)[:, jnp.newaxis]

        q_values_pi = critic.apply_fn(
            {"params": critic.params},
            observations,
            actions_pi,
            rngs={"dropout": dropout_key}
        )
        min_qf_pi = jnp.min(q_values_pi, axis=1)
        actor_loss = ((alpha * prior_divergence) - min_qf_pi).mean()

        higher_self_ll = action_dist.log_prob(actions_pi)
        _infos = {
            "higher_actor_loss": actor_loss.mean(),
            "higher_self_ll": higher_self_ll.mean(),
            "prior_divergence": prior_divergence.mean(),
            "prior_self_ll": self_ll.mean(),

            "_prior_mean": prior_z_dist.mean(),
            "_prior_var": prior_z_dist.stddev(),

            "_actor_mean": action_dist.mean(),
            "_actor_var": action_dist.stddev()
        }
        return actor_loss, _infos
    actor, infos = actor.apply_gradient(actor_loss_fn)
    return actor, infos


def hrl_lower_actor_update(         # This is runned by SAC, hence use an entropy.
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    log_alpha: Model,

    observations: jnp.ndarray,
    conditions: jnp.ndarray     # == z
):
    alpha = jnp.exp(log_alpha())
    rng, dropout_key = jax.random.split(rng)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, *_ = actor.apply_fn(
            {'params': actor_params},
            observations,
            conditions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        actions_pi = dist.sample(seed=rng)
        log_prob = dist.log_prob(actions_pi)
        log_prob = log_prob[:, jnp.newaxis]

        q_values_pi = critic(
            observations,
            actions_pi,
            conditions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        min_qf_pi = jnp.min(q_values_pi, axis=1)
        actor_loss = (alpha * log_prob - min_qf_pi).mean()
        return actor_loss, {'lower_actor_loss': actor_loss, 'entropy': -log_prob.mean()}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def skill_prior_regularized_critic_update(
    rng: jnp.ndarray,

    skill_prior: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float
):

    dropout_key, forwarding_key = jax.random.split(rng)

    alpha = jnp.exp(log_alpha())

    _, next_higher_actions_dist = actor.apply_fn(
        {"params": actor.params, "batch_stats": actor.batch_stats},
        next_observations,
        deterministic=False,
        training=False,
        rngs={"dropout": dropout_key, "sampling": forwarding_key},
        method=ThreeComposedActor.sample_higher_action
    )

    prior_z_dist, *_ = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        next_observations,
        deterministic=False,
        training=False
    )
    prior_z_sampled = prior_z_dist.sample(seed=rng)
    prior_divergence = - next_higher_actions_dist.log_prob(prior_z_sampled)[:, np.newaxis]

    (next_actions_dist, *_) = actor.apply_fn(
        {"params": actor.params, "batch_stats": actor.batch_stats},
        next_observations,
        deterministic=False,
        training=False,
        rngs={"dropout": dropout_key, "forwarding": forwarding_key},
    )
    next_actions = next_actions_dist.sample(seed=rng)

    next_q_values = critic_target.apply_fn(
        {"params": critic_target.params},
        next_observations,
        next_actions,
        deterministic=False,
    )
    next_q_values = jnp.min(next_q_values, axis=1)

    next_q_values = next_q_values - alpha * prior_divergence

    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network using action from the replay buffer
        q_values = critic.apply_fn(
            {"params": params},
            observations,
            actions,
            rngs={"dropout_key": dropout_key},
        )

        # Compute skill prior regularized critic loss
        n_qs = q_values.shape[1]
        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)]) / n_qs

        _infos = {
            "higher_critic_loss": critic_loss,
        }
        return critic_loss, _infos

    critic, infos = critic.apply_gradient(critic_loss_fn)
    return critic, critic_target, actor, infos


# @clock(fmt="[{name}: {elapsed: 0.8f}s]")
def hrl_skill_prior_regularized_critic_update(
    rng: jnp.ndarray,

    skill_prior: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float
):
    dropout_key, forwarding_key = jax.random.split(rng)

    alpha = jnp.exp(log_alpha())
    next_action_dist, *_ = actor.apply_fn(
        {"params": actor.params, "batch_stats": actor.batch_stats},
        next_observations,
        deterministic=False,
        training=False,
        rngs={"dropout": dropout_key}
    )
    next_prior_z_dist, *_ = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        next_observations,
        deterministic=False,
        training=False
    )
    _prior_divergence =  tfd.kl_divergence(next_action_dist, next_prior_z_dist, allow_nan_stats=False)[:, jnp.newaxis]

    next_actions = next_action_dist.sample(seed=rng)

    next_q_values = critic_target(next_observations, next_actions, deterministic=False, rngs={"dropout": dropout_key})
    next_q_values = jnp.min(next_q_values, axis=1)
    next_q_values = next_q_values - alpha * _prior_divergence
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        q_values = critic.apply_fn(
            {'params': critic_params},
            observations,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs

        _infos = {
            "higher_critic_loss": critic_loss,
            "higher_min_qf_pi": q_values.mean(),
            "higher_next_q_val": next_q_values.mean(),
            "critic_prior_divergence": _prior_divergence.mean()
        }

        return critic_loss, _infos

    critic, info = critic.apply_gradient(critic_loss_fn)
    return critic, info


def hrl_lower_critic_update(
    rng: jnp.ndarray,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    conditions: jnp.ndarray,            # == z
    next_conditions: jnp.ndarray,

    gamma:float
):
    rng, dropout_key = jax.random.split(rng)
    dist, *_ = actor(next_observations, next_conditions, deterministic=False, rngs={"dropout": dropout_key})
    next_actions = dist.sample(seed=rng)
    next_log_prob = dist.log_prob(next_actions)

    # Compute the next Q values: min over all critics targets
    next_q_values = critic_target(
        next_observations,
        next_actions,
        next_conditions,
        deterministic=False,
        rngs={"dropout": dropout_key}
    )
    next_q_values = jnp.min(next_q_values, axis=1)

    ent_coef = jnp.exp(log_ent_coef())

    # add entropy term
    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    # td error + entropy term
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        q_values = critic.apply_fn(
            {'params': critic_params},
            observations,
            actions,
            conditions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        # Compute critic loss
        n_qs = q_values.shape[1]

        critic_loss = sum([jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs
        _infos = {
            'lower_critic_loss': critic_loss,
            'lower_min_qf_pi': q_values.mean(),
            "n_qs": n_qs,
            "lower_next_q_val": next_q_values.mean(),
            "test_ent_coef": ent_coef.mean(),
            "test_next_log_prob": next_log_prob.mean()
        }
        return critic_loss, _infos

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info

def log_alpha_update(
    rng: jnp.ndarray,
    skill_prior: Model,
    log_alpha: Model,
    actor: Model,
    observations: jnp.ndarray,
    target_alpha: float,
):

    rng, prior_sampling_key, dropout_key, forwarding_key = jax.random.split(rng, 4)
    prior_z_dist, *_ = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        observations,
        deterministic=False,
        training=False,
        rngs={"dropout": dropout_key, "forwarding": forwarding_key},
    )
    prior_z = prior_z_dist.sample(seed=prior_sampling_key)

    def alpha_loss_fn(params: Params):
        (action_dist, higher_action_dist) = actor.apply_fn(
            {"params": actor.params, "batch_stats": actor.batch_stats},
            observations,
            deterministic=False,
            rngs={"dropout": dropout_key, "forwarding": forwarding_key},
            training=False
        )
        prior_divergence = -higher_action_dist.log_prob(prior_z)
        _log_alpha = log_alpha.apply_fn({'params': params})
        alpha_loss = -(_log_alpha * (-target_alpha + prior_divergence)).mean()

        _infos = {
            "log_alpha": _log_alpha,
            "alpha_loss": alpha_loss,
            "alpha_update_prior_divergence": prior_divergence
        }
        return alpha_loss, _infos

    log_alpha, infos = log_alpha.apply_gradient(alpha_loss_fn)
    return log_alpha, actor, infos


# @clock(fmt="[{name}: {elapsed: 0.8f}s]")
def hrl_higher_log_alpha_update(
    rng: jnp.ndarray,
    skill_prior: Model,
    log_alpha: Model,
    actor: Model,           # This is higher actor
    observations: jnp.ndarray,
    target_alpha: float,
):
    rng, dropout_key = jax.random.split(rng)
    action_dist, *_ = actor.apply_fn(
        {"params": actor.params, "batch_stats": actor.batch_stats},
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key},
        training=False
    )
    skill_prior_dist, *_ = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key},
        training=False
    )
    prior_divergence = tfd.kl_divergence(action_dist, skill_prior_dist, allow_nan_stats=False)

    def alpha_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _log_alpha = log_alpha.apply_fn({'params': params})
        alpha = jnp.exp(_log_alpha)
        alpha_loss = - (alpha * (-target_alpha + prior_divergence)).mean()

        return alpha_loss, {"higher_alpha": alpha, "higher_alpha_loss": alpha_loss}

    log_alpha, info = log_alpha.apply_gradient(alpha_loss_fn)
    return log_alpha, info


def hrl_lower_log_alpha_update(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    observations: jnp.ndarray,
    conditions: jnp.ndarray,
    target_entropy: float,
) -> Tuple[Model, InfoDict]:

    rng, dropout_key = jax.random.split(rng)
    dist, *_ = actor(observations, conditions, deterministic=False, rngs={"dropout": dropout_key})
    actions_pi = dist.sample(seed=rng)
    log_prob = dist.log_prob(actions_pi)

    def temperature_loss_fn(ent_params: Params):
        ent_coef = jnp.exp(log_ent_coef.apply_fn({'params': ent_params}))
        ent_coef_loss = -(ent_coef * (target_entropy + log_prob)).mean()
        _infos = {
            'lower_alpha': ent_coef,
            'lower_alpha_loss': ent_coef_loss,
            "lower_log_prob": log_prob.mean(),
            "test_ent_coef!": ent_coef.mean(),
            "test_target_entropy!": target_entropy.mean(),
            "test_log_prob!": log_prob.mean()
        }
        return ent_coef_loss, _infos

    new_ent_coef, info = log_ent_coef.apply_gradient(temperature_loss_fn)
    return new_ent_coef, info


def image_encoder_update(
    rng: jnp.ndarray,
    encoder: Model,
    images: jnp.ndarray,
    beta: float
):
    rng, latent_sampling_key = jax.random.split(rng)

    def encoder_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        decoded_img, (mean, log_std) = encoder.apply_fn(
            {"params": params},
            images,
            rngs={"latent_sampling": latent_sampling_key}
        )
        std = jnp.exp(log_std)
        recon_loss = jnp.mean((decoded_img - images) ** 2)
        kl_loss = jnp.mean((jnp.log(STD_DEV) - log_std) + std ** 2 + (mean - STD_MEAN) ** 2 / (2 * STD_DEV ** 2) - 0.5)

        encoder_loss = recon_loss + beta * kl_loss
        _infos = {
            "encoder_loss": encoder_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "mean": mean.mean(),
            "log_std": log_std.mean(),
            "decoded_img": decoded_img
        }
        return encoder_loss, _infos

    encoder, info = encoder.apply_gradient(encoder_loss_fn)
    return encoder, info


@jax.jit
def ra_hrl_skill_prior_regularized_actor_update(
    rng: jnp.ndarray,

    skill_prior: Model,
    actor: Model,       # Higher actor
    critic: Model,      # Higher critic
    log_alpha: Model,

    observations: jnp.ndarray,
):
    """Calculate the prior divergence empirically"""
    alpha = jnp.exp(log_alpha())
    rng, forwarding_key, dropout_key, prior_sampling = jax.random.split(rng, 4)

    # Training = False for skill prior, e.g., running mean = True for batch normalization
    prior_z_dist, prior_mu, prior_log_std = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key},
        training=False
    )
    prior_z = prior_z_dist.sample(seed=prior_sampling)
    self_ll = prior_z_dist.log_prob(prior_z)[:, jnp.newaxis]

    def actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        action_dist, actor_mu, actor_log_std = actor.apply_fn(
            {"params": params, "batch_stats": actor.batch_stats},
            observations,
            deterministic=False,
            training=False,
            rngs={"dropout": dropout_key}
        )
        actions_pi = action_dist.sample(seed=rng)

        actor_dist = tfd.MultivariateNormalDiag(loc=actor_mu, scale_diag=jnp.exp(actor_log_std))

        prior_divergence = tfd.kl_divergence(actor_dist, prior_z_dist, allow_nan_stats=False).reshape(-1, 1)
        prior_divergence = jnp.clip(jnp.array(prior_divergence), -100.0, 100.0)

        q_values_pi = critic.apply_fn(
            {"params": critic.params},
            observations,
            actions_pi,
            rngs={"dropout": dropout_key}
        )
        min_qf_pi = jnp.min(q_values_pi, axis=1)
        actor_loss = ((alpha * prior_divergence) - min_qf_pi).mean()

        higher_self_ll = action_dist.log_prob(actions_pi)
        _infos = {
            "higher_actor_loss": actor_loss.mean(),
            "higher_self_ll": higher_self_ll.mean(),
            "prior_divergence": prior_divergence.mean(),
            "prior_self_ll": self_ll.mean()
        }
        return actor_loss, _infos
    new_actor, infos = actor.apply_gradient(actor_loss_fn)
    return new_actor, infos


@partial(jax.jit, static_argnames=("gamma",))
def ra_hrl_skill_prior_regularized_critic_update(
    rng: jnp.ndarray,

    skill_prior: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float
):
    dropout_key, forwarding_key = jax.random.split(rng)

    alpha = jnp.exp(log_alpha())
    next_action_dist, actor_mu, actor_log_std = actor.apply_fn(
        {"params": actor.params, "batch_stats": actor.batch_stats},
        next_observations,
        deterministic=False,
        training=False,
        rngs={"dropout": dropout_key}
    )
    next_prior_z_dist, prior_z_mu, prior_z_log_std = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        next_observations,
        deterministic=False,
        training=False
    )

    next_actor_dist = tfd.MultivariateNormalDiag(loc=actor_mu, scale_diag=jnp.exp(actor_log_std))
    prior_divergence = tfd.kl_divergence(next_actor_dist, next_prior_z_dist, allow_nan_stats=False).reshape(-1, 1)
    # prior_divergence = tfd.kl_divergence(next_actor_dist, next_prior_z_dist, allow_nan_stats=False).reshape(-1, 1)
    prior_divergence = jnp.clip((jnp.array(prior_divergence)), -100.0, 100.0)

    next_actions = next_action_dist.sample(seed=rng)

    next_q_values = critic_target(next_observations, next_actions, deterministic=False, rngs={"dropout": dropout_key})
    next_q_values = jnp.min(next_q_values, axis=1)
    next_q_values = next_q_values - alpha * prior_divergence
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        q_values = critic.apply_fn(
            {'params': critic_params},
            observations,
            actions,
            deterministic=False,
            rngs={"dropout": dropout_key}
        )

        # Compute critic loss
        n_qs = q_values.shape[1]
        critic_loss = sum([0.5 * jnp.mean((target_q_values - q_values[:, i, ...]) ** 2) for i in range(n_qs)])
        critic_loss = critic_loss / n_qs

        return critic_loss, {"higher_critic_loss": critic_loss, "higher_min_qf_pi": q_values.mean(), "higher_next_q_val": next_q_values.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


@partial(jax.jit, static_argnames=("target_alpha",))
def ra_hrl_higher_log_alpha_update(
    rng: jnp.ndarray,
    skill_prior: Model,
    log_alpha: Model,
    actor: Model,           # This is higher actor

    observations: jnp.ndarray,

    target_alpha: float,
):
    rng, dropout_key = jax.random.split(rng)
    action_dist, actor_mu, actor_log_std = actor.apply_fn(
        {"params": actor.params, "batch_stats": actor.batch_stats},
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key},
        training=False
    )
    prior_z_dist, prior_mu, prior_log_std = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        observations,
        deterministic=False,
        rngs={"dropout": dropout_key},
        training=False
    )

    actor_dist = tfd.MultivariateNormalDiag(loc=actor_mu, scale_diag=jnp.exp(actor_log_std))
    prior_divergence = tfd.kl_divergence(actor_dist, prior_z_dist, allow_nan_stats=False)
    prior_divergence = jnp.clip(jnp.array(prior_divergence), -100.0, 100.0)

    def alpha_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _log_alpha = log_alpha.apply_fn({'params': params})
        alpha = jnp.exp(_log_alpha)
        alpha_loss = alpha * (target_alpha - prior_divergence).mean()

        return alpha_loss, {"higher_alpha": alpha, "higher_alpha_loss": alpha_loss, "alpha_loss_prior_divergence": prior_divergence.mean()}

    new_log_alpha, info = log_alpha.apply_gradient(alpha_loss_fn)
    return new_log_alpha, info


def ppo_style_higher_actor_update(
    rng: jnp.ndarray,

    higher_actor: Model,
    skill_prior: Model,     # For PPO style regularization

    higher_observations: jnp.ndarray,
    higher_actions: jnp.ndarray,
    returns: jnp.ndarray,   # [nct, 1]      # nct = n_collected_transitions for one episode

    clip_range: float
):
    rng, dropout_key = jax.random.split(rng)
    higher_actions = jnp.clip(higher_actions, -2.0 + 1E-4, 2.0 - 1E-4)

    prior_dist, *_ = skill_prior.apply_fn(
        {"params": skill_prior.params, "batch_stats": skill_prior.batch_stats},
        higher_observations,
        deterministic=True,
        rngs={"dropout": dropout_key},
        training=False
    )
    prior_log_prob = jnp.clip(prior_dist.log_prob(higher_actions).reshape(-1, 1), -100, 50)

    def higher_actor_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        action_dist, *_ = higher_actor.apply_fn(
            {"params": params, "batch_stats": higher_actor.batch_stats},
            higher_observations,
            deterministic=True,
            rngs={"dropout": dropout_key},
            training=False
        )
        actor_log_prob = jnp.clip(action_dist.log_prob(higher_actions).reshape(-1, 1), -100, 50)

        # NOTE: PPO maximizes ratio
        ratio = jnp.exp(actor_log_prob - prior_log_prob).reshape(-1, 1)        # [nct, 1]

        actor_loss_1 = returns * ratio
        actor_loss_2 = returns * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)

        actor_loss = jnp.squeeze(jnp.array([actor_loss_1, actor_loss_2]), axis=2)
        actor_loss = -jnp.min(actor_loss, axis=0).mean()
        # NOTE: PPO END

        # # NOTE: If we want 'not' to maximize the ratio
        # print("Returns shape", returns.shape)
        # print("Actor log prob", actor_log_prob.shape)
        # actor_loss = jnp.mean(returns * jnp.clip(actor_log_prob, -100, 50))

        # Logging
        clip_fraction = jnp.mean((jnp.abs(ratio - 1) > clip_range))
        _infos = {
            "higher_actor_loss": actor_loss,
            "clip_fraction": clip_fraction,
            "actor_log_prob": actor_log_prob.mean(),
            "prior_log_prob": prior_log_prob.mean(),
            "actor_loss_1": actor_loss_1.mean(),
            "actor_loss_2": actor_loss_2.mean(),
            "ratio": ratio.mean(),
            "ratio_before_exp": (actor_log_prob - prior_log_prob).mean()
        }
        return actor_loss, _infos

    new_actor, infos = higher_actor.apply_gradient(higher_actor_loss_fn)
    return rng, new_actor, infos