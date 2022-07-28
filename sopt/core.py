import functools
from collections import defaultdict

import jax
import jax.numpy as jnp

from offline_baselines_jax.common.jax_layers import polyak_update
from offline_baselines_jax.common.policies import Model
from .core_comp import (

    sac_actor_update,
    sac_critic_update,

    log_ent_coef_update,
    sensor_based_double_state_discriminator_update,
    sensor_based_double_state_last_conditioned_discriminator_update,
    image_encoder_update,
    skill_generator_update,
    lowlevel_policy_update,
    learned_skill_prior_update,
    deterministic_lowlevel_policy_update,
    skill_generator_det_low_update,

    three_composed_actor_update,
    skill_prior_regularized_critic_update,
    log_alpha_update,

    hrl_skill_prior_regularized_actor_update,
    hrl_skill_prior_regularized_critic_update,
    hrl_higher_log_alpha_update,
    hrl_lower_actor_update,
    hrl_lower_critic_update,
    hrl_lower_log_alpha_update,
    ppo_style_higher_actor_update,

    batch_skill_generator_update,
    batch_lowlevel_policy_update,

    ra_hrl_skill_prior_regularized_actor_update,
    ra_hrl_skill_prior_regularized_critic_update,
    ra_hrl_higher_log_alpha_update
)

target_update = polyak_update
CURRENT = 0
NEXT = 1


@functools.partial(jax.jit, static_argnames=('gamma', 'target_entropy', 'tau', 'target_update_cond', 'entropy_update'))
def sac_update(
    rng: int,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float,
    target_entropy: float,
    tau: float,
    target_update_cond: bool,
    entropy_update: bool
):

    rng, key = jax.random.split(rng, 2)
    new_critic, critic_info = sac_critic_update(
        rng=rng,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_ent_coef=log_ent_coef,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        gamma=gamma
    )

    if target_update_cond:
        new_critic_target = polyak_update(new_critic, critic_target, tau)
    else:
        new_critic_target = critic_target

    rng, key = jax.random.split(rng, 2)
    new_actor, actor_info = sac_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_ent_coef=log_ent_coef,
        observations=observations
    )

    rng, key = jax.random.split(rng, 2)
    if entropy_update:
        new_temp, ent_info = log_ent_coef_update(
            rng=rng,
            log_ent_coef=log_ent_coef,
            actor=actor,
            observations=observations,
            target_entropy=target_entropy
        )
    else:
        new_temp, ent_info = log_ent_coef, {'ent_coef': jnp.exp(log_ent_coef()), 'ent_coef_loss': 0}

    new_models = {
        "critic": new_critic,
        "critic_target": new_critic_target,
        "actor": new_actor,
        "log_ent_coef": new_temp
    }
    return rng, new_models, {**critic_info, **actor_info, **ent_info}


@functools.partial(jax.jit, static_argnames=("target_update_cond", "entropy_update", "target_entropy", "gamma", "tau"))
def sensor_based_single_state_amsopt_sac_update_without_action_matching_update(
    rng: jnp.ndarray,
    log_ent_coef: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    double_state_discriminator: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    expert_observation: jnp.ndarray,
    expert_next_observation: jnp.ndarray,

    target_update_cond: bool,              # For SAC
    entropy_update: bool,                  # For SAC
    target_entropy: float,                 # For SAC
    gamma: float,                          # Discount factor
    tau: float,                            # Soft target update,

    **kwargs
):
    r"""
    여기서는, discriminator의 score가 policy나 critic의 학습에 직접 영향을 주지는 않는다.
    discriminator는 단순히 intrinsic reward를 정의해주기 위해 학습한다.

    Here, the score of the discriminator does not directly affect the learning of the policy or the critic.
    The discriminator simply learns to define an intrinsic reward.
    """

    double_state_discriminator, double_disc_info = sensor_based_double_state_discriminator_update(
        rng=rng,
        discriminator=double_state_discriminator,
        expert_observation=expert_observation,
        expert_next_observation=expert_next_observation,
        observation=observations,
        next_observation=next_observations
    )

    rng, models, infos = sac_update(
        rng=rng,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_ent_coef=log_ent_coef,

        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,

        gamma=gamma,
        target_entropy=target_entropy,
        tau=tau,
        target_update_cond=target_update_cond,
        entropy_update=entropy_update
    )

    models.update({"double_state_discriminator": double_state_discriminator})
    infos.update({**double_disc_info})

    return rng, models, infos


@functools.partial(jax.jit, static_argnames=("gamma", "target_entropy", "tau", "target_update_cond", "entropy_update", "discriminator_update_cond"))
def adversarial_imitation_learning_update(
    rng: jnp.ndarray,

    discriminator: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_ent_coef: Model,

    expert_observations: jnp.ndarray,
    expert_next_observations: jnp.ndarray,
    expert_last_observations: jnp.ndarray,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
    last_observations: jnp.ndarray,

    gamma: float,
    target_entropy: float,
    tau: float,
    target_update_cond: bool,
    discriminator_update_cond: bool,
    entropy_update: bool
):
    rng, new_models, infos = sac_update(
        rng=rng,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_ent_coef=log_ent_coef,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
        gamma=gamma,
        target_entropy=target_entropy,
        tau=tau,
        target_update_cond=target_update_cond,
        entropy_update=entropy_update
    )

    if discriminator_update_cond:
        discriminator, disc_info = sensor_based_double_state_last_conditioned_discriminator_update(
            rng=rng,
            discriminator=discriminator,
            expert_observation=expert_observations,
            expert_next_observation=expert_next_observations,
            expert_last_observation=expert_last_observations,
            observation=observations,
            next_observation=next_observations,
            last_observation=last_observations
        )
    else:
        disc_info = {"discriminator_loss": 0, "expert_disc_score": 0, "policy_disc_score": 0}

    new_models.update({"discriminator": discriminator})
    infos.update({**disc_info})

    return rng, new_models, infos


@jax.jit
def skill_prior_update(
    rng: jnp.ndarray,

    lowlevel_policy: Model,      # Output: pseudo-action
    skill_generator: Model,
    skill_prior: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,       # NOTE: This is pseudo-action. Original dataset doesn't containing actions
    last_observations: jnp.ndarray,

):
    """
    >> observations, next_observations: [batch_size, subseq_len, obs_dim * n_frames]
    >> last_observations: [batch_size, obs_dim * n_frames]
    >> actions: [batch_size, subseq_len, action_dim]     # In this code, action_dim = obs_dim
    NOTE:
        actions: This has a shape [batch_size, subseq_len, pseudo_action_dim]
        Lowlevel policy is trained by behavior cloning the 'next action'.
        So actions must be sliced into the size of [batch_size, 'subseq_len-1', pseudo_action_dim]
        and the last action is a target of behavior cloning.
    """
    rng, _ = jax.random.split(rng)
    new_skill_generator, lowlevel_policy_2, skill_generator_infos = batch_skill_generator_update(
        rng=rng,
        lowlevel_policy=lowlevel_policy,
        skill_generator=skill_generator,
        observations=observations,
        actions=actions,
        last_observations=last_observations
    )

    rng, _ = jax.random.split(rng)
    new_lowlevel_policy, lowlevel_policy_infos = batch_lowlevel_policy_update(
        rng=rng,
        lowlevel_policy=lowlevel_policy_2,
        skill_generator=skill_generator,

        observations=observations,
        actions=actions,
        last_observations=last_observations
    )

    rng, _ = jax.random.split(rng)
    new_skill_prior, skill_prior_infos = learned_skill_prior_update(
        rng=rng,
        skill_generator=skill_generator,
        skill_prior=skill_prior,
        observations=observations,
        actions=actions,
        last_observations=last_observations,
    )

    new_models = {
        "skill_generator": new_skill_generator,
        "lowlevel_policy": new_lowlevel_policy,
        "skill_prior": new_skill_prior
    }
    infos = defaultdict(int)
    infos.update({**skill_generator_infos, **lowlevel_policy_infos, **skill_prior_infos})

    return rng, new_models, infos


@jax.jit
def det_skill_prior_update(     # Train with deterministic lowlevel policy
    rng: jnp.ndarray,

    lowlevel_policy: Model,      # Output: pseudo-action
    skill_generator: Model,
    skill_prior: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,       # NOTE: This is pseudo-action. Original dataset doesn't containing actions
    last_observations: jnp.ndarray,

):
    """
    >> observations, next_observations: [batch_size, subseq_len, obs_dim * n_frames]
    >> last_observations: [batch_size, obs_dim * n_frames]
    >> actions: [batch_size, subseq_len, action_dim]     # In this code, action_dim = obs_dim
    NOTE:
        actions: This has a shape [batch_size, subseq_len, pseudo_action_dim]
        Lowlevel policy is trained by behavior cloning the 'next action'.
        So actions must be sliced into the size of [batch_size, 'subseq_len-1', pseudo_action_dim]
        and the last action is a target of behavior cloning.
    """
    rng, _ = jax.random.split(rng)
    skill_generator, lowlevel_policy, skill_generator_infos = skill_generator_det_low_update(
        rng=rng,
        lowlevel_policy=lowlevel_policy,
        skill_generator=skill_generator,
        observations=observations,
        actions=actions,
        last_observations=last_observations
    )
    z = skill_generator_infos["z"]

    rng, _ = jax.random.split(rng)
    lowlevel_policy, lowlevel_policy_infos = deterministic_lowlevel_policy_update(
        rng=rng,
        lowlevel_policy=lowlevel_policy,
        observations=observations,
        actions=actions,
        z=z
    )

    rng, _ = jax.random.split(rng)
    skill_prior, skill_prior_infos = learned_skill_prior_update(
        rng=rng,
        skill_prior=skill_prior,
        observations=observations,
        z=z
    )

    new_models = {
        "skill_generator": skill_generator,
        "lowlevel_policy": lowlevel_policy,
        "skill_prior": skill_prior
    }
    infos = defaultdict(int)
    infos.update({**skill_generator_infos, **lowlevel_policy_infos, **skill_prior_infos})

    return rng, new_models, infos

@functools.partial(jax.jit, static_argnames=("target_update_cond", "alpha_update", "gamma", "target_alpha", "bc_reg_coef", "tau"))
def skill_regularized_sac_update(           # Update higher actor
    rng: jnp.ndarray,

    skill_prior: Model,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float,
    target_alpha: float,
    tau: float,
    target_update_cond: bool,
    alpha_update: bool
):
    rng, _ = jax.random.split(rng)
    actor, critic, actor_update_infos = three_composed_actor_update(
        rng=rng,
        skill_prior=skill_prior,
        three_composed_actor=actor,
        critic=critic,
        log_alpha=log_alpha,

        observations=observations,
    )

    rng, _ = jax.random.split(rng)
    critic, critic_target, actor, critic_update_infos = skill_prior_regularized_critic_update(
        rng=rng,
        skill_prior=skill_prior,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_alpha=log_alpha,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        gamma=gamma
    )

    if alpha_update:
        rng, _ = jax.random.split(rng)
        log_alpha, actor, alpha_update_infos = log_alpha_update(
            rng=rng,
            skill_prior=skill_prior,
            log_alpha=log_alpha,
            actor=actor,
            observations=observations,
            target_alpha=target_alpha
        )
    else:
        alpha_update_infos = {"log_alpha": log_alpha(), "alpha_loss": 0}

    if target_update_cond:
        critic_target = polyak_update(critic, critic_target, tau)

    new_models = {
        "actor": actor,
        "critic": critic,
        "critic_target": critic_target,
        "log_alpha": log_alpha,
    }
    infos = {**actor_update_infos, **critic_update_infos, **alpha_update_infos}

    return rng, new_models, infos


@functools.partial(jax.jit, static_argnames=("target_update_cond", "alpha_update", "gamma", "target_alpha", "tau", "target_update_cond", "alpha_update"))
def hrl_higher_policy_update(
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    skill_prior: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float,
    target_alpha: float,
    tau: float,
    target_update_cond: bool,
    alpha_update: bool
):
    rng, _ = jax.random.split(rng)
    updated_actor, actor_infos = hrl_skill_prior_regularized_actor_update(
        rng=rng,
        skill_prior=skill_prior,
        actor=actor,
        critic=critic,
        log_alpha=log_alpha,
        observations=observations
    )

    rng, _ = jax.random.split(rng)
    updated_critic, critic_infos = hrl_skill_prior_regularized_critic_update(
        rng=rng,

        skill_prior=skill_prior,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_alpha=log_alpha,

        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        gamma=gamma
    )

    if alpha_update:
        rng, _ = jax.random.split(rng)
        updated_log_alpha, log_alpha_infos = hrl_higher_log_alpha_update(
            rng=rng,
            skill_prior=skill_prior,
            log_alpha=log_alpha,
            actor=actor,
            observations=observations,
            target_alpha=target_alpha,
        )

    else:
        updated_log_alpha = log_alpha
        log_alpha_infos = {"higher_log_alpha": log_alpha, "higher_alpha_loss": 0.0}

    if target_update_cond:
        updated_critic_target = polyak_update(critic, critic_target, tau)

    else:
        updated_critic_target = updated_critic

    new_models = {
        "higher_actor": updated_actor,
        "higher_critic": updated_critic,
        "higher_critic_target": updated_critic_target,
        "higher_log_alpha": updated_log_alpha
    }

    infos = {**actor_infos, **critic_infos, **log_alpha_infos}
    return rng, new_models, infos


@functools.partial(jax.jit, static_argnames=("target_update_cond", "alpha_update", "gamma", "target_alpha", "tau", "target_update_cond", "alpha_update"))
def hrl_lower_policy_update(
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,
    conditions: jnp.ndarray,
    next_conditions: jnp.ndarray,

    gamma: float,
    target_alpha: float,
    tau: float,
    target_update_cond: bool,
    alpha_update: bool
):
    rng, _ = jax.random.split(rng)
    new_actor, actor_infos = hrl_lower_actor_update(
        rng=rng,
        actor=actor,
        critic=critic,
        log_alpha=log_alpha,
        observations=observations,
        conditions=conditions
    )

    rng, _ = jax.random.split(rng)
    new_critic, critic_infos = hrl_lower_critic_update(
        rng=rng,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_ent_coef=log_alpha,

        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        conditions=conditions,
        next_conditions=next_conditions,
        gamma=gamma
    )

    if alpha_update:
        rng, _ = jax.random.split(rng)
        log_alpha, log_alpha_infos = hrl_lower_log_alpha_update(
            rng=rng,
            log_ent_coef=log_alpha,
            actor=actor,
            observations=observations,
            conditions=conditions,
            target_entropy=target_alpha
        )
    else:
        log_alpha_infos = {"lower_log_alpha": log_alpha, "lower_alpha_loss": 0.0}

    if target_update_cond:
        critic_target = polyak_update(critic, critic_target, tau)

    new_models = {
        "lower_actor": new_actor,
        "lower_critic": new_critic,
        "lower_critic_target": critic_target,
        "lower_log_alpha": log_alpha
    }
    infos = {**actor_infos, **critic_infos, **log_alpha_infos}
    return rng, new_models, infos


@functools.partial(jax.jit, static_argnames=("beta",))
def encoder_update(
    rng: jnp.ndarray,

    encoder: Model,
    images: jnp.ndarray,
    beta: float
):
    encoder, infos = image_encoder_update(rng=rng, encoder=encoder, images=images, beta=beta)
    return rng, encoder, infos


# @functools.partial(jax.jit, static_argnames=("target_update_cond", "alpha_update", "gamma", "target_alpha", "tau", "target_update_cond", "alpha_update"))
def ra_hrl_higher_policy_update(
    rng: jnp.ndarray,

    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    skill_prior: Model,

    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    next_observations: jnp.ndarray,
    dones: jnp.ndarray,

    gamma: float,
    target_alpha: float,
    tau: float,
    target_update_cond: bool,
    alpha_update: bool
):
    if alpha_update:
        rng, _ = jax.random.split(rng)
        new_log_alpha, log_alpha_infos = ra_hrl_higher_log_alpha_update(
            rng=rng,
            skill_prior=skill_prior,
            log_alpha=log_alpha,
            actor=actor,
            observations=observations,
            target_alpha=target_alpha,
        )
    else:
        new_log_alpha = log_alpha
        log_alpha_infos = {"higher_log_alpha": log_alpha, "higher_alpha_loss": 0.0}

    rng, _ = jax.random.split(rng)
    new_actor, actor_infos = ra_hrl_skill_prior_regularized_actor_update(
        rng=rng,
        skill_prior=skill_prior,
        actor=actor,
        critic=critic,
        log_alpha=log_alpha,
        observations=observations
    )

    rng, _ = jax.random.split(rng)
    new_critic, critic_infos = ra_hrl_skill_prior_regularized_critic_update(
        rng=rng,
        skill_prior=skill_prior,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        log_alpha=log_alpha,

        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        dones=dones,
        gamma=gamma
    )

    if target_update_cond:
        new_critic_target = polyak_update(critic, critic_target, tau)
    else:
        new_critic_target = critic_target

    new_models = {
        "higher_actor": new_actor,
        "higher_critic": new_critic,
        "higher_critic_target": new_critic_target,
        "higher_log_alpha": new_log_alpha
    }
    infos = {**actor_infos, **critic_infos, **log_alpha_infos}
    return rng, new_models, infos


@functools.partial(jax.jit, static_argnames=("clip_range", ))
def hrl_higher_policy_onpolicy_update(
    rng: jnp.ndarray,

    higher_actor: Model,
    skill_prior: Model,         # For PPO style regularization

    higher_observations: jnp.ndarray,
    higher_actions: jnp.ndarray,
    returns: jnp.ndarray,       # [nct, 1]      # nct = n_collected_transitions for one episode

    clip_range: float
):
    rng, new_higher_actor, infos = ppo_style_higher_actor_update(
        rng=rng,
        higher_actor=higher_actor,
        skill_prior=skill_prior,
        higher_observations=higher_observations,
        higher_actions=higher_actions,
        returns=returns,
        clip_range=clip_range
    )

    return rng, {"higher_actor": new_higher_actor}, infos