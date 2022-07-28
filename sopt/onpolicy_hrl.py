from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from offline_baselines_jax.common.policies import Model
from .buffer import HigherReplayBuffer, LowerReplayBuffer
from .core import hrl_lower_policy_update, hrl_higher_policy_onpolicy_update
from .sopt_skill_prior_hrl import SkillBasedHRLAgent
from .networks import SquashedMLPSkillPrior
from .ra_hrl import check_nested_dict_numpy_equation


@jax.jit
def _get_lower_q_values(
    lower_actor: Model,
    lower_critic: Model,
    higher_observations: jnp.ndarray,
    higher_actions: jnp.ndarray,        # == z
) -> jnp.ndarray:

    action_dist, action_mu, action_log_std = lower_actor.apply_fn(
        {"params": lower_actor.params},
        higher_observations,
        higher_actions,
        deterministic=True
    )

    q_vals = lower_critic.apply_fn(
        {"params": lower_critic.params},
        higher_observations,
        action_mu,
        higher_actions,     # == z
    )
    min_q_val = jnp.min(q_vals, axis=1)
    return min_q_val



class OnpolicySkillBasedHRLAgent(SkillBasedHRLAgent):
    """
    Higher policy: On-policy
    Lower policy: Off-policy
    """
    def __init__(self, n_epoch: int, clip_range: float, *args, **kwargs):
        super(OnpolicySkillBasedHRLAgent, self).__init__(*args, **kwargs)
        self.n_epoch = n_epoch
        self.clip_range = clip_range

    @property
    def skill_prior_class(self):
        return SquashedMLPSkillPrior

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
        super(OnpolicySkillBasedHRLAgent, self).build_higher_policy(higher_policy_config)
        del self.higher_policy.critic, self.higher_policy.critic_target

    def hrl_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Here, we train only the lower policy.
        Higher policy is trained when episode dones.
        """
        lower_alpha_losses, lower_alphas = [], []
        lower_actor_losses, lower_critic_losses = [], []

        for gradient_step in range(gradient_steps):

            lower_replay_data = self.lower_replay_buffer.sample(batch_size)
            target_update_cond = ((self.num_timesteps % self.target_update_interval) == 0)

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

                lower_actor_losses.append(lower_infos["lower_actor_loss"].mean())
                lower_critic_losses.append(lower_infos["lower_critic_loss"].mean())
                lower_alphas.append(lower_infos["lower_alpha"].mean())
                lower_alpha_losses.append(lower_infos["lower_alpha_loss"].mean())

        if self.num_timesteps % 100 == 0:
            self.logger.record_mean("lower/actor_loss(l)", np.mean(lower_actor_losses))
            self.logger.record_mean("lower/critic_loss(l)", np.mean(lower_critic_losses))
            self.logger.record_mean("lower/alpha_loss(l)", np.mean(lower_alpha_losses))
            self.logger.record_mean("lower/alpha(l)", np.mean(lower_alphas))

    def postprocess_episode_done(self):
        print("Enter the episode done callback")
        """Train higher policy"""
        n_collected_transitions = self.higher_replay_buffer.pos
        replay_buffer = self.higher_replay_buffer.sample(batch_size=n_collected_transitions)
        # replay_buffer.observations: [batch_size, subseq_len, obs_dim]
        # Hence, we collect only the first observation of sub-trajectories.

        # NOTE: View batch size as a timestep.
        # Term: nct = n_collected_transitions
        observations = replay_buffer.observations[:, 0, :]      # [nct, obs_dim]
        higher_actions = replay_buffer.higher_actions         # [nct, higher_action_dim]
        q_values = np.array(_get_lower_q_values(
            self.lower_policy.actor,
            self.lower_policy.critic,
            observations,
            higher_actions
        ))

        cumulated_reward = 0.
        higher_rewards = replay_buffer.rewards                  # [nct, higher_action_dim]
        relabeled_returns = np.zeros_like(q_values).reshape(-1, 1) + q_values
        for t in reversed(range(n_collected_transitions)):
            current_reward = higher_rewards[t]
            cumulated_reward = self.gamma * cumulated_reward + current_reward
            relabeled_returns[t] += cumulated_reward

        relabeled_returns = (relabeled_returns - relabeled_returns.mean()) / (relabeled_returns.std() + 1e-8)

        for _ in range(self.n_epoch):
            # Subtract baseline
            self.rng, new_models, infos = hrl_higher_policy_onpolicy_update(
                rng=self.rng,
                higher_actor=self.higher_policy.actor,
                skill_prior=self.skill_prior,
                higher_observations=observations,
                higher_actions=higher_actions,
                returns=relabeled_returns,
                clip_range=self.clip_range
            )
            self.higher_policy.actor = self.higher_policy.actor.replace(params=new_models["higher_actor"].params)
            # self.higher_policy.actor = new_models["higher_actor"]

            print("="*30)
            for k, v in infos.items():
                print(k, v)

        self.higher_replay_buffer.reset()