from typing import Dict

import jax.tree_util
import numpy as np

from typing import Dict

import numpy as np

from offline_baselines_jax.common.jax_layers import (
    FlattenExtractor,
)
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.utils import get_basic_rngs
from .core import hrl_higher_policy_update, ra_hrl_higher_policy_update
from .networks import (
    MLPSkillPrior,
)
from .ra_networks import RAMLPSkillPrior, RASquashedMLPSkillPrior


EPS = 1E-8

from .sopt_skill_prior_hrl import SkillBasedHRLAgent
from .hrl_policies import RAHigherPolicy, RALowerPolicy


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


class RASkillBasedHRLAgent(SkillBasedHRLAgent):
    def __init__(self, *args, **kwargs):
        super(RASkillBasedHRLAgent, self).__init__(*args, **kwargs)

    @property
    def save_lowlevel_episodes(self) -> bool:
        return False

    def load_skill_prior(self, pretrained_kwargs: Dict, load_dir: str, load_epoch: int):
        skill_prior_def = RAMLPSkillPrior(**pretrained_kwargs["skill_prior"])

        self.rng, rngs = get_basic_rngs(self.rng)
        init_obs = self.higher_observation_space.sample()[np.newaxis, ...]
        skill_prior = Model.create(skill_prior_def, inputs=[rngs, init_obs])  # Skill prior is not trained
        skill_prior = skill_prior.load_dict(load_dir + f"skill_prior_{load_epoch}")  # Load param
        skill_prior = skill_prior.load_batch_stats(
            load_dir + f"skill_prior_batch_stats_{load_epoch}")  # Load batch stats (for batch normalization)
        self.skill_prior = skill_prior
        print(f"Skill prior params are loaded from {load_dir}\n" * 10)

    def build_higher_policy(self, higher_policy_config: Dict) -> None:
        # NOTE: Inside the build, higher policy is initialized by the pretrained skill prior
        self.higher_policy = RAHigherPolicy(
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
        self.lower_policy = RALowerPolicy(
            rng=self.rng,
            observation_space=self.lower_observation_space,
            action_space=self.action_space,
            conditioned_dim=self.skill_dim,
            lr_schedule=self.lr_schedule,
            net_arch=lower_policy_config["net_arch"],
            features_extractor_class=FlattenExtractor,
            n_critics=lower_policy_config["n_critics"],
            dropout=self.dropout
        )
        self.lower_policy.build_model(lower_policy_config)

    def load_pseudo_action_policy(self, pretrained_kwargs: Dict, load_dir: str, load_epoch: int):
        pass        # Not used

    def calculate_intrinsic_reward(self, **kwargs):
        return 0.0

    def hrl_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        higher_alpha_losses, higher_alphas = [], []
        higher_actor_losses, higher_critic_losses = [], []
        higher_prior_divergences = []

        for gradient_step in range(gradient_steps):
            higher_replay_data = self.higher_replay_buffer.sample(batch_size)

            self.rng, higher_new_models, higher_infos = ra_hrl_higher_policy_update(
                rng=self.rng,

                actor=self.higher_policy.actor,
                critic=self.higher_policy.critic,
                critic_target=self.higher_policy.critic_target,
                log_alpha=self.higher_log_ent_coef,
                skill_prior=self.skill_prior,

                observations=self.get_representation(higher_replay_data.observations[:, 0, ...]),
                actions=higher_replay_data.higher_actions,
                rewards=higher_replay_data.rewards,
                next_observations=self.get_representation(higher_replay_data.next_observations),
                dones=higher_replay_data.dones,
                gamma=self.gamma,
                target_alpha=5.0,
                tau=self.tau,
                target_update_cond=True,
                alpha_update=True,
            )

            same = check_nested_dict_numpy_equation(self.skill_prior.params, self.higher_policy.actor.params)

            self.higher_policy.actor = higher_new_models["higher_actor"]
            self.higher_policy.critic = higher_new_models["higher_critic"]
            self.higher_policy.critic_target = higher_new_models["higher_critic_target"]
            self.higher_log_ent_coef = higher_new_models["higher_log_alpha"]

            higher_actor_losses.append(higher_infos["higher_actor_loss"].mean())
            higher_prior_divergences.append(higher_infos["prior_divergence"].mean())
            higher_critic_losses.append(higher_infos["higher_critic_loss"].mean())
            higher_alphas.append(higher_infos["higher_alpha"].mean())
            higher_alpha_losses.append(higher_infos["higher_alpha_loss"].mean())

        if self.num_timesteps % 100 == 0:
            print("rewards", higher_replay_data.rewards.mean())
            print("dones", higher_replay_data.dones.mean())
            print("alpha_loss_prior_divergence", higher_infos["alpha_loss_prior_divergence"])

        if self.num_timesteps % 100 == 0:
            self.logger.record_mean("higher/actor_loss(h)", np.mean(higher_actor_losses))
            self.logger.record_mean("higher/prior_div(h)", np.mean(higher_prior_divergences))
            self.logger.record_mean("higher/critic_loss(h)", np.mean(higher_critic_losses))
            self.logger.record_mean("higher/alpha_loss(h)", np.mean(higher_alpha_losses))
            self.logger.record_mean("higher/alpha(h)", np.mean(higher_alphas))