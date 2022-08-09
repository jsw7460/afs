from contextlib import contextmanager
from typing import Dict, Tuple

import numpy as np
import optax

from offline_baselines_jax.common.jax_layers import FlattenExtractor
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.utils import get_basic_rngs
from .core import (
    skill_prior_update,
    skill_prior_update_wobn,
)
from .ra_networks import LSTMSubTrajectoryBasedSkillGenerator, RAMLPSkillPrior, RALowLevelSkillPolicy
from .sopt_skill_prior import SOPTSkillEmpowered


class RASkillPrior(SOPTSkillEmpowered):
    def __init__(self, *args, **kwargs):
        super(RASkillPrior, self).__init__(*args, **kwargs)
        self.use_bn = True

    @contextmanager
    def not_apply_bn(self):
        self.use_bn = False
        yield

    @property
    def skill_prior_training_ft(self):
        if self.use_bn:
            return skill_prior_update
        else:
            return skill_prior_update_wobn

    @property
    def skill_generator_class(self):
        return LSTMSubTrajectoryBasedSkillGenerator

    @property
    def skill_prior_class(self):
        return RAMLPSkillPrior

    @property
    def skill_decoder_class(self):
        return RALowLevelSkillPolicy

    def build_hrl_models(self, hrl_config: Dict) -> Tuple[int, int]:
        raise NotImplementedError()

    def build_skill_prior(self) -> Dict:
        features_extractor_class = FlattenExtractor
        init_obs = self.observation_space.sample()[np.newaxis, ...]

        self.rng, rngs = get_basic_rngs(self.rng)
        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        skill_prior_kwargs = {
            "features_extractor": features_extractor,
            "observation_space": self.observation_space,
            "latent_dim": 64,
            "skill_dim": self.skill_dim,
            "dropout": self.dropout,
            "net_arch": self.model_archs["skill_prior"],
            "log_std_coef": self.sp_logstd_coef,
        }
        skill_prior_def = self.skill_prior_class(**skill_prior_kwargs)

        self.skill_prior = Model.create(
            skill_prior_def,
            inputs=[rngs, init_obs],
            tx=optax.adam(learning_rate=5e-4)
        )
        return skill_prior_kwargs

    def skill_prior_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        skill_generator_kl_losses = []
        lowerlevel_policy_losses = []
        skill_prior_losses = []

        skill_means, skill_log_stds, skill_self_lls = [], [], []
        skill_prior_means, skill_prior_log_stds, skill_prior_self_lls = [], [], []
        lowlevel_means, lowlevel_log_stds, lowlevel_self_lls = [], [], []

        for gradient_step in range(gradient_steps):

            # Train using pseudo action, not real one
            observations, actions, last_observations = self.get_skill_prior_training_input()

            self.rng, new_models, infos = self.skill_prior_training_ft(
                rng=self.rng,

                lowlevel_policy=self.lowlevel_policy,
                skill_generator=self.skill_generator,
                skill_prior=self.skill_prior,

                observations=observations,
                actions=actions,
                last_observations=last_observations
            )

            self.apply_new_models(new_models)
            self.num_timesteps += 1

            # Log
            skill_generator_kl_losses.append(infos["skill_generator_kl_loss"])
            lowerlevel_policy_losses.append(infos["lowlevel_policy_loss"])
            skill_prior_losses.append(infos["skill_prior_loss"])

            skill_means.append(infos["skill_mean"])
            skill_log_stds.append(infos["skill_log_std"])
            skill_self_lls.append(infos["skill_generator_self_ll"])

            skill_prior_means.append(infos["skill_prior_mean"])
            skill_prior_log_stds.append(infos["skill_prior_log_std"])
            skill_prior_self_lls.append(infos["skill_prior_self_ll"])

            lowlevel_means.append(infos["lowlevel_policy_mean"])
            lowlevel_log_stds.append(infos["lowlevel_log_std"])
            lowlevel_self_lls.append(infos["lowlevel_self_ll"])

        if (self.num_timesteps % 50) == 0:
            self.logger.record("config/cur_dataset_pos", self.current_dataset_pos)
            self.logger.record_mean("train/skill_gen_kl_loss", np.mean(skill_generator_kl_losses))
            self.logger.record_mean("train/lowlevel_bc_ll", -np.mean(lowerlevel_policy_losses))
            self.logger.record_mean("train/skill_prior_loss", np.mean(skill_prior_losses))

            self.logger.record_mean("skill_gen/mean(g)", np.mean(skill_means))
            self.logger.record_mean("skill_gen/std(g)", np.mean(np.exp(skill_log_stds)))
            self.logger.record_mean("skill_gen/log_std(g)", np.mean(skill_log_stds))
            self.logger.record_mean("skill_gen/self_ll(g)", np.mean(skill_self_lls))

            self.logger.record_mean("skill_prior/mean(p)", np.mean(skill_prior_means))
            self.logger.record_mean("skill_prior/std(p)", np.mean(np.exp(skill_prior_log_stds)))
            self.logger.record_mean("skill_prior/log_std(p)", np.mean(skill_prior_log_stds))
            self.logger.record_mean("skill_prior/self_ll(p)", np.mean(skill_prior_self_lls))

            self.logger.record_mean("lowlevel_policy/mean(l)", np.mean(lowlevel_means))
            self.logger.record_mean("lowlevel_policy/log_std(l)", np.mean(lowlevel_log_stds))
            self.logger.record_mean("lowlevel_policy/self_ll(l)", np.mean(lowlevel_self_lls))

        if (self.num_timesteps % 500) == 0:
            print("=" * 30)
            print("Generated action", infos["pseudoaction_mean"][0])
            print("Real action", actions[0][0])
            self.logger.record_mean("timestep", self.num_timesteps)
            self.logger.dump(self.num_timesteps)

        if (self.num_timesteps % self.skill_prior_model_save_interval) == 0:
            print("*" * 30, f"model saved in {self.skill_prior_model_save_dir}")
            self.skill_prior.save_dict(self.skill_prior_model_save_dir + f"skill_prior_{self.num_timesteps}")
            self.skill_prior.save_batch_stats(self.skill_prior_model_save_dir + f"skill_prior_batch_stats_{self.num_timesteps}")

            self.lowlevel_policy.save_dict(self.skill_prior_model_save_dir + f"lowlevel_policy_{self.num_timesteps}")
            self.skill_generator.save_dict(self.skill_prior_model_save_dir + f"skill_generator_{self.num_timesteps}")

            self.lowlevel_policy.save_batch_stats(self.skill_prior_model_save_dir + f"lowlevel_policy_batch_stats_{self.num_timesteps}")
            self.skill_generator.save_batch_stats(self.skill_prior_model_save_dir + f"skill_generator_batch_stats_{self.num_timesteps}")

        if (self.num_timesteps % self.expert_data_load_interval) == 0:
            self.load_next_expert_buffer()