import pprint

import gym
import hydra
from omegaconf import DictConfig, OmegaConf

from sopt.policies import SkillBasedComposedPolicy
from sopt.utils import (
    RewardMDPSensorObservationStackWrapper,
    get_kitchen_env
)

PRETTY_PRINTER = pprint.PrettyPrinter(width=41, compact=True)


@hydra.main(config_path="./sopt/conf", config_name="sp_ra_kitchen_wobn")
def main(cfg: DictConfig) -> None:
    pp_cfg = OmegaConf.to_container(cfg, resolve=True)
    PRETTY_PRINTER.pprint(pp_cfg)
    SoptRLWorkSpace(cfg)


if __name__ == '__main__':

    class SoptRLWorkSpace(object):

        def __init__(self, cfg: DictConfig):
            self.cfg = cfg

            env = get_kitchen_env(self.cfg.kitchen_env)
            env = RewardMDPSensorObservationStackWrapper(env, n_frames=cfg.n_frames, max_len=cfg.env_max_len)
            self.env = gym.wrappers.FlattenObservation(env)

            self.model = None
            self.model, skill_prior_total_timesteps = self.get_model()

            with self.model.skill_prior_learning_phase():
                with self.model.use_real_action_sequence():
                    with self.model.not_apply_bn():
                        self.model.learn(
                            skill_prior_total_timesteps,
                            log_interval=1,
                            tb_log_name="RASkillPriorTrainingWobn"
                        )

        def get_model(self):
            # Set model. To use real action sequence, pseudo action dim is explicitely given.
            pseudo_action_dim = self.env.action_space.shape[-1]
            model = hydra.utils.instantiate(
                self.cfg.sopt_model,
                env=self.env,
                policy=SkillBasedComposedPolicy,
                pseudo_action_dim=pseudo_action_dim
            )

            # Set expert state buffer
            model.set_expert_buffer(
                hydra.utils.get_class(self.cfg.expert_buffer_class),
                self.cfg.expert_buffer_path,
                self.cfg.n_frames,
                self.cfg.subseq_len,
                max_traj_len=self.cfg.max_expert_traj_len,
            )

            # Build skill prior model
            skill_prior_total_timesteps = \
                model.build_skill_prior_models(OmegaConf.to_container(self.cfg.skill_prior_config, resolve=True))

            return model, skill_prior_total_timesteps

    main()
