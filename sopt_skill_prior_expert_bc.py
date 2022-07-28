import pprint

import gym
import hydra
from omegaconf import DictConfig, OmegaConf

from sopt import SOPTSkillEmpowered
from sopt.policies import SkillBasedComposedPolicy
from sopt.utils import (
    RewardMDPSensorObservationStackWrapper,
    get_kitchen_env
)

PRETTY_PRINTER = pprint.PrettyPrinter(width=41, compact=True)


# NOTE: Transfer layer를 expert action으로 behavior cloning 시키는 실험
@hydra.main(config_path="./sopt/conf", config_name="sopt_skill_prior_conf_kitchen_bc")
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

            self.model = None           # type: SOPTSkillEmpowered
            self.model, skill_prior_total_timesteps, warmup_total_timesteps, rl_total_timesteps = self.get_model()

            with self.model.skill_prior_learning_phase():
                self.model.learn(skill_prior_total_timesteps, log_interval=1)

            with self.model.warmup_phase():
                self.model.learn(warmup_total_timesteps, log_interval=1)

            with self.model.rl_phase():
                self.model.learn(rl_total_timesteps, log_interval=1, reset_num_timesteps=False)

        def get_model(self):

            # Set replay buffer class
            replay_buffer_class = hydra.utils.get_class(self.cfg.replay_buffer_class)

            # replay_buffer_kwargs = OmegaConf.to_container(cfg.replay_buffer_kwargs, resolve=True)

            # Set model
            model = hydra.utils.instantiate(
                self.cfg.sopt_model,
                env=self.env,
                policy=SkillBasedComposedPolicy,
                replay_buffer_class=replay_buffer_class
            )

            # Set expert state buffer
            model.set_expert_buffer(
                self.cfg.expert_buffer_path,
                self.cfg.n_frames,
                self.cfg.subseq_len,
                max_traj_len=self.cfg.max_expert_traj_len,
            )

            # Build skill prior model
            skill_prior_total_timesteps = \
                model.build_skill_prior_models(OmegaConf.to_container(self.cfg.skill_prior_config, resolve=True))

            # Build rl components
            warmup_total_timesteps, rl_total_timesteps = 0, 0
            if self.cfg["rl_build"]:
                warmup_total_timesteps, rl_total_timesteps = model.build_rl_models(
                    OmegaConf.to_container(self.cfg.rl_config, resolve=True)
                )

            return model, skill_prior_total_timesteps, warmup_total_timesteps, rl_total_timesteps

    main()
