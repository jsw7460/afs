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


@hydra.main(config_path="./sopt/conf", config_name="hrl_ra_kitchen")
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
            self.model, rl_total_timesteps = self.get_model()

            with self.model.hrl_phase():
                print(f"Hrl phase, {rl_total_timesteps}\n" * 30)
                self.model.learn(
                    rl_total_timesteps,
                    log_interval=1,
                    reset_num_timesteps=False,
                    tb_log_name="RAHRLKitchen"
                )

        def get_model(self):

            # Set replay buffer class
            replay_buffer_class = hydra.utils.get_class(self.cfg.replay_buffer_class)

            # replay_buffer_kwargs = OmegaConf.to_container(cfg.replay_buffer_kwargs, resolve=True)

            # Set model
            model = hydra.utils.instantiate(
                self.cfg.sopt_model,
                env=self.env,
                policy=SkillBasedComposedPolicy,
                replay_buffer_class=replay_buffer_class,
                train_freq=self.cfg.subseq_len      # Update for every subseq_len step
            )

            # Set expert state buffer
            if self.cfg["skill_prior_build"]:
                model.set_expert_buffer(
                    self.cfg.expert_buffer_path,
                    self.cfg.n_frames,
                    self.cfg.subseq_len,
                    max_traj_len=self.cfg.max_expert_traj_len,
                )

            # Build rl components
            rl_total_timesteps = model.build_hrl_models(OmegaConf.to_container(self.cfg.hrl_config, resolve=True))

            return model, rl_total_timesteps

    main()
