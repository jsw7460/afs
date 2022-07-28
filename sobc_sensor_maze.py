import pprint

import gym
import hydra
from omegaconf import DictConfig, OmegaConf

from sopt.policies import SkillBasedComposedPolicy
from sopt.utils import (
    RewardMDPSensorObservationStackWrapper,
    get_maze_env
)

PRETTY_PRINTER = pprint.PrettyPrinter(width=41, compact=True)


@hydra.main(config_path="./sopt/conf", config_name="sobc_sensor_maze")
def main(cfg: DictConfig) -> None:
    pp_cfg = OmegaConf.to_container(cfg, resolve=True)
    PRETTY_PRINTER.pprint(pp_cfg)
    SoptRLWorkSpace(cfg)


if __name__ == '__main__':

    class SoptRLWorkSpace(object):

        def __init__(self, cfg: DictConfig):
            self.cfg = cfg

            env = get_maze_env(self.cfg.maze_env)
            env = RewardMDPSensorObservationStackWrapper(env, n_frames=2, max_len=cfg.env_max_len)
            self.env = gym.wrappers.FlattenObservation(env)

            self.model = None
            self.model, bc_timesteps = self.get_model()

            with self.model.pretraining_phase():
                self.model.learn(bc_timesteps, log_interval=1)

        def get_model(self):

            # Set model
            model = hydra.utils.instantiate(
                self.cfg.sopt_model,
                env=self.env,
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
            bc_timesteps = \
                model.build_bc_model(OmegaConf.to_container(self.cfg.bc_config, resolve=True))

            return model, bc_timesteps

    main()
