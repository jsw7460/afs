import pprint

import gym
import hydra
from omegaconf import DictConfig, OmegaConf

from sopt.utils import (
    RewardMDPSensorObservationStackWrapper,
    EvalCallback,
    get_kitchen_env,
)

PRETTY_PRINTER = pprint.PrettyPrinter(width=41, compact=True)


@hydra.main(config_path="./sopt/conf", config_name="rl_sobc_kitchen")
def main(cfg: DictConfig) -> None:
    pp_cfg = OmegaConf.to_container(cfg, resolve=True)
    PRETTY_PRINTER.pprint(pp_cfg)
    SoptRLWorkSpace(cfg)


if __name__ == '__main__':

    class SoptRLWorkSpace(object):
        TAG = "rl_sobc_kitchen"

        def __init__(self, cfg: DictConfig):
            self.cfg = cfg

            env = get_kitchen_env(self.cfg.kitchen_env)
            env = RewardMDPSensorObservationStackWrapper(env, n_frames=2, max_len=cfg.env_max_len)
            self.env = gym.wrappers.FlattenObservation(env)

            eval_env = get_kitchen_env(self.cfg.kitchen_env)
            eval_env = RewardMDPSensorObservationStackWrapper(eval_env, n_frames=2, max_len=cfg.env_max_len)
            eval_env = gym.wrappers.FlattenObservation(eval_env)
            callback = EvalCallback(
                eval_env=eval_env,
                n_eval_episodes=1,
                eval_freq=10000,
                log_path=f"/workspace/callback_results/{self.TAG}_seed{self.cfg.eval_seed}",
                best_model_save_path=None,
                deterministic=True,
                render=False,
                verbose=1,
                warn=True
            )

            self.model = None
            self.model, rl_total_timesteps = self.get_model()

            print(f"rl phase, {rl_total_timesteps}\n" * 30)
            self.model.learn(
                rl_total_timesteps,
                log_interval=1,
                tb_log_name=f"{self.TAG}",
                callback=callback
            )

        def get_model(self):
            # Set model
            model = hydra.utils.instantiate(
                self.cfg.sopt_model,
                env=self.env,
            )

            # Load bc model to calculate an intrinsic reward
            rl_timesteps = \
                model.load_bc_model(OmegaConf.to_container(self.cfg.rl_config, resolve=True))

            return model, rl_timesteps

    main()
