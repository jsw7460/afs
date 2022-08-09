import pprint

import gym
import hydra
from omegaconf import DictConfig, OmegaConf

from sopt.policies import SkillBasedComposedPolicy
from sopt.utils import (
    RewardMDPSensorObservationStackWrapper,
    SpirlMazeEnvWrapper,
    get_maze_env,
    EvalCallback
)

PRETTY_PRINTER = pprint.PrettyPrinter(width=41, compact=True)


@hydra.main(config_path="./sopt/conf", config_name="hrl_dropout_maze")
def main(cfg: DictConfig) -> None:
    pp_cfg = OmegaConf.to_container(cfg, resolve=True)
    PRETTY_PRINTER.pprint(pp_cfg)
    SoptRLWorkSpace(cfg)


if __name__ == '__main__':
    class SoptRLWorkSpace(object):
        TAG = "hrl_dropout_maze"

        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # NOTE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # Hiro 풀어라 ~! Higher buffer size도 현재 늘어나있다

        def __init__(self, cfg: DictConfig):
            self.cfg = cfg

            env = get_maze_env(OmegaConf.to_container(self.cfg.maze_env, resolve=True))
            env = RewardMDPSensorObservationStackWrapper(env, n_frames=cfg.n_frames, max_len=cfg.env_max_len)
            env = SpirlMazeEnvWrapper(env, success_thresh=cfg.maze_env.success_thresh, reward_type="sparse")
            self.env = gym.wrappers.FlattenObservation(env)

            eval_env = get_maze_env(OmegaConf.to_container(self.cfg.maze_env, resolve=True))
            eval_env = RewardMDPSensorObservationStackWrapper(eval_env, n_frames=cfg.n_frames, max_len=cfg.env_max_len)
            eval_env = SpirlMazeEnvWrapper(eval_env, success_thresh=cfg.maze_env.success_thresh, reward_type="sparse")
            eval_env = gym.wrappers.FlattenObservation(eval_env)
            callback = EvalCallback(
                eval_env=eval_env,
                n_eval_episodes=5,
                eval_freq=20000,
                log_path=f"/workspace/callback_results/{self.TAG}",
                best_model_save_path=None,
                deterministic=True,
                render=False,
                verbose=1,
                warn=True
            )

            self.model = None
            self.model, rl_total_timesteps = self.get_model()

            with self.model.hrl_phase():
                print(f"Hrl phase, {rl_total_timesteps}\n" * 30)
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
                policy=SkillBasedComposedPolicy,
            )

            # Build rl components
            rl_total_timesteps = model.build_hrl_models(OmegaConf.to_container(self.cfg.hrl_config, resolve=True))

            return model, rl_total_timesteps


    main()
