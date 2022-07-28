import pprint

import hydra
from omegaconf import DictConfig, OmegaConf

from sopt.policies import SkillBasedComposedPolicy
from sopt.utils import (
    get_maze_env,
    PixelObservationWrapper,
    ImgSpirlMazeEnvWrapper
)

PRETTY_PRINTER = pprint.PrettyPrinter(width=41, compact=True)


@hydra.main(config_path="./sopt/conf", config_name="hrl_image_maze")
def main(cfg: DictConfig) -> None:
    pp_cfg = OmegaConf.to_container(cfg, resolve=True)
    PRETTY_PRINTER.pprint(pp_cfg)
    WorkSpace(cfg)


if __name__ == '__main__':
    class WorkSpace(object):

        def __init__(self, cfg: DictConfig):
            self.cfg = cfg

            env = get_maze_env(self.cfg.maze_env)
            env = PixelObservationWrapper(
                env,
                pixels_only=False,
                pixel_keys=("image",),
                render_kwargs={"mode": "rgb_array"},
                channel_first=False
            )

            self.env = ImgSpirlMazeEnvWrapper(env)
            self.model = None
            self.model, rl_total_timesteps = self.get_model()

            with self.model.hrl_phase():
                print(f"Hrl phase, {rl_total_timesteps}\n" * 30)
                self.model.learn(rl_total_timesteps, log_interval=1, reset_num_timesteps=False)

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
