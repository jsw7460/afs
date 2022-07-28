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


@hydra.main(config_path="./sopt/conf", config_name="sp_image_maze")
def main(cfg: DictConfig) -> None:
    pp_cfg = OmegaConf.to_container(cfg, resolve=True)
    PRETTY_PRINTER.pprint(pp_cfg)
    SoptRLWorkSpace(cfg)


if __name__ == '__main__':

    class SoptRLWorkSpace(object):

        def __init__(self, cfg: DictConfig):
            self.cfg = cfg

            env = get_maze_env(self.cfg.maze_env)
            env = PixelObservationWrapper(
                env,
                pixels_only=False,
                pixel_keys=("image", ),
                render_kwargs={"mode": "rgb_array"},
                channel_first=False
            )

            self.env = ImgSpirlMazeEnvWrapper(env)
            self.model = None
            self.model, require_encoder_train, skill_prior_total_timesteps = self.get_model()

            if require_encoder_train:
                print("Encoder train mode\n" * 10)
                with self.model.encoder_learning_phase():
                    self.model.learn(1)

            with self.model.skill_prior_learning_phase():
                print("Skill prior train mode\n" * 10)
                self.model.learn(skill_prior_total_timesteps, log_interval=1)

        def get_model(self):

            # replay_buffer_kwargs = OmegaConf.to_container(cfg.replay_buffer_kwargs, resolve=True)

            # Set model
            model = hydra.utils.instantiate(
                self.cfg.sopt_model,
                env=self.env,
                policy=SkillBasedComposedPolicy,
            )

            # Set expert state buffer
            model.set_image_expert_buffer(self.cfg.expert_buffer_path, self.cfg.expert_buffer_kwargs)

            # Build skill prior model
            require_encoder_train, skill_prior_total_timesteps = \
                model.build_skill_prior_models(OmegaConf.to_container(self.cfg.skill_prior_config, resolve=True))

            return model, require_encoder_train, skill_prior_total_timesteps

    main()
