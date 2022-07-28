import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.noise import ActionNoise

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.jax_layers import FlattenExtractor
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import (
    GymEnv,
    Schedule
)
from offline_baselines_jax.common.utils import get_basic_rngs
from .core import det_skill_prior_update
from .networks import DeterministicLowLevelSkillPolicy
from .policies import SkillBasedComposedPolicy
from .sopt_skill_prior import SOPTSkillEmpowered


# Define: PseudoAction model is deterministic
class DetPseudoActionSkillEmpowered(SOPTSkillEmpowered):
    def __init__(
            self,
            env: Union[GymEnv, str],
            policy: Union[str] = SkillBasedComposedPolicy,
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[ReplayBuffer] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: int = 0,
            _init_setup_model: bool = True,
            without_exploration: bool = False,
            ###
            dropout: float = 0.0,
            n_frames: int = 3,
            subseq_len: int = 10,
            batch_dim: int = 256,
            hidden_dim: int = 128,
            skill_dim: int = 5,

            model_archs: Optional[Dict[str, List]] = {},
            bc_reg_coef: float = 0.5,

            expert_dataset_load_interval: int = 500_000
    ):
        super(DetPseudoActionSkillEmpowered, self).__init__(
            env=env,
            policy=policy,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            _init_setup_model=_init_setup_model,
            without_exploration=without_exploration,
            dropout=dropout,
            n_frames=n_frames,
            subseq_len=subseq_len,
            batch_dim=batch_dim,
            hidden_dim=hidden_dim,
            skill_dim=skill_dim,
            model_archs=model_archs,
            bc_reg_coef=bc_reg_coef,
            expert_dataset_load_interval=expert_dataset_load_interval
        )

    def build_lowlevel_policy(self) -> Dict:
        self.rng, rngs = get_basic_rngs(self.rng)

        features_extractor_class = FlattenExtractor
        init_obs = self.observation_space.sample()[np.newaxis, ...]
        obs_dim = init_obs.shape[-1]

        features_extractor = features_extractor_class(_observation_space=self.observation_space)
        lowlevel_policy_kwargs = {
            "features_extractor": features_extractor,
            "observation_space": self.observation_space,
            "lowlevel_action_dim": obs_dim,
            "net_arch": self.model_archs["lowlevel_policy"],
            "dropout": 0.1
        }
        lowlevel_policy_def = DeterministicLowLevelSkillPolicy(**lowlevel_policy_kwargs)
        init_skill = jnp.zeros((1, self.skill_dim))

        self.lowlevel_policy = Model.create(
            lowlevel_policy_def,
            inputs=[rngs, init_obs, init_skill],
            tx=optax.adam(1e-4)
        )

        return lowlevel_policy_kwargs

    def build_skill_prior_models(self, skill_prior_config: Dict) -> int:
        if not skill_prior_config["build"]: return 0        # Return required training step

        # Define: A dictionary which saves a parameters of pretrained model. This will be saved with model together.
        pretrained_kwargs = {}

        # Define: LowLevel Skill Policy (Learning with pseudo actions (e.g., delta sequence))
        lowlevel_policy_kwargs = self.build_lowlevel_policy()
        # Define: skill prior
        skill_generator_kwargs = self.build_skill_generator()
        # Save kwargs of pretrained models
        skill_prior_kwargs = self.build_skill_prior()

        pretrained_kwargs.update({
            "lowlevel_policy": lowlevel_policy_kwargs,
            "skill_generator": skill_generator_kwargs,
            "skill_prior": skill_prior_kwargs,
            "normalizing_max": self.expert_buffer.normalizing_max.copy(),
            "normalizing_min": self.expert_buffer.normalizing_min.copy()    # required to finetuning
        })

        os.makedirs(skill_prior_config["config_save_dir"], exist_ok=True)
        config_path = skill_prior_config["config_save_dir"] + "det_config"
        with open(config_path, "wb") as f:
            pickle.dump(pretrained_kwargs, f)
        print(f"Config saved in {config_path}")

        self.skill_prior_model_save_dir = skill_prior_config["model_save_dir"]
        return skill_prior_config["total_timesteps"]

    def skill_prior_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        skill_generator_kl_losses = []
        lowerlevel_policy_losses = []
        skill_prior_losses = []
        skill_means, skill_log_stds, skill_self_lls = [], [], []
        skill_prior_means, skill_prior_log_stds, skill_prior_self_lls = [], [], []

        REAL_ACTIONS = 0
        for gradient_step in range(gradient_steps):

            # Train using pseudo action, not real one
            replay_data = self.expert_buffer.sample_skill_prior_training_data(batch_size, real_actions=False)
            actions = np.array(replay_data.actions, dtype=np.float)
            self.rng, new_models, infos = det_skill_prior_update(
                rng=self.rng,

                lowlevel_policy=self.lowlevel_policy,
                skill_generator=self.skill_generator,
                skill_prior=self.skill_prior,

                observations=replay_data.observations,
                actions=actions,
                last_observations=replay_data.last_observations
            )
            # print("RUN?" * 999)
            self.apply_new_models(new_models)
            self.num_timesteps += 1
            if np.random.uniform(0, 1) < 0.01:
                print("XXX" * 10, infos["lowlevel_sample"][0][:30])
                print("YYY" * 10, infos["lowlevel_sample"][0][30:])
                print("ZZZ" * 10, actions[0][-1][:30])
                print("WWW" * 10, jnp.mean((infos["lowlevel_sample"][0][:30] - actions[0][-1][:30]) ** 2))
                # print("Pseudoaction policy Sample", infos["pseudoaction_sample"])

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

        if (self.num_timesteps % 50) == 0:
            self.logger.record("config/real_action_train", REAL_ACTIONS)
            self.logger.record("config/cur_dataset_pos", self.current_dataset_pos)
            self.logger.record_mean("train/skill_gen_kl_loss", np.mean(skill_generator_kl_losses))
            self.logger.record_mean("train/lowlevel_bc_mse", np.mean(lowerlevel_policy_losses))
            self.logger.record_mean("train/skill_prior_loss", np.mean(skill_prior_losses))

            self.logger.record_mean("skill_gen/mean(g)", np.mean(skill_means))
            self.logger.record_mean("skill_gen/std(g)", np.mean(np.exp(skill_log_stds)))
            self.logger.record_mean("skill_gen/log_std(g)", np.mean(skill_log_stds))
            self.logger.record_mean("skill_gen/self_ll(g)", np.mean(skill_self_lls))

            self.logger.record_mean("skill_prior/mean(p)", np.mean(skill_prior_means))
            self.logger.record_mean("skill_prior/std(p)", np.mean(np.exp(skill_prior_log_stds)))
            self.logger.record_mean("skill_prior/log_std(p)", np.mean(skill_prior_log_stds))
            self.logger.record_mean("skill_prior/self_ll(p)", np.mean(skill_prior_self_lls))

        if (self.num_timesteps % 500) == 0:
            self.logger.record_mean("timestep", self.num_timesteps)
            self.logger.dump(self.num_timesteps)

        if (self.num_timesteps % 100000) == 0:
            print("*" * 30, f"model saved in {self.skill_prior_model_save_dir}")
            self.skill_prior.save_dict(self.skill_prior_model_save_dir + f"skill_prior_{self.num_timesteps}")
            self.lowlevel_policy.save_dict(self.skill_prior_model_save_dir + f"det_lowlevel_policy_{self.num_timesteps}")
            self.skill_generator.save_dict(self.skill_prior_model_save_dir + f"skill_generator_{self.num_timesteps}")

            self.skill_prior.save_batch_stats(self.skill_prior_model_save_dir + f"skill_prior_batch_stats_{self.num_timesteps}")
            self.lowlevel_policy.save_batch_stats(self.skill_prior_model_save_dir + f"det_lowlevel_policy_batch_stats_{self.num_timesteps}")
            self.skill_generator.save_batch_stats(self.skill_prior_model_save_dir + f"skill_generator_batch_stats_{self.num_timesteps}")

        if (self.num_timesteps % self.expert_data_load_interval) == 0:
            self.load_next_expert_buffer()