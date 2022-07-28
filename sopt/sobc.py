import numpy as np
import optax
from contextlib import contextmanager
import os
from typing import Any, Dict, Optional, Tuple, Type, Union, Callable

from stable_baselines3.common.noise import ActionNoise

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.type_aliases import GymEnv, Schedule
from offline_baselines_jax.sac import SAC
from offline_baselines_jax.sac.policies import SACPolicy
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.jax_layers import create_mlp
from offline_baselines_jax.common.utils import get_basic_rngs
from .buffer import SkillPriorTrainingBuffer
from offline_baselines_jax.common.type_aliases import Params, InfoDict
import jax
import jax.numpy as jnp

@jax.jit
def bc_model_update(
    bc_model: Model,
    observations: jnp.ndarray,
    next_observations: jnp.ndarray,
) -> Tuple[Model, Dict]:
    def bc_model_loss(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred = bc_model.apply_fn(
            {"params": params},
            observations,
        )
        loss = jnp.mean((pred - next_observations) ** 2)
        return loss, {"bc_loss": loss}
    bc_model, infos = bc_model.apply_gradient(bc_model_loss)

    return bc_model, infos


class StateOnlyBC(SAC):

    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, Type[SACPolicy]] = SACPolicy,
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
        tensorboard_log: Optional[str] = "StateOnlyBC",
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: int = 0,
        _init_setup_model: bool = True,
        without_exploration: bool = False,
    ):
        super(StateOnlyBC, self).__init__(
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
            without_exploration=without_exploration
        )

        self.expert_dataset_dir = None
        self.expert_dataset_list = []
        self.expert_buffer = None
        self.bc_model = None       # type: Model
        self.bc_save_dir = None
        self.bc_timestep = 0
        self._train = None      # type: Callable[..., None]
        self.current_dataset_pos = 0

    @contextmanager
    def pretraining_phase(self):
        self.without_exploration = True
        self._train = self.bc_train
        yield

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        return self._train(gradient_steps, batch_size)

    def offline_train(self, gradient_steps: int, batch_size: int) -> None:
        return self._train(gradient_steps, batch_size)

    def build_bc_model(self, cfg):
        init_observations = self.observation_space.sample()
        state_dim = init_observations.shape[-1] // 2       # Observation is stacked 2 frame
        bc_def = create_mlp(state_dim, cfg["net_arch"])
        self.rng, rngs = get_basic_rngs(self.rng)
        self.bc_model = Model.create(bc_def, inputs=[rngs, init_observations], tx=optax.adam(1e-4))
        self.bc_save_dir = cfg["save_dir"]
        return cfg["total_timesteps"]

    def load_bc_model(self, cfg):
        self.build_bc_model()
        self.bc_model = Model.load_dict(cfg["load_dir"])

    def set_expert_buffer(
            self,
            buffer_class,
            path: str,  # Directory. Not a file.
            n_frames: int,
            subseq_len: int,
            max_traj_len: int = 1_000_000
    ):
        self.expert_buffer_class = buffer_class
        self.expert_dataset_dir = path
        self.expert_dataset_list = sorted([f for f in os.listdir(path)])

        current_dataset_path = path + self.expert_dataset_list[self.current_dataset_pos]
        self.expert_buffer = buffer_class(
            data_path=current_dataset_path,
            n_frames=n_frames,
            subseq_len=subseq_len,
            max_traj_len=max_traj_len
        )
        self.expert_buffer.relabel_action_by_obs_difference()
        print(f"Expert data is load from {current_dataset_path}\n" * 10)
        print(f"\t Num trajectories: {len(self.expert_buffer)}")
        print(f"\t Num transitions: {self.expert_buffer.n_transitions}")

    def load_next_expert_buffer(self):
        self.current_dataset_pos += 1
        self.current_dataset_pos %= len(self.expert_dataset_list)
        current_dataset_path = self.expert_dataset_dir + self.expert_dataset_list[self.current_dataset_pos]
        print(f"Expert data is load from {current_dataset_path}")
        self.expert_buffer = self.expert_buffer_class(
            data_path=current_dataset_path,
            n_frames=1,
            subseq_len=10
        )
        self.expert_buffer.relabel_action_by_obs_difference()

    def bc_train(self, gradient_steps, batch_size: int = 64):
        replay_data: SkillPriorTrainingBuffer
        for gradient_step in range(gradient_steps):
            replay_data = self.expert_buffer.sample(batch_size=batch_size)       # Output a sequence of observations.
            observations = replay_data.observations[:, :2, ...]     # Input: Frame
            observations = observations.reshape(batch_size, -1)
            next_observations = replay_data.observations[:, 2, ...]     # Prediction: one step observation

            self.bc_model, infos = bc_model_update(
                self.bc_model,
                observations,
                next_observations
            )
            self.bc_timestep += 1

            if self.bc_timestep % 100 == 0:
                for k, v in infos.items():
                    print(f"Timestep: {self.bc_timestep}", k, v)

            if self.bc_timestep % 100000 == 0:
                self.bc_model.save_dict(self.bc_save_dir + f"sobc_{self.bc_timestep}")

            if self.bc_timestep % 100000 == 0:
                self.load_next_expert_buffer()