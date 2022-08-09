import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple, Type, Union
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import VecEnv

from offline_baselines_jax.common.buffers import ReplayBuffer
from offline_baselines_jax.common.jax_layers import create_mlp
from offline_baselines_jax.common.policies import Model
from offline_baselines_jax.common.type_aliases import (
    GymEnv,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
    RolloutReturn,
    Params
)
from offline_baselines_jax.common.type_aliases import InfoDict
from offline_baselines_jax.common.utils import get_basic_rngs
from offline_baselines_jax.common.utils import should_collect_more_steps
from offline_baselines_jax.sac import SAC
from offline_baselines_jax.sac.policies import SACPolicy
from .buffer import SkillPriorTrainingBuffer


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


@jax.jit
def get_intrinsic_reward(
    rng: jnp.ndarray,
    bc_model: Model,
    observations: jnp.ndarray,
    next_observations: jnp.ndarray
):
    rng, _ = jax.random.split(rng)
    predicted_expert_obs = bc_model.apply_fn(
        {"params": bc_model.params},
        observations,
        next_observations
    )
    intrinsic_reward = -jnp.mean((predicted_expert_obs - next_observations) ** 2)

    return rng, intrinsic_reward


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

        self.single_obs_dim = int(self.observation_space.shape[-1] // 2)
        self.intrinsic_rewards = []

    @contextmanager
    def pretraining_phase(self):
        self.without_exploration = True
        self._train = self.bc_train
        yield
        self.without_exploration = False
        self._train = None

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

    def load_bc_model(self, cfg) -> int:
        self.build_bc_model(cfg)
        self.bc_model = self.bc_model.load_dict(cfg["load_dir"] + "/sobc_3000000")

        return cfg["total_timesteps"]

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
            next_observations = replay_data.observations[:, 2, ...]     # Prediction: one step next observation (Not concatenated)

            self.bc_model, infos = bc_model_update(
                self.bc_model,
                observations,
                next_observations
            )
            self.bc_timestep += 1

            if self.bc_timestep % 100000 == 0:
                for k, v in infos.items():
                    print(f"Timestep: {self.bc_timestep}", k, v)

            if self.bc_timestep % 500000 == 0:
                self.bc_model.save_dict(self.bc_save_dir + f"sobc_{self.bc_timestep}")

            if self.bc_timestep % 100000 == 0:
                self.load_next_expert_buffer()

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        assert not self.without_exploration
        assert self.bc_model is not None

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.rng, intrinsic_reward = get_intrinsic_reward(
                rng=self.rng,
                bc_model=self.bc_model,
                observations=self.last_obs.copy(),
                next_observations=new_obs[0][self.single_obs_dim:],
            )
            # rewards += np.array(intrinsic_reward)

            self.intrinsic_rewards.append(intrinsic_reward)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Log intrinsic rewards
                    self.logger.record("rollout/intr_rewards", np.mean(self.intrinsic_rewards))
                    self.intrinsic_rewards = []

                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
