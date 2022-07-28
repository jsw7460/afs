import warnings
from typing import NamedTuple, List, Dict, Any, Union

import numpy as np
from gym import spaces

from offline_baselines_jax.common.buffers import BaseBuffer

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

EPS = 1E-8


class HigherReplayBufferSample(NamedTuple):
    observations: np.ndarray
    higher_actions: np.ndarray
    lower_actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    traj_lengths: np.ndarray


class LowerReplayBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    higher_actions: np.ndarray
    next_higher_actions: np.ndarray


class HigherReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        higher_action_space: spaces.Space,
        lower_action_space: spaces.Space,
        subseq_len: int,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(HigherReplayBuffer, self).__init__(buffer_size, observation_space, higher_action_space, n_envs=n_envs)
        assert isinstance(observation_space, spaces.Box)
        self.subseq_len = subseq_len

        # Adjust buffer size
        self.buffer_size = max(buffer_size // (n_envs * subseq_len), 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        # Higher replay buffer saves the 'trajectory' of states
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, self.subseq_len) + self.obs_shape,
            dtype=observation_space.dtype
        )

        # If subseq_len is 10 and episode is doned at 37th steps, then rear 3 transitions should be marked.
        self.traj_lengths = np.zeros((self.buffer_size, ))

        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        # This is not necessarily a trajectory
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=higher_action_space.dtype)

        # Higher replay buffer saves the 'trajectory' of lower actions
        lower_action_dim = lower_action_space.shape[-1]
        self.lower_actions = np.zeros((self.buffer_size, self.n_envs, self.subseq_len, lower_action_dim), dtype=lower_action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.observation_dim = None

        try:
            self.observation_dim = self.observation_space.shape[0]
        except:
            pass

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

        # Temporaly save the transitions until the pos touch the subsequence length
        self.temporal_observations = np.zeros((self.n_envs, self.subseq_len) + self.obs_shape, dtype=observation_space.dtype)
        self.temporal_higher_actions = np.zeros((self.n_envs, self.action_dim), dtype=higher_action_space.dtype)
        self.temporal_lower_actions = np.zeros((self.n_envs, self.subseq_len, lower_action_dim), dtype=lower_action_space.dtype)
        self.temporal_rewards = np.zeros((self.n_envs, ), dtype=np.float32)
        self.last_higher_actions = np.zeros((self.n_envs, self.action_dim), dtype=higher_action_space.dtype)
        self.temporal_pos = 0

    def temporal_reset(self):
        self.temporal_pos = 0
        self.temporal_rewards = np.zeros((self.n_envs, ), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        lower_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        higher_action: np.ndarray = None,
        cur_time = None
    ) -> None:
        tp = self.temporal_pos

        # Add to temporal buffer
        self.temporal_observations[0, tp] = obs
        self.temporal_lower_actions[0, tp] = lower_action
        self.temporal_rewards += reward

        if higher_action is not None:
            self.last_higher_actions[0] = np.array(higher_action[0])

        self.temporal_pos += 1

        if (self.temporal_pos == self.subseq_len) or done:

            # If self.temporal_pos < self.subseq_len, then masking the temporal observations after current timestep
            self.temporal_observations[0, self.temporal_pos: ] *= 0
            self.temporal_lower_actions[0, self.temporal_pos: ] *= 0

            # Add to buffer
            self.observations[self.pos] = self.temporal_observations.copy()     # Trajectory
            self.actions[self.pos] = self.last_higher_actions.copy()
            self.rewards[self.pos] = np.array(self.temporal_rewards).copy()
            self.next_observations[self.pos] = np.array(next_obs).copy()
            self.lower_actions[self.pos] = self.temporal_lower_actions.copy()
            self.traj_lengths[self.pos] = self.temporal_pos

            if self.handle_timeout_termination:
                self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

            self.temporal_pos = 0
            self.temporal_rewards = 0.0

            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0


    def _get_samples(
        self, batch_inds: np.ndarray, env = None
    ) -> Union[HigherReplayBufferSample]:
        env_indices = 0

        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        """
        observations: [b, subseq_len, obs_dim]
        higher_actions: [b, action_dim]
        lower_actions: [b, subseq_len, action_dim]
        next_observations: [b, obs_dim]
        dones: [b, 1]
        rewards: [b, 1]
        """

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),        # s_{0: T-1}. Trajectory is required to apply the HIRO.
            self.actions[batch_inds, env_indices, :],                                       # Higher actions
            self.lower_actions[batch_inds, env_indices, :],                                 # Lower actions
            next_obs,                                                                       # s_T
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.traj_lengths[batch_inds]
        )

        return HigherReplayBufferSample(*tuple(data))


class LowerReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        higher_action_space: spaces.Space,
        lower_action_space: spaces.Space,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(LowerReplayBuffer, self).__init__(buffer_size, observation_space, lower_action_space, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=lower_action_space.dtype)

        self.higher_action_dim = higher_action_space.shape[-1]
        self.higher_actions = np.zeros((self.buffer_size, self.n_envs, self.higher_action_dim), dtype=higher_action_space.dtype)
        self.next_higher_actions = np.zeros((self.buffer_size, self.n_envs, self.higher_action_dim), dtype=higher_action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.observation_dim = None
        try:
            self.observation_dim = self.observation_space.shape[0]
        except:
            pass

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,

        higher_action: np.ndarray,          # added
        next_higher_action: np.ndarray,     # added

        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.higher_actions[self.pos] = np.array(higher_action).copy()                  # added
        self.next_higher_actions[self.pos] = np.array(next_higher_action).copy()        # added

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env = None) -> LowerReplayBufferSample:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env = None) -> LowerReplayBufferSample:
        # Sample randomly the env idx
        env_indices = 0

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.higher_actions[batch_inds, env_indices],
            self.next_higher_actions[batch_inds, env_indices]
        )
        return LowerReplayBufferSample(*tuple(data))
