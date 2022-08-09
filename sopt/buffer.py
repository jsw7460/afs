import pickle
import warnings
from typing import NamedTuple, List, Dict, Any, Union

import gym
import numpy as np
from gym import spaces

from offline_baselines_jax.common.buffers import ReplayBuffer, BaseBuffer

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

EPS = 1E-8


def clamp(n, smallest, largest): return max(smallest, min(n, largest))


def flatten(array: np.ndarray) -> np.ndarray:
    batch_size = array.shape[0]
    return array.reshape(batch_size, -1).copy()


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


class SkillPriorTrainingBuffer(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    last_observations: np.ndarray


class PosGoalReplayBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    goals: np.ndarray


class GoalGeneratorBufferSample(NamedTuple):
    observations: np.ndarray
    subgoal_observations: np.ndarray
    goal_observations: np.ndarray
    target_future_hop: np.ndarray


class LastObsContainedBufferSample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    last_observations: np.ndarray


class ExpertObservationsSample(NamedTuple):
    observations: np.ndarray    # [batch_size, subseq_len, obs_dim]
    last_observations: np.ndarray   # [batch_size, obs_dim]. 특정 sequence의 마지막 obs. Subseq의 마지막 아님


# class PosGoalReplayBuffer(ReplayBuffer):
#     def __init__(
#         self,
#         buffer_size: int,
#         observation_space: gym.Space,
#         action_space: gym.Space,
#         n_envs: int,
#         optimize_memory_usage: bool = False,
#     ):
#         super(PosGoalReplayBuffer, self).__init__(
#             buffer_size=buffer_size,
#             observation_space=observation_space,
#             action_space=action_space,
#             n_envs=n_envs,
#             optimize_memory_usage=optimize_memory_usage,
#             handle_timeout_termination=True
#         )
#
#         self.goals = np.zeros((self.buffer_size, self.n_envs, 2))        # goals: xy - pos
#
#     def add(
#         self,
#         obs: np.ndarray,
#         next_obs: np.ndarray,
#         action: np.ndarray,
#         reward: np.ndarray,
#         done: np.ndarray,
#         infos: List[Dict[str, Any]],
#     ) -> None:
#         # Reshape needed when using multiple envs with discrete observations
#         # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
#
#         # Copy to avoid modification by reference
#         self.observations[self.pos] = np.array(obs).copy()
#
#         if self.optimize_memory_usage:
#             self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
#         else:
#             self.next_observations[self.pos] = np.array(next_obs).copy()
#
#         self.actions[self.pos] = np.array(action).copy()
#         self.rewards[self.pos] = np.array(reward).copy()
#         self.dones[self.pos] = np.array(done).copy()
#         self.goals[self.pos] = np.array([info.get("goal", None) for info in infos]).copy()
#
#         if self.handle_timeout_termination:
#             self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
#
#         self.pos += 1
#         if self.pos == self.buffer_size:
#             self.full = True
#             self.pos = 0
#
#     def _get_samples(self, batch_inds: np.ndarray, env = None) -> PosGoalReplayBufferSample:
#         # Sample randomly the env idx
#         env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
#
#         if self.optimize_memory_usage:
#             next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
#         else:
#             next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
#
#         data = (
#             self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
#             self.actions[batch_inds, env_indices, :],
#             next_obs,
#             # Only use dones that are not due to timeouts
#             # deactivated by default (timeouts is initialized as an array of False)
#             (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
#             self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
#             self.goals[batch_inds, env_indices]
#         )
#         return PosGoalReplayBufferSample(*tuple(data))


class LastObservationSavedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int,
        optimize_memory_usage: bool = False,
    ):
        """
        Although last observation can be obtained in the fly,
        but I choose to save by hand due to sampling speed.
        :param buffer_size:
        :param observation_space:
        :param action_space:
        :param n_envs:
        :param optimize_memory_usage:
        """
        super(LastObservationSavedReplayBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=True
        )
        self.last_observations = np.zeros_like(self.observations)

    def add_with_last_observation(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        last_obs: np.ndarray,
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, gym.spaces.Discrete):
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
        self.last_observations[self.pos] = np.array(last_obs).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> LastObsContainedBufferSample:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

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
            self.last_observations[batch_inds, env_indices]
        )
        return LastObsContainedBufferSample(*tuple(data))


# class CondVaeGoalGeneratorBuffer(object):
#     def __init__(self, data_path: str, n_subgoal: int = 20):
#         with open(data_path + ".pkl", "rb") as f:
#             self.dataset = pickle.load(f)
#
#         self.n_traj = None
#         self.buffer_size = None
#         self.observations = None
#         self.traj_lengths = None
#         self.subgoal_cand = None
#         self.observation_traj = None
#         self.goal_pos = None
#         self.max_traj_len = None
#         self.mean_target_future_hop = None
#
#         self.current_trunc = 0
#         self.n_subgoal = n_subgoal
#
#         self.truncation_reset(0)
#
#     def truncation_reset(self, current_trunc: int):
#         """
#         :param current_trunc: 데이터가 커서, 전체 dataset을 truncation해서 사용해야 한다.
#         그 truncation이 몇 번째인지 알려주는 수가 current_trunc이다.
#         :return:
#         """
#         dataset = self.dataset[500 * current_trunc: 500 * (current_trunc + 1)]
#
#         self.n_traj = len(dataset)
#         observations = [data["observations"].copy() for data in dataset]
#         traj_lengths = [len(observation) for observation in observations]
#         subgoal_cand = [np.floor(np.linspace(0, traj_len - 1, self.n_subgoal)) for traj_len in traj_lengths]
#
#         self.traj_lengths = np.array(traj_lengths)  # [5166, ]
#         self.mean_target_future_hop = np.floor(np.mean(self.traj_lengths / self.n_subgoal))     # 평균 몇 step이후의 goal을 예측하는지
#         self.subgoal_cand = np.array(subgoal_cand, dtype=np.int32)  # [5166, 20]
#         self.buffer_size = len(self.traj_lengths)
#
#         obs_sample = observations[0][0]
#         max_traj_len = np.max(self.traj_lengths)
#         self.observation_traj = np.zeros(([self.n_traj, max_traj_len] + [*obs_sample.shape]))  # [5166, 501, 32, 32, 3]
#         self.goal_pos = np.zeros((self.n_traj, 2))  # [5166, 2]
#
#         for traj_idx, traj_len in zip(range(self.n_traj), self.traj_lengths):
#             self.observation_traj[traj_idx, :traj_len, ...] = dataset[traj_idx]["observations"].copy()
#             # Goal is fixed during episode. So we just need 0th goal.
#             self.goal_pos[traj_idx] = dataset[traj_idx]["goals"][0].copy()
#
#         self.max_traj_len = max_traj_len
#
#     def sample(self, batch_size: int = 256) -> GoalGeneratorBufferSample:
#         batch_inds = np.random.randint(0, self.n_traj, size=batch_size)
#         timesteps = np.random.randint(0, self.traj_lengths)[batch_inds]
#         subgoal_inds = self.subgoal_cand[batch_inds]
#         subgoal_inds[timesteps[:, np.newaxis] > subgoal_inds] = 99999
#         subgoal_inds = np.min(subgoal_inds, axis=1)
#
#         # If timestep == 0, then subgoal index is zero. So we insert 1 for such indices.
#         subgoal_inds[subgoal_inds == 0] = 1
#
#         observation = self.observation_traj[batch_inds, timesteps, ...].copy()
#         subgoal_observation = self.observation_traj[batch_inds, subgoal_inds, ...].copy()
#         goal_observation = self.goal_pos[batch_inds].copy()
#         target_future_hop = (subgoal_inds - timesteps)[..., np.newaxis]
#
#         data = (observation / 255, subgoal_observation / 255, goal_observation, target_future_hop)
#
#         return GoalGeneratorBufferSample(*data)

class SensorBasedExpertBuffer:
    def __init__(
        self,
        data_path: str = None,
        n_frames: int = 1,
        max_traj_len: int = 1_000_000,
        dataset: Union[List] = None,
        **kwargs
    ):
        if not dataset:
            with open(data_path + ".pkl", "rb") as f:
                dataset = pickle.load(f)
                dataset = dataset[1: max_traj_len]

        n_traj = len(dataset)
        observations = [data["observations"].copy() for data in dataset]
        actions = [data["actions"].copy() for data in dataset]
        traj_lengths = [len(obs) for obs in observations]

        obs_sample = observations[0][0]
        act_sample = actions[0][0]
        max_traj_len = np.max(traj_lengths) + n_frames      # trajectory의 맨 앞부분을 n_frames만큼 늘려준다.       # 1009

        # +1: Relabel action 할 때, 문제있는 부분이 생김
        self.observation_traj = np.zeros(([n_traj, max_traj_len] + [*obs_sample.shape]))
        self.action_traj = np.zeros(([n_traj, max_traj_len] + [*act_sample.shape]))
        self.relabled_action_traj = None  # It will be labled later

        self.observation_dim = obs_sample.shape[-1]
        self.n_frames = n_frames

        # Save the normalizing factor
        self.normalizing_max = 0.0
        self.normalizing_min = 0.0

        # Save goal observation if exist
        self.goal_traj = None
        if dataset[0].get("goals", None) is not None:
            goal_sample = dataset[0].get("goals")[0]
            self.goal_traj = np.zeros(([n_traj, max_traj_len + 1] + [*goal_sample.shape]))

        # Save
        truncated_n_traj = 0
        truncated_traj_lengths = []
        removed_traj_n = 0

        traj_idx_z = 0
        for traj_idx, traj_len in zip(range(n_traj), traj_lengths):
            if self.should_remove_data(dataset[traj_idx]):
                removed_traj_n += 1
                continue

            # if traj_len < n_frames + 1:
            #     removed_traj_n += 1
            #     continue
            # pos = dataset[traj_idx]["observations"][-1][:2]
            # if pos[0] >= 16.0 and pos[1] >= 16.0:
            #     removed_traj_n += 1
            #     continue


            # The environment observation is temporally stacked

            original_obs = dataset[traj_idx]["observations"]
            original_act = dataset[traj_idx]["actions"]

            # 첫 번째 observation은 history가 없기 때문에, n_frames-1 만큼 늘려준다. 첫번째는 이미 들어있기 때문에 -1 해준다.
            frame_aug_obs = np.repeat(dataset[traj_idx]["observations"][0][np.newaxis, ...], repeats=n_frames-1, axis=0)
            frame_aug_obs = np.vstack((frame_aug_obs, original_obs))

            # Action은 stacking 안해주지만, 그래도 dataset에서는 augment한다. 안그러면 sampling할 때 index가 안맞는다.
            frame_aug_act = np.repeat(dataset[traj_idx]["actions"][0][np.newaxis, ...], repeats=n_frames-1, axis=0)
            frame_aug_act = np.vstack((frame_aug_act, original_act))

            self.observation_traj[traj_idx_z, :traj_len + n_frames - 1] = frame_aug_obs.copy()
            self.action_traj[traj_idx_z, :traj_len + n_frames - 1] = frame_aug_act.copy()

            if self.goal_traj is not None:
                original_goal = dataset[traj_idx]["goals"]
                frame_aug_goals = np.repeat(dataset[traj_idx]["goals"][0][np.newaxis, ...], repeats=n_frames-1, axis=0)
                frame_aug_goals = np.vstack((frame_aug_goals, original_goal))
                self.goal_traj[traj_idx_z, :traj_len + n_frames - 1] = frame_aug_goals.copy()

            truncated_n_traj += 1
            truncated_traj_lengths.append(traj_len)

            traj_idx_z += 1

        print(f"\tTotal {removed_traj_n} trajectories are removed")
        print(f"\tThus for skill prior training, {traj_idx_z} trajectories are trained")

        # Delete removed trajectory
        self.observation_traj = np.delete(self.observation_traj, range(n_traj - removed_traj_n, n_traj), axis=0)
        self.action_traj = np.delete(self.action_traj, range(n_traj - removed_traj_n, n_traj), axis=0)

        if self.goal_traj is not None:
            self.goal_traj = np.delte(self.goal_traj, range(n_traj - removed_traj_n, n_traj), axis=0)

        self.n_traj = truncated_n_traj
        self.traj_lengths = truncated_traj_lengths
        self.n_transitions = sum(self.traj_lengths)

        self.max_traj_len = max_traj_len

        self.lower_action_dim = self.observation_dim        # type: int

    def should_remove_data(self, trajectory: Dict):
        raise NotImplementedError()

    def relabel_action_by_obs_difference(self, observation_part: slice = None) -> None:
        """
        :param observation_part: Required for multitask environment to avoid relabeling the onehot part.
        :return:
        """
        # Define action_traj := delta sequence
        action_traj = (self.observation_traj[:, 1:, ...] - self.observation_traj[:, :-1, ...])

        if observation_part:        # this run for multitask dataset
            action_traj = action_traj[..., observation_part]

        # State가 끝나는 부분에서 state간의 차이를 action으로 정의하면, 하나의 수치가 비정상적으로 커진다. 그래서 강제로 0으로 만든다.
        # 실제로 마지막 action은 의미가 없기 때문에 0으로 두는것이 옳다.
        action_traj[np.arange(self.n_traj), np.array(self.traj_lengths) + self.n_frames - 2, ...] = 0       # [n_traj, max_traj_len-1, 4]

        # 위에서 [:, 1:, ...] - [:, :-1, ...] 따위로 slicing을 했기 때문에 길이가 하나 줄어든다.
        # observation과 길이를 맞춰야 해서, zero padding action을 넣어준다.
        last_action = np.zeros((self.n_traj, 1, self.observation_dim))
        action_traj = np.concatenate((action_traj, last_action), axis=1)            # [n_traj, max_traj_len, 4]

        # Normalize and clipping
        reshaped_action_traj = action_traj.reshape(action_traj.shape[0] * action_traj.shape[1], -1)

        max_act = reshaped_action_traj.max(axis=0, keepdims=True)
        min_act = reshaped_action_traj.min(axis=0, keepdims=True)

        self.normalizing_max = max_act
        self.normalizing_min = min_act

        action_traj = 2 * ((action_traj - min_act) / (max_act - min_act + EPS)) - 1
        action_traj = np.clip(action_traj, -1.0 + EPS, 1.0 - EPS)

        self.lower_action_dim = action_traj.shape[-1]
        self.relabled_action_traj = action_traj

    def sample(
        self,
        batch_size: int = 256,
        relabled_action: bool = True,
        return_action: bool = True,
        return_last_observation: bool = False,
    ):
        """
        NOTE: This returns the self.n_frames stacking observation. Thus the output observation has the size of
        NOTE: [batch_size, n_frames, obs_dim]
        :param batch_size:
        :param relabled_action: If "Fake" action, set relabled_action True, else False.
        :param return_action:
        :param return_last_observation:
        :return:
        """

        batch_inds = np.random.randint(0, self.n_traj, size=batch_size)
        timesteps = np.random.randint(self.n_frames - 1, np.array(self.traj_lengths)+self.n_frames-1)     # [n_traj, ]

        # For next timestep
        # +2 까지 해야 실제 arange는 +1 까지 됨. 뒤에서 obs, next_obs로 나뉠 것이므로 일부러 하나 더 timestep을 만들어주는 중
        idxs = np.array([np.arange(timestep - self.n_frames + 1, timestep + 2) for timestep in timesteps])
        observations_chunk = self.observation_traj[np.arange(self.n_traj)[:, np.newaxis], idxs][batch_inds].copy()
        observations = observations_chunk[:, : -1, ...]
        next_observations = observations_chunk[:, 1:, ...]

        if return_last_observation:
            # Trajectory의 맨 마지막부분을 return한다. Expert의 마지막 상태가 task를 추론하기에 적합하다고 판단했기 때문이다.
            last_observation_idxs = np.array(
                [np.arange(traj_length - 1, traj_length) for traj_length in self.traj_lengths]
            )
            last_observations = self.observation_traj[np.arange(self.n_traj)[:, np.newaxis], last_observation_idxs][batch_inds].copy()
            return observations, np.squeeze(last_observations, axis=1)

        if self.goal_traj is not None:
            goals_chunk = self.goal_traj[np.arange(self.n_traj)[:, np.newaxis], idxs][batch_inds].copy()
            goals = goals_chunk[:, : -1, ...]
            next_goals = goals_chunk[:, 1:, ...]

            observations = {
                "observations": observations,
                "goals": goals
            }

            next_observations = {
                "observations": next_observations,
                "goals": next_goals
            }

        if return_action:
            action_traj_buffer = self.relabled_action_traj if relabled_action else self.action_traj
            actions = action_traj_buffer[np.arange(self.n_traj), timesteps, ...][batch_inds].copy()
            return observations, actions, next_observations

        else:
            return observations, next_observations


class SoptFiSensorBasedExpertBuffer(SensorBasedExpertBuffer):
    def __init__(
        self,
        data_path: str,
        n_frames: int,
        subseq_len: int,
        max_traj_len: int = 1_000_000,
        **kwargs,
    ):
        # Frame만큼 이어서 하나의 observation을 만들어 줌.
        # Timestep 단위로  subseq_len만큼 buffer를 늘려주는건 superclass에서 한다.
        if "pkl" not in data_path.lower():
            data_path = data_path + ".pkl"
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
            dataset = dataset[1:]
            obs_sample = dataset[0]["observations"]
            observation_dim = obs_sample.shape[-1]

        dataset = dataset[:max_traj_len]
        reformed_dataset = []
        # Before making buffer, assure that state is stacked 'n_frames' times.
        for data in dataset:        # NOTE: Fori -- episode
            obs = data["observations"]      # traj_len, obs_dim
            actions = data["actions"]  # traj_len, act_dim

            first_obs = obs[0]
            traj_len = len(obs)
            if traj_len < n_frames + 1: continue  # Discard short trajectory

            # From here, we stack n_frames of expert observations.
            observations = np.empty((traj_len, n_frames, observation_dim))

            for traj_idx in range(traj_len):        # NOTE: Fori -- timestep
                history_start_idx = clamp(traj_idx - n_frames, 0, np.inf)       # e.g., 0
                padding_size = clamp(n_frames - traj_idx, 0, np.inf)            # n_frames
                history = obs[history_start_idx: traj_idx, ...]                 # []
                padding = np.repeat(first_obs[np.newaxis, ...], repeats=padding_size, axis=0)   # [n_frames, ]
                history = np.concatenate((padding, history), axis=0)
                observations[traj_idx] = history

            observations = observations.reshape((traj_len, -1))
            reformed_data = {"observations": observations, "actions": actions}
            reformed_dataset.append(reformed_data)

        del dataset
        super(SoptFiSensorBasedExpertBuffer, self).__init__(dataset=reformed_dataset, n_frames=subseq_len)

    def should_remove_data(self, trajectory: Dict):
        raise NotImplementedError()

    def sample(
        self,
        batch_size: int = 256,
        **kwargs
    ):
        observations, last_observations = super(SoptFiSensorBasedExpertBuffer, self).sample(
            batch_size=batch_size,
            return_last_observation=True,
            **kwargs
        )
        # observations: [batch_size, subseq_len, obs_dim * n_frames] (obs_dim = 17, n_frames = 3 --> 51 dim is outputed)
        # last_observations: [batch_size, obs_dim * n_frames]
        return ExpertObservationsSample(observations, last_observations)

    def sample_skill_prior_training_data(
        self,
        batch_size: int = 256,
        real_actions: bool = False,
        **kwargs
    ):
        batch_inds = np.random.randint(0, self.n_traj, size=batch_size)
        timesteps = np.random.randint(self.n_frames, np.array(self.traj_lengths) - 1)  # [n_traj, ]

        # For next timestep
        # +2 까지 해야 실제 arange는 +1 까지 됨. 뒤에서 obs, next_obs로 나뉠 것이므로 일부러 하나 더 timestep을 만들어주는 중
        idxs = np.array([np.arange(timestep - self.n_frames + 1, timestep + 2) for timestep in timesteps])

        # Observation & Next observation
        observations_chunk = self.observation_traj[np.arange(self.n_traj)[:, np.newaxis], idxs][batch_inds]
        observations = observations_chunk[:, : -1, ...]
        next_observations = observations_chunk[:, 1:, ...]

        # Last observation
        last_observation_idxs = np.array([np.arange(traj_length - 1, traj_length) for traj_length in self.traj_lengths])
        last_observations \
            = self.observation_traj[np.arange(self.n_traj)[:, np.newaxis], last_observation_idxs][batch_inds].copy()
        last_observations = np.squeeze(last_observations, axis=1)

        # Action
        action_traj = self.action_traj if real_actions else self.relabled_action_traj
        actions = action_traj[np.arange(self.n_traj)[:, np.newaxis], idxs][batch_inds].copy()
        actions = actions[:, : -1, ...]
        actions = np.clip(actions, -1 + 1e-6, 1 - 1e-6)
        return SkillPriorTrainingBuffer(observations, actions, next_observations, last_observations)

    def __len__(self):
        return len(self.observation_traj)


class KitchenExpertBuffer(SoptFiSensorBasedExpertBuffer):
    def __init__(self, data_path: str, n_frames: int, subseq_len: int, max_traj_len: int = 1_000_000, **kwargs):
        super(KitchenExpertBuffer, self).__init__(
            data_path=data_path,
            n_frames=n_frames,
            subseq_len=subseq_len,
            max_traj_len=max_traj_len,
            **kwargs
        )

    def should_remove_data(self, trajectory: Dict):
        return len(trajectory["observations"]) < self.n_frames


class MazeExpertBuffer(SoptFiSensorBasedExpertBuffer):
    def __init__(self, data_path: str, n_frames: int, subseq_len: int, max_traj_len: int = 1_000_000, **kwargs):
        super(MazeExpertBuffer, self).__init__(
            data_path=data_path,
            n_frames=n_frames,
            subseq_len=subseq_len,
            max_traj_len=max_traj_len,
            **kwargs
        )

    def should_remove_data(self, trajectory: Dict):
        len_condition = len(trajectory["observations"]) < self.n_frames
        qpos = trajectory["observations"][-1][:2]
        # pos_condition = qpos[0] >= 16.0 and qpos[1] >= 16.0       # For size 20
        # pos_condition = qpos[0] >= 16.0 and qpos[1] >= 16.0
        pos_condition = qpos[0] <= 1.70 and qpos[1] >= 9.30         # For LargeMaze layout
        return len_condition or pos_condition


# class MultiTaskReplayBuffer:
#     # Multitask replay buffer. This stores an N replay buffers(N == #domain).
#     # In this project,
#     # "Domain" = drawer-close, reach, window-close, ...
#     # "Task" = Various initial - goal position of each domain.
#     def __init__(
#         self,
#         buffer_size_per_domain: int,
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         domain_name_list: List,
#         **kwargs,
#     ):
#         self.domain_name_list = domain_name_list
#         self.buffer_size_per_domain = buffer_size_per_domain
#         self.observation_space = observation_space
#         self.action_space = action_space
#
#         self.n_domains = len(domain_name_list)
#         self.__domain_name = None   # type: str
#
#         # Define replay buffer. We call it multi"domain".
#         self.multidomain_buffer = {}
#         for domain_name in self.domain_name_list:
#             self.multidomain_buffer[domain_name] = ReplayBuffer(
#                 buffer_size=buffer_size_per_domain,
#                 observation_space=observation_space,
#                 action_space=action_space
#             )
#
#     def __str__(self):
#         return "MultiTaskReplayBuffer with tasks " + ", ".join(self.domain_name_list)
#
#     def add(
#         self,
#         obs: np.ndarray,
#         next_obs: np.ndarray,
#         action: np.ndarray,
#         reward: np.ndarray,
#         done: np.ndarray,
#         infos: List[Dict[str, Any]]
#     ) -> None:
#         # Insert the domain's transition to corresponding buffer
#         # [0] means 0th environment (Not use parallel env here)
#         domain_name = infos[0]["domain_name"]
#         self.multidomain_buffer[domain_name].add(obs, next_obs, action, reward, done, infos)
#
#     def reset(self):
#         raise NotImplementedError()
#
#     def sample(self, batch_size: int, *args, **kwargs):
#         """
#         :param batch_size: NOTE: This is batch size "per domain buffer"
#         """
#
#         observations = []
#         actions = []
#         next_observations = []
#         dones = []
#         rewards = []
#         for domain_name, buffer in self.multidomain_buffer.items():
#             replay_data = buffer.sample(batch_size)
#             observations.append(replay_data.observations)
#             actions.append(replay_data.actions)
#             next_observations.append(replay_data.next_observations)
#             dones.append(replay_data.dones)
#             rewards.append(replay_data.rewards)
#
#         observations = np.vstack(observations)
#         actions = np.vstack(actions)
#         next_observations = np.vstack(next_observations)
#         dones = np.vstack(dones)
#         rewards = np.vstack(rewards)
#         return ReplayBufferSamples(
#             observations=observations,
#             actions=actions,
#             next_observations=next_observations,
#             dones=dones,
#             rewards=rewards
#         )
#
#
# class MultiTaskSensorBasedExpertBuffer:
#     def __init__(
#         self,
#         data_dir: str,
#         n_frames: int,
#         domain_name2idx: Dict[str, int],
#         obs_dim_without_onehot: int,
#         max_traj_len: int
#     ):
#         """
#         :param domain_name2idx: Get from environment
#         E.g., {"drawer-close": 0, "window-open": 1, ...}
#         """
#         self.domain_name2idx = domain_name2idx
#         self.n_domains = len(domain_name2idx)
#         self.multidomain_expert_buffer = {}
#         self.obs_dim_without_onehot = obs_dim_without_onehot
#
#         self.observation_part = slice(0, obs_dim_without_onehot)
#
#         for domain_name, domain_idx in domain_name2idx.items():
#             buffer = SensorBasedExpertBuffer(
#                 data_path=data_dir+f"/{domain_name}-noise0.5-seed0",
#                 n_frames=n_frames,
#                 max_traj_len=max_traj_len
#             )
#             self.multidomain_expert_buffer[domain_name] = buffer
#
#     def relabel_action_by_obs_difference(self) -> None:
#         for domain_name, buffer in self.multidomain_expert_buffer.items():
#             buffer.relabel_action_by_obs_difference(observation_part=self.observation_part)
#
#     def sample(self, batch_size: int, *args, **kwargs):
#         """
#         :param batch_size: NOTE: This is batch size "per domain buffer"
#         """
#         observations = []
#         actions = []
#         next_observations = []
#
#         for domain_name, buffer in self.multidomain_expert_buffer.items():
#             observation, action, next_observation = buffer.sample(batch_size, return_action=True)
#             observation = flatten(observation)
#             actions.append(action)
#             next_observation = flatten(next_observation)
#
#             one_hot = np.zeros((batch_size, self.n_domains))
#             domain_idx = self.domain_name2idx[domain_name]
#             one_hot[:, domain_idx] = 1
#
#             observation = np.hstack((observation, one_hot))
#             next_observation = np.hstack((next_observation, one_hot))
#
#             observations.append(observation)
#             next_observations.append(next_observation)
#
#         observations = np.vstack(observations)
#         actions = np.vstack(actions)
#         next_observations = np.vstack(next_observations)
#         return observations, actions, next_observations
#
#
# @dataclasses.dataclass()
# class Trajectory:
#     def __init__(
#         self,
#         observations: Union[List[np.ndarray], np.ndarray],      # [traj_len, obs_dim] or [batch_size, traj_len, obs_dim]
#         actions: Union[List[np.ndarray], np.ndarray],           # [traj_len, act_dim] or [batch_size, traj_len, act_dim]
#         **kwargs
#     ):
#         self.observations = np.array(observations)
#         self.actions = np.array(actions)
#         self.traj_len = len(observations)
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#
#     def __len__(self):
#         assert len(self.observations) == len(self.actions)
#         return len(self.observations)
#
#
# class StateActionTrajectoryDataset(Dataset):
#
#     def __init__(
#         self,
#         subseq_len: int = 10,
#         data_path: Union[str] = None,
#         dataset: Union[List[Trajectory]] = None,
#         max_traj_len: int = 1_000_000,
#         n_frames: int = 1,
#     ):
#         """
#         :param subseq_len: Skill embedding subsequence length
#         :param data_path:
#         :param dataset:
#         :param max_traj_len:
#         :param n_frames: This is part of environment. Independent of subseq_len.
#         """
#         self.seqs = []      # type: List[Trajectory]
#         self.subseq_len = subseq_len
#         self.n_transitions = 0
#
#         if data_path is not None:
#             with open(data_path + ".pkl", "rb") as f:
#                 dataset = pickle.load(f)
#                 obs_sample = dataset[0]["observations"]
#                 observation_dim = obs_sample.shape[-1]
#
#             for data in dataset:                # For each trajectory
#                 obs = data["observations"]      # traj_len, obs_dim
#                 first_obs = obs[0]
#
#                 actions = data["actions"]  # traj_len, act_dim
#                 traj_len = len(actions)
#                 if traj_len < self.subseq_len: continue  # Discard short trajectory
#
#                 # From here, we stack n_frames of expert observations.
#                 observations = np.empty((traj_len, n_frames, observation_dim))
#
#                 for traj_idx in range(traj_len):
#                     history_start_idx = clamp(traj_idx - n_frames, 0, np.inf)
#                     padding_size = clamp(n_frames - traj_idx, 0, np.inf)
#                     history = obs[history_start_idx: traj_idx, ...]
#                     padding = np.repeat(first_obs[np.newaxis, ...], repeats=padding_size, axis=0)
#                     history = np.concatenate((padding, history), axis=0)
#                     observations[traj_idx] = history
#
#                 observations = observations.reshape((traj_len, -1))
#                 self.seqs.append(Trajectory(observations, actions))  # Transform to attribute dictionary
#
#         else:
#             if dataset is None: raise LookupError("one of data path or dataset should be given")
#             dataset = dataset[:max_traj_len]
#             self.seqs = dataset
#
#         self.n_transitions = sum([len(traj) for traj in self.seqs])
#
#         def collate_func(trajectories: List[Trajectory]):
#             _observations, _actions, _last_observation = [], [], []
#
#             for traj in trajectories:
#                 _observations.append(traj.observations)
#                 _actions.append(traj.actions)
#                 _last_observation.append(traj.last_observation)
#
#             return Trajectory(_observations, _actions, last_observation=np.array(_last_observation))
#
#         self._collate_func = collate_func
#
#     def __getitem__(self, item):
#         seq = random.choice(self.seqs)  # type: Trajectory
#         start_idx = np.random.randint(0, seq.traj_len - self.subseq_len)
#
#         observations = seq.observations[start_idx: start_idx + self.subseq_len]
#         actions = seq.actions[start_idx: start_idx + self.subseq_len]
#         # NOTE: Last observation is sequence's last observation. Not 'sub'sequence's last observation.
#         last_observation = seq.observations[-1]
#         return Trajectory(observations, actions, last_observation=np.array(last_observation))
#
#     def __len__(self):
#         return len(self.seqs)
#
#     def get_dataloader(self, *, batch_size: int, **loader_kwargs):
#         return DataLoader(self, batch_size=batch_size, collate_fn=self._collate_func, **loader_kwargs, drop_last=True)


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
        self.buffer_size = max(buffer_size // n_envs, 1)

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
        if isinstance(lower_action_space, gym.spaces.Box):
            lower_action_dim = lower_action_space.shape[-1]
        elif isinstance(lower_action_space, gym.spaces.Discrete):
            lower_action_dim = 1
        else:
            raise NotImplementedError()
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

        self.temporal_observations *= 0
        self.temporal_higher_actions *= 0
        self.temporal_lower_actions *= 0

    def reset(self) -> None:
        super(HigherReplayBuffer, self).reset()
        self.temporal_reset()

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        lower_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        higher_action: np.ndarray = None,
        cur_time=None
    ) -> None:
        # Add to temporal buffer
        self.temporal_observations[0, self.temporal_pos] = obs.copy()
        self.temporal_lower_actions[0, self.temporal_pos] = lower_action.copy()
        self.temporal_rewards += reward

        if higher_action is not None:
            higher_action = np.array(higher_action).copy()
            self.last_higher_actions[0] = np.array(higher_action[0])

        self.temporal_pos += 1

        if (self.temporal_pos == self.subseq_len) or done:
            # If self.temporal_pos < self.subseq_len, then masking the temporal observations after current timestep
            self.temporal_observations[0, self.temporal_pos:] *= 0
            self.temporal_lower_actions[0, self.temporal_pos:] *= 0

            # Add to buffer
            self.observations[self.pos] = self.temporal_observations.copy()     # Trajectory
            self.actions[self.pos] = self.last_higher_actions.copy()
            self.rewards[self.pos] = np.array(self.temporal_rewards).copy()
            self.next_observations[self.pos] = np.array(next_obs).copy()        # Not a trajectory
            self.lower_actions[self.pos] = self.temporal_lower_actions.copy()
            self.traj_lengths[self.pos] = self.temporal_pos
            self.dones[self.pos] = np.array(done).copy()
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

        next_obs = self.next_observations[batch_inds, env_indices, :]

        """
        observations: [b, subseq_len, obs_dim]
        higher_actions: [b, action_dim]
        lower_actions: [b, subseq_len, action_dim]
        next_observations: [b, obs_dim]
        dones: [b, 1]
        rewards: [b, 1]
        """

        data = (
            self.observations[batch_inds, env_indices, :],        # s_{0: T-1}. Trajectory is required to apply the HIRO.
            self.actions[batch_inds, env_indices, :],                                       # Higher actions
            self.lower_actions[batch_inds, env_indices, :],                                 # Lower actions
            next_obs,                                                                       # s_T
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.traj_lengths[batch_inds]
        )
        return HigherReplayBufferSample(*tuple(data))


class HigherRolloutBuffer(HigherReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(HigherRolloutBuffer, self).__init__(*args, **kwargs)


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
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.higher_actions[batch_inds, env_indices],
            self.next_higher_actions[batch_inds, env_indices]
        )
        return LowerReplayBufferSample(*tuple(data))
