from collections import deque

from typing import Union

import gym
import numpy as np

from contextlib import contextmanager


class MTRewardMDPSensorObservationStackWrapper(gym.Wrapper):
    def __init__(self, env: Union[gym.Env,], n_frames: int):
        super(MTRewardMDPSensorObservationStackWrapper, self).__init__(env)
        self.env = env
        self._n_frames = n_frames
        self.obs_frames = deque(maxlen=n_frames)

        org_obs_space = env.observation_space
        low = np.repeat(org_obs_space.low[np.newaxis, ...], repeats=n_frames, axis=0)
        high = np.repeat(org_obs_space.high[np.newaxis, ...], repeats=n_frames, axis=0)

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32, seed=0)

    def reset(self, **kwargs):
        obs = self.env.reset()
        for _ in range(self._n_frames):
            self.obs_frames.append(obs.copy())
        observation = np.array(self.obs_frames).copy()
        return observation

    def step(self, action: np.ndarray):
        assert len(self.obs_frames) == self._n_frames
        obs, reward, done, info = self.env.step(action)
        self.obs_frames.append(obs.copy())

        observation = np.array(self.obs_frames).copy()
        return observation, reward, done, info

    @contextmanager
    def set_task(self, task):
        self.env.set_task(task)
        yield
        pass