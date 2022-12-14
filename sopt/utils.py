import dataclasses
import os
import random
import sys
import time
import warnings
from collections import OrderedDict, deque
from collections.abc import MutableMapping
from copy import copy
from copy import deepcopy
from functools import wraps
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Union
from typing import Tuple

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.vec_env import VecMonitor, is_vecenv_wrapped

# import d4rl_mujoco
Schedule = Callable[[float], float]


def get_linear_fn(start: float, end: float, end_fraction: float) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return:
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def clock(fmt="[{elapsed: 0.8f}s {name}({arg_str}) --> {result}]"):

    def decorate(func):
        @wraps(func)
        def clocked(*args, **kwargs):
            t0 = time.time()
            _result = func(*args, **kwargs)
            elapsed = time.time() - t0
            name = func.__name__
            arg_list = []
            if args:
                arg_list.append(', '.join(repr(arg) for arg in args))
            if kwargs:
                pairs = [f"{k}={v}" for k, v in sorted(kwargs.items())]
                arg_list.append(', '.join(pairs))

            arg_str = ', '.join(arg_list)
            result = repr(_result)
            print(fmt.format(**locals()))
            return _result

        return clocked
    return decorate


def get_maze_env(cfg):
    # import here to avoid all dependency matters.
    sys.path.append("/workspace/spirl")
    from spirl.rl.envs.maze import MazeEnv, ParamDict

    # Create my own maze env
    class RandMazeEnv(MazeEnv):
        START_POS = np.array(cfg["start_pos"])
        TARGET_POS = np.array(cfg["target_pos"])

        def _default_hparams(self):
            default_dict = ParamDict({
                "name": cfg["name"]
            })
            return super()._default_hparams().overwrite(default_dict)

        def render(self, mode="rgb_array"):
            img = super(RandMazeEnv, self).render(mode)     # ??? ????????? 255??? ??? ???????
            img = np.array(img * 255., dtype=np.uint8)
            return img
    return RandMazeEnv(cfg)


def get_kitchen_env(cfg):
    # import here to avoid all dependency matters.
    sys.path.append("/workspace/spirl")
    from spirl.rl.envs.kitchen import KitchenEnv
    env = KitchenEnv({**cfg})
    return env


STATE_KEY = "state"
class PixelObservationWrapper(gym.ObservationWrapper):
    """Augment observations by pixel values."""

    def __init__(
        self, env, pixels_only=True, render_kwargs=None, pixel_keys=("pixels",), channel_first: bool = True,
    ):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            pixels_only: If `True` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If `False`, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs: Optional `dict` containing keyword arguments passed
                to the `self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the `OrderedDict` of observations.
                Defaults to 'pixels'.

        Raises:
            ValueError: If `env`'s observation spec is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If `env`'s observation already contains any of the
                specified `pixel_keys`.
        """

        super(PixelObservationWrapper, self).__init__(env)

        if channel_first:
            self.transposing = lambda x: x.transpose((2, 0, 1))
        else:
            self.transposing = lambda x: x

        if render_kwargs is None:
            render_kwargs = {}

        for key in pixel_keys:
            render_kwargs.setdefault(key, {})

            render_mode = render_kwargs[key].pop("mode", "rgb_array")
            assert render_mode == "rgb_array", render_mode
            render_kwargs[key]["mode"] = "rgb_array"

        wrapped_observation_space = env.observation_space

        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = set([STATE_KEY])
        elif isinstance(wrapped_observation_space, (spaces.Dict, MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only:
            # Make sure that now keys in the `pixel_keys` overlap with
            # `observation_keys`
            overlapping_keys = set(pixel_keys) & set(invalid_keys)
            if overlapping_keys:
                raise ValueError(
                    "Duplicate or reserved pixel keys {!r}.".format(overlapping_keys)
                )

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict()
            self.observation_space.spaces[STATE_KEY] = wrapped_observation_space

        # Extend observation space with pixels.
        pixels_spaces = {}
        for pixel_key in pixel_keys:
            pixels = self.env.render(**render_kwargs[pixel_key])

            if np.issubdtype(pixels.dtype, np.integer):
                low, high = (0, 255)
            elif np.issubdtype(pixels.dtype, np.float):
                low, high = (-float("inf"), float("inf"))
            else:
                raise TypeError(pixels.dtype)

            pixels_space = spaces.Box(
                shape=self.transposing(pixels).shape, low=low, high=high, dtype=pixels.dtype
            )
            pixels_spaces[pixel_key] = pixels_space

        self.observation_space.spaces.update(pixels_spaces)

        self._env = env
        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, wrapped_observation):
        if self._pixels_only:
            observation = OrderedDict()
        elif self._observation_is_dict:
            observation = type(wrapped_observation)(wrapped_observation)
        else:
            observation = OrderedDict()
            observation[STATE_KEY] = wrapped_observation

        pixel_observations = {
            pixel_key: self.transposing(self.env.render(**self._render_kwargs[pixel_key]))
            for pixel_key in self._pixel_keys
        }

        observation.update(pixel_observations)
        return observation


class SpirlMazeEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, success_thresh: float = 3.0, reward_type: str = "sparse"):
        """
        Original Spirl's forked d4rl maze environment always outputs done=False and
        setting target goal's position is doned outsided of the environment.
        We fix it here.
        :param env:
        :param success_thresh:
        """
        super(SpirlMazeEnvWrapper, self).__init__(env)
        self.success_thresh = success_thresh
        self.check_done = lambda pos, tar: np.linalg.norm(pos - tar)
        self.reward_type = reward_type

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, time_limit_done, info = super(SpirlMazeEnvWrapper, self).step(action)
        if obs.shape[0] == 1:       # No stack
            cur_pos = np.squeeze(obs, axis=0)[0: 2]
        else:
            cur_pos = obs[-1][0: 2]     # Stacked --> last timestep observation is current position

        distance = self.check_done(cur_pos, self.env.target)
        done = (distance <= self.success_thresh)
        if self.reward_type.lower() == "dense":
            reward = -distance
        else:
            reward = float(done)
        return obs, reward, time_limit_done, info

    def render(self, mode="human", **kwargs):
        return self.env.render(mode=mode, **kwargs)


class MazeActionDiscretizer(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(MazeActionDiscretizer, self).__init__(env)
        self.action_space = gym.spaces.Discrete(n=4)
        # 0 --> (1, 0)
        # 1 --> (-1, 0)
        # 2 --> (0, -1)
        # 3 --> (0, 1)

    def step(self, action):
        cont_action = None
        if action == 0:
            cont_action = (1., 0.)
        elif action == 1:
            cont_action = (-1., 0.)
        elif action == 2:
            cont_action = (0., -1.)
        elif action == 3:
            cont_action = (0., 1.)
        return super(MazeActionDiscretizer, self).step(cont_action)


class ImgSpirlMazeEnvWrapper(SpirlMazeEnvWrapper):
    """
    Stack image. Don't stack state.
    """
    def __init__(self, env: gym.Env, succes_thresh: float = 3.0, n_frames: int = 2, max_len: int = 2000):
        super(ImgSpirlMazeEnvWrapper, self).__init__(env, succes_thresh)

        self.n_frames = n_frames
        self.stacked_image = deque(maxlen=n_frames)

        self.__timestep = 0
        self.max_len = max_len

        org_space = self.env.observation_space
        state_dim = org_space["state"].shape[-1]
        img_shape = org_space["image"].shape[:2]
        n_channel = org_space["image"].shape[-1]

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(state_dim, )),
            "image": gym.spaces.Box(low=0, high=255, shape=(*img_shape, n_channel * n_frames), dtype=np.uint8)
        })

    @property
    def timestep(self):
        return self.__timestep

    @timestep.setter
    def timestep(self, value):
        self.__timestep = value

    def reset(self, **kwargs):
        self.timestep = 0
        obs = self.env.reset()
        for _ in range(100):
            self.env.render(mode="rgb_array")       # so that camera can "reach" agent

        for _ in range(self.n_frames):
            self.stacked_image.append(obs["image"])

        image = np.concatenate(self.stacked_image, axis=-1)
        state = obs["state"]

        obs = {"image": image, "state": state}
        return obs

    def step(self, action):
        obs, _, _, info = self.env.step(action)
        self.timestep += 1
        self.stacked_image.append(obs["image"])

        image = np.concatenate(self.stacked_image, axis=-1)
        state = obs["state"]

        distance = self.check_done(state[0:2], self.env.target)
        done = (distance <= self.success_thresh)
        reward = float(done)

        # print(f"Timestep {self.timestep}, Cur: {state[0:2]}, Target: {self.env.target}, Distance: {distance}")
        time_done = False
        if self.timestep >= self.max_len:
            time_done = True

        obs = {"image": image, "state": state}
        return obs, reward, done or time_done, info


class GoalMDPSensorObservationStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_frames: int):
        super(GoalMDPSensorObservationStackWrapper, self).__init__(env)
        from offline_baselines_jax.common.preprocessing import is_image_space
        assert not is_image_space(env.observation_space)
        self.env = env
        self._n_frames = n_frames
        self.obs_frames = deque(maxlen=n_frames)
        self.goal_frames = deque(maxlen=n_frames)

        dummy_obs = self.reset()

        self.observation_space = gym.spaces.Dict({
            "observations": gym.spaces.Box(-np.inf, np.inf, shape=dummy_obs["observations"].shape),
            "goals": gym.spaces.Box(-np.inf, np.inf, shape=dummy_obs["goals"].shape)
        })

    def __getattr__(self, item):
        return getattr(self.env, item)

    def reset(self, **kwargs):
        obs = self.env.reset()
        goals = self.env.get_target()
        for _ in range(self._n_frames):
            self.obs_frames.append(obs.copy())
            self.goal_frames.append(copy(goals))
        _obs = np.array(self.obs_frames).copy()
        _goals = np.array(self.goal_frames).copy()

        observation = {
            "observations": _obs,
            "goals": _goals
        }
        return observation

    def step(self, action: np.ndarray):
        assert len(self.obs_frames) == self._n_frames
        obs, reward, done, info = self.env.step(action)
        goals = info["goal"]
        self.obs_frames.append(obs.copy())
        self.goal_frames.append(goals.copy())

        _obs = np.array(self.obs_frames).copy()
        _goals = np.array(self.goal_frames).copy()

        observation = {
            "observations": _obs,
            "goals": _goals
        }
        return observation, reward, done, info


class RewardMDPSensorObservationStackWrapper(gym.Wrapper):
    def __init__(self, env: Union[str, gym.Env], n_frames: int, max_len: int = 2_000, eval_checker: bool = False):

        if isinstance(env, str): env = gym.make(env)
        else: pass

        super(RewardMDPSensorObservationStackWrapper, self).__init__(env)
        self.env = env
        self._n_frames = n_frames
        self.max_len = max_len
        self.obs_frames = deque(maxlen=n_frames)
        self.eval_checker = eval_checker

        # dummy_obs = self.reset()
        org_obs_space = env.observation_space
        low = np.repeat(org_obs_space.low[np.newaxis, ...], repeats=n_frames, axis=0)
        high = np.repeat(org_obs_space.high[np.newaxis, ...], repeats=n_frames, axis=0)

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.__timestep = 0

    @property
    def timestep(self):
        return self.__timestep

    @timestep.setter
    def timestep(self, value):
        self.__timestep = value

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.timestep = 0
        for _ in range(self._n_frames):
            self.obs_frames.append(obs.copy())
        observation = np.array(self.obs_frames).copy()
        return observation

    def step(self, action: np.ndarray):
        assert len(self.obs_frames) == self._n_frames
        obs, reward, done, info = self.env.step(action)
        self.timestep += 1
        self.obs_frames.append(obs.copy())

        observation = np.array(self.obs_frames).copy()

        if self.timestep >= self.max_len:
            done = True

        return observation, reward, done, info


class SensorBasedStackedMultiTaskMetaWorld(gym.Env):
    def __init__(
        self,
        max_episode_length: int,
        domain_name_list: List,
        n_frames: int,
        seed: int,
    ):
        """
        # NOTE: Observation space is [stacked_observation @ one_hot vector]
        # NOTE: Thus, dim = obs_dim + n_domains. This is different from [stacked, obs_dim],
        # NOTE: and the reason is to avoid the multiple one hot vector.
        :param max_episode_length:
        :param domain_name_list:
        :param n_frames:
        :param seed:
        """
        assert len(domain_name_list) > 0, "At least one domain required"

        # Sort by first letters. This makes consistent one-hot task vector regardless selection of subset of domains
        domain_name_list = sorted(domain_name_list, key=lambda x: x[0])
        self.max_episode_length = max_episode_length
        self.domain_name_list = domain_name_list
        self.n_domains = len(domain_name_list)
        self.n_frames = n_frames

        # Import here to avoid dependency matter
        from metaworld import MT50
        self.metaworld_env = MT50(seed=seed)
        self.__env = None            # type: MT50
        self.__timestep = None       # type: int
        self.__domain_idx = 0        # Window-open, Peg-insert, ...
        self.__task_idx = 0          # Various init-goal positions

        # Sample dummy environment to define the observation-action space
        env_cls = self.metaworld_env.train_classes[domain_name_list[0]]
        env = env_cls()
        task = random.choice([task for task in self.metaworld_env.train_tasks if task.env_name == domain_name_list[0]])
        env.set_task(task)

        # Save the original observation space before the wrapping.
        self.obs_dim_without_onehot = env.observation_space.shape[0]

        env = RewardMDPSensorObservationStackWrapper(env, n_frames=n_frames)

        low = env.observation_space.low.ravel()
        high = env.observation_space.high.ravel()
        one_hot_task = np.zeros(len(self.domain_name_list))

        self.observation_space = gym.spaces.Box(
            low=np.hstack((low, one_hot_task)),
            high=np.hstack((high, one_hot_task)),    # NOTE ?????? hstack??????
        )
        self.action_space = gym.spaces.Box(
            low=env.action_space.low,
            high=env.action_space.high,
        )

        self.__domain_name2idx = {}
        self.__domain_idx2name = {}
        for domain_idx, domain_name in enumerate(domain_name_list):
            self.__domain_idx2name[str(domain_idx)] = domain_name
            self.__domain_name2idx[domain_name] = domain_idx
        # Frozendict
        self.__domain_name2idx = MappingProxyType(self.domain_name2idx)
        self.__domain_idx2name = MappingProxyType(self.domain_idx2name)

    @property
    def env(self):
        return self.__env

    @property
    def timestep(self):
        return self.__timestep

    @property
    def domain_idx(self):
        return self.__domain_idx

    @property
    def domain_name(self):
        return self.__domain_idx2name[str(self.domain_idx)]

    @property
    def domain_name2idx(self):
        return self.__domain_name2idx

    @property
    def domain_idx2name(self):
        return self.__domain_idx2name

    @property
    def current_domain_tasks(self):
        tasks = [task for task in self.metaworld_env.train_tasks if task.env_name == self.domain_name]
        return tasks

    @property
    def task_idx(self):
        return self.__task_idx

    @property
    def task_one_hot(self):
        """
        Term: task one hot == domain one hot
        """
        one_hot = np.zeros(len(self.domain_name_list))
        one_hot[self.domain_idx] = 1
        return one_hot

    @env.setter
    def env(self, value):
        self.__env = value

    @timestep.setter
    def timestep(self, value):
        self.__timestep = value

    @domain_idx.setter
    def domain_idx(self, value):
        self.__domain_idx = value

    @task_idx.setter
    def task_idx(self, value):
        self.__task_idx = value

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)
        self.timestep = self.timestep + 1

        observation = np.hstack((observation.ravel(), self.task_one_hot))
        additional_info = [
            ("domain_name", self.domain_name),
            ("task_idx", self.task_idx),
            ("is_success", info["success"])
        ]
        info.update(additional_info)

        if self.timestep > self.max_episode_length or info["success"] == 1:
            done = True

        return observation, reward, done, info

    def reset(self, **kwargs):
        # Always change the domain and task
        self.domain_idx = (self.domain_idx + 1) % self.n_domains
        self.task_idx = (self.task_idx + 1) % 50

        # Set domain
        env_cls = self.metaworld_env.train_classes[self.domain_name]
        env = env_cls()
        self.timestep = 0

        # Set task
        tasks = self.current_domain_tasks
        task = tasks[self.task_idx]
        env.set_task(task)

        # Wrap to stack the frame
        self.env = RewardMDPSensorObservationStackWrapper(env, n_frames=self.n_frames)

        observation = self.env.reset()
        observation = np.hstack((observation.ravel(), self.task_one_hot))

        return observation

    def render(self, mode="human"):
        pass


@dataclasses.dataclass
class DomainLog:
    def __init__(self, name:str, idx: int):
        self.name = name
        self.idx = idx
        self.episodic_returns = deque(maxlen=10)
        self.episodic_successes = deque(maxlen=10)

    def get_episodic_returns_mean(self):
        return np.mean(self.episodic_returns) if len(self.episodic_returns) != 0 else 0.0

    def get_episodic_success_ratio(self):
        return np.mean(self.episodic_successes) if len(self.episodic_successes) != 0 else 0.0

    def add_episodic_info(self, returns: float, success: Optional[float]):
        self.episodic_returns.append(returns)
        self.episodic_successes.append(success)


class MultiTaskLogHelper:
    def __init__(self, domain_name2idx: Dict[str, int]):
        self.domain_name2idx = domain_name2idx
        for domain_name, domain_idx in domain_name2idx.items():
            setattr(self, f"__{domain_name}", DomainLog(domain_name, domain_idx))
        self.domain_names = tuple(name for name in self.domain_name2idx.keys())

    def __getitem__(self, domain_name) -> DomainLog:
        return getattr(self, f"__{domain_name}")

    @property
    def current_domain_informations(self):
        for domain_name in self.domain_name2idx.keys():
            yield (
                domain_name,
                self[domain_name].get_episodic_returns_mean(),
                self[domain_name].get_episodic_success_ratio()
            )


class MultiItemGetter:
    __slots__ = ("_items", "_call")

    def __init__(self, item, *items):
        if not items:
            self._items = items

            def func(obj):
                return obj[item]
            self._call = func

        else:
            self._items = items = (item,) + items

            def func(obj):
                return tuple(obj[i] for i in items)
            self._call = func

    def __call__(self, obj):
        return self._call(obj)

    def __reduce__(self):
        return self.__class__, self._items


def evaluate_hrl_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    dones = False

    last_z = None

    while (episode_counts < episode_count_targets).any():
        actions, last_z = model.predict(
            observations,
            cur_timestep=current_lengths,
            last_z=last_z
        )
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_hrl_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)