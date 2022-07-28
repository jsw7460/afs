import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from offline_baselines_jax import MTCQL, MTSAC, TD3, PCGrad, SoftModularization

from offline_baselines_jax.td3.policies import MultiInputPolicy
TD3Policy = MultiInputPolicy
from offline_baselines_jax.sac.policies import MultiInputPolicy
SACPolicy = MultiInputPolicy
from offline_baselines_jax.soft_modularization.policies import SoftModulePolicy
SFPolicy = SoftModulePolicy
from envs.env_dict import TimeLimitGoalMDP, TimeLimitRewardMDP, MT10_TASK, CDS_TASK, MT50_TASK
from envs.meta_world import MetaWorldIndexedMultiTaskTester
from offline_data.offline_data_collector import OfflineDatasets
from stable_baselines3.common.evaluation import evaluate_policy

from offline_baselines_jax.common.callbacks import CheckpointCallback

from collections import deque
import numpy as np
import random
import argparse

import flax.linen as nn
from cfg.sampling_policy_cfg import sampling_policy, default_static_num_data, default_dynamic_num_data

seed = 777
np.random.seed(seed)
random.seed(seed)


def GenerateOnlineMultiTaskPolicy(timesteps: int, buffer_size:int, algos:str):
    if algos == 'MTSAC':
        model_algo = MTSAC
        policy = SACPolicy
    elif algos == 'TD3':
        model_algo = TD3
        policy = TD3Policy
    elif algos == 'SoftModule':
        model_algo = SoftModularization
        policy = SFPolicy
    elif algos == 'PCGrad':
        model_algo = PCGrad
        policy = TD3Policy
    else:
        NotImplementedError()

    seed = 777
    mode = 'mixed_medium'
    num_data = 'dynamic'

    policy_quality = sampling_policy[mode]
    np.random.shuffle(policy_quality)
    print(policy_quality)

    if num_data == 'static':
        num_data_dict = default_static_num_data
    elif num_data == 'dynamic':
        num_data_dict = default_dynamic_num_data
    else:
        num_data_dict = None

    replay_buffer = None
    for idx, t_n in enumerate(MT10_TASK):
        offline_datasets = OfflineDatasets()
        offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(t_n, policy_quality[idx]))
        offline_datasets = offline_datasets.sample(num_data_dict[policy_quality[idx]])
        replay_buffer = offline_datasets.get_task_ID_replay_buffer(5000000, idx, 10, replay_buffer)

    train_env = TimeLimitRewardMDP(MetaWorldIndexedMultiTaskTester(mode='dict', task_name_list=MT10_TASK))
    test_env = TimeLimitRewardMDP(MetaWorldIndexedMultiTaskTester(mode='dict', task_name_list=MT10_TASK))


    min_parameter = dict(alpha=2.5)

    if algos == 'TD3' or algos == 'PCGrad':
        kwargs = min_parameter
    else:
        kwargs = dict()

    # Generate RL Model
    if algos == 'SoftModule':
        policy_kwargs = dict(net_arch=[64], activation_fn=nn.relu)
    else:
        policy_kwargs = dict(net_arch=[256, 256, 256], activation_fn=nn.relu)

    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='../results_jax/models/TID{}_seed_{}'.format(algos, seed),
                                             name_prefix='model')

    model = model_algo(policy, train_env, seed=seed, verbose=1, batch_size=1280, buffer_size=buffer_size,
                  train_freq=2000, policy_kwargs=policy_kwargs, learning_rate=3e-4, gradient_steps=200,
                  tensorboard_log='../results_jax/tensorboard/TID{}_seed_{}'.format(algos, seed), **kwargs)

    model.replay_buffer = replay_buffer

    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, eval_freq=100000, log_interval=100,
                n_eval_episodes=100, eval_log_path='../results_jax/models/TID{}_seed_{}'.format(algos, seed), eval_env=test_env, tb_log_name='exp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default='Multitask')
    parser.add_argument('--buffer-size', type=int, default=5_000_000)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--algos', type=str, default='MTSAC')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    GenerateOnlineMultiTaskPolicy(timesteps=args.timesteps, buffer_size=args.buffer_size, algos=args.algos)