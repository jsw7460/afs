import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from offline_baselines_jax import MTCQL, MTSAC, TD3, PCGrad

from offline_baselines_jax.td3.policies import MultiInputPolicy
TD3Policy = MultiInputPolicy
from offline_baselines_jax.sac.policies import MultiInputPolicy
SACPolicy = MultiInputPolicy
from offline_baselines_jax.soft_modularization.policies import SoftModuleTD3Policy
SFPolicy = SoftModuleTD3Policy
from envs.env_dict import TimeLimitGoalMDP, TimeLimitRewardMDP, MT10_TASK, CDS_TASK, NormalizeEnv
from offline_baselines_jax.common.buffers import ModelBasedDictReplayBuffer
from envs.meta_world import MetaWorldIndexedMultiTaskTester
from offline_data.offline_data_collector import OfflineDatasets
from stable_baselines3.common.evaluation import evaluate_policy
from PGTE.jax_models import TransitionDecoder, RewardDecoder, Model, Params, InfoDict

from offline_baselines_jax.common.callbacks import CheckpointCallback

from collections import deque
import numpy as np
import random
import argparse
import jax

import flax.linen as nn
from cfg.sampling_policy_cfg import sampling_policy, default_static_num_data, default_dynamic_num_data

seed = 777
np.random.seed(seed)
random.seed(seed)

def GenerateOfflineMultiTaskPolicy(task_name:str, timesteps: int, buffer_size:int, mode:str, num_data_dict, policy_quality, num_data, algos:str,
                                   data_augmentation:bool, reward_path:str = None, transition_path:str = None):
    if task_name == 'Multitask':
        task_name_list = MT10_TASK
    elif task_name == 'CDS':
        task_name_list = CDS_TASK
    else:
        task_name_list = [task_name]

    if algos == 'MTCQL':
        model_algo = MTCQL
        policy = SACPolicy
    elif algos == 'TD3':
        model_algo = TD3
        policy = TD3Policy
    elif algos == 'SoftModule':
        model_algo = TD3
        policy = SFPolicy
    elif algos == 'PCGrad':
        model_algo = PCGrad
        policy = TD3Policy
    else:
        NotImplementedError()

    offline_datasets = OfflineDatasets()
    offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(task_name_list[0], policy_quality[0]))

    if data_augmentation:
        states = np.zeros((offline_datasets.state_size,))
        actions = np.zeros((offline_datasets.action_size,))
        latents = np.zeros((10, ))
        reward_decoder_def = RewardDecoder(net_arch=[256, 256, 1])
        transition_decoder_def = TransitionDecoder(net_arch=[256, 256, offline_datasets.state_size])

        key = jax.random.PRNGKey(0)
        reward_decoder = Model.create(reward_decoder_def, inputs=[key, states, actions, latents])
        transition_decoder = Model.create(transition_decoder_def, inputs=[key, states, actions, latents])

        reward_decoder = reward_decoder.load(reward_path)
        transition_decoder = transition_decoder.load(transition_path)

    replay_buffer = None

    for idx, t_n in enumerate(task_name_list):
        offline_datasets = OfflineDatasets()
        offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(t_n, policy_quality[idx]))
        offline_datasets = offline_datasets.sample(num_data_dict[policy_quality[idx]])
        replay_buffer = offline_datasets.get_task_ID_replay_buffer(5000000, idx, 1, replay_buffer, data_augmentation=data_augmentation)

    if data_augmentation:
        replay_buffer.set_data_point()
        replay_buffer.get_model(reward_model=reward_decoder, transition_model=transition_decoder)

    mean, std = replay_buffer.normalize_states()
    train_env = NormalizeEnv(TimeLimitRewardMDP(MetaWorldIndexedMultiTaskTester(mode='dict', task_name_list=task_name_list)), mean=mean, std=std)
    test_env = NormalizeEnv(TimeLimitRewardMDP(MetaWorldIndexedMultiTaskTester(mode='dict', task_name_list=task_name_list)), mean=mean, std=std)

    cql_parameter = dict(conservative_weight=1.,  alpha_coef=1.)
    min_parameter = dict(alpha=2.5)

    if algos == 'MTCQL' :
        kwargs = cql_parameter
    elif algos == 'TD3' or algos == 'PCGrad' or algos=='SoftModule':
        kwargs = min_parameter

    # Generate RL Model
    if algos == 'SoftModule':
        policy_kwargs = dict(net_arch=[64], activation_fn=nn.relu)
    elif algos=='PCGrad':
        policy_kwargs = dict(net_arch=[160, 160, 160, 160, 160, 160], activation_fn=nn.relu)
    else:
        policy_kwargs = dict(net_arch=[256, 256, 256], activation_fn=nn.relu)

    # if data_augmentation:
    #     replay_buffer = ModelBasedDictReplayBuffer(buffer_size)

    checkpoint_callback = CheckpointCallback(save_freq=100000,
                                             save_path='../results_jax/models/TID{}_seed_{}_{}_'.format(algos, seed, num_data) + mode,
                                             name_prefix='model')

    model = model_algo(policy, train_env, seed=seed, verbose=1, batch_size=1280, buffer_size=buffer_size, #data_augmentation=data_augmentation,
                  train_freq=1, policy_kwargs=policy_kwargs, without_exploration=True,  learning_rate=1e-4, gradient_steps=1000,
                  tensorboard_log='../results_jax/tensorboard/TID{}_seed_{}_{}_'.format(algos, seed, num_data) + mode, **kwargs)

    model.replay_buffer = replay_buffer
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, eval_freq=200000, log_interval=1000,
                n_eval_episodes=500, eval_log_path='../results_jax/models/TID{}_seed_{}_{}_'.format(algos, seed, num_data) + mode, eval_env=test_env, tb_log_name='exp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default='Multitask')
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--mode', type=str, default='replay_25')
    parser.add_argument('--num-data', type=str, default='static')
    parser.add_argument('--data-augmentation', action='store_true')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--algos', type=str, default='TD3')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    policy_quality = sampling_policy[args.mode]
    np.random.shuffle(policy_quality)
    if args.num_data == 'static':
        num_data = default_static_num_data
    elif args.num_data == 'dynamic':
        num_data = default_dynamic_num_data
    else:
        num_data = None

    if args.data_augmentation:
        path = '../results_jax/models/MB'
        reward_path = os.path.join(path, 'reward_decoder_MB_{}_seed_{}_{}.jax'.format(args.mode, args.seed, args.num_data))
        transition_path = os.path.join(path, 'transition_decoder_MB_{}_seed_{}_{}.jax'.format(args.mode, args.seed, args.num_data))
    else:
        reward_path = None
        transition_path = None

    GenerateOfflineMultiTaskPolicy(task_name=args.task_name, timesteps=args.timesteps, buffer_size=args.buffer_size, algos=args.algos,
                                   mode=args.mode, num_data_dict=num_data, policy_quality=policy_quality, num_data=args.num_data,
                                   data_augmentation=args.data_augmentation, reward_path=reward_path, transition_path=transition_path)