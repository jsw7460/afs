import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from stable_baselines3 import MTSAC, MTCQL
from envs.env_dict import TimeLimitGoalMDP, TimeLimitRewardMDP, MT10_TASK, CDS_TASK, TaskLabelsEncoderEnv
from envs.meta_world import MetaWorldIndexedMultiTaskTester
from stable_baselines3.common.callbacks import CheckpointCallback
from offline_data.offline_data_collector import OfflineDatasets
from stable_baselines3.common.evaluation import evaluate_policy

from collections import deque

from models import TaskIDTE
import numpy as np
import random
import torch
import argparse
import torch.nn as nn

seed = 777
np.random.seed(seed)
random.seed(seed)

def GenerateMultiTaskPolicy(task_name:str, timesteps: int, path: str, buffer_size:int, save_freq:int=100000):
    if task_name == 'Multitask':
        task_name_list = MT10_TASK
    elif task_name == 'CDS':
        task_name_list = CDS_TASK
    else:
        return

    train_env = MetaWorldIndexedMultiTaskTester(mode='dict')
    task_encoder = TaskIDTE(num_task=len(task_name_list), state_space=train_env.observation_space['obs'].shape[0],
                      action_space=train_env.action_space.shape[0], latent_dim=8, hidden_dim=256).eval()

    task_encoder.load_state_dict(torch.load('te_encoder/te_model.pth'))

    train_env = TaskLabelsEncoderEnv(TimeLimitRewardMDP(train_env), task_encoder.encoder, 8)
    test_env = TaskLabelsEncoderEnv(TimeLimitGoalMDP(MetaWorldIndexedMultiTaskTester(mode='dict')), task_encoder.encoder, 8)

    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./{}/{}'.format(path, 'TEMTSAC_' + task_name),
                                             name_prefix='model')

    # Generate RL Model
    policy_kwargs = dict(net_arch=[400, 400, 400], activation_fn=nn.ReLU)

    model = MTSAC('MultiInputPolicy', train_env, seed=seed, verbose=1, batch_size=1280, buffer_size=buffer_size,
                  train_freq=1, policy_kwargs=policy_kwargs, learning_rate=1e-4,
                  tensorboard_log='./{}/{}'.format(path, 'TEMTSAC_' + task_name), )

    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, eval_freq=save_freq,
                n_eval_episodes=100, eval_log_path='./{}/{}'.format(path, 'TEMTSAC_' + task_name),
                eval_env=test_env, tb_log_name='exp')

def GenerateOfflineMultiTaskPolicy(task_name:str, timesteps: int, path: str, buffer_size:int, mode:str):
    if task_name == 'Multitask':
        task_name_list = MT10_TASK
    elif task_name == 'CDS':
        task_name_list = CDS_TASK
    else:
        return

    train_env = MetaWorldIndexedMultiTaskTester(mode='dict')
    task_encoder = TaskIDTE(num_task=len(task_name_list), state_space=train_env.observation_space['obs'].shape[0],
                      action_space=train_env.action_space.shape[0], latent_dim=8, hidden_dim=256).eval()

    task_encoder.load_state_dict(torch.load('../results/models/TaskIDTE/model_{}.pth'.format(mode)))

    train_env = TaskLabelsEncoderEnv(TimeLimitRewardMDP(train_env), task_encoder.encoder, 8)
    test_env = TaskLabelsEncoderEnv(TimeLimitGoalMDP(MetaWorldIndexedMultiTaskTester(mode='dict')), task_encoder.encoder, 8)

    replay_buffer = None
    for idx, t_n in enumerate(task_name_list):
        offline_datasets = OfflineDatasets()
        offline_datasets.load('../offline_data/offline_data/{}/{}.pkl'.format(t_n, mode))
        offline_datasets = offline_datasets.sample(250)
        replay_buffer = offline_datasets.get_task_label_replay_buffer(500000, task_encoder.encoder, 8, idx, 10, replay_buffer)

    # Generate RL Model
    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)

    model = MTCQL('MultiInputPolicy', train_env, seed=seed, verbose=1, batch_size=1280, buffer_size=buffer_size,
                  train_freq=1, policy_kwargs=policy_kwargs, without_exploration=True,  learning_rate=1e-4,
                  tensorboard_log=os.path.join(path, 'tensorboard/{}'.format('TIDTEMTCQL_' + mode)))

    model.reload_buffer = False
    model.replay_buffer = replay_buffer
    evaluation_model = MTCQL('MultiInputPolicy', test_env, seed=seed, policy_kwargs=policy_kwargs,)

    task_rewards = {}
    task_rewards_std = {}

    for task_name in task_name_list:
        task_rewards[task_name] = deque(maxlen=10)
        task_rewards_std[task_name] = deque(maxlen=10)

    model.task_rewards = task_rewards
    model.task_rewards_std = task_rewards_std

    for i in range(100):
        model.learn(total_timesteps=timesteps // 100, tb_log_name='exp', reset_num_timesteps=False, )

        evaluation_model.set_parameters(model.get_parameters())
        epi_reward = []
        for t_i, t_n in enumerate(task_name_list):
            test_env.env.set_task(t_i)
            episode_reward, epi_len = evaluate_policy(evaluation_model, test_env, n_eval_episodes=10,
                                                      return_episode_rewards=True, )
            model.task_rewards[t_n].append(np.mean(episode_reward))
            model.task_rewards_std[t_n].append(np.std(episode_reward))
            epi_reward.extend(episode_reward)

        model.offline_rewards.append(np.mean(epi_reward))
        model.offline_rewards_std.append(np.std(epi_reward))
        model._dump_logs()

    if not os.path.isdir(os.path.join(path, 'models/{}'.format('TIDTEMTCQL_' + mode))):
        os.mkdir(os.path.join(path, 'models/{}'.format('TIDTEMTCQL_' + mode)))

    model.save(os.path.join(path, 'models/{}/model.zip'.format('TIDTEMTCQL_' + mode)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default='Multitask')
    parser.add_argument('--buffer-size', type=int, default=500_000)
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--path', type=str, default='../results')
    parser.add_argument('--mode', type=str, default='replay_50')
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()

    if args.online:
        GenerateMultiTaskPolicy(task_name=args.task_name, timesteps=args.timesteps, path=args.path,
                                buffer_size=args.buffer_size)
    else:
        GenerateOfflineMultiTaskPolicy(task_name=args.task_name, timesteps=args.timesteps, path=args.path,
                                buffer_size=args.buffer_size, mode=args.mode)