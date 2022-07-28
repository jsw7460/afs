import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from stable_baselines3 import MTSAC
from envs.env_dict import TimeLimitGoalMDP, TimeLimitRewardMDP, MT10_TASK, CDS_TASK, TaskEncoderEnv
from meta_world_env.meta_world import MetaWorldIndexedMultiTaskTester
from stable_baselines3.common.callbacks import CheckpointCallback

from models import TDE
import numpy as np
import random
import torch
import argparse
import torch.nn as nn

seed = 777
np.random.seed(seed)
random.seed(seed)

def GenerateMultiTaskPolicy(task_name:str, timesteps: int, path: str, buffer_size:int, save_freq:int):
    if task_name == 'Multitask':
        task_name_list = MT10_TASK
    elif task_name == 'CDS':
        task_name_list = CDS_TASK
    else:
        return

    train_env = MetaWorldIndexedMultiTaskTester(mode='dict')
    transition_size = train_env.observation_space['obs'].shape[0] * 2 + train_env.action_space.shape[0] + 1
    task_encoder = TDE(transition_size=transition_size, latent_dim=16, hidden_dim=256).eval()

    task_encoder.load_state_dict(torch.load('tde_encoder/tde_model.pth'))

    train_env = TaskEncoderEnv(TimeLimitRewardMDP(train_env), task_encoder.encoder, 16)
    test_env = TaskEncoderEnv(TimeLimitGoalMDP(MetaWorldIndexedMultiTaskTester(mode='dict')), task_encoder.encoder, 16)

    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./{}/{}'.format(path, 'TDEMTSAC_' + task_name),
                                             name_prefix='model')

    # Generate RL Model
    policy_kwargs = dict(net_arch=[400, 400, 400], activation_fn=nn.ReLU)

    model = MTSAC('MultiInputPolicy', train_env, seed=seed, verbose=1, batch_size=1280, buffer_size=buffer_size,
                  train_freq=1, policy_kwargs=policy_kwargs, learning_rate=1e-4,
                  tensorboard_log='./{}/{}'.format(path, 'TDEMTSAC_' + task_name), )

    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, eval_freq=save_freq,
                n_eval_episodes=100, eval_log_path='./{}/{}'.format(path, 'TDEMTSAC_' + task_name),
                eval_env=test_env, tb_log_name='exp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default='Multitask')
    parser.add_argument('--save-freq', type=int, default=100_000)
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--timesteps', type=int, default=2_000_000)
    parser.add_argument('--path', type=str, default='tde_encoder')
    args = parser.parse_args()

    GenerateMultiTaskPolicy(task_name=args.task_name, timesteps=args.timesteps, path=args.path,
                            buffer_size=args.buffer_size, save_freq=args.save_freq)

