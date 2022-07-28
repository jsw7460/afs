import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from stable_baselines3 import TEMTSAC
from stable_baselines3.common.callbacks import CheckpointCallback
from meta_world_env.meta_world import MetaWorldIndexedMultiTaskTester
from envs.env_dict import TimeLimitRewardMDP, TimeLimitGoalMDP

import argparse
import torch.nn as nn

seed = 777

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default=None)
    parser.add_argument('--save-freq', type=int, default=500_000)
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--path', type=str, default='data')
    args = parser.parse_args()


    if args.task_name == 'online':
        checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path='./{}/{}'.format(args.path, 'TEMTSAC_' + args.task_name),
                                                 name_prefix='model')

        train_env = TimeLimitRewardMDP(MetaWorldIndexedMultiTaskTester(mode='dict'))
        test_env = TimeLimitGoalMDP(MetaWorldIndexedMultiTaskTester(mode='dict'))
        # Generate RL Model
        policy_kwargs = dict(net_arch=[400, 400, 400], activation_fn=nn.ReLU)

        model = TEMTSAC('MultiTaskInputPolicy', train_env, seed=seed, verbose=1, batch_size=1280, buffer_size=args.buffer_size, train_freq=1,
                   policy_kwargs=policy_kwargs, task_embedding_dim=4, tensorboard_log='./{}/{}'.format(args.path,  'TEMTSAC_' + args.task_name),)

        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, eval_freq=args.save_freq, n_eval_episodes=100,
                    eval_log_path='./{}/{}'.format(args.path,  'TEMTSAC_' + args.task_name), eval_env=test_env, tb_log_name='exp')

    else:
        NotImplementedError()
