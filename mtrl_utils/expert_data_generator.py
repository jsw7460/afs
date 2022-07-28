from metaworld import MT50
from envs.env_dict import TimeLimitMDP, MT50_TASK, MT10_TASK, MT10_EXPERT

import numpy as np
import random
import tqdm
import argparse
import cv2
import os

seed = 0
np.random.seed(seed)
random.seed(seed)


def GetExpertData(task_name: str, episodes: int, path: str):
    mt = MT50(seed=seed)
    print('Get expert data for {}'.format(task_name))
    # make environment
    env_cls = mt.train_classes[task_name]
    env = env_cls()
    task = random.choice([task for task in mt.train_tasks if task.env_name == task_name])
    env.set_task(task)
    fourcc =  cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(task_name + '.avi', fourcc , 20, (640, 480))
    # set meta-world environment to goal MDP
    test_env = TimeLimitMDP(env)

    success = 0
    timesteps = 0
    progress_bar = tqdm.tqdm(range(episodes))
    policy = MT10_EXPERT[task_name]()
    for epochs in progress_bar:
        obs = test_env.reset()
        img = test_env.render()
        video.write(img)
        done = False
        tt = 0
        # success_check = False
        while not done:
            tt += 1
            action = policy.get_action(obs) + np.random.normal(size=4) * 0.1
            next_obs, reward, done, info = test_env.step(action)
            timesteps += 1
            obs = next_obs
            img = test_env.render()

            video.write(img)
            if info['success'] == 1:
                success += 1
                done = True

        progress_bar.set_description("Success rate %.3f%% Timesteps %d"%(success / (epochs+1) * 100, timesteps))

    save_path = os.path.join(path, task_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # expert_replay_buffer.save(os.path.join(save_path, 'expert_replay.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default="mt10")
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--path', type=str, default='policy')
    args = parser.parse_args()

    if args.task_name is None:
        print("Input task name for generating expert data")
        print(MT50_TASK)
    elif args.task_name == 'mt10':
        for task_name in MT10_TASK:
            GetExpertData(task_name=task_name, episodes=args.episodes, path=args.path)
    else:
        GetExpertData(task_name=args.task_name, episodes=args.episodes, path=args.path)