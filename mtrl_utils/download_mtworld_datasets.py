import pickle
import random

import numpy as np
import tqdm

from envs.env_dict import TimeLimitMDP, MT10_TASK, MT10_EXPERT
from metaworld import MT50

MIN_LEN = 50
SEED = 0
NOISE = 0.5

if __name__ == "__main__":
    datasets = []
    mt = MT50(seed=SEED)
    for domain_name in MT10_TASK:
        # Set environments
        env_cls = mt.train_classes[domain_name]
        env = env_cls()

        progress_bar = tqdm.tqdm(range(999999999))      # Break when the num of saved transitions > 1M.

        # Set rule based expert policy
        policy = MT10_EXPERT[domain_name]()

        # Prepare task dataset
        domain_dataset = []
        n_transitions = 0

        tasks = [task for task in mt.train_tasks if task.env_name == domain_name]

        for epochs in progress_bar:

            task_idx = random.choice(range(len(tasks)))
            task = tasks[task_idx]

            env.set_task(task)
            test_env = TimeLimitMDP(env)

            observation = test_env.reset()
            done = False

            # Prepare dataset for one episode
            observations = []
            next_observations = []
            actions = []
            rewards = []
            terminals = []
            infos = []

            ep_len = 0
            while not done:
                action = policy.get_action(observation) + np.random.normal(size=4) * NOISE
                # action = policy.get_action(observation)
                action = np.clip(action, -1.0, 1.0)
                next_observation, reward, done, info = test_env.step(action)
                zz = test_env.render(mode="rgb_array")
                print(zz.shape)
                exit()

                observations.append(observation)
                next_observations.append(next_observation)
                actions.append(action)
                rewards.append(reward)
                terminals.append(done)

                observation = next_observation

                info.update(
                    {
                        "domain_name": domain_name,
                        "task_idx": task_idx
                    }
                )
                if info["success"] == 1:
                    done = True

                ep_len += 1

            # Avoid short length episode to stack the transitions properly later
            if ep_len < MIN_LEN:
                continue

            # Save Each episode by a dictionary type
            np_observations = np.array(observations)
            np_next_observations = np.array(next_observations)
            np_actions = np.array(actions)
            np_rewards = np.array(rewards)
            np_terminals = np.array(terminals)

            one_episodes = {
                "observations": np_observations,
                "next_observations": np_next_observations,
                "actions": np_actions,
                "rewards": np_rewards,
                "terminals": np_terminals
            }

            domain_dataset.append(one_episodes)
            n_transitions += ep_len

            if n_transitions > 1_000_000:
                break

        returns = np.array([np.sum(p['rewards']) for p in domain_dataset])
        num_samples = np.sum([p['rewards'].shape[0] for p in domain_dataset])
        print(f'Number of samples collected: {num_samples}')
        print(
            f'Trajectory returns: mean = {np.mean(returns)},'
            f' std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}'
        )
        # Save current task dataset
        with open(f"{domain_name}-noise{NOISE}-seed{SEED}.pkl", "wb") as f:
            pickle.dump(domain_dataset, f)
