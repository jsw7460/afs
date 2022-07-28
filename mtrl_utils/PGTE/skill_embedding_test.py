import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import random
import jax
import jax.numpy as jnp

from envs.env_dict import SkillBasedMetaWrapper, TimeLimitRewardMDP, MT10_TASK
from jax_models import TaskEncoder, PolicyTaskEncoder, Model, TaskEncoderAE, PolicyTaskEncoderAE, PolicyEncoder, \
    RewardDecoder, TransitionDecoder, BehaviorDecoder
from collections import deque
from offline_data.offline_data_collector import OfflineDatasets
from envs.meta_world import MetaWorldIndexedMultiTaskTester
import functools
from typing import Any, Dict, Callable, Tuple
from offline_baselines_jax.common.type_aliases import Params
from cfg.sampling_policy_cfg import sampling_policy, default_static_num_data, default_dynamic_num_data


@functools.partial(jax.jit, static_argnames=('task_embedding_fn'))
def inference_task_embeddings_AE(task_embedding_fn: Callable[..., Any], encoder_params: Params,
                    traj: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    task_embeddings = task_embedding_fn({'params': encoder_params}, traj)
    return task_embeddings

@jax.jit
def normal_sampling(key:Any, task_latents_mu:jnp.ndarray, task_latents_log_std:jnp.ndarray):
    rng, key = jax.random.split(key)
    return task_latents_mu + jax.random.normal(key, shape=(task_latents_log_std.shape[-1], )) * jnp.exp(0.5 * task_latents_log_std)

latent_dim = 16

seed = 777
mode = 'replay_25'
num_data = 'dynamic'
np.random.seed(seed)
random.seed(seed)

policy_quality = sampling_policy[mode]
np.random.shuffle(policy_quality)
print(policy_quality)

for idx, task_name in enumerate(MT10_TASK):
    path = '../single_task/offline_data/{}/{}.pkl'.format(task_name, policy_quality[idx])
    # Load Dataloader
    task_replay_buffer = OfflineDatasets()
    task_replay_buffer.load(path)
    task_replay_buffer = task_replay_buffer.sample(default_dynamic_num_data[policy_quality[idx]])
    print(np.mean(task_replay_buffer.get_episodic_rewards()))

train_env = MetaWorldIndexedMultiTaskTester(mode='dict', seed=seed)
state_size = train_env.observation_space['obs'].shape[0]
action_size = train_env.action_space.shape[0]
n_steps = 4

latents = np.zeros((latent_dim,))
key = jax.random.PRNGKey(seed)

trajectories = np.zeros((n_steps * (state_size + action_size + 1),))
states = np.zeros((state_size,))
actions = np.zeros((action_size,))
seq = np.zeros((8,))

model_path = os.path.join('../results_jax/models/PGTE',
                          'policy_task_encoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))
reward_path = os.path.join('../results_jax/models/PGTE',
                           'reward_decoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))
transition_path = os.path.join('../results_jax/models/PGTE',
                               'transition_decoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))
behavior_path = os.path.join('../results_jax/models/PGTE',
                             'behavior_decoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))

policy_task_encoder_def = TaskEncoderAE(net_arch=[256, 256], latent_dim=latent_dim)
task_encoder = Model.create(policy_task_encoder_def, inputs=[key, trajectories])

reward_decoder_def = RewardDecoder(net_arch=[256, 256, 1])
transition_decoder_def = TransitionDecoder(net_arch=[256, 256, state_size])

reward_decoder = Model.create(reward_decoder_def, inputs=[key, states, actions, latents])
transition_decoder = Model.create(transition_decoder_def, inputs=[key, states, actions, latents])

task_encoder = task_encoder.load(model_path)
reward_decoder = reward_decoder.load(reward_path)
transition_decoder = transition_decoder.load(transition_path)

behavior_decoder_def = BehaviorDecoder(net_arch=[256, 256, action_size])
behavior_decoder = Model.create(behavior_decoder_def, inputs=[key, states, latents, seq])
behavior_decoder = behavior_decoder.load(behavior_path)

train_env = SkillBasedMetaWrapper(train_env, skill_decoder=behavior_decoder, latent_space=4)
eval_env = SkillBasedMetaWrapper(TimeLimitRewardMDP(MetaWorldIndexedMultiTaskTester(mode='dict', seed=seed)), skill_decoder=behavior_decoder, latent_space=4)

epi_reward = np.zeros(10)

for t in range(500):
    sum_reward = 0
    state = eval_env.reset()
    state_history = list()
    action_history = list()
    reward_history = list()
    state_history.append(state['obs'])
    skills = np.zeros(latent_dim)
    done = False
    while not done:
        state, reward, done, info = eval_env.step(skills)
        sum_reward += reward
        action = info['action']
        action_history.append(action)
        state_history.append(state['obs'])
        reward_history.append(reward)

        traj = []
        for i in range(n_steps):
            transitions = np.zeros((state_history[0].shape[0] + action.shape[0] + 1, ))
            idx = len(action_history) - n_steps + i
            if idx >= 0:
                transitions[:action.shape[0]] = action_history[idx]
                transitions[action.shape[0]: action.shape[0] + state_history[0].shape[0]] = state_history[idx]
                transitions[action.shape[0] + state_history[0].shape[0]] = reward_history[idx]
            traj.append(transitions)

        traj = np.concatenate(traj)
        skills = inference_task_embeddings_AE(task_encoder.apply_fn, task_encoder.params, traj)
        # key, task_key = jax.random.split(key, 2)
        # noise = jnp.clip(normal_sampling(task_key, jnp.zeros_like(skills), jnp.ones_like(skills) * -4.6),a_min=-0.5, a_max=0.5)
        # skills += noise

    epi_reward[t % 10] += sum_reward
    # print(sum_reward)

print(epi_reward / 50)
