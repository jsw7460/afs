import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import random
import jax
import jax.numpy as jnp
import flax.linen as nn
from offline_baselines_jax import MTCQL, MTSAC, TD3, CQL, SAC
from offline_baselines_jax.td3.policies import MultiInputPolicy
TD3Policy = MultiInputPolicy
from offline_baselines_jax.sac.policies import MultiInputPolicy
SACPolicy = MultiInputPolicy

from envs.env_dict import SkillBasedMetaWrapper, TimeLimitRewardMDP, MT10_TASK, TrajectoriesEncoderEnv, MT50_TASK
from jax_models import TaskEncoder, PolicyTaskEncoder, Model, TaskEncoderAE, PolicyTaskEncoderAE, PolicyEncoder, \
    RewardDecoder, TransitionDecoder, BehaviorDecoder
from collections import deque
from offline_data.offline_data_collector import OfflineDatasets
from offline_baselines_jax.common.evaluation import evaluate_policy
from envs.meta_world import MetaWorldIndexedMultiTaskTester
import functools
from typing import Any, Dict, Callable, Tuple
from offline_baselines_jax.common.type_aliases import Params, TrainFreq, TrainFrequencyUnit
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

latent_dim = 4

seed = 1234
mode = 'replay_25'
num_data = 'dynamic'
task_embeddings_model = 'PGTE'
np.random.seed(seed)
random.seed(seed)

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

offline_datasets = OfflineDatasets()

for idx, t_n in enumerate(MT10_TASK):
    _offline_datasets = OfflineDatasets()
    _offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(t_n, policy_quality[idx]))
    _offline_datasets = _offline_datasets.sample(num_data_dict[policy_quality[idx]])
    offline_datasets.extend(_offline_datasets)

train_env = MetaWorldIndexedMultiTaskTester(mode='dict', seed=seed)
state_size = train_env.observation_space['obs'].shape[0]
action_size = train_env.action_space.shape[0]
n_steps = 4

key = jax.random.PRNGKey(seed)

latents = np.zeros((latent_dim,))
trajectories = np.zeros((n_steps * (state_size + action_size + 1),))
states = np.zeros((state_size,))
actions = np.zeros((action_size,))
seq = np.zeros((8,))

if task_embeddings_model == "PGTE":
    model_path = os.path.join('../results_jax/models/PGTE', 'policy_task_encoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))
    behavior_path = os.path.join('../results_jax/models/PGTE', 'behavior_decoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))
    behavior_decoder_def = BehaviorDecoder(net_arch=[256, 256, 256, offline_datasets.action_size])
    behavior_decoder = Model.create(behavior_decoder_def, inputs=[key, states, latents, seq])
    behavior_decoder = behavior_decoder.load(behavior_path)

elif task_embeddings_model == 'TE':
    model_path = os.path.join('../results_jax/models/TE', 'task_encoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))

policy_task_encoder_def = TaskEncoderAE(net_arch=[256, 256], latent_dim=latent_dim)
task_encoder = Model.create(policy_task_encoder_def, inputs=[key, trajectories])
task_encoder = task_encoder.load(model_path)

print("load_replay_buffer")
key, rng = jax.random.split(key, 2)
replay_buffer = offline_datasets.get_trajectories_replay_buffer(rng, 5_000_000, None, latent_dim, idx, 10,
                                                                replay_buffer=replay_buffer, n_steps=4, AE=True,
                                                                obs_space=train_env.observation_space['obs'])


mean, std = replay_buffer.normalize_states()

train_env = TrajectoriesEncoderEnv(TimeLimitRewardMDP(MetaWorldIndexedMultiTaskTester(mode='dict', task_name_list=MT10_TASK)), task_encoder, latent_dim, seed=seed, AE=True, mean=mean, std=std)
test_env = TrajectoriesEncoderEnv(TimeLimitRewardMDP(MetaWorldIndexedMultiTaskTester(mode='dict', task_name_list=MT50_TASK)), task_encoder, latent_dim, n_steps=n_steps, seed=seed, AE=True, mean=mean, std=std)

model = TD3.load(os.path.join(os.path.join('../results_jax', 'models/{}_seed_{}_{}_'.format('{}{}'.format(task_embeddings_model, 'TD3'), seed, num_data) + mode + '_da'), 'model_1000000_steps.zip'), train_env)

epi_r, epi_suc, task_r, task_suc = evaluate_policy(model, test_env, n_eval_episodes=2500, return_episode_rewards=True)

for task_id in task_r.keys():
    print(task_id, np.mean(task_r[task_id]), np.mean(task_suc[task_id]))

# history = test_env.get_histories()
# history.save('./{}_behavior_data_{}_seed_{}_{}.jax'.format(task_embeddings_model, mode, seed, num_data))