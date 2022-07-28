from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
import jax
import random

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from cfg.sampling_policy_cfg import sampling_policy, default_static_num_data, default_dynamic_num_data
# from models import PGTE, TE
from offline_data.offline_data_collector import OfflineDatasets
from jax_models import TaskEncoderAE, Model, PolicyTaskEncoder, BehaviorDecoder, PolicyEncoderAE
from jax_models import TaskEncoder, PolicyTaskEncoder, Model, TaskEncoderAE, PolicyTaskEncoderAE, PolicyEncoder, \
    RewardDecoder, TransitionDecoder, BehaviorDecoder
import gym

seed = 777
MT10_TASK = ['drawer-close-v2', 'reach-v2', 'window-close-v2', 'window-open-v2', 'button-press-topdown-v2',
             'door-open-v2', 'drawer-open-v2', 'pick-place-v2', 'peg-insert-side-v2', 'push-v2']
color=['olive', 'mediumpurple', 'grey', 'rosybrown', 'maroon', 'hotpink', 'teal', 'steelblue', 'gold', 'yellowgreen']
onoff = [True, True, True, True, True, True, True, True, True, True]
from sklearn.decomposition import PCA

if __name__ == '__main__':

    latent_dim = 4

    seed = 7777
    mode = 'replay_0'
    num_data = 'dynamic'
    # task_embeddings_model = 'PGTE'
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
        _offline_datasets = _offline_datasets.sample(10)
        offline_datasets.extend(_offline_datasets)

    _offline_datasets = OfflineDatasets()
    _offline_datasets.load('../single_task/offline_data/{}/replay_100.pkl'.format(MT10_TASK[3]))
    _offline_datasets = _offline_datasets.sample(10)
    offline_datasets.extend(_offline_datasets)

    state_size = 39
    action_size = 4
    n_steps = 4

    _, _, _, _, _, sum_rewards, prev_states, prev_actions = offline_datasets.export_dataset(
        mode='trajectories', n_steps=n_steps)

    latents = np.zeros((latent_dim,))
    key = jax.random.PRNGKey(seed)

    sam_prev_states = np.zeros((1, state_size * (n_steps * 2)))
    sam_prev_actions = np.zeros((1, action_size * (n_steps * 2)))

    behavior_encoder_path = os.path.join('../results_jax/models/PGTE',
                               'policy_encoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))

    policy_encoder_def = PolicyEncoderAE(net_arch=[256, 256, 256], latent_dim=latent_dim)
    skill_encoder = Model.create(policy_encoder_def, inputs=[key, sam_prev_states, sam_prev_actions])

    prev_states = np.reshape(prev_states, (prev_states.shape[0], -1))
    prev_actions = np.reshape(prev_actions, (prev_actions.shape[0], -1))

    print(prev_states.shape, prev_actions.shape, sum_rewards.shape)

    original_skill_embeddings = skill_encoder(prev_states, prev_actions)
    print(original_skill_embeddings.shape)

    behavior_data = OfflineDatasets()
    behavior_data.load('{}_behavior_data_{}_seed_{}_{}.jax'.format("PGTE", mode, seed, num_data))
    behavior_data = behavior_data.sample(10)
    _, _, _, _, _, _, prev_states, prev_actions = behavior_data.export_dataset(
        mode='trajectories', n_steps=n_steps)
    prev_states = np.reshape(prev_states, (prev_states.shape[0], -1))
    prev_actions = np.reshape(prev_actions, (prev_actions.shape[0], -1))
    print(prev_states.shape, prev_actions.shape)
    PGTE_skill_embeddings = skill_encoder(prev_states, prev_actions)

    behavior_data = OfflineDatasets()
    behavior_data.load('{}_behavior_data_{}_seed_{}_{}.jax'.format("TE", mode, seed, num_data))
    behavior_data = behavior_data.sample(10)
    _, _, _, _, _, _, prev_states, prev_actions = behavior_data.export_dataset(
        mode='trajectories', n_steps=n_steps)
    prev_states = np.reshape(prev_states, (prev_states.shape[0], -1))
    prev_actions = np.reshape(prev_actions, (prev_actions.shape[0], -1))
    print(prev_states.shape, prev_actions.shape)
    TE_skill_embeddings = skill_encoder(prev_states, prev_actions)

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=777, n_iter=5000, perplexity=15.0)
    skill_embeddings = np.concatenate([original_skill_embeddings, PGTE_skill_embeddings, TE_skill_embeddings])
    embedded = tsne.fit_transform(skill_embeddings)

    print(sum_rewards.mean())
    skill_idx = np.where(sum_rewards > 0.25)

    plt.scatter(embedded[20000:22000, 0], embedded[20000:22000, 1])
    plt.savefig('behavior_embedding.png')
    plt.cla()

    plt.scatter(embedded[22000:24000, 0], embedded[22000:24000, 1])
    plt.savefig('PGTE_behavior_embedding.png')
    plt.cla()

    plt.scatter(embedded[24000:26000, 0], embedded[24000:26000, 1])
    plt.savefig('TE_behavior_embedding.png')