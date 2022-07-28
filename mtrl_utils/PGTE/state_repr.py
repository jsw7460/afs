from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
import jax
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

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
    task_name_list = 'window-close-v2'
    mode = 'medium_replay'
    np.random.seed(seed)
    offline_datasets = OfflineDatasets()
    offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(task_name_list, mode))

    states_4 = np.array(offline_datasets.sample_states())
    mode = 'offline_data_random'
    offline_datasets = OfflineDatasets()
    offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(task_name_list, mode))

    states_5 = np.array(offline_datasets.sample_states())


    mode = 'replay_100'

    offline_datasets = OfflineDatasets()
    offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(task_name_list, mode))

    states_1 = np.array(offline_datasets.sample_states())

    mode = 'replay_0'

    offline_datasets = OfflineDatasets()
    offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(task_name_list, mode))
    states_2 = np.array(offline_datasets.sample_states())

    states_noise = np.copy(states_2)
    rand_shape = states_noise.shape
    states_noise[:, :39] += np.random.randn(rand_shape[0], 39) * 0.2

    mode = 'replay_100'

    offline_datasets = OfflineDatasets()
    offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(task_name_list, mode))
    # states_2 = np.array(offline_datasets.sample_states())

    latent_dim = 16


    mode = 'replay_0'
    num_data = 'dynamic'

    state_size = 39
    action_size = 4
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

    replay_buffer, states_3 = offline_datasets.get_trajectories_replay_buffer(key, 1000000, task_encoder, latent_dim, 0, 10,
                                                                replay_buffer=None, n_steps=n_steps, AE=True,
                                                                obs_space=gym.spaces.Box(shape=(39, ), low=0, high=1),
                                                                data_augmentation=True,
                                                                transition_decoder=transition_decoder,
                                                                reward_decoder=reward_decoder,
                                                                behavior_decoder=behavior_decoder)

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=777, n_iter=1000, perplexity=15.0)
    data = np.concatenate([states_1, states_noise, states_2, states_3, states_4, states_5], axis=0)
    print(data.shape)
    embedded = tsne.fit_transform(data)
    minaxis = np.min(embedded, axis=0)
    maxaxis = np.max(embedded, axis=0)

    x_min = minaxis[0]
    y_min = minaxis[1]
    x_max = maxaxis[0]
    y_max = maxaxis[1]

    x_bin = (x_max - x_min) / 30 + 1e-4
    y_bin = (y_max - y_min) / 30 + 1e-4

    print(x_bin, y_bin)

    h_1 = np.zeros((30, 30))
    h_2 = np.zeros((30, 30))
    h_3 = np.zeros((30, 30))
    h_4 = np.zeros((30, 30))

    for i in range(20000):
        h_1[int((embedded[i][0] - x_min) // x_bin)][int((embedded[i][1] - y_min) // y_bin )] += 1

    for i in range(20000, 40000):
        h_2[int((embedded[i][0] - x_min) // x_bin)][int((embedded[i][1] - y_min) // y_bin )] += 1

    for i in range(40000, 60000):
        h_3[int((embedded[i][0] - x_min) // x_bin)][int((embedded[i][1] - y_min) // y_bin )] += 1

    for i in range(60000, 79900):
        h_4[int((embedded[i][0] - x_min) // x_bin)][int((embedded[i][1] - y_min) // y_bin )] += 1

    plt.pcolor(h_1, cmap='plasma')
    plt.savefig('state_repr_0.png')

    plt.cla()
    plt.pcolor(h_2, cmap='plasma')
    plt.savefig('state_repr_1.png')

    plt.cla()
    plt.pcolor(h_3, cmap='plasma')
    plt.savefig('state_repr_2.png')

    plt.cla()
    plt.pcolor(h_4, cmap='plasma')
    plt.savefig('state_repr_3.png')

    plt.scatter(embedded[:20000, 0],
                embedded[:20000, 1],
                c=color[8], marker='o', s=9)
    plt.savefig('state_dot_0.png')

    plt.cla()
    plt.scatter(embedded[20000:40000, 0],
                embedded[20000:40000, 1],
                c=color[8],  marker='o', s=9)
    plt.savefig('state_dot_1.png')

    plt.cla()
    plt.scatter(embedded[40000:59900, 0],
                embedded[40000:59900, 1],
                c=color[8], marker='o', s=9)
    plt.savefig('state_dot_2.png')

    plt.cla()