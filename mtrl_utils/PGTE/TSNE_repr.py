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


MT10_TASK = ['drawer-close-v2', 'reach-v2', 'window-close-v2', 'window-open-v2', 'button-press-topdown-v2',
             'door-open-v2', 'drawer-open-v2', 'pick-place-v2', 'peg-insert-side-v2', 'push-v2']
color=['olive', 'mediumpurple', 'grey', 'rosybrown', 'maroon', 'hotpink', 'teal', 'steelblue', 'gold', 'yellowgreen']
onoff = [True, True, True, True, True, True, True, True, True, True]

if __name__ == '__main__':
    task_name_list = MT10_TASK
    mode = 'replay_25'
    task_embeddings_model = "PGTE"
    num_data = 'dynamic'
    seed = 777

    offline_datasets = OfflineDatasets()
    offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(task_name_list[0], "replay_100"))

    dataset_list = []
    for idx, t_n in enumerate(task_name_list):
        offline_datasets = OfflineDatasets()
        offline_datasets.load('../single_task/offline_data/{}/{}.pkl'.format(t_n, mode))
        dataset_list.append(offline_datasets)

    key = jax.random.PRNGKey(seed)
    key, task_encoder_key = jax.random.split(key, 2)

    task_encoder_def = TaskEncoderAE(net_arch=[256, 256], latent_dim=4)
    # ps = np.zeros((39 * 3, ))
    # pa = np.zeros((4 * 3))
    if task_embeddings_model == "PGTE":
        model_path = os.path.join('../results_jax/models/PGTE',
                                  'policy_task_encoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))
    elif task_embeddings_model == 'TE':
        model_path = os.path.join('../results_jax/models/TE',
                                  'task_encoder_{}_seed_{}_{}.jax'.format(mode, seed, num_data))

    trajectories = np.zeros((176, ))
    task_encoder = Model.create(task_encoder_def, inputs=[task_encoder_key, trajectories])
    task_encoder = task_encoder.load(model_path)

    embeddings_list = []
    task_len_list = np.zeros(len(task_name_list) + 1)
    for i in range(len(dataset_list)):
        traj, embeddings = dataset_list[i].sample_traj_embeddings(task_encoder, mode='te')
        embeddings_list.append(embeddings)
        task_len_list[i + 1] = task_len_list[i] + len(traj)

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=777, n_iter=1000,)
    data = np.concatenate(embeddings_list, axis=0)
    print(data.shape)
    embedded = tsne.fit_transform(data)

    print(task_len_list)

    for i in range(10):
        if onoff[i]:
            plt.scatter(embedded[int(task_len_list[i]):int(task_len_list[i+1]), 0],
                     embedded[int(task_len_list[i]):int(task_len_list[i+1]), 1],
                     c=color[i], label=task_name_list[i], marker='o', s=36)
    # plt.legend()
    plt.savefig('{}_{}_{}_TSNE.png'.format(task_embeddings_model, mode, seed))