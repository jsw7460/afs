from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import pickle as pkl
import torch.utils.data as data
import numpy as np
from torchvision import transforms

from offline_data.offline_data_collector import OfflineDatasets


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = torch.from_numpy(sample).float()
        return sample


class ConcatDataset(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

    @property
    def size(self):
        return self.datasets[0].size

    @property
    def state_size(self):
        return self.datasets[0].state_size

    @property
    def action_size(self):
        return self.datasets[0].action_size


class Transitions(data.Dataset):
    def __init__(self, dataset: OfflineDatasets, mode='transition'):
        super().__init__()
        self.states, self.actions, self.rewards, self.next_states = dataset.export_dataset()
        self.mode = mode
        self.dataset = np.concatenate([self.states, self.actions, self.rewards, self.next_states], axis=1)
        self.size = self.state_size * 2 + self.action_size + 1

    def __getitem__(self, index):
        if self.mode == 'transition':
            data = np.concatenate([self.states[index], self.actions[index], self.rewards[index], self.next_states[index]])
            return self.transform(data)
        else:
            return self.transform(self.states[index]), self.transform(self.actions[index]), \
                   self.transform(self.rewards[index]), self.transform(self.next_states[index])

    def __len__(self):
        return len(self.states)

    def transform(self, sample):
        composed_transforms = transforms.Compose([ToTensor()])
        return composed_transforms(sample)

    @property
    def state_size(self):
        return self.states[0].shape[-1]

    @property
    def action_size(self):
        return self.actions[0].shape[-1]


class Trajectories(data.Dataset):
    def __init__(self, datasets, n_steps: int, jax: bool = True):
        super().__init__()
        t, s, a, r, n, sr, tk, ps, pa= [], [], [], [], [], [], [], [], []
        for i, dataset in enumerate(datasets):
            trajs, states, actions, rewards, next_states, sum_rewards, prev_states, prev_actions= dataset.export_dataset(mode='trajectories', n_steps=n_steps)
            t.append(trajs)
            s.append(states)
            a.append(actions)
            r.append(rewards)
            n.append(next_states)
            ps.append(prev_states)
            pa.append(prev_actions)
            sr.append(sum_rewards)
            task = np.zeros((trajs.shape[0], len(datasets)))
            task[:, i] = 1
            tk.append(task)

        self.states = np.concatenate(s, axis=0)
        self.trajs = np.concatenate(t, axis=0)
        self.actions = np.concatenate(a, axis=0)
        self.rewards = np.concatenate(r, axis=0)
        self.sum_rewards = np.concatenate(sr, axis=0)
        self.next_states = np.concatenate(n, axis=0)
        self.tasks = np.concatenate(tk, axis=0)
        self.prev_states = np.concatenate(ps, axis=0)
        self.prev_actions = np.concatenate(pa, axis=0)

        self.state_size = self.states[0].shape[-1]
        self.action_size = self.actions[0].shape[-1]
        self.size = self.trajs[0].shape[0]
        self.jax = jax


    def __getitem__(self, index):
        return self.transform(self.trajs[index]), self.transform(self.states[index]), self.transform(self.actions[index]),\
               self.transform(self.rewards[index]), self.transform(self.next_states[index]), self.transform(self.sum_rewards[index]),\
               self.transform(self.tasks[index]), self.transform(self.prev_states[index]), self.transform(self.prev_actions[index])

    def __len__(self):
        return self.trajs.shape[0]

    def transform(self, sample):
        if self.jax:
            return sample
        else:
            composed_transforms = transforms.Compose([ToTensor()])
            return composed_transforms(sample)


class PairedTransitions(data.Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = []
        self.generated_dataset = []

    def __getitem__(self, index):
        return self.transform(self.dataset[index]), self.transform(self.generated_dataset[index])

    @property
    def size(self):
        return self.dataset[0].shape[0]

    def __len__(self):
        return len(self.dataset)

    def add(self, transitions, generated_transitions):
        if len(transitions.shape) == 1:
            self.dataset.append(np.array(transitions))
            self.generated_dataset.append(np.array(generated_transitions))
        elif len(transitions.shape) == 2:
            for idx in range(transitions.shape[0]):
                self.dataset.append(np.array(transitions[idx]))
                self.generated_dataset.append(np.array(generated_transitions[idx]))

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump({'dataset': self.dataset, 'generated_dataset': self.generated_dataset}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)

        self.dataset = data['dataset']
        self.generated_dataset = data['generated_dataset']

    def transform(self, sample):
        composed_transforms = transforms.Compose([ToTensor()])
        return composed_transforms(sample)

    def extend(self, dataset):
        self.dataset.extend(dataset.dataset)
        self.generated_dataset.extend(dataset.generated_dataset)