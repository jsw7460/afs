import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import jax
import jax.numpy as jnp
import numpy as np
import functools
import optax
import argparse
import random

from envs.env_dict import MT10_TASK
from offline_data.offline_data_collector import OfflineDatasets
from utils.dataloader import Trajectories
from cfg.sampling_policy_cfg import sampling_policy, default_static_num_data, default_dynamic_num_data
from typing import Tuple, Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from jax_models import TransitionDecoder, RewardDecoder, Model, Params, InfoDict


def l2_loss(x):
    return (x ** 2).mean()

"""
  actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))
"""

def decoder_update(task_labels: jnp.ndarray, reward_decoder: Model, transition_decoder: Model,
                   states: jnp.ndarray, actions: jnp.ndarray, next_states: jnp.array, rewards: jnp.ndarray):

    def reward_loss(reward_decoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_rewards = reward_decoder.apply_fn({'params': reward_decoder_params}, states, actions, task_labels)
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(reward_decoder_params))
        loss = jnp.mean(jnp.square(pred_rewards - rewards)) + 1e-3 * l2_reg
        return loss, {'reward_loss': loss}

    def transition_loss(transition_decoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_next_states = transition_decoder.apply_fn({'params': transition_decoder_params}, states, actions, task_labels)
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(transition_decoder_params))
        loss = jnp.mean(jnp.square(pred_next_states - next_states)) + 1e-3 * l2_reg
        return loss, {'transition_loss': loss}

    new_reward_decoder, reward_info = reward_decoder.apply_gradient(reward_loss)
    new_transition_decoder, transition_info = transition_decoder.apply_gradient(transition_loss)

    return new_reward_decoder, new_transition_decoder, {**reward_info, **transition_info}

@functools.partial(jax.jit)
def _update_te(rng:Any, reward_decoder: Model, transition_decoder: Model, task_labels:jnp.ndarray,
               states:jnp.ndarray, actions:jnp.ndarray, rewards: jnp.ndarray, next_states:jnp.ndarray) \
        -> Tuple[Any, Model, Model, InfoDict]:

    new_reward_decoder, new_transition_decoder, decoder_info = \
        decoder_update(task_labels, reward_decoder, transition_decoder, states, actions, next_states, rewards)

    return rng,  new_reward_decoder, new_transition_decoder, decoder_info


class MBTrainer(object):
    def __init__(self, datasets, args):
        self.train_loader = DataLoader(datasets, batch_size=args.num_batch, shuffle=True, )
        self.num_task = len(datasets)

        self.key = jax.random.PRNGKey(args.seed)
        self.mode = args.mode

        state_size = datasets.state_size
        action_size = datasets.action_size

        states = jnp.zeros((state_size,))
        actions = jnp.zeros((action_size,))
        latents = jnp.zeros((args.num_task))

        # Define network
        reward_decoder_def = RewardDecoder(net_arch=[256, 256, 1])
        transition_decoder_def = TransitionDecoder(net_arch=[256, 256, state_size])

        # create model
        self.key, task_encoder_key, reward_decoder_key, transition_decoder_key = jax.random.split(self.key, 4)
        self.reward_decoder = Model.create(reward_decoder_def, inputs=[reward_decoder_key, states, actions, latents],
                                           tx=optax.adam(learning_rate=args.lr))
        self.transition_decoder = Model.create(transition_decoder_def, inputs=[transition_decoder_key, states, actions, latents],
                                               tx=optax.adam(learning_rate=args.lr))


    def train(self, epoch):
        for i in range(epoch):
            tbar = tqdm(self.train_loader)
            transition_loss, reward_loss, kld_loss = 0, 0, 0
            for idx, sample in enumerate(tbar):
                _, states, actions, rewards, next_states, _, task_id, _, _ = sample

                self.key, key = jax.random.split(self.key, 2)
                self.key, new_reward_decoder, new_transition_decoder, info = \
                    _update_te(key, self.reward_decoder, self.transition_decoder, task_id.numpy(),
                               states.numpy(), actions.numpy(), rewards.numpy(), next_states.numpy())

                self.reward_decoder = new_reward_decoder
                self.transition_decoder = new_transition_decoder

                transition_loss += info['transition_loss']
                reward_loss += info['reward_loss']
                tbar.set_description('Epochs %d: Trans. loss: %.4f Rew. loss: %.4f '%(i, transition_loss / (idx + 1), reward_loss / (idx + 1)))


    def save(self, path, seed, num_data):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.reward_decoder.save(os.path.join(path, 'reward_decoder_MB_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)))
        self.transition_decoder.save(os.path.join(path, 'transition_decoder_MB_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)))
        return os.path.join(path, 'reward_decoder_MB_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)), \
               os.path.join(path, 'transition_decoder_MB_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment setting')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-batch', type=int, default=2048)
    parser.add_argument('--task-name', type=str, default='Multitask')
    parser.add_argument('--num-task', type=int, default=10)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--mode', type=str, default='replay_25')
    parser.add_argument('--path', type=str, default='../results_jax')
    parser.add_argument('--num-data', type=str, default='static')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    policy_quality = sampling_policy[args.mode]
    np.random.shuffle(policy_quality)
    if args.num_data == 'static':
        num_data = default_static_num_data
    elif args.num_data == 'dynamic':
        num_data = default_dynamic_num_data
    else:
        num_data = None

    datasets = []
    for idx, task_name in enumerate(MT10_TASK):
        path = '../single_task/offline_data/{}/{}.pkl'.format(task_name, policy_quality[idx])
        # Load Dataloader
        task_replay_buffer = OfflineDatasets()
        task_replay_buffer.load(path)
        task_replay_buffer = task_replay_buffer.sample(num_data[policy_quality[idx]])
        datasets.append(task_replay_buffer)

    datasets = Trajectories(datasets, n_steps=1, jax=True)

    trainer = MBTrainer(datasets, args)
    trainer.train(args.epochs)
    reward_path, transition_path = trainer.save('../results_jax/models/MB', seed, args.num_data)