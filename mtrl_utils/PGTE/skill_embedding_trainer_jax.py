import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import jax
import jax.numpy as jnp
import functools
import optax
import numpy as np
import time
import flax.linen as nn

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from typing import Tuple, Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from jax_models import PolicyEncoderAE, TaskEncoderAE, TransitionDecoder, RewardDecoder, BehaviorDecoder, Model, Params, InfoDict, SumRewardDecoder, PolicyEncoder, SimpleLinearModel, TaskEncoderPrior

from utils.dataloader import ConcatDataset
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -10

@jax.jit
def normal_sampling(key:Any, task_latents_mu:jnp.ndarray, task_latents_log_std:jnp.ndarray):
    return task_latents_mu + jax.random.normal(key, shape=task_latents_log_std.shape) * jnp.exp(0.5 * task_latents_log_std)

def l2_loss(x):
    return (x ** 2).mean()

def compute_mmd(z: jnp.ndarray, z1:jnp.ndarray=None, reg_weight: float=100) -> jnp.ndarray:
    # Sample from prior (Gaussian) distribution
    key = jax.random.PRNGKey(0)
    batch_size = z.shape[0]
    reg_weight = reg_weight / (batch_size * (batch_size - 1))
    if z1 is None:
        prior_z =jax.random.normal(key, shape=z.shape)
    else:
        prior_z = z1

    prior_z__kernel = compute_inv_mult_quad(prior_z, prior_z)
    z__kernel = compute_inv_mult_quad(z, z)
    priorz_z__kernel = compute_inv_mult_quad(prior_z, z)

    mmd = reg_weight * prior_z__kernel.mean() + \
          reg_weight * z__kernel.mean() - \
          2 * reg_weight * priorz_z__kernel.mean()
    return mmd

def compute_inv_mult_quad(x1: jnp.ndarray, x2: jnp.ndarray, eps: float = 1e-7, latent_var: float = 2.) -> jnp.ndarray:
    D, N = x1.shape

    x1 = jnp.expand_dims(x1, axis=-2)  # Make it into a column tensor
    x2 = jnp.expand_dims(x2, axis=-3)  # Make it into a row tensor

    x1 = jnp.repeat(x1, D, axis=-2)
    x2 = jnp.repeat(x2, D, axis=-3)

    z_dim = x2.shape[-1]
    C = 2 * z_dim * latent_var
    kernel = C / (eps + C + jnp.sum((x1 - x2)**2, axis=-1))

    # Exclude diagonal elements
    result = jnp.sum(kernel) - jnp.sum(jnp.diag(kernel))

    return result


def policy_task_encoder_update(key:Any, task_encoder: Model, policy_encoder:Model, behavior_decoder: Model,  states: jnp.ndarray,
                               prev_states: jnp.ndarray, prev_actions: jnp.ndarray, seq: jnp.ndarray, task_id:jnp.ndarray):

    key, task_key, policy_key = jax.random.split(key, 3)
    def task_encoder_loss(task_encoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        task_latent = task_encoder.apply_fn({'params': task_encoder_params}, states, task_id)
        policy_latent = policy_encoder(jnp.reshape(prev_states, (prev_states.shape[0], -1)), jnp.reshape(prev_actions, (prev_actions.shape[0], -1)))

        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(task_encoder_params))

        policy_embedding_loss = jnp.sum(jnp.clip(jnp.square(task_latent - jax.lax.stop_gradient(policy_latent)) - 0.05, a_min=0), axis=-1).mean()
        loss = policy_embedding_loss + 1e-3 * l2_reg
        return loss, {'policy_embedding_loss': policy_embedding_loss, 'task_latent': task_latent}

    def policy_encoder_loss(policy_encoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # task_latent = task_encoder(trajectories)
        policy_latent = policy_encoder.apply_fn({'params': policy_encoder_params}, jnp.reshape(prev_states, (prev_states.shape[0], -1)), jnp.reshape(prev_actions, (prev_actions.shape[0], -1)))
        pred_actions = behavior_decoder(jnp.reshape(prev_states, (-1, prev_states.shape[-1])), jnp.repeat(policy_latent, repeats=8, axis=0), seq)

        reconstruction_loss = jnp.mean(jnp.sum(jnp.square(pred_actions - jnp.reshape(prev_actions, (-1, prev_actions.shape[-1]))), axis=1), axis=0)

        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(policy_encoder_params))

        reg_loss = compute_mmd(policy_latent)

        loss = reconstruction_loss + reg_loss + l2_reg * 1e-3
        return loss, {'policy_reg_loss': reg_loss, 'policy_latent': policy_latent}

    new_task_encoder, task_info = task_encoder.apply_gradient(task_encoder_loss)
    new_policy_encoder, policy_info = policy_encoder.apply_gradient(policy_encoder_loss)
    return new_task_encoder, new_policy_encoder, {**task_info, **policy_info}

def decoder_update(key:Any, policy_latent: jnp.ndarray, behavior_decoder: Model,
                   prev_states: jnp.ndarray, prev_actions: jnp.ndarray, seq: jnp.ndarray):

    def behavior_loss(behavior_decoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_actions = behavior_decoder.apply_fn({'params': behavior_decoder_params}, jnp.reshape(prev_states, (-1, prev_states.shape[-1])), jnp.repeat(policy_latent, repeats=8, axis=0), seq)
        l2_reg = sum(l2_loss(w) for w in jax.tree_leaves(behavior_decoder_params))
        loss = jnp.mean(jnp.sum(jnp.square(pred_actions - jnp.reshape(prev_actions, (-1, prev_actions.shape[-1]))), axis=1), axis=0) + 1e-3 * l2_reg
        return loss, {'behavior_loss': loss, 'pred_actions': pred_actions}

    new_behavior_decoder, behavior_info = behavior_decoder.apply_gradient(behavior_loss)
    return new_behavior_decoder, {**behavior_info}

@functools.partial(jax.jit)
def _update_pgte(rng:Any, task_encoder: Model, policy_encoder:Model, behavior_decoder:Model,
                 states:jnp.ndarray, prev_states:jnp.ndarray, prev_actions:jnp.ndarray, seq:jnp.ndarray,
                 task_id:jnp.ndarray) \
        -> Tuple[Any, Model, Model, Model, InfoDict]:

    rng, task_key, decoder_key, key1, key2  = jax.random.split(rng, 5)

    states = prev_states[:, 0, :]
    print(states.shape)

    new_task_encoder, new_policy_encoder, task_info = policy_task_encoder_update(task_key, task_encoder, policy_encoder,  behavior_decoder,
                                                       states, prev_states, prev_actions, seq, task_id)

    new_behavior_decoder, decoder_info = decoder_update(decoder_key, task_info['policy_latent'], behavior_decoder, prev_states, prev_actions,  seq)

    return rng, new_task_encoder, new_policy_encoder, new_behavior_decoder, {**task_info, **decoder_info}


class SimplTrainer(object):
    def __init__(self, datasets, args):
        self.train_loader = DataLoader(datasets, batch_size=args.num_batch, shuffle=True, )
        self.num_task = len(datasets)
        self.num_batch = args.num_batch

        self.key = jax.random.PRNGKey(args.seed)
        self.mode = args.mode
        state_size = datasets.state_size
        action_size = datasets.action_size
        print(action_size, state_size)

        states = jnp.zeros((state_size, ))
        latents = jnp.zeros((args.latent_dim, ))
        prev_states = jnp.zeros((state_size * (args.n_steps * 2)))
        prev_actions = jnp.zeros((action_size * (args.n_steps * 2)))
        seq = jnp.zeros((args.n_steps * 2, ))
        task_id = jnp.zeros((args.latent_dim, ))

        # Define network
        task_encoder_def = TaskEncoderPrior(net_arch=[256, 256],  latent_dim=args.latent_dim)
        policy_encoder_def = PolicyEncoderAE(net_arch = [256, 256], latent_dim=args.latent_dim)
        behavior_decoder_def = BehaviorDecoder(net_arch=[256, 256, action_size])

        # create model
        self.key, task_encoder_key, policy_encoder_key, reward_decoder_key, transition_decoder_key, behavior_decoder_key, task_to_policy_key = jax.random.split(self.key, 7)
        self.task_encoder = Model.create(task_encoder_def, inputs=[task_encoder_key, states, task_id],
                                         tx=optax.adam(learning_rate=args.lr))
        self.policy_encoder = Model.create(policy_encoder_def, inputs=[policy_encoder_key, prev_states, prev_actions],
                                         tx=optax.adam(learning_rate=args.lr))
        self.behavior_decoder = Model.create(behavior_decoder_def, inputs=[behavior_decoder_key, states, latents, seq],
                                             tx=optax.adam(learning_rate=args.lr))

    def train(self, epoch):
        for i in range(epoch):
            tbar = tqdm(self.train_loader)
            transition_loss, reward_loss, behavior_loss, kld_loss, kld_loss_2, policy_embedding_loss = 0, 0, 0, 0, 0, 0
            for idx, sample in enumerate(tbar):
                _, states, _, _, _, _, task_id, prev_states, prev_actions = sample
                self.key, key = jax.random.split(self.key, 2)

                seq = np.zeros((8, 8))
                for j in range(8):
                    seq[j, j] = 1
                seq = np.tile(seq, (prev_states.shape[0], 1))

                self.key, new_task_encoder, new_policy_encoder, new_behavior_encoder, info = \
                    _update_pgte(key, self.task_encoder, self.policy_encoder, self.behavior_decoder, states.numpy(),
                                prev_states.numpy(), prev_actions.numpy(), seq, task_id.numpy())

                self.task_encoder = new_task_encoder
                self.policy_encoder = new_policy_encoder
                self.behavior_decoder = new_behavior_encoder

                kld_loss += info['policy_reg_loss']
                policy_embedding_loss += info['policy_embedding_loss']
                behavior_loss += info['behavior_loss']

                tbar.set_description('Epochs %d: Trans. loss: %.4f Rew. loss: %.4f REG loss: %.4f TREG loss: %.4f Beh. loss: %.4f Pol.loss %.4f'
                                     %(i, transition_loss / (idx + 1), reward_loss / (idx + 1), kld_loss / (idx + 1), kld_loss_2 / (idx + 1),
                                       behavior_loss / (idx + 1),  policy_embedding_loss / (idx + 1)))

    def save(self, path, seed, num_data):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.task_encoder.save(os.path.join(path, 'policy_task_encoder_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)))
        self.policy_encoder.save(os.path.join(path, 'policy_encoder_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)))
        self.behavior_decoder.save(os.path.join(path, 'behavior_decoder_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)))
        return os.path.join(path, 'policy_task_encoder_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data)), \
               os.path.join(path, 'behavior_decoder_{}_seed_{}_{}.jax'.format(self.mode, seed, num_data))