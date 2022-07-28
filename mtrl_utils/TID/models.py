import torch
import torch.nn as nn
import gym
from torch.nn.utils import spectral_norm

class TaskIDTE(nn.Module):
    def __init__(self, num_task, state_space, action_space, latent_dim, hidden_dim):
        super().__init__()
        self.encoder = TaskEncoder(num_task, latent_dim, hidden_dim)
        self.decoder = TaskDecoder(state_space, action_space, latent_dim, hidden_dim)

    def forward(self, task_label, states, actions, rewards, next_states):
        latent = self.encoder(task_label)
        latent_state_action = torch.cat((latent, states, actions), 1)

        pred_next_states, pred_rewards = self.decoder(latent_state_action)

        transition_loss = nn.MSELoss()(pred_next_states, next_states)
        reward_loss = nn.MSELoss()(pred_rewards, rewards)
        out = dict(
            transition_loss=transition_loss,
            reward_loss=reward_loss,
            pred_next_states=pred_next_states,
            pred_rewards=pred_rewards,
            latent=latent
        )
        return out

class TaskEncoder(nn.Module):
    def __init__(self, num_task, latent_dim, hidden_dim):
        super().__init__()

        model = [spectral_norm(nn.Linear(num_task, hidden_dim)),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, latent_dim))]

        self.model = nn.Sequential(*model)

    def forward(self, task_label):
        return self.model(task_label)

class TaskDecoder(nn.Module):
    def __init__(self, state_space, action_space, latent_dim, hidden_dim):
        super().__init__()

        model = [spectral_norm(nn.Linear(latent_dim + state_space + action_space, hidden_dim)),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                 nn.ReLU()]

        self.model = nn.Sequential(*model)
        self.next_state_layer = spectral_norm(nn.Linear(hidden_dim, state_space))
        self.reward_layer = spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, latent_state_action):
        x = self.model(latent_state_action)
        next_state = self.next_state_layer(x)
        reward = self.reward_layer(x)
        return next_state, reward