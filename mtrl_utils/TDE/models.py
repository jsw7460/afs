import torch
import torch.nn as nn
import gym
from torch.nn.utils import spectral_norm

class CycleGAN(nn.Module):
    def __init__(self, transition_size, latent_dim, hidden_dim):
        super().__init__()
        self.generator_1 = Generator(transition_size, latent_dim, hidden_dim)
        self.generator_2 = Generator(transition_size, latent_dim, hidden_dim)
        self.discriminator_1 = Discriminator(transition_size, hidden_dim)
        self.discriminator_2 = Discriminator(transition_size, hidden_dim)

    def forward(self, transitions_1, transitions_2):
        generated_transitions_2 = self.generator_1(transitions_1)
        generated_transitions_1 = self.generator_2(transitions_2)

        cycle_transitions_1 = self.generator_2(generated_transitions_2)
        cycle_transitions_2 = self.generator_1(generated_transitions_1)

        discriminated_result_1 = self.discriminator_1(generated_transitions_1)
        discriminated_result_2 = self.discriminator_2(generated_transitions_2)

        cycle_consistency_loss = nn.MSELoss()(transitions_1, cycle_transitions_1) + nn.MSELoss()(transitions_2, cycle_transitions_2)

        # fake data is 0 / true data is 1
        labels = torch.ones([transitions_1.shape[0], 1])
        gan_loss = nn.L1Loss()(discriminated_result_1, labels) + nn.L1Loss()(discriminated_result_2, labels)

        ret = dict(cycle_consistency_loss=cycle_consistency_loss,
                   gan_loss=gan_loss / 2,
                   generated_transitions_1=generated_transitions_1,
                   generated_transitions_2=generated_transitions_2,
                   cycle_transitions_1=cycle_transitions_1,
                   cycle_transitions_2=cycle_transitions_2,
                   discriminated_result_1=discriminated_result_1,
                   discriminated_result_2=discriminated_result_2,)

        return ret

    def inference_discriminator(self, transitions_1, transitions_2):
        generated_transitions_2 = self.generator_1(transitions_1)
        generated_transitions_1 = self.generator_2(transitions_2)

        fake_result_1 = self.discriminator_1(generated_transitions_1)
        fake_result_2 = self.discriminator_2(generated_transitions_2)
        true_result_1 = self.discriminator_1(transitions_1)
        true_result_2 = self.discriminator_2(transitions_2)

        # fake data is 0 / true data is 1
        true_labels = torch.ones([transitions_1.shape[0], 1])
        fake_labels = torch.zeros([transitions_1.shape[0], 1])

        discriminator_loss = nn.MSELoss()(fake_result_1, fake_labels) + nn.MSELoss()(fake_result_2, fake_labels) + \
                             nn.MSELoss()(true_result_1, true_labels) + nn.MSELoss()(true_result_2, true_labels)

        ret = dict(discriminator_loss=discriminator_loss / 4,
                   fake_result_1=fake_result_1,
                   fake_result_2=fake_result_2,
                   true_result_1=true_result_1,
                   true_result_2=true_result_2,
                   generated_transitions_1=generated_transitions_1,
                   generated_transitions_2=generated_transitions_2)

        return ret

    def generate_transitions(self, transitions_1, transitions_2):
        generated_transitions_2 = self.generator_1(transitions_1)
        generated_transitions_1 = self.generator_2(transitions_2)

        ret = dict(generated_transitions_1=generated_transitions_1,
                   generated_transitions_2=generated_transitions_2)

        return ret


class Generator(nn.Module):
    def __init__(self, transition_size, latent_dim, hidden_dim):
        super().__init__()

        model = [spectral_norm(nn.Linear(transition_size, hidden_dim)),
                 nn.BatchNorm1d(hidden_dim),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, latent_dim)),
                 spectral_norm(nn.Linear(latent_dim, hidden_dim)),
                 nn.BatchNorm1d(hidden_dim),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, transition_size))]

        self.model = nn.Sequential(*model)

    def forward(self, transitions):
        return self.model(transitions)


class Discriminator(nn.Module):
    def __init__(self, transition_size, hidden_dim):
        super().__init__()

        model = [spectral_norm(nn.Linear(transition_size, hidden_dim)),
                 nn.BatchNorm1d(hidden_dim),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                 nn.BatchNorm1d(hidden_dim),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, 1))]

        self.model = nn.Sequential(*model)

    def forward(self, transitions):
        return self.model(transitions)

class TDE(nn.Module):
    def __init__(self, transition_size, latent_dim, hidden_dim):
        super().__init__()
        self.encoder = TaskDecompositionEncoder(transition_size, latent_dim, hidden_dim)
        self.decoder = Decoder(transition_size, latent_dim, hidden_dim)

    def forward(self, transitions_1, transitions_2):
        time_invariant_latent_1, time_variant_latent_1 = self.encoder(transitions_1)
        time_invariant_latent_2, time_variant_latent_2 = self.encoder(transitions_2)

        reconstructed_transitions_1 = self.decoder(time_invariant_latent_1, time_variant_latent_1)
        reconstructed_transitions_2 = self.decoder(time_invariant_latent_2, time_variant_latent_2)
        time_invariant_transitions_1 = self.decoder(time_invariant_latent_1)
        time_invariant_transitions_2 = self.decoder(time_invariant_latent_2)

        reconstruction_loss = nn.MSELoss()(time_invariant_transitions_1, transitions_1) \
                              + nn.MSELoss()(time_invariant_transitions_2, transitions_2)

        ae_loss = nn.MSELoss()(reconstructed_transitions_1, transitions_1) \
                  + nn.MSELoss()(reconstructed_transitions_2, transitions_2)

        similarity_loss = nn.MSELoss()(time_invariant_transitions_1, time_invariant_transitions_2)

        ret = dict(reconstruction_loss=reconstruction_loss,
                   ae_loss=ae_loss,
                   similarity_loss=similarity_loss,
                   time_invariant_latent_1=time_invariant_latent_1,
                   time_variant_latent_1=time_variant_latent_1,
                   time_invariant_latent_2=time_invariant_latent_2,
                   time_variant_latent_2=time_variant_latent_2,
                   reconstructed_transitions_1=reconstructed_transitions_1,
                   reconstructed_transitions_2=reconstructed_transitions_2,
                   time_invariant_transitions_1=time_invariant_transitions_1,
                   time_invariant_transitions_2=time_invariant_transitions_2)

        return ret


class TaskDecompositionEncoder(nn.Module):
    def __init__(self, transition_size, latent_dim, hidden_dim):
        super().__init__()

        model = [spectral_norm(nn.Linear(transition_size, hidden_dim)),
                 nn.BatchNorm1d(hidden_dim),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                 nn.BatchNorm1d(hidden_dim),
                 nn.ReLU()]

        self.model = nn.Sequential(*model)
        self.time_invariant_layer = spectral_norm(nn.Linear(hidden_dim, latent_dim))
        self.time_variant_layer = spectral_norm(nn.Linear(hidden_dim, latent_dim))

    def forward(self, transitions):
        x = self.model(transitions)
        time_variant_latent = self.time_variant_layer(x)
        time_invariant_latent = self.time_invariant_layer(x)
        return time_invariant_latent, time_variant_latent


class Decoder(nn.Module):
    def __init__(self, transition_size, latent_dim, hidden_dim):
        super().__init__()

        model = [spectral_norm(nn.Linear(latent_dim * 2, hidden_dim)),
                 nn.BatchNorm1d(hidden_dim),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                 nn.BatchNorm1d(hidden_dim),
                 nn.ReLU(),
                 spectral_norm(nn.Linear(hidden_dim, transition_size))]

        self.model = nn.Sequential(*model)

    def forward(self, time_invariant_latent, time_variant_latent=None):
        if time_variant_latent is None:
            time_variant_latent = torch.zeros(time_invariant_latent.shape)

        input_latent = torch.cat((time_invariant_latent, time_variant_latent), 1)

        return self.model(input_latent)