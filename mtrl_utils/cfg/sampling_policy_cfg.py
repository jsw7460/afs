sampling_policy = {
    'low_expert': ['offline_data_random'] * 7 + ['replay_50'] * 3,
    'very_low': ['offline_data_random'] * 8 + ['replay_25'] * 2,
    'low': ['offline_data_random'] * 7 + ['replay_25'] * 3,
    'medium': ['offline_data_random'] * 5 + ['replay_25'] * 5,
    'high': ['offline_data_random'] * 3 + ['replay_25'] * 7,
    'low_replay': ['offline_data_random'] * 7 + ['replay_0'] * 3,
    'medium_replay': ['offline_data_random'] * 5 + ['replay_0'] * 5,
    'high_replay': ['offline_data_random'] * 3 + ['replay_0'] * 7,

    'motiv_medium': ['replay_50'] * 5 + ['replay_0'] * 5,
    'motiv_low': ['replay_50'] * 3 + ['replay_0'] * 7,
    'motiv_high': ['replay_50'] * 7 + ['replay_0'] * 3,
    'mixed_low': ['replay_50'] * 2 + ['replay_25'] * 3 + ['replay_0'] * 5,
    'mixed_medium': ['replay_50' ] * 3 + ['replay_25'] * 3 + ['replay_0'] * 4,
    'mixed_high': ['replay_50'] * 4 + ['replay_25'] * 3 + ['replay_0'] * 3,

    'expert_low': ['replay_50'] * 3 + ['replay_25'] * 7,
    'expert_medium': ['replay_50'] * 5 + ['replay_25'] * 5,
    'zero_replay_0': ['replay_0'] * 8,
    'zero_replay_25': ['replay_25'] * 8,
    'zero_replay_50': ['replay_25'] * 8,
    'airsim_medium_expert': ['medium_expert'] * 6,
    'airsim_medium_replay': ['medium_replay'] * 6,
    'airsim_replay': ['replay'] * 6,
}

default_static_num_data = dict()
sampling_policy['random'] = ['offline_data_random'] * 10
for i in range(21):
    sampling_policy['replay_{}'.format(i * 5)] = ['replay_{}'.format(i * 5)] * 10
    default_static_num_data['replay_{}'.format(i * 5)] = 100
    default_static_num_data['offline_data_random'] = 100

default_dynamic_num_data = {'offline_data_random': 200, 'replay_0': 150, 'replay_25': 100, 'replay_50': 50, 'medium_expert': 40, 'medium_replay': 120, 'replay': 80}