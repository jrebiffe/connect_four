from stable_baselines3.dqn import DQN
from torch.nn import Tanh
import numpy as np

seed = 43

config = {
    'game' : {
        'name': 'connect_four',
        'first_player_rule': lambda : np.random.choice([1,2]), #lambda info['nb']: np.random.choice(range(nb))
        'action': {'n': 7},
        'observation':{'height': 7, 'width':7},
    },
    'state': lambda obs: obs['board'],
    'reward': lambda obs: -1 if obs['illegal'] else 10 if obs['win'] else -10 if obs['loose'] else 0,
    'end_condition': lambda obs: True if obs['win'] or obs['loose'] else False,  #obs['full'] or
    'action': lambda act: {'column': act},
    'agent': {
        'agent_type':DQN,
        'load_pretrained_model': False,
        'model_path':'.zip',
        'evaluate':False,
        'load_replay_buffer': False,
        'buffer_path':'.buf',
        'pretrain':False,
        'total_timestep':100,
        'kwargs':{
            'policy': 'MlpPolicy',
            'train_freq':1,
            'seed': seed,
            'learning_rate':0.0003,
            'gamma':0.999,
            'batch_size': 250,
            'buffer_size': 500000,
            'learning_starts': 250,
            'gradient_steps': 1,
            'policy_kwargs':{
                'activation_fn': Tanh,
                'net_arch': [64, 64],
            },
        },
    },
}