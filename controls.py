from stable_baselines3.dqn import DQN
from torch.nn import Tanh

seed = 42

config = {
    'init' : {
        'action': {'n': 7},
        'observation':{'height': 7, 'width':7},
    },
    'state': lambda obs: obs['board'],
    'reward': lambda obs: -1 if obs['illegal'] else 10 if obs['win'] else 0,
    'action': lambda act: {'column': act},
    'agent': {
        'agent_type':DQN,
        'load_pretrained_model': False,
        'model_path':'.zip',
        'evaluate':False,
        'load_replay_buffer': False,
        'buffer_path':'.buf',
        'pretrain':False,
        'total_timestep':50,
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