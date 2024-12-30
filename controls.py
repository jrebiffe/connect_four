from stable_baselines3.dqn import DQN
from agent.policies import CustomCNN
from torch.nn import Tanh
import numpy as np

seed = 43

config = {
    'game' : {
        'name': 'connect_four',
        'first_player_rule': lambda : np.random.choice([1,2]), #lambda info['nb']: np.random.choice(range(nb))
        # 'action': {'n': 7}, #action = width
        'observation':{'height': 6, 'width':7},
    },
    'state': lambda obs: obs['board'],
    'state_transformer': lambda state: (np.arange(2) == state[...,None]-1).astype(int)*125, # only used with cnn policy
    'reward': lambda obs: -0.25 if obs['illegal'] else 
            5 if (obs['win'] and obs['col']) else 
            7 if (obs['win'] and obs['row']) else 
            7 if (obs['win'] and obs['diag']) else 
            -7 if (obs['loose'] and obs['col']) else
            -5 if (obs['loose'] and obs['row']) else  
            -5 if (obs['loose'] and obs['diag']) else 
            1 if (obs['full']) else 0,
    'end_condition': lambda obs: True if obs['full'] or obs['win'] or obs['loose'] or (obs['illegal_tot']>30) else False,  #
    'action': lambda act: {'column': act},   
    'monitor_param': ['win', 'diag', 'col', 'row', 'loose', 'full', 'illegal_tot'],
    'agent_eval': {
        'use_while_training':True,
        'output': r"run\\eval\\",
        # 'mode': 'human',
        # 'mode': 'rdm_agent',
        'mode': 'load_agent',
        'agent_type': DQN,
        'policy_path':r"run\\model\\oldest.zip", # previous agent
        'kwargs':{'policy': 'MlpPolicy'},
        'eval_kwargs':{
            'n_eval_episodes': 1, 
            'eval_freq': 1000,
            },
        },
    'agent': {
        'output': r"run\\agent_1\\",
        'agent_type':DQN,
        'load_pretrained_model': False,
        'pretrained_model_path':r"run\\model\\okish.zip",
        'model_path':r"run\\model\\",
        'save_freq': 25_000,
        'evaluate_policy':False,
        'policy_path':r"run\\model\\rl_model_100000_steps.zip",
        'load_replay_buffer': False,
        'buffer_path':'.buf',
        'pretrain':False,
        'total_timestep':10_000_000,
        'kwargs':{
            'train_freq':1,
            'seed': seed,
            'learning_rate':0.0003,
            'gamma':0.999,
            'batch_size': 100,
            'buffer_size': 500000,
            'learning_starts': 100,
            'gradient_steps': 1,
            # 'policy': 'MlpPolicy', #"CnnPolicy", #
            # 'policy_kwargs':{
            #     'activation_fn': Tanh,
            #     'net_arch': [64, 64],
            # },
            'policy': "CnnPolicy", 
            'policy_kwargs':{
                'features_extractor_class': CustomCNN,
                'features_extractor_kwargs': dict(features_dim=21),
            },
        },
    },
}