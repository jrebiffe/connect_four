from stable_baselines3.dqn import DQN
from agent.human_agent import HumanAgent
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
    'reward': lambda obs: -1 if obs['illegal'] else 10 if obs['win'] else -10 if obs['loose'] else 0,
    'end_condition': lambda obs: True if obs['full'] or obs['win'] or obs['loose'] else False,  #
    'action': lambda act: {'column': act},   
    'agent_eval': {
        'use_while_training':False,
        'output': r"run\\eval\\",
        'type':HumanAgent(),
        'eval_kwargs':{
            'n_eval_episodes': 1, 
            'eval_freq': 100
            },
        },
    'agent': {
        'output': r"run\\agent_1\\",
        'agent_type':DQN,
        'load_pretrained_model': False,
        'model_path':r"run\\model\\rl_model_1000_steps",
        'save_freq': 500,
        'evaluate':True,
        'load_replay_buffer': False,
        'buffer_path':'.buf',
        'pretrain':False,
        'total_timestep':10000,
        'kwargs':{
            'policy': 'MlpPolicy',
            'train_freq':1,
            'seed': seed,
            'learning_rate':0.003,
            'gamma':0.999,
            'batch_size': 100,
            'buffer_size': 500000,
            'learning_starts': 100,
            'gradient_steps': 1,
            'policy_kwargs':{
                'activation_fn': Tanh,
                'net_arch': [64, 64],
            },
        },
    },
}