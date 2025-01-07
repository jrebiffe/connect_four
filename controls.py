from stable_baselines3.dqn import DQN
from stable_baselines3.ppo import PPO
from agent.policies import CustomCNN
from torch.nn import Tanh
import numpy as np

seed = 42

config = {
    'game' : {
        'name': 'connect_four',
        'first_player_rule': lambda : np.random.choice([1,2]), #lambda info['nb']: np.random.choice(range(nb))
        # 'action': {'n': 7}, #action = width
        'observation':{'height': 6, 'width':7},
    },
    'state': lambda obs: obs['board'],
    'state_transformer': lambda state: np.transpose((np.arange(2) == np.transpose(state)[...,None]-1).astype(int))*255, #np.array([state])*125, # only used with cnn policy
    'reward': lambda obs:  
            1 if obs['previous_player_lost'] else 
            -1 if obs['aborted_the_game'] else 
            -1 if obs['previous_player_won'] else 0,
    'end_condition': lambda obs: True if obs['full'] or obs['previous_player_won'] or obs['previous_player_lost'] or obs['illegal'] else False,  #
    'action': lambda act: {'column': act},   
    'monitor_param': ['previous_player_won', 'diag', 'col', 'row', 'previous_player_lost', 'full', 'illegal_tot'],
    'agent_eval': {
        'use_while_training':True,
        'output': r"run\\eval_agent_1\\",
        # 'mode': 'human',
        'mode': 'rdm_agent',
        # 'mode': 'load_agent',
        # 'agent_type': DQN,
        # 'policy_path':r"run\\model\\oldest.zip", # previous agent
        # 'kwargs':{'policy': 'MlpPolicy'},
        'eval_kwargs':{
            'n_eval_episodes': 1, 
            'eval_freq': 100,
            'deterministic':False,
            },
        },
    'agent': {
        'output': r"run\\training_agent_1\\",
        'agent_type':DQN,
        'load_pretrained_model': False,
        'pretrained_model_path':r"run\\model\\okish.zip",
        'model_path':r"run\\model\\",
        'save_freq': 20_000,
        'evaluate_policy': False,
        'policy_path':r"run\\model\\rl_model_500000_steps.zip",
        'load_replay_buffer': False,
        'buffer_path':'.buf',
        'pretrain':False,
        'total_timesteps':500_000,
        'kwargs':{
            'train_freq':1,
            'seed': seed,
            # 'learning_rate':0.003,
            'gamma':0.999,
            # 'batch_size': 100,
            # 'buffer_size': 500_000,
            # 'learning_starts': 100,
            'gradient_steps': 1,
            # 'policy': 'MlpPolicy', 
            # 'policy_kwargs':{
            #     'activation_fn': Tanh,
            #     'net_arch': [64, 64],
            # },
            'policy': "CnnPolicy", 
            'policy_kwargs':{
                'features_extractor_class': CustomCNN,
                'features_extractor_kwargs': dict(features_dim=21),
                'net_arch': [64, 64],
            },
        },
    },
}