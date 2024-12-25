from controls import config

import gymnasium as gym
from environment.gym_adapter import ConnectFourAdapter
from environment.gym_wrappers import RewardWrapper, ObservationWrapper, ActionWrapper, PlayerIDWrapper, SwitchWrapper, customMonitorWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from agent.utils import agent_follow
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import spaces
import numpy as np

init_config = config['game']

env = gym.make(init_config['name'], config=init_config)

# custom player
starter_fct = init_config['first_player_rule']
env = PlayerIDWrapper(env, starter_fct)

# custom reward
reward_fct = config['reward']
termination_fct = config['end_condition']
env = RewardWrapper(env, reward_fct, termination_fct)

# custom observation
state_fct = config['state']
observation_space = spaces.Box(
    low=0, 
    high=2,
    shape=( 
        init_config['observation']['width'],
        init_config['observation']['height']), 
    dtype=np.uint8
    )
env = ObservationWrapper(env, state_fct, observation_space)

# custom action
action_space = spaces.Discrete(init_config['observation']['width'])
action_fct = config['action']
env = ActionWrapper(env, action_fct, action_space)

# eval
eval_config = config['agent_eval']
eval_kwargs = eval_config['eval_kwargs']
agent_eval = eval_config['type']
eval_env = SwitchWrapper(env, agent_eval)

# agent switch
agent_config = config['agent']
agent_type = agent_config['agent_type']
agent = agent_follow(agent_type)
kwargs = agent_config['kwargs']
env = SwitchWrapper(env, agent(env=env, **kwargs))

# Monitor
file_name = config['output_agent_1']
env = customMonitorWrapper(env, file_name, info_keywords=tuple(['win', 'loose', 'full']))

class TestWrapper(gym.Wrapper):
    count = 0
    def step(self, action):
        
        observation, reward, done, truncated, info = self.env.step(action)
        print('reward agent 1: ', reward)

        # self.env.get_wrapper_attr('player_id')==1
        return observation, reward, done, truncated, info
env = TestWrapper(env) 

# AGENT
agent_config = config['agent']
agent = agent_config['agent_type']
kwargs = agent_config['kwargs']

agent = agent(env=env, **kwargs)

if agent_config.get('load_pretrained_model', False):
    pretrained = agent_config.get('model_path')
    agent.load(pretrained, env=env)
    if config['agent'].get('evaluate', False):  
        agent.set_training_mode(False)

if agent_config.get('evaluate_policy', False):  
    pretrained = agent_config.get('model_path')
    kwargs = agent_config['eval_kwargs']
    evaluate_policy(pretrained, env=env, **kwargs)

if agent_config.get('load_replay_buffer', False):  
    pretrained = agent_config.get('buffer_path')
    agent.load_replay_buffer(pretrained)
    if agent_config.get('pretrain', False):
        agent.train()

#TRAINING
total_timestep = agent_config['total_timestep']

save_path = agent_config['model_path']
save_freq = agent_config['save_freq']
# Save a checkpoint
checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_path)
eval_callback =EvalCallback(eval_env)


agent.learn(total_timesteps=total_timestep, callback=[checkpoint_callback, eval_callback])
env.close()

# action = [1,2,3]

# for _ in range(2):
#     obs, info = env.reset()
#     done = False
#     while not done:
#         obs, reward, done, truncated, info = env.step(action=np.random.choice(action))
#         if done:
#             print('party done:', done)

# env.close()