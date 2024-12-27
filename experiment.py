from controls import config

import gymnasium as gym
from environment.gym_adapter import ConnectFourAdapter
from environment.gym_wrappers import RewardWrapper, ObservationWrapper, ActionWrapper, PlayerIDWrapper, SwitchWrapper, customMonitorWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from agent.utils import agent_follow, attach_eval_agent
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
# agent_eval = eval_config['type']
file_name = eval_config['output']
eval_config['env'] = env

# agent_config = config['agent']
# agent = agent_config['agent_type']
# kwargs = agent_config['kwargs']
# agent = agent(env=env, **kwargs)
# pretrained = agent_config.get('model_path')
# agent.load(pretrained)
agent_eval = attach_eval_agent(eval_config)
eval_env = SwitchWrapper(env, agent_eval)
eval_env = customMonitorWrapper(eval_env, file_name, info_keywords=tuple(config['monitor_param']))


# agent switch
agent_config = config['agent']
# agent_type = agent_config['agent_type']
eval_agent = agent_follow(agent_config)
eval_kwargs = agent_config['kwargs']
env = SwitchWrapper(env, eval_agent(env=env, **eval_kwargs))

# Monitor
file_name = agent_config['output']
env = customMonitorWrapper(env, file_name, info_keywords=tuple(config['monitor_param']))

class TestWrapper(gym.Wrapper):
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        print('reward agent 1: ', reward)
        return observation, reward, done, truncated, info

env = TestWrapper(env) 

# AGENT
agent_config = config['agent']
agent = agent_config['agent_type']
kwargs = agent_config['kwargs']

agent = agent(env=env, **kwargs)

if agent_config.get('load_pretrained_model', False):
    pretrained = agent_config.get('pretrained_model_path')
    agent.load(pretrained, env=env)
    if config['agent'].get('evaluate', False):  
        agent.set_training_mode(False)

if agent_config.get('evaluate_policy', False):  
    pretrained = agent_config.get('policy_path')
    agent.load(pretrained)
    evaluate_policy(agent, env=eval_env)

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

callbacks = [checkpoint_callback]
if eval_config.get('use_while_training', False):

    eval_callback =EvalCallback(eval_env, **eval_config['eval_kwargs'])
    callbacks.append(eval_callback)

agent.learn(total_timesteps=total_timestep, callback=callbacks)
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