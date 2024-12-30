from controls import config

import gymnasium as gym
from environment.gym_adapter import ConnectFourAdapter
from environment.gym_wrappers import RewardWrapper, ObservationWrapper, ActionWrapper, PlayerIDWrapper, SwitchWrapper, customMonitorWrapper #, TransposeWrapper
from agent.utils import inner_agent_follow, attach_eval_agent
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import TransformObservation
import numpy as np

init_config = config['game']
agent_config = config['agent']

env = gym.make(init_config['name'], config=init_config)

# custom player
starter_fct = init_config['first_player_rule']
env = PlayerIDWrapper(env, starter_fct)

# custom reward
reward_fct = config['reward']
termination_fct = config['end_condition']
env = RewardWrapper(env, reward_fct, termination_fct)

# custom observation to state transformation
state_fct = config['state']
observation_space = gym.spaces.Box(
    low=0, 
    high=2,
    shape=(
        init_config['observation']['width'],
        init_config['observation']['height'],
        ), 
    dtype=np.uint8
    )
env = ObservationWrapper(env, state_fct, observation_space)

# additional transformation for image like state
state_transformer = config['state_transformer']
state_space = gym.spaces.Box(low=0, high=255, 
        shape=(*observation_space.shape,2),
        dtype=np.uint8
        )

# custom action
action_space = gym.spaces.Discrete(init_config['observation']['width'])
action_fct = config['action']
env = ActionWrapper(env, action_fct, action_space)

# eval
eval_config = config['agent_eval']
# eval_kwargs = eval_config['eval_kwargs']
file_name = eval_config['output']
eval_config['env'] = env

# custom environment for eval inner agent
if eval_config.get('kwargs',{}).get('policy','') == "CnnPolicy": 
    eval_env = TransformObservation(env, state_transformer, state_space)
else:
    eval_env = env

eval_agent = attach_eval_agent(eval_env, eval_config)
# eval_env = deepcopy(env)
eval_env = SwitchWrapper(eval_env, eval_agent)
eval_env = customMonitorWrapper(eval_env, file_name, info_keywords=tuple(config['monitor_param']))

# agent switch

# custom environment for training inner agent
# TODO dissociate inner and outer agent
if agent_config['kwargs'].get('policy','') == "CnnPolicy": 
    inner_env = TransformObservation(env, state_transformer, state_space)
else:
    inner_env = env

inner_agent = inner_agent_follow(inner_env, agent_config)
env = SwitchWrapper(env, inner_agent)

# one hot encoding for cnn policy
if agent_config['kwargs'].get('policy','') == "CnnPolicy":
    env = TransformObservation(env, state_transformer, state_space)
    eval_env = TransformObservation(eval_env, state_transformer, state_space)

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