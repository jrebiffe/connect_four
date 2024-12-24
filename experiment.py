from controls import config

import gymnasium as gym
from environment.gym_adapter import ConnectFourAdapter
from gymnasium.wrappers import TransformReward, TransformObservation, TransformAction
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import spaces
import numpy as np
print(config['init']['action']['n'])

init_config = config["init"]
gym.register('connect_four', entry_point=ConnectFourAdapter)
env = gym.make("connect_four", config=init_config)

# custom reward
reward_fct = config['reward']
env = TransformReward(env, reward_fct)

# custom observation
state_fct = config['state']
observation_space = spaces.Box(
    low=0, 
    high=2,
    shape=(
        init_config['observation']['height'], 
        init_config['observation']['width']), 
    dtype=np.uint8
    )
env = TransformObservation(env, state_fct, observation_space)

# custom action
action_space = spaces.Discrete(init_config['action']['n'])
action_fct = config['action']
env = TransformAction(env, action_fct, action_space)


agent_1 = config['agent']['agent_type']
class customAgent(agent_1):
    def _store_transition(
        self,
        replay_buffer,
        buffer_action,
        new_obs,
        reward,
        dones,
        infos,
    ) -> None:
        self.temp = new_obs, reward, dones, infos
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)


# agent switch
class SwitchWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SwitchWrapper, self).__init__(env)
        self.player = True
        agent = customAgent
        kwargs = config['agent']['kwargs']
        self.inner_agent = agent(env=env, **kwargs)
        total_timestep = config['agent']['total_timestep']
        self.inner_agent.learn(total_timesteps=total_timestep)

    def step(self, action):
        if self.player:
            observation, reward, done, truncated, info = self.env.step(action)
            self.player = not self.player
        else:
            observation, reward, done, truncated, info = self.inner_agent.temp
            self.player = not self.player

        return observation, reward, done, truncated, info

env = SwitchWrapper(env)

# AGENT
agent = config['agent']['agent_type']
kwargs = config['agent']['kwargs']

agent = agent(env=env, **kwargs)

if config['agent'].get('load_pretrained_model', False):
    pretrained = config['agent'].get('model_path')
    agent.load(pretrained, env=env)
    if config['agent'].get('evaluate', False):  
        agent.set_training_mode(False)

if config['agent'].get('evaluate_policy', False):  
    pretrained = config['agent'].get('model_path')
    kwargs = config['agent']['eval_kwargs']
    evaluate_policy(pretrained, env=env, **kwargs)

if config['agent'].get('load_replay_buffer', False):  
    pretrained = config['agent'].get('buffer_path')
    agent.load_replay_buffer(pretrained)
    if config['agent'].get('pretrain', False):
        agent.train()

#TRAINING
total_timestep = config['agent']['total_timestep']

# agent.learn(total_timesteps=total_timestep)
# env.close()

action = [1,4,5,6,2,2]

for _ in range(4):
    obs, info = env.reset()
    for i in range(6):
        obs, reward, done, truncated, info = env.step(action=action[i])
        print(obs)

        if truncated or done:
            break

env.close()