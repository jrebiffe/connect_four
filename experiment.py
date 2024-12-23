from controls import config
import gymnasium as gym
from environment.gym_adapter import ConnectFourAdapter
from gym.wrappers import TransformReward, TransformObservation, TransformAction
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces
import numpy as np

init_config = config["init"]
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
        kwargs['seed'] = config['seed']
        self.inner_agent = agent(env=env, **kwargs)
        total_timestep = config['agent']['total_timestep']
        agent.learn(total_timesteps=total_timestep)

    def step(self, action):
        if self.player:
            observation, reward, done, truncated, info = self.env.step(action)
            self.player = ~self.player
        else:
            observation, reward, done, truncated, info = self.inner_agent.temp
            self.player = ~self.player

        return observation, reward, done, truncated, info


# AGENT
agent = config['agent']['agent_type']
kwargs = config['agent']['kwargs']
kwargs['seed'] = config['seed']

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

agent.learn(total_timesteps=total_timestep)
env.close()