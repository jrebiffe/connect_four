from controls import config

import gymnasium as gym
from environment.gym_adapter import ConnectFourAdapter
from environment.gym_wrappers import RewardWrapper, ObservationWrapper, ActionWrapper, PlayerIDWrapper
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
env = RewardWrapper(env, reward_fct)

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
env = ObservationWrapper(env, state_fct, observation_space)

# custom action
action_space = spaces.Discrete(init_config['action']['n'])
action_fct = config['action']
env = ActionWrapper(env, action_fct, action_space)


agent_1 = config['agent']['agent_type']
class customAgent(agent_1):
    def learn(self, callback = None, log_interval = 4,
        tb_log_name = "run", reset_num_timesteps = False, progress_bar = False,
    ):
        _, callback = self._setup_learn(
            1,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        # assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        rollout = self.collect_rollouts(
            self.env,
            train_freq=self.train_freq,
            action_noise=self.action_noise,
            callback=callback,
            learning_starts=self.learning_starts,
            replay_buffer=self.replay_buffer,
            log_interval=log_interval,
        )


        if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        return self

    def _store_transition(self, replay_buffer, buffer_action,
        new_obs, reward, dones, infos) -> None:
        # save temporary what the agent sees
        self.temp = new_obs, reward, dones[0], infos[0]['TimeLimit.truncated'], infos[0]
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)


# agent switch
class SwitchWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SwitchWrapper, self).__init__(env)
        agent = customAgent
        kwargs = config['agent']['kwargs']
        self.inner_agent = agent(env=env, **kwargs)

    def step(self, action):
        if self.env.get_wrapper_attr('player_id')==1:
            observation, reward, done, truncated, info = self.env.step(action)
        else:
            self.inner_agent.learn()
            observation, reward, done, truncated, info = self.inner_agent.temp

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

action = [1,4,5,6,2,2,3,4,6,2,1,3,5]

for _ in range(2):
    obs, info = env.reset()
    for i in range(10):
        obs, reward, done, truncated, info = env.step(action=action[i])
        print('party done:', done)
        print(obs)

        if truncated or done:
            break

env.close()