from collections import deque
from agent.human_agent import HumanAgent
import numpy as np

def agent_follow(agent_type):
    class customAgent(agent_type):
        num_collected_steps = 0
        def __init__(self, env, **kwargs) -> None:
            super().__init__(env=env, **kwargs)

            if self.ep_info_buffer is None :
                # Initialize buffers if they don't exist, or reinitialize if resetting counters
                self.ep_info_buffer = deque(maxlen=self._stats_window_size)
                self.ep_success_buffer = deque(maxlen=self._stats_window_size)

            if self.action_noise is not None:
                self.action_noise.reset()

            self._setup_learn(total_timesteps=10000000000000)

    return customAgent

def create_inner_agent(env, agent_config):
    agent_type = agent_config['agent_type']
    agent = agent_follow(agent_type)

    class inner_agent_follow(agent):
        def __init__(self, env, agent_config) -> None:
            kwargs = agent_config['kwargs']
            super().__init__(env=env, **kwargs)

        def observe(self, obs):
            self._last_obs = np.array([obs])

        def call_action(self):
            actions, buffer_actions = self._sample_action(self.learning_starts)
            self.buffer_actions = buffer_actions
            return actions[0]
        
        def result(self,
            new_obs, reward, done, info) -> None:

            self.num_collected_steps += 1
            self.num_timesteps += self.env.num_envs

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer([info], [done])


            # self.temp = new_obs, reward[0], dones[0], infos[0]['TimeLimit.truncated'], infos[0]
            self._store_transition(self.replay_buffer, self.buffer_actions, np.array([new_obs]), [reward], [done], [info])

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()
            
            # save temporary what the agent sees
            print('reward player 2: ', reward)
            print('timestep', self.num_timesteps)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else self.num_collected_steps 
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    print('inner agent training with gradient ', gradient_steps)
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

    return inner_agent_follow(env, agent_config)


def pretainedAgent(env, agent_config):
    agent = agent_config['agent_type']
    class loadedAgent(agent):
        def __init__(self, env, agent_config) -> None:
            kwargs = agent_config.get('kwargs', {})
            super().__init__(env=env, **kwargs)  
            pretrained = agent_config['policy_path']
            self = self.load(pretrained)
            self.info = {}

        def observe(self, board):
            self._last_obs = board

        def call_action(self):

            if self.info.get('illegal',False):
                action = (self.previous_action+1) % 7
            else:
                action = self.predict(self._last_obs, deterministic=False)[0]
            self.previous_action = action

            return action
        
        def result(self,
            new_obs, reward, done, info) -> None:
            self.info = info
            return
    return loadedAgent(env, agent_config)

def rdmAgent(env, agent_config):
    class createRdmAgent():
        def __init__(self) -> None:
            self.info = {}
            self.env = env
        def observe(self, board):
            return

        def call_action(self):

            if self.info.get('illegal',False):
                action = (self.previous_action+1) % 7
            else:
                action = np.random.choice(range(7))
            self.previous_action = action

            return action
        
        def result(self,
            new_obs, reward, done, info) -> None:
            self.info = info
            return
    return createRdmAgent()

def create_eval_agent(env, agent_config):
    if agent_config['mode']=='human':
        return HumanAgent(env, agent_config)
    elif agent_config['mode']=='load_agent':
        return pretainedAgent(env, agent_config)
    elif agent_config['mode']=='rdm_agent':
        return rdmAgent(env, agent_config)
