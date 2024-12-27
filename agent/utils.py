from collections import deque
from agent.human_agent import HumanAgent
import numpy as np

def agent_follow(agent_config):
    agent_type = agent_config['agent_type']
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

        def render(self, board):
            self.last_obs = board

        def call_action(self):
            actions, buffer_actions = self._sample_action(self.learning_starts)
            self.buffer_actions = buffer_actions
            return actions[0]
        
        def result(self,
            new_obs, reward, done, info) -> None:

            self.num_collected_steps += 1

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer([info], [done])


            # self.temp = new_obs, reward[0], dones[0], infos[0]['TimeLimit.truncated'], infos[0]
            super()._store_transition(self.replay_buffer, self.buffer_actions, [new_obs], [reward], [done], [info])

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()
            
            # save temporary what the agent sees
            print('reward player 2: ', reward)
            print(new_obs)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else self.num_collected_steps 
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

    return customAgent

def pretainedAgent(agent_config):
    class loadedAgent():
        def __init__(self) -> None:
            agent = agent_config['agent_type']
            kwargs = agent_config.get('agent_kwargs', {})
            env = agent_config['env']
            pretrained = agent_config['policy_path']
            self.agent = agent(env=env, **kwargs)  
            self.agent.load(pretrained)
            self.info = {}

        def render(self, board):
            self.last_obs = board

        def call_action(self):

            if self.info.get('illegal',False):
                action = (self.previous_action+1) % 7
            else:
                action = self.agent.predict(self.last_obs, deterministic=False)[0]
            self.previous_action = action

            return action
        
        def result(self,
            new_obs, reward, done, info) -> None:
            self.info = info
            return
    return loadedAgent()

def rdmAgent():
    class createRdmAgent():
        def __init__(self) -> None:
            self.info = {}
        def render(self, board):
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

def attach_eval_agent(agent_config):
    if agent_config['mode']=='human':
        return HumanAgent()
    elif agent_config['mode']=='load_agent':
        return pretainedAgent(agent_config)
    elif agent_config['mode']=='rdm_agent':
        return rdmAgent()
