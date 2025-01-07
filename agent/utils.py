from collections import deque
from agent.human_agent import HumanAgent
import numpy as np

def create_inner_agent(env, agent_config):
    agent_type = agent_config['agent_type']

    class inner_agent_follow(agent_type):
        def __init__(self, env, agent_config) -> None:
            kwargs = agent_config['kwargs']
            super().__init__(env=env, **kwargs)
            self.log_interval = 4
            total_timesteps = agent_config['total_timesteps']
            self.callback = agent_config.get('callback', None)
            total_timesteps, self.callback = self._setup_learn(total_timesteps, self.callback)
            self.callback.on_training_start(locals(), globals())
            # self.num_collected_steps = 0

        def observe(self, obs):
            self._last_obs = np.array([obs])

        def call_action(self):
            self.policy.set_training_mode(False)

            if self.use_sde:
                self.actor.reset_noise(self.env.num_envs)

            self.callback.on_rollout_start()

            if self.use_sde and self.sde_sample_freq > 0: # and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            actions, buffer_actions = self._sample_action(self.learning_starts, self.action_noise, self.env.num_envs)
            self.buffer_actions = buffer_actions

            return actions[0]
        
        def result(self,
            new_obs, reward, done, truncated, info) -> None:

            self.num_timesteps += self.env.num_envs

            # Give access to local variables
            self.callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not self.callback.on_step():
                return

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer([info], [done])

            self._store_transition(self.replay_buffer, self.buffer_actions, np.array([new_obs]), np.array([reward]), np.array([done]), np.array([info]))

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(np.array([done])):
                if done:
                    # Update stats
                    self._episode_num += 1

                    if self.action_noise is not None:
                        kwargs = dict(indices=[idx]) if self.env.num_envs > 1 else {}
                        self.action_noise.reset(**kwargs)

                    # Log training infos
                    if self.log_interval is not None and self._episode_num % self.log_interval == 0:
                        self._dump_logs()
            
            self.callback.on_rollout_end()

            print('reward player 2: ', reward)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else 1 #self.num_collected_steps 
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
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
                action = self.predict(self._last_obs, deterministic=True)[0]
            self.previous_action = action

            return action
        
        def result(self,
            new_obs, reward, done, truncated, info) -> None:
            self.info = info
            return
    return loadedAgent(env, agent_config)

def rdmAgent(env, agent_config):
    class createRdmAgent():
        def __init__(self) -> None:
            self.info = {}
            self.env = env

        def observe(self, board):
            self.board = board
            return

        def call_action(self):

            action = np.random.choice(range(7))

            i = 0
            while np.count_nonzero(self.board[action,:])>5 and i<7:
                action = np.random.choice(range(7))
                i += 1

            return action
        
        def result(self,
            new_obs, reward, done, truncated, info) -> None:
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
