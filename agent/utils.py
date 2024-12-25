from collections import deque

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