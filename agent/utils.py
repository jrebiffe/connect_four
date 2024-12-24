def agent_follow(agent_type):
    class customAgent(agent_type):
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
            if infos[0]['win']:
                print('player 2 won')
            if infos[0]['loose']:
                print('player 2 lost')
            print('reward player 2:', reward[0])
            print(new_obs)
            self.temp = new_obs, reward[0], dones[0], infos[0]['TimeLimit.truncated'], infos[0]
            super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    return customAgent