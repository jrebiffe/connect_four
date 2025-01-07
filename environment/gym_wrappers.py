import traceback
import numpy as np
from typing import Tuple, Any, Callable, SupportsFloat
import gymnasium as gym
import time
from stable_baselines3.common.monitor import Monitor


class RewardWrapper(gym.Wrapper):
    """
    RewardWrapper

    This wrapper injects the computation at the end of the
    environment.step() function.
    """
    def __init__(self, environment: gym.Env, compute_reward: Callable, compute_termination: Callable):
        super().__init__(env=environment)
        self.compute_reward = compute_reward
        self.compute_termination = compute_termination

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:
        observation, _, terminated, truncated, info = self.env.step(action=action)

        # Compute reward
        _reward = self.compute_reward(info)
        _terminated = self.compute_termination(info)

        return observation, _reward, _terminated, truncated, info

class ActionWrapper(gym.Wrapper):
    """
    """
    def __init__(self,
                 environment: gym.Env,
                 action_transformation_function: Callable,
                 action_space: gym.Space = None):

        super().__init__(env=environment)

        # Update action space if provided.
        if action_space is not None:
            self.action_space = action_space
        self.transform_action = action_transformation_function

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:
        """
        """
        _action = self.transform_action(action)
        return self.env.step(_action)


class ObservationWrapper(gym.Wrapper):

    def __init__(self,
                 environment: gym.Env,
                 observation_transformation_function: Callable,
                 observation_space: gym.Space = None):
        """
        """
        super().__init__(env=environment)

        # Update observation space, if given.
        if observation_space is not None:
            self.observation_space = observation_space

        self.transform_observation = observation_transformation_function

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:
        """
        step
        """
        observation, reward, terminated, truncated, info = self.env.step(action=action)
        _observation = self.transform_observation(info)
        return _observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs) -> Tuple[Any, dict[Any]]:
        """
        reset
        """
        _, info = self.env.reset(*args, **kwargs)
        _observation = self.transform_observation(info)
        return _observation, info

class PlayerIDWrapper(gym.Wrapper):

    def __init__(self, environment: gym.Env, starter_rule):
        self.nb_players = 2         
        self.starter_rule = starter_rule       
        super().__init__(env=environment)

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:

        action.update({'player_id': self.player_id})
        observation, reward, terminated, truncated, info = self.env.step(action=action)

        self.player_id = self.player_id % self.nb_players +1
        # if not info['illegal']: # or info['loose']:
        #     self.player_id = self.player_id % self.nb_players +1
        # else:
        #     print('illegal action by player ', self.player_id)

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs) -> Tuple[Any, dict[Any]]:
        self.player_id = self.starter_rule()

        print('reset called') 

        return self.env.reset(*args, **kwargs)    

class SwitchWrapper(gym.Wrapper):
    def __init__(self, env, agent):
        super(SwitchWrapper, self).__init__(env)
        self.inner_agent = agent
        extracted_env = self.inner_agent.env
        try:
            self.transform = extracted_env.envs[0].get_wrapper_attr('func')
            # self.transform = lambda obs: np.transpose(self.transform(np.transpose(obs)))
        except AttributeError:
            self.transform = lambda obs: obs

    def step(self, action):
        transition = list(self.env.step(action))
        player_1_done = transition[2]
        player_2_done = False

        while self.env.get_wrapper_attr('player_id')!=1:
            transition[0] = self.transform(transition[0])
            if self.player_2_has_played:
                self.inner_agent.result(*transition)
            self.inner_agent.observe(transition[0])
            action = self.inner_agent.call_action()
            transition = list(self.env.step(action))
            self.player_2_has_played = True
            player_2_done = transition[2]

        # if player 2 ended the game, we inform it before the player 1 reset the episode
        if player_2_done and (not player_1_done):
            trans = list(self.env.step(None))
            trans[0] = self.transform(trans[0])
            self.inner_agent.result(*trans)

        return transition
    
    def reset(self, *args, **kwargs) -> Tuple[Any, dict[Any]]:
        observation, info = self.env.reset(*args, **kwargs) 
        self.init_obs = observation
        self.init_info = info
        self.player_2_has_played = False

        if self.env.get_wrapper_attr('player_id')!=1:
            self.inner_agent.observe(self.transform(observation))
            action = self.inner_agent.call_action()
            observation, _, _, _, info = self.env.step(action)
            self.player_2_has_played = True

        return observation, info

class customMonitorWrapper(Monitor):
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:
        observation, reward, done, truncated, info = super().step(action)
        self.needs_reset = False
        return observation, reward, done, truncated, info 
    
class innerMonitorWrapper(Monitor):
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:
        
        observation, reward, done, truncated, info = self.env.step(action)

        player_id = self.env.get_wrapper_attr('player_id')
        if player_id !=1:
            self.rewards.append(float(reward))
            if done or truncated:
                self.needs_reset = True
                ep_rew = sum(self.rewards)
                ep_len = len(self.rewards)
                ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    ep_info[key] = info[key]
                self.episode_returns.append(ep_rew)
                self.episode_lengths.append(ep_len)
                self.episode_times.append(time.time() - self.t_start)
                ep_info.update(self.current_reset_info)
                if self.results_writer:
                    self.results_writer.write_row(ep_info)
                info["episode"] = ep_info
            self.total_steps += 1
        self.needs_reset = False
        return observation, reward, done, truncated, info 