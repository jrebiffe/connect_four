import traceback
from typing import Tuple, Any, Callable, SupportsFloat
import gymnasium as gym
from itertools import cycle
from copy import deepcopy


class RewardWrapper(gym.Wrapper):
    """
    RewardWrapper

    Apply transformation function to the observation, resulting in new
    _reward value. This wrapper injects the computation at the end of the
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
    ActionWrapper

    Transform action before passing it to the environment
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
        step

        Parameters
        ----------
        action : Any

        Returns
        -------
        Tuple[Any, SupportsFloat, bool, bool, dict[Any]]
            observation, reward, terminated, truncated, info
        """
        _action = self.transform_action(action)
        return self.env.step(_action)


class ObservationWrapper(gym.Wrapper):

    def __init__(self,
                 environment: gym.Env,
                 observation_transformation_function: Callable,
                 observation_space: gym.Space = None):
        """
        Parameters
        ----------
        environment
        observation_transformation_function
        """
        super().__init__(env=environment)

        # Update observation space, if given.
        if observation_space is not None:
            self.observation_space = observation_space

        self.transform_observation = observation_transformation_function

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:
        """
        step

        Parameters
        ----------
        action : Any

        Returns
        -------
        Tuple[Any, SupportsFloat, bool, bool, dict[Any]]
            observation, reward, terminated, truncated, info
        """
        observation, reward, terminated, truncated, info = self.env.step(action=action)
        _observation = self.transform_observation(info)
        return _observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs) -> Tuple[Any, dict[Any]]:
        """
        reset

        Returns
        -------
        Tuple[Any, dict[Any]]
            observation, info
        """
        observation, info = self.env.reset(*args, **kwargs)
        _observation = self.transform_observation(info)
        return _observation, info

class PlayerIDWrapper(gym.Wrapper):

    def __init__(self, environment: gym.Env, starter_rule):
        self.players = cycle(range(1,3))         
        self.starter_rule = starter_rule       
        super().__init__(env=environment)
        self.player_id = 1
        self.info_round = False

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:

        action.update({'player_id': self.player_id})
        observation, reward, terminated, truncated, info = self.env.step(action=action)

        # info['loose'] = False                
        # if info['win']:
        #     self.info_round = True
        #     self.winner = deepcopy(self.player_id)
        #     info['win'] = False
        #     info['loose'] = True

        if not info['illegal']: # or info['loose']:
            self.player_id = next(self.players)

        # if self.info_round:
        #     print('player:', self.player_id)
        #     if self.winner == self.player_id:
        #         info['win'] = True
        #         info['loose'] = False

        print('illegal:', info['illegal'], 'player', self.player_id)
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs) -> Tuple[Any, dict[Any]]:
        target_id = self.starter_rule()
        while self.player_id != target_id :
            self.player_id = next(self.players)
        observation, info = self.env.reset(*args, **kwargs)
        print('reset called')    
        # traceback.print_stack()
        
        return observation, info     

class SwitchWrapper(gym.Wrapper):
    def __init__(self, env, agent, kwargs):
        super(SwitchWrapper, self).__init__(env)
        self.inner_agent = agent(env=env, **kwargs)

    def step(self, action):
        if self.env.get_wrapper_attr('player_id')==1:
            observation, reward, done, truncated, info = self.env.step(action)
            if done: #info['win']:
                # # if the main agent has won, the inner agent needs to be informed before closing the env
                print('player 1 won')
                self.inner_agent.learn()
                observation, reward, done, truncated, info = self.inner_agent.temp
        else:
            self.inner_agent.learn()
            observation, reward, done, truncated, info = self.inner_agent.temp
        
            if done: #info['win']:
                observation, reward, done, truncated, info = self.env.step(None)

        return observation, reward, done, truncated, info     