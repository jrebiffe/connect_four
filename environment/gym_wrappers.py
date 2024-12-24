from typing import Tuple, Any, Callable, SupportsFloat
import gymnasium as gym
from itertools import cycle


class RewardWrapper(gym.Wrapper):
    """
    RewardWrapper

    Apply transformation function to the observation, resulting in new
    _reward value. This wrapper injects the computation at the end of the
    environment.step() function.
    """
    def __init__(self, environment: gym.Env, compute_reward: Callable):
        super().__init__(env=environment)
        self.compute_reward = compute_reward

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:
        observation, _, terminated, truncated, info = self.env.step(action=action)

        # Compute reward
        _reward = self.compute_reward(info)

        return observation, _reward, terminated, truncated, info

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

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict[Any]]:
        action.update({'player_id': self.player_id})
        observation, reward, terminated, truncated, info = self.env.step(action=action)
        if not info['illegal']:
            self.player_id = next(self.players)
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs) -> Tuple[Any, dict[Any]]:
        observation, info = self.env.reset(*args, **kwargs)
        print('reset called')
        # self.player_id = self.starter_rule(info)
        return observation, info          