import gymnasium as gym
from game import Game
from copy import deepcopy


class ConnectFourAdapter(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, config):
        super().__init__()
         #config['init'])
        self.action_space = gym.spaces.Box(low=-1, high=1)
        self.observation_space = gym.spaces.Box(low=-1, high=1)

    def step(self, action):
        if self.game_over:
            obs = self.last_obs
            if obs['win']:
                obs['win'] = False
                obs['loose'] = True
        else:
            obs = self.game.step(action)
            obs['loose'] = False
            self.last_obs = deepcopy(obs)
            self.game_over = obs['win'] | obs['full']

        terminated = False
        truncated = False
        reward = 0

        info = deepcopy(obs)
        info.update({'board': self.game.observe()})

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation = None #{'board': self.game.observe()}
        self.game = Game()
        info = {'board': self.game.observe()}
        self.game_over = False
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


gym.register('connect_four', entry_point=ConnectFourAdapter)