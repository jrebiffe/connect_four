import gymnasium as gym
from environment import Game


class ConnectFourAdapter(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, config):
        super().__init__()
        self.game = Game() #config['init'])

    def step(self, action):
        obs = self.game.step(action)
        obs.append({'board': self.game.observe()})
        terminated = False
        truncated = False
        reward = 0
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation = self.game.observe()
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


