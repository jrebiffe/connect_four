import gymnasium as gym
from game import Game


class ConnectFourAdapter(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, config):
        super().__init__()
        self.game = Game() #config['init'])
        self.action_space = gym.spaces.Box(low=-1, high=1)
        self.observation_space = gym.spaces.Box(low=-1, high=1)

    def step(self, action):
        obs = self.game.step(action)
        obs.update({'board': self.game.observe()})
        terminated = False
        truncated = False
        reward = obs
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation = {'board': self.game.observe()}
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


