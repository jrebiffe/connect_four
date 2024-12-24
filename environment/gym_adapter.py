import gymnasium as gym
from game import Game
from copy import deepcopy


class ConnectFourAdapter(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, config):
        super().__init__()
        self.game = Game() #config['init'])
        self.action_space = gym.spaces.Box(low=-1, high=1)
        self.observation_space = gym.spaces.Box(low=-1, high=1)

    def step(self, action):
        action.update({'player_id':self.player_id})
        obs = self.game.step(action)
        info = deepcopy(obs)
        info.update({'board': self.game.observe()})
        terminated = False
        truncated = False
        reward = 0
        # info = obs
        print(info)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation = None #{'board': self.game.observe()}
        info = {'board': self.game.observe()}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


gym.register('connect_four', entry_point=ConnectFourAdapter)