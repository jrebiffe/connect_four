import gymnasium as gym
from game import Game
from copy import deepcopy


class ConnectFourAdapter(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self, config):
        super().__init__()
        self.height = config['observation']['height']
        self.width = config['observation']['width']
        self.action_space = gym.spaces.Box(low=0, high=self.width)
        self.observation_space = gym.spaces.Box(low=0, high=2)
        # self.game_over = False

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0

        if action['column'] is None:
            obs = self.last_obs
            if obs['win']:
                obs['win'] = False
                obs['loose'] = True
            info = deepcopy(obs)
            # self.game_over = False
            return obs, reward, terminated, truncated, info

        obs = self.game.step(action)
        info = deepcopy(obs)
        info.update({'board': self.game.observe()})
        info.update({'loose': False})

        if not info['illegal']:
            self.illegal_count = 0
        self.illegal_count += int(info['illegal'])
        info.update({'illegal_tot': self.illegal_count})

        self.last_obs = deepcopy(info)
        # self.game_over = obs['win'] or obs['full']
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        # if self.game_over:
        #     return None, self.init_obs #TODO renvoyer les board avec le premier jeton du prochain joueur si c'est lui qui commence

        observation = None #{'board': self.game.observe()}
        self.game = Game(self.height, self.width)
        info = {'board': self.game.observe()}
        self.game_over = False
        self.illegal_count = 0
        self.init_obs = deepcopy(info)
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


gym.register('connect_four', entry_point=ConnectFourAdapter)