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

        if self.last_obs['previous_player_won']:
            obs = self.last_obs
            obs['previous_player_won'] = False
            obs['previous_player_lost'] = True
            info = deepcopy(obs)
            return obs, reward, terminated, truncated, info

        if self.last_obs['previous_player_aborted']:
            obs = self.last_obs
            obs['previous_player_aborted'] = False
            obs['aborted_the_game'] = True
            info = deepcopy(obs)
            return obs, reward, terminated, True, info

        if self.last_obs['full']:
            obs = self.last_obs
            info = deepcopy(obs)
            return obs, reward, terminated, truncated, info
            

        obs = self.game.step(action)
        info = deepcopy(obs)
        info.update({'board': self.game.observe()})
        info.update({'previous_player_lost': False})
        info.update({'previous_player_aborted': info['illegal']})
        info.update({'aborted_the_game': False})

        self.illegal_count += int(info['illegal'])
        info.update({'illegal_tot': self.illegal_count})

        self.last_obs = deepcopy(info)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        observation = None
        self.game = Game(self.height, self.width)
        info = {'board': self.game.observe(), 'previous_player_won': False, 'full':False, 'previous_player_aborted':False}
        self.game_over = False
        self.illegal_count = 0
        self.last_obs = deepcopy(info)
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


gym.register('connect_four', entry_point=ConnectFourAdapter)