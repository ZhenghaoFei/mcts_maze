import numpy as np

class RandomController(object):
    """docstring for RandomController"""
    def __init__(self, env):
        self.env = env
        self.action_dim = self.env.ACTION_DIM

    def get_action(self, s_t):
        return np.random.randint(0, self.action_dim)
