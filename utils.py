import numpy as np

from mcts_utils import StateNode, StateActionNode

class ModelWrapper(object):
    # wrap env model to fit mcts 
    def __init__(self, env):
        self.env = env
        
    def step(self, state, action):
        self.env.load_states(state)
        _, reward, done, status = self.env.step(action)
        state_nxt = self.env.save_states()

        return state_nxt, reward, done





