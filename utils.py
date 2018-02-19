import numpy as np

from queue import *
from mdp_nodes import StateNode, StateActionNode

class ModelWrapper(object):
    # wrap env model to fit mcts 
    def __init__(self, env):
        self.env = env
        
    def step(self, state, action):
        self.env.load_states(state)
        _, reward, done, status = self.env.step(action)
        state_nxt = self.env.save_states()

        return state_nxt, reward, done

def print_tree(node):
    open_set = Queue()
    closed_set = set()

    root_node = node
    open_set.put(node)
    while not open_set.empty():
        cr_node = open_set.get()
        print_node_info(cr_node)

        for child in cr_node.children:
            if child in closed_set:
                continue
            # if child not in open_set:
            open_set.put(child)
            
        closed_set.add(cr_node)

def print_node_info(node):
    print("")
    print("depth: %i " %node.depth)
    print("type: ", node.type)
    if node.type == "state_action_node":    
        print("action: ", node.action)
    print("visited_times: ", node.visited_times)
    print("cumulative_reward: ", node.cumulative_reward)
    print("num_children: ", node.num_children())
    print("value: ", node.value())



