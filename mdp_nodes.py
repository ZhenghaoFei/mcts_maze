import numpy as np

class StateNode(object):
    def __init__(self, state, parent, depth):
        self.type = "state_node"
        self.state = state
        self.parent = parent
        self.children = []
        self.visited_times = 0
        self.cumulative_reward = 0.
        self.depth = depth

    def find_child(self, action):
        # check if this child already exist
        # for determinisitc and stochasitc action
        
        exist = False
        exist_child = None
        for child in self.children:
            if child.action == action:
                exist = True
                exist_child = child

        return exist_child, exist

    def append_child(self, child_node):
        self.children.append(child_node)

    def value(self):
        value = self.cumulative_reward / self.visited_times
        return value

    def num_children(self):
        num_children = len(self.children)
        return num_children

class StateActionNode(object):
    def __init__(self, state, action, parent, depth):
        self.type = "state_action_node"
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visited_times = 0
        self.cumulative_reward = 0.
        self.reward = 0.
        self.depth = depth

    def compare_states(self, states1, states2):
        same = True
        for i, state in enumerate(states1):
            if np.size(state) > 1:
                if (state != states2[i]).any():
                    same = False
            else:
                if (state != states2[i]):
                    same = False
        return same

    def find_child(self, state_nxt):
        # check if this child already exist
        # for determinisitc model only

        exist = False
        exist_child = None


        # for child in self.children:
        #     # print("same: ", self.compare_states(child.state, state_nxt))
        #     if self.compare_states(child.state, state_nxt):
        #         exist = True
        #         exist_child = child

        if self.num_children() != 0:
            exist = True
            exist_child = self.children[0]

        return exist_child, exist


    def append_child(self, child_node):
        self.children.append(child_node)


    def value(self):
        value = self.cumulative_reward / self.visited_times
        return value

    def num_children(self):
        num_children = len(self.children)
        return num_children