import numpy as np
import copy

from mdp_nodes import StateNode, StateActionNode
from utils import print_tree

class Mcts(object):
    """
    Monte Carlo Tree Search Planning Method
    0. Normal
    For finite action space and finite state space

    1. Simple Progressive Widening
    For infinite action space finite state space

    """
    def __init__(self, env, exploration_parameter, default_policy_fn, model_fn):
        self.cp = exploration_parameter
        self.default_policy_fn = default_policy_fn
        self.env = env
        self.model_fn = model_fn

    def run(self, st, rollout_times, debug=False):
        root_node = StateNode(st, parent=None, depth=0)

        # grow the tree for N times
        for t in range(rollout_times):
            self.grow_tree(root_node)

        action = self.best_action(root_node)

        if debug:
            print_tree(root_node)


        self.env.load_states(root_node.state)

        root_value = root_node.value()

        return action, root_value

    def best_action(self, node):
        if node.num_children() == 0:
            print("no child in root node")
            action = self.default_policy_fn.get_action(node.state)
        else:
            qs = []
            acs = []
            for child in node.children:

                q = child.value()
                qs.append(q)
                acs.append(child.action)
            qs = np.asarray(qs)
            best_q_idx = np.argmax(qs) 
            action = acs[best_q_idx]
        return action

    def select_action(self, state_node):

        qs = []
        acs = []
        for child in node.children:
            q = child.value() + self.cp * np.sqrt((np.log(state_node.visited_times)/child.visited_times))
            qs.append(q)
            acs.append(acs)
        qs = np.asarray(qs)
        best_q_idx = np.argmax(qs) 
        action = acs[best_q_idx]

        return action

    def select_action_random(self, state_node):

        action = np.random.randint(self.env.ACTION_DIM)

        return action

    def select_action_best_q(self, state_node):

        best_action = np.random.randint(self.env.ACTION_DIM)
        best_q = -np.inf

        for action in range(self.env.ACTION_DIM):
            sa_node, exist = state_node.find_child(action)
            if not exist:
                value = np.inf
            else:
                value = sa_node.value()

            if value > best_q:
                best_action = action
                best_q = value

        return best_action

    def select_action_uct(self, state_node):

        best_action = np.random.randint(self.env.ACTION_DIM)
        best_q = -np.inf

        for action in range(self.env.ACTION_DIM):
            sa_node, exist = state_node.find_child(action)
            if not exist:
                value = np.inf
            else:
                # print("value", sa_node.value())
                # print("cp", self.cp * np.sqrt(np.log(state_node.visited_times)/sa_node.visited_times))
                value = sa_node.value() + self.cp * np.sqrt(np.log(state_node.visited_times)/sa_node.visited_times)

            if value > best_q:
                best_action = action
                best_q = value

        return best_action

    def expansion(self, leaf_state_node):
        action = self.default_policy_fn.get_action(leaf_state_node.state)
        new_sa_node = StateActionNode(leaf_state_node.state, action, parent=leaf_state_node, depth=leaf_state_node.depth+1)
        leaf_state_node.append_child(new_sa_node)


        state_nxt, reward, done = self.model_fn.step(leaf_state_node.state, action)
        new_s_node = StateNode(state_nxt, parent=new_sa_node, depth=new_sa_node.depth+1)
        new_sa_node.append_child(new_s_node)
        new_sa_node.reward = reward


        return new_s_node, done

    def grow_tree(self, root_node):
        current_s_node = root_node
        cumulative_reward = 0.

        # forward phase
        while True:

            # select action add a (s,a) node into tree
            # action = self.select_action_random(current_s_node)
            action = self.select_action_uct(current_s_node)

            new_sa_node, exist = current_s_node.find_child(action)

            if not exist:
                new_sa_node = StateActionNode(current_s_node.state, action, parent=current_s_node, depth=current_s_node.depth+1)
                current_s_node.append_child(new_sa_node)

            # model generate next state add a (s) node into tree
            state_nxt, reward, done = self.model_fn.step(current_s_node.state, action)
            new_s_node, exist = new_sa_node.find_child(state_nxt)

            if not exist:
                new_s_node = StateNode(state_nxt, parent=new_sa_node, depth=new_sa_node.depth+1)
                new_sa_node.append_child(new_s_node)

            new_sa_node.reward = reward


            current_s_node = new_s_node

            if current_s_node.visited_times == 0 or current_s_node.num_children() == 0:
                if not done:
                    current_s_node, done = self.expansion(current_s_node)
                break



        if not done:
            cumulative_reward = self.eval(current_s_node)

        # backward phase
        while True:
            current_s_node.visited_times += 1
            current_s_node.cumulative_reward += cumulative_reward


            if current_s_node.parent == None:
                break

            current_sa_node = current_s_node.parent
            cumulative_reward += current_sa_node.reward
            current_sa_node.cumulative_reward += cumulative_reward
            current_sa_node.visited_times += 1
            current_s_node = current_sa_node.parent

    def eval(self, current_s_node, max_horizon=10):
        horizon = 0
        cumulative_reward = 0

        while True:
            horizon += 1
            action = self.default_policy_fn.get_action(current_s_node.state)
            state_nxt, reward, done = self.model_fn.step(current_s_node.state, action)
            cumulative_reward += reward

            if done or horizon > max_horizon:
                break
        # print("cumulative_reward: ", cumulative_reward)
        return cumulative_reward


class MctsSwp(object):
    """
    Monte Carlo Tree Search Planning Method
    0. Normal
    For finite action space and finite state space

    1. Simple Progressive Widening
    For infinite action space finite state space

    """
    def __init__(exploration_parameter, alpha, default_policy_fn, model_fn):
        self.cp = exploration_parameter
        self.alpha = alpha
        self.default_policy_fn = default_policy_fn
        self.model_fn = model_fn

    def run(st, model, rollout_times):
        root_node = Node(st, parent=None, depth=0)

        # grow the tree for N times
        for t in range(rollout_times):
            self.grow_tree(root_node)

        action = self.best_action(root_node)

        return action

    def best_action(node):
        if node.num_children() == 0:
            print("no child in root node")
            action = self.default_policy_fn.get_action(node.state)
        else:
            qs = []
            acs = []
            for child in node.children:
                q = child.value()
                qs.append(q)
                acs.append(acs)
            qs = np.asarray(qs)
            best_q_idx = np.argmax(qs) 
            action = acs[best_q_idx]
        return best_action

    def select_action_swp(state_node):

        if (state_node.visited_times)**self.alpha > len(state_node.children):
            action = default_policy_fn.get_action(state_node.state)
            new_sa_node = StateActionNode(state_node.state, action, parent=state_node, depth=state_node.depth+1)
            state_node.append_child(new_sa_node)
        else:
            qs = []
            acs = []
            for child in node.children:
                q = child.value() + self.cp * np.sqrt((np.log(state_node.visited_times)/child.visited_times))
                qs.append(q)
                acs.append(acs)
            qs = np.asarray(qs)
            best_q_idx = np.argmax(qs) 
            action = acs[best_q_idx]

        return action

    def select_action(state_node):

        qs = []
        acs = []
        for child in node.children:
            q = child.value() + self.cp * np.sqrt((np.log(state_node.visited_times)/child.visited_times))
            qs.append(q)
            acs.append(acs)
        qs = np.asarray(qs)
        best_q_idx = np.argmax(qs) 
        action = acs[best_q_idx]

        return action

    def grow_tree(root_node):
        current_s_node = root_node
        cumulative_reward = 0.

        while True:
            if current_s_node.visited_times == 0 or node.num_children() == 0:
                break

            # select action add a (s,a) node into tree
            action = select_action(current_s_node)
            new_sa_node = StateActionNode(current_s_node.state, action, parent=current_s_node, depth=current_s_node.depth+1)
            current_s_node.append_child(new_sa_node)

            # model generate next state add a (s) node into tree
            state_nxt, reward = model_fn(current_s_node.state, action)
            new_s_node = StateNode(state_nxt, parent=new_sa_node, depth=new_sa_node.depth+1)
            new_sa_node.append_child(new_s_node)

            current_s_node = new_s_node


        cumulative_reward = self.eval(current_s_node)

        