import copy
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(object):
    def __init__(self, dim, probability, fix_start=False):
        self.STATE_DIM = dim
        self.ACTION_DIM = 4
        self.MAX_STEP = 40

        self.PROBABILITY = probability
        self.FIX_START = fix_start

        self.WALL_VALUE = 50
        self.CAR_VALUE = 100
        self.GOAL_VALUE = 200

        # states
        self.map_matrix = np.zeros([self.STATE_DIM, self.STATE_DIM])
        self.car_location = [0, 0]
        self.current_step = 0
        self.done = False

        self.reset()

    def get_output_st(self):
        if self.done:
            s_t = self.map_matrix = np.zeros([self.STATE_DIM, self.STATE_DIM, 1])
        else:
            s_t = copy.copy(self.map_matrix)
            s_t[self.car_location[0], self.car_location[1]] = self.CAR_VALUE
            s_t = np.reshape(s_t, [self.STATE_DIM, self.STATE_DIM, 1])
        return s_t

    def save_states(self):
        # save states
        states = copy.deepcopy([self.map_matrix, self.car_location, self.current_step, self.done])
        return states

    def load_states(self, saved_states):
        # load states
        self.map_matrix, self.car_location, self.current_step, self.done = copy.deepcopy(saved_states)

    def reset(self):
        # create random map
        self.map_matrix = np.random.rand(self.STATE_DIM, self.STATE_DIM)
        self.map_matrix[self.map_matrix<self.PROBABILITY] = self.WALL_VALUE
        self.map_matrix[self.map_matrix!=self.WALL_VALUE] = 0

        # random starting point
        if self.FIX_START:
            self.start = [0, 0]
        else:
            self.start = np.random.random_integers(0, self.STATE_DIM-1, 2)

        self.map_matrix[self.start[0], self.start[1]] = 0 # make sure not start in wall

        self.car_location = self.start

        # random goal point
        if self.FIX_START:
            self.goal = self.STATE_DIM-1, self.STATE_DIM-1
        else:
            self.goal = np.random.random_integers(0, self.STATE_DIM-1, 2)
            
        self.map_matrix[self.goal[0], self.goal[1]] = self.GOAL_VALUE
        self.current_step = 0

        s_t = self.get_output_st()

        return s_t

    def plot_st(self, ion=False):
        s_t = self.get_output_st()
        s_t = np.squeeze(s_t, axis=2)
        plt.imshow(s_t, interpolation='none')
        
        if ion:
          plt.ion()
          plt.pause(0.1)
        else:
          plt.show()

    def collision_check(self):
        # check status
        status = 'normal'
        reward = 0.
        done = False

        # out of boundry check
        if (self.car_location[0] < 0) or (self.car_location[1] < 0) \
        or (self.car_location[0] >= self.STATE_DIM) or (self.car_location[1] >= self.STATE_DIM):

            reward = -1 # collision reward
            done = True
            status = 'out_of_boundry' 

        elif self.map_matrix[self.car_location[0], self.car_location[1]] == self.WALL_VALUE:
            reward = -1 # collision reward
            done = True
            status = 'collision'

        elif self.map_matrix[self.car_location[0], self.car_location[1]] == self.GOAL_VALUE:
            # print ("congratulations! You arrive destination")
            reward = 1 # get goal reward
            done = True
            status = 'arrive'


        elif self.map_matrix[self.car_location[0], self.car_location[1]] == 0 and self.current_step >= self.MAX_STEP:
            reward = -1 # exceed max step
            done = True
            status = 'max_step'

        return reward, done, status

    def step(self, action, verbose=False):
        assert not self.done
        self.current_step += 1
        reward = 0. # default reward 

        # do action, move the car


        if action == 0:
            self.car_location[0] -= 1
        elif action == 1:
            self.car_location[0] += 1
        elif action == 2:
            self.car_location[1] += 1
        elif action == 3:
            self.car_location[1] -= 1
        else:
            print('action error!')
    
        reward, done, status = self.collision_check()

        self.done = done

        if status == 'normal':
          self.map_matrix[self.car_location[0], self.car_location[1]] == self.CAR_VALUE

        s_t = self.get_output_st()

        if verbose:
          print("cr_step: ", self.current_step)
          print("car_location: ", self.car_location)
          print("action: ", action)
          print("status: ", status)
          print("reward: ", reward)
          print("done: ", done)



        return s_t, reward, done, status