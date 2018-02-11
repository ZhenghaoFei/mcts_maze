import copy
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(object):
  def __init__(self, dim, probability, fix_start=False):
    self.STATE_DIM = dim
    self.ACTION_DIM = 4
    self.MAX_STEP = 40

    self.PROBABILITY = probability
    self.FIX_START = True

    self.WALL_VALUE = 50
    self.CAR_VALUE = 100
    self.GOAL_VALUE = 200

    # states
    self.map_matrix = np.zeros([self.STATE_DIM, self.STATE_DIM])
    self.car_location = [0, 0]
    self.current_step = 0

    self.reset()

  def get_output_st(self):
    s_t = copy.copy(self.map_matrix)
    s_t[self.car_location[0], self.car_location[1]] = self.CAR_VALUE
    s_t = np.reshape(s_t, [self.STATE_DIM, self.STATE_DIM, 1])
    return s_t

  def save_states(self):
    pass


  def reset(self):
    # create random map
    self.map_matrix = np.random.rand(self.STATE_DIM, self.STATE_DIM)
    self.map_matrix[self.map_matrix<self.PROBABILITY] = self.WALL_VALUE
    self.map_matrix[self.map_matrix!=self.WALL_VALUE] = 0

    # random starting point
    if self.FIX_START:
        self.start = 0, 0
    else:
        self.start = np.random.random_integers(0, self.STATE_DIM, 2)
        self.start = self.start[0], self.start[1]
    self.map_matrix[self.start] = 0 # make sure not start in wall

    self.car_location = self.start

    # random goal point
    if self.FIX_START:
        self.goal = self.STATE_DIM-1, self.STATE_DIM-1
    else:
        self.goal = np.random.random_integers(0, self.STATE_DIM, 2)
        self.goal = self.goal[0], self.goal[1]
    self.map_matrix[self.goal] = self.GOAL_VALUE
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

  def step(self, action):
    self.current_step += 1
    self.done = False

    reward = -0.0 # default reward 
    # env =  np.zeros([10, 10])	
    env_distance = 1 # env use car as center, sensing distance
    self.s_t = np.pad(self.map_matrix, env_distance,'constant', constant_values=self.WALL_VALUE)
    # self.s_t = np.copy(self.map_matrix)
    # if action == None:
    #     self.car_location = self.start
    #     car_x, car_y = self.car_location
    #     env_x = car_x + env_distance
    #     env_y = car_y + env_distance
    #     # env use car as center, sensing distance
    #     self.s_t[env_x, env_y] = CAR_VALUE
    #     # env = self.s_t[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
    #     return self.car_location, reward, self.s_t

    # check if initial_location legal
    if self.map_matrix[self.car_location] == self.WALL_VALUE:
        status = "initial position error"
        print ("initial position error")
        # print("check car loc", self.car_location)
        # print(self.map_matrix)
        self.car_location = self.start
        self.done = True
        return self.s_t, reward, self.done, status

    # do action, move the car
    car_x, car_y = self.car_location

    if action == 0:
        car_x -= 1
    elif action == 1:
        car_x += 1
    elif action == 2:
        car_y += 1
    elif action == 3:
        car_y -= 1
    else:
        print('action error!')
	
    self.car_location = car_x, car_y

    env_x = car_x + env_distance
    env_y = car_y + env_distance
    env_location = env_x, env_y
    # env = self.s_t[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
    # goal_distance = np.sqrt(np.sum((np.asarray(self.goal) - np.asarray(self.car_location))**2)) # the distance from goal
    # print "goal_distance: ", goal_distance

    # print "step: ", step
    # check status
    status = 'normal'
    reward = -0.1
    if self.s_t[env_location] == self.WALL_VALUE:
        # print ("collision")
        reward = -1 # collision reward
        self.done = True
        status = 'collision'
        # env = self.s_t[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]

    elif self.s_t[env_location] == 0:
        # improve = last_goaldistance - goal_distance # whether approach goal
        # if improve > 0:
        #     reward = 0.001 # good moving reward
        # elif improve < 0:
        #     reward = -0.002 # bad moving reward
        if self.current_step >= self.MAX_STEP:
            reward = -1
            self.done = True
            status = 'max_step'

    elif self.s_t[env_location] == self.GOAL_VALUE:
        # print ("congratulations! You arrive destination")
        reward = 1 # get goal reward
        self.done = True
        status = 'arrive'
    self.s_t[env_location] = self.CAR_VALUE

    # self.s_t = self.s_t.ravel
    # env = self.s_t[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
    self.s_t = np.reshape(self.s_t, [self.STATE_DIM[0], self.STATE_DIM[1], 1 ])
    return self.s_t, reward, self.done, status