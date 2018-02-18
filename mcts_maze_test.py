from maze import MazeEnv
from mcts import Mcts
from controller import RandomController

maze_env = MazeEnv(10, 0.2)
# maze_env.plot_st(ion=False)

random_policy_fn = RandomController(maze_env)
mcts = Mcts(maze_env, 1., random_policy_fn)

s_t = maze_env.reset()

while True:
    maze_env.plot_st(ion=True)

    st = maze_env.save_states()
    action, root_value = mcts.run(st, 5000, debug=True)
    print("best action: ", action, "root_value: ", root_value)

    s_t, reward, done, status = maze_env.step(action, verbose=False)

    if done:
        print("status: ", status)
        break

# s_t = maze.reset()
# for i in range(10):
#     maze.plot_st(ion=False)
#     action = policy_fn.get_action(s_t)
#     maze.step(action, verbose=False)

