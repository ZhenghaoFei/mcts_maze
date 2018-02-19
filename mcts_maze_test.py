from maze import MazeEnv
from mcts import Mcts, MctsSwp
from controller import RandomController
from utils import ModelWrapper


maze_env = MazeEnv(6, 0.2)
# maze_env.plot_st(ion=False)

random_policy_fn = RandomController(maze_env)
model_fn = ModelWrapper(maze_env)
mcts = MctsSwp(maze_env, 1., random_policy_fn, model_fn)

s_t = maze_env.reset()

while True:
    maze_env.plot_st(ion=True)

    st = maze_env.save_states()
    action, root_value = mcts.run(st, 100, debug=True)
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

