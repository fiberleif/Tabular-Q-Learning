import argparse


# base class of environment
class Environment(object):
    def __init__(self, env_path):
        print("[Info]: Start to create maze env component from raw environment txt.")
        self.env_path = env_path
        self._get_raw_env()
        self._distil_raw_env()
        print("[Info]: Finish creating maze env component from raw environment txt.")

    def _get_raw_env(self):
        # read raw environment from txt file
        # input: nothing
        # return: raw_env from List class
        fo = open(self.env_path, "r")
        self.raw_env = [line.strip("\n") for line in fo.readlines()]
        assert (len(self.raw_env) >= 0)
        self.row_num = len(self.raw_env)
        self.col_num = len(self.raw_env[0])

    def _distil_raw_env(self):
        # extract state space from raw environment. (remove obstacle locations)
        state_space = []
        start_state_space = []
        goal_state_space = []
        obstacle_space = []
        for row_idx in range(self.row_num):
            for col_idx in range(self.col_num):
                if self.raw_env[row_idx][col_idx] != "*":
                    state_space.append([row_idx, col_idx])
                    if self.raw_env[row_idx][col_idx] == "G":
                        goal_state_space.append([row_idx, col_idx])
                    if self.raw_env[row_idx][col_idx] == "S":
                        start_state_space.append([row_idx, col_idx])
                else:
                    obstacle_space.append([row_idx, col_idx])
        self.state_space = state_space
        self.start_state_space = start_state_space
        self.goal_state_space = goal_state_space
        self.obstacle_space = obstacle_space
        self.action_space = [0, 1, 2, 3]

    def get_transition(self, state, action):
        # return next state when taking action over state
        # check
        if state not in self.state_space:
            print("[Error]: Input state is illegal!")
            raise ValueError
        if action not in self.action_space:
            print("[Error]: Input action is illegal!")
            raise ValueError

        # if state is from goal state space
        if state in self.goal_state_space:
            return state

        # compute pre next_state
        next_state = [None, None]
        if action == 0:
            next_state[0] = state[0]
            next_state[1] = state[1] - 1
        elif action == 1:
            next_state[0] = state[0] - 1
            next_state[1] = state[1]
        elif action == 2:
            next_state[0] = state[0]
            next_state[1] = state[1] + 1
        elif action == 3:
            next_state[0] = state[0] + 1
            next_state[1] = state[1]

        # compute final next_state
        if (next_state[0] not in range(self.row_num)) or (next_state[1] not in range(self.col_num)): # consider wall situation
            next_state = state
        if next_state in self.obstacle_space:  # consider obstacle situation
            next_state = state
        return next_state

    def get_reward(self, state, action):
        # return reward when taking action over state
        # check
        if state not in self.state_space:
            print("[Error]: Input state is illegal!")
            raise ValueError
        if action not in self.action_space:
            print("[Error]: Input action is illegal!")
            raise ValueError
        # compute reward
        if state in self.goal_state_space:
            return 0
        else:
            return -1


# subclass of environment for Value Iteration
class VIEnvironment(Environment):
    def __init__(self, env_path):
        super(VIEnvironment, self).__init__(env_path)

    def step(self, state, action):
        return self.get_transition(state, action), self.get_reward(state, action)


# subclass of environment for Q-Learning
class QLEnvironment(Environment):
    def __init__(self, env_path):
        super(QLEnvironment, self).__init__(env_path)
        self.current_state = self.start_state_space[0]

    def step(self, action):
        next_state = self.get_transition(self.current_state, action)
        reward = self.get_reward(self.current_state, action)
        self.current_state = next_state
        is_terminal = 0
        if next_state in self.goal_state_space:
            is_terminal = 1
        return next_state, reward, is_terminal


    def reset(self):
        self.current_state = self.start_state_space[0]
        return self.current_state


def parse_arguments():
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--maze_input', type=str, default='../env/maze_0.txt',
                        help=' path to the environment input.txt described previously')
    parser.add_argument('--output_file', type=str, default='./output_file.txt',
                        help='path to output the value function')
    parser.add_argument('--action_seq_file', type=str, default='./action_seq_file.txt',
                        help='path to output the q_value function')

    # parse arguments
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse arguments
    args = parse_arguments()
    # create env
    maze_env = QLEnvironment(args.maze_input)
    # get actions sequence
    action_sequence = []
    file = open(args.action_seq_file, "r")
    for line in file.readlines():
        truncated_line = [ int(e) for e in line.strip("\n").split(" ")]
        action_sequence += truncated_line
    file.close()

    # interact with environment and record simultaneously
    file = open(args.output_file, "w")
    for action in action_sequence:
        next_state, reward, is_terminal = maze_env.step(action)
        file.write("{0} {1} {2} {3}\n".format(next_state[0], next_state[1], reward, is_terminal))
    file.close()


