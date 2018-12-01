import argparse
from environment import VIEnvironment


def parse_arguments():
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--maze_input', type=str, default='../env/tiny_maze.txt',
                        help=' path to the environment input.txt described previously')
    parser.add_argument('--value_file', type=str, default='./value_file.txt',
                        help='path to output the value function')
    parser.add_argument('--q_value_file', type=str, default='./q_value_file.txt',
                        help='path to output the q_value function')
    parser.add_argument('--policy_file', type=str, default='./policy_file.txt',
                        help ='path to output the optimal policy')
    parser.add_argument('--num_epoch', type=int, default=5,
                        help='the number of train epochs')
    parser.add_argument('--discount_factor', type=float, default=0.9,
                        help='the discount factor')
    # parse arguments
    args = parser.parse_args()
    return args


def train(maze_env, value_file, q_value_file, policy_file, num_epoch, discount_factor, **kwargs):
    # create value function and q value function
    value_function = {}
    q_value_function = {}
    for state in maze_env.state_space:
        hashing_state = hash_state(state)
        value_function[hashing_state] = 0
        for action in maze_env.action_space:
            hashing_state_action = hash_state_action(state, action)
            q_value_function[hashing_state_action] = 0

    # train
    for epoch in range(num_epoch):
        for state in maze_env.state_space:
            max_value_target = -float("inf")
            for action in maze_env.action_space:
                next_state, reward = maze_env.step(state, action)
                max_q_value_target = -float("inf")
                current_value_target = reward + discount_factor * value_function[hash_state(next_state)]
                if current_value_target > max_value_target:
                    max_value_target = current_value_target
                for next_action in maze_env.action_space:
                    current_q_value_target = reward + discount_factor * \
                                             q_value_function[hash_state_action(next_state, next_action)]
                    if current_q_value_target > max_q_value_target:
                        max_q_value_target = current_q_value_target
                q_value_function[hash_state_action(state, action)] = max_q_value_target
            value_function[hash_state(state)] = max_value_target

    # output
    file = open(value_file, "w")
    for key, value in value_function.items():
        state = reverse_hashing_state(key)
        file.write("{0} {1} {2}\n".format(state[0], state[1], value))
    file.close()

    file = open(q_value_file, "w")
    for key, value in q_value_function.items():
        state, action = reverse_hashing_state_action(key)
        file.write("{0} {1} {2} {3}\n".format(state[0], state[1], action, value))
    file.close()

    file = open(policy_file, "w")
    for state in maze_env.state_space:
        max_action = None
        max_q_value = -float('inf')
        for action in maze_env.action_space:
            current_q_value = q_value_function[hash_state_action(state, action)]
            if current_q_value > max_q_value:
                max_q_value = current_q_value
                max_action = action
        file.write("{0} {1} {2}\n".format(state[0], state[1], max_action))
    file.close()


def hash_state(state):
    return str(state[0]) + "-" + str(state[1])


def reverse_hashing_state(hashing_state):
    return [int(e) for e in hashing_state.split("-")]


def hash_state_action(state, action):
    return str(state[0]) + "-" + str(state[1]) + "|" + str(action)


def reverse_hashing_state_action(hashing_state_action):
    state , action = hashing_state_action.split("|")
    state = reverse_hashing_state(state)
    action = int(action)
    return state, action


def main():
    # parse arguments
    args = parse_arguments()
    # create env
    maze_env = VIEnvironment(args.maze_input)
    # train agent
    train(maze_env, **vars(args))


if __name__ == '__main__':
    main()
