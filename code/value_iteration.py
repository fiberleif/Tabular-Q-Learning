import argparse
import time
from copy import copy
from environment import VIEnvironment


def parse_arguments():
    print("[Info]: Start to parse arguments from commandline.")
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--maze_input', type=str, default='../env/maze_2.txt',
                        help=' path to the environment input.txt described previously')
    parser.add_argument('--value_file', type=str, default='./vi_value_file.txt',
                        help='path to output the value function')
    parser.add_argument('--q_value_file', type=str, default='./vi_q_value_file.txt',
                        help='path to output the q_value function')
    parser.add_argument('--policy_file', type=str, default='./vi_policy_file.txt',
                        help ='path to output the optimal policy')
    parser.add_argument('--num_epoch', type=int, default=1000,
                        help='the number of train epochs')
    parser.add_argument('--discount_factor', type=float, default=0.9,
                        help='the discount factor')
    # parse arguments
    args = parser.parse_args()
    print("[Info]: Finish parsing arguments from commandline.")
    return args


def train(maze_env, value_file, q_value_file, policy_file, num_epoch, discount_factor, **kwargs):

    # create value function and q value function
    print("[Info]: Start to create value function and q value function component.")
    value_function = {}
    q_value_function = {}
    for state in maze_env.state_space:
        hashing_state = hash_state(state)
        value_function[hashing_state] = 0
        for action in maze_env.action_space:
            hashing_state_action = hash_state_action(state, action)
            q_value_function[hashing_state_action] = 0
    print("[Info]: Finish creating value function and q value function component.")

    # train agent
    start = time.time()
    print("[Info]: Start to train value function and q value function.")
    for epoch in range(num_epoch):
        old_value_function = copy(value_function)
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
        distance = compute_distance(old_value_function, value_function, maze_env)
        if distance < 1e-3:
            end = time.time()
            print("[Convergence]: Epoch {0} and Time {1}s".format(epoch, end - start))
            break
    print("[Info]: Finish training value function and q value function.")

    # output
    print("[Info]: Start to output value function, q value function and policy to file.")
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
    print("[Info]: Finish outputting value function, q value function and policy to file.")


# hash state
def hash_state(state):
    return str(state[0]) + "-" + str(state[1])


# reverse hashing state to state
def reverse_hashing_state(hashing_state):
    return [int(e) for e in hashing_state.split("-")]


# hash state, action
def hash_state_action(state, action):
    return str(state[0]) + "-" + str(state[1]) + "|" + str(action)


# reverse hashing state-action to state, action
def reverse_hashing_state_action(hashing_state_action):
    state , action = hashing_state_action.split("|")
    state = reverse_hashing_state(state)
    action = int(action)
    return state, action


# compute l-infty distance of two function
def compute_distance(old_value_function, value_function, maze_env):
    max_distance = -float("inf")
    for state in maze_env.state_space:
        current_distance = abs(old_value_function[hash_state(state)] - value_function[hash_state(state)])
        if current_distance > max_distance:
            max_distance = current_distance
    return max_distance


def main():
    # parse arguments
    args = parse_arguments()
    # create env
    maze_env = VIEnvironment(args.maze_input)
    # train agent
    train(maze_env, **vars(args))


if __name__ == '__main__':
    main()
