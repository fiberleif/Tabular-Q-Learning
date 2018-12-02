# Tabular-Q-Learning
This repo is to implement the value iteration and Q-Learning algorithms to solve mazes.
## Maze Environment
The files in env directory describle structure of the maze.  Any maze is rectangular with a start state in the bottom left corner and agoal state in the upper right corner. We use S to indicate the initial state (start state), G to indicate the terminal state (goal state), * to indicate an obstacle, and .  indicate a state an agent can go through. The agent cannot go through states with obstacles. You can assume that there are no obstacles at S and G. There are walls around the border of the maze. For example, an input file could look like the following:
```bash
.*..*..G
S......*
```
The environment comprises four key parts as follows:
### state space
The state is described by a tuple (x,y) where x is the row and y is the column (zero-indexed). obstacle locations are not including.

### action space
There  are  four  available  actions,  0,1,2,3,  at  each  state  and  they  correspond  to  the  directions West, North, East, South respectively.

### dynamic transition
The dynamic transition is **deterministic**:
* if an agent is at the initial state S and takes action 2 (East), then the next state of the agent is (1,1). 
* If the agent takes an action that would result in going into a state with an obstacle, it will instead stay in the same state. 
* If the agent takes an action that would result in hitting the wall, it will also stay in the same state.

### reward function
The goal of the agent is to take as few steps as possible to get to the goal state from the start state.  Hence, the reward is −1 whenever an agent takes an action, regardless of the state at which it takes an action.
## Value iteration
```bash
python code/value_iteration.py <maze_input> <value_file> <q_value_file> <policy file> <num_epoch> <discount_factor>
```
6 command-line arguments are described in detail below:
* **mazeinput**:  path to the environment input.txt.
* **value_file**: path to output the values $V(s)$.
* **q_value_file**: path to output the q-values $Q(s, a)$.
* **policy_file**: path to output the optimal actions $\pi(s)$.
* **num_epoch**: the number of train epochs.
* **discount_factor**:  the discount factor $\gamma$.

A simple example (tinymaze):
```bash
python code/value_iteration.py --maze_input ./env/tiny_maze.txt --value_file vi_value_output.txt --q_value_file vi_q_value_output.txt --policy_file vi_policy_output.txt --discount_factor 5 --discount_factor 0.9
```

##  Q-Learning
```bash
python code/q_learning.py <maze_input> <value_file> <q_value_file> <policy file> <num_episode> <learning_rate> <discount_factor> <epsilon>
```
6 command-line arguments are described in detail below:
* **mazeinput**:  path to the environment input.txt.
* **value_file**: path to output the values $V(s)$.
* **q_value_file**: path to output the q-values $Q(s, a)$.
* **policy_file**: path to output the optimal actions $\pi(s)$.
* **num_episode**: the number of train episodes.
* **max_episode_length**:  the maximum of the length of an episode.
* **learning_rate**: the learning rate $\alpha$.
* **discount_factor**:  the discount factor $\gamma$.
* **epsilon**: the value $\epsilon$ for the epsilon-greedy strategy.

A simple example (tinymaze):
```bash
python code/q_learning.py --maze_input ./env/tiny_maze.txt --value_file ql_value_output.txt --q_value_file ql_q_value_output.txt --policy_file ql_policy_output.txt --num_episode 1000 --max_episode_length 20 --learning_rate 0.8 --discount_factor 0.9 --epsilon 0.05
```
