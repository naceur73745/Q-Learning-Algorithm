import random
import numpy as np
from ReplayBuffer import ReplayBuffer 

class Agent:
    """
    Q-learning agent for the Prisoners Dilemma game.

    Parameters:
    - input_dimension: Dimension of the input state
    - n_actions: Number of possible actions
    - lr: Learning rate for updating Q-values
    - gamma: Discount factor for future rewards
    - epsilon: Exploration-exploitation tradeoff factor
    - max_epsilon: Maximum exploration factor
    - min_epsilon: Minimum exploration factor
    - decay_rate: Rate at which exploration factor decays
    - mem_size: Size of the replay buffer
    - butch_size: Batch size for sampling experiences from the replay buffer
    """

    def __init__(self, input_dimension, n_actions, lr, gamma, epsilon, max_epsilon, min_epsilon, decay_rate, mem_size, butch_size):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.input_dimension = input_dimension
        self.n_actions = n_actions
        self.Q_table = np.zeros((input_dimension, n_actions))
        self.mem_size = mem_size
        self.butch_size = butch_size

    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy policy.

        Parameters:
        - state: Current state

        Returns:
        - action: Chosen action
        """
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > self.epsilon:
            state = self.convert_function(state)
            action = np.argmax(self.Q_table[state, :])
        else:
            action = random.choice([0, 1])

        return action

    def convert_function(self, state):
        """
        Convert the given state to an index for Q-table.

        Parameters:
        - state: Current state

        Returns:
        - index: Index corresponding to the state in the Q-table
        """
        if state == (0, 0):
            return 0
        elif state == (0, 1):
            return 1
        elif state == (1, 0):
            return 2
        elif state == (1, 1):
            return 3
        else:
            raise ValueError("Invalid state")

    def learn(self, state, action, reward, new_state):
        """
        Update Q-values based on the Q-learning algorithm.

        Parameters:
        - state: Current state
        - action: Chosen action
        - reward: Received reward
        - new_state: New state after the action
        """
        state = self.convert_function(state)
        new_state = self.convert_function(new_state)

        self.Q_table[state, action] = self.Q_table[state, action] + self.lr * (
            reward + self.gamma * np.max(self.Q_table[new_state, :]) - self.Q_table[state, action]
        )

"""#Env for the Game"""
