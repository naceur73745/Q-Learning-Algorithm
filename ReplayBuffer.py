import torch
import numpy as np

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences for reinforcement learning.

    Parameters:
    - mem_size: Maximum size of the replay buffer
    - input_dim: Dimension of the input (state)
    - n_actions: Number of possible actions
    - butch_size: Batch size for sampling experiences
    """

    def __init__(self, mem_size, input_dim, n_actions, butch_size):
        self.mem_size = mem_size
        self.state_mem = np.zeros((mem_size, input_dim))
        self.new_state_mem = np.zeros((mem_size, input_dim))
        self.action_mem = np.zeros((mem_size))
        self.reward_mem = np.zeros((mem_size))
        self.done_mem = np.zeros((mem_size,))
        self.mem_cntr = 0
        self.butch_size = butch_size

    def store_action(self, state, new_state, action, reward, done):
        """
        Store an experience (state, new state, action, reward, done) in the replay buffer.

        Parameters:
        - state: Current state
        - new_state: New state after the action
        - action: Chosen action
        - reward: Received reward
        - done: Boolean indicating if the episode is done
        """
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.done_mem[index] = done
        self.mem_cntr += 1

    def sample_mem(self):
        """
        Sample a batch of experiences from the replay buffer.

        Returns:
        - state: Batch of current states
        - new_state: Batch of new states after actions
        - action: Batch of chosen actions
        - reward: Batch of received rewards
        - done: Batch of booleans indicating if episodes are done
        - butch: Indices of the sampled experiences in the replay buffer
        """
        # Find the first empty memory
        mem_empty = min(self.mem_size, self.mem_cntr)

        # Choose indices for the batch
        if mem_empty >= self.butch_size:
            butch = np.random.choice(mem_empty, self.butch_size, replace=False)
        else:
            butch = np.random.choice(mem_empty, self.butch_size, replace=True)

        # Retrieve batches for each component
        state = self.state_mem[butch]
        new_state = self.new_state_mem[butch]
        action = self.action_mem[butch]
        reward = self.reward_mem[butch]
        done = self.done_mem[butch]

        return state, new_state, action, reward, done, butch
