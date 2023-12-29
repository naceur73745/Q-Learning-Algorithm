import random
import numpy as np 

class Prisoners:
    """
    Class representing the Prisoners Dilemma environment.

    Parameters:
    - episode_len: Maximum length of each episode
    - n_round: Number of rounds or episodes
    """

    def  __init__(self, episode_len, n_round):
        self.action_space = 2
        self.n_round = n_round
        self.observation_space = 4
        self.current_round = 0
        self.current_step = 0
        self.episode_len = episode_len
        self.done = False
        self.state = (random.choice([0,1]), random.choice([0,1]))
        self.steps = 0
        self.grudge = False
        self.payoff_matrix = {(0,0): (2,2), (0,1): (0,3), (1,0): (3,0), (1,1): (1,1)}
        self.strategies = ["Always_cooperate", "Always_defect", "Grudge", "Tit_for_Tat"]
        self.index = 0
        self.state_total = []
        self.round_state_list = []
        self.choosen_startegy_each_round = []
        self.reward_total = []
        self.reward_round_list = []

    def reset(self):
        """
        Reset the environment for a new episode.

        Returns:
        - state: Initial state of the environment
        """
        # Starting with a random state is better
        self.state = (random.choice([0,1]), random.choice([0,1]))

        # Check if all strategies have been used, reset index if true
        if self.index == len(self.strategies):
            self.index = 0

        self.choosen_startegy_each_round.append(self.strategies[self.index])
        self.current_step = 0
        self.grudge = False
        self.done = False
        self.round_state_list = []
        self.reward_round_list = []

        return self.state

    def coop(self, action):
        """
        Define cooperation action.

        Parameters:
        - action: Action chosen by the agent

        Returns:
        - 0: Cooperation action
        """
        return 0

    def defect(self, action):
        """
        Define defection action.

        Parameters:
        - action: Action chosen by the agent

        Returns:
        - 1: Defection action
        """
        return 1

    def Grudge(self, action):
        """
        Define Grudge action.

        Parameters:
        - action: Action chosen by the agent

        Returns:
        - 1: Grudge action if the opponent's action is 1, otherwise 0
        """
        if action == 1:
            self.grudge = True
        if self.grudge:
            return 1
        else:
            return 0

    def Tit_for_Tat(self, action):
        """
        Define Tit-for-Tat action.

        Parameters:
        - action: Action chosen by the agent

        Returns:
        - self.state[1]: Opponent's last action
        """
        return self.state[1]

    def evaluate(self, action, value):
        """
        Evaluate the agent's action.

        Parameters:
        - action: Action chosen by the agent
        - value: Value indicating the strategy used by the opponent

        Returns:
        - state: Current state after the agent's action
        - reward[0]: Agent's reward
        - self.done: Boolean indicating if the episode is done
        - reward: Tuple containing the rewards
        """
        # Check if the episode is done
        if self.current_step == self.episode_len:
            print("Episode is done")
            self.done = True

        # Set the current state
        self.state = (action, int(random.choice([0,1])))

        # Get the reward
        reward = self.payoff_matrix[self.state]

        # Increment the current step
        self.current_step += 1

        return self.state, reward[0], self.done, reward

    def step(self, action):
        """
        Perform one step in the environment.

        Parameters:
        - action: Action chosen by the agent

        Returns:
        - state: Current state after the agent's action
        - reward[0]: Agent's reward
        - self.done: Boolean indicating if the episode is done
        - reward: Tuple containing the rewards
        """
        reward = (0, 0)

        # Choose a random action for the opponent
        player2_action = np.random.choice([0,1])

        # Check if training is over
        if self.current_round == self.n_round:
            print("Training over")
        elif self.current_step == self.episode_len:
            self.index += 1
            self.reward_total.append(self.reward_round_list)
            self.state_total.append(self.round_state_list)
            self.done = True
        else:
            # Set the current state based on the agent's and opponent's actions
            self.state = (action, player2_action)

            # Increment the current step
            self.current_step += 1

            # Get the reward based on the current state
            reward = self.payoff_matrix[self.state]

            # Append the state and reward to lists
            self.round_state_list.append(self.state)
            self.reward_round_list.append(reward)

        return self.state, reward[0], self.done, reward
