# Author: Minh Hua
# Date: 09/29/2022
# Purpose: Class that handles the intialization and updates of the QAM model with Cellular Automata.

import numpy as np
import random as rd
import math
import matplotlib
import matplotlib.pyplot as plt
from pylab import *

class QAMCA:
    """
    A class to model and simulate the QAM model using Cellular Automata
    """
    def __init__(self, 
    n:int=4, 
    constellation_type:str='square',
    percent_satisfied_agents:float=1.0,
    random_swap_every:int=10,
    random_swap_n_agents:int=1, 
    tolerance:float=0.8, 
    initial_ratio:float=0.5, 
    empty_perc:float=0.1, 
    n_type:str='moore', 
    boundary_cond:str='periodic', 
    rd_seed=None) -> None:
        """
        Description:
            Initialize a QAM CA model.

        Arguments:
            n: the number of bits to encode. 2^n = number of constellation points.
            constellation_type: the type of constellation. For even n the constellation is usually 'square'. For odd n the constellation is usually 'cross'.
            percent_satisfied_agents: the percentage of satisfied agents that are good enough.
            random_swap_every: specify the number of steps to randomly swap agents while satisfaction is not percent_satisfied_agents%.
            random_swap_n_agents: how many agents to randomly swap when satisfaction is not percent_satisfied_agents%.
            tolerance: the tolerance of individuals against others.
            initial_ratio: the initial percentage of one population.
            empty_perc: percentage of empty spaces.
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.
            rd_seed: a random seed to pass to the random number generator. Used to reproduce specific initial configurations.

        Return:
            (None)
        """
        if rd_seed: # set random seed for reproducible experiments
            rd.seed(rd_seed)
        
        # initialization
        self.n = n
        self.constellation_type = constellation_type
        self.percent_satisfied_agents = percent_satisfied_agents
        self.random_swap_every = random_swap_every
        self.random_swap_n_agents = random_swap_n_agents
        self.tolerance = tolerance
        self.initial_ratio = initial_ratio
        self.empty_perc = empty_perc
        self.n_type = n_type
        self.boundary_cond = boundary_cond
        self.cmap = matplotlib.colors.ListedColormap(['red', 'blue']) # used to configure the colors of the CA

    def initialize(self) -> None:
        """
        Description:
            Initialize a configuration for the Schelling Segregation CA model.

        Arguments:
            None

        Return:
            (None)
        """
        # initialize an array of m constellation diagrams
        self.cur_diagrams = []
        # self.next_diagrams = []
        self.total_satisfied = [0 for _ in range(self.n)]
        # calculate configuration dimensions
        self.config_dim = int(math.sqrt(2 ** self.n))
        # Initialize n CA configurations
        for config_idx in range(self.n):
            config = np.ones((self.config_dim, self.config_dim), dtype=np.int8)
            # sample indices for the empty spaces and each population
            og_idx = [i for i in range(self.n * self.n)] # a list of indices from the entire configuration
            first_pop_idx = np.random.choice(og_idx, int(math.floor(self.initial_ratio * len(og_idx))), replace=False) # index for the first population
            # assign levels for the empty spaces and the two populations: -1 = empty space, 0 = first population, 1 = second population
            config = config.flatten()
            config[first_pop_idx] = 0
            config = config.reshape(self.config_dim, self.config_dim)
            # save the configurations
            self.cur_diagrams.append(config)
            # self.next_diagrams.append(config)
        self.step = 0

    def observe(self) -> None:
        """
        Description:
            Call matplotlib to draw the CA configuration.

        Arguments:
            None

        Return:
            (None)
        """
        for config_idx, config in enumerate(self.cur_diagrams):
            plt.subplot(1, self.n, config_idx + 1)
            plt.cla()
            # plt.imshow(config, vmin = 0, vmax = 1, cmap = plt.cm.binary)
            plt.imshow(config, vmin = 0, vmax = 1, cmap = self.cmap)
            percent_satisfied = self.total_satisfied[config_idx] / (self.config_dim ** 2)
            plt.title('Rail {} \nPercent Satisfied \n= {}'.format(config_idx + 1, percent_satisfied))
            # ax = plt.gca()
            # ax.set_xticks(np.arange(0, self.config_dim, 1))
            # ax.set_yticks(np.arange(0, self.config_dim, 1))
            # ax.grid(color='black', linestyle='-', linewidth=1)
            # ax.tick_params(axis='x', colors='white')
            # ax.tick_params(axis='y', colors='white')
        plt.show()

    def get_neighbors(self, x:int, y:int, n_type:str='moore', boundary_cond:str='periodic') -> list:
        """
        Description:
            Return the neighbors of a point at (x, y) according to the neighborhood type and boundary conditions

        Arguments:
            x: the cell's x location.
            y: the cell's y location.
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.

        Return:
            (list) a list of coordinate tuples of the neighbors
        """
        neighbors = []
        if n_type == 'moore':
            delta = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]
        elif n_type == 'neumann':
            delta = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in delta:
                neighbor = None
                if boundary_cond == 'periodic':
                    neighbor = ((x + dx) % self.config_dim, (y + dy) % self.config_dim)
                elif boundary_cond == 'cut-off':
                    if not (x + dx >= self.config_dim or x + dx < 0 or y + dy >= self.config_dim or y + dy < 0):
                        neighbor = (x + dx, y + dy)
                if neighbor:
                    neighbors.append(neighbor)
        return neighbors


    def determine_satisfaction(self, config:np.array, x:int, y:int) -> int:
        """
        Description:
            Determines whether the individual at coordinates (x, y) is satisfied by looking at their neighbors in a Moore neighborhood with periodic boundary conditions.

        Arguments:
            config: the current configuration of the model.
            x: the x-coordinate of the individual.
            y: the y-coordinate of the individual.

        Return:
            (int) 0 for dissatisfied and 1 for satisfied
        """
        # assert x < len(self.config) and y < len(self.config)
        neighbors = self.get_neighbors(x, y, self.n_type, self.boundary_cond)
        current_agent = config[x][y]
        count_similar = 0
        total_n = 0
        # determine whether we have enough neighbors of the same population
        for neighbor_x, neighbor_y in neighbors:
            neighbor = config[neighbor_x][neighbor_y]
            if neighbor != -1 and neighbor == current_agent:
                count_similar += 1
            if neighbor != -1:
                total_n += 1
        # if the agent is isolated with no neighbors
        if total_n == 0:
            return 1
        if (count_similar / total_n) >= self.tolerance:
            return 1 # satisfied
        return 0 # dissastified
    
    def agent_reallocation(self, x:int, y:int) -> tuple:
        """
        Description:
            Re-allocate the individual at coordinate (x, y) to a random empty space.

        Arguments:
            x: the x-coordinate of the individual.
            y: the y-coordinate of the individual.

        Return:
            (tuple) the location of the new individual or None if there are no empty spaces left
        """
        if len(self.empty_spaces) == 0: # cannot move if there are no empty spaces
            return None
        # randomly sample an empty space and remove that space from the running list of empty spaces
        random_empty_space_idx = np.random.randint(0, len(self.empty_spaces))
        new_home = self.empty_spaces[random_empty_space_idx]
        self.empty_spaces.pop(random_empty_space_idx) # remove empty space from list of empty space
        self.empty_spaces.append((x, y)) # add the individual's past location into the list of empty spaces
        return new_home

    def update(self) -> None:
        """
        Description:
            Loop through all current configurations and randomly choose an agent to check for satisfaction.

        Arguments:
            None

        Return:
            (None)
        """
        # update a random agent
        # randomly sample an agent and remove that agent from the list of agents (remove their old location)
        # random_agent_idx = np.random.randint(0, len(self.agents))
        # x, y = self.agents[random_empty_space_idx]
        # self.agents.pop(random_agent_idx) # pop the agent's old location from the list of agent locations
        for config_idx, config in enumerate(self.cur_diagrams):
            nextconfig = config
            # loop through the agents list and look for all dissatisfied agents
            dissatisfied_agents = self.get_dissatisfied_agents(config)
            self.total_satisfied[config_idx] = (self.config_dim * self.config_dim) - len(dissatisfied_agents)
            swap_pair_idx = [i for i in range(len(dissatisfied_agents))]
            swap_pair_first_idx = np.random.choice(len(dissatisfied_agents), math.floor(len(dissatisfied_agents) / 2), replace=False)
            for idx in swap_pair_first_idx:
                swap_pair_idx.remove(idx)
            for first_idx_idx, first_idx in enumerate(swap_pair_first_idx):
                first_dis_agent_x = dissatisfied_agents[first_idx][0]
                first_dis_agent_x = dissatisfied_agents[first_idx][1]
                second_dis_agent_x = dissatisfied_agents[swap_pair_idx[first_idx_idx]][0]
                second_dis_agent_x = dissatisfied_agents[swap_pair_idx[first_idx_idx]][1]
                temp = config[first_dis_agent_x][first_dis_agent_x]
                nextconfig[first_dis_agent_x][first_dis_agent_x] = config[second_dis_agent_x][second_dis_agent_x]
                nextconfig[second_dis_agent_x][second_dis_agent_x] = temp
            # if we are not at satisfactory level of satisfaction
            if (self.total_satisfied[config_idx] / (self.config_dim ** 2)) < self.percent_satisfied_agents and ((self.step % self.random_swap_every) == 0):
                # randomly swap a number of agents:
                # get random pairs of agents to swap
                indices = [(i, j) for i in range(self.config_dim) for j in range(self.config_dim)]
                swap_pairs_idx = np.random.choice(len(indices), self.random_swap_n_agents, replace=False)
                swap_pairs = [indices[i] for i in swap_pairs_idx]
                print(swap_pairs)
                for idx1 in range(0, len(swap_pairs), 2):
                    first_dis_agent_x = swap_pairs[idx1][0]
                    first_dis_agent_y = swap_pairs[idx1][1]
                    second_dis_agent_x = swap_pairs[idx1 + 1][0]
                    second_dis_agent_y = swap_pairs[idx1 + 1][1]
                    temp = config[first_dis_agent_x][first_dis_agent_y]
                    nextconfig[first_dis_agent_x][first_dis_agent_y] = config[second_dis_agent_x][second_dis_agent_y]
                    nextconfig[second_dis_agent_x][second_dis_agent_y] = temp
            # left_dis_agent_idx = 0
            # right_dis_agent_idx = len(dissatisfied_agents) - 1
            # while left_dis_agent_idx < right_dis_agent_idx:
            #     # swap a dissatisified agent with a randomly chosen dissatisfied agent in the list of dissatisfied agents
            #     left_dis_agent_x = dissatisfied_agents[left_dis_agent_idx][0]
            #     left_dis_agent_y = dissatisfied_agents[left_dis_agent_idx][1]
            #     right_dis_agent_x = dissatisfied_agents[right_dis_agent_idx][0]
            #     right_dis_agent_y = dissatisfied_agents[right_dis_agent_idx][1]
            #     temp = config[left_dis_agent_x][left_dis_agent_y]
            #     config[left_dis_agent_x][left_dis_agent_y] = config[right_dis_agent_x][right_dis_agent_y]
            #     config[right_dis_agent_x][right_dis_agent_y] = temp
            #     left_dis_agent_idx += 1
            #     right_dis_agent_idx -= 1                        
            # step the config forward
            self.cur_diagrams[config_idx] = nextconfig
        self.step = self.step + 1

    def get_dissatisfied_agents(self, config:np.array) -> list:
        """
        Description:
            Get the locations of the dissatisified agents.

        Arguments:
            config: the current configuration.

        Return:
            (list) a list of the dissatisfied agents.
        """
        dissatisfied_agents = []
        # loop through the agents list and look for all dissatisfied agents
        for x in range(self.config_dim):
            for y in range(self.config_dim):
                satisfaction = self.determine_satisfaction(config, x, y)
                if satisfaction == 0:
                    dissatisfied_agents.append((x, y))
        return dissatisfied_agents

    def sum_satisfaction(self, config:np.array) -> float:
        """
        Description:
            Get the satisfaction of the entire configuration.

        Arguments:
            config: the current configuration.

        Return:
            (float) the ratio of satisfied agents
        """
        total_satisfaction = 0
        for x in range(self.config_dim):
            for y in range(self.config_dim):
                if config[x][y] != -1: # if we have an agent
                    total_satisfaction += self.determine_satisfaction(x, y)
        return total_satisfaction

def run_loop(model, hyperparams:dict, max_steps:int=300):
    """
    Description:
        Run the model for a certain number of steps or until a threshold is met.

    Arguments:
        model: the model to run the loop for.
        hyperparams: the set of hyperparameters for the model.
        max_steps: the maximum number of steps to run the model.

    Return:
        (tuple) the percent of panicked individuals at each time step and the final configuration
    """
    dim = [] # holds the dimension of the hyperparameters so we can initialize an np array
    for param_vals in hyperparams.values():
        dim.append(len(param_vals))
    # empty np array to hold the results corresponding to a parameter combination
    satisfaction_results = np.empty((dim))
    seg_results = np.empty((dim))
    last_ns = np.empty((dim))
    # loop through every parameter combination and find the best one
    try:
        for idx, _ in np.ndenumerate(satisfaction_results):
            # get the current combination of parameters
            cur_args_list = []
            for cur_param, param_key in zip(idx, hyperparams.keys()):
                # print(param_key, hyperparams[param_key][cur_param])
                cur_args_list.append(hyperparams[param_key][cur_param])
            print("Current args: {}".format(cur_args_list))

            # initialize configuration
            model = model(*cur_args_list)
            model.initialize()

            # update config until satisfaction == 100% or max_steps
            step = 0
            perc_satisfied = model.sum_satisfaction() / model.total_agents
            while (perc_satisfied) < 1 and step < max_steps:
                model.update()

                # update step params
                perc_satisfied = model.sum_satisfaction() / model.total_agents
                step += 1
            
            final_satisfaction = model.sum_satisfaction() / model.total_agents
            final_segregation = model.get_metrics('interface density')
            print("Final percent of satisfied: {}, Final percent of segregation: {}".format(final_satisfaction * 100, final_segregation * 100))
            satisfaction_results[idx] = final_satisfaction
            seg_results[idx] = final_segregation
            last_ns[idx] = step
    except KeyboardInterrupt:
        return satisfaction_results, seg_results, last_ns
    return satisfaction_results, seg_results, last_ns