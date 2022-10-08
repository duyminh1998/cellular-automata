# Author: Minh Hua
# Date: 09/07/2022
# Purpose: Class that handles the intialization and updates of the Schelling Segregation model with Cellular Automata.

import numpy as np
import random as rd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import copy

class SchellingSegregationCA:
    """
    A class to model and simulate the Schelling Segregation model using Cellular Automata
    """
    def __init__(self, n:int=100, tolerance:float=0.8, initial_ratio:float=0.5, empty_perc:float=0.1, n_type:str='moore', boundary_cond:str='periodic', rd_seed=None) -> None:
        """
        Description:
            Initialize a Schelling Segregation CA model.

        Arguments:
            n: the dimension of the board. Equivalent to generating n x n agents.
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
        self.tolerance = tolerance
        self.initial_ratio = initial_ratio
        self.empty_perc = empty_perc
        self.n_type = n_type
        self.boundary_cond = boundary_cond

        if self.empty_perc > 0:
            self.cmap = matplotlib.colors.ListedColormap(['white', 'red', 'blue']) # used to configure the colors of the CA
        else:
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
        # CA configurations
        config = np.ones((self.n, self.n), dtype=np.int8)
        nextconfig = np.ones((self.n, self.n), dtype=np.int8)
        # sample indices for the empty spaces and each population
        og_idx = [i for i in range(self.n * self.n)] # a list of indices from the entire configuration
        if self.empty_perc > 0.0: # if we have empty spaces
            empty_idx = np.random.choice(og_idx, int(math.floor(self.empty_perc * self.n * self.n)), replace=False)
            # remove the indices of the empty spaces from the original index list
            og_idx = list(set(og_idx) - set(empty_idx))
        first_pop_idx = np.random.choice(og_idx, int(math.floor(self.initial_ratio * len(og_idx))), replace=False) # index for the first population
        # assign levels for the empty spaces and the two populations: -1 = empty space, 0 = first population, 1 = second population
        config = config.flatten()
        if self.empty_perc:
            config[empty_idx] = -1
        config[first_pop_idx] = 0
        config = config.reshape(self.n, self.n)
        # save a list of empty spaces and agents for re-allocation
        self.empty_spaces = []
        for ix, iy in np.ndindex(config.shape):
            if config[ix][iy] == -1:
                self.empty_spaces.append((ix, iy))
        # save the configurations
        self.config = config
        self.nextconfig = copy.deepcopy(config)
        self.step = 0
        self.total_satisfied = self.sum_satisfaction()
        self.total_agents = self.n * self.n - (int(math.floor(self.empty_perc * self.n * self.n)))

    def observe(self) -> None:
        """
        Description:
            Call matplotlib to draw the CA configuration.

        Arguments:
            None

        Return:
            (None)
        """        
        plt.cla()
        satisfied_ratio = self.total_satisfied / self.total_agents
        extent = [0, self.n, 0, self.n]
        perc_seg = self.get_metrics('interface density')
        im2 = plt.imshow(self.config, cmap=self.cmap, extent=extent)
        plt.title("Schelling Segregation Cellular Automata \nPercent of satisfied agents: {:.2f}% \nPercent of segregation: {:.2f}% \nStep = {}".format(satisfied_ratio * 100, perc_seg * 100, self.step))
        ax = plt.gca()
        ax.set_xticks(np.arange(0, self.n, 1))
        ax.set_yticks(np.arange(0, self.n, 1))
        ax.grid(color='black', linestyle='-', linewidth=1)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        # ax.set_title("Percent of satisfied agents: {}%".format(satisfied_ratio * 100))
        # plt.grid(which='major', linestyle='-', linewidth=0.5)
        plt.show()

    def get_neighbors(self, x:int, y:int, n_type:str='moore', boundary_cond:str='periodic') -> list:
        """
        Description:
            Return the neighbors of a point at (x, y) according to the neighborhood type and boundary conditions

        Arguments:
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
                    neighbor = ((x + dx) % self.n, (y + dy) % self.n)
                elif boundary_cond == 'cut-off':
                    if not (x + dx >= self.n or x + dx < 0 or y + dy >= self.n or y + dy < 0):
                        neighbor = (x + dx, y + dy)
                if neighbor:
                    neighbors.append(neighbor)
        return neighbors


    def determine_satisfaction(self, x:int, y:int) -> int:
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
        current_agent = self.config[x][y]
        count_similar = 0
        total_n = 0
        # determine whether we have enough neighbors of the same population
        for neighbor_x, neighbor_y in neighbors:
            neighbor = self.config[neighbor_x][neighbor_y]
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
            Loop through the current configuration and randomly choose an agent to check for satisfaction.

        Arguments:
            None

        Return:
            (None)
        """
        assert self.config.shape == self.nextconfig.shape
        # update a random agent
        # randomly sample an agent and remove that agent from the list of agents (remove their old location)
        # random_agent_idx = np.random.randint(0, len(self.agents))
        # x, y = self.agents[random_empty_space_idx]
        # self.agents.pop(random_agent_idx) # pop the agent's old location from the list of agent locations
        self.nextconfig = self.config
        self.total_satisfied = 0
        # loop through the agents list and look for all dissatisfied agents
        for x in range(self.n):
            for y in range(self.n):
                if self.config[x][y] != -1: # if we have an agent
                    satisfaction = self.determine_satisfaction(x, y)
                    if satisfaction == 0: # re-allocate agent if not satisfied
                        new_home = self.agent_reallocation(x, y)
                        if new_home: # if a new home is found, then relocate. Else, keep the agent where they are
                            new_home_x = new_home[0]
                            new_home_y = new_home[1]
                            self.nextconfig[new_home_x][new_home_y] = self.config[x][y]
                            self.nextconfig[x][y] = -1 # make the agent's old location empty
                    else: # satisfied
                        self.total_satisfied = self.total_satisfied + 1
        # step the config forward
        self.config = self.nextconfig
        self.step = self.step + 1

    def sum_satisfaction(self) -> float:
        """
        Description:
            Get the satisfaction of the entire configuration.

        Arguments:
            None

        Return:
            (float) the ratio of satisfied agents
        """
        total_satisfaction = 0
        for x in range(self.n):
            for y in range(self.n):
                if self.config[x][y] != -1: # if we have an agent
                    total_satisfaction += self.determine_satisfaction(x, y)
        return total_satisfaction

    def get_metrics(self, metric_type:str) -> float:
        # interface denstiy
        # if metric_type == 'interface density':
        #     count_sim = 0
        #     count_dis = 0
        #     for x in range(self.n):
        #         for y in range(self.n):
        #             agent = self.config[x][y]
        #             if agent != -1:
        #                 neighbors = self.get_neighbors(x, y, self.n_type, self.boundary_cond)
        #                 for neighbor_x, neighbor_y in neighbors:
        #                     neighbor = self.config[neighbor_x][neighbor_y]
        #                     if neighbor != -1:
        #                         if neighbor == agent:
        #                             count_sim += 1
        #                         else:
        #                             count_dis += 1
        #     return count_sim / (count_sim + count_dis)
        if metric_type == 'interface density':
            count_sim = 0
            count_dis = 0
            for x in range(self.n):
                for y in range(self.n):
                    agent = self.config[x][y]
                    if agent != -1:
                        # neighbors: (x + 1 % n, y), (x, y + 1 % n), (x + 1 % n, y + 1 % n)
                        d = [(0, 1), (1, 0), (1, -1), (1, 1)]
                        for dx, dy in d:
                            new_x = x + dx
                            new_y = y + dy
                            if new_x < self.n and new_y < self.n and new_x >= 0 and new_y >= 0:
                                neighbor = self.config[new_x][new_y]
                                if neighbor != -1:
                                    if neighbor == agent:
                                        count_sim += 1
                                        # print("x: {}, y: {}, nx: {}, ny: {}, Same".format(x, y, new_x, new_y))
                                    elif neighbor != agent:
                                        count_dis += 1
                                        # print("x: {}, y: {}, nx: {}, ny: {}, Diff".format(x, y, new_x, new_y))
            # print("Similar: {}, Different: {}".format(count_sim, count_dis))
            # print(self.config)
            return count_sim / (count_sim + count_dis)

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