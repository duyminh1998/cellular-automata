# Author: Minh Hua
# Date: 10/07/2022
# Purpose: Class that handles the intialization and updates of the Lasers model with Cellular Automata.

import numpy as np
import random as rd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import copy

class LaserCA:
    """
    A class to model and simulate the Lasers model using Cellular Automata
    """
    def __init__(self, n:int=100, 
    pumping_probability:float=0.01,
    photon_lifetime:int=14,
    electron_lifetime:int=160,
    max_photons:int=45000,
    threshold_delta:float=1.0,
    n_type:str='moore', 
    boundary_cond:str='periodic', 
    rd_seed=None) -> None:
        """
        Description:
            Initialize a Lasers CA model.

        Arguments:
            n: the dimension of the board. Equivalent to generating n x n agents.
            pumping_probability: the probability that an electron goes to a high energy state.
            photon_lifetime: the maximum lifetime of a photon.
            electron_lifetime: the maximum lifetime of an electron.
            max_photons: the maximum number of photons in a cell.
            threshold_delta: the threshold to compare the number of photons in the neighborhood.
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
        self.pumping_probability = pumping_probability
        self.photon_lifetime = photon_lifetime
        self.electron_lifetime = electron_lifetime
        self.max_photons = max_photons
        self.threshold_delta = threshold_delta
        self.n_type = n_type
        self.boundary_cond = boundary_cond

        # self.cmap = matplotlib.colors.ListedColormap(['white', 'red', 'blue']) # used to configure the colors of the CA

    def initialize(self) -> None:
        """
        Description:
            Initialize a configuration for the Lasers CA model.

        Arguments:
            None

        Return:
            (None)
        """
        # CA configurations
        self.a_cur = np.zeros((self.n, self.n), dtype=np.int8) # state of the electrons in cell i at time step t
        self.a_next = np.zeros((self.n, self.n), dtype=np.int8) # state of the electrons in cell i at time step t + 1
        self.a_tilde = np.zeros((self.n, self.n), dtype=np.int8) # the time an electron has spent in the excited state

        self.c_cur = np.zeros((self.n, self.n), dtype=np.int8) # number of photons in cell i at time step t
        self.c_next = np.zeros((self.n, self.n), dtype=np.int8) # number of photons in cell i at time step t + 1
        self.c_tilde = np.zeros((self.n, self.n, self.max_photons), dtype=np.int8) # the amount of time since a photon j was created at node i

        # step variable
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
        plt.cla()
        im2 = plt.imshow(self.c_cur, vmin=0, vmax=2, cmap=plt.cm.binary)
        plt.title("Lasers Cellular Automata \nStep = {}".format(self.step))
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0, self.n, 1))
        # ax.set_yticks(np.arange(0, self.n, 1))
        # ax.grid(color='black', linestyle='-', linewidth=1)
        # ax.tick_params(axis='x', colors='white')
        # ax.tick_params(axis='y', colors='white')
        plt.show()

    def get_neighbor_photons(self, x:int, y:int, n_type:str='moore', boundary_cond:str='periodic') -> int:
        """
        Description:
            Return the photon count in the neighborhood of a point at (x, y) according to the neighborhood type and boundary conditions

        Arguments:
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.

        Return:
            (int) the photon count in the neighborhood.
        """
        neighbor_photons = 0
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
                    neighbor_photons += self.c_cur[neighbor[0]][neighbor[1]]
        return neighbor_photons

    def update(self) -> None:
        """
        Description:
            Loop through the current configuration and update the model

        Arguments:
            None

        Return:
            (None)
        """
        # update time step t + 1
        self.a_next = self.a_cur
        self.c_next = self.c_cur
        # loop through each cell
        for x in range(self.n):
            for y in range(self.n):
                # flag to denote whether new photon was created
                photon_created = False
                # fire the pumping process with probability lambda
                if self.a_cur[x][y] == 0:
                    if rd.random() < self.pumping_probability:
                        self.a_next[x][y] == 1
                # stimulated emission
                else:
                    num_photons_in_neighborhood = self.get_neighbor_photons(x, y, self.n_type, self.boundary_cond)
                    if num_photons_in_neighborhood > self.threshold_delta:
                        self.c_next[x][y] = self.c_cur[x][y] + 1
                        self.a_next[x][y] = 0
                        # after a photon has been created, we must increment its lifetime
                        for photon_idx, photon in enumerate(self.c_tilde[x][y]):
                            if photon == 0:
                                self.c_tilde[x][y][photon_idx] = 1
                                photon_created = photon_idx
                                break
                # photon decay
                if self.c_cur[x][y] > 0:
                    # check whether a photon in the cell has reached its lifetime
                    for photon_idx, photon in enumerate(self.c_tilde[x][y]):
                        if photon >= self.photon_lifetime:
                            self.c_next[x][y] = self.c_cur[x][y] - 1
                            self.c_tilde[x][y][photon_idx] = 0
                            break
                # electron decay
                if self.a_cur[x][y] == 1:
                    # check whether an electron in the cell has reached its lifetime
                    if self.a_tilde[x][y] >= self.electron_lifetime:
                        self.a_next = 0
                    else: # increment the electron lifetime
                        self.a_tilde[x][y] = self.a_tilde[x][y] + 1
                # update the photon lifetimes
                for photon_idx, photon in enumerate(self.c_tilde[x][y]):
                    if photon > 0:
                        if photon_created:
                            if photon_idx != photon_created:
                                self.c_tilde[x][y][photon_idx] = self.c_tilde[x][y][photon_idx] + 1
                        else:
                            self.c_tilde[x][y][photon_idx] = self.c_tilde[x][y][photon_idx] + 1
                
        # step the config forward
        self.a_cur = self.a_next
        self.c_cur = self.c_next
        self.step = self.step + 1

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