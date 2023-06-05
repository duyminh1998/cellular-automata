# Author: Minh Hua
# Date: 11/13/2022
# Purpose: Class that handles the intialization and updates of the Rats SIR model with Cellular Automata.

import numpy as np
import random as rd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import copy

class RatsCA:
    """
    A class to model and simulate the Rats SIR model using Cellular Automata
    """
    def __init__(self, 
    T:list,
    Q:list,
    q:float,
    d:float,
    N:int,
    n:int=100,
    n_type:str='moore', 
    boundary_cond:str='periodic', 
    update_stats:bool=False,
    rd_seed=None) -> None:
        """
        Description:
            Initialize a Lasers CA model.

        Arguments:
            T: the probability of an epizootic triggering cases in the neighboring subpopulations.
            Q: the probability that these triggered cases give rise to a persistent endemic.
            q: factor to reduce cases in the endemic phase.
            d: death rate.
            N: the number of susceptible substates.
            n: the dimension of the board. Equivalent to generating n x n agents.
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.
            update_stats: whether or not to save population statistics at each iteration.
            rd_seed: a random seed to pass to the random number generator. Used to reproduce specific initial configurations.

        Return:
            (None)
        """
        if rd_seed: # set random seed for reproducible experiments
            rd.seed(rd_seed)
        
        # initialization
        self.T = T
        self.Q = Q
        self.q = q
        self.d = d
        self.N = N
        self.n = n
        self.n_type = n_type
        self.boundary_cond = boundary_cond
        self.update_stats = update_stats
        if n_type == 'moore':
            self.delta = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]
        elif n_type == 'neumann':
            self.delta = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        self.cmap = matplotlib.colors.ListedColormap(['white', 'gray', 'black']) # used to configure the colors of the CA

        self.E = []
        self.P = []

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
        self.config = np.zeros((self.n, self.n), dtype=np.int8)
        self.S = np.zeros((self.n, self.n), dtype=np.int8)

        # randomly distribute initial persistent endemics
        rand_x_idx = np.random.choice(self.n, int(0.01 * self.n * self.n))
        rand_y_idx = np.random.choice(self.n, int(0.01 * self.n * self.n))
        for x, y in zip(rand_x_idx, rand_y_idx):
            self.config[x][y] = 2    

        self.next_config = self.config

        if self.update_stats:
            E_count = self.count_disease('E')
            P_count = self.count_disease('P')
            self.E.append(E_count)
            self.P.append(P_count)

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
        # im2 = plt.imshow(self.c_cur, vmin=0, vmax=2, cmap=plt.cm.binary)
        im2 = plt.imshow(self.config, cmap=self.cmap)
        plt.title("Rats SIR CA \nStep = {}\n Proportion of epidemic: {}\n Proportion of endemic: {}".format(self.step, self.E[-1], self.P[-1]))
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0, self.n, 1))
        # ax.set_yticks(np.arange(0, self.n, 1))
        # ax.grid(color='black', linestyle='-', linewidth=1)
        # ax.tick_params(axis='x', colors='white')
        # ax.tick_params(axis='y', colors='white')
        plt.show()

    def get_neighbors(self, x:int, y:int, n_type:str='moore', boundary_cond:str='periodic') -> list:
        """
        Description:
            Return the photon count in the neighborhood of a point at (x, y) according to the neighborhood type and boundary conditions

        Arguments:
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.

        Return:
            (list) the neighbors for a cell.
        """
        neighbors = []
        for dx, dy in self.delta:
            neighbor = None
            if boundary_cond == 'periodic':
                neighbor = ((x + dx) % self.n, (y + dy) % self.n)
            elif boundary_cond == 'cut-off':
                if not (x + dx >= self.n or x + dx < 0 or y + dy >= self.n or y + dy < 0):
                    neighbor = (x + dx, y + dy)
            if neighbor:
                neighbors.append(neighbor)
        return neighbors

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
        self.next_config = self.config

        # loop through each cell
        for x in range(self.n):
            for y in range(self.n):
                state = self.config[x][y]
                # if we are in state S
                if state == 0:
                    S = self.S[x][y]
                    # check to see if we transition to a short-lived epidemic or persisten endemic state
                    neighbors = self.get_neighbors(x, y, self.n_type, self.boundary_cond)
                    E_seen = False
                    P_seen = False
                    for neighbor_x, neighbor_y in neighbors:
                        neighbor = self.config[neighbor_x][neighbor_y]
                        if neighbor == 2: # P
                            P_seen = True
                        elif neighbor == 1: # E
                            E_seen = True
                    chance = rd.random()
                    # transitions to E
                    if (E_seen and chance <= self.T[S]*(1-self.Q[S])) or (P_seen and chance <= self.q*self.T[S]*(1-self.Q[S])):
                        self.next_config[x][y] = 1
                    # transitions to P
                    elif (E_seen and chance <= self.T[S]*self.Q[S]) or (P_seen and chance <= self.q*self.T[S]*self.Q[S]):
                        self.next_config[x][y] = 2
                    # increment from Si to Si+1 with 100% probability
                    if self.S[x][y] < self.N - 1:   
                        self.S[x][y] = self.S[x][y] + 1
                # if we are in state E
                elif state == 1:
                    self.next_config[x][y] = 0
                    self.S[x][y] = 0
                # if we are in state P
                elif state == 2:
                    if rd.random() < self.d:
                        self.next_config[x][y] = 0
                        self.S[x][y] = np.random.choice(self.N - 1, size = 1)[0]
                    else:
                        self.next_config[x][y] = 2
                
        # step the config forward
        self.step = self.step + 1
        self.config = self.next_config

        # update population stats if desired
        if self.update_stats:
            E_count = self.count_disease('E')
            P_count = self.count_disease('P')
            self.E.append(E_count)
            self.P.append(P_count)

    def count_disease(self, type:str='E') -> float:
        count = 0
        for x in range(self.n):
            for y in range(self.n):
                if type == 'E' and self.config[x][y] == 1:
                    count += 1
                elif type == 'P' and self.config[x][y] == 2:
                    count += 1
        return count / (self.n * self.n)

if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np
    import imageio

    def create_image(model, t:int) -> tuple:
        # Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
        for t in range(t):
            model.update()
        c = np.uint8(255 * (model.config - model.config.min()) / (model.config.max() - model.config.min()))
        return c, model

    def create_frames(n, model, t:int) -> list:
        # Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
        c_frames = []
        for _ in tqdm(range(n)):
            c, model = create_image(model, t)
            c_frames.append(c)
        return c_frames

    mu_r = 0.03
    mu_f = 0.008

    N = 5
    T_min = 0
    T_max = 0.4
    T = np.arange(T_min, T_max, (T_max - T_min) / N, dtype=float)
    Q_min = 0.3
    Q_max = 1
    Q = np.arange(Q_min, Q_max, (Q_max - Q_min) / N, dtype=float)
    Q[0] = 0
    # Q = [0.125, 0.75, 0.125]
    q = 0.5
    d = 0.2
    n = 300
    n_type = 'moore'
    boundary_cond = 'periodic'
    update_stats = True
    r_seed = 42

    # choose test type
    test = 0

    if test > 0:
        update_stats = True

    # initialize the model
    model = RatsCA(T, Q, q, d, N, n, n_type, boundary_cond, update_stats, r_seed)
    model.initialize()
    
    # create gifs
    if test == 0:
        # update config until max_steps
        frames = 100
        steps_per_frame = 1
        c_frames = create_frames(frames, model, steps_per_frame)

        file_name = 'T_{:.2f}.gif'.format(T[-1])
        gif_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Independent Study\\src\\output\\rats\\img\\"
        imageio.mimsave(gif_path + file_name, c_frames, format='gif', fps=15)