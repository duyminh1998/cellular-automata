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
    percent_random_excitation:float=0.0001,
    n_type:str='moore', 
    boundary_cond:str='periodic', 
    update_stats:bool=False,
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
            percent_random_excitation: percentage of cells to randomly excite.
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
        self.n = n
        self.pumping_probability = pumping_probability
        self.photon_lifetime = photon_lifetime
        self.electron_lifetime = electron_lifetime
        self.max_photons = max_photons
        self.threshold_delta = threshold_delta
        self.percent_random_excitation_cells = int(percent_random_excitation * n * n)
        self.n_type = n_type
        self.boundary_cond = boundary_cond
        self.update_stats = update_stats
        self.n_pop = 0
        self.N_pop = 0
        if n_type == 'moore':
            self.delta = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]
        elif n_type == 'neumann':
            self.delta = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        self.cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'white']) # used to configure the colors of the CA

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
        # self.c_tilde = np.zeros((self.n, self.n, self.max_photons), dtype=np.int8) # the amount of time since a photon j was created at node i
        self.c_tilde = [] # the amount of time since a photon j was created at node i
        for x in range(self.n):
            self.c_tilde.append([])
            for y in range(self.n):
                self.c_tilde[x].append([])

        # initialize some noise photons
        # rand_x_idx = np.random.choice(self.n, int(0.1 * self.n * self.n))
        # rand_y_idx = np.random.choice(self.n, int(0.1 * self.n * self.n))
        # for x, y in zip(rand_x_idx, rand_y_idx):
        #     self.c_cur[x][y] = np.random.randint(1, self.max_photons)
        #     for photon in range(self.c_cur[x][y]):
        #         self.c_tilde[x][y][photon] = np.random.randint(1, self.photon_lifetime)

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
        im2 = plt.imshow(self.c_cur, cmap=self.cmap)
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
        for dx, dy in self.delta:
            neighbor = None
            if boundary_cond == 'periodic':
                neighbor = ((x + dx) % self.n, (y + dy) % self.n)
            elif boundary_cond == 'cut-off':
                if not (x + dx >= self.n or x + dx < 0 or y + dy >= self.n or y + dy < 0):
                    neighbor = (x + dx, y + dy)
            if neighbor:
                neighbor_photons += self.c_cur[neighbor[0]][neighbor[1]]
                if neighbor_photons > self.threshold_delta: # premature return once we exceed threshold
                    return neighbor_photons
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
        # get a random number of cells to random add noise photons
        rand_locs = [(x, y) for x, y in zip(np.random.choice(self.n, self.percent_random_excitation_cells), np.random.choice(self.n, self.percent_random_excitation_cells))]
        for x, y in rand_locs:
            # randomly add noise photon
            if self.c_cur[x][y] < self.max_photons:
                self.c_next[x][y] = self.c_cur[x][y] + 1
                 # after a photon has been created, we must increment its lifetime
                self.c_tilde[x][y].append(0)
        # loop through each cell
        for x in range(self.n):
            for y in range(self.n):
                # fire the pumping process with probability lambda
                if self.a_cur[x][y] == 0:
                    if rd.random() < self.pumping_probability:
                        self.a_next[x][y] = 1
                # stimulated emission
                else:
                    num_photons_in_neighborhood = self.get_neighbor_photons(x, y, self.n_type, self.boundary_cond)
                    # if there is a sufficient number of photons in the neighborhood
                    if num_photons_in_neighborhood > self.threshold_delta:
                        self.a_next[x][y] = 0
                        # add a photon
                        if self.c_cur[x][y] < self.max_photons:
                            self.c_next[x][y] = self.c_cur[x][y] + 1
                            # after a photon has been created, we must increment its lifetime
                            self.c_tilde[x][y].append(0)
                # electron decay
                if self.a_cur[x][y] == 1:
                    # check whether an electron in the cell has reached its lifetime
                    if self.a_tilde[x][y] >= self.electron_lifetime:
                        self.a_next[x][y] = 0
                    else: # increment the electron lifetime
                        self.a_tilde[x][y] += 1
                # update the photon lifetimes
                if self.c_cur[x][y] > 0:
                    # increment photon lifetime
                    for photon_idx in range(self.c_cur[x][y]):
                        self.c_tilde[x][y][photon_idx] += 1
                    # photon decay
                    photon_idx = 0
                    while photon_idx < len(self.c_tilde[x][y]):
                        if self.c_tilde[x][y][photon_idx] >= self.photon_lifetime:
                            del self.c_tilde[x][y][photon_idx]
                            self.c_next[x][y] -= 1
                        else:
                            photon_idx += 1
                
        # step the config forward
        self.a_cur = self.a_next
        self.c_cur = self.c_next
        self.step = self.step + 1

        # update population statistics if desired
        if self.update_stats:
            self.n_pop = self.c_cur.sum()
            self.N_pop = self.a_cur.sum()

if __name__ == '__main__':
    from tqdm import tqdm
    import numpy as np
    import imageio

    def create_image(model, t:int) -> tuple:
        # Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
        for t in range(t):
            model.update()
        c = np.uint8(255 * (model.c_cur - model.c_cur.min()) / (model.c_cur.max() - model.c_cur.min()))
        return c, model

    def create_frames(n, model, t:int) -> list:
        # Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
        c_frames = []
        for _ in tqdm(range(n)):
            c, model = create_image(model, t)
            c_frames.append(c)
        return c_frames

    n = 300
    pumping_probability = 0.1
    photon_lifetime = 8
    electron_lifetime = 30
    max_photons = 20
    threshold_delta = 1.0
    percent_random_excitation = 0.0001
    n_type = 'moore'
    boundary_cond = 'periodic'
    update_stats = False
    r_seed = 42

    # choose test type
    test = 0

    if test > 0:
        update_stats = True

    # initialize the model
    model = LaserCA(n, pumping_probability, photon_lifetime, electron_lifetime, max_photons, threshold_delta, percent_random_excitation, n_type, boundary_cond, update_stats, r_seed)
    model.initialize()
    
    # create gifs
    if test == 0:
        # update config until max_steps
        frames = 400
        steps_per_frame = 1
        c_frames = create_frames(frames, model, steps_per_frame)

        file_name = 'n_{}_t_{}_lambda_{:.2f}_tau_a_{:.2f}_tau_c_{:.2f}_id_{:.3f}.gif'.format(n, frames, model.pumping_probability, model.electron_lifetime, model.photon_lifetime, rd.random())
        gif_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Independent Study\\src\\output\\lasers\\img\\"
        imageio.mimsave(gif_path + file_name, c_frames, format='gif', fps=15)
    # generate data and plot time series
    elif test == 1:
        max_t = 50
        t = [i for i in range(max_t)]
        n_pop = []
        N_pop = []

        # evolve the model
        for _ in tqdm(t):
            model.update()
            n_pop.append(model.n_pop)
            N_pop.append(model.N_pop)

        plt.scatter(t, n_pop, label='$n(t)$', marker = 'x')
        plt.scatter(t, N_pop, label='$N(t)$', marker = '+')
        plt.legend(loc='best')
        plt.xlabel('$t$')
        plt.title('Time series for $n(t)$ and $N(t)$ \n$R$={:.2f}, $\\tau_c$={:.2f}, $\\tau_a$={:.2f}'.format(model.pumping_probability, model.photon_lifetime, model.electron_lifetime))
        plt.grid()
        plt.show()
    # generate data and plot 2D phase plane
    elif test == 2:
        max_t = 100
        t = [i for i in range(max_t)]
        n_pop = []
        N_pop = []

        # evolve the model
        for _ in range(t):
            model.update()
            n_pop.append(model.n_pop)
            N_pop.append(model.N_pop)

        plt.scatter(N_pop, n_pop, marker = '+')
        plt.xlabel('N')
        plt.ylabel('n')
        plt.title('2D Phase Plane \n$R$={:.2f}, $\\tau_c$={:.2f}, $\\tau_a$={:.2f}'.format(model.pumping_probability, model.photon_lifetime, model.electron_lifetime))
        plt.grid()
        plt.show()
    # generate data only
    elif test == 3:
        max_t = 400
        t = [i for i in range(max_t)]
        n_pop = []
        N_pop = []

        # evolve the model
        for _ in tqdm(t):
            model.update()
            n_pop.append(model.n_pop)
            N_pop.append(model.N_pop)

        pickle_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Independent Study\\src\\output\\lasers\\pickle\\"
        n_file_name = 'photons_n_{}_t_{}_lambda_{:.2f}_tau_a_{:.2f}_tau_c_{:.2f}.pickle'.format(n, max_t, model.pumping_probability, model.electron_lifetime, model.photon_lifetime)
        with open(pickle_path + n_file_name, 'wb') as fh:
            pickle.dump(n_pop, fh)

        NN_fn = 'pop_inverse_n_{}_t_{}_lambda_{:.2f}_tau_a_{:.2f}_tau_c_{:.2f}.pickle'.format(n, max_t, model.pumping_probability, model.electron_lifetime, model.photon_lifetime)
        with open(pickle_path + NN_fn, 'wb') as fh1:
            pickle.dump(N_pop, fh1)
    # plot time series only
    elif test == 4:
        max_t = 400
        t = [i for i in range(max_t)]
        pickle_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Independent Study\\src\\output\\lasers\\pickle\\"
        file_name = 'photons_n_{}_t_{}_lambda_{:.2f}_tau_a_{:.2f}_tau_c_{:.2f}.pickle'.format(n, max_t, pumping_probability, electron_lifetime, photon_lifetime)
        with open(pickle_path + file_name, 'rb') as fh:
            n_pop = pickle.load(fh)
        file_name = 'pop_inverse_n_{}_t_{}_lambda_{:.2f}_tau_a_{:.2f}_tau_c_{:.2f}.pickle'.format(n, max_t, pumping_probability, electron_lifetime, photon_lifetime)
        with open(pickle_path + file_name, 'rb') as fh:
             N_pop = pickle.load(fh)

        plt.scatter(t, n_pop, label='$n(t)$', marker = 'x')
        plt.scatter(t, N_pop, label='$N(t)$', marker = '+')
        plt.legend(loc='best')
        plt.xlabel('$t$')
        plt.title('Time series for $n(t)$ and $N(t)$ \n$R$={:.2f}, $\\tau_c$={:.2f}, $\\tau_a$={:.2f}'.format(model.pumping_probability, model.photon_lifetime, model.electron_lifetime))
        plt.grid()
        plt.show()
    # plot 2D phase plane only
    elif test == 5:
        max_t = 400
        t = [i for i in range(max_t)]
        pickle_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Independent Study\\src\\output\\lasers\\pickle\\"
        file_name = 'photons_n_{}_t_{}_lambda_{:.2f}_tau_a_{:.2f}_tau_c_{:.2f}.pickle'.format(n, max_t, pumping_probability, electron_lifetime, photon_lifetime)
        with open(pickle_path + file_name, 'rb') as fh:
            n_pop = pickle.load(fh)
        file_name = 'pop_inverse_n_{}_t_{}_lambda_{:.2f}_tau_a_{:.2f}_tau_c_{:.2f}.pickle'.format(n, max_t, pumping_probability, electron_lifetime, photon_lifetime)
        with open(pickle_path + file_name, 'rb') as fh:
             N_pop = pickle.load(fh)

        plt.scatter(N_pop, n_pop, marker = '+')
        plt.xlabel('N')
        plt.ylabel('n')
        plt.title('2D Phase Plane \n$R$={:.2f}, $\\tau_c$={:.2f}, $\\tau_a$={:.2f}'.format(model.pumping_probability, model.photon_lifetime, model.electron_lifetime))
        plt.grid()
        plt.show()