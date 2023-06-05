# Author: Minh Hua
# Date: 12/5/2022
# Purpose: This module contains the Constellation class used in GACE.

import random
import numpy as np
import math
import copy
import os

def hamming(str1:str, str2:str) -> int:
    """
    Description:
        Calculate the Hamming distance between two strings.

    Arguments:
        str1: the first string.
        str2: the second string.

    Return:
        (int) the Hamming distance between the two strings.
    """
    dist = 0
    for b1, b2 in zip(str1, str2):
        if b1 != b2:
            dist += 1
    return dist

class Constellation:
    """A class that represents a constellation diagram."""
    def __init__(self, 
        n:int,
        fitness:float=None
    ):
        """
        Description:
            Initializes an inter-task mapping.

        Arguments:
            n: the number of bits.
            fitness: the initial fitness of the mapping.

        Return:
            (None)
        """
        # constellation attributes
        self.n = n
        self.fitness = None
        self.initial_ratio = 0.5
        if n % 2 == 0:
            self.constellation_type = 'square'
            self.config_dim = int(math.sqrt(2 ** self.n))
        else:
            self.constellation_type = 'cross'
            self.config_dim = int(math.sqrt(2 ** self.n + (4 * 4 ** ((self.n - 5) / 2))))

        # initialize the constellation
        self.constellation, self.zeros_locs, self.ones_locs = [], [], []

        # assign a unique ID to the offspring as a function of its state and action mapping
        self.ID = self.create_ID()

    def create_ID(self) -> str:
        """
        Description:
            Creates an ID for the mapping as a function of its state and action mapping.

        Arguments:
            None

        Return:
            (None)
        """
        ID = ""
        for rail in self.constellation:
            for x in range(self.config_dim):
                for y in range(self.config_dim):
                    if rail[x][y] != -1:
                        ID += str(rail[x][y])
        return ID

    def record_01_locs(self, rail:np.array, empty_coords=None) -> tuple:
        """
        Description:
            Record the locations of the ones and zeros.

        Arguments:
            rail: the current rail.
            empty_coords: a list of empty coordinates for cross constellations.

        Return:
            (tuple) the locations of the zeros and the locations of the ones.
        """
        zeros_locs = []
        ones_locs = []
        # record where the ones and zeros are
        for x in range(self.config_dim):
            for y in range(self.config_dim):
                if (empty_coords and (x, y) not in empty_coords) or not empty_coords:
                    if rail[x][y] == 0 :
                        zeros_locs.append((x, y))
                    else:
                        ones_locs.append((x, y))
        return zeros_locs, ones_locs

    def initialize_constellation(self) -> tuple:
        """
        Description:
            Initialize a constellation diagram.

        Arguments:
            None

        Return:
            (tuple) the rails, the locations of the zeros, and the locations of the ones.
        """
        # initialize an array of n constellation diagrams of rails
        cur_diagrams = []
        self.empty_coords = None
        # Initialize n rails
        if self.constellation_type == 'square':
            zeros_locs = []
            ones_locs = []
            for _ in range(self.n):
                zeros_locs.append([])
                ones_locs.append([])                
                config = np.ones((self.config_dim, self.config_dim), dtype=np.int8)
                # sample indices for the empty spaces and each population
                og_idx = [i for i in range(int(self.config_dim ** 2))] # a list of indices from the entire configuration
                first_pop_idx = np.random.choice(og_idx, int(math.floor(self.initial_ratio * len(og_idx))), replace=False) # index for the first population
                # assign levels for the empty spaces and the two populations: -1 = empty space, 0 = first population, 1 = second population
                config = config.flatten()
                config[first_pop_idx] = 0
                config = config.reshape(self.config_dim, self.config_dim)
                # save the configurations
                cur_diagrams.append(config)
                # record where the ones and zeros are
                zeros_locs[-1], ones_locs[-1] = self.record_01_locs(config)
        elif self.constellation_type == 'cross':
            vertical_offset = int(2 ** ((self.n - 5) / 2))
            horizontal_offset = int(2 ** ((self.n - 5) / 2))
            vertical_offsets = [v for v in range(vertical_offset)] + [self.config_dim - v for v in range(1, vertical_offset + 1)]
            horizontal_offsets = [h for h in range(horizontal_offset)] + [self.config_dim - h for h in range(1, horizontal_offset + 1)]
            self.empty_coords = []
            for v in vertical_offsets:
                for h in horizontal_offsets:
                    self.empty_coords.append((v, h))
            empty_coords_flattened = []
            og_idx = []
            flatten_idx = 0
            for i in range(self.config_dim):
                for j in range(self.config_dim):
                    if (i, j) in self.empty_coords:
                        empty_coords_flattened.append(flatten_idx)
                    else:
                        og_idx.append(flatten_idx)
                    flatten_idx += 1
            zeros_locs = []
            ones_locs = []
            for _ in range(self.n):
                zeros_locs.append([])
                ones_locs.append([])
                config = np.ones((self.config_dim, self.config_dim), dtype=np.int8)
                # sample indices for the empty spaces and each population
                first_pop_idx = np.random.choice(og_idx, int(math.floor(self.initial_ratio * len(og_idx))), replace=False) # index for the first population
                # assign levels for the empty spaces and the two populations: -1 = empty space, 0 = first population, 1 = second population
                config = config.flatten()
                config[first_pop_idx] = 0
                config[empty_coords_flattened] = -1
                config = config.reshape(self.config_dim, self.config_dim)
                # save the configurations
                cur_diagrams.append(config)
                # record where the ones and zeros are
                zeros_locs[-1], ones_locs[-1] = self.record_01_locs(config, self.empty_coords)
        else:
            raise ValueError("Invalid constellation type.")
        return cur_diagrams, zeros_locs, ones_locs

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

    def swap_cells(self, config:np.array, x1:int, y1:int, x2:int, y2:int) -> None:
        """
        Description:
            Swap agents at (x1, y1) to another agent at (x2, y2).

        Arguments:
            config: the current configuration.
            x1: the x-coordinate of the first cell.
            y1: the y-coordinate of the first cell.
            x2: the x-coordinate of the second cell.
            y2: the y-coordinate of the second cell.            

        Return:
            (None)
        """
        temp = copy.deepcopy(config[x1][y1])
        config[x1][y1] = copy.deepcopy(config[x2][y2])
        config[x2][y2] = temp

    def compose_config(self, rails:list=None) -> np.array:
        """
        Description:
            Compose all the rails into the constellation diagram.

        Arguments:
            rails: optionally supplied rails instead of the class' current rails.           

        Return:
            (np.array) an np array representing the constellation diagram.
        """
        constellation_diagram = np.zeros(shape = self.constellation[0].shape, dtype = object)
        if rails:
            cur_rails = rails
        else:
            cur_rails = self.constellation
        for x in range(self.config_dim):
            for y in range(self.config_dim):
                cell_str = ""
                for config in cur_rails:
                    if config[x][y] == -1:
                        cell_str = None
                    else:
                        cell_str += str(config[x][y])
                constellation_diagram[x][y] = cell_str
        return constellation_diagram              

    def calc_gray_score(self, constellation_diagram:np.chararray, n:int=None) -> float:
        """
        Description:
            Calculate the Gray penalty score of the current constellation diagram.

        Arguments:
            constellation_diagram: the constellation diagram.
            n: the number of bits in each constellation point.

        Return:
            (float) the Gray penalty score of the constellation diagram.
        """
        if not n:
            n = self.n
        gray_code_sum = 0
        for x in range(len(constellation_diagram)):
            for y in range(len(constellation_diagram[0])):
                # get current cell's value
                current_code = constellation_diagram[x][y]
                if current_code:
                    # get neigbors
                    neighbors = self.get_neighbors(x, y, 'neumann', 'cut-off')
                    count_neighbors = 0
                    # sum hamming distances from neighbors
                    sum_hd_neighbors = 0
                    for neighbor in neighbors:
                        neighbor_code = constellation_diagram[neighbor[0]][neighbor[1]]
                        if neighbor_code:
                            sum_hd_neighbors += hamming(current_code, neighbor_code)
                            count_neighbors += 1
                    gray_code_sum += (sum_hd_neighbors / count_neighbors)
        return gray_code_sum / (2 ** n)

    def random_swap(self) -> None:
        """
        Description:
            Loop through all current configurations and randomly swap a zero and a one.

        Arguments:
            None

        Return:
            (None)
        """
        for config_idx, config in enumerate(self.constellation):
            nextconfig = copy.deepcopy(config)
            # consider a random swap and keep it if it improves overall satisfaction
            random_zero_idx = np.random.choice(len(self.zeros_locs[config_idx]), size = 1)[0]
            random_one_idx = np.random.choice(len(self.ones_locs[config_idx]), size = 1)[0]
            first_x, first_y = self.zeros_locs[config_idx][random_zero_idx]
            second_x, second_y = self.ones_locs[config_idx][random_one_idx]
            self.swap_cells(nextconfig, first_x, first_y, second_x, second_y)
            self.constellation[config_idx] = nextconfig
            # update list of ones and zeros after swap
            self.zeros_locs[config_idx][random_zero_idx] = (second_x, second_y)
            self.ones_locs[config_idx][random_one_idx] = (first_x, first_y)

    def __str__(self) -> str:
        return 'ID: {}, Fitness: {}'.format(self.ID, self.fitness)