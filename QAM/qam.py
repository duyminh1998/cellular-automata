# Author: Minh Hua
# Date: 09/29/2022
# Purpose: Class that handles the intialization and updates of the QAM model with Cellular Automata.

import numpy as np
import random as rd
import math
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import copy

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

class BaseQAMCA:
    """
    A base class to model and simulate the QAM model using Cellular Automata
    """
    def __init__(self, 
    n:int=4,
    prev_rails:list=None,
    convergence_steps:int=10,
    tolerance:float=0.8, 
    initial_ratio:float=0.5,
    n_type:str='moore',
    boundary_cond:str='periodic',
    rd_seed=None,
    constellation_type:str='square',
    cmap:list=['red', 'blue']) -> None:
        """
        Description:
            Initialize a QAM CA model.

        Arguments:
            n: the number of bits to encode. 2^n = number of constellation points.
            prev_rails: previous rails to be used as constraints.
            convergence_steps: the number of steps to check for convergence.
            tolerance: the tolerance of individuals against others.
            initial_ratio: the initial percentage of one population.
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.
            rd_seed: a random seed to pass to the random number generator. Used to reproduce specific initial configurations.
            constellation_type: the type of constellation. For even n the constellation is usually 'square'. For odd n the constellation is usually 'cross'.            
            cmap: the color map to use for the cells.

        Return:
            (None)
        """
        if rd_seed: # set random seed for reproducible experiments
            rd.seed(rd_seed)
        
        # initialization
        self.n = n
        self.prev_rails = prev_rails
        self.convergence_steps = convergence_steps
        if n % 2 == 0:
            self.constellation_type = 'square'
            self.cmap = matplotlib.colors.ListedColormap(['blue', 'red'])
            self.vmin = 0
            self.config_dim = int(math.sqrt(2 ** self.n))
        else:
            self.constellation_type = 'cross'
            self.cmap = matplotlib.colors.ListedColormap(['blue', 'red', 'white'])
            self.vmin = 0
            self.config_dim = int(math.sqrt(2 ** self.n + (4 * 4 ** ((self.n - 5) / 2))))
        # calculate configuration dimensions
        self.tolerance = tolerance
        self.initial_ratio = initial_ratio
        self.n_type = n_type
        self.boundary_cond = boundary_cond

        # get the subset constraints if we have previous rails for constraints
        if self.prev_rails:
            self.subsets = self.determine_subsets()
            self.subsets_keys = list(self.subsets.keys())
        else:
            self.subsets = None
            self.subsets_keys = None

    def determine_subsets(self) -> dict:
        """
        Description:
            Determine the subsets for which we will be performing the swaps in.

        Arguments:
            None

        Return:
            (dict) a dictionary of all the subsets.
        """
        # concatenate all the rails
        concat_rails = self.compose_config(self.prev_rails)
        # loop through concatenated rails and determine the positions for which the constellation points are the same.
        subsets = {}
        for x in range(len(concat_rails)):
            for y in range(len(concat_rails[x])):
                constellation_point = concat_rails[x][y]
                if constellation_point:
                    if constellation_point not in subsets.keys():
                        subsets[constellation_point] = [(x, y)]
                    else:
                        subsets[constellation_point].append((x, y))
        return subsets

    def get_ones_and_zeros_locs(self, rail:np.array, subset_idxs:list=None) -> tuple:
        """
        Description:
            Get the locations of the ones and zeros in the rail for a certain subset.

        Arguments:
            rail: the current rail:
            subset_idxs: the indices of the constellation points in the subset.

        Return:
            (tuple) the locations of the ones and the zeros.
        """
        zeros_locs = []
        ones_locs = []
        if subset_idxs:
            for (const_pt_x, const_pt_y) in subset_idxs:
                const_pt = rail[const_pt_x][const_pt_y]
                if const_pt == 0:
                    zeros_locs.append((const_pt_x, const_pt_y))
                elif const_pt == 1:
                    ones_locs.append((const_pt_x, const_pt_y))
        else:
            for x in range(self.config_dim):
                for y in range(self.config_dim):
                    if rail[x][y] == 0 :
                        self.zeros_locs.append((x, y))
                    elif rail[x][y] == 1:
                        self.ones_locs.append((x, y))
        return zeros_locs, ones_locs       

    def initialize(self) -> None:
        """
        Description:
            Initialize a configuration for the Schelling Segregation CA model.

        Arguments:
            None

        Return:
            (None)
        """
        # initialize a step counter
        self.step = 0
        self.steps_per_subset = 0
        if self.subsets_keys:
            self.current_subset_idx = self.subsets_keys[0]
            self.current_subset_num_idx = 0
            self.subsets_best_scores = {}
            for key in self.subsets_keys:
                self.subsets_best_scores[key] = []
            self.subsets_converged = 0            
        else:
            self.current_subset_idx = None
            self.current_subset_num_idx = None
            self.subsets_best_scores = None
            self.rail_best_scores = []            
        # Initialize n rails
        if self.constellation_type == 'square':
            self.zeros_locs = []
            self.ones_locs = []              
            self.config = np.ones((self.config_dim, self.config_dim), dtype=np.int8)
            # sample indices for the empty spaces and each population
            # sample randomly if we do not have subsets
            if not self.prev_rails:
                og_idx = [i for i in range(int(self.config_dim ** 2))] # a list of indices from the entire configuration
                first_pop_idx = np.random.choice(og_idx, int(math.floor(self.initial_ratio * len(og_idx))), replace=False) # index for the first population
                # assign levels for the empty spaces and the two populations: -1 = empty space, 0 = first population, 1 = second population
                self.config = self.config.flatten()
                self.config[first_pop_idx] = 0
                self.config = self.config.reshape(self.config_dim, self.config_dim)
            else:
                # assign 50/50 splits of zeros and ones in each subset
                for key in self.subsets_keys:
                    current_subset_locs = self.subsets[key]
                    random_positions = np.random.choice(len(current_subset_locs), size = int(len(current_subset_locs) / 2), replace = False)
                    for random_position in random_positions:
                        rand_x, rand_y = current_subset_locs[random_position][0], current_subset_locs[random_position][1]
                        self.config[rand_x][rand_y] = 0
            # record where the ones and zeros are
            for x in range(self.config_dim):
                for y in range(self.config_dim):
                    if self.config[x][y] == 0:
                        self.zeros_locs.append((x, y))
                    elif self.config[x][y] == 1:
                        self.ones_locs.append((x, y))
            # get the ones and zeros for each subset
            if self.subsets:
                self.subsets_zeros = {}
                self.subsets_ones = {}
                for key in self.subsets_keys:
                    self.subsets_zeros[key], self.subsets_ones[key] = self.get_ones_and_zeros_locs(self.config, self.subsets[key])
            else:
                self.subsets_ones_and_zeros = None                    
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
            self.zeros_locs = []
            self.ones_locs = []
            self.config = np.ones((self.config_dim, self.config_dim), dtype=np.int8)
            # sample indices for the empty spaces and each population
            # sample randomly if we do not have subsets
            if not self.prev_rails:            
                # og_idx = [(i, j) for i, j in zip(range(self.config_dim), range(self.config_dim)) if (i, j) not in empty_coords] # a list of indices from the entire configuration
                first_pop_idx = np.random.choice(og_idx, int(math.floor(self.initial_ratio * len(og_idx))), replace=False) # index for the first population
                # assign levels for the empty spaces and the two populations: -1 = empty space, 0 = first population, 1 = second population
                self.config = self.config.flatten()
                self.config[first_pop_idx] = 0
                self.config[empty_coords_flattened] = 2
                self.config = self.config.reshape(self.config_dim, self.config_dim)
            else:
                self.config = self.config.flatten()
                self.config[empty_coords_flattened] = 2
                self.config = self.config.reshape(self.config_dim, self.config_dim)
                # assign 50/50 splits of zeros and ones in each subset
                for key in self.subsets_keys:
                    current_subset_locs = self.subsets[key]
                    random_positions = np.random.choice(len(current_subset_locs), size = int(len(current_subset_locs) / 2), replace = False)
                    for random_position in random_positions:
                        rand_x, rand_y = current_subset_locs[random_position][0], current_subset_locs[random_position][1]
                        self.config[rand_x][rand_y] = 0
            # record where the ones and zeros are
            for x in range(self.config_dim):
                for y in range(self.config_dim):
                    if (x, y) not in self.empty_coords:
                        if self.config[x][y] == 0 :
                            self.zeros_locs.append((x, y))
                        else:
                            self.ones_locs.append((x, y))
            # get the ones and zeros for each subset
            if self.subsets:
                self.subsets_zeros = {}
                self.subsets_ones = {}
                for key in self.subsets_keys:
                    self.subsets_zeros[key], self.subsets_ones[key] = self.get_ones_and_zeros_locs(self.config, self.subsets[key])
            else:
                self.subsets_ones_and_zeros = None                         
        else:
            raise ValueError("Invalid constellation type.")   

    def get_ID(self, rail:np.array) -> str:
        """
        Description:
            Generate a unique ID for the current rail.

        Arguments:
            rail: the current rail.

        Return:
            (str) the ID of the rail.
        """
        ID = ""
        for x in range(len(rail)):
            for y in range(len(rail[x])):
                ID += str(rail[x][y])
        return ID

    def ID_to_rail(self, ID:str, config_dim:int) -> np.array:
        """
        Description:
            Convert an ID into a rail.

        Arguments:
            ID: the rail's ID.
            config_dim: the dimensions of the rail.

        Return:
            (np.array) the rail.
        """
        rail = np.zeros(shape = (config_dim, config_dim))
        rail = rail.flatten()
        for c_idx in range(len(ID)):
            rail[c_idx] = int(ID[c_idx])
        rail = rail.reshape(config_dim, config_dim)
        return rail                            

    def observe(self) -> None:
        """
        Description:
            Call matplotlib to draw the CA configuration.

        Arguments:
            None

        Return:
            (None)
        """
        if self.prev_rails:
            cur_diagrams = self.prev_rails + [self.config]
        else:
            cur_diagrams = [self.config]
        for rail_idx, rail in enumerate(cur_diagrams):
            plt.subplot(1, self.n, rail_idx + 1)
            plt.cla()
            # plt.imshow(config, vmin = 0, vmax = 1, cmap = plt.cm.binary)
            plt.imshow(rail, vmin = self.vmin, vmax = 2, cmap = self.cmap)
            percent_satisfied = self.sum_satisfaction(rail) / (2 ** self.n)
            plt.title('Rail {} \nSatisfied \n= {}'.format(rail_idx + 1, percent_satisfied))
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
            if neighbor != 2 and neighbor == current_agent:
                count_similar += 1
            if neighbor != 2:
                total_n += 1
        # if the agent is isolated with no neighbors
        if total_n == 0:
            return 1
        if (count_similar / total_n) >= self.tolerance:
            return 1 # satisfied
        return 0 # dissastified

    def update(self) -> None:
        """
        Description:
            Loop through all current configurations and randomly choose an agent to check for satisfaction.

        Arguments:
            None

        Return:
            (None)
        """
        self.step = self.step + 1

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
                if config[x][y] != 2: # if we have an agent
                    total_satisfaction += self.determine_satisfaction(config, x, y)
        return total_satisfaction

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
        constellation_diagram = np.zeros(shape = (self.config_dim, self.config_dim), dtype = object)
        if type(rails) != None:
            cur_rails = rails
        elif self.prev_rails:
            cur_rails = self.prev_rails
        else:
            cur_rails = [self.config]
        for x in range(self.config_dim):
            for y in range(self.config_dim):
                cell_str = ""
                for config in cur_rails:
                    if config[x][y] == 2:
                        cell_str = None
                    else:
                        cell_str += str(config[x][y])
                constellation_diagram[x][y] = cell_str
        return constellation_diagram

    def calc_gray_score(self, constellation_diagram:np.chararray, n:int) -> float:
        """
        Description:
            Calculate the Gray penalty score of the current constellation diagram.

        Arguments:
            constellation_diagram: the constellation diagram.
            n: the number of bits in each constellation point.

        Return:
            (float) the Gray penalty score of the constellation diagram.
        """
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

    def count_repeats(self, constellation_diagram) -> int:
        """
        Description:
            Count the number of repeated constellation points in the constellation diagram.

        Arguments:
            constellation_diagram: the constellation diagram.

        Return:
            (int) the number of repeated constellation points.
        """        
        const_dic = {}
        for const_pt_x in range(len(constellation_diagram)):
            for const_pt_y in range(len(constellation_diagram[const_pt_x])):
                const_pt = constellation_diagram[const_pt_x][const_pt_y]
                if const_pt:
                    if const_pt not in const_dic.keys():
                        const_dic[const_pt] = 1
                    else:
                        const_dic[const_pt] = const_dic[const_pt] + 1
        return sum(1 for v in const_dic.values() if v > 1)

    def determine_xor(self, rail1:np.array, rail2:np.array) -> int:
        """
        Description:
            Determine the XOR of two rails.

        Arguments:
            rail1: the first rail.
            rail2: the second rail.

        Return:
            (int) the sum of the XOR.
        """    
        xor = 0
        for x in range(self.config_dim):
            for y in range(self.config_dim):
                if rail1[x][y] != 2 and rail2[x][y] != 2:
                    xor += (rail1[x][y] ^ rail2[x][y])
        return xor

    def batch_xor(self, rails:list, cur_rail:np.array) -> list:
        """
        Description:
            Determine the XOR of the current rail and the fixed rails.

        Arguments:
            rails: a list of rails that are fixed.
            cur_rail: the current rail.

        Return:
            (list) a list of the XOR'd rails.
        """    
        xor_scores = []
        for rail in rails:
            xor_score = self.determine_xor(rail, cur_rail) / (2 ** self.n)
            xor_scores.append(xor_score)
        return xor_scores

    def calc_xor_error(self, xor_scores:list) -> float:
        """
        Description:
            Calculate MSE(xor_scores, 0.5).

        Arguments:
            xor_scores: a list of XOR scores.

        Return:
            (float) the mean squared difference between the XOR scores and 0.5.
        """
        return (sum((score - 0.5) ** 2 for score in xor_scores) / len(xor_scores))

class SAHCQAMCA(BaseQAMCA):
    """
    A class to model and simulate the QAM model using Cellular Automata with a steepest ascent hill climbing update rule
    """
    def __init__(self, 
    n:int=4, 
    prev_rails:list=None,
    convergence_steps:int=10,    
    tolerance:float=0.8, 
    initial_ratio:float=0.5,
    n_type:str='moore',
    boundary_cond:str='periodic',
    rd_seed=None,
    constellation_type:str='square',
    cmap:list=['red', 'blue'],
    print_debug:bool=False) -> None:
        """
        Description:
            Initialize a QAM CA model.

        Arguments:
            n: the number of bits to encode. 2^n = number of constellation points.
            prev_rails: previous rails to be used as constraints.
            convergence_steps: the number of steps to check for convergence.            
            tolerance: the tolerance of individuals against others.
            initial_ratio: the initial percentage of one population.
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.
            rd_seed: a random seed to pass to the random number generator. Used to reproduce specific initial configurations.
            constellation_type: the type of constellation. For even n the constellation is usually 'square'. For odd n the constellation is usually 'cross'.            
            cmap: the color map to use for the cells.
            print_debug: whether to print debug information.

        Return:
            (None)
        """
        super(SAHCQAMCA, self).__init__(n, prev_rails, convergence_steps, tolerance, initial_ratio, n_type, boundary_cond, rd_seed, constellation_type, cmap)
        self.print_debug = print_debug
    
    def update(self):
        """
        Description:
            Loop through all current subsets and randomly choose an agent to check for satisfaction.

        Arguments:
            None

        Return:
            (None)
        """
        # if we have previous rails constraints
        # loop through all the subsets and record the value of all possible swaps
        if self.prev_rails:
            cur_subset_zeros, cur_subset_ones = self.subsets_zeros[self.current_subset_idx], self.subsets_ones[self.current_subset_idx]
            if self.print_debug:
                print("Current subset index: {}".format(self.current_subset_idx))
                print("Current subset indices: {}".format(self.subsets[self.current_subset_idx]))
                print("Current subset zeros: {}".format(cur_subset_zeros))
                print("Current subset ones: {}".format(cur_subset_ones))
        else:
            # else we do not have previous rails constraints
            cur_subset_zeros, cur_subset_ones = self.zeros_locs, self.ones_locs
            
        swap_values = {}
        nextconfig = copy.deepcopy(self.config)
        current_satisfaction = self.sum_satisfaction(self.config) / (2 ** self.n)
        current_rail_gray = self.calc_gray_score(self.compose_config([self.config]), self.n)
        # consider all possible swaps and record the value
        for one_loc_idx, (one_loc_x, one_loc_y) in enumerate(cur_subset_ones):
            for zero_loc_idx, (zero_loc_x, zero_loc_y) in enumerate(cur_subset_zeros):
                # try out the swap
                self.swap_cells(nextconfig, one_loc_x, one_loc_y, zero_loc_x, zero_loc_y)
                # record the new satisfaction
                new_satisfaction = self.sum_satisfaction(nextconfig) / (2 ** self.n)
                new_rail_gray = self.calc_gray_score(self.compose_config([nextconfig]), self.n)
                # swap_values['{}_{}'.format(one_loc_idx, zero_loc_idx)] = new_satisfaction
                swap_values['{}_{}'.format(one_loc_idx, zero_loc_idx)] = new_rail_gray
                # swap back the cells
                self.swap_cells(nextconfig, one_loc_x, one_loc_y, zero_loc_x, zero_loc_y)
                # if self.print_debug:
                #     print("Current satisfaction: {}, New satisfaction: {}".format(current_satisfaction, new_satisfaction))
        
        # find the swap with the max increase
        if len(swap_values) > 0:
            # max_swap = max(swap_values.values())
            max_swap = min(swap_values.values())
            # if max_swap > current_satisfaction:
            if max_swap < current_rail_gray:
                # max_swap_key = max(swap_values, key=swap_values.get)
                max_swap_key = min(swap_values, key=swap_values.get)
                max_one_idx, max_zero_idx = int(max_swap_key.split('_')[0]), int(max_swap_key.split('_')[1])
                first_x, first_y = cur_subset_zeros[max_zero_idx]
                second_x, second_y = cur_subset_ones[max_one_idx]
                self.swap_cells(nextconfig, first_x, first_y, second_x, second_y)
                self.config = nextconfig
                # update list of ones and zeros after swap
                cur_subset_zeros[max_zero_idx] = (second_x, second_y)
                cur_subset_ones[max_one_idx] = (first_x, first_y)                
            
        if self.print_debug:
            if self.prev_rails:
                constellation_diagram = self.compose_config(rails = self.prev_rails + [self.config])
            else:
                constellation_diagram = self.compose_config(rails = [self.config])
            print(constellation_diagram)
            print("Gray score: {}".format(self.calc_gray_score(constellation_diagram, self.n)))
        
        self.step = self.step + 1
        self.steps_per_subset += 1

        # check for convergence in the subsets
        if self.prev_rails:
            self.subsets_best_scores[self.current_subset_idx].append(max_swap)
            if len(self.subsets_best_scores[self.current_subset_idx]) >= self.convergence_steps:
                # check to see if the best fitness has changed enough from the average of the past window
                moving_average = sum(score for score in self.subsets_best_scores[self.current_subset_idx][-self.convergence_steps:]) / self.convergence_steps
                if self.print_debug:
                    print("Average best fitness of the past {} generations: {}".format(self.convergence_steps, moving_average))
                if abs(self.subsets_best_scores[self.current_subset_idx][-1] - moving_average) <= 10**-3:
                    # we can stop early
                    if self.print_debug:
                        print("Moving to next subset.")
                    # check to see if there are no longer any improvements to be made across all subsets
                    if self.steps_per_subset == 1:
                        self.subsets_converged += 1
                        if self.subsets_converged == len(self.prev_rails) + 1:
                            if self.print_debug:
                                print("Every rail has converged.")
                            return self.config
                    else:
                        self.subsets_converged = 0
                    self.current_subset_num_idx = (self.current_subset_num_idx + 1) % len(self.subsets_keys)
                    self.current_subset_idx = self.subsets_keys[self.current_subset_num_idx]
                    self.steps_per_subset = 0
        # check for convergence in a single rail
        else:
            self.rail_best_scores.append(max_swap)
            if len(self.rail_best_scores) >= self.convergence_steps:
                # check to see if the best fitness has changed enough from the average of the past window
                moving_average = sum(score for score in self.rail_best_scores[-self.convergence_steps:]) / self.convergence_steps
                if self.print_debug:
                    print("Average best fitness of the past {} generations: {}".format(self.convergence_steps, moving_average))
                if abs(self.rail_best_scores[-1] - moving_average) <= 10**-3:
                    # we can stop early
                    if self.print_debug:
                        print("Stopping early.")
                    # check to see if there are no longer any improvements to be made across all subsets
                    return self.config
        return None

    def fixed_update(self):
        """
        Description:
            Update one configuration at a time.

        Arguments:
            None

        Return:
            (None)
        """
        if self.current_config_idx < self.n:
            # consider only the current configuration until convergence
            # check XOR condition
            fixed_rails = self.cur_diagrams[:self.current_config_idx]
            config = self.cur_diagrams[self.current_config_idx]
            config_idx = self.current_config_idx
            current_xor_error = self.calc_xor_error(self.batch_xor(fixed_rails, config))
            current_satisfaction = self.sum_satisfaction(config) / (2 ** self.n)
            if self.print_debug:
                # print("{}".format(self.batch_xor(fixed_rails, config)))
                current_constellation = self.compose_config()
                current_gray_score = self.calc_gray_score(current_constellation, self.n)
                current_repeats = self.count_repeats(current_constellation)

            # if xor condition is not met
            if (current_xor_error > 0.0 or current_satisfaction < 1.0) and not self.cur_swap_per_rail > self.max_swaps_per_rail:
                nextconfig = copy.deepcopy(config)
                swap_sat_vals = {}
                swap_xor_err_vals = {}
                # consider all possible swaps and record the value
                for one_loc_idx, (one_loc_x, one_loc_y) in enumerate(self.ones_locs[config_idx]):
                    for zero_loc_idx, (zero_loc_x, zero_loc_y) in enumerate(self.zeros_locs[config_idx]):
                        # try out the swap
                        self.swap_cells(nextconfig, one_loc_x, one_loc_y, zero_loc_x, zero_loc_y)
                        # record the new satisfaction
                        new_satisfaction = self.sum_satisfaction(nextconfig) / (2 ** self.n)
                        new_xor_error = self.calc_xor_error(self.batch_xor(fixed_rails, nextconfig))
                        swap_sat_vals['{}_{}'.format(one_loc_idx, zero_loc_idx)] = new_satisfaction
                        swap_xor_err_vals['{}_{}'.format(one_loc_idx, zero_loc_idx)] = new_xor_error
                        # swap back the cells
                        self.swap_cells(nextconfig, one_loc_x, one_loc_y, zero_loc_x, zero_loc_y)

                # find the swap with the max increase in satisfaction or decrease in XOR error
                max_swap_by_sat = max(swap_sat_vals.values())
                min_swap_by_xor_err = min(swap_xor_err_vals.values())
                # randomly choose one gradient to follow
                if max_swap_by_sat > current_satisfaction and min_swap_by_xor_err < current_xor_error:
                    choice = np.random.choice([0, 1], size = 1, p = [0.4, 0.6])[0]
                    if choice == 0: # follow satisfaction gradient
                        swap_key = max(swap_sat_vals, key=swap_sat_vals.get)
                    else: # follow XOR error gradient
                        swap_key = min(swap_xor_err_vals, key=swap_xor_err_vals.get)
                # follow the satisfaction gradient
                elif max_swap_by_sat > current_satisfaction:
                    swap_key = max(swap_sat_vals, key=swap_sat_vals.get)
                # follow the XOR error gradient
                elif min_swap_by_xor_err < current_xor_error:
                    swap_key = min(swap_xor_err_vals, key=swap_xor_err_vals.get)
                # perform the swap
                swap_one_idx, swap_zero_idx = int(swap_key.split('_')[0]), int(swap_key.split('_')[1])
                first_x, first_y = self.zeros_locs[config_idx][swap_zero_idx]
                second_x, second_y = self.ones_locs[config_idx][swap_one_idx]
                self.swap_cells(nextconfig, first_x, first_y, second_x, second_y)
                self.cur_diagrams[config_idx] = nextconfig
                # update list of ones and zeros after swap
                self.zeros_locs[config_idx][swap_zero_idx] = (second_x, second_y)
                self.ones_locs[config_idx][swap_one_idx] = (first_x, first_y)  
                
                # print debug if necessary
                if self.print_debug:
                    new_constellation = self.compose_config()
                    new_gray_score = self.calc_gray_score(new_constellation, self.n)
                    new_repeats = self.count_repeats(new_constellation)
                    print("Rail: {}, Old Gray: {}, New Gray: {}, Old repeats: {}, New repeats: {}, Old Sat: {}, New Sat: {}, Old XOR: {}, New XOR: {}".format(config_idx, 
                    current_gray_score, new_gray_score, current_repeats, new_repeats, current_satisfaction, max_swap_by_sat, current_xor_error, min_swap_by_xor_err))
            else:
                self.current_config_idx += 1
                self.cur_swap_per_rail = 0
                if self.print_debug:
                    print("Rail: {}, Final Gray: {}, Final repeats: {}, Final Sat: {}, Final XOR: {}".format(config_idx, current_gray_score, current_repeats, current_satisfaction, current_xor_error))
                    print("Moving on to rail {}".format(self.current_config_idx))
            self.step = self.step + 1
            self.cur_swap_per_rail += 1
        else:
            final_constellation = self.compose_config()
            final_gray_score = self.calc_gray_score(final_constellation, self.n)
            final_repeats = self.count_repeats(final_constellation)
            print("Final gray score: {}, Final repeats: {}".format(final_gray_score, final_repeats))
            print("{}".format(final_constellation))
            raise ValueError("Terminated.")