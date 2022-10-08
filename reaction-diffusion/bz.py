# Author: Minh Hua
# Date: 09/19/2022
# Purpose: Class that handles the intialization and updates of the Belousov-Zhabotinsky (BZ) reaction model with Cellular Automata.

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.colors
from pylab import *
from lib.utils import *

class BZCA:
    """
    A class to model and simulate the Belousov-Zhabotinsky (BZ) reaction model using Cellular Automata
    """
    def __init__(self, epsilon:float, q:float, f:float, Dh:float, Dt:float=0.02, Du:float=0.00001, Dv:float=0.00001, n:int=100, lo:float=0.0, hi:float=1.0, n_type:str='moore', boundary_cond:str='periodic', rd_seed=None) -> None:
        """
        Description:
            Initialize a Belousov-Zhabotinsky (BZ) reaction CA model.

        Arguments:
            epsilon: parameter in the BZ equation.
            q: parameter in the BZ equation.
            f: parameter in the BZ equation.
            Dh: spatial resolution.
            Dt: temporal resolution. Default is 0.02.
            Du: diffusion constant of u. Default is 10e-5.
            Dv: diffusion constant of v. Default is 10e-5.
            n: the dimension of the board. Equivalent to generating n x n spaces.
            lo: the lower bound for the range of values u and v can take on.
            hi: the upper bound for the range of values u and v can take on.
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
        self.epsilon = epsilon
        self.q = q
        self.f = f
        self.Dh = Dh
        self.Dt = Dt
        self.Du = Du
        self.Dv = Dv
        self.n_type = n_type
        self.boundary_cond = boundary_cond
        self.lo = lo
        self.hi = hi
        self.name = 'BZ'

    def initialize(self) -> None:
        """
        Description:
            Initialize a configuration for the Belousov-Zhabotinsky (BZ) reaction CA model.

        Arguments:
            (None)

        Return:
            (None)
        """
        # variables for CA configurations
        # u = np.random.uniform(self.lo, self.hi, (self.n, self.n))
        u = np.zeros((self.n, self.n), dtype=np.float)
        v = np.random.uniform(self.lo, self.hi, (self.n, self.n))
        nextu = np.zeros((self.n, self.n), dtype=np.float)
        nextv = np.zeros((self.n, self.n), dtype=np.float)
        # initialize with ones and add small noise
        for x in range(self.n):
            for y in range(self.n):
                if (x > self.n/2) and (y < self.n/2 + 1) and (y > self.n/2 - 1):
                    u[x][y] = self.hi
                # v[x][y] = self.hi
        # save the configurations
        self.u = u
        self.v = v
        self.nextu = nextu
        self.nextv = nextv

    def observe(self) -> None:
        """
        Description:
            Call matplotlib to draw the CA configuration.

        Arguments:
            None

        Return:
            (None)
        """
        plt.subplot(1, 2, 1)
        plt.cla()
        plt.imshow(self.u, vmin = self.lo, vmax = self.hi, cmap = plt.cm.binary)
        plt.title('u')
        plt.subplot(1, 2, 2)
        plt.cla()
        plt.imshow(self.v, vmin = self.lo, vmax = self.hi, cmap = plt.cm.binary)
        plt.title('v')        
        plt.show()

    def update(self) -> None:
        """
        Description:
            Update the simulation.

        Arguments:
            None

        Return:
            (None)
        """
        for x in range(self.n):
            for y in range(self.n):
                # state-transition function
                uC, uR, uL, uU, uD = self.u[x][y], self.u[(x+1)%self.n][y], self.u[(x-1)%self.n][y], self.u[x][(y+1)%self.n], self.u[x][(y-1)%self.n]
                uUL, uUR, uDL, uDR = self.u[(x-1)%self.n][(y+1)%self.n], self.u[(x+1)%self.n][(y+1)%self.n], self.u[(x-1)%self.n][(y-1)%self.n], self.u[(x+1)%self.n][(y-1)%self.n]
                vC, vR, vL, vU, vD = self.v[x][y], self.v[(x+1)%self.n][y], self.v[(x-1)%self.n][y], self.v[x][(y+1)%self.n], self.v[x][(y-1)%self.n]
                vUL, vUR, vDL, vDR = self.v[(x-1)%self.n][(y+1)%self.n], self.v[(x+1)%self.n][(y+1)%self.n], self.v[(x-1)%self.n][(y-1)%self.n], self.v[(x+1)%self.n][(y-1)%self.n]
                uLap = (uR + uL + uU + uD - 4 * uC) / (self.Dh ** 2)
                vLap = (vR + vL + vU + vD - 4 * vC) / (self.Dh ** 2)
                # uLap = (uR + uL + uU + uD + uUL + uUR + uDL + uDR - 8 * uC) / (self.Dh ** 2)
                # vLap = (vR + vL + vU + vD + vUL + vUR + vDL + vDR - 8 * vC) / (self.Dh ** 2)
                self.nextu[x][y] = uC + (((uC * (1 - uC) - ((uC - self.q) / (uC + self.q)) * self.f * vC) / self.epsilon + self.Du * uLap)) * self.Dt
                self.nextv[x][y] = vC + (uC - vC + self.Dv * vLap) * self.Dt
        # update configurations
        self.u, self.nextu = self.nextu, self.u
        self.v, self.nextv = self.nextv, self.v

class NPBZCA(BZCA):
    """
    A class to model and simulate the Belousov-Zhabotinsky (BZ) reaction model using Cellular Automata

    Implementation using numpy provided by Loic Gouarin. Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
    """
    def __init__(self, epsilon:float, q:float, f:float, Dh:float, Dt:float=0.02, Du:float=0.00001, Dv:float=0.00001, n:int=100, lo:float=0.0, hi:float=1.0, n_type:str='moore', boundary_cond:str='periodic', rd_seed=None) -> None:
        """
        Description:
            Initialize a Belousov-Zhabotinsky (BZ) reaction CA model.

        Arguments:
            epsilon: parameter in the BZ equation.
            q: parameter in the BZ equation.
            f: parameter in the BZ equation.
            Dh: spatial resolution.
            Dt: temporal resolution. Default is 0.02.
            Du: diffusion constant of u. Default is 10e-5.
            Dv: diffusion constant of v. Default is 10e-5.
            n: the dimension of the board. Equivalent to generating n x n spaces.
            lo: the lower bound for the range of values u and v can take on.
            hi: the upper bound for the range of values u and v can take on.
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.
            rd_seed: a random seed to pass to the random number generator. Used to reproduce specific initial configurations.

        Return:
            (None)
        """
        super().__init__(epsilon, q, f, Dh, Dt, Du, Dv, n, lo, hi, n_type, boundary_cond, rd_seed)

    def initialize(self, from_file:str=None) -> None:
        """
        Description:
            Initialize a configuration for the Gray-Scott reaction CA model.

        Arguments:
            None

        Return:
            (None)
        """
        # variables for CA configurations
        u = np.random.uniform(0, 1, (self.n + 2, self.n + 2))
        # im = Image.open("../img/sample_CA_init_config_300x300.bmp")
        # p = np.array(im)
        # u = 1 * (p) / (255)
        v = np.random.uniform(0, 1, (self.n + 2, self.n + 2))
        # x, y = np.meshgrid(np.linspace(0, 1, self.n + 2), np.linspace(0, 1, self.n + 2))
        # mask = (y > 0.5) & (x < 0.51) & (x > 0.49)
        # u[mask] = 1
        # v[mask] = 0.25
        # save the configurations
        self.u = u
        self.v = v
    
    def periodic_bc(self, u) -> np.array:
        """
        Description:
            Apply periodic boundary conditions to the configuration.

        Arguments:
            u: one of the chemicals of the current configuration.

        Return:
            (np.array) the input configuration with periodic boundary conditions applied.
        """
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]
        return u

    def diffusion(self, u) -> np.array:
        """
        Description:
            Calculate the Laplacian between a cell and its neighbors.

        Arguments:
            u: one of the chemicals of the current configuration.

        Return:
            (np.array) the Laplacian at each point of u
        """
        # left neighbor, bottom neighbor, top neighbor, left neighbor
        return (u[:-2, 1:-1] + u[1:-1, :-2] - 4*u[1:-1, 1:-1] + u[1:-1, 2:] + u[2:, 1:-1]) / (self.Dh**2)

    def update(self) -> None:
        """
        Description:
            Update the simulation.

        Arguments:
            None

        Return:
            (None)
        """
        self.u = np.clip(self.u, self.lo, self.hi)
        self.v = np.clip(self.v, self.lo, self.hi)
        u, v = self.u[1:-1, 1:-1], self.v[1:-1, 1:-1]
        Lu = self.diffusion(self.u)
        Lv = self.diffusion(self.v)
        u += (self.Du*Lu + (1/self.epsilon)*(u*(1-u) - ((u - self.q)/(u + self.q))*self.f*v))*self.Dt
        v += (self.Dv*Lv + u - v)*self.Dt
        self.u = self.periodic_bc(self.u)
        self.v = self.periodic_bc(self.v)

class NPBZ3CA(BZCA):
    """
    A class to model and simulate the Belousov-Zhabotinsky (BZ) reaction model using Cellular Automata. Three chemical quantities.

    Implementation using numpy provided by Loic Gouarin. Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
    """
    def __init__(self, alpha:float, beta:float, gamma:float, Dh:float, Dt:float=0.02, Du:float=0.00001, Dv:float=0.00001, n:int=100, lo:float=0.0, hi:float=1.0, n_type:str='moore', boundary_cond:str='periodic', rd_seed=None) -> None:
        """
        Description:
            Initialize a Belousov-Zhabotinsky (BZ) reaction CA model.

        Arguments:
            alpha: parameter in the BZ equation.
            beta: parameter in the BZ equation.
            gamma: parameter in the BZ equation.
            Dh: spatial resolution.
            Dt: temporal resolution. Default is 0.02.
            Du: diffusion constant of u. Default is 10e-5.
            Dv: diffusion constant of v. Default is 10e-5.
            n: the dimension of the board. Equivalent to generating n x n spaces.
            lo: the lower bound for the range of values u and v can take on.
            hi: the upper bound for the range of values u and v can take on.
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.
            rd_seed: a random seed to pass to the random number generator. Used to reproduce specific initial configurations.

        Return:
            (None)
        """
        super().__init__(None, None, None, Dh, Dt, Du, Dv, n, lo, hi, n_type, boundary_cond, rd_seed)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.u_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","cyan"])
        self.v_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","blue"])
        self.w_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","magenta"])
        self.uvw_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['cyan', 'blue', 'magenta']) # used to configure the colors of the CA

    def initialize(self, from_file:str=None) -> None:
        """
        Description:
            Initialize a configuration for the Gray-Scott reaction CA model.

        Arguments:
            None

        Return:
            (None)
        """
        # variables for CA configurations
        u = np.random.uniform(0, 1, (self.n + 2, self.n + 2))
        v = np.random.uniform(0, 1, (self.n + 2, self.n + 2))
        w = np.random.uniform(0, 1, (self.n + 2, self.n + 2))
        # save the configurations
        self.u = u
        self.v = v
        self.w = w

    def observe(self) -> None:
        """
        Description:
            Call matplotlib to draw the CA configuration.

        Arguments:
            None

        Return:
            (None)
        """
        plt.subplot(1, 4, 1)
        plt.cla()
        plt.imshow(self.u, vmin = self.lo, vmax = self.hi, cmap = self.u_cmap)
        plt.title('u')
        plt.subplot(1, 4, 2)
        plt.cla()
        plt.imshow(self.v, vmin = self.lo, vmax = self.hi, cmap = self.v_cmap)
        plt.title('v')        
        plt.subplot(1, 4, 3)
        plt.cla()
        plt.imshow(self.w, vmin = self.lo, vmax = self.hi, cmap = self.w_cmap)
        plt.title('w')
        # compute the array combining all three substances
        self.uvw = np.zeros(shape=self.u.shape)
        for x in range(self.n + 2):
            for y in range(self.n + 2):
                substances = [self.u[x][y], self.v[x][y], self.w[x][y]]
                self.uvw[x][y] = substances.index(max(substances))
        plt.subplot(1, 4, 4)
        plt.cla()
        plt.imshow(self.uvw, cmap = self.uvw_cmap)
        plt.title('uvw')
        plt.show()
    
    def periodic_bc(self, u) -> np.array:
        """
        Description:
            Apply periodic boundary conditions to the configuration.

        Arguments:
            u: one of the chemicals of the current configuration.

        Return:
            (np.array) the input configuration with periodic boundary conditions applied.
        """
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]
        if self.n_type == 'moore':
            u[0][0] = u[-2][-2]
            u[0][-1] = u[-2][1]
            u[-1][0] = u[1][-2]
            u[-1][-1] = u[1][1]
        return u

    def diffusion(self, u) -> np.array:
        """
        Description:
            Calculate the average quantity in a cell and its eight neighbors.

        Arguments:
            u: one of the chemicals of the current configuration.

        Return:
            (np.array) the Laplacian at each point of u
        """
        # left neighbor, bottom neighbor, top neighbor, left neighbor
        return (u[:-2, 1:-1] + u[1:-1, :-2] + u[1:-1, 1:-1] + u[1:-1, 2:] + u[2:, 1:-1] + u[:-2, 0:-2] + u[:-2, 2:] + u[2:, 0:-2] + u[2:, 2:]) / 9

    def update(self) -> None:
        """
        Description:
            Update the simulation.

        Arguments:
            None

        Return:
            (None)
        """
        self.u = np.clip(self.u, self.lo, self.hi)
        self.v = np.clip(self.v, self.lo, self.hi)
        self.w = np.clip(self.w, self.lo, self.hi)
        u, v, w = self.u[1:-1, 1:-1], self.v[1:-1, 1:-1], self.w[1:-1, 1:-1]
        Lu = self.diffusion(self.u)
        Lv = self.diffusion(self.v)
        Lw = self.diffusion(self.w)
        u += Lu * (self.alpha * Lv - self.gamma * Lw)
        v += Lv * (self.beta * Lw - self.alpha * Lu)
        w += Lw * (self.gamma * Lu - self.beta * Lv)
        self.u = self.periodic_bc(self.u)
        self.v = self.periodic_bc(self.v)
        self.w = self.periodic_bc(self.w)

if __name__ == '__main__':
    out = 'batch'
    if out == 'single':
        pass
    elif out == 'batch':
        n_frames = 300
        steps_per_frame = 40
        out_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Independent Study\\src\\output\\reaction-diffusion\\bz\\"
        hyperparams = {
            'epsilon': np.linspace(0.1, 1, 10),
            'q': np.linspace(0, 1, 10),
            'f': np.linspace(0, 1.6, 10),
            'Dh': [1],
            'Dt': [1],
            'Du': np.linspace(0.016, 0.16, 3),
            'Dv': np.linspace(0.008, 0.08, 3),
            'n': [100]
        }
        run_loop_reaction_diff_models(NPBZCA, hyperparams, out_path, n_frames, steps_per_frame)