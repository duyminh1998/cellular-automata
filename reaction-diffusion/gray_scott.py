# Author: Minh Hua
# Date: 09/19/2022
# Purpose: Class that handles the intialization and updates of the Gray-Scott reaction model with Cellular Automata.

import numpy as np
import random as rd
import matplotlib.pyplot as plt
from pylab import *
from lib.utils import *

class GrayScottCA:
    """
    A class to model and simulate the Gray-Scott reaction model using Cellular Automata
    """
    def __init__(self, F:float, k:float, Dh:float, Dt:float=0.02, Du:float=0.00002, Dv:float=0.00001, n:int=100, lo:float=0.0, hi:float=1.0, n_type:str='moore', boundary_cond:str='periodic', rd_seed=None) -> None:
        """
        Description:
            Initialize a Gray-Scott reaction CA model.

        Arguments:
            F: parameter in the Gray-Scott equation.
            k: parameter in the Gray-Scott equation.
            Dh: spatial resolution.
            Dt: temporal resolution. Default is 0.02.
            Du: diffusion constant of u. Default is 2 x 10e-5.
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
        self.F = F
        self.k = k
        self.Dh = Dh
        self.Dt = Dt
        self.Du = Du
        self.Dv = Dv
        self.n_type = n_type
        self.boundary_cond = boundary_cond
        self.lo = lo
        self.hi = hi
        self.name = 'GS'

    def initialize(self) -> None:
        """
        Description:
            Initialize a configuration for the Gray-Scott reaction CA model.

        Arguments:
            None

        Return:
            (None)
        """
        # variables for CA configurations
        # u = np.random.uniform(self.lo, self.hi, (self.n, self.n))
        u = np.ones((self.n, self.n), dtype=np.float)
        v = np.zeros((self.n, self.n), dtype=np.float)
        nextu = np.zeros((self.n, self.n), dtype=np.float)
        nextv = np.zeros((self.n, self.n), dtype=np.float)
        # initialize the center with (u, v) = (0, 1)
        # u[math.floor(self.n/2)][math.floor(self.n/2)] = 0
        # v[math.floor(self.n/2)][math.floor(self.n/2)] = 1
        x, y = np.meshgrid(np.linspace(0, 1, self.n), np.linspace(0, 1, self.n))
        mask = (0.4 < x) & (x < 0.6) & (0.4 < y) & (y < 0.6)
        # initialize the center with (u, v) = (0, 1)
        u[mask] = 0.5
        v[mask] = 0.25
        nextu[mask] = 0.5
        nextv[mask] = 0.25
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
                self.nextu[x][y] = uC + (self.F * (1 - uC) - uC * (vC**2) + self.Du * uLap) * self.Dt
                self.nextv[x][y] = vC + (-(self.F + self.k)*vC + uC * (vC**2) + self.Dv * vLap) * self.Dt
        # update configurations
        self.u, self.nextu = self.nextu, self.u
        self.v, self.nextv = self.nextv, self.v

class NPGrayScottCA(GrayScottCA):
    """
    A class to model and simulate the Gray-Scott reaction model using Cellular Automata.

    Implementation using numpy provided by Loic Gouarin. Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
    """
    def __init__(self, F:float, k:float, Dh:float, Dt:float=0.02, Du:float=0.00002, Dv:float=0.00001, n:int=100, lo:float=0.0, hi:float=1.0, n_type:str='moore', boundary_cond:str='periodic', rd_seed=None) -> None:
        """
        Description:
            Initialize a Gray-Scott reaction CA model.

        Arguments:
            F: parameter in the Gray-Scott equation.
            k: parameter in the Gray-Scott equation.
            Dh: spatial resolution.
            Dt: temporal resolution. Default is 0.02.
            Du: diffusion constant of u. Default is 2 x 10e-5.
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
        super().__init__(F, k, Dh, Dt, Du, Dv, n, lo, hi, n_type, boundary_cond, rd_seed)

    def initialize(self) -> None:
        """
        Description:
            Initialize a configuration for the Gray-Scott reaction CA model.

        Arguments:
            None

        Return:
            (None)
        """
        # variables for CA configurations
        u = np.ones((self.n + 2, self.n + 2), dtype=float)
        v = np.zeros((self.n + 2, self.n + 2), dtype=float)
        x, y = np.meshgrid(np.linspace(0, 1, self.n + 2), np.linspace(0, 1, self.n + 2))
        mask = (0.5 < x) & (x < 0.51) & (0.5 < y) & (y < 0.51)
        u[mask] = 0.5
        v[mask] = 0.25
        # u = np.random.uniform(0, 1, (self.n + 2, self.n + 2))
        # v = np.random.uniform(0, 1, (self.n + 2, self.n + 2))
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
        # return (u[:-2, 1:-1] + u[1:-1, :-2] - 4*u[1:-1, 1:-1] + u[1:-1, 2:] + u[2:, 1:-1])
        # moore neighborhood
        return (u[:-2, 1:-1] + u[1:-1, :-2] - 8*u[1:-1, 1:-1] + u[1:-1, 2:] + u[2:, 1:-1] + u[:-2, :-2] + u[:-2, 2:] + u[2:, :-2] + u[2:, 2:])

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
        uvv = u*v*v
        u += self.Du*Lu - uvv + self.F*(1-u)
        v += self.Dv*Lv + uvv - (self.F + self.k)*v
        self.u = self.periodic_bc(self.u)
        self.v = self.periodic_bc(self.v)

if __name__ == "__main__":
    out = 'batch'
    if out == 'single':
        n = 500 # size of grid: n * n
        Dh = 1 # spatial resolution, assuming space is [0,1] * [0,1]
        Dt = 1 # temporal resolution
        F, k = 0.0545, 0.062 # parameter values
        Du = 0.1 # diffusion constant of u
        Dv = 0.05 # diffusion constant of v
        lo = 0
        hi = 1
        n_type = 'moore'
        boundary_cond = 'cut-off'
        r_seed = 42
        model = NPGrayScottCA(F, k, Dh, Dt, Du, Dv, n, lo, hi, n_type, boundary_cond, r_seed)
        model.initialize()
        frames = create_frames(500, model, 40)

        import imageio
        # frames_scaled = [np.uint8(frame) for frame in frames]
        out_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Independent Study\\src\\output\\reaction-diffusion\\"
        imageio.mimsave(out_path + 'GS_n_{}_F_{}_k_{}_Du_{}_Dv_{}.gif'.format(n, F, k, Du, Dv), frames, format='gif', fps=60)
    elif out == 'batch':
        n_frames = 100
        steps_per_frame = 40
        out_path = "E:\\VAULT 419\\Files\\School\\JHU Archive\\Fall 2022\\Independent Study\\src\\output\\reaction-diffusion\\"
        hyperparams = {
            'F': np.linspace(0.08, 0.5, 10),
            'k': np.linspace(0.01, 0.5, 10),
            'Dh': [1],
            'Dt': [1],
            'Du': np.linspace(0.016, 0.16, 4),
            'Dv': np.linspace(0.008, 0.08, 4),
            'n': [300]
        }
        run_loop_reaction_diff_models(NPGrayScottCA, hyperparams, out_path, n_frames, steps_per_frame)