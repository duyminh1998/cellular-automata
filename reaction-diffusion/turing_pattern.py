# Author: Minh Hua (adapted from Hiroki Sayama)
# Date: 09/19/2022
# Purpose: Class that handles the intialization and updates of the Turing Pattern model with Cellular Automata.

import numpy as np
import random as rd
import pickle
import matplotlib.pyplot as plt
from pylab import *

class TuringPatternCA:
    """
    A class to model and simulate the Turing Pattern model using Cellular Automata
    """
    def __init__(self, a:float, b:float, c:float, d:float, h:float, k:float, Dh:float, Dt:float=0.02, Du:float=0.0001, Dv:float=0.0006, n:int=100, n_type:str='moore', boundary_cond:str='periodic', rd_seed=None) -> None:
        """
        Description:
            Initialize a Turing Pattern CA model.

        Arguments:
            a: parameter in Turing's equation.
            b: parameter in Turing's equation.
            c: parameter in Turing's equation.
            d: parameter in Turing's equation.
            h: parameter in Turing's equation.
            k: parameter in Turing's equation.
            Dh: spatial resolution.
            Dt: temporal resolution. Default is 0.02.
            Du: diffusion constant of u. Default is 0.0001.
            Dv: diffusion constant of v. Default is 0.0006.
            n: the dimension of the board. Equivalent to generating n x n spaces.
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
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.h = h
        self.k = k
        self.Dh = Dh
        self.Dt = Dt
        self.Du = Du
        self.Dv = Dv
        self.n_type = n_type
        self.boundary_cond = boundary_cond
        self.name = 'TP'

    def initialize(self, lo:float=-0.03, hi:float=0.03) -> None:
        """
        Description:
            Initialize a configuration for the Turing Pattern CA model.

        Arguments:
            lo: lower bound for noise.
            hi: upper bound for noise.

        Return:
            (None)
        """
        # variables for CA configurations
        u = np.ones((self.n, self.n), dtype=np.float)
        v = np.ones((self.n, self.n), dtype=np.float)
        nextu = np.zeros((self.n, self.n), dtype=np.float)
        nextv = np.zeros((self.n, self.n), dtype=np.float)
        # initialize with ones and add small noise
        for x in range(self.n):
            for y in range(self.n):
                u[x][y] = u[x][y] + uniform(lo, hi)
                v[x][y] = v[x][y] + uniform(lo, hi)
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
        plt.imshow(self.u, vmin = 0, vmax = 2, cmap = plt.cm.binary)
        plt.title('u')
        plt.subplot(1, 2, 2)
        plt.cla()
        plt.imshow(self.v, vmin = 0, vmax = 2, cmap = plt.cm.binary)
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
                vC, vR, vL, vU, vD = self.v[x][y], self.v[(x+1)%self.n][y], self.v[(x-1)%self.n][y], self.v[x][(y+1)%self.n], self.v[x][(y-1)%self.n]
                uLap = (uR + uL + uU + uD - 4 * uC) / (self.Dh ** 2)
                vLap = (vR + vL + vU + vD - 4 * vC) / (self.Dh ** 2)
                self.nextu[x][y] = uC + (self.a * (uC - self.h) + self.b * (vC - self.k) + self.Du * uLap) * self.Dt
                self.nextv[x][y] = vC + (self.c * (uC - self.h) + self.d * (vC - self.k) + self.Dv * vLap) * self.Dt
        # update configurations
        self.u, self.nextu = self.nextu, self.u
        self.v, self.nextv = self.nextv, self.v

class NPTuringPatternCA(TuringPatternCA):
    """
    A class to model and simulate the Turing Pattern model using Cellular Automata.

    Implementation using numpy provided by Loic Gouarin. Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
    """
    def __init__(self, a:float, b:float, c:float, d:float, h:float, k:float, Dh:float, Dt:float=0.02, Du:float=0.0001, Dv:float=0.0006, n:int=100, n_type:str='moore', boundary_cond:str='periodic', rd_seed=None) -> None:
        """
        Description:
            Initialize a Turing Pattern CA model.

        Arguments:
            a: parameter in Turing's equation.
            b: parameter in Turing's equation.
            c: parameter in Turing's equation.
            d: parameter in Turing's equation.
            h: parameter in Turing's equation.
            k: parameter in Turing's equation.
            Dh: spatial resolution.
            Dt: temporal resolution. Default is 0.02.
            Du: diffusion constant of u. Default is 0.0001.
            Dv: diffusion constant of v. Default is 0.0006.
            n: the dimension of the board. Equivalent to generating n x n spaces.
            n_type: the type of neighborhood. Currently supports 'moore' and 'neumann'.
            boundary_cond: the boundary conditions. Currenty supports 'cut-off' and 'periodic'.
            rd_seed: a random seed to pass to the random number generator. Used to reproduce specific initial configurations.

        Return:
            (None)
        """
        super().__init__(a, b, c, d, h, k, Dh, Dt, Du, Dv, n, n_type, boundary_cond, rd_seed)
        

    def initialize(self, lo:float=-0.03, hi:float=0.03) -> None:
        """
        Description:
            Initialize a configuration for the Gray-Scott reaction CA model.

        Arguments:
            lo: lower bound for noise.
            hi: upper bound for noise.

        Return:
            (None)
        """
        # variables for CA configurations
        u = np.ones((self.n + 2, self.n + 2), dtype=np.float)
        v = np.ones((self.n + 2, self.n + 2), dtype=np.float)
        # initialize with ones and add small noise
        for x in range(self.n):
            for y in range(self.n):
                u[x][y] = u[x][y] + uniform(lo, hi)
                v[x][y] = v[x][y] + uniform(lo, hi)
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
        u, v = self.u[1:-1, 1:-1], self.v[1:-1, 1:-1]
        Lu = self.diffusion(self.u)
        Lv = self.diffusion(self.v)
        u += (self.a * (u - self.h) + self.b * (v - self.k) + self.Du * Lu) * self.Dt
        v += (self.c * (u - self.h) + self.d * (v - self.k) + self.Dv * Lv) * self.Dt
        self.u = self.periodic_bc(self.u)
        self.v = self.periodic_bc(self.v)