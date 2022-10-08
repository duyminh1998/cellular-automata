from turing_pattern import TuringPatternCA, NPTuringPatternCA
from bz import BZCA, NPBZCA, NPBZ3CA
from gray_scott import GrayScottCA, NPGrayScottCA
from lib import pycxsimulator
import matplotlib
matplotlib.use('TkAgg')

simulation = 2

if simulation == 0:
    n = 100 # size of grid: n * n
    Dh = 1. / n # spatial resolution, assuming space is [0,1] * [0,1]
    Dt = 0.02 # temporal resolution
    a, b, c, d, h, k = 1., -1., 2., -1.5, 1., 1. # parameter values
    Du = 0.0001 # diffusion constant of u
    Dv = 0.0006 # diffusion constant of v
    n_type = 'moore'
    boundary_cond = 'cut-off'
    r_seed = 42
    model = NPTuringPatternCA(a, b, c, d, h, k, Dh, Dt, Du, Dv, n, n_type, boundary_cond, r_seed)
elif simulation == 1:
    n = 300 # size of grid: n * n
    Dh = 0.125 # spatial resolution
    Dt = 0.001 # temporal resolution
    epsilon, q, f = 0.2, 10**-3, 1.0 # parameter values
    Du = 10**-5 # diffusion constant of u
    Dv = 10**-5 # diffusion constant of v
    lo = 0
    hi = 1
    n_type = 'moore'
    boundary_cond = 'cut-off'
    r_seed = 42
    model = NPBZCA(epsilon, q, f, Dh, Dt, Du, Dv, n, lo, hi, n_type, boundary_cond, r_seed)
elif simulation == 2:
    n = 300 # size of grid: n * n
    Dh = 1 # spatial resolution
    Dt = 1 # temporal resolution
    F, k = 0.014, 0.039 # parameter values
    Du = 0.1 # diffusion constant of u
    Dv = 0.05 # diffusion constant of v
    lo = 0
    hi = 1
    n_type = 'moore'
    boundary_cond = 'cut-off'
    r_seed = 42
    model = NPGrayScottCA(F, k, Dh, Dt, Du, Dv, n, lo, hi, n_type, boundary_cond, r_seed)
elif simulation == 3:
    n = 100 # size of grid: n * n
    Dh = 0.125 # spatial resolution
    Dt = 0.001 # temporal resolution
    alpha, beta, gamma = 0.8, 1, 1 # parameter values
    Du = 10**-5 # diffusion constant of u
    Dv = 10**-5 # diffusion constant of v
    lo = 0
    hi = 1
    n_type = 'moore'
    boundary_cond = 'cut-off'
    r_seed = 42
    model = NPBZ3CA(alpha, beta, gamma, Dh, Dt, Du, Dv, n, lo, hi, n_type, boundary_cond, r_seed)

pycxsimulator.GUI().start(func=[model.initialize, model.observe, model.update])