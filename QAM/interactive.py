from qam import *
from lib import pycxsimulator

n = 5
# prev_rails = None
# prev_rails = [
#     [[0, 0, 0, 0],
#     [0, 0, 0, 0],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1]]
# ]
prev_rails = [
    [[0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1]],
    [[0, 0, 1, 1],
    [0, 0, 1, 1], 
    [0, 0, 1, 1],
    [0, 0, 1, 1]]
]
prev_rails = [
    [[0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1]],
    [[0, 0, 1, 1],
    [0, 0, 1, 1], 
    [0, 0, 1, 1],
    [0, 0, 1, 1]],
    [[0, 0, 0, 0],
    [1, 1, 1, 1], 
    [1, 1, 1, 1], 
    [0, 0, 0, 0]]
]
prev_rails = [
    [[2, 0, 0, 1, 1, 2],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [2, 0, 0, 1, 1, 2]]
]
convergence_steps = 10
tolerance = 1.0
initial_ratio = 0.5
n_type = 'moore'
boundary_cond = 'cut-off'
r_seed = 42
model = SAHCQAMCA(n, prev_rails, convergence_steps, tolerance, initial_ratio, n_type, boundary_cond, r_seed, print_debug = True)

pycxsimulator.GUI().start(func=[model.initialize, model.observe, model.update])