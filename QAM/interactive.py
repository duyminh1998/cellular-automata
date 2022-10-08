from qam import QAMCA
from lib import pycxsimulator

n = 4
constellation_type = 'square'
percent_satisfied_agents = 0.7838
random_swap_every = 5
random_swap_n_agents = 16
tolerance = 0.5
initial_ratio = 0.5
empty_perc = 0
n_type = 'moore'
boundary_cond = 'cut-off'
r_seed = 42
model = QAMCA(n, constellation_type, percent_satisfied_agents, random_swap_every, random_swap_n_agents, tolerance, initial_ratio, empty_perc, n_type, boundary_cond, r_seed)

pycxsimulator.GUI().start(func=[model.initialize, model.observe, model.update])