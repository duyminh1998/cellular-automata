from schelling import SchellingSegregationCA
from lib import pycxsimulator

n = 4
tolerance = 0.7
initial_ratio = 0.5
empty_perc = 0.1
n_type = 'moore'
boundary_cond = 'cut-off'
r_seed = 42
model = SchellingSegregationCA(n, tolerance, initial_ratio, empty_perc, n_type, boundary_cond, r_seed)

pycxsimulator.GUI().start(func=[model.initialize, model.observe, model.update])