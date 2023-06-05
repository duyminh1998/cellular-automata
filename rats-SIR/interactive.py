from ratsSIR import RatsCA
from lib import pycxsimulator
import numpy as np

mu_r = 0.03
mu_f = 0.008

N = 5
T_min = 0
T_max = 0.1
T = np.arange(T_min, T_max, (T_max - T_min) / N, dtype=float)
Q_min = 0.1
Q_max = 1
Q = np.arange(Q_min, Q_max, (Q_max - Q_min) / N, dtype=float)
Q[0] = 0
# Q = [0.125, 0.75, 0.125]
q = 0.5
d = 0.2
n = 100
n_type = 'moore'
boundary_cond = 'periodic'
update_stats = True
r_seed = 42
model = RatsCA(T, Q, q, d, N, n, n_type, boundary_cond, update_stats, r_seed)

pycxsimulator.GUI().start(func=[model.initialize, model.observe, model.update])