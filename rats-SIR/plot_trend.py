from ratsSIR import RatsCA
from lib import pycxsimulator
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

mu_r = 0.03
mu_f = 0.008
N = 5
q = 0.5
d = 0.2
n = 50
n_type = 'moore'
boundary_cond = 'periodic'
r_seed = 42
update_stats = True
T_min = 0
Q_min = 0.1
Q_max = 1
Q = np.arange(Q_min, Q_max, (Q_max - Q_min) / N, dtype=float)
Q[0] = 0

max_t = 200

X = []
Y_epidemic = []
Y_endemic = []

# try out different values for T
for T_max in tqdm(np.arange(0.1, 1, (1 - 0.1) / 15, dtype=float)):
    T = np.arange(T_min, T_max, (T_max - T_min) / N, dtype=float)
    model = RatsCA(T, Q, q, d, N, n, n_type, boundary_cond, update_stats, r_seed)
    model.initialize()
    for t in range(max_t):
        model.update()
    # get pop stats
    X.append(T_max)
    Y_epidemic.append(np.mean(model.E[-5]))
    Y_endemic.append(np.mean(model.P[-5]))

# plot
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(X, Y_epidemic, 'g-', label = 'epidemic')
ax2.plot(X, Y_endemic, 'b-', label = 'endemic')

ax1.set_xlabel('coupling')
ax1.set_ylabel('proportion of epidemics', color='g')
ax2.set_ylabel('proportion of endemics', color='b')

fig.legend()
plt.show()