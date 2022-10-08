# Author: Minh Hua
# Date: 09/07/2022
# Purpose: Plots the results of the Schelling Segregation model with Cellular Automata.

import pickle
import matplotlib.pyplot as plt
from schelling import SchellingSegregationCA

tolerance = []
max_tol = 0.75
min_tol = 0
delta_tol = 0.001
for t in range(int((max_tol - min_tol) / delta_tol)):
    cur_t = round(min_tol + t * delta_tol, 3)
    tolerance.append(cur_t)
tolerance.append(cur_t + delta_tol)

hyperparams = {
    'n': [50],
    'tolerance': tolerance,
    'initial_ratio': [0.5],
    'empty_perc': [0.1],
    'n_type': ['moore'],
    'boundary_cond': ['cut-off'],
    'r_seed': [42]
}

data_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Independent Study\\data\\schelling\\09102022\\"
r_seeds = [42, 69, 420, 100, 500, 1874, 192, 2, 991, 2350]
# for r_seed in r_seeds:
#     hyperparams['r_seed'] = [r_seed]
#     satisfaction_results, seg_results, last_ns = run_loop(SchellingSegregationCA, hyperparams)
#     with open(data_path + 'satisfaction_results_r({}).pickle'.format(r_seed), 'wb') as f:
#         pickle.dump(satisfaction_results, f)
#     with open(data_path + 'seg_results_r({}).pickle'.format(r_seed), 'wb') as f:
#         pickle.dump(seg_results, f)
#     with open(data_path + 'last_ns_r({}).pickle'.format(r_seed), 'wb') as f:
#         pickle.dump(last_ns, f)

avg_sat = [0 for _ in range(len(tolerance))]
min_sat = [1.1 for _ in range(len(tolerance))]
max_sat = [0 for _ in range(len(tolerance))]
avg_seg = [0 for _ in range(len(tolerance))]
min_seg = [1.1 for _ in range(len(tolerance))]
max_seg = [0 for _ in range(len(tolerance))]

for r_seed in r_seeds:
    with open(data_path + 'satisfaction_results_r({}).pickle'.format(r_seed), 'rb') as f:
        sat = pickle.load(f).flatten()
    with open(data_path + 'seg_results_r({}).pickle'.format(r_seed), 'rb') as f:
        seg = pickle.load(f).flatten()
    for i in range(len(sat)):
        avg_sat[i] = avg_sat[i] + sat[i]
        if sat[i] > max_sat[i]:
            max_sat[i] = sat[i]
        if sat[i] < min_sat[i]:
            min_sat[i] = sat[i]
        avg_seg[i] = avg_seg[i] + seg[i]
        if seg[i] > max_seg[i]:
            max_seg[i] = seg[i]
        if seg[i] < min_seg[i]:
            min_seg[i] = seg[i]

avg_sat = [sat / len(r_seeds) for sat in avg_sat]
avg_seg = [seg / len(r_seeds) for seg in avg_seg] 

# with open('seg_results.pickle', 'rb') as f:
#     seg_results = pickle.load(f).flatten()

# print(matplotlib.get_backend())
fig, ax = plt.subplots(1,1)
fig.set_size_inches(16, 9)
avg_plot = ax.scatter([t * 100 for t in tolerance], [s * 100 for s in avg_seg], s = 5, label = 'Average Segregation', c = 'green')
min_plot = ax.scatter([t * 100 for t in tolerance], [s * 100 for s in min_seg], s = 5, label = 'Minimum Segregation', c = 'blue')
max_plot = ax.scatter([t * 100 for t in tolerance], [s * 100 for s in max_seg], s = 5, label = 'Maximum Segregation', c = 'red')
ax.set_xlabel('Tolerance (%)')
ax.set_ylabel('Segregation Measure (%)')
ax.set_title('Dependence of segregation on tolerance')
# center text
note = 'Note: 50/50 population split \n10% empty spaces \nMoore neighborhood with cut-off boundary conditions'
fig.text(.1, .015, note, ha='left')
plt.legend()
plt.show()