from qam import *
from lib import pycxsimulator
import random as rd
import pickle

n = 5
rail = 5
prev_rails = None
# prev_rails = [
#     [[1, 1, 0, 0],
#     [1, 1, 0, 0],
#     [1, 1, 0, 0],
#     [1, 1, 0, 0]],
#     [
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [1, 1, 1, 1],
#         [1, 1, 1, 1]
#     ],
#     [
#         [0, 1, 1, 0],
#         [0, 1, 1, 0],
#         [0, 1, 1, 0],
#         [0, 1, 1, 0]        
#     ]
# ]
prev_rails = [
    [[2, 0, 0, 1, 1, 2],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [2, 0, 0, 1, 1, 2]],
    [[2, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 2]],
    [
        [2, 1, 1, 1, 1, 2],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [2, 1, 1, 1, 1, 2]
    ],
    [[2, 0, 1, 1, 0, 2,],
    [0, 0, 1, 1, 0, 0,],
    [0, 1, 1, 1, 1, 0,],
    [0, 1, 1, 1, 1, 0,],
    [0, 1, 1, 1, 1, 0,],
    [2, 0, 0, 0, 0, 2]]    
]
convergence_steps = 20
tolerance = 1.0
initial_ratio = 0.5
n_type = 'moore'
boundary_cond = 'cut-off'
r_seed = 42

final_rails_count = {}
final_rails_score = {}

# run 100 different random ICs
tests = 100000
for test in range(tests):
    print("Run {}".format(test))
    # r_seed = rd.randint(-10000, 10000)
    model = SAHCQAMCA(n, prev_rails, convergence_steps, tolerance, initial_ratio, n_type, boundary_cond, print_debug = False)
    model.initialize()
    update_result = model.update()
    while type(update_result) == None:
        update_result = model.update()
    # save the final configuration
    final_rail_ID = model.get_ID(model.config)
    if final_rail_ID not in final_rails_count.keys():
        final_rails_count[final_rail_ID] = 1
        final_rails_score[final_rail_ID] = model.calc_gray_score(model.compose_config([model.config]), model.config_dim)
    else:
        final_rails_count[final_rail_ID] = final_rails_count[final_rail_ID] + 1

with open('output/{}-QAM_rail{}_counts.pickle'.format(2 ** n, rail), 'wb') as f:
    pickle.dump(final_rails_count, f)
with open('output/{}-QAM_rail{}_scores.pickle'.format(2 ** n, rail), 'wb') as f:
    pickle.dump(final_rails_score, f)

print('Rail with optimal score:\n {}'.format(model.ID_to_rail(min(final_rails_score, key=final_rails_score.get), model.config_dim)))
print('Rail with most convergences:\n {}'.format(model.ID_to_rail(max(final_rails_count, key=final_rails_count.get), model.config_dim)))