from lasers import LaserCA
from lib import pycxsimulator

n = 200
pumping_probability = 0.1
photon_lifetime = 8
electron_lifetime = 30
max_photons = 10
threshold_delta = 1.0
n_type = 'moore'
boundary_cond = 'periodic'
r_seed = 42
model = LaserCA(n, pumping_probability, photon_lifetime, electron_lifetime, max_photons, threshold_delta, n_type, boundary_cond, r_seed)

pycxsimulator.GUI().start(func=[model.initialize, model.observe, model.update])