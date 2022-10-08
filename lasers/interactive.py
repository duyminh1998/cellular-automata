from lasers import LaserCA
from lib import pycxsimulator

n = 10
pumping_probability = 0.01
photon_lifetime = 14
electron_lifetime = 160
max_photons = 50
threshold_delta = 1.0
n_type = 'moore'
boundary_cond = 'cut-off'
r_seed = 42
model = LaserCA(n, pumping_probability, photon_lifetime, electron_lifetime, max_photons, threshold_delta, n_type, boundary_cond, r_seed)

pycxsimulator.GUI().start(func=[model.initialize, model.observe, model.update])