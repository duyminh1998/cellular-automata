from turing_pattern import TuringPatternCA, NPTuringPatternCA
from bz import BZCA, NPBZCA, NPBZ3CA
from gray_scott import GrayScottCA, NPGrayScottCA
from lib import pycxsimulator
import imageio
from lib.utils import *

simulation = 4

if simulation == 0:
    n = 300 # size of grid: n * n
    Dh = 1. / n # spatial resolution, assuming space is [0,1] * [0,1]
    Dt = 0.02 # temporal resolution
    a, b, c, d, h, k = 1., -1., 2., -1.5, 1., 1. # parameter values
    Du = 0.0001 # diffusion constant of u
    Dv = 0.0006 # diffusion constant of v
    n_type = 'moore'
    boundary_cond = 'cut-off'
    r_seed = 42
    model = NPTuringPatternCA(a, b, c, d, h, k, Dh, Dt, Du, Dv, n, n_type, boundary_cond, r_seed)
    file_name = '{}_n_{}_a_{}_b_{}_c_{}_d_{}_h_{}_k_{}_Du_{}_Dv_{}.gif'.format(model.name, n, a, b, c, d, h, k, Du, Dv)
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
    file_name = '{}_n_{}_e_{}_q_{}_f_{}_Du_{}_Dv_{}.gif'.format(model.name, n, epsilon, q, f, Du, Dv)
elif simulation == 2:
    n = 300 # size of grid: n * n
    Dh = 1 # spatial resolution
    Dt = 1 # temporal resolution
    F, k = 0.0545, 0.062 # parameter values
    Du = 0.1 # diffusion constant of u
    Dv = 0.05 # diffusion constant of v
    lo = 0
    hi = 1
    n_type = 'moore'
    boundary_cond = 'cut-off'
    r_seed = 42
    model = NPGrayScottCA(F, k, Dh, Dt, Du, Dv, n, lo, hi, n_type, boundary_cond, r_seed)
    file_name = '{}_n_{}_F_{}_k_{}_Du_{}_Dv_{}.gif'.format(model.name, n, F, k, Du, Dv)
elif simulation == 4:
    n = 500 # size of grid: n * n
    Dh = 0.125 # spatial resolution
    Dt = 0.001 # temporal resolution
    alpha, beta, gamma = 1.2, 1, 1 # parameter values
    Du = 10**-5 # diffusion constant of u
    Dv = 10**-5 # diffusion constant of v
    lo = 0
    hi = 1
    n_type = 'moore'
    boundary_cond = 'cut-off'
    r_seed = 42
    model = NPBZ3CA(alpha, beta, gamma, Dh, Dt, Du, Dv, n, lo, hi, n_type, boundary_cond, r_seed)

fps = 30
seconds = 30
n_frames = seconds * fps
steps_per_frame = 1
out_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Independent Study\\src\\output\\reaction-diffusion\\"
model.initialize()
# Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
U_frames = []
V_frames = []
W_frames = []
UVW_frames = []
for _ in tqdm(range(n_frames)):
    model.update()
    model.uvw = np.zeros(shape=model.u.shape)
    for x in range(model.n + 2):
        for y in range(model.n + 2):
            substances = [model.u[x][y], model.v[x][y], model.w[x][y]]
            model.uvw[x][y] = substances.index(max(substances))
    U_scaled = np.uint8(model.u_cmap(model.u)*255)
    V_scaled = np.uint8(model.v_cmap(model.v)*255)
    W_scaled = np.uint8(model.w_cmap(model.w)*255)
    UVW_scaled = np.uint8(model.uvw_cmap(model.uvw / 2)*255)
    U_frames.append(U_scaled)
    V_frames.append(V_scaled)
    W_frames.append(W_scaled)
    UVW_frames.append(UVW_scaled)
imageio.mimsave(out_path + 'U.gif', U_frames, format='gif', fps=fps)
imageio.mimsave(out_path + 'V.gif', V_frames, format='gif', fps=fps)
imageio.mimsave(out_path + 'W.gif', W_frames, format='gif', fps=fps)
imageio.mimsave(out_path + 'UVW.gif', UVW_frames, format='gif', fps=fps)