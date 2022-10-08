# Author: Minh Hua
# Date: 09/23/2022
# Purpose: Utility functions for various Cellular Automata models.

from tqdm import tqdm
import numpy as np
import imageio

def create_image(model, t:int) -> tuple:
    # Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
    for t in range(t):
        model.update()
    U_scaled = np.uint8(255 * (model.u - model.u.min()) / (model.u.max() - model.u.min()))
    V_scaled = np.uint8(255 * (model.v - model.v.min()) / (model.v.max() - model.v.min()))
    return U_scaled, V_scaled, model

def create_frames(n, model, t:int) -> list:
    # Source: https://pnavaro.github.io/python-fortran/06.gray-scott-model.html
    U_frames = []
    V_frames = []
    for _ in tqdm(range(n)):
        U_scaled, V_scaled, model = create_image(model, t)
        U_frames.append(U_scaled)
        V_frames.append(V_scaled)
    return U_frames, V_frames

def run_loop_reaction_diff_models(base_model, hyperparams:dict, gif_path:str, frames:int=500, steps_per_frame:int=50) -> None:
    """
    Description:
        Run the model for a certain number of steps or until a threshold is met.

    Arguments:
        base_model: the model to run the loop for.
        hyperparams: the set of hyperparameters for the model.
        gif_path: the location to save the gifs.
        frames: the number of frames to generate.
        steps_per_frame: the number of time steps in the simulation per frame.

    Return:
        (None)
    """
    dim = [] # holds the dimension of the hyperparameters so we can initialize an np array
    for param_vals in hyperparams.values():
        dim.append(len(param_vals))
    # empty np array to hold the results corresponding to a parameter combination
    results = np.empty((dim))
    # loop through every parameter combination and find the best one
    for idx, _ in np.ndenumerate(results):
        # get the current combination of parameters
        cur_args_list = []
        for cur_param, param_key in zip(idx, hyperparams.keys()):
            cur_args_list.append(hyperparams[param_key][cur_param])
        print("Current args: {}".format(cur_args_list))

        # initialize configuration
        model = base_model(*cur_args_list)
        model.initialize()

        # update config until max_steps
        U_frames, V_frames = create_frames(frames, model, steps_per_frame)
        if model.name == 'GS':
            file_name = '{}_n_{}_F_{}_k_{}_Du_{}_Dv_{}_U.gif'.format(model.name, model.n, model.F, model.k, model.Du, model.Dv)
            imageio.mimsave(gif_path + file_name, U_frames, format='gif', fps=60)
            file_name = '{}_n_{}_F_{}_k_{}_Du_{}_Dv_{}_V.gif'.format(model.name, model.n, model.F, model.k, model.Du, model.Dv)
            imageio.mimsave(gif_path + file_name, V_frames, format='gif', fps=60)
        elif model.name == 'BZ':
            file_name = '{}_n_{}_e_{}_q_{}_f_{}_Du_{}_Dv_{}_U.gif'.format(model.name, model.n, model.epsilon, model.q, model.f, model.Du, model.Dv)
            imageio.mimsave(gif_path + file_name, U_frames, format='gif', fps=60)
            file_name = '{}_n_{}_e_{}_q_{}_f_{}_Du_{}_Dv_{}_V.gif'.format(model.name, model.n, model.epsilon, model.q, model.f, model.Du, model.Dv)
            imageio.mimsave(gif_path + file_name, V_frames, format='gif', fps=60)