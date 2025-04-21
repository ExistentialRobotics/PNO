import sys

import numpy as np

import argparse

import os

current_script_path = os.getcwd()
main_folder_path = os.path.dirname(current_script_path)

if main_folder_path not in sys.path:
    sys.path.append(main_folder_path)


from tqdm import tqdm 

import numpy as np
import matplotlib.pyplot as plt
import pykonal
import random
import skfmm

from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndimage

def calculate_signed_distance(velocity_matrix):
    distance = distance_transform_edt(velocity_matrix != 0)
    return distance

def scikitFMM(map,goal):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map'''

    env_size_x, env_size_y, env_size_z, env_size_a = map.shape
    phi = np.ones((env_size_x, env_size_y, env_size_z, env_size_a))
    phi[goal[0].astype(int),goal[1].astype(int),goal[2].astype(int), goal[3].astype(int)] = 0
    velocity_matrix = (map)
    valuefunction = skfmm.travel_time(phi, speed = velocity_matrix).filled()
    return valuefunction


def pykonalFMM(map,goal):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map'''

    env_size_x, env_size_y, env_size_z = map.shape
    velocity_matrix = (map)
    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = 1, 1, 1
    solver.velocity.npts = env_size_x, env_size_y, env_size_z
    solver.velocity.values = velocity_matrix.reshape(env_size_x, env_size_y, env_size_z)
    src_idx = goal[0].astype(int), goal[1].astype(int), goal[2].astype(int)
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    solver.solve()
    valuefunction = solver.traveltime.values[:, :, :]
    return valuefunction

def create_dataset(maps, num_trials,goal_trials,env_size):

    count = 0

    velocity_matrices_array = np.ones((num_trials*goal_trials, env_size[0], env_size[1], env_size[2], env_size[3]))
    travel_time_values_array = np.zeros((num_trials*goal_trials, env_size[0], env_size[1], env_size[2], env_size[3]))
    signed_distance_array = np.zeros((num_trials*goal_trials,env_size[0],env_size[1], env_size[2], env_size[3]))
    goals = np.zeros((num_trials*goal_trials, 4))

    
    for trial in tqdm(range(num_trials)):
        im_np = maps[trial,:,:,:]
        original_maze = im_np
        condition1 = original_maze == 1
        row_indices, col_indices, z_indices, a_indices = np.indices(original_maze.shape)
        condition2 = (row_indices < env_size[0]) & (col_indices < env_size[1]) & (z_indices < env_size[2]) & (a_indices < env_size[3])
        # condition3 = (row_indices == 8) & (col_indices == 8) & (z_indices == 8)
        combined_condition = condition1 & condition2 

        for goal_trial in range(goal_trials):
            passable_indices = np.argwhere(combined_condition)
            high_values_mask = None
            if passable_indices.size > 0:
                if(trial * goal_trials + goal_trial!=count):
                    print("count not matching")
                    break

                environment = np.array(original_maze)
                while np.all(travel_time_values_array[trial * goal_trials + goal_trial, :, :, :, :] == 0):
                    velocity_matrix = environment
                    goal_index = random.choice(passable_indices)
                    goal = goal_index[0], goal_index[1], goal_index[2], goal_index[3]
                    velocity_matrices_array[trial*goal_trials + goal_trial, :, :,:, :] = environment.reshape(env_size[0],env_size[1], env_size[2], env_size[3])
                    goals[trial * goal_trials + goal_trial,:] = goal
                    travel_time_values_array[trial * goal_trials + goal_trial, :, :, :, :] = scikitFMM(velocity_matrix,goal)
                    signed_distance_array[trial * goal_trials + goal_trial, :, :, :, :] = calculate_signed_distance(environment)
                    high_values_mask = travel_time_values_array[trial * goal_trials + goal_trial, :, :, :, :] > 10e5
                    input_mask = (velocity_matrix == 0)
                    travel_time_values_array[trial * goal_trials + goal_trial, high_values_mask] = 0

                if(high_values_mask!=input_mask).all():
                    continue    
                
                if(high_values_mask!=input_mask).any():
                    velocity_matrices_array[trial * goal_trials + goal_trial, high_values_mask] = 0

                if np.any(travel_time_values_array[trial * goal_trials + goal_trial, :, :, :, :] > 10e5):
                    print("Improper Data")
                    break
                if np.all(travel_time_values_array[trial * goal_trials + goal_trial, :, :, :, :] == 0):
                    print("Improper Data")
                    break
            else:
                print("Improper Data")
                break

            count+=1 
       
    return travel_time_values_array, signed_distance_array, velocity_matrices_array, goals           
