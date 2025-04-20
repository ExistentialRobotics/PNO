import sys

import numpy as np
from dataset import *

import argparse

import os

current_script_path = os.getcwd()
main_folder_path = os.path.dirname(current_script_path)

if main_folder_path not in sys.path:
    sys.path.append(main_folder_path)


from domains.gridworld import *
from generators.obstacle_gen import *
from tqdm import tqdm 

import numpy as np
import matplotlib.pyplot as plt
import pykonal
import random

from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndimage

def calculate_signed_distance(velocity_matrix):
    distance = distance_transform_edt(velocity_matrix != 0)
    return distance


def create_dataset(training_data, num_trials,goal_trials,env_size, erosion_trials = 1, a_min = 0, a_max = 1):

    count = 0
    a_min = 0
    a_max = 1

    velocity_matrices_array = np.ones((num_trials*goal_trials*erosion_trials, env_size, env_size))
    travel_time_values_array = np.zeros((num_trials*goal_trials*erosion_trials, env_size, env_size))
    goals = np.zeros((num_trials*goal_trials*erosion_trials, 2))

    for trial in tqdm(range(num_trials)):
        im_np = training_data[trial,:,:]
        original_maze = 1 - im_np
        condition1 = original_maze == 1
        row_indices, col_indices = np.indices(original_maze.shape)
        condition2 = (row_indices < env_size) & (col_indices < env_size)
        combined_condition = condition1 & condition2


        for goal_trial in range(goal_trials):
            passable_indices = np.argwhere(combined_condition)
            high_values_mask = None
            if passable_indices.size > 0:
                if(trial * goal_trials + goal_trial!=count):
                    break

                environment = np.array(original_maze)
                while np.all(travel_time_values_array[trial * goal_trials + goal_trial, :, :] == 0):
                    signeddistance = calculate_signed_distance(environment)
                    velocity_matrix = (1/a_max)*np.clip(signeddistance,a_min=a_min,a_max=a_max)*environment

                    goal_index = random.choice(passable_indices)
                    goal = goal_index[0], goal_index[1]

                    # Set up the Eikonal solver
                    solver = pykonal.EikonalSolver(coord_sys="cartesian")
                    solver.velocity.min_coords = 0, 0, 0
                    solver.velocity.node_intervals = 1, 1, 1
                    solver.velocity.npts = env_size, env_size, 1
                    solver.velocity.values = velocity_matrix.reshape(env_size, env_size, 1)
                    src_idx = goal[0], goal[1], 0
                    solver.traveltime.values[src_idx] = 0
                    solver.unknown[src_idx] = False
                    solver.trial.push(*src_idx)
                    solver.solve()

                    velocity_matrices_array[trial*goal_trials + goal_trial, :, :] = environment.reshape(env_size,env_size)
                    goals[trial * goal_trials + goal_trial,:] = goal
                    travel_time_values_array[trial * goal_trials + goal_trial, :, :] = solver.traveltime.values[:, :, 0]
                    high_values_mask = solver.traveltime.values[:, :, 0] > 1000
                    input_mask = (velocity_matrix == 0)
                    travel_time_values_array[trial * goal_trials + goal_trial, high_values_mask] = 0

                if(high_values_mask!=input_mask).any():
                    velocity_matrices_array[trial * goal_trials + goal_trial, high_values_mask] = 0

                if np.any(travel_time_values_array[trial * goal_trials + goal_trial, :, :] > 1000):
                    print("Improper Data")
                    break
                if np.all(travel_time_values_array[trial * goal_trials + goal_trial, :, :] == 0):
                    print("Improper Data")
                    break
            else:
                print("Improper Data")
                break

            count+=1 

    if(erosion_trials>1):        
        offset = (num_trials * goal_trials)
        for trial in tqdm(range(num_trials)):
            im_np = training_data[trial,:,:]
            original_maze = 1 - im_np
            condition1 = original_maze == 1
            row_indices, col_indices = np.indices(original_maze.shape)
            condition2 = (row_indices < env_size) & (col_indices < env_size)
            combined_condition = condition1 & condition2


            for goal_trial in range(goal_trials):
                passable_indices = np.argwhere(combined_condition)
                high_values_mask = None
                if passable_indices.size > 0:
                    if(trial * goal_trials + goal_trial +offset!=count):
                        break

                    environment = 1-ndimage.binary_erosion(1-np.array(original_maze),iterations=np.random.randint(1,4)).astype(np.array(original_maze).dtype)
                    while np.all(travel_time_values_array[trial * goal_trials + goal_trial + offset, :, :] == 0):
                        signeddistance = calculate_signed_distance(environment)
                        velocity_matrix = (1/a_max)*np.clip(signeddistance,a_min=a_min,a_max=a_max)*environment
                        goal_index = random.choice(passable_indices)
                        goal = goal_index[0], goal_index[1]

                        # Set up the Eikonal solver
                        solver = pykonal.EikonalSolver(coord_sys="cartesian")
                        solver.velocity.min_coords = 0, 0, 0
                        solver.velocity.node_intervals = 1, 1, 1
                        solver.velocity.npts = env_size, env_size, 1
                        solver.velocity.values = velocity_matrix.reshape(env_size, env_size, 1)
                        src_idx = goal[0], goal[1], 0
                        solver.traveltime.values[src_idx] = 0
                        solver.unknown[src_idx] = False
                        solver.trial.push(*src_idx)
                        solver.solve()
                        velocity_matrices_array[trial*goal_trials + goal_trial + offset, :, :] = environment.reshape(env_size,env_size)
                        goals[trial * goal_trials + goal_trial + offset,:] = goal
                        travel_time_values_array[trial * goal_trials + goal_trial+offset, :, :] = solver.traveltime.values[:, :, 0]
                        high_values_mask = solver.traveltime.values[:, :, 0] > 1000
                        input_mask = (velocity_matrix == 0)
                        travel_time_values_array[trial * goal_trials + goal_trial + offset, high_values_mask] = 0

                    if(high_values_mask!=input_mask).any():
                        velocity_matrices_array[trial * goal_trials + offset, high_values_mask] = 0

                    if np.any(travel_time_values_array[trial * goal_trials + goal_trial + offset, :, :] > 1000):
                        print("Improper Data: Something exceeded max limit")
                        break

                    if np.all(travel_time_values_array[trial * goal_trials + goal_trial + offset, :, :] == 0):
                        print("Improper Data: All values are zero")
                        break
                else:
                    print("Improper Data: No valid goal positions")
                    break

                count+=1

    signed_distance_array = np.zeros((num_trials*goal_trials*erosion_trials,env_size,env_size))

    for i in tqdm(range(0, num_trials*goal_trials*erosion_trials)):
        signed_distance_array[i,:,:] = calculate_signed_distance(velocity_matrices_array[i,:,:])         


    return travel_time_values_array, signed_distance_array, velocity_matrices_array, goals           
