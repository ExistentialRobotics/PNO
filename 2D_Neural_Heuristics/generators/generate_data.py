import numpy as np
from dijkstra_data_generator import create_dataset
import matplotlib.pyplot as plt

import os
import sys

os.chdir('..')

training_data = np.load("/dataset/street_maps_1024.npy")
np.random.shuffle(training_data)

train_trial = 25
test_trial = 5

env_size = 1024

travel_time_values_array_train, signed_distance_array_train, velocity_matrices_array_train, goals_train = create_dataset(training_data=training_data[:train_trial,:,:], num_trials=train_trial, goal_trials=10, env_size=env_size,erosion_trials = 2)
travel_time_values_array_test,  signed_distance_array_test,  velocity_matrices_array_test,  goals_test =  create_dataset(training_data=training_data[train_trial:,:,:], num_trials=test_trial,  goal_trials=10, env_size=env_size,erosion_trials = 1)


np.save("goals.npy",np.concatenate((goals_train,goals_test), axis=0))

# Save velocity_matrices_array as "mask.npy"
np.save("mask.npy", np.concatenate((velocity_matrices_array_train,velocity_matrices_array_test), axis=0))

# Save travel_time_values_array as "output.npy"
np.save("output.npy", np.concatenate((travel_time_values_array_train,travel_time_values_array_test), axis=0))

# Save signed_distance_array as "dist_in.npy"
np.save("dist_in.npy", np.concatenate((signed_distance_array_train,signed_distance_array_test),axis=0))