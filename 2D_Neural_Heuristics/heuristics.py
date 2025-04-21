import numpy as np
import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

# FMM Packages
import pykonal
# import skfmm

import matplotlib.pyplot as plt


current_script_path = os.getcwd()
main_folder_path = os.path.dirname(current_script_path)

if main_folder_path not in sys.path:
    sys.path.append(main_folder_path)

from astar.astar import  AStar
from astar.environment_simple import Environment2D
from astar.utilities import tic,toc,drawMap,drawPath2D,plotClosedNodes,plotInconsistentNodes

from generators.dijkstra_data_generator import calculate_signed_distance

from scipy import ndimage
from scipy.ndimage import distance_transform_edt


from train.TrainPlanningOperator2D import PlanningOperator2D
from train.TrainPlanningOperator2D import smooth_chi


def generaterandompos(maps):
    "Generates random positions in the free space of given maps"

    numofmaps = maps.shape[0]
    env_size = maps.shape[1]
    pos = np.zeros((numofmaps,2))

    assert maps.shape[1] == maps.shape[2]

    for i,map in enumerate(maps):

        condition1 = map == 1
        row_indices, col_indices = np.indices(map.shape)
        condition2 = (row_indices < env_size) & (col_indices < env_size)
        combined_condition = condition1 & condition2
        passable_indices = np.argwhere(combined_condition)
        point = random.choice(passable_indices)
        pos[i,:] = np.array([point[0],point[1]])

    return pos.astype(int)

def euclideannorm(map,goal):
    
    t0 = tic()
    env_size_x, env_size_y = map.shape
    x, y = np.meshgrid(np.arange(env_size_x), np.arange(env_size_y), indexing='ij')
    positions = np.stack([x, y], axis=-1)
    valuefunction = np.linalg.norm(positions - goal, axis=-1)
    dt = toc(t0)
    
    return valuefunction, dt

def manhattandistance(map, goal):
    
    t0 = tic()
    env_size_x, env_size_y = map.shape
    x, y = np.meshgrid(np.arange(env_size_x), np.arange(env_size_y), indexing='ij')
    positions = np.stack([x, y], axis=-1)
    valuefunction = np.abs(positions[..., 0] - goal[0]) + np.abs(positions[..., 1] - goal[1])
    dt = toc(t0)
    
    return valuefunction, dt    

def FMM(map,goal):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map'''

    t0 = tic()
    env_size_x, env_size_y = map.shape
    velocity_matrix = (1-map)
    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = 1, 1, 1
    solver.velocity.npts = env_size_x, env_size_y, 1
    solver.velocity.values = velocity_matrix.reshape(env_size_x, env_size_y, 1)
    src_idx = goal[0].astype(int), goal[1].astype(int), 0
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    solver.solve()
    valuefunction = solver.traveltime.values[:, :, 0]
    dt = toc(t0)
    return valuefunction,dt

def highresFMM(map,goal):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map
    For high res output insert high res map (2x the orignal map)'''

    t0 = tic()
    env_size_x, env_size_y = map.shape
    velocity_matrix = (1-map)
    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = 0.5, 0.5, 0.5
    solver.velocity.npts = env_size_x, env_size_y, 1
    solver.velocity.values = velocity_matrix.reshape(env_size_x, env_size_y, 1)
    src_idx = 2*goal[0].astype(int), 2*goal[1].astype(int), 0
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    solver.solve()
    valuefunction = solver.traveltime.values[:, :, 0]
    valuefunction = valuefunction[1::2, 1::2]
    dt = toc(t0)
    return valuefunction,dt


def scikitFMM(map,goal):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map'''

    t0 = tic()
    env_size_x, env_size_y = map.shape
    phi = np.ones((env_size_x, env_size_y))
    phi[goal[0].astype(int),goal[1].astype(int)] = 0
    velocity_matrix = (1-map)
    valuefunction = skfmm.travel_time(phi, speed = velocity_matrix)
    dt = toc(t0)

    return valuefunction,dt


def dijkstra(map,goal):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map'''
    t0 = tic()
    env = Environment2D(goal,map)
    valuefunction = AStar.getDistances(env)
    dt = toc(t0)

    return valuefunction,dt

def sdfoperator(map, model):
    print("Hello")


def planningoperator(map,goal,model,erosion=4):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map'''

    mask = 1-map
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_size_x, env_size_y = map.shape

    t0 = tic()

    # Erode Map to under approximate value function
    eroded_map =  1-ndimage.binary_erosion(1-np.array(mask),iterations=erosion).astype(np.array(mask).dtype)
    terode = toc(t0)
    # print("Erode Time",terode)

    # Calculate SDF of eroded map
    sdf = calculate_signed_distance(eroded_map)
    eroded_map = eroded_map.reshape(1,env_size_x,env_size_y,1)
    eroded_map = torch.tensor(eroded_map,dtype=torch.float)
    sdf = sdf.reshape(1,env_size_x,env_size_y,1)
    sdf = torch.tensor(sdf,dtype=torch.float)
    tsdf = toc(t0) - terode
    # print("SDF Time",tsdf)

    # Calculate Chi for smoothening
    smooth_coef=5. #Depends on what is it trained on
    chi = smooth_chi(eroded_map, sdf, smooth_coef).to(device)

    # Load Goal Position
    goal_coord = np.array([goal[0],goal[1]])
    gg = goal_coord.reshape(1,2,1)
    gg = torch.tensor(gg, dtype=torch.float).to(device)

    #Infer value function 
    valuefunction = model(chi,gg)
    valuefunction = valuefunction.detach().cpu().numpy().reshape(env_size_x,env_size_y)
    valuefunction = valuefunction/(mask+10e-10)
    tno = toc(t0) - tsdf -terode
    # print("NO time:",tno)

    # Calculate Max
    euclideanvalue, _ = euclideannorm(mask, goal)
    valuefunction = np.maximum(euclideanvalue, valuefunction)
    dt = toc(t0)

    return valuefunction,dt

def doEikplanningoperator(map,goal,model,erosion=4):
    '''Map is a 2D binary occupancy map with 1 representing obstacle and 0 representing free space
    Goal is an index in the map'''

    mask = 1-map
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_size_x, env_size_y = map.shape

    t0 = tic()

    # Erode Map to under approximate value function
    eroded_map =  1-ndimage.binary_erosion(1-np.array(mask),iterations=erosion).astype(np.array(mask).dtype)
    terode = toc(t0)
    # print("Erode Time",terode)

    # Calculate SDF of eroded map
    sdf = calculate_signed_distance(eroded_map)
    eroded_map = eroded_map.reshape(1,env_size_x,env_size_y,1)
    eroded_map = torch.tensor(eroded_map,dtype=torch.float)
    sdf = sdf.reshape(1,env_size_x,env_size_y,1)
    sdf = torch.tensor(sdf,dtype=torch.float)
    tsdf = toc(t0) - terode
    # print("SDF Time",tsdf)

    # Calculate Chi for smoothening
    smooth_coef=5. #Depends on what is it trained on
    chi = smooth_chi(eroded_map, sdf, smooth_coef).to(device)

    # Load Goal Position
    goal_coord = np.array([goal[0],goal[1]])
    gg = goal_coord.reshape(1,2,1)
    gg = torch.tensor(gg, dtype=torch.float).to(device)

    #Infer value function 
    valuefunction = model(chi,gg)
    valuefunction = valuefunction.detach().cpu().numpy().reshape(env_size_x,env_size_y)
    valuefunction = valuefunction/(mask+10e-10)
    tno = toc(t0) - tsdf -terode
    # print("NO time:",tno)

    # Calculate Max
    euclideanvalue, _ = euclideannorm(mask, goal)
    valuefunction = np.maximum(euclideanvalue, valuefunction)
    dt = toc(t0)

    eroded_map = eroded_map.detach().cpu().numpy().reshape(env_size_x,env_size_y)

    return valuefunction,dt


def testheuristiconsinglemap(start, goal, map, heuristic, plotresults = False, **kwargs):
    valuefunction, dt_infer = heuristic(map, goal,**kwargs)
        
    env = Environment2D(goal, map, valuefunction)
    
    t0 = tic()
    path_cost, path, action_idx, nodes_count, sss = AStar.plan(start, env)
    dt_plan = toc(t0)

    if path_cost < 10e10:
        succ = True

    path_array = np.asarray(path)
    
    if plotresults:
        f, ax = plt.subplots()
        drawMap(ax, map)
        plotClosedNodes(ax,sss)
        plotInconsistentNodes(ax,sss)
        drawPath2D(ax, path_array)

    print(  'Path Cost:', path_cost, 
        '\nPlanning Time:', dt_plan,
        '\nInference Time:', dt_infer, 
        '\nNumber of Node Expansions:', nodes_count,
        '\nSuccess:', succ)    




    
# def testheuristiconmaps(starts, goals, maps, heuristic, plotresults = False, printvalues = True, saveplotdata= False,**kwargs):
#     avgpathcost, avgplantime, avginfertime, avgnodesexp, avgsuccessrate = 0, 0, 0, 0, 0
#     totpathcost, totplantime, totinfertime, totnodesexp, succcount = 0, 0, 0, 0, 0

#     numofmaps = maps.shape[0]

#     for start, goal, map in zip(starts, goals, 1-maps):

#         # Call the heuristic function with additional arguments
#         valuefunction, dt_infer = heuristic(map, goal,**kwargs)
        
#         env = Environment2D(goal, map, valuefunction)
        
#         t0 = tic()
#         path_cost, path, action_idx, nodes_count, sss = AStar.plan(start, env)
#         dt_plan = toc(t0)

#         if path_cost < 10e10:
#             succcount += 1

#         totpathcost += path_cost
#         totplantime += dt_plan
#         totinfertime += dt_infer
#         totnodesexp += nodes_count

#         path_array = np.asarray(path)
        
#         if plotresults:
#             f, ax = plt.subplots()
#             drawMap(ax, map)
#             closednodes = plotClosedNodes(ax,sss)
#             # plotInconsistentNodes(ax,sss,env)
#             drawPath2D(ax, path_array)

#             if saveplotdata:
#             # Save the closed nodes and the path
#                 closed_coords = [sss['hm'][key].coord for key in sss['closed_list']]
#                 closed_array = np.array(closed_coords)

#                 # Save to files using a unique identifier (idx)
#                 np.save(f'closednodes.npy', closed_array)
#                 np.save(f'path.npy', path_array)
#                 np.save(f'map.npy', map)
#                 np.save(f'valuefunction.npy',valuefunction)
    
    

#     # Calculate averages
#     avgpathcost = totpathcost / numofmaps
#     avgplantime = totplantime / numofmaps
#     avginfertime = totinfertime / numofmaps
#     avgnodesexp = totnodesexp / numofmaps
#     avgsuccessrate = succcount / numofmaps

#     if printvalues:
#         print(  'Average Path Cost:', avgpathcost, 
#                 '\nAverage Planning Time:', avgplantime,
#                 '\nAverage Inference Time:', avginfertime, 
#                 '\nAverage Number of Node Expansions:', avgnodesexp,
#                 '\nAverage Success Rate:', avgsuccessrate)

#     return avgpathcost, avgplantime, avginfertime, avgnodesexp, avgsuccessrate


def testheuristiconmaps(starts, goals, maps, heuristic, plotresults=False, printvalues=True, saveplotdata=False, **kwargs):
    avgpathcost, avgplantime, avginfertime, avgnodesexp, avgsuccessrate = 0, 0, 0, 0, 0
    totpathcost, totplantime, totinfertime, totnodesexp, succcount = 0, 0, 0, 0, 0

    numofmaps = maps.shape[0]
    erosion = kwargs.get('erosion', None)  # Get the erosion value if passed in kwargs
    if erosion is not None:
        assert( 1 <= erosion) #Erosion function works only for values equal to and above 1
    heuristic_name = heuristic.__name__  # Get the name of the heuristic function

    for start, goal, map in zip(starts, goals, 1 - maps):

        # Call the heuristic function with additional arguments
        valuefunction, dt_infer = heuristic(map, goal, **kwargs)
        
        env = Environment2D(goal, map, valuefunction)
        
        t0 = tic()
        path_cost, path, action_idx, nodes_count, sss = AStar.plan(start, env)
        dt_plan = toc(t0)

        if path_cost < 10e10:
            succcount += 1

        totpathcost += path_cost
        totplantime += dt_plan
        totinfertime += dt_infer
        totnodesexp += nodes_count

        path_array = np.asarray(path)
        
        if plotresults:
            f, ax = plt.subplots()
            drawMap(ax, map)
            closednodes = plotClosedNodes(ax, sss)
            drawPath2D(ax, path_array)

            if saveplotdata:
                # Save the closed nodes and the path
                closed_coords = [sss['hm'][key].coord for key in sss['closed_list']]
                closed_array = np.array(closed_coords)

                # Create a unique filename using the heuristic name and erosion value (if provided)
                erosion_str = f'_erosion_{erosion}' if erosion is not None else ''
                filename_prefix = f'{heuristic_name}{erosion_str}'

                # Save to files using the generated unique prefix
                np.save(f'{filename_prefix}_closednodes.npy', closed_array)
                np.save(f'{filename_prefix}_path.npy', path_array)
                np.save(f'{filename_prefix}_map.npy', map)
                np.save(f'{filename_prefix}_valuefunction.npy', valuefunction)

    # Calculate averages
    avgpathcost = totpathcost / numofmaps
    avgplantime = totplantime / numofmaps
    avgnodesexp = totnodesexp / numofmaps
    avgsuccessrate = succcount / numofmaps

    if printvalues:
        print('Average Path Cost:', avgpathcost, 
              '\nAverage Planning Time:', avgplantime,
              '\nAverage Number of Node Expansions:', avgnodesexp,
              '\nAverage Success Rate:', avgsuccessrate)

    return avgpathcost, avgplantime, avginfertime, avgnodesexp, avgsuccessrate


def testheuristiconmaps_eplan(starts, goals, maps, heuristic, plotresults = False, printvalues = True, **kwargs):
    avgpathcost, avgplantime, avginfertime, avgnodesexp, avgsuccessrate = 0, 0, 0, 0, 0
    totpathcost, totplantime, totinfertime, totnodesexp, succcount = 0, 0, 0, 0, 0

    numofmaps = maps.shape[0]

    for start, goal, map in zip(starts, goals, 1-maps):

        # Call the heuristic function with additional arguments
        valuefunction, dt_infer = heuristic(map, goal,**kwargs)
        
        env = Environment2D(goal, map, valuefunction)
        
        t0 = tic()
        path_cost, path, action_idx, nodes_count, sss = AStar.eplan(start, env)
        dt_plan = toc(t0)

        if path_cost < 10e10:
            succcount += 1

        totpathcost += path_cost
        totplantime += dt_plan
        totinfertime += dt_infer
        totnodesexp += nodes_count

        path_array = np.asarray(path)
        
        if plotresults:
            f, ax = plt.subplots()
            drawMap(ax, map)
            plotClosedNodes(ax,sss)
            plotInconsistentNodes(ax,sss,env)
            drawPath2D(ax, path_array)
    

    # Calculate averages
    avgpathcost = totpathcost / numofmaps
    avgplantime = totplantime / numofmaps
    avginfertime = totinfertime / numofmaps
    avgnodesexp = totnodesexp / numofmaps
    avgsuccessrate = succcount / numofmaps

    if printvalues:
        print(  'Average Path Cost:', avgpathcost, 
                '\nAverage Planning Time:', avgplantime,
                '\nAverage Inference Time:', avginfertime, 
                '\nAverage Number of Node Expansions:', avgnodesexp,
                '\nAverage Success Rate:', avgsuccessrate)

    return avgpathcost, avgplantime, avginfertime, avgnodesexp, avgsuccessrate
























