import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.image import AxesImage
from matplotlib.collections import PolyCollection, LineCollection, PathCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Path3DCollection

def tic():
  return time.time()
def toc(tstart, nm=""):
  return (time.time() - tstart)

def drawMap(ax,cmap):
  '''
    Draws the matrix cmap as a grayscale image in the axes ax
    and returns a handle to the plot.
    The handle can be used later to update the map like so:
    
    f, ax = plt.subplots()
    h = drawMap(ax, cmap)
    # update cmap
    h = drawMap(h,cmap)
  '''
  if type(ax) is AxesImage:
    # update image data
    ax.set_data(cmap)
  else:
    # setup image data for the first time
    # transpose because imshow places the first dimension on the y-axis
    h = ax.imshow( cmap.T, interpolation="none", cmap='gray_r', origin='lower', \
                   extent=(-0.5,cmap.shape[0]-0.5, -0.5, cmap.shape[1]-0.5) )
    ax.axis([-0.5, cmap.shape[0]-0.5, -0.5, cmap.shape[1]-0.5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return h

def drawPath2D(ax, traj):
    ''' 
    h = drawPath2D(h, traj)
    
    traj = num_traj x num_pts x num_dim
    '''
    if traj.ndim < 3:
        traj = traj[None, ...]

    if isinstance(ax, LineCollection):
        ax.set_verts(traj)
    else:
        colors = [mcolors.to_rgba(c)
                  for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        h = ax.add_collection(LineCollection(traj, colors=colors))
        
        # Extracting start and goal positions
        start = traj[0, 0, :]
        goal = traj[0, -1, :]

        # Mark the start and goal positions
        ax.plot(start[0], start[1], 'go', label='Start')  # Green circle for start
        ax.plot(goal[0], goal[1], 'bo', label='Goal')   # Red circle for goal

        # Adding a legend to differentiate start and goal
        ax.legend()

        return h

def drawPath3D(ax, traj):
  ''' h = drawPath3D(h,traj)
      
      traj = num_traj x num_pts x num_dim
  ''' 
  if(traj.ndim < 3):
    traj = traj[None,...]

  if type(ax) is Line3DCollection:
    ax.set_verts(traj)
  else:
    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    h = ax.add_collection(Line3DCollection(traj,colors=colors))
    return h
  
def plotClosedNodes(ax, sss, color='red', marker='x', label='Closed Nodes'):
  """
  Plots all closed nodes on the provided axes.
  
  Parameters:
      ax (matplotlib.axes.Axes): The axes on which to plot the closed nodes.
      sss (dict): The AStateSpace dictionary containing the closed_list and other data.
      color (str): The color of the closed nodes.
      marker (str): The marker style for the closed nodes.
      label (str): The label for the closed nodes in the plot legend.
  """
  closed_coords = [sss['hm'][key].coord for key in sss['closed_list']]
  
  if not closed_coords:
      print("No closed nodes to plot.")
      return

  # Extract x and y coordinates
  x_coords = [coord[0] for coord in closed_coords]
  y_coords = [coord[1] for coord in closed_coords]

  # Plot the closed nodes on the provided axes
  ax.scatter(x_coords, y_coords, c=color, marker=marker, label=label)

  return closed_coords


def plotInconsistentNodes(ax, sss, env):
    """
    Plots the nodes in the inconsistent list on the provided axis.

    Parameters:
    - sss: The state space structure containing the inconsistent list.
    - env: The environment object that provides grid size information.
    - ax: The matplotlib axis to plot on.
    """
    # Extract grid size
    grid_size = env.getSize()
    
    # Extract and plot inconsistent nodes
    inconsistent_coords = [node.coord for node in sss['il']]
    
    if inconsistent_coords:
        inconsistent_coords = np.array(inconsistent_coords)
        ax.scatter(inconsistent_coords[:, 0], inconsistent_coords[:, 1], 
                   c='blue', marker='x', label='Inconsistent Nodes')
    # else:
    #    print("No inconistent nodes")    

    ax.legend()
