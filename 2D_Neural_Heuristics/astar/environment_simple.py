from astar.environment_abc import EnvironmentABC
import numpy as np

class Environment2D(EnvironmentABC):
  __U = np.array([[-1, -1, -1, 0, 0, 1, 1, 1],[-1, 0, 1, -1, 1, -1, 0, 1]])
  __iU = np.array([0, 1, 2, 3, 4, 5, 6, 7]) 
  __cU =  np.array([np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)])
  
  def __init__(self, goal_coord, cmap, valuefunction = None):
    self.__goal_coord = goal_coord
    self.__cmap = cmap
    self.__cdim = np.array([cmap.shape]).T
    self.valuefunction = valuefunction
    
  def isGoal(self, curr):
    return np.array_equal(curr,self.__goal_coord)
  

  def getGoal(self):
    return self.__goal_coord


  #def coord_to_idx(self, curr):
  #  return np.ravel_multi_index(curr, self.__cmap.shape)

  def getSuccessors(self, curr):
    # get neighbors
    succ = curr[:, None] + self.__U
    # remove neighbors outside of the map
    valid = np.all(np.logical_and(np.zeros((2,1)) <= succ, succ  < self.__cdim),axis = 0)
    succ = succ[:,valid]
    succ_cost = self.__cU[valid]
    action_idx = self.__iU[valid]
    # remove neighbors that hit obstacles
    valid = self.__cmap[succ[0,:],succ[1,:]] == 0
    succ = succ[:,valid]
    succ_cost = succ_cost[valid]
    #succ_idx = self.coord_to_idx(succ)
    action_idx = action_idx[valid]
    return succ, succ_cost, action_idx
    #return succ, succ_idx, succ_cost, action_idx

  def getHeuristic(self, curr):
    if self.valuefunction is None:
     return 0
    else:
      return self.valuefunction[curr[0],curr[1]]   

  def forwardAction(self, curr, action_id):
    return curr + self.__U[:,action_id]
  
  def getSize(self):
    return self.__cmap.shape


if __name__ == '__main__':
  print("Running small unit test...")
  myE = Environment2D(np.array([1,1]),np.array([[0, 1, 0],[0, 0, 0], [0,0,0]]))
  curr = np.array([0,1])
  print(myE.isGoal(curr)==False)
  print(np.array_equal(myE.forwardAction(curr, 0),np.array([-1,0])))
  succ, succ_cost, action_idx = myE.getSuccessors(curr)
  print(np.array_equal(succ,np.array([[0,0,1,1,1],[0,2,0,1,2]])))
  print(np.isclose(myE.getHeuristic(curr),1.0))


