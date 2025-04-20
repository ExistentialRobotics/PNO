from abc import ABC, abstractmethod

class EnvironmentABC(ABC):
    
    @abstractmethod
    def isGoal(self, curr):
      return True
    
    #@abstractmethod
    #def coord_to_idx(self, curr):
    #  return 0

    @abstractmethod
    def getSuccessors(self, curr):
      succ = [curr]
      #succ_idx = [coord_to_idx(curr)]
      succ_cost = [0.0]
      action_idx = [0]
      return succ, succ_cost, action_idx #succ, succ_idx, succ_cost, action_idx
    
    @abstractmethod
    def getHeuristic(self, curr):
      return 0.0
    
    #@abstractmethod
    #def forward_action(self, curr, action_id):
    #  next_micro = [curr]
    #  return next_micro

