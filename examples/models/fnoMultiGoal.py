"""
@author: Zongyi Li
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use current relative directory
import os
import sys
dirname = os.path.dirname(__file__)
sys.path.insert(1, dirname)
from spectralLayer import SpectralConv2d

class FNO2dMultiGoal(nn.Module):
    def __init__(self, num_layers, padding, modes1, modes2,  width):
        super(FNO2dMultiGoal, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1, self.width) # input channel is 3: (a(x, y), x, y)
        self.num_layers = num_layers
        self.fc1 = torch.nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

        for i in range(self.num_layers):
            self.add_module('conv%d' % i, SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.add_module('w%d' % i, nn.Conv2d(self.width, self.width, 1))

    def forward(self, x, goal):
        # indicate goal with a -1
        for i in range(x.shape[0]):
            x[i][goal[i][1]][goal[i][0]] = -1
        
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        for i in range(self.num_layers):
            x1 = self._modules['conv%d' % i](x)
            x2 = self._modules['w%d' % i](x)
            x = x1+x2 
            x = F.gelu(x)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
