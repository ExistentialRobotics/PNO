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



class DAFNO2dMultiGoal(nn.Module):
    def __init__(self, num_layers, modes1, modes2,  width):
        super(DAFNO2dMultiGoal, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 3: (a(x, y), x, y)
        self.num_layers = num_layers

        for i in range(self.num_layers):
            self.add_module('conv%d' % i, SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.add_module('w%d' % i, nn.Conv2d(self.width, self.width, 1))

        self.fc1 = torch.nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, chi, goal):
        # indicate goal with a -1
        for i in range(chi.shape[0]):
            chi[i][goal[i][1]][goal[i][0]] = -1
            
        batchsize = chi.shape[0]
        size_x = size_y = chi.shape[1]

        grid = self.get_grid(batchsize, size_x, size_y, chi.device)

        chi = chi.permute(0, 3, 1, 2)
        chi = chi.expand(batchsize, self.width, size_x, size_y)
        x = grid

        # Lifting layer:
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.num_layers):
            conv_chi = self._modules['conv%d' % i](chi)
            conv_chix = self._modules['conv%d' % i](chi * x)
            xconv_chi = x * conv_chi
            wx = self._modules['w%d' % i](x)
            x = chi * (conv_chix - xconv_chi + wx)
            if i < self.num_layers - 1: x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)

        return x

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
