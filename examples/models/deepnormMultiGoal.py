import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer

# Use current relative directory
import os
import sys
dirname = os.path.dirname(__file__)
sys.path.insert(1, dirname)
from spectralLayer import SpectralConv2d
from metrics.deepnorm import DeepNormMetric
from metrics.deepnorm import MaxReLUPairwiseActivation
from metrics.deepnorm import ConcaveActivation

class DEEPNORM2dMultiGoal(nn.Module):
    def __init__(self, nlayers, modes1, modes2, width):
        super(DEEPNORM2dMultiGoal, self).__init__()

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
        self.inp_size = 2
        self.nlayers = nlayers

        self.fc0 = nn.Linear(self.inp_size, self.width)

        for i in range(self.nlayers):
            self.add_module('conv%d' % i, SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.add_module('w%d' % i, nn.Conv2d(self.width, self.width, 1))

        self.fc1 =  DeepNormMetric(self.width, (128, 128), concave_activation_size=20, activation=lambda: MaxReLUPairwiseActivation(128), symmetric=True)

    def forward(self, chi, goal):
        batchsize = chi.shape[0]
        size_x = size_y = chi.shape[1]

        grid = self.get_grid(batchsize, size_x, size_y, chi.device)

        chi = chi.permute(0, 3, 1, 2)
        chi = chi.expand(batchsize, self.width, size_x, size_y)
        x = grid

        # Lifting layer:
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.nlayers):
            conv_chi = self._modules['conv%d' % i](chi)
            conv_chix = self._modules['conv%d' % i](chi * x)
            xconv_chi = x * conv_chi
            wx = self._modules['w%d' % i](x)
            x = chi * (conv_chix - xconv_chi + wx)
            if i < self.nlayers - 1: x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        g = x.clone()
    
        # find better way to do this
        goal_y_indices = goal[:, 1]
        goal_x_indices = goal[:, 0]
        g = x[torch.arange(batchsize), goal_y_indices, goal_x_indices, :].unsqueeze(1).repeat(1, size_x, 1).unsqueeze(1).repeat(1, size_x, 1, 1)
        
        feature1 = g
        feature2 = x
        reshapedfeature1 = feature1.reshape(-1,self.width)
        reshapedfeature2 = feature2.reshape(-1,self.width)
        output = self.fc1(reshapedfeature1,reshapedfeature2)
        reshapedoutput = output.reshape(batchsize,size_x,size_y,1)
        return reshapedoutput

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
