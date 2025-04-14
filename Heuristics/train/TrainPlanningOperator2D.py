"""
This code uses the Planning Operator on the Maze dataset described in the paper "Planning Operator: Generalizable Robot Motion Planning via Operator Learning"
"""

import os
import sys

# Add the current script directory to sys.path
current_script_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_script_path)

if current_folder_path not in sys.path:
    sys.path.append(current_folder_path)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

import matplotlib.pyplot as plt
from utilities import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os
import sys
from itertools import chain

# Helpers:

class ConstrainedLinear(nn.Linear):
  def forward(self, x):
    return F.linear(x, torch.min(self.weight ** 2, torch.abs(self.weight)))

# Activations:

class MaxReLUPairwiseActivation(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.weights = nn.Parameter(torch.zeros(1, num_features))
    self.avg_pool = nn.AvgPool1d(2, 2)

  def forward(self, x):
    x = x.unsqueeze(1)
    max_component = F.max_pool1d(x, 2)
    relu_component = F.avg_pool1d(F.relu(x * F.softplus(self.weights)), 2)
    output = torch.cat((max_component, relu_component), dim=-1).squeeze(1)
    return output


class MaxAvgGlobalActivation(nn.Module):
  def __init__(self):
    super().__init__()
    self.alpha = nn.Parameter(-torch.ones(1))

  def forward(self, x):
    alpha = torch.sigmoid(self.alpha)
    return alpha * x.max(dim=-1)[0] + (1 - alpha) * x.mean(dim=-1)


class MaxPoolPairwiseActivation(nn.Module):
  def forward(self, x):
    x = x.unsqueeze(1)
    x = F.max_pool1d(x, 2)
    return x.squeeze(1)


class ConcaveActivation(nn.Module):
  def __init__(self, num_features, concave_activation_size):
    super().__init__()
    assert concave_activation_size > 1

    self.bs_nonzero = nn.Parameter(1e-3 * torch.randn((1, num_features, concave_activation_size - 1)) - 1)
    self.bs_zero    = torch.zeros((1, num_features, 1))
    self.ms = nn.Parameter(1e-3 * torch.randn((1, num_features, concave_activation_size)))

  def forward(self, x):
    bs = torch.cat((F.softplus(self.bs_nonzero), self.bs_zero), -1)
    ms = 2 * torch.sigmoid(self.ms)
    x = x.unsqueeze(-1)

    x = x * ms + bs
    return x.min(-1)[0]


# Metrics:

class ReduceMetric(nn.Module):
  def __init__(self, mode):
    super().__init__()
    if mode == 'avg':
      self.forward = self.avg_forward
    elif mode == 'max':
      self.forward = self.max_forward
    elif mode == 'maxavg':
      self.maxavg_activation = MaxAvgGlobalActivation()
      self.forward = self.maxavg_forward
    else:
      raise NotImplementedError

  def maxavg_forward(self, x):
    return self.maxavg_activation(x)

  def max_forward(self, x):
    return x.max(-1)[0]

  def avg_forward(self, x):
    return x.mean(-1)


class EuclideanMetric(nn.Module):
  def forward(self, x, y):
    return torch.norm(x - y, dim=-1)


class MahalanobisMetric(nn.Module):
  def __init__(self, num_features, size):
    super().__init__()
    self.layer = nn.Linear(num_features, size, bias=False)

  def forward(self, x, y):
    return torch.norm(self.layer(x - y), dim=-1)


class WideNormMetric(nn.Module):
  def __init__(self,
               num_features,
               num_components,
               component_size,
               concave_activation_size=None,
               mode='avg',
               symmetric=True):
    super().__init__()
    self.symmetric = symmetric
    self.num_components = num_components
    self.component_size = component_size

    output_size = component_size*num_components
    if not symmetric:
      num_features = num_features * 2
      self.f = ConstrainedLinear(num_features, output_size)
    else:
      self.f = nn.Linear(num_features, output_size)
      
    self.activation = ConcaveActivation(num_components, concave_activation_size) if concave_activation_size else nn.Identity()
    self.reduce_metric = ReduceMetric(mode)

  def forward(self, x, y):
    h = x - y
    if not self.symmetric:
      h = torch.cat((F.relu(h), F.relu(-h)), -1)
    h = torch.reshape(self.f(h), (-1, self.num_components, self.component_size))
    h = torch.norm(h, dim=-1)
    h = self.activation(h)
    return self.reduce_metric(h)


class DeepNormMetric(nn.Module):
  def __init__(self, num_features, layers, activation=nn.ReLU, concave_activation_size=None, mode='avg', symmetric=False):
    super().__init__()
    self.num_features = num_features

    assert len(layers) >= 2

    self.Us = nn.ModuleList([nn.Linear(num_features, layers[0], bias=False)])
    self.Ws = nn.ModuleList([])

    for in_features, out_features in zip(layers[:-1], layers[1:]):
      self.Us.append(nn.Linear(num_features, out_features, bias=False))
      self.Ws.append(ConstrainedLinear(in_features, out_features, bias=False))

    self.activation = activation()
    self.output_activation = ConcaveActivation(layers[-1], concave_activation_size) if concave_activation_size else nn.Identity()
    self.reduce_metric = ReduceMetric(mode)

    self.symmetric = symmetric

  def _asym_fwd(self, h):
    h1 = self.Us[0](h)
    for U, W in zip(self.Us[1:], self.Ws):
      h1 = self.activation(W(h1) + U(h))
    return h1

  def forward(self, x, y):
    h = x - y
    if self.symmetric:
      h = self._asym_fwd(h) + self._asym_fwd(-h)
    else:
      h = self._asym_fwd(-h)
    h = self.activation(h)
    return self.reduce_metric(h)


################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class PlanningOperator2D(nn.Module):
    def __init__(self, modes1, modes2, width, nlayers):
        super(PlanningOperator2D, self).__init__()

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

    def forward(self, chi, gs):
        batchsize = chi.shape[0]
        size_x = size_y = chi.shape[1]

        assert(gs.shape[0]==batchsize)

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

        batch_indices = torch.arange(batchsize, device=gs.device)  
        x_indices = gs[:, 0, 0].long()
        y_indices = gs[:, 1, 0].long()  
        g = x[batch_indices, x_indices, y_indices, :]  
        g = g.unsqueeze(1).unsqueeze(1).repeat(1, size_x, size_y, 1)

        g = g.reshape(-1,self.width)
        x = x.reshape(-1,self.width)

        # Projection layer:
        output = self.fc1(x,g)
        output = output.reshape(batchsize,size_x,size_y,1)

        return output

    def get_grid(self, batchsize, size_x, size_y, device):
        gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float, requires_grad=True)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float, requires_grad=True)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def smooth_chi(mask, dist, smooth_coef):
    return torch.mul(torch.tanh(dist * smooth_coef), (mask - 0.5)) + 0.5


if __name__ == '__main__':
    # define hyperparameters
    print("Started Script")
    os.chdir("/mountvol/2D-1024-Dataset-0")
    lrs = [5e-3]
    gammas = [0.5]
    wds = [3e-6]
    smooth_coefs = [5.]
    smooth_coef = smooth_coefs[0]
    # experiments to be replicated with different seeds
    seeds = [5, 2000, 14000, 16000, 100000]
    seeds = [seeds[0]]

    ################################################################
    #                       configs
    ################################################################
    Ntotal = 550
    ntrain = 500

    ntest = Ntotal-ntrain
    batch_size = 5

    epochs = 401
    scheduler_step = 100
    tol_early_stop = 400

    modes = 8
    width = 28
    nlayers = 1

    ################################################################
    # load data and data normalization
    ################################################################
    t1 = default_timer()

    sub = 1
    Sx = int(((1024 - 1) / sub) + 1)
    Sy = Sx

    print("Loading Data......")

    mask = np.load('mask.npy')
    mask = torch.tensor(mask, dtype=torch.float)
    dist_in = np.load('dist_in.npy')
    dist_in = torch.tensor(dist_in[:Ntotal, :, :], dtype=torch.float)
    input = smooth_chi(mask, dist_in, smooth_coef)
    output = np.load('output.npy')
    output = torch.tensor(output, dtype=torch.float)

    goals = np.load('goals.npy')
    goals = torch.tensor(goals, dtype=torch.float)

    print("Data Loaded!")



    mask_train = mask[:Ntotal][:ntrain, ::sub, ::sub][:, :Sx, :Sy]
    mask_test = mask[:Ntotal][-ntest:, ::sub, ::sub][:, :Sx, :Sy]

    mask_train = mask_train.reshape(ntrain, Sx, Sy, 1)
    mask_test = mask_test.reshape(ntest, Sx, Sy, 1)

    chi_train = input[:Ntotal][:ntrain, ::sub, ::sub][:, :Sx, :Sy]
    chi_test = input[:Ntotal][-ntest:, ::sub, ::sub][:, :Sx, :Sy]

    chi_train = chi_train.reshape(ntrain, Sx, Sy, 1)
    chi_test = chi_test.reshape(ntest, Sx, Sy, 1)

    y_train = output[:Ntotal][:ntrain, ::sub, ::sub][:, :Sx, :Sy]
    y_test = output[:Ntotal][-ntest:, ::sub, ::sub][:, :Sx, :Sy]

    y_train = y_train.reshape(ntrain, Sx, Sy, 1)
    y_test = y_test.reshape(ntest, Sx, Sy, 1)

    goals_train = goals[:ntrain]
    goals_test = goals[-ntest:]

    goals_train = goals_train.reshape(ntrain, 2, 1)
    goals_test = goals_test.reshape(ntest, 2, 1)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_train, chi_train, y_train, goals_train),
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_test, chi_test, y_test, goals_test),
                                              batch_size=batch_size,
                                              shuffle=False)

    op_type = 'street_maps_1024x1024_8m_28w_1l_b5'
    res_dir = './planningoperator2D_%s' % op_type
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    f = open("%s/n%d.txt" % (res_dir, ntrain), "w")
    f.write(f'ntrain, seed, learning_rate, scheduler_gamma, weight_decay, smooth_coef, '
            f'best_train_loss, best_valid_loss, best_epoch\n')
    f.close()

    t2 = default_timer()
    print(f'>> Preprocessing finished, time used: {(t2 - t1):.2f}s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    for learning_rate in lrs:
        for scheduler_gamma in gammas:
            for wd in wds:
                for isd in seeds:
                    torch.manual_seed(isd)
                    torch.cuda.manual_seed(isd)
                    np.random.seed(isd)

                    print(f'>> random seed: {isd}')

                    base_dir = './planningoperator_%s/n%d_lr%e_gamma%e_wd%e_seed%d' % (op_type, ntrain, learning_rate,
                                                                                scheduler_gamma, wd, isd)
                    if not os.path.exists(base_dir):
                        os.makedirs(base_dir)

                    ################################################################
                    #                      train and eval
                    ################################################################
                    myloss = LpLoss(size_average=False)
                    print("-" * 100)
                    model = PlanningOperator2D(modes, modes, width, nlayers).to(device)
                    print(f'>> Total number of model parameters: {count_params(model)}')

                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
                    model_filename = '%s/model1024.ckpt' % base_dir

                    ttrain, ttest = [], []
                    best_train_loss = best_test_loss = 1e8
                    best_epoch = 0
                    early_stop = 0
                    for ep in range(epochs):
                        t1 = default_timer()
                        optimizer = scheduler(optimizer,
                                              LR_schedule(learning_rate, ep, scheduler_step, scheduler_gamma))
                        model.train()
                        train_l2 = 0
                        for mm, xx, yy, gg in train_loader:
                            mm, xx, yy, gg = mm.to(device), xx.to(device), yy.to(device), gg.to(device)

                            optimizer.zero_grad()
                            out = model(xx,gg)

                            out = out*mm
                            yy= yy*mm

                            loss = myloss(out, yy)
                            train_l2 += loss.item()

                            loss.backward()
                            optimizer.step()

                        train_l2 /= ntrain
                        ttrain.append([ep, train_l2])

                        if train_l2 < best_train_loss:
                            model.eval()
                            test_l2 = 0
                            with torch.no_grad():
                                for mm, xx, yy ,gg in test_loader:
                                    mm, xx, yy , gg= mm.to(device), xx.to(device), yy.to(device), gg.to(device)

                                    out = model(xx, gg)

                                    out = out*mm
                                    yy *= mm

                                    test_l2 += myloss(out, yy).item()

                            test_l2 /= ntest
                            ttest.append([ep, test_l2])
                            if test_l2 < best_test_loss:
                                early_stop = 0
                                best_train_loss = train_l2
                                best_test_loss = test_l2
                                best_epoch = ep
                                torch.save(model.state_dict(), model_filename)
                                t2 = default_timer()
                                print(f'>> s: {smooth_coef}, '
                                      f'epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: {(t2 - t1):.2f}s, '
                                      f'train loss: {train_l2:.5f}, test loss: {test_l2:.5f}')
                            else:
                                early_stop += 1
                                t2 = default_timer()
                                print(f'>> s: {smooth_coef}, '
                                      f'epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}](best:{best_epoch + 1}), '
                                      f'runtime: {(t2 - t1):.2f}s, train loss: {train_l2:.5f} (best: '
                                      f'{best_train_loss:.5f}/{best_test_loss:.5f})')

                        else:
                            early_stop += 1
                            t2 = default_timer()
                            print(f'>> s: {smooth_coef}, '
                                  f'epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}](best:{best_epoch + 1}), '
                                  f'runtime: {(t2 - t1):.2f}s, train loss: {train_l2:.5f} (best: '
                                  f'{best_train_loss:.5f}/{best_test_loss:.5f})')

                        if early_stop > tol_early_stop: break

                    with open('%s/loss_train.txt' % base_dir, 'w') as file:
                        np.savetxt(file, ttrain)
                    with open('%s/loss_test.txt' % base_dir, 'w') as file:
                        np.savetxt(file, ttest)

                    print("-" * 100)
                    print("-" * 100)
                    print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, weight decay: {wd}')
                    print(f'>> Best train error: {best_train_loss:.5f}')
                    print(f'>> Best validation error: {best_test_loss:.5f}')
                    print(f'>> Best epochs: {best_epoch}')
                    print("-" * 100)
                    print("-" * 100)

                    f = open("%s/n%d.txt" % (res_dir, ntrain), "a")
                    f.write(f'{ntrain}, {isd}, {learning_rate}, {scheduler_gamma}, {wd}, {smooth_coef}, '
                            f'{best_train_loss}, {best_test_loss}, {best_epoch}\n')
                    f.close()

            print(f'********** Training completed! **********')
