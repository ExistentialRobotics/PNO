"""
This code uses the Planning Operator on the Maze dataset described in the paper "Planning Operator: Generalizable Robot Motion Planning via Operator Learning"
"""
import os
import gc
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

# Add the current script directory to sys.path
current_script_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_script_path)

if current_folder_path not in sys.path:
    sys.path.append(current_folder_path)

import matplotlib.pyplot as plt
from utilities import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
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
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class PlanningOperator3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, nlayers):
        super(PlanningOperator3D, self).__init__()

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
        self.modes3 = modes3
        self.width = width
        self.inp_size = 3
        self.nlayers = nlayers

        self.fc0 = nn.Linear(self.inp_size, self.width)

        for i in range(self.nlayers):
            self.add_module('conv%d' % i, SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3))
            self.add_module('w%d' % i, nn.Conv3d(self.width, self.width, 1))

        self.fc1 =  DeepNormMetric(self.width, (128, 128), concave_activation_size=20, activation=lambda: MaxReLUPairwiseActivation(128), symmetric=True)

    def forward(self, chi, gs):
        batchsize = chi.shape[0]
        size_x = chi.shape[1]
        size_y = chi.shape[2]
        size_z = chi.shape[3]

        grid = self.get_grid(batchsize, size_x, size_y, size_z, chi.device)


        chi = chi.permute(0, 4, 1, 2, 3)
        chi = chi.expand(batchsize, self.width, size_x, size_y, size_z)
        x = grid

        # Lifting layer:
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i in range(self.nlayers):
            conv_chi = self._modules['conv%d' % i](chi)
            conv_chix = self._modules['conv%d' % i](chi * x)
            xconv_chi = x * conv_chi
            wx = self._modules['w%d' % i](x)
            x = chi * (conv_chix - xconv_chi + wx)
            if i < self.nlayers - 1: x = F.gelu(x)

        x = x.permute(0, 2, 3, 4, 1)
        g = x.clone()

        batch_indices = torch.arange(batchsize, device=gs.device)  # Ensure indices are on the same device
        x_indices = gs[:, 0, 0].long()  # Shape: [batchsize]
        y_indices = gs[:, 1, 0].long()  # Shape: [batchsize]
        z_indices = gs[:, 2, 0].long()  # Shape: [batchsize]

        g = x[batch_indices, x_indices, y_indices, z_indices, :] 

        # If you need to reshape g to match the original 5D shape of [batchsize, D1, D2, D3, C], repeat it:
        g = g.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, size_x, size_y, size_z, 1)

        x = x.reshape(-1,self.width)
        g = g.reshape(-1,self.width)

        # Projection Layer
        output = self.fc1(x,g)
        # print(output.shape)
        output = output.reshape(batchsize,size_x,size_y,size_z,1)
        # output = torch.randn(batchsize, size_x, size_y, size_z, 1)

        return output

    def get_grid(self, batchsize, size_x, size_y, size_z, device):
        gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        
        gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        
        gridz = torch.tensor(np.linspace(-1, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


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
    os.chdir("/mountvol/igib-dataset-160-5G")

    lrs = [3e-3]
    gammas = [0.6]
    wds = [3e-6]
    smooth_coefs = [5.]
    smooth_coef = smooth_coefs[0]
    # experiments to be replicated with different seeds
    seeds = [5, 2000, 14000, 16000, 100000]
    seeds = [seeds[0]]

    ################################################################
    #                       configs
    ################################################################
    Ntotal = 400
    ntrain = 320
    ntest =  80

    batch_size = 5

    epochs = 401
    scheduler_step = 100
    tol_early_stop = 400

    modes = 3
    width = 8
    nlayers = 1

    ################################################################
    # load data and data normalization
    ################################################################
    t1 = default_timer()

    sub = 1
    Sx = int(((160 - 1) / sub) + 1)
    Sy = Sx
    Sz = int(((62 - 1) / sub) + 1)

    print("Loading Data.......")
    mask = np.load('mask.npy')[:Ntotal,:,:,:]
    mask = torch.tensor(mask, dtype=torch.float)
    dist_in = np.load('dist_in.npy')[:Ntotal,:,:,:]
    dist_in = torch.tensor(dist_in[:Ntotal, :, :, :], dtype=torch.float)
    input = smooth_chi(mask, dist_in, smooth_coef)
    output = np.load('output.npy')[:Ntotal,:,:,:]
    output = torch.tensor(output, dtype=torch.float)

    goals = np.load('goals.npy')[:Ntotal,:]
    goals = torch.tensor(goals, dtype=torch.float)
    print("Data Loaded!")


    mask_train = mask[:Ntotal][:ntrain, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]
    mask_test = mask[:Ntotal][-ntest:, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]

    mask_train = mask_train.reshape(ntrain, Sx, Sy, Sz, 1)
    mask_test = mask_test.reshape(ntest, Sx, Sy, Sz, 1)

    chi_train = input[:Ntotal][:ntrain, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]
    chi_test = input[:Ntotal][-ntest:, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]

    chi_train = chi_train.reshape(ntrain, Sx, Sy, Sz, 1)
    chi_test = chi_test.reshape(ntest, Sx, Sy, Sz,1)

    y_train = output[:Ntotal][:ntrain, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]
    y_test = output[:Ntotal][-ntest:, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]

    y_train = y_train.reshape(ntrain, Sx, Sy, Sz, 1)
    y_test = y_test.reshape(ntest, Sx, Sy, Sz, 1)

    goals_train = goals[:ntrain]
    goals_test = goals[-ntest:]

    goals_train = goals_train.reshape(ntrain, 3, 1)
    goals_test = goals_test.reshape(ntest, 3, 1)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_train, chi_train, y_train, goals_train),
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_test, chi_test, y_test, goals_test),
                                              batch_size=batch_size,
                                              shuffle=False)
    
    print("Training Started")
    op_type = 'env160_m12_w32_l1_b5_lr3e-3_5g_30nov_2400'
    res_dir = './planningoperator3D_%s' % op_type
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
                    myloss = LpLoss(size_average=False, d=3)
                    print("-" * 100)
                    model = PlanningOperator3D(modes, modes, modes, width, nlayers).to(device)

                    # if torch.cuda.device_count() > 1:
                    #     model = nn.DataParallel(model)


                    print(f'>> Total number of model parameters: {count_params(model)}')

                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
                    model_filename = '%s/model3d.ckpt' % base_dir

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

                            # print(yy)

                            optimizer.zero_grad()
                            out = model(xx,gg)
                            # print(out[0,40,40,10:13,0])

                            out = out*mm
                            yy= yy*mm
                            # print(out[0,40,40,10:13,0])

                            grad_out = torch.linalg.vector_norm(
                                torch.stack(
                                    list(
                                        torch.gradient(
                                            out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3]),
                                            dim=[1, 2, 3]
                                        )
                                    ),
                                    dim=0
                                ),
                                dim=0
                            )
                            grad_y = torch.linalg.vector_norm(
                                torch.stack(
                                    list(
                                        torch.gradient(
                                            yy.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3]),
                                            dim=[1, 2, 3]
                                        )
                                    ),
                                    dim=0
                                ),
                                dim=0
                            )
                            loss1 = myloss(out, yy)
                            loss2 = myloss(
                                grad_out.view(batch_size, yy.shape[1], yy.shape[2], yy.shape[3]),
                                grad_y.view(batch_size, yy.shape[1], yy.shape[2], yy.shape[3])
                            )

                            loss = loss1 + loss2
                            train_l2 += loss.item()
                            # print(loss)

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

                                    grad_out = torch.linalg.vector_norm(
                                        torch.stack(
                                            list(
                                                torch.gradient(
                                                    out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3]),
                                                    dim=[1, 2, 3]
                                                )
                                            ),
                                            dim=0
                                        ),
                                        dim=0
                                    )
                                    grad_y = torch.linalg.vector_norm(
                                        torch.stack(
                                            list(
                                                torch.gradient(
                                                    yy.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3]),
                                                    dim=[1, 2, 3]
                                                )
                                            ),
                                            dim=0
                                        ),
                                        dim=0
                                    )

                                    loss1 = myloss(out, yy)
                                    loss2 = myloss(
                                        grad_out.view(batch_size, yy.shape[1], yy.shape[2], yy.shape[3]),
                                        grad_y.view(batch_size, yy.shape[1], yy.shape[2], yy.shape[3])
                                    )

                                    loss = loss1 + loss2
                                    test_l2 += loss.item()

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

                        gc.collect()
                        torch.cuda.empty_cache()
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
