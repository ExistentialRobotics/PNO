import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os

torch.manual_seed(0)
np.random.seed(0)

activation = F.relu

################################################################
# 3d fourier layers
################################################################

def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 3, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.irfft(out_ft, 3, normalized=True, onesided=True, signal_sizes=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes, modes, width)


    def forward(self, x):
        print("Forward Input Shape:",x.shape)
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
    
def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))   

def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


################################################################
# configs
################################################################
if __name__ == '__main__':
    print("Started Script")
    os.chdir("/mountvol/dataset-80-15g")
    lrs = [3e-3]
    gammas = [0.8]
    wds = [3e-6]
    smooth_coefs = [5.]
    smooth_coef = smooth_coefs[0]
    # experiments to be replicated with different seeds
    seeds = [5, 2000, 14000, 16000, 100000]
    seeds = [seeds[0]]

    ################################################################
    #                       configs
    ################################################################
   
    batch_size = 20

    epochs = 801
    scheduler_step = 100
    tol_early_stop = 800

    modes = 5
    width = 6

    ################################################################
    # load data and data normalization
    ################################################################
    t1 = default_timer()

    sub = 1
    Sx = int(((80 - 1) / sub) + 1)
    Sy = Sx
    Sz = int(((30 - 1) / sub) + 1)

    Ntotal = 32+8
    ntrain = 32
    ntest =  8

    print("Loading Data.......")
    mask = np.load('mask.npy')
    mask = torch.tensor(mask, dtype=torch.float)
    output = np.load('dist_in.npy')
    output = torch.tensor(output, dtype=torch.float)
    print("Data Loaded!")

    print(mask.shape)

    # Selecting every 15th mask
    mask_train = mask[:ntrain*15:15, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]
    mask_test = mask[-ntest*15::15, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]

    print(mask_train.shape)

    # Reshape to add the channel dimension
    mask_train = mask_train.reshape(ntrain, Sx, Sy, Sz, 1)
    mask_test = mask_test.reshape(ntest, Sx, Sy, Sz, 1)

    # Similarly for output (y_train and y_test)
    y_train = output[:ntrain*15:15, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]
    y_test = output[-ntest*15::15, ::sub, ::sub, ::sub][:, :Sx, :Sy, :Sz]


    y_normalizer = GaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    y_train = y_train.reshape(ntrain, Sx, Sy, Sz, 1)
    y_test = y_test.reshape(ntest, Sx, Sy, Sz, 1)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_train,y_train),
                                                batch_size=batch_size,
                                                shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(mask_test, y_test),
                                                batch_size=batch_size,
                                                shuffle=False)


    print("Training Started")
    op_type = 'igibsonenv_sdf_80_m8_w18_l2_b20_lr3e-3_18sep'
    res_dir = './sdfoperator3D_%s' % op_type
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

    t2 = default_timer()

    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda')

    ################################################################
    # training and evaluation
    ################################################################
   
    # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

  


    ################################################################
    #                      train and eval
    ################################################################
    myloss = LpLoss(size_average=False)
    print("-" * 100)
    model  = Net2d(modes, width).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lrs[0], weight_decay=1e-4)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    print(f'>> Total number of model parameters: {model.count_params()}')

    ttrain, ttest = [], []
    best_train_loss = best_test_loss = 1e8
    best_epoch = 0
    early_stop = 0

    for learning_rate in lrs:
            for scheduler_gamma in gammas:
                for wd in wds:
                    for isd in seeds:
                        torch.manual_seed(isd)
                        torch.cuda.manual_seed(isd)
                        np.random.seed(isd)

                        print(f'>> random seed: {isd}')

                        base_dir = './sdfoperator_%s/n%d_lr%e_gamma%e_wd%e_seed%d' % (op_type, ntrain, learning_rate,
                                                                                scheduler_gamma, wd, isd)
                        if not os.path.exists(base_dir):
                            os.makedirs(base_dir)

                        ################################################################
                        #                      train and eval
                        ################################################################
                        myloss = LpLoss(size_average=False)
                        print("-" * 100)
                        model = Net2d(modes, width).to(device)

                        # if torch.cuda.device_count() > 1:
                        #     model = nn.DataParallel(model)


                        print(f'>> Total number of model parameters: {model.count_params()}')

                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
                        model_filename = '%s/model3dsdf.ckpt' % base_dir

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
                            for xx, yy in train_loader:
                                xx, yy = xx.to(device), yy.to(device)

                                optimizer.zero_grad()
                                out = model(xx)
                                out = y_normalizer.decode(out) * xx
                                yy = y_normalizer.decode(yy) * xx

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
                                    for xx, yy in test_loader:
                                        xx, yy =  xx.to(device), yy.to(device)

                                        out = model(xx)

                                        out = y_normalizer.decode(out) * xx
                                        yy *= y_normalizer.decode(yy)  *xx

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


