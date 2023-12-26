import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append('..')
from data.dataset import HRRR
from torch.utils.data import DataLoader


class Loss(nn.Module):
    def __init__(self, sj, wj, ai=None):
        super().__init__()
        self.sj = sj
        self.wj = wj
        self.ai = ai

    def forward(self, predict, target, T=1):
        # predict:[batch, location, feature]
        # sj:[feature]
        # wj:[feature]
        # ai:[location]
        len_features = len(self.sj)
        len_locations = len(self.ai)
        B, L, F = predict.shape
        self.sj = torch.reshape(self.sj, [1, 1, len_features]).expand([B, L, -1])
        self.wj = torch.reshape(self.wj, [1, 1, len_features]).expand([B, L, -1])
        # self.ai = torch.reshape(self.ai, [1, len_locations, 1]).expand([B, -1, F])

        # loss = self.sj * self.wj * self.ai * torch.pow((predict - target), 2)
        loss = self.sj * self.wj * torch.pow((predict - target), 2)
        print(loss)
        loss = 1 / (B * T * L) * torch.sum(loss)

        return loss           

def get_sj():
    hrrr = HRRR('/ssd1/hrrr_data/train', 1, 1)
    data_loader = DataLoader(hrrr, batch_size=2000)
    diffs = []
    device = torch.device('cuda')
    H_mean = 0
    H_var = 0
    H_num = 0
    for input, label, _, _ in data_loader:
        input.to(device)
        label.to(device)
        diff = label-input
        # [num_grid, features]
        B, N, F = diff.shape
        diff = torch.reshape(diff, [B * N, F])
        A_mean = torch.mean(diff, dim=0)
        A_num = B * N
        A_var = torch.var(diff, dim=0)
        X_mean = (H_mean * H_num + A_mean * A_num) / (H_num + A_num)
        X_var = (H_num * (H_var + torch.pow(X_mean - H_mean, 2)) + A_num * (A_var + torch.pow(X_mean - A_mean, 2))) / (H_num + A_num)
        print(X_var)
        H_num = H_num + A_num
        H_var = X_var
        H_mean = X_mean
    
    sj = H_var
    np.save('/home/wwh/graphcast/location/sj.npy', sj)
# get_sj()
# def get_ai(latitude):

def get_wj():
    pressure = np.array([50., 500, 850, 1000])
    pressure_normalization = pressure / np.sum(pressure)
    pressure_normalization = np.repeat(pressure_normalization, 5)
    surface = np.array([0.1, 0.1, 0.1, 1])
    wj = np.concatenate([pressure_normalization, surface])
    np.save('/home/wwh/graphcast/location/wj.npy', wj)

# def get_Cij():
    #valid = 


# a1 = torch.arange(24).reshape([2, 1, 3, 4])
# a2 = torch.arange(1, 25).reshape([2, 1, 3, 4])
# print('a1', a1)
# print('a2:', a2)
# s1 = torch.arange(4)
# s2 = torch.arange(4)
# s3 = torch.arange(3)
# loss = Loss(s1, s2, s3)
# loss1 = loss(a1, a2)
# print(loss1)
    