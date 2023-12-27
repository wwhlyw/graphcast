# {0: 20657.43289983611, 1: 5751.477673070288, 2: 1515.681617375804, 3: 147.89313015520383, 
#  4: 211.5233345934269, 5: 260.2125557439184, 6: 283.05791319356734, 7: 289.3527680646734, 
#  8: 3.9980989888479385e-06, 9: 0.0011294531919208942, 10: 0.006366958498976991, 11: 0.00918317163877328, 
#  12: 2.911837106393454, 13: 14.170063293266587, 14: 4.3964357904958495, 15: 0.1782575642922985, 
#  16: 0.1847573400469163, 17: 1.391608085358267, 18: 1.5916701674730873, 19: 0.3625728877539512, 
#  20: 101733.0130280215, 21: 0.2636278827605439, 22: 0.31753442026055945, 23: 288.1850850679897}
# [2.0657781e+04 5.7515933e+03 1.5157103e+03 1.4789497e+02 2.1152121e+02
#  2.6021439e+02 2.8306021e+02 2.8935880e+02 3.9991646e-06 1.1292103e-03
#  6.3678054e-03 9.1851633e-03 2.9089484e+00 1.4158347e+01 4.3943605e+00
#  1.7851296e-01 1.8370540e-01 1.3803493e+00 1.5891752e+00 3.6263445e-01
#  1.0173423e+05 2.6379207e-01 3.1767491e-01 2.8818991e+02]


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import pandas as pd


class HRRR(Dataset):
    def __init__(
            self, 
            file_path: str,
            num_input_timestamps: int = 1,
            num_label_timestamps: int = 1,
        ):
        self.file_path = file_path
        self.num_input_timestamps = num_input_timestamps
        self.num_label_timestamps = num_label_timestamps
        self.files = self._read_data()
        self.num_days = len(self.files)
        self.num_samples_per_day = 24
        self.num_samples = self.num_days * self.num_samples_per_day
        self.std = StandardScaler()
        self.timefeature = TimeFeature()

    def __len__(self):
        return self.num_samples - self.num_label_timestamps

    def _read_data(self):
        paths = []
        for root, dirs, files in os.walk(self.file_path):
            for file in files:
                paths.append(os.path.join(root, file))
        files = []
        for path in paths:
            file = h5py.File(path, 'r')
            files.append([file['fields'], path[-13: -3]])
        return files

    def set_time(self, time):
        time = pd.to_datetime(time, format='%Y/%m/%d/%H')

        time_feature = np.array(self.timefeature(time))
        time_feature = torch.from_numpy(time_feature.squeeze())

        return time_feature


    def __getitem__(self, global_idx):
        if global_idx >= self.num_samples - self.num_label_timestamps:
            return self.__getitem__(np.random.randint(self.__len__()))
        
        input_item_list = []
        input_time_list = []
        label_item_list = []
        label_time_list = []

        for i in range(self.num_input_timestamps):
            input_day_idx = (global_idx + i) // self.num_samples_per_day
            input_hour_idx = (global_idx + i) % self.num_samples_per_day
            input_file = self.files[input_day_idx][0]
            if len(input_file.shape) == 1:
                return self.__getitem__(np.random.randint(self.__len__()))
            input_item = input_file[input_hour_idx]
            input_time = [self.files[input_day_idx][1] + "/" + str(input_hour_idx)]
            input_time = self.set_time(input_time)
            input_item_list.append(input_item)
            input_time_list.append(input_time)
        for j in range(self.num_label_timestamps):
            label_day_idx = (global_idx + j + 1) // self.num_samples_per_day
            label_hour_idx = (global_idx + j + 1) % self.num_samples_per_day
            label_file = self.files[label_day_idx][0]
            if len(label_file.shape) == 1:
                return self.__getitem__(np.random.randint(self.__len__()))
            label_item = label_file[label_hour_idx]
            label_time = [self.files[label_day_idx][1] + "/" + str(label_hour_idx)]
            label_time = self.set_time(label_time)
            label_item_list.append(label_item)
            label_time_list.append(label_time)

        input_item_list = np.stack(input_item_list, axis=0)
        input_time_list = np.stack(input_time_list, axis=0)
        label_item_list = np.stack(label_item_list, axis=0)
        label_time_list = np.stack(label_time_list, axis=0)

        input_item_list = self.std.transform(input_item_list)
        label_item_list = self.std.transform(label_item_list)
        
        T, V, H, W = input_item_list.shape
        T1, V1, H1, W1 = label_item_list.shape
        input_item_list = np.reshape(input_item_list, [T*V, H*W]).transpose([1, 0])
        label_item_list = np.reshape(label_item_list, [T1*V1, H1*W1]).transpose([1, 0])
        input_time_list = np.reshape(input_time_list, [-1])
        label_time_list = np.reshape(label_time_list, [-1])

        return input_item_list, label_item_list, input_time_list, label_time_list


class StandardScaler:
    def __init__(self):
        self.mean = np.load('/ssd1/hrrr_data/stat/mean_crop.npy').reshape(-1, 1, 1).astype(np.float32)
        self.std = np.load('/ssd1/hrrr_data/stat/std_crop.npy').reshape(-1, 1, 1).astype(np.float32)

    def transform(self, input_data):
        return (input_data - self.mean) / self.std
    
    def inverse_transform(self, inverse_data):
        return inverse_data * self.std + self.mean 
    

class TimeFeature:
    def __call__(self, date: pd.DatetimeIndex):
        self.hourofday = self.HourOfDay(date)
        self.dayofyear = self.DayOfYear(date)
        return self.hourofday, self.dayofyear
    
    def HourOfDay(self, date):
        return date.hour / 23.0 - 0.5
    
    def DayOfYear(self, date):
        return (date.dayofyear - 1) / 365 - 0.5

# filePath = '/ssd1/hrrr_data/train'
# data = HRRR(file_path=filePath, num_input_timestamps=2, num_label_timestamps=4)
# dataloader = DataLoader(dataset=data, batch_size=1)
# for input, label, input_l, label_l in dataloader:
#     print(input.shape)
#     print(label.shape)
#     print(input_l.shape)
#     print(label_l.shape)
#     break