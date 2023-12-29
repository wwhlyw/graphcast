from scipy.spatial.transform import Rotation
import numpy as np
import torch
import math
import matplotlib.pyplot as plt 
from scipy.spatial import transform
from torch_scatter import scatter


datas = np.load('./predict.npy')
for data in datas[0]:
    if data < 20000:
        print(data)