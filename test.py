from scipy.spatial.transform import Rotation
import numpy as np
import torch
import math
import matplotlib.pyplot as plt 


x = torch.arange(1000).reshape(100, 10)
index = torch.ones(10000000, dtype=torch.int32)
y = torch.index_select(x, dim=0, index=index)
print(y.shape)