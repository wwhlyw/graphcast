from scipy.spatial.transform import Rotation
import numpy as np
import torch
import math
import matplotlib.pyplot as plt 
from scipy.spatial import transform
from torch_scatter import scatter


x = torch.arange(48).reshape(2, 3, 8)
y = torch.arange(2, 34).reshape(2, 2, 8)
print(y)
indices = torch.tensor([0, 2])
x[:, indices] = y
print(x)