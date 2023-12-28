from scipy.spatial.transform import Rotation
import numpy as np
import torch
import math
import matplotlib.pyplot as plt 
from scipy.spatial import transform
from torch_scatter import scatter


x = torch.tensor([5, 1, 7, 2, 3, 2, 1, 3])
index = torch.tensor([1, 1, 1, 1, 2, 2, 3, 3])
x = scatter(x, index)
print(x)
