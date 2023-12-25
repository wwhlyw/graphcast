from scipy.spatial.transform import Rotation
import numpy as np
import torch
import math
import matplotlib.pyplot as plt 


lon = np.load('./location/lats.npy')[253:693,970:1378]
print(lon)