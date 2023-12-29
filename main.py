import graphcast
import yaml
import torch
import random
import numpy as np


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

with open('/home/wwh/graphcast/config/params.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)
trainer = graphcast.GraphCast(config)
trainer.train()
