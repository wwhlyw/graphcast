import graphcast
import graphcast2d
import yaml
import torch
import random
import numpy as np


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
model = 'graphcast2d'

if model == 'graphcast':
    with open('/home/wwh/graphcast/config/params.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    trainer = graphcast.GraphCast(config)
    trainer.train()
elif model == 'graphcast2d':
    with open('/home/wwh/graphcast/config/params1.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    trainer = graphcast2d.GraphCast(config)
    trainer.train()