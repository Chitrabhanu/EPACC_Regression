# Contains some training utilities

import torch
from itertools import product

def weights_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)

def get_adaptive_learning_rate(epoch, initial_rate=0.01, step_size=50):
    return initial_rate / (10 ** (epoch // step_size))

def grid_generator(param_dict):
    keys, values = zip(*param_dict.items())
    return [dict(zip(keys, v)) for v in product(*values)]