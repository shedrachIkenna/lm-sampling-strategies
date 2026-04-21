import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import csv 
from pathlib import Path 
from copy import deepcopy


# Reproducibility 
SEED = 42 

def set_seed(seed: int): 
    """Fix all random sources for full reproducibilty"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 64 
batch_size = 32 
dropout = 0.1 
n_embd = 128 
n_head = 4 
n_layer = 4 
lr = 3e-4 
max_iters = 2000 
eval_interval = 100 # record loss every N steps 
eval_batches = 20 # average val loss over 20 batches 

CONVERGENCE_THRESHOLD = 1.5 
print(f"Device: {device}")
print(f"Seed: {SEED}")
print(f"iters: {max_iters} | Eval every {eval_interval} steps")
print(f"Val loss averaged over {eval_batches} batches per checkpoint")
print("="*60)