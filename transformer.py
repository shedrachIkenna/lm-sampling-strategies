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