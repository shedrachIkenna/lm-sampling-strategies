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


# Data Loading 
data_path = Path("/content/tiny.txt")
assert data_path.exists(), (
    "tiny.txt not found. Upload your text file to /content/tiny.txt"
)

text = data_path.read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)} # dictionary of characters to integer IDs 
itos = {i: ch for ch, i in stoi.items()} # dictionary of interger IDs to character tokens 

def encode(s: str) -> list:
    return [stoi[c] for c in s]

def decode(lst: list) -> str: 
    return "".join(itos[i] for i in lst)


data = torch.tensor(encode(text), dtype=torch.long) # convert token to tensor
n = len(data)
train_data = data[:int(0.9 * n)]
val_data = data[int(0.9 * n):]

print(f"Corpus: {n:,} tokens")
print(f"Vocab: {vocab_size} unique characters")
print(f"Train: {len(train_data):,} tokens")
print(f"Val: {len(val_data):,} tokens")
print(f"Valid starting positions: (train): {len(train_data) - block_size}")
print("=" * 60)


# Sampling Strategies 

# Baseline - Random sampling with replacement 
def get_batch_random(split: str): 
    """
    This is the baseline sampling approach: Samples batch_size starting indices uniformly from [0, len(src) - block_size - 1]
    Same index can appear multiple times in the same batch or across different steps. No coverage guarantee
    """
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i: i + block_size] for i in ix])
    y = torch.stack([src[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# Shuffle without replacement 
class ShuffleSampler: 
    """
    Epoch-based sampler. Precomputes all valid starting positions 
    shuffles them, then yields batches in order 
    When all positions are exhausted, reshuffles and starts a new epoch 
    Every position is seen exactly once per epoch - no repetition
    """

    def __init__(self, src: torch.Tensor) -> None: 
        self.src = src 
        self.n_valid = len(src) - block_size
        self._reset()

    def _reset(self) -> None:
        perm = torch.randperm(self.n_valid)
        self.queue = perm.tolist()
    
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]: 
        if len(self.queue) < batch_size:
            self._reset() # start new epoch 
        ix = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        ix = torch.tensor(ix, dtype=torch.long)
        x = torch.stack([self.src[i : i + block_size] for i in ix])
        y = torch.stack([self.src[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
    

# Circular sampling 
def get_batch_circular(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Treats the data as a ring. Every index in [0, len(src) - 1] is a valid starting position. 
    Indices near the end wrap back to the beginning using modular arithmetic. 

    Trade-off: occasionally produces a context window that stitches the end of the corpus to the beginning
    """
    src = train_data if split == "train" else val_data
    N = len(src)
    ix = torch.randint(N, (batch_size,))
    x = torch.stack([src[torch.arange(i, i + block_size) % N] for i in ix])
    y = torch.stack([src[torch.arange(i + 1, i + block_size + 1) % N] for i in ix])
    return x.to(device), y.to(device)


# Circular + Shuffle Sampling 
class CircularShuffleSampler:
    """
    Combines the logic of shuffle without replacement and circular sampling \
    
    - Full index range [0, len(src) - 1] are valid starting positions via circular wrapping 
    - Shuffle without replacement ensures every position is visited exactly once per epoch before any index is revisited 
    """

    def __init__(self, src: torch.Tensor) -> None: 
        self.src = src 
        self.N = len(src)
        self._reset()

    def _reset(self) -> None: 
        perm = torch.randperm(self.N)
        self.queue = perm.tolist()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.queue) < batch_size:
            self._reset()
        ix = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        ix = torch.tensor(ix, dtype=torch.long)
        N = self.N
        x = torch.stack([self.src[torch.arange(i, i + block_size) % N] for i in ix])
        y = torch.stack([self.src[torch.arange(i + 1, i + block_size + 1) % N] for i in ix])
        return x.to(device), y.to(device)
    

# Rotary Positional Embedding 
"""
Implementation Note: 
    -- The cos/sin tables are precomputed once up to max_seq_len and cached as non-learnable buffers 
    -- apply_rotary_emb slices the right prefix at runtime, so sequences shorter than max_seq_len are handled with no waste 
"""

class RotaryEmbedding(nn.Module): 
    """
    Precomputes and caches the cos/sin rotation tables for RoPE

    Args: 
        d_head: dimension of a single attention head (must be even)
        max_seq: maximum sequence length to precompute 
        base: frequency base (default = 10000 as used in the original paper)
    """

    def __init__(self, d_head: int, max_seq: int = 2048, base: int = 10_000) -> None: 
        super().__init__()
        assert d_head % 2 == 0, "Rope requires even head dimension"
        # 0_i = 1 / base^(2i / d_head) for i = 0, 1, ..., d_head/2 - 1
        # shape: (d_head/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)


        # Precompute tables up to max_seq_len -> (max_seq, d_head/2)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int) -> None: 
        pos = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device) # (T,)
        freqs = torch.outer(pos, self.inv_freq) # (T, d_head/2)
        # Concatenate so that cos/sin cover all d_head dimensions 
        emb = torch.cat([freqs, freqs], dim=-1) # (T, d_head)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]: 
        """Returns the cos and sin tables sliced to the actual sequence length"""
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len)
        # Shapes returned: (1, 1, T, d_head) ready to be broadcasted over (B, H, T, d_head)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin
    
    