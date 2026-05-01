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
    

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation details: 
        - split the dimensions into two equal contiguous arrays. 
        - Pair the dimensions element-wise and apply the rotation matrix logic 
    """
    half = x.shape[-1] // 2 
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies RoPE to query and key tensors

    Returns 
        q_rot, k_rot: same shape as q and k respectively 
    """
    q_rot = q * cos + _rotate_half(q) * sin 
    k_rot = k * cos + _rotate_half(k) * sin 
    return q_rot, k_rot

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None: 
        super().__init__()
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=True)
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rotary_emb: RotaryEmbedding | None = None) -> None: 
        super.__init__()
        assert d_model % n_heads == 0 
        self.n_heads = n_heads 
        self.d_heads = d_model // n_heads 
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = rotary_emb # shared across all layers 

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor: 
        B, T, D = x.shape 
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.d_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Change the shape from [B, T, n_heads, d_head] -> [B, n_heads, T, d_head]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # apply RoPE to q and k (v is left unrotated)
        if self.rotary_emb is not None: 
            cos, sin = self.rotary_emb(T)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Scaled dot product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply mask 
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, float=("-inf"))
        
        probs = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(probs, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out(out)
    

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None: 
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), # Expansion: Projects the token into higher dimension 
            nn.GELU(), # Activation: Where non-linear and complex relations are learned 
            nn.Linear(d_ff, d_model), # Contraction: Token's dimension are projected back down to its original dimension 
            nn.Dropout(dropout), # Regularization: Prevents the model from relying on some particular neuron to prevent overfitting
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float=0.1, rotary_emb: RotaryEmbedding | None = None) -> None: 
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, rotary_emb)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor: 
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x         

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, block_size: int, dropout: float = 0.1) -> None: 
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # RoPE uses one shared instance whose cos/sin tables are reused by every layer. No extra parameters and weight-tying across layers is free
        d_head = d_model // n_heads 
        self.rotary_emb = RotaryEmbedding(d_head, max_seq=block_size)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, self.rotary_emb)
            for _ in range(n_layers)
        ])
        self.ln_f = LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.block_size = block_size

        # weight tying 
        self.head.weight = self.token_emb.weight

        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))

        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None: 
        """
        Initializes the weights of the model with default values that will make the model stable and more likely to learn 
        effectively from the very first second of training 
        Set 
            - standard deviation = 0.02 
            - bias = 0 
            - mean = 0.0 
        """
        if isinstance(module, nn.Linear): 
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): 
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert T <= self.block_size
        x = self.token_emb(idx)
        # RoPE is applied inside of each layer 
        mask = self.causal_mask[:, :, :T, :T]
        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.head(self.ln_f(x))
        loss = None 
        if targets is not None: 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss 
    

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor: 
        for _ in range(max_new_tokens): 
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
    
# Evaluation Metrics 
@torch.no_grad()
def evaluate(model: TinyTransformerLM, n_batches: int = eval_batches) -> dict: 
    """
    Compute mean train and val loss over n_batches batches 

    Using multiple batches gives a much more stable loss estimate than a single batch
    """
    model.eval()
    out = {}
    for split in ("train", "loss"):
        losses = []
        for _ in range(n_batches): 
            xb, yb = get_batch_random(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out

# Training Loop 
def train_run(technique_name: str, get_train_batch_fn) -> dict: 
    """
    Run a full training loop for one sampling technique 

    Parameters 
        - technique name: sampling technique labels used in plots and logs 
        - get_train_batch_fn: callable() => (x,y) tensors for training 

    Returns dict with keys 
        - iter_history, train_history, val_history 
        - final_train_loss, final_val_loss, 
        - convergence_step (step where val loss <= CONVERGENCE_THRESHOLD or None if threshold is never reached)
    """

    print(f"\n{'-' * 60}")
    print(f" Run: {technique_name}")
    print(f"{'=' * 60}")

    # Reload identical initial weights for a fair comparision 
    set_seed(SEED)
    model = TinyTransformerLM(vocab_size, n_embd, n_layer, n_head, n_embd * 4, block_size, dropout).to(device)
    model.load_state_dict(torch.load("init_weights.pth", map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    iter_history = [] 
    train_history = [] 
    val_history = [] 
    convergence_step = None

    for step in range(max_iters): 
        # training step 
        xb, yb = get_train_batch_fn()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Checkpoint 
        if step % eval_interval == 0 or step == max_iters - 1: 
            metrics = evaluate(model)
            iter_history.append(metrics["train"])
            val_history.append(metrics["val"])

            if convergence_step is None and metrics["val"] <= CONVERGENCE_THRESHOLD: 
                convergence_step = step

            converged_tag = " <- converged" if convergence_step == step else print(
                f"Step {step:>4d} | "
                f"Train {metrics['train']:.4f} | "
                f"Val {metrics['val']:.4f}"
                f"{converged_tag}"
            )
    
    return {
        "iter_history": iter_history, 
        "train_history": train_history,
        "val_history": val_history, 
        "final_train_loss": train_history[-1], 
        "final_val_loss": val_history[-1], 
        "convergence_step": convergence_step
    }

# Initialize model and save starting weights 
print("\nInitializing model and saving starting weights...")
set_seed(SEED)
_init_model = TinyTransformerLM(vocab_size, n_embd, n_layer, n_head, n_embd * 4, block_size, dropout).to(device)
torch.save(_init_model.state_dict(), "init_weight.pth")
n_params = sum(p.numel() for p in _init_model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")
print("init_weight.pth saved - all four runs will start from this checkpoint")
del _init_model

