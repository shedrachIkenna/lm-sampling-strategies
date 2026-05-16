# Sampling Strategy Comparison for GPT-Style Language Model Training 

> **Research Question:** Does the method used to sample training batches affect the speed of convergence and final loss of a character-level transformer language model trained for a fix number of iterations? 

--- 

## Overview 

This project empirically tests four batch sampling strategies under identical controlled conditions for a character-level transformer trained on the [tinyshakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset. 

**Short answer:** No. Sampling strategy has no measurable effect on convergence speed or final loss in this setting. The dominant variables were corpus size, model capacity, and regularization strength - each producing changes an order of magnitude larger than any difference attributable to sampling strategy. 

**Full paper:** [`sampling_strategy_research_paper.pdf`](./sampling_strategy_research_paper.pdf)

--- 

## The Four Strategies 

| # | Strategy | Description | Coverage guarantee |
|---|---|---|---|
| 1 | **Random (baseline)** | `torch.randint` over valid positions. Same index can repeat across batches. | None |
| 2 | **Shuffle without replacement** | All valid positions shuffled via `torch.randperm`, yielded in order. Reshuffled each epoch. | Every valid position seen once per epoch |
| 3 | **Circular buffer** | Indices sampled from full corpus range. Sequences wrap around end-to-start via modular arithmetic. | None (but no wasted tail positions) |
| 4 | **Circular + Shuffle** | Full index range via wrapping + shuffle without replacement. | Every valid position seen once per epoch |

--- 

## Key Results 

All four strategies converged identically across 15,000 training steps 

```
═══════════════════════════════════════════════════════════════
  Technique             Final Train   Min Val    Final Val
───────────────────────────────────────────────────────────────
  Random (baseline)         1.345      1.557       1.557
  Shuffle                   1.360      1.542       1.584
  Circular                  1.343      1.556       1.556
  Circular + Shuffle        1.355      1.533       1.568
═══════════════════════════════════════════════════════════════
  Strategy spread in min val loss: 0.024  (within measurement noise)
```

![Final run results](./figures/sampling_comparison_run4.png)

### What actually moved the needle
 
| Variable | Change | Effect on val loss |
|---|---|---|
| Corpus size | 10k → 1M tokens | Eliminated catastrophic overfitting |
| Model capacity | 103k → 800k params | Floor dropped from 1.77 → 1.53 |
| Dropout | 0.1 → 0.3 | Closed train/val gap from 3.0 → 0.21 |
| **Sampling strategy** | **Any of four** | **~0.02–0.05 — within noise** |

---

## Why the Theory Didn't Show Up in Practice 
Rajput, Gupta and Papailiopoulos (ICML 2020) prove that SGD without replacement converges at O(n/T²) vs O(1/T) for with-replacement — but only when **T grows faster than n**.

In this experiment: 
- **n** ≈ 1,003,790 valid starting positions
- **T** = 15,000 steps
- T is ~66× **smaller** than n — the theoretical condition was never satisfied


At less than half an epoch, even random sampling rarely repeats positions. The coverage advantage of shuffle-based strategies is real in theory, negligible here in practice.

**This isn't just a limitation of this experiment - its structural** 
Modern LLMs are trained on trillions of tokens. Even with massive compute budget, T never approaches n. The theoretical regime where without-replacement sampling provably wins is the opposite of how languague models are trained at meaningful scale. The theory is mathematically sound, but the conditions it requires simply doesn't exist in practice.

## Model Architecture 

Decoder-only transformer with Rotary Positional Embedding (RoPE)

```
TinyTransformerLM(
  token_emb     :Embedding(65, 128)
  rotary_emb    :RotaryEmbedding(d_head=32, max_seq=64)
  layers x4     : TransformerBlock(
      ln1       : LayerNorm(128)
      attn      : MultiHeadSelfAttention(d_model=128, n_heads=4)
      ln2       : LayerNorm(128)
      ffn       : FeedForward(128 -> 512 -> 128, GELU, dropout=0.3)
  )
  ln_f          : LayerNorm(128)
  head          : Linear(128 -> 65)  [weight-tied to token_emb]
)
Total trainable parameters: 800,512
```

**Why RoPE?** Unlike sinusoidal positional encoding, RoPE encodes position by rotating query and key vectors inside attention, not by adding a signal to the token embedding. This encodes relative position rather than absolute position and generalizes more naturally to longer sequences. 

---