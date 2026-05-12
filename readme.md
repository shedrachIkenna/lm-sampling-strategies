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