# Sampling Strategy Comparison for GPT-Style Language Model Training 

> **Research Question:** Does the method used to sample training batches affect the speed of convergence and final loss of a character-level transformer language model trained for a fix number of iterations? 

--- 

## Overview 

This project empirically tests four batch sampling strategies under identical controlled conditions for a character-level transformer trained on the [tinyshakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset. 

**Short answer:** No. Sampling strategy has no measurable effect on convergence speed or final loss in this setting. The dominant variables were corpus size, model capacity, and regularization strength - each producing changes an order of magnitude larger than any difference attributable to sampling strategy. 

**Full paper:** [`sampling_strategy_research_paper.pdf`](./sampling_strategy_research_paper.pdf)