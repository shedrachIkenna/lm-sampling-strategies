# Sampling Strategy Comparison for GPT-Style Language Model Training 

> **Research Question:** Does the method used to sample training batches affect the speed of convergence and final loss of a character-level transformer language model trained for a fix number of iterations? 

--- 

## Overview 

This project empirically tests four batch sampling strategies under identical controlled conditions for a character-level transformer trained on the [tinyshakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset. 