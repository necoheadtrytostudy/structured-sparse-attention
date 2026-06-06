# Structured Sparse Attention Prototype

A PyTorch prototype developed for a CS 657 course project exploring structured sparse attention for Transformer inference.

## Overview

This project compares three attention implementations:

1. Standard dense attention
2. Loop-based local sparse attention
3. Mask-based local sparse attention

The program benchmarks their execution time across multiple sequence lengths and compares sparse outputs with dense attention.

## Technologies

- Python
- PyTorch
- Tensor operations
- Performance benchmarking

## Features

- Dense attention implementation
- Local sliding-window sparse attention
- Loop-based and mask-based approaches
- Runtime comparison across multiple input sizes
- Output difference measurement
- Reproducible random inputs through fixed seeds

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

The default experiment tests sequence lengths of 128, 256, and 512 with an embedding dimension of 64.

## What I Learned

- How attention scores and weighted values are calculated
- How local attention restricts token interactions
- The performance difference between Python loops and vectorized tensor operations
- How implementation structure can affect runtime
- How to benchmark and compare experimental implementations

## Limitations

This is an educational prototype rather than a production-optimized sparse attention kernel. The mask-based implementation still computes the full attention score matrix before applying the local mask.

## Background

Completed as coursework for a UWM computer science course. The repository is shared for learning and portfolio purposes.
