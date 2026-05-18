# Structured Sparse Attention Prototype

This repository contains the code for the CS 790/657 final project on compiler-visible structured sparse attention for Transformer inference.

## Files

- `main.py`: PyTorch prototype comparing dense attention, loop-based sparse attention, and mask-based sparse attention.
- `requirements.txt`: Python dependency list.
- `.gitignore`: Excludes virtual environments, IDE files, cache files, and PDFs.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
