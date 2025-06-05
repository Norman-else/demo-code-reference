# Replenishment Recommendation Module

This module provides a simple implementation of DNN and GNN based models for
a replenishment recommendation workflow. It demonstrates how to prepare data,
define models, and run a minimal training loop using PyTorch.

## Contents
- `datasets.py` – utilities to generate synthetic datasets for training
- `models.py` – implementations of a feed-forward DNN and a simple GNN
- `train.py` – example training script that trains both models
- `requirements.txt` – python dependencies

## Usage
Install dependencies and run the training script:

```bash
pip install -r requirements.txt
python train.py
```

The script generates dummy data and trains the models for a few epochs, then
prints the training loss.
