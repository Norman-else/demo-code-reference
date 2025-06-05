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
Install dependencies and run the training and prediction scripts:

```bash
pip install -r requirements.txt
python train.py       # trains and saves DNN and GNN models
python predict.py     # loads the saved models and prints predictions
```

`train.py` generates dummy data, trains the models for a few epochs and saves
the weights to `dnn_model.pth` and `gnn_model.pth`. `predict.py` then loads
these weights and performs inference on fresh synthetic data.
