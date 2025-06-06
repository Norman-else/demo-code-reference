# Package Predictor

This directory contains a simple example of training a neural-network-like model
from a small dataset of product names to predict package size and type. The model
is implemented from scratch without external machine learning libraries.

## Files

- `model.py` – minimal implementation of the classifier and persistence
- `train.py` – trains the model using `data.csv`
- `predict.py` – loads the saved model and predicts for a given product name
- `data.csv` – example training data

## Usage

Train the model (this creates `model.json`):

```bash
python train.py
```

Run a prediction:

```bash
python predict.py "cola"
```

The script outputs the predicted package size and type along with their
confidence values.
