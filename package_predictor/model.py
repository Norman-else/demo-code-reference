"""Simple neural network for package size and type prediction."""

import json
import math
import random
from typing import List, Tuple, Dict


def tokenize(name: str) -> List[str]:
    """Split product name into words."""
    return name.lower().split()


class Vocabulary:
    def __init__(self):
        self.token_to_index: Dict[str, int] = {}

    def build(self, names: List[str]):
        for name in names:
            for token in tokenize(name):
                if token not in self.token_to_index:
                    self.token_to_index[token] = len(self.token_to_index)

    def vectorize(self, name: str) -> List[float]:
        vec = [0.0] * len(self.token_to_index)
        for token in tokenize(name):
            idx = self.token_to_index.get(token)
            if idx is not None:
                vec[idx] += 1.0
        return vec


def softmax(logits: List[float]) -> List[float]:
    exps = [math.exp(x) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


class SimpleClassifier:
    def __init__(self, input_dim: int, num_classes: int, lr: float = 0.1):
        self.lr = lr
        self.num_classes = num_classes
        self.weights = [
            [random.uniform(-0.01, 0.01) for _ in range(input_dim)]
            for _ in range(num_classes)
        ]
        self.biases = [0.0 for _ in range(num_classes)]

    def predict_proba(self, vec: List[float]) -> List[float]:
        logits = []
        for c in range(self.num_classes):
            logit = self.biases[c]
            for i, val in enumerate(vec):
                logit += self.weights[c][i] * val
            logits.append(logit)
        return softmax(logits)

    def train(self, X: List[List[float]], y: List[int], epochs: int = 50):
        for _ in range(epochs):
            for vec, label in zip(X, y):
                probs = self.predict_proba(vec)
                for c in range(self.num_classes):
                    error = (1.0 if c == label else 0.0) - probs[c]
                    for i, val in enumerate(vec):
                        self.weights[c][i] += self.lr * error * val
                    self.biases[c] += self.lr * error


class PackagePredictor:
    def __init__(self, vocab: Vocabulary, size_classes: List[str], type_classes: List[str]):
        self.vocab = vocab
        self.size_classes = size_classes
        self.type_classes = type_classes
        self.size_model = SimpleClassifier(len(vocab.token_to_index), len(size_classes))
        self.type_model = SimpleClassifier(len(vocab.token_to_index), len(type_classes))

    def train(self, names: List[str], size_labels: List[str], type_labels: List[str]):
        X = [self.vocab.vectorize(name) for name in names]
        y_size = [self.size_classes.index(lbl) for lbl in size_labels]
        y_type = [self.type_classes.index(lbl) for lbl in type_labels]
        self.size_model.train(X, y_size)
        self.type_model.train(X, y_type)

    def predict(self, name: str) -> Tuple[Tuple[str, float], Tuple[str, float]]:
        vec = self.vocab.vectorize(name)
        size_probs = self.size_model.predict_proba(vec)
        type_probs = self.type_model.predict_proba(vec)
        size_idx = max(range(len(size_probs)), key=lambda i: size_probs[i])
        type_idx = max(range(len(type_probs)), key=lambda i: type_probs[i])
        return (
            (self.size_classes[size_idx], size_probs[size_idx]),
            (self.type_classes[type_idx], type_probs[type_idx]),
        )

    def save(self, path: str):
        data = {
            "vocab": self.vocab.token_to_index,
            "size_classes": self.size_classes,
            "type_classes": self.type_classes,
            "size_weights": self.size_model.weights,
            "size_biases": self.size_model.biases,
            "type_weights": self.type_model.weights,
            "type_biases": self.type_model.biases,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "PackagePredictor":
        with open(path, "r") as f:
            data = json.load(f)
        vocab = Vocabulary()
        vocab.token_to_index = data["vocab"]
        predictor = cls(vocab, data["size_classes"], data["type_classes"])
        predictor.size_model.weights = data["size_weights"]
        predictor.size_model.biases = data["size_biases"]
        predictor.type_model.weights = data["type_weights"]
        predictor.type_model.biases = data["type_biases"]
        return predictor
