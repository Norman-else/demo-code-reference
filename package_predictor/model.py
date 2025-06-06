"""Simple neural network for package size and type prediction."""

import json
import math
import random
from typing import List, Tuple, Dict


def tokenize(text: str) -> List[str]:
    """Split a piece of text into lowercase tokens."""
    return text.lower().split()


class Vocabulary:
    def __init__(self):
        self.token_to_index: Dict[str, int] = {}

    def build(self, texts: List[str]):
        """Build the vocabulary from a list of texts."""
        for text in texts:
            for token in tokenize(text):
                if token not in self.token_to_index:
                    self.token_to_index[token] = len(self.token_to_index)

    def vectorize(self, name: str, description: str) -> List[float]:
        """Convert a name/description pair into a vector."""
        text = f"{name} {description}"
        vec = [0.0] * len(self.token_to_index)
        for token in tokenize(text):
            idx = self.token_to_index.get(token)
            if idx is not None:
                vec[idx] += 1.0
        return vec


def softmax(logits: List[float]) -> List[float]:
    exps = [math.exp(x) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


class SimpleNN:
    """Minimal two-layer neural network with tanh activation."""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, lr: float = 0.1
    ):
        self.lr = lr
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.hidden_weights = [
            [random.uniform(-0.01, 0.01) for _ in range(input_dim)]
            for _ in range(hidden_dim)
        ]
        self.hidden_biases = [0.0 for _ in range(hidden_dim)]
        self.out_weights = [
            [random.uniform(-0.01, 0.01) for _ in range(hidden_dim)]
            for _ in range(num_classes)
        ]
        self.out_biases = [0.0 for _ in range(num_classes)]

    def _forward(self, vec: List[float]) -> Tuple[List[float], List[float]]:
        hidden = []
        for j in range(self.hidden_dim):
            h = self.hidden_biases[j]
            for i, val in enumerate(vec):
                h += self.hidden_weights[j][i] * val
            hidden.append(math.tanh(h))

        logits = []
        for c in range(self.num_classes):
            logit = self.out_biases[c]
            for j, h in enumerate(hidden):
                logit += self.out_weights[c][j] * h
            logits.append(logit)
        return hidden, softmax(logits)

    def predict_proba(self, vec: List[float]) -> List[float]:
        _, probs = self._forward(vec)
        return probs

    def train(self, X: List[List[float]], y: List[int], epochs: int = 50):
        for _ in range(epochs):
            for vec, label in zip(X, y):
                hidden, probs = self._forward(vec)
                # Gradient for output layer
                delta_out = [probs[c] - (1.0 if c == label else 0.0) for c in range(self.num_classes)]
                # Update output weights and biases
                for c in range(self.num_classes):
                    for j in range(self.hidden_dim):
                        self.out_weights[c][j] -= self.lr * delta_out[c] * hidden[j]
                    self.out_biases[c] -= self.lr * delta_out[c]
                # Gradient for hidden layer
                delta_hidden = [0.0 for _ in range(self.hidden_dim)]
                for j in range(self.hidden_dim):
                    dh = 0.0
                    for c in range(self.num_classes):
                        dh += self.out_weights[c][j] * delta_out[c]
                    dh *= 1.0 - hidden[j] ** 2  # derivative of tanh
                    delta_hidden[j] = dh
                    for i in range(len(vec)):
                        self.hidden_weights[j][i] -= self.lr * dh * vec[i]
                    self.hidden_biases[j] -= self.lr * dh


class PackagePredictor:
    def __init__(self, vocab: Vocabulary, size_classes: List[str], type_classes: List[str]):
        self.vocab = vocab
        self.size_classes = size_classes
        self.type_classes = type_classes
        input_dim = len(vocab.token_to_index)
        hidden_dim = max(2, input_dim // 2)
        self.size_model = SimpleNN(input_dim, hidden_dim, len(size_classes))
        self.type_model = SimpleNN(input_dim, hidden_dim, len(type_classes))

    def train(
        self,
        names: List[str],
        descriptions: List[str],
        size_labels: List[str],
        type_labels: List[str],
    ):
        X = [self.vocab.vectorize(n, d) for n, d in zip(names, descriptions)]
        y_size = [self.size_classes.index(lbl) for lbl in size_labels]
        y_type = [self.type_classes.index(lbl) for lbl in type_labels]
        self.size_model.train(X, y_size)
        self.type_model.train(X, y_type)

    def predict(self, name: str, description: str) -> Tuple[Tuple[str, float], Tuple[str, float]]:
        vec = self.vocab.vectorize(name, description)
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
            "size_hidden_weights": self.size_model.hidden_weights,
            "size_hidden_biases": self.size_model.hidden_biases,
            "size_out_weights": self.size_model.out_weights,
            "size_out_biases": self.size_model.out_biases,
            "type_hidden_weights": self.type_model.hidden_weights,
            "type_hidden_biases": self.type_model.hidden_biases,
            "type_out_weights": self.type_model.out_weights,
            "type_out_biases": self.type_model.out_biases,
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
        predictor.size_model.hidden_weights = data["size_hidden_weights"]
        predictor.size_model.hidden_biases = data["size_hidden_biases"]
        predictor.size_model.out_weights = data["size_out_weights"]
        predictor.size_model.out_biases = data["size_out_biases"]
        predictor.type_model.hidden_weights = data["type_hidden_weights"]
        predictor.type_model.hidden_biases = data["type_hidden_biases"]
        predictor.type_model.out_weights = data["type_out_weights"]
        predictor.type_model.out_biases = data["type_out_biases"]
        return predictor
