"""Model implementations for the replenishment recommendation workflow."""

from typing import Iterable

import torch
from torch import nn


class DNNRecommender(nn.Module):
    """Simple feed-forward network for tabular data."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], output_dim: int = 1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SimpleGNNRecommender(nn.Module):
    """Very small GNN that aggregates neighbor information using adjacency matrices."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Aggregate neighbor features
        h = self.fc1(x)
        h = torch.relu(h)
        agg = torch.matmul(adj, h)
        out = self.fc2(agg)
        return out
