"""Synthetic datasets for the replenishment recommendation models."""

from typing import Tuple

import torch

def generate_tabular_data(num_samples: int, feature_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy tabular data for the DNN model.

    Args:
        num_samples: Number of samples.
        feature_dim: Dimension of each sample feature vector.

    Returns:
        features: Tensor of shape (num_samples, feature_dim)
        targets: Tensor of shape (num_samples, 1)
    """
    features = torch.randn(num_samples, feature_dim)
    targets = (features.sum(dim=1, keepdim=True) > 0).float()
    return features, targets


def generate_graph_data(num_nodes: int, feature_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate dummy graph data for the GNN model.

    Creates a simple ring graph represented by an adjacency matrix.

    Args:
        num_nodes: Number of nodes in the graph.
        feature_dim: Dimension of node feature vectors.

    Returns:
        features: Tensor of shape (num_nodes, feature_dim)
        adjacency: Tensor of shape (num_nodes, num_nodes)
    """
    features = torch.randn(num_nodes, feature_dim)
    adjacency = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        adjacency[i, (i + 1) % num_nodes] = 1
        adjacency[(i + 1) % num_nodes, i] = 1
    return features, adjacency
