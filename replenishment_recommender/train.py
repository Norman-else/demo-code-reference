"""Example training script for the replenishment recommendation models."""

import torch
from torch import nn, optim

from datasets import generate_tabular_data, generate_graph_data
from models import DNNRecommender, SimpleGNNRecommender


def train_dnn():
    features, targets = generate_tabular_data(num_samples=256, feature_dim=8)
    model = DNNRecommender(input_dim=8, hidden_dims=[16, 8])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"DNN Epoch {epoch+1}, loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "dnn_model.pth")


def train_gnn():
    features, adj = generate_graph_data(num_nodes=20, feature_dim=8)
    model = SimpleGNNRecommender(input_dim=8, hidden_dim=16)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    target = torch.zeros(features.size(0), 1)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(features, adj)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        print(f"GNN Epoch {epoch+1}, loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "gnn_model.pth")


if __name__ == "__main__":
    train_dnn()
    train_gnn()
