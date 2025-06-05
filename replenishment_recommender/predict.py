"""Example prediction script using trained models."""

import torch

from datasets import generate_tabular_data, generate_graph_data
from models import DNNRecommender, SimpleGNNRecommender


def predict_dnn(model_path: str):
    """Load a trained DNN model and run predictions on synthetic data."""
    features, _ = generate_tabular_data(num_samples=5, feature_dim=8)
    model = DNNRecommender(input_dim=8, hidden_dims=[16, 8])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        outputs = torch.sigmoid(model(features))
    print("DNN predictions:", outputs.squeeze().tolist())


def predict_gnn(model_path: str):
    """Load a trained GNN model and run predictions on synthetic data."""
    features, adj = generate_graph_data(num_nodes=5, feature_dim=8)
    model = SimpleGNNRecommender(input_dim=8, hidden_dim=16)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        outputs = model(features, adj)
    print("GNN predictions:", outputs.squeeze().tolist())


if __name__ == "__main__":
    predict_dnn("dnn_model.pth")
    predict_gnn("gnn_model.pth")
