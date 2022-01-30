import torch
import torch.nn


class LogisticRegression(torch.nn.Module):
    _linear: torch.nn.Module
    _activation: torch.nn.Sigmoid

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._linear = torch.nn.Linear(input_dim, output_dim)
        self._activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self._activation(self._linear(x))
        return x
