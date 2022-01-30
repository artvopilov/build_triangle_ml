import numpy as np
import torch
import torch.nn


class TriangleReconstructorModel(torch.nn.Module):
    _layer1: torch.nn.Linear
    _activation1: torch.nn.Sigmoid
    _layer2: torch.nn.Linear
    _activation2: torch.nn.Sigmoid

    def __init__(self):
        super().__init__()
        self._layer1 = torch.nn.Linear(2, 3)
        self._activation1 = torch.nn.Sigmoid()
        self._layer2 = torch.nn.Linear(3, 1)
        self._activation2 = torch.nn.Sigmoid()

    def forward(self, x: np.array) -> np.array:
        x = self._activation1(self._layer1(x))
        x = self._activation2(self._layer2(x))
        return x



