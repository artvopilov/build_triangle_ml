import numpy as np
import torch
import torch.nn
import torch.optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

from src.triangle_reconstructor_model import TriangleReconstructorModel


class TriangleReconstructor:
    _model: torch.nn.Module
    _loss_function: torch.nn.Module
    _optimizer: torch.optim.Optimizer
    _epochs: int

    def __init__(self, learning_rate: float, epochs: int):
        self._model = TriangleReconstructorModel()
        self._loss_function = torch.nn.BCELoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate)
        self._epochs = epochs

    def reconstruct(self, points: np.array, labels: np.array) -> None:
        x_train, x_test, y_train, y_test = train_test_split(points, labels, test_size=0.2, random_state=42)

        x_train = torch.Tensor(x_train)
        x_test = torch.Tensor(x_test)
        y_train = torch.unsqueeze(torch.Tensor(y_train), dim=-1)
        y_test = torch.unsqueeze(torch.Tensor(y_test), dim=-1)

        self._train(x_train, y_train)
        self._evaluate(x_test, y_test)
        self._plot(points, labels)

    def _train(self, x: np.array, y: np.array) -> None:
        self._model.train()
        for epoch in tqdm_notebook(range(self._epochs), desc='Training Epochs'):
            self._optimizer.zero_grad()

            y_pred = self._model(x)
            loss = self._loss_function(y_pred, y)
            loss.backward()

            self._optimizer.step()

            print(f'Epoch of training: {epoch + 1}, loss: {loss.item()}')

    def _evaluate(self, x: np.array, y: np.array) -> None:
        self._model.eval()
        y_pred = self._model(x)
        loss = self._loss_function(y_pred, y)
        print('Testing loss', loss.item())

    def _plot(self, points: np.array, labels: np.array):
        # plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='jet')
        # plt.show()
        pass
