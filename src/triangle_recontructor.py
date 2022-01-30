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
        points, labels = self._down_sample_data(points, labels)
        x_train, x_test, y_train, y_test = train_test_split(points, labels, test_size=0.2, random_state=42)

        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_train = torch.unsqueeze(torch.tensor(y_train, dtype=torch.float32), dim=-1)
        y_test = torch.unsqueeze(torch.tensor(y_test, dtype=torch.float32), dim=-1)

        self._train(x_train, y_train)
        self._evaluate(x_test, y_test)

    @staticmethod
    def _down_sample_data(points: np.array, labels: np.array):
        pos_labels = labels.sum()
        neg_labels = len(labels) - pos_labels
        label_size = min(pos_labels, neg_labels)

        selected_points = []
        selected_labels = []
        pos_count, neg_count = 0, 0
        for i in range(len(points)):
            point = points[i]
            label = labels[i]

            if label and pos_count >= label_size:
                continue
            if not label and neg_count >= label_size:
                continue

            pos_count += 1 if label else 0
            neg_count += 1 if not label else 0

            selected_points.append(point)
            selected_labels.append(label)

        print(f'Positives: {pos_count}, negatives: {neg_count}, selected: {len(selected_points)}')
        return np.array(selected_points), np.array(selected_labels)

    def _train(self, x: np.array, y: np.array) -> None:
        self._model.train()
        for epoch in tqdm_notebook(range(self._epochs), desc='Training Epochs'):
            self._optimizer.zero_grad()

            y_pred = self._model(x)
            loss = self._loss_function(y_pred, y)
            loss.backward()

            self._optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch of training: {epoch + 1}, loss: {loss.item()}')

    def _evaluate(self, x: np.array, y: np.array) -> None:
        self._model.eval()
        y_pred = self._model(x)
        loss = self._loss_function(y_pred, y)
        print('Testing loss', loss.item())

    def compute_labels(self, points: np.array) -> np.array:
        x_test = torch.tensor(points, dtype=torch.float32)

        self._model.eval()
        y_test = self._model(x_test)
        labels = y_test.round().detach().numpy()
        return labels
