from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def build_sequence_tensor(values: np.ndarray, *, lookback: int = 5) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("sequence values must be a 2D feature matrix")
    if lookback < 1:
        raise ValueError("lookback must be positive")
    rows: list[np.ndarray] = []
    for end in range(len(values)):
        start = max(0, end - lookback + 1)
        window = values[start : end + 1]
        if len(window) < lookback:
            pad = np.repeat(window[:1], lookback - len(window), axis=0)
            window = np.vstack([pad, window])
        rows.append(window)
    return np.stack(rows).astype("float32")


@dataclass
class LaggedRidgeRegressor:
    alpha: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LaggedRidgeRegressor":
        if x.ndim != 3:
            raise ValueError("LaggedRidgeRegressor expects a 3D sequence tensor")
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(x.reshape(len(x), -1), y.astype(float))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("LaggedRidgeRegressor expects a 3D sequence tensor")
        return self.model_.predict(x.reshape(len(x), -1))


class _TorchRecurrentNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cell_type: str):
        super().__init__()
        if cell_type == "lstm":
            self.recurrent = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif cell_type == "gru":
            self.recurrent = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        else:
            raise ValueError(f"unsupported recurrent cell: {cell_type}")
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.recurrent(x)
        return self.head(output[:, -1, :]).squeeze(-1)


@dataclass
class TorchSequenceRegressor:
    cell_type: str
    hidden_size: int = 8
    epochs: int = 24
    learning_rate: float = 0.01
    seed: int = 7

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchSequenceRegressor":
        if x.ndim != 3:
            raise ValueError("TorchSequenceRegressor expects a 3D sequence tensor")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.feature_mean_ = x.reshape(-1, x.shape[-1]).mean(axis=0)
        self.feature_std_ = x.reshape(-1, x.shape[-1]).std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ <= 1e-8, 1.0, self.feature_std_)
        self.target_mean_ = float(np.mean(y))
        self.target_std_ = float(np.std(y))
        if self.target_std_ <= 1e-8:
            self.target_std_ = 1.0

        x_scaled = (x - self.feature_mean_) / self.feature_std_
        y_scaled = (y.astype("float32") - self.target_mean_) / self.target_std_
        x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32)
        self.model_ = _TorchRecurrentNet(x.shape[-1], self.hidden_size, self.cell_type)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        self.model_.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("TorchSequenceRegressor expects a 3D sequence tensor")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            output = self.model_(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_
