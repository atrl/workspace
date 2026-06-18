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


def _frequency_features(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("frequency features expect a 3D sequence tensor")
    last = x[:, -1, :]
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    spectrum = np.abs(np.fft.rfft(x, axis=1))[:, 1:, :]
    if spectrum.size == 0:
        spectrum = np.zeros((len(x), 1, x.shape[-1]))
    return np.concatenate([last, mean, std, spectrum.reshape(len(x), -1)], axis=1)


def _trend_residual_features(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("decomposition features expect a 3D sequence tensor")
    trend = x.mean(axis=1)
    seasonal = x[:, -1, :] - trend
    slope = x[:, -1, :] - x[:, 0, :]
    return np.concatenate([trend, seasonal, slope], axis=1)


def _patch_features(x: np.ndarray, *, patch_size: int = 2) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("patch features expect a 3D sequence tensor")
    patches: list[np.ndarray] = []
    for start in range(0, x.shape[1], patch_size):
        patch = x[:, start : start + patch_size, :]
        patches.append(np.concatenate([patch.mean(axis=1), patch.std(axis=1)], axis=1))
    return np.concatenate(patches, axis=1)


def _event_decay_features(x: np.ndarray, *, decay: float = 0.7) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("event decay features expect a 3D sequence tensor")
    steps = np.arange(x.shape[1] - 1, -1, -1, dtype="float32")
    weights = np.exp(-decay * steps)
    weighted = (x * weights.reshape(1, -1, 1)).sum(axis=1) / weights.sum()
    event_intensity = np.abs(np.diff(x, axis=1)).sum(axis=1)
    return np.concatenate([x[:, -1, :], weighted, event_intensity], axis=1)


def _mixer_features(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("mixer features expect a 3D sequence tensor")
    token_mixed = x.mean(axis=2)
    channel_mixed = x.mean(axis=1)
    interactions = np.einsum("bt,bf->btf", token_mixed, channel_mixed).reshape(len(x), -1)
    return np.concatenate([x.reshape(len(x), -1), token_mixed, channel_mixed, interactions], axis=1)


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


@dataclass
class FrequencyRidgeRegressor:
    alpha: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FrequencyRidgeRegressor":
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(_frequency_features(x), y.astype(float))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict(_frequency_features(x))


@dataclass
class FeatureRidgeRegressor:
    feature_kind: str
    alpha: float = 1.0

    def _features(self, x: np.ndarray) -> np.ndarray:
        if self.feature_kind == "event_decay":
            return _event_decay_features(x)
        if self.feature_kind == "mixer":
            return _mixer_features(x)
        if self.feature_kind == "dlinear":
            return _trend_residual_features(x)
        if self.feature_kind == "patch":
            return _patch_features(x)
        if self.feature_kind == "autoformer":
            return np.concatenate([_trend_residual_features(x), _patch_features(x)], axis=1)
        if self.feature_kind == "fedformer":
            return np.concatenate([_trend_residual_features(x), _frequency_features(x)], axis=1)
        raise ValueError(f"unsupported sequence feature kind: {self.feature_kind}")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FeatureRidgeRegressor":
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(self._features(x), y.astype(float))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict(self._features(x))


def _moving_average_trend(x: np.ndarray, *, kernel_size: int = 3) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("DLinear moving average expects a 3D sequence tensor")
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if kernel_size == 1:
        return x.copy()
    left = kernel_size // 2
    right = kernel_size - 1 - left
    padded = np.pad(x, ((0, 0), (left, right), (0, 0)), mode="edge")
    windows = [padded[:, start : start + x.shape[1], :] for start in range(kernel_size)]
    return np.mean(windows, axis=0)


@dataclass
class DLinearDecompositionRegressor:
    alpha: float = 1.0
    kernel_size: int = 3

    def _decompose(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim != 3:
            raise ValueError("DLinearDecompositionRegressor expects a 3D sequence tensor")
        trend = _moving_average_trend(x, kernel_size=self.kernel_size)
        residual = x - trend
        return trend.reshape(len(x), -1), residual.reshape(len(x), -1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DLinearDecompositionRegressor":
        trend, residual = self._decompose(x)
        self.lookback_ = int(x.shape[1])
        self.feature_count_ = int(x.shape[2])
        self.trend_model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.residual_model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        target = y.astype(float)
        self.trend_model_.fit(trend, target)
        residual_target = target - self.trend_model_.predict(trend)
        self.residual_model_.fit(residual, residual_target)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        trend, residual = self._decompose(x)
        return self.trend_model_.predict(trend) + self.residual_model_.predict(residual)


def _autocorrelation_scores(residual: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    if not lags:
        return np.zeros((len(residual), 0, residual.shape[-1]), dtype=float)
    centered = residual - residual.mean(axis=1, keepdims=True)
    scores: list[np.ndarray] = []
    for lag in lags:
        left = centered[:, :-lag, :]
        right = centered[:, lag:, :]
        numerator = (left * right).mean(axis=1)
        denominator = np.sqrt((left**2).mean(axis=1) * (right**2).mean(axis=1)) + 1e-8
        scores.append(numerator / denominator)
    return np.stack(scores, axis=1)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    total = float(weights.sum())
    if total <= 1e-12:
        return np.full_like(values, 1.0 / max(len(values), 1), dtype=float)
    return weights / total


def _time_delay_aggregate(residual: np.ndarray, scores: np.ndarray, lags: tuple[int, ...], *, top_k: int) -> np.ndarray:
    if not lags:
        return residual[:, -1, :]
    k = max(1, min(int(top_k), len(lags)))
    aggregate = np.zeros((len(residual), residual.shape[-1]), dtype=float)
    for row in range(len(residual)):
        for feature in range(residual.shape[-1]):
            feature_scores = np.abs(scores[row, :, feature])
            selected = np.argsort(feature_scores)[-k:]
            weights = _softmax(feature_scores[selected])
            delayed_values = np.array([residual[row, -1 - lags[index], feature] for index in selected])
            aggregate[row, feature] = float(np.dot(weights, delayed_values))
    return aggregate


@dataclass
class AutoformerDecompositionRegressor:
    alpha: float = 1.0
    kernel_size: int = 3
    max_lag: int = 4
    top_k: int = 2

    def _decompose(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim != 3:
            raise ValueError("AutoformerDecompositionRegressor expects a 3D sequence tensor")
        trend = _moving_average_trend(x, kernel_size=self.kernel_size)
        residual = x - trend
        return trend, residual

    def _seasonal_features(self, residual: np.ndarray) -> np.ndarray:
        scores = _autocorrelation_scores(residual, self.autocorrelation_lags_)
        aggregated = _time_delay_aggregate(residual, scores, self.autocorrelation_lags_, top_k=self.top_k)
        seasonal_last = residual[:, -1, :]
        seasonal_mean = residual.mean(axis=1)
        seasonal_std = residual.std(axis=1)
        return np.concatenate(
            [
                seasonal_last,
                seasonal_mean,
                seasonal_std,
                aggregated,
                scores.reshape(len(residual), -1),
            ],
            axis=1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AutoformerDecompositionRegressor":
        trend, residual = self._decompose(x)
        self.lookback_ = int(x.shape[1])
        self.feature_count_ = int(x.shape[2])
        max_lag = max(0, min(int(self.max_lag), self.lookback_ - 1))
        self.autocorrelation_lags_ = tuple(range(1, max_lag + 1))
        trend_features = trend.reshape(len(x), -1)
        seasonal_features = self._seasonal_features(residual)
        target = y.astype(float)
        self.trend_model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.seasonal_model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.trend_model_.fit(trend_features, target)
        seasonal_target = target - self.trend_model_.predict(trend_features)
        self.seasonal_model_.fit(seasonal_features, seasonal_target)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        trend, residual = self._decompose(x)
        trend_features = trend.reshape(len(x), -1)
        seasonal_features = self._seasonal_features(residual)
        return self.trend_model_.predict(trend_features) + self.seasonal_model_.predict(seasonal_features)


def _frequency_mode_features(residual: np.ndarray, modes: tuple[int, ...]) -> np.ndarray:
    if residual.ndim != 3:
        raise ValueError("frequency mode features expect a 3D sequence tensor")
    spectrum = np.fft.rfft(residual, axis=1)
    if not modes:
        return np.zeros((len(residual), 0), dtype=float)
    pieces: list[np.ndarray] = []
    for mode in modes:
        coeff = spectrum[:, mode, :]
        pieces.extend([coeff.real, coeff.imag, np.abs(coeff)])
    return np.concatenate(pieces, axis=1)


def _frequency_reconstruction_features(residual: np.ndarray, modes: tuple[int, ...]) -> np.ndarray:
    if not modes:
        return residual[:, -1, :]
    spectrum = np.fft.rfft(residual, axis=1)
    filtered = np.zeros_like(spectrum)
    for mode in modes:
        filtered[:, mode, :] = spectrum[:, mode, :]
    reconstruction = np.fft.irfft(filtered, n=residual.shape[1], axis=1)
    return np.concatenate([reconstruction[:, -1, :], reconstruction.mean(axis=1), reconstruction.std(axis=1)], axis=1)


@dataclass
class FEDformerFrequencyRegressor:
    alpha: float = 1.0
    kernel_size: int = 3
    top_k_modes: int = 2

    def _decompose(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim != 3:
            raise ValueError("FEDformerFrequencyRegressor expects a 3D sequence tensor")
        trend = _moving_average_trend(x, kernel_size=self.kernel_size)
        residual = x - trend
        return trend, residual

    def _frequency_features(self, residual: np.ndarray) -> np.ndarray:
        mode_features = _frequency_mode_features(residual, self.frequency_modes_)
        reconstruction_features = _frequency_reconstruction_features(residual, self.frequency_modes_)
        return np.concatenate(
            [
                residual[:, -1, :],
                residual.mean(axis=1),
                residual.std(axis=1),
                reconstruction_features,
                mode_features,
            ],
            axis=1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FEDformerFrequencyRegressor":
        trend, residual = self._decompose(x)
        self.lookback_ = int(x.shape[1])
        self.feature_count_ = int(x.shape[2])
        max_mode = max(0, min(int(self.top_k_modes), self.lookback_ // 2))
        self.frequency_modes_ = tuple(range(1, max_mode + 1))
        trend_features = trend.reshape(len(x), -1)
        frequency_features = self._frequency_features(residual)
        target = y.astype(float)
        self.trend_model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.frequency_model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.trend_model_.fit(trend_features, target)
        frequency_target = target - self.trend_model_.predict(trend_features)
        self.frequency_model_.fit(frequency_features, frequency_target)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        trend, residual = self._decompose(x)
        trend_features = trend.reshape(len(x), -1)
        frequency_features = self._frequency_features(residual)
        return self.trend_model_.predict(trend_features) + self.frequency_model_.predict(frequency_features)


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


class _TorchAttentionNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cell_type: str, use_input_gate: bool = False):
        super().__init__()
        self.use_input_gate = use_input_gate
        if use_input_gate:
            self.input_gate = nn.Sequential(nn.Linear(input_size, input_size), nn.Sigmoid())
        if cell_type == "lstm":
            self.recurrent = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif cell_type == "gru":
            self.recurrent = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        else:
            raise ValueError(f"unsupported recurrent cell: {cell_type}")
        self.attention = nn.Linear(hidden_size, 1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.use_input_gate:
            x = x * self.input_gate(x)
        output, _ = self.recurrent(x)
        weights = self.attention(output).squeeze(-1).softmax(dim=1)
        context = (output * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(context).squeeze(-1)


class _TorchTemporalConvNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output = self.net(x.transpose(1, 2))
        pooled = output[:, :, -1]
        return self.head(pooled).squeeze(-1)


class _SFMNet(nn.Module):
    def __init__(self, *, lookback: int, feature_count: int, hidden_size: int, frequency_count: int):
        super().__init__()
        import torch

        self.lookback = lookback
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.frequency_count = frequency_count
        self.input_projection = nn.Linear(feature_count, hidden_size)
        self.input_gate = nn.Sequential(nn.Linear(feature_count, hidden_size), nn.Sigmoid())
        self.frequency_selector = nn.Sequential(
            nn.Linear(hidden_size * frequency_count * 2, hidden_size),
            nn.Sigmoid(),
        )
        self.head = nn.Linear(hidden_size * 3, 1)

        steps = torch.arange(lookback, dtype=torch.float32) / float(max(lookback - 1, 1))
        frequencies = torch.arange(1, frequency_count + 1, dtype=torch.float32)
        angles = 2.0 * torch.pi * steps.reshape(-1, 1) * frequencies.reshape(1, -1)
        self.register_buffer("cos_basis", torch.cos(angles))
        self.register_buffer("sin_basis", torch.sin(angles))

    def forward(self, x):
        import torch

        hidden = self.input_projection(x).tanh() * self.input_gate(x)
        cos_state = torch.einsum("bth,tk->bkh", hidden, self.cos_basis) / float(self.lookback)
        sin_state = torch.einsum("bth,tk->bkh", hidden, self.sin_basis) / float(self.lookback)
        frequency_memory = torch.cat([cos_state, sin_state], dim=1)
        frequency_flat = frequency_memory.reshape(x.shape[0], -1)
        selected_frequency = frequency_memory.mean(dim=1) * self.frequency_selector(frequency_flat)
        context = torch.cat([hidden[:, -1, :], hidden.mean(dim=1), selected_frequency], dim=1)
        return self.head(context).squeeze(-1)


class _PatchTSTNet(nn.Module):
    def __init__(
        self,
        *,
        lookback: int,
        feature_count: int,
        patch_size: int,
        stride: int,
        d_model: int,
        nhead: int,
        num_layers: int,
    ):
        super().__init__()
        import torch

        self.lookback = lookback
        self.feature_count = feature_count
        self.patch_size = patch_size
        self.stride = stride
        self.patch_count = max(1, (lookback - patch_size) // stride + 1)
        self.patch_projection = nn.Linear(patch_size, d_model)
        self.position = nn.Parameter(torch.zeros((1, self.patch_count, d_model), dtype=torch.float32))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.channel_head = nn.Linear(d_model, 1)
        self.output_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, x):
        import torch

        if x.shape[1] < self.patch_size:
            pad = self.patch_size - x.shape[1]
            x = torch.cat([x[:, :1, :].repeat(1, pad, 1), x], dim=1)
        channel_first = x.transpose(1, 2)
        patches = channel_first.unfold(dimension=2, size=self.patch_size, step=self.stride)
        patches = patches[:, :, : self.patch_count, :]
        tokens = patches.reshape(x.shape[0] * self.feature_count, self.patch_count, self.patch_size)
        encoded = self.patch_projection(tokens) + self.position
        encoded = self.encoder(encoded)
        channel_scores = self.channel_head(encoded.mean(dim=1)).reshape(x.shape[0], self.feature_count)
        return channel_scores.mean(dim=1) + self.output_bias


class _MLPMixerNet(nn.Module):
    def __init__(
        self,
        *,
        lookback: int,
        feature_count: int,
        token_hidden_size: int,
        channel_hidden_size: int,
        num_blocks: int,
    ):
        super().__init__()
        self.lookback = lookback
        self.feature_count = feature_count
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "token_norm": nn.LayerNorm(feature_count),
                        "token_mixer": nn.Sequential(
                            nn.Linear(lookback, token_hidden_size),
                            nn.GELU(),
                            nn.Linear(token_hidden_size, lookback),
                        ),
                        "channel_norm": nn.LayerNorm(feature_count),
                        "channel_mixer": nn.Sequential(
                            nn.Linear(feature_count, channel_hidden_size),
                            nn.GELU(),
                            nn.Linear(channel_hidden_size, feature_count),
                        ),
                    }
                )
            )
        self.final_norm = nn.LayerNorm(feature_count)
        self.head = nn.Linear(feature_count, 1)

    def forward(self, x):
        for block in self.blocks:
            token_input = block["token_norm"](x).transpose(1, 2)
            x = x + block["token_mixer"](token_input).transpose(1, 2)
            x = x + block["channel_mixer"](block["channel_norm"](x))
        pooled = self.final_norm(x).mean(dim=1)
        return self.head(pooled).squeeze(-1)


@dataclass
class TorchSequenceRegressor:
    cell_type: str
    hidden_size: int = 8
    epochs: int = 24
    learning_rate: float = 0.01
    seed: int = 7
    architecture: str = "recurrent"
    use_input_gate: bool = False

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
        if self.architecture == "attention":
            self.model_ = _TorchAttentionNet(
                x.shape[-1],
                self.hidden_size,
                self.cell_type,
                use_input_gate=self.use_input_gate,
            )
        elif self.architecture == "tcn":
            self.model_ = _TorchTemporalConvNet(x.shape[-1], self.hidden_size)
        else:
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


@dataclass
class SFMFrequencyMemoryRegressor:
    hidden_size: int = 8
    frequency_count: int = 3
    epochs: int = 24
    learning_rate: float = 0.01
    seed: int = 31

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SFMFrequencyMemoryRegressor":
        if x.ndim != 3:
            raise ValueError("SFMFrequencyMemoryRegressor expects a 3D sequence tensor")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.lookback_ = int(x.shape[1])
        self.feature_count_ = int(x.shape[2])
        self.frequency_modes_ = tuple(range(1, int(self.frequency_count) + 1))
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
        self.model_ = _SFMNet(
            lookback=self.lookback_,
            feature_count=self.feature_count_,
            hidden_size=int(self.hidden_size),
            frequency_count=int(self.frequency_count),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=float(self.learning_rate))
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(int(self.epochs)):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        self.model_.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("SFMFrequencyMemoryRegressor expects a 3D sequence tensor")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            output = self.model_(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class PatchTSTTransformerRegressor:
    patch_size: int = 2
    stride: int = 1
    d_model: int = 8
    nhead: int = 2
    num_layers: int = 1
    epochs: int = 24
    learning_rate: float = 0.01
    seed: int = 23

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PatchTSTTransformerRegressor":
        if x.ndim != 3:
            raise ValueError("PatchTSTTransformerRegressor expects a 3D sequence tensor")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.lookback_ = int(x.shape[1])
        self.feature_count_ = int(x.shape[2])
        self.patch_count_ = max(1, (self.lookback_ - int(self.patch_size)) // int(self.stride) + 1)
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
        self.model_ = _PatchTSTNet(
            lookback=self.lookback_,
            feature_count=self.feature_count_,
            patch_size=int(self.patch_size),
            stride=int(self.stride),
            d_model=int(self.d_model),
            nhead=int(self.nhead),
            num_layers=int(self.num_layers),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=float(self.learning_rate))
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(int(self.epochs)):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        self.model_.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("PatchTSTTransformerRegressor expects a 3D sequence tensor")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            output = self.model_(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_


@dataclass
class MLPMixerSequenceRegressor:
    token_hidden_size: int = 8
    channel_hidden_size: int = 16
    num_blocks: int = 1
    epochs: int = 24
    learning_rate: float = 0.01
    seed: int = 29

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPMixerSequenceRegressor":
        if x.ndim != 3:
            raise ValueError("MLPMixerSequenceRegressor expects a 3D sequence tensor")
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.lookback_ = int(x.shape[1])
        self.feature_count_ = int(x.shape[2])
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
        self.model_ = _MLPMixerNet(
            lookback=self.lookback_,
            feature_count=self.feature_count_,
            token_hidden_size=int(self.token_hidden_size),
            channel_hidden_size=int(self.channel_hidden_size),
            num_blocks=int(self.num_blocks),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=float(self.learning_rate))
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(int(self.epochs)):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        self.model_.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("MLPMixerSequenceRegressor expects a 3D sequence tensor")
        import torch

        torch.set_num_threads(1)
        x_scaled = (x - self.feature_mean_) / self.feature_std_
        with torch.no_grad():
            output = self.model_(torch.as_tensor(x_scaled, dtype=torch.float32)).detach().cpu().numpy()
        return output.astype(float) * self.target_std_ + self.target_mean_
