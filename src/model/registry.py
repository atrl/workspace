from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    objective: str
    capabilities: dict[str, Any] = field(default_factory=dict)


def quantbench_supervised_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("linear_mse", "linear", "mse", {"uses_qlib_dataset": True, "supports_feature_importance": True}),
        ModelSpec("linear_ic", "linear", "ic", {"uses_qlib_dataset": True}),
        ModelSpec("fast_tree_rank", "tree", "ranking", {"uses_qlib_dataset": True, "supports_feature_importance": True}),
        ModelSpec("gbr_utility", "tree", "utility", {"uses_qlib_dataset": True}),
        ModelSpec("lstm_mse", "sequence", "mse", {"uses_qlib_dataset": True, "supports_window_sensitivity": True}),
        ModelSpec("transformer_rank", "transformer", "ranking", {"uses_qlib_dataset": True, "supports_window_sensitivity": True}),
    ]


class QlibMeanReturnModel:
    qlib_dataset_api = ["DatasetH", "DataHandlerLP"]

    def __init__(self, *, target_col: str | None = None, prediction: float = 0.0) -> None:
        self.target_col = target_col
        self.prediction = float(prediction)

    def _target_column(self, frame: pd.DataFrame) -> str:
        if self.target_col:
            return self.target_col
        label_cols = [column for column in frame.columns if str(column).startswith("Ref(")]
        if not label_cols:
            raise ValueError("qlib dataset frame does not include a label column")
        return str(label_cols[0])

    def fit(self, dataset, *, segment: str = "train") -> "QlibMeanReturnModel":
        frame = dataset.prepare(segment)
        target_col = self._target_column(frame)
        values = frame[target_col].dropna()
        self.target_col = target_col
        self.prediction = float(values.mean()) if not values.empty else 0.0
        return self

    def predict(self, dataset, *, segment: str = "test") -> pd.Series:
        frame = dataset.prepare(segment)
        return pd.Series(self.prediction, index=frame.index, name="score")

    def save(self, path: Path) -> None:
        payload = {
            "model": "qlib_mean_return",
            "target_col": self.target_col,
            "prediction": self.prediction,
            "qlib_dataset_api": self.qlib_dataset_api,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "QlibMeanReturnModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(target_col=payload["target_col"], prediction=float(payload["prediction"]))
