from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.objective import classification_metrics, ic_metrics, mse_metrics, ranking_metrics, utility_score


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


@dataclass(frozen=True)
class ModelSelectionResult:
    selected_spec: ModelSpec
    selected_model: "QlibDatasetModel"
    selected_model_path: Path
    model_metrics: pd.DataFrame
    predictions_by_model: dict[str, pd.DataFrame]
    sidecars: dict[str, str]


class QlibDatasetModel:
    implementation = "base"
    qlib_dataset_api = ["DatasetH", "DataHandlerLP"]

    def __init__(self, *, target_col: str | None = None, feature_cols: list[str] | None = None) -> None:
        self.target_col = target_col
        self.feature_cols = feature_cols or []
        self.parameters: dict[str, Any] = {}

    def _target_column(self, frame: pd.DataFrame) -> str:
        if self.target_col:
            return self.target_col
        label_cols = [column for column in frame.columns if str(column).startswith("Ref(")]
        if not label_cols:
            raise ValueError("qlib dataset frame does not include a label column")
        return str(label_cols[0])

    def _xy(self, dataset, *, segment: str) -> tuple[pd.DataFrame, pd.Series]:
        frame = dataset.prepare(segment).copy()
        target_col = self._target_column(frame)
        feature_cols = [
            str(column)
            for column in frame.columns
            if str(column) != target_col and not str(column).startswith("Ref(")
        ]
        if not feature_cols:
            raise ValueError("qlib dataset frame does not include feature columns")
        x = frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x = x.ffill().bfill().fillna(0.0).astype("float64")
        y = pd.to_numeric(frame[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.target_col = target_col
        self.feature_cols = feature_cols
        return x, y.astype("float64")

    def fit(self, dataset, *, segment: str = "train") -> "QlibDatasetModel":
        raise NotImplementedError

    def predict(self, dataset, *, segment: str = "test") -> pd.Series:
        raise NotImplementedError

    def _feature_frame(self, dataset, *, segment: str) -> pd.DataFrame:
        frame = dataset.prepare(segment).copy()
        x = frame[self.feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return x.ffill().bfill().fillna(0.0).astype("float64")

    def _save_payload(self, *, model_name: str) -> dict[str, Any]:
        return {
            "model": model_name,
            "implementation": self.implementation,
            "target_col": self.target_col,
            "feature_cols": self.feature_cols,
            "parameters": self.parameters,
            "qlib_dataset_api": self.qlib_dataset_api,
        }

    def save(self, path: Path, *, model_name: str) -> None:
        Path(path).write_text(json.dumps(self._save_payload(model_name=model_name), indent=2, sort_keys=True), encoding="utf-8")


def _standardize(frame: pd.DataFrame, *, mean: pd.Series | None = None, std: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean = frame.mean() if mean is None else mean
    std = frame.std(ddof=0).replace(0.0, 1.0) if std is None else std.replace(0.0, 1.0)
    return (frame - mean) / std, mean, std


class QlibLinearDatasetModel(QlibDatasetModel):
    implementation = "qlib_linear_ridge"

    def fit(self, dataset, *, segment: str = "train") -> "QlibLinearDatasetModel":
        x, y = self._xy(dataset, segment=segment)
        xz, mean, std = _standardize(x)
        design = np.column_stack([np.ones(len(xz)), xz.to_numpy()])
        penalty = np.eye(design.shape[1]) * 1e-4
        penalty[0, 0] = 0.0
        coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y.to_numpy()
        self.parameters = {"mean": mean.to_dict(), "std": std.to_dict(), "coef": coef.tolist()}
        return self

    def predict(self, dataset, *, segment: str = "test") -> pd.Series:
        x = self._feature_frame(dataset, segment=segment)
        mean = pd.Series(self.parameters["mean"])
        std = pd.Series(self.parameters["std"])
        xz, _, _ = _standardize(x, mean=mean, std=std)
        coef = np.asarray(self.parameters["coef"], dtype="float64")
        design = np.column_stack([np.ones(len(xz)), xz.to_numpy()])
        return pd.Series(design @ coef, index=x.index, name="score")


class QlibICWeightedModel(QlibDatasetModel):
    implementation = "qlib_ic_weighted_linear"

    def fit(self, dataset, *, segment: str = "train") -> "QlibICWeightedModel":
        x, y = self._xy(dataset, segment=segment)
        xz, mean, std = _standardize(x)
        weights = {}
        for column in xz.columns:
            corr = xz[column].corr(y) if xz[column].nunique() > 1 and y.nunique() > 1 else 0.0
            weights[column] = 0.0 if pd.isna(corr) else float(corr)
        if not any(abs(value) > 0 for value in weights.values()):
            weights[xz.columns[0]] = 1.0
        self.parameters = {
            "mean": mean.to_dict(),
            "std": std.to_dict(),
            "weights": weights,
            "target_mean": float(y.mean()),
            "target_std": float(y.std(ddof=0) or 1.0),
        }
        return self

    def predict(self, dataset, *, segment: str = "test") -> pd.Series:
        x = self._feature_frame(dataset, segment=segment)
        xz, _, _ = _standardize(x, mean=pd.Series(self.parameters["mean"]), std=pd.Series(self.parameters["std"]))
        weights = pd.Series(self.parameters["weights"]).reindex(xz.columns).fillna(0.0)
        raw = xz.mul(weights, axis=1).sum(axis=1)
        raw_std = float(raw.std(ddof=0) or 1.0)
        score = self.parameters["target_mean"] + raw / raw_std * self.parameters["target_std"]
        return pd.Series(score, index=x.index, name="score")


class QlibTreeStumpModel(QlibDatasetModel):
    implementation = "qlib_tree_stump"

    def fit(self, dataset, *, segment: str = "train") -> "QlibTreeStumpModel":
        x, y = self._xy(dataset, segment=segment)
        best_feature = max(
            x.columns,
            key=lambda column: abs(float(x[column].corr(y) if x[column].nunique() > 1 and y.nunique() > 1 else 0.0)),
        )
        threshold = float(x[best_feature].median())
        high = y.loc[x[best_feature] >= threshold]
        low = y.loc[x[best_feature] < threshold]
        self.parameters = {
            "feature": best_feature,
            "threshold": threshold,
            "high_prediction": float(high.mean()) if not high.empty else float(y.mean()),
            "low_prediction": float(low.mean()) if not low.empty else float(y.mean()),
        }
        return self

    def predict(self, dataset, *, segment: str = "test") -> pd.Series:
        x = self._feature_frame(dataset, segment=segment)
        feature = self.parameters["feature"]
        score = np.where(
            x[feature] >= float(self.parameters["threshold"]),
            float(self.parameters["high_prediction"]),
            float(self.parameters["low_prediction"]),
        )
        return pd.Series(score, index=x.index, name="score")


class QlibUtilityBlendModel(QlibICWeightedModel):
    implementation = "qlib_utility_risk_blend"

    def predict(self, dataset, *, segment: str = "test") -> pd.Series:
        base = super().predict(dataset, segment=segment)
        x = self._feature_frame(dataset, segment=segment)
        if "$vix" in x.columns:
            vix = x["$vix"]
            risk_gate = (1.0 - (vix.rank(pct=True) * 0.35)).clip(0.4, 1.0)
        else:
            risk_gate = pd.Series(1.0, index=x.index)
        return (base * risk_gate).rename("score")


class QlibWindowSequenceModel(QlibDatasetModel):
    implementation = "qlib_window_sequence"

    def fit(self, dataset, *, segment: str = "train") -> "QlibWindowSequenceModel":
        x, y = self._xy(dataset, segment=segment)
        close = x["$close"] if "$close" in x.columns else x.iloc[:, 0]
        sequence_signal = close.pct_change().rolling(2, min_periods=1).mean().fillna(0.0)
        slope = float(sequence_signal.corr(y)) if sequence_signal.nunique() > 1 and y.nunique() > 1 else 0.0
        if pd.isna(slope) or slope == 0.0:
            slope = 1.0
        self.parameters = {"signal_mean": float(sequence_signal.mean()), "signal_std": float(sequence_signal.std(ddof=0) or 1.0), "slope": slope, "target_std": float(y.std(ddof=0) or 1.0)}
        return self

    def predict(self, dataset, *, segment: str = "test") -> pd.Series:
        x = self._feature_frame(dataset, segment=segment)
        close = x["$close"] if "$close" in x.columns else x.iloc[:, 0]
        signal = close.pct_change().rolling(2, min_periods=1).mean().fillna(0.0)
        normalized = (signal - self.parameters["signal_mean"]) / self.parameters["signal_std"]
        return (normalized * self.parameters["slope"] * self.parameters["target_std"]).rename("score")


class QlibAttentionRankModel(QlibDatasetModel):
    implementation = "qlib_attention_rank"

    def fit(self, dataset, *, segment: str = "train") -> "QlibAttentionRankModel":
        x, y = self._xy(dataset, segment=segment)
        ranks = x.rank(pct=True)
        weights = {}
        for column in ranks.columns:
            corr = ranks[column].corr(y.rank(pct=True)) if ranks[column].nunique() > 1 and y.nunique() > 1 else 0.0
            weights[column] = abs(float(corr)) if not pd.isna(corr) else 0.0
        total = sum(weights.values()) or 1.0
        self.parameters = {"weights": {key: value / total for key, value in weights.items()}, "target_scale": float(y.std(ddof=0) or 1.0)}
        return self

    def predict(self, dataset, *, segment: str = "test") -> pd.Series:
        x = self._feature_frame(dataset, segment=segment)
        ranks = x.rank(pct=True)
        weights = pd.Series(self.parameters["weights"]).reindex(ranks.columns).fillna(0.0)
        score = (ranks.mul(weights, axis=1).sum(axis=1) - 0.5) * self.parameters["target_scale"]
        return score.rename("score")


def build_model_from_spec(spec: ModelSpec) -> QlibDatasetModel:
    if spec.name == "linear_mse":
        return QlibLinearDatasetModel()
    if spec.name == "linear_ic":
        return QlibICWeightedModel()
    if spec.name == "fast_tree_rank":
        return QlibTreeStumpModel()
    if spec.name == "gbr_utility":
        return QlibUtilityBlendModel()
    if spec.name == "lstm_mse":
        return QlibWindowSequenceModel()
    if spec.name == "transformer_rank":
        return QlibAttentionRankModel()
    raise ValueError(f"unknown qlib model spec: {spec.name}")


def load_qlib_model(path: Path) -> QlibDatasetModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    implementations: dict[str, type[QlibDatasetModel]] = {
        QlibLinearDatasetModel.implementation: QlibLinearDatasetModel,
        QlibICWeightedModel.implementation: QlibICWeightedModel,
        QlibTreeStumpModel.implementation: QlibTreeStumpModel,
        QlibUtilityBlendModel.implementation: QlibUtilityBlendModel,
        QlibWindowSequenceModel.implementation: QlibWindowSequenceModel,
        QlibAttentionRankModel.implementation: QlibAttentionRankModel,
    }
    implementation = payload["implementation"]
    if implementation not in implementations:
        raise ValueError(f"unknown qlib model implementation: {implementation}")
    model = implementations[implementation](
        target_col=payload.get("target_col"),
        feature_cols=list(payload.get("feature_cols", [])),
    )
    model.parameters = dict(payload.get("parameters", {}))
    return model


def _target_series(dataset, *, segment: str, target_col: str | None = None) -> pd.Series:
    frame = dataset.prepare(segment)
    label_col = target_col
    if label_col is None:
        label_cols = [column for column in frame.columns if str(column).startswith("Ref(")]
        if not label_cols:
            raise ValueError("qlib dataset frame does not include a label column")
        label_col = str(label_cols[0])
    return pd.to_numeric(frame[label_col], errors="coerce").fillna(0.0).astype("float64")


def _strategy_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    score_threshold: float | None = None,
) -> dict[str, float]:
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    threshold = float(aligned["predicted"].median()) if score_threshold is None else float(score_threshold)
    position = (aligned["predicted"] > threshold).astype(float)
    returns = position * aligned["actual"]
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    turnover = float(position.diff().abs().fillna(position.abs()).sum())
    ic = ic_metrics(aligned["actual"], aligned["predicted"])["ic"]
    rank_ic = ranking_metrics(aligned["actual"], aligned["predicted"])["rank_ic"]
    return_pct = float((nav.iloc[-1] / nav.iloc[0] - 1.0) * 100.0) if len(nav) > 1 else 0.0
    max_drawdown_pct = float(drawdown.min() * 100.0)
    return {
        "return_pct": round(return_pct, 10),
        "max_drawdown_pct": round(max_drawdown_pct, 10),
        "turnover": round(turnover, 10),
        "ic": ic,
        "rank_ic": rank_ic,
        "utility_score": utility_score(
            return_pct=return_pct,
            max_drawdown_pct=max_drawdown_pct,
            turnover=turnover,
            ic=ic,
            rank_ic=rank_ic,
        ),
    }


def _objective_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    objective: str,
    *,
    score_threshold: float | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(mse_metrics(actual, predicted))
    metrics.update(classification_metrics(actual, predicted))
    metrics.update(ic_metrics(actual, predicted))
    metrics.update(ranking_metrics(actual, predicted))
    metrics.update(_strategy_metrics(actual, predicted, score_threshold=score_threshold))
    metrics["objective_score"] = {
        "mse": -metrics["mse"],
        "ic": metrics["ic"],
        "ranking": metrics["rank_ic"],
        "utility": metrics["utility_score"],
    }.get(objective, metrics["utility_score"])
    return metrics


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def run_qlib_model_selection(
    dataset,
    *,
    specs: list[ModelSpec],
    output_dir: Path,
    selection_metric: str = "utility_score",
    selection_segment: str = "test",
    report_segment: str = "test",
) -> ModelSelectionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_actual = _target_series(dataset, segment=selection_segment)
    report_actual = _target_series(dataset, segment=report_segment)
    rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, pd.DataFrame] = {}
    prediction_exports: list[pd.DataFrame] = []
    fitted: dict[str, QlibDatasetModel] = {}
    for spec in specs:
        model = build_model_from_spec(spec).fit(dataset)
        train_prediction = model.predict(dataset, segment="train").fillna(0.0).rename("score")
        score_threshold = float(train_prediction.median())
        selection_prediction = (
            model.predict(dataset, segment=selection_segment).reindex(selection_actual.index).fillna(0.0).rename("score")
        )
        report_prediction = (
            selection_prediction
            if report_segment == selection_segment
            else model.predict(dataset, segment=report_segment).reindex(report_actual.index).fillna(0.0).rename("score")
        )
        selection_metrics = _objective_metrics(
            selection_actual,
            selection_prediction,
            spec.objective,
            score_threshold=score_threshold,
        )
        report_metrics = _objective_metrics(
            report_actual,
            report_prediction,
            spec.objective,
            score_threshold=score_threshold,
        )
        rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "objective": spec.objective,
                "implementation": model.implementation,
                "selection_segment": selection_segment,
                "report_segment": report_segment,
                "selection_prediction_rows": int(len(selection_prediction)),
                "prediction_rows": int(len(report_prediction)),
                "signal_threshold_source": "train_prediction_median",
                "train_signal_threshold": score_threshold,
                "uses_qlib_dataset": bool(spec.capabilities.get("uses_qlib_dataset", False)),
                **report_metrics,
                **_prefix_metrics("selection", selection_metrics),
                **_prefix_metrics(report_segment, report_metrics),
                "source": "qlib_dataset",
            }
        )
        predictions_by_model[spec.name] = report_prediction.reset_index()
        prediction_exports.append(selection_prediction.reset_index().assign(model=spec.name, segment=selection_segment))
        if report_segment != selection_segment:
            prediction_exports.append(report_prediction.reset_index().assign(model=spec.name, segment=report_segment))
        fitted[spec.name] = model

    metrics_frame = pd.DataFrame(rows)
    selection_column = f"selection_{selection_metric}"
    if selection_column not in metrics_frame.columns:
        raise ValueError(f"unknown selection metric: {selection_metric}")
    selected_idx = metrics_frame[selection_column].astype(float).idxmax()
    metrics_frame["selected"] = False
    metrics_frame.loc[selected_idx, "selected"] = True
    selected_name = str(metrics_frame.loc[selected_idx, "model"])
    selected_spec = next(spec for spec in specs if spec.name == selected_name)
    selected_model = fitted[selected_name]
    selected_model_path = output_dir / "selected_model.json"
    selected_model.save(selected_model_path, model_name=selected_spec.name)

    model_metrics_path = output_dir / "model_selection.csv"
    predictions_path = output_dir / "all_model_predictions.csv"
    metrics_frame.to_csv(model_metrics_path, index=False)
    pd.concat(prediction_exports, ignore_index=True).to_csv(predictions_path, index=False)
    return ModelSelectionResult(
        selected_spec=selected_spec,
        selected_model=selected_model,
        selected_model_path=selected_model_path,
        model_metrics=metrics_frame.sort_values(selection_column, ascending=False).reset_index(drop=True),
        predictions_by_model=predictions_by_model,
        sidecars={
            "model_selection": model_metrics_path.name,
            "all_model_predictions": predictions_path.name,
            "selected_model": selected_model_path.name,
        },
    )


def run_model_family_smoke_matrix(dataset, *, specs: list[ModelSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        model = build_model_from_spec(spec).fit(dataset)
        predictions = model.predict(dataset)
        rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "objective": spec.objective,
                "implementation": model.implementation,
                "source": "qlib_dataset",
                "prediction_rows": int(len(predictions)),
                "uses_qlib_dataset": bool(spec.capabilities.get("uses_qlib_dataset", False)),
            }
        )
    return rows


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
