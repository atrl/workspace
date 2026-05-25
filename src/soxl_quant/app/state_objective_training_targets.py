from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.soxl_quant.app.state_conditioned_objective import (
    StateConditionedObjectiveConfig,
    daily_path_metrics,
)


@dataclass(frozen=True)
class StateObjectiveTrainingTargetConfig:
    target_relative_return: float = 30.0
    target_max_dd: float = 15.0
    cluster_relative_floor: float = -30.0
    cluster_participation_floor: float = 0.35
    downtrend_position_cap: float = 0.25
    drawdown_state_threshold: float = -0.10
    relative_gap_weight: float = 1.0
    max_dd_gap_weight: float = 10.0
    cluster_relative_gap_weight: float = 1.0
    cluster_participation_gap_weight: float = 100.0
    cluster_max_dd_gap_weight: float = 5.0

    def objective_config(self) -> StateConditionedObjectiveConfig:
        return StateConditionedObjectiveConfig(
            target_relative_return=self.target_relative_return,
            target_max_dd=self.target_max_dd,
            cluster_relative_floor=self.cluster_relative_floor,
            cluster_participation_floor=self.cluster_participation_floor,
            relative_gap_weight=self.relative_gap_weight,
            max_dd_gap_weight=self.max_dd_gap_weight,
            cluster_relative_gap_weight=self.cluster_relative_gap_weight,
            cluster_participation_gap_weight=self.cluster_participation_gap_weight,
            cluster_max_dd_gap_weight=self.cluster_max_dd_gap_weight,
        )


def _window_weight(window: str, config: StateConditionedObjectiveConfig) -> float:
    if window in config.window_weights:
        return float(config.window_weights[window])
    if window.startswith("wf_"):
        return 1.0
    return 0.75


def _json_safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return float(value)


def _normalise_daily_path(daily: pd.DataFrame, *, window: str, candidate: str) -> pd.DataFrame:
    frame = daily.copy()
    if "position" not in frame.columns:
        for alias in ("router_position", "target_position", "base_position"):
            if alias in frame.columns:
                frame["position"] = frame[alias]
                break
    missing = sorted({"datetime", "position"} - set(frame.columns))
    if missing:
        raise ValueError(f"{window} daily path missing columns: {missing}")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime").reset_index(drop=True)
    frame["candidate"] = str(candidate)
    frame["window"] = str(window)
    frame["teacher_position"] = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
    if "rebound_state" not in frame.columns:
        frame["rebound_state"] = False
    if "risk_state" not in frame.columns:
        frame["risk_state"] = False
    frame["rebound_state"] = frame["rebound_state"].fillna(False).astype(bool)
    frame["risk_state"] = frame["risk_state"].fillna(False).astype(bool)
    return frame


def _metrics_for_window(
    *,
    window: str,
    candidate: str,
    daily: pd.DataFrame,
    window_metrics: pd.DataFrame,
) -> dict[str, float]:
    metrics = window_metrics.copy()
    if not metrics.empty and "candidate" in metrics.columns:
        metrics = metrics[metrics["candidate"].astype(str) == str(candidate)]
    if not metrics.empty and "window" in metrics.columns:
        metrics = metrics[metrics["window"].astype(str) == str(window)]
    if not metrics.empty:
        row = metrics.iloc[0]
        return {
            "relative_return": _json_safe_float(row.get("relative_return"), 0.0),
            "max_dd": _json_safe_float(row.get("max_dd"), 0.0),
        }
    if {"datetime", "strategy_nav", "buyhold_nav"}.issubset(daily.columns):
        observed = daily_path_metrics(daily)
        return {
            "relative_return": _json_safe_float(observed.get("relative_return"), 0.0),
            "max_dd": _json_safe_float(observed.get("max_dd"), 0.0),
        }
    return {"relative_return": 0.0, "max_dd": 0.0}


def _candidate_cluster_metrics(cluster_metrics: pd.DataFrame, *, candidate: str) -> pd.DataFrame:
    if cluster_metrics.empty:
        return pd.DataFrame(
            columns=[
                "cluster",
                "start",
                "end",
                "relative_return",
                "max_dd",
                "avg_pos",
                "cluster_relative_gap",
                "cluster_participation_gap",
                "cluster_drawdown_gap",
                "cluster_loss",
            ]
        )
    clusters = cluster_metrics.copy()
    if "candidate" in clusters.columns:
        clusters = clusters[clusters["candidate"].astype(str) == str(candidate)]
    if clusters.empty:
        return clusters
    required = {"cluster", "start", "end", "relative_return", "max_dd", "avg_pos"}
    missing = sorted(required - set(clusters.columns))
    if missing:
        raise ValueError(f"cluster_metrics missing columns: {missing}")
    clusters["start"] = pd.to_datetime(clusters["start"])
    clusters["end"] = pd.to_datetime(clusters["end"])
    return clusters.reset_index(drop=True)


def _cluster_loss_columns(
    clusters: pd.DataFrame,
    config: StateObjectiveTrainingTargetConfig,
) -> pd.DataFrame:
    if clusters.empty:
        return clusters
    result = clusters.copy()
    result["cluster_relative_gap"] = (
        float(config.cluster_relative_floor) - pd.to_numeric(result["relative_return"], errors="coerce")
    ).clip(lower=0.0)
    result["cluster_participation_gap"] = (
        float(config.cluster_participation_floor) - pd.to_numeric(result["avg_pos"], errors="coerce")
    ).clip(lower=0.0)
    result["cluster_drawdown_gap"] = (
        pd.to_numeric(result["max_dd"], errors="coerce") - float(config.target_max_dd)
    ).clip(lower=0.0)
    result["cluster_loss"] = (
        result["cluster_relative_gap"] * float(config.cluster_relative_gap_weight)
        + result["cluster_participation_gap"] * float(config.cluster_participation_gap_weight)
        + result["cluster_drawdown_gap"] * float(config.cluster_max_dd_gap_weight)
    )
    return result


def _attach_cluster_terms(
    frame: pd.DataFrame,
    clusters: pd.DataFrame,
    config: StateObjectiveTrainingTargetConfig,
) -> pd.DataFrame:
    result = frame.copy()
    result["cluster"] = ""
    result["cluster_relative_gap"] = 0.0
    result["cluster_participation_gap"] = 0.0
    result["cluster_drawdown_gap"] = 0.0
    result["is_rebound_cluster"] = False
    if clusters.empty:
        return result

    for cluster in clusters.sort_values("cluster_loss", ascending=False).itertuples(index=False):
        mask = (result["datetime"] >= cluster.start) & (result["datetime"] <= cluster.end)
        mask = mask & (result["cluster"] == "")
        if not mask.any():
            continue
        result.loc[mask, "cluster"] = str(cluster.cluster)
        result.loc[mask, "cluster_relative_gap"] = float(cluster.cluster_relative_gap)
        result.loc[mask, "cluster_participation_gap"] = float(cluster.cluster_participation_gap)
        result.loc[mask, "cluster_drawdown_gap"] = float(cluster.cluster_drawdown_gap)
        result.loc[mask, "is_rebound_cluster"] = True
    result["rebound_target_position"] = np.where(
        result["is_rebound_cluster"],
        np.maximum(result["teacher_position"], float(config.cluster_participation_floor)),
        result["teacher_position"],
    )
    return result


def build_state_objective_training_targets(
    daily_paths: Mapping[str, pd.DataFrame],
    *,
    window_metrics: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    candidate: str,
    config: StateObjectiveTrainingTargetConfig = StateObjectiveTrainingTargetConfig(),
) -> pd.DataFrame:
    """Build per-date teacher targets from a failed state-conditioned objective screen.

    The table is intended for behavior-cloning warm starts, checkpoint scoring,
    or reward shaping. It keeps rebound participation and downtrend risk caps as
    explicit terms so a training run can optimize them without hiding the
    trade-off in one scalar position rule.
    """

    clusters = _cluster_loss_columns(_candidate_cluster_metrics(cluster_metrics, candidate=candidate), config)
    objective_config = config.objective_config()
    rows: list[pd.DataFrame] = []
    for window, daily in daily_paths.items():
        frame = _normalise_daily_path(daily, window=str(window), candidate=candidate)
        metrics = _metrics_for_window(
            window=str(window),
            candidate=candidate,
            daily=frame,
            window_metrics=window_metrics,
        )
        relative_gap = max(0.0, float(config.target_relative_return) - float(metrics["relative_return"]))
        drawdown_gap = max(0.0, float(metrics["max_dd"]) - float(config.target_max_dd))
        weight = _window_weight(str(window), objective_config)
        frame["window_relative_return"] = float(metrics["relative_return"])
        frame["window_max_dd"] = float(metrics["max_dd"])
        frame["window_relative_gap"] = relative_gap
        frame["window_drawdown_gap"] = drawdown_gap
        frame["window_weight"] = weight
        frame = _attach_cluster_terms(frame, clusters, config)
        if "rebound_target_position" not in frame.columns:
            frame["rebound_target_position"] = frame["teacher_position"]
        if "drawdown_63" in frame.columns:
            drawdown_state = (
                pd.to_numeric(frame["drawdown_63"], errors="coerce").fillna(0.0)
                <= float(config.drawdown_state_threshold)
            )
        else:
            drawdown_state = pd.Series(False, index=frame.index)
        frame["is_downtrend_state"] = frame["risk_state"] | drawdown_state
        frame["risk_target_position"] = np.where(
            frame["is_downtrend_state"],
            np.minimum(frame["teacher_position"], float(config.downtrend_position_cap)),
            frame["teacher_position"],
        )
        frame["walkforward_relative_weight"] = (
            frame["window_weight"] * frame["window_relative_gap"] * float(config.relative_gap_weight)
        )
        frame["downtrend_drawdown_weight"] = np.where(
            frame["is_downtrend_state"],
            frame["window_weight"] * frame["window_drawdown_gap"] * float(config.max_dd_gap_weight),
            0.0,
        )
        frame["rebound_participation_weight"] = np.where(
            frame["is_rebound_cluster"],
            frame["cluster_relative_gap"] * float(config.cluster_relative_gap_weight)
            + frame["cluster_participation_gap"] * float(config.cluster_participation_gap_weight),
            0.0,
        )
        frame["cluster_drawdown_weight"] = np.where(
            frame["is_rebound_cluster"],
            frame["cluster_drawdown_gap"] * float(config.cluster_max_dd_gap_weight),
            0.0,
        )
        frame["objective_weight"] = (
            1.0
            + frame["walkforward_relative_weight"]
            + frame["downtrend_drawdown_weight"]
            + frame["rebound_participation_weight"]
            + frame["cluster_drawdown_weight"]
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame(
            columns=[
                "datetime",
                "candidate",
                "window",
                "teacher_position",
                "rebound_target_position",
                "risk_target_position",
                "objective_weight",
            ]
        )

    targets = pd.concat(rows, ignore_index=True).sort_values(["window", "datetime"]).reset_index(drop=True)
    columns = [
        "datetime",
        "candidate",
        "window",
        "cluster",
        "teacher_position",
        "rebound_target_position",
        "risk_target_position",
        "objective_weight",
        "walkforward_relative_weight",
        "downtrend_drawdown_weight",
        "rebound_participation_weight",
        "cluster_drawdown_weight",
        "window_relative_return",
        "window_max_dd",
        "window_relative_gap",
        "window_drawdown_gap",
        "cluster_relative_gap",
        "cluster_participation_gap",
        "cluster_drawdown_gap",
        "is_rebound_cluster",
        "is_downtrend_state",
        "rebound_state",
        "risk_state",
    ]
    passthrough = [
        column
        for column in [
            "price",
            "strategy_nav",
            "buyhold_nav",
            "base_position",
            "upside_position",
            "router_position",
            "mom_5",
            "mom_21",
            "drawdown_63",
            "rv_21",
            "vol_of_vol_21",
            "soxl_soxx_ratio_widen_21",
            "soxl_soxx_ratio_z_252",
            "soxx_mom_21",
        ]
        if column in targets.columns
    ]
    return targets[columns + passthrough]


def summarize_state_objective_training_targets(targets: pd.DataFrame) -> dict[str, object]:
    if targets.empty:
        return {
            "rows": 0,
            "windows": [],
            "top_weighted_window": "",
            "rebound_rows": 0,
            "downtrend_rows": 0,
            "required_terms": [],
        }
    by_window = (
        targets.groupby("window", dropna=False)
        .agg(
            rows=("datetime", "count"),
            mean_objective_weight=("objective_weight", "mean"),
            max_objective_weight=("objective_weight", "max"),
            rebound_rows=("is_rebound_cluster", "sum"),
            downtrend_rows=("is_downtrend_state", "sum"),
            mean_teacher_position=("teacher_position", "mean"),
            mean_rebound_target_position=("rebound_target_position", "mean"),
            mean_risk_target_position=("risk_target_position", "mean"),
        )
        .reset_index()
        .sort_values(["max_objective_weight", "mean_objective_weight", "window"], ascending=[False, False, True])
    )
    top = by_window.iloc[0]
    required_terms: list[str] = []
    if float(targets["walkforward_relative_weight"].max()) > 0.0:
        required_terms.append("walkforward_relative_return_gap_penalty")
    if float(targets["window_drawdown_gap"].max()) > 0.0 or float(targets["downtrend_drawdown_weight"].max()) > 0.0:
        required_terms.append("walkforward_drawdown_gap_penalty")
    if int(targets["is_rebound_cluster"].sum()) > 0:
        required_terms.append("high_vol_rebound_participation_floor")
    if int(targets["is_downtrend_state"].sum()) > 0:
        required_terms.append("downtrend_drawdown_state_cap")
    return {
        "rows": int(len(targets)),
        "windows": [str(value) for value in by_window["window"].tolist()],
        "top_weighted_window": str(top["window"]),
        "max_objective_weight": float(top["max_objective_weight"]),
        "rebound_rows": int(targets["is_rebound_cluster"].sum()),
        "downtrend_rows": int(targets["is_downtrend_state"].sum()),
        "mean_teacher_position": float(pd.to_numeric(targets["teacher_position"], errors="coerce").mean()),
        "mean_rebound_target_position": float(
            pd.to_numeric(targets["rebound_target_position"], errors="coerce").mean()
        ),
        "mean_risk_target_position": float(pd.to_numeric(targets["risk_target_position"], errors="coerce").mean()),
        "window_summary": by_window.to_dict(orient="records"),
        "required_terms": required_terms,
    }


__all__ = [
    "StateObjectiveTrainingTargetConfig",
    "build_state_objective_training_targets",
    "summarize_state_objective_training_targets",
]
