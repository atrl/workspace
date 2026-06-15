from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

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


@dataclass(frozen=True)
class StateObjectiveDrawdownControlTargetConfig:
    target_relative_return: float = 30.0
    target_max_dd: float = 15.0
    drawdown_state_threshold: float = 0.10
    downtrend_position_cap: float = 0.25
    min_position_delta: float = 0.02
    drawdown_gap_weight: float = 10.0
    interval_weight: float = 5.0
    max_objective_weight: float = 25.0


@dataclass(frozen=True)
class StateObjectiveCheckpointSelectionConfig:
    alignment_loss_weight: float = 1.0
    window_gate_loss_weight: float = 0.01
    missing_target_penalty: float = 10.0
    rebound_mae_weight: float = 1.0
    risk_cap_violation_weight: float = 2.0
    legacy_score_reward: float = 0.0


@dataclass(frozen=True)
class StateObjectiveBCReadinessConfig:
    max_dominant_action_share: float = 0.90
    min_unique_actions: int = 3
    min_samples: int = 100


@dataclass(frozen=True)
class StateObjectiveReboundClusterConfig:
    min_rows: int = 3
    date_col: str = "datetime"
    rebound_col: str = "rebound_state"


@dataclass(frozen=True)
class StateObjectiveReboundTargetEnrichmentConfig:
    radius: int = 1
    min_target_position: float = 0.35
    objective_weight_decay: float = 0.50
    exclude_downtrend: bool = True
    group_cols: tuple[str, ...] = ("fold", "window")


@dataclass(frozen=True)
class TrendParticipationTargetConfig:
    enabled: bool = False
    min_target_position: float = 0.45
    max_target_position: float = 0.65
    max_teacher_position: float = 0.35
    min_anchor_gap_180: float | None = 0.50
    min_mom_63: float | None = 0.40
    min_trend_slope_63: float | None = 0.10
    min_trend_gap_63: float | None = 0.10
    max_vix_percentile_252: float | None = 0.45
    max_realized_vol_63: float | None = 1.00
    min_drawdown_63: float | None = -0.25
    objective_weight_bonus: float = 4.0
    exclude_downtrend: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.min_target_position) <= 1.0:
            raise ValueError("min_target_position must be within [0, 1]")
        if not 0.0 <= float(self.max_target_position) <= 1.0:
            raise ValueError("max_target_position must be within [0, 1]")
        if float(self.min_target_position) > float(self.max_target_position):
            raise ValueError("min_target_position must be <= max_target_position")
        if not 0.0 <= float(self.max_teacher_position) <= 1.0:
            raise ValueError("max_teacher_position must be within [0, 1]")
        if float(self.objective_weight_bonus) < 0.0:
            raise ValueError("objective_weight_bonus must be non-negative")


@dataclass(frozen=True)
class BullReentryParticipationTargetConfig:
    enabled: bool = False
    min_target_position: float = 0.60
    max_target_position: float = 0.85
    objective_weight_bonus: float = 5.0
    include_release_gate: bool = True
    exclude_downtrend: bool = True
    exclude_drawdown_control: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.min_target_position) <= 1.0:
            raise ValueError("min_target_position must be within [0, 1]")
        if not 0.0 <= float(self.max_target_position) <= 1.0:
            raise ValueError("max_target_position must be within [0, 1]")
        if float(self.min_target_position) > float(self.max_target_position):
            raise ValueError("min_target_position must be <= max_target_position")
        if float(self.objective_weight_bonus) < 0.0:
            raise ValueError("objective_weight_bonus must be non-negative")


@dataclass(frozen=True)
class ShortVolshockDeRiskTargetConfig:
    target_cap: float = 0.05
    min_feature_hits: int = 2
    min_position_delta: float = 0.02
    objective_weight_bonus: float = 6.0
    max_objective_weight: float = 25.0
    threshold_suffix: str = "q80"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.target_cap) <= 1.0:
            raise ValueError("target_cap must be within [0, 1]")
        if int(self.min_feature_hits) < 1:
            raise ValueError("min_feature_hits must be >= 1")
        if float(self.min_position_delta) < 0.0:
            raise ValueError("min_position_delta must be non-negative")
        if float(self.objective_weight_bonus) < 0.0:
            raise ValueError("objective_weight_bonus must be non-negative")


def _fold_threshold_key(value: object) -> str:
    fold = str(value or "")
    if fold.endswith("_train"):
        return fold[: -len("_train")]
    return fold


def _normalise_feature_frame(features: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    frame = features.copy()
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    elif isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
        if "instrument" in frame.columns:
            frame = frame.drop(columns=["instrument"])
        if "datetime" not in frame.columns:
            raise ValueError("feature frame multi-index must include `datetime`")
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    else:
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"feature frame missing columns: {missing}")
    return frame[["datetime", *required]].sort_values("datetime").reset_index(drop=True)


def _short_volshock_threshold_map(
    thresholds: pd.DataFrame,
    config: ShortVolshockDeRiskTargetConfig,
) -> dict[str, dict[str, float]]:
    required = {
        "fold",
        f"realized_vol_21_{config.threshold_suffix}",
        f"vol_of_vol_21_{config.threshold_suffix}",
        f"downside_semivol_21_{config.threshold_suffix}",
    }
    missing = sorted(required - set(thresholds.columns))
    if missing:
        raise ValueError(f"short-volshock thresholds missing columns: {missing}")
    result: dict[str, dict[str, float]] = {}
    for row in thresholds.to_dict("records"):
        fold = _fold_threshold_key(row.get("fold"))
        result[fold] = {
            "realized_vol_21": _json_safe_float(row[f"realized_vol_21_{config.threshold_suffix}"], np.nan),
            "vol_of_vol_21": _json_safe_float(row[f"vol_of_vol_21_{config.threshold_suffix}"], np.nan),
            "downside_semivol_21": _json_safe_float(row[f"downside_semivol_21_{config.threshold_suffix}"], np.nan),
        }
    return result


def apply_short_volshock_de_risk_target_gate(
    targets: pd.DataFrame,
    feature_frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    config: ShortVolshockDeRiskTargetConfig = ShortVolshockDeRiskTargetConfig(),
) -> pd.DataFrame:
    """Add fold-local short-volshock de-risk targets to an existing target table.

    This helper is intentionally target-generation only: it merges features by
    date, reads thresholds by the target fold, and records a cap candidate for
    later checkpoint scoring or BC. It does not use OOS replay outcomes as
    direct labels.
    """

    if targets.empty:
        result = targets.copy()
        result["short_volshock_state"] = False
        result["short_volshock_target_state"] = False
        result["short_volshock_feature_hits"] = 0
        result["short_volshock_risk_cap_delta"] = 0.0
        return result
    required_targets = {"datetime", "teacher_position", "risk_target_position"}
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_targets:
        raise ValueError(f"short-volshock targets missing columns: {missing_targets}")
    feature_cols = ["realized_vol_21", "vol_of_vol_21", "downside_semivol_21"]
    features = _normalise_feature_frame(feature_frame, feature_cols)
    threshold_map = _short_volshock_threshold_map(thresholds, config)

    result = targets.copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    result = result.drop(columns=[column for column in feature_cols if column in result.columns])
    result = result.merge(features, on="datetime", how="left")
    result["teacher_position"] = pd.to_numeric(result["teacher_position"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    result["risk_target_position"] = (
        pd.to_numeric(result["risk_target_position"], errors="coerce")
        .fillna(result["teacher_position"])
        .clip(0.0, 1.0)
    )
    if "objective_weight" not in result.columns:
        result["objective_weight"] = 1.0
    result["objective_weight"] = pd.to_numeric(result["objective_weight"], errors="coerce").fillna(1.0)
    if "is_downtrend_state" not in result.columns:
        result["is_downtrend_state"] = False
    result["is_downtrend_state"] = result["is_downtrend_state"].fillna(False).astype(bool)
    if "risk_state" not in result.columns:
        result["risk_state"] = False
    result["risk_state"] = result["risk_state"].fillna(False).astype(bool)

    if "fold" in result.columns:
        fold_keys = result["fold"].map(_fold_threshold_key)
    elif "window" in result.columns:
        fold_keys = result["window"].map(_fold_threshold_key)
    else:
        fold_keys = pd.Series("", index=result.index)
    result["short_volshock_threshold_fold"] = fold_keys.fillna("").astype(str)

    hit_counts: list[int] = []
    state_rows: list[bool] = []
    threshold_json: list[str] = []
    for idx, fold in result["short_volshock_threshold_fold"].items():
        row_thresholds = threshold_map.get(str(fold))
        if row_thresholds is None:
            hit_counts.append(0)
            state_rows.append(False)
            threshold_json.append("{}")
            continue
        hits = 0
        valid = True
        for column, threshold in row_thresholds.items():
            value = result.at[idx, column]
            if pd.isna(value) or pd.isna(threshold):
                valid = False
                break
            hits += int(float(value) >= float(threshold))
        hit_counts.append(int(hits if valid else 0))
        state_rows.append(bool(valid and hits >= int(config.min_feature_hits)))
        threshold_json.append(
            "{" + ",".join(f'"{key}":{float(value):.10g}' for key, value in sorted(row_thresholds.items())) + "}"
        )
    result["short_volshock_feature_hits"] = hit_counts
    result["short_volshock_state"] = state_rows
    result["short_volshock_thresholds_json"] = threshold_json
    result["risk_target_position_before_short_volshock"] = result["risk_target_position"]
    cap = float(config.target_cap)
    cap_delta = (result["risk_target_position_before_short_volshock"] - cap).clip(lower=0.0)
    result["short_volshock_target_state"] = result["short_volshock_state"] & (
        cap_delta > float(config.min_position_delta)
    )
    result["short_volshock_risk_cap_delta"] = np.where(
        result["short_volshock_target_state"],
        cap_delta,
        0.0,
    ).round(10)
    result["risk_target_position"] = np.where(
        result["short_volshock_target_state"],
        np.minimum(result["risk_target_position"], cap),
        result["risk_target_position"],
    )
    result["short_volshock_de_risk_weight"] = np.where(
        result["short_volshock_target_state"],
        1.0 + result["short_volshock_risk_cap_delta"] * float(config.objective_weight_bonus) * 10.0,
        0.0,
    )
    result["objective_weight"] = np.where(
        result["short_volshock_target_state"],
        np.maximum(result["objective_weight"], result["short_volshock_de_risk_weight"]),
        result["objective_weight"],
    ).clip(max=float(config.max_objective_weight))
    result["is_downtrend_state"] = result["is_downtrend_state"] | result["short_volshock_target_state"]
    result["risk_state"] = result["risk_state"] | result["short_volshock_state"]
    if "target_recipe" in result.columns:
        result["target_recipe"] = np.where(
            result["short_volshock_target_state"],
            result["target_recipe"].fillna("").astype(str) + "+short_volshock_de_risk_cap",
            result["target_recipe"].fillna("").astype(str),
        )
    else:
        result["target_recipe"] = np.where(
            result["short_volshock_target_state"],
            "short_volshock_de_risk_cap",
            "",
        )
    return result.reset_index(drop=True)


def summarize_short_volshock_de_risk_targets(targets: pd.DataFrame) -> dict[str, object]:
    if targets.empty:
        return {
            "decision": "short_volshock_de_risk_targets_empty",
            "rows": 0,
            "short_volshock_state_rows": 0,
            "target_state_rows": 0,
            "next_mechanism": "collect_fold_local_short_volshock_targets",
        }
    state_rows = int(targets.get("short_volshock_state", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    target_rows = int(targets.get("short_volshock_target_state", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    folds = (
        sorted(str(value) for value in targets["fold"].dropna().unique().tolist())
        if "fold" in targets.columns
        else []
    )
    decision = (
        "short_volshock_de_risk_targets_ready"
        if target_rows > 0
        else "short_volshock_de_risk_targets_no_training_signal"
    )
    next_mechanism = (
        "score_checkpoints_or_run_bc_with_short_volshock_targets"
        if target_rows > 0
        else "revise_short_volshock_thresholds_or_state_representation"
    )
    return {
        "decision": decision,
        "rows": int(len(targets)),
        "folds": folds,
        "short_volshock_state_rows": state_rows,
        "target_state_rows": target_rows,
        "max_risk_cap_delta": float(
            pd.to_numeric(targets.get("short_volshock_risk_cap_delta", pd.Series(dtype=float)), errors="coerce")
            .fillna(0.0)
            .max()
        ),
        "mean_objective_weight": float(
            pd.to_numeric(targets.get("objective_weight", pd.Series(dtype=float)), errors="coerce")
            .fillna(0.0)
            .mean()
        ),
        "next_mechanism": next_mechanism,
        "required_terms": [
            "fold_local_short_volshock_state",
            "short_volshock_de_risk_target_cap",
            "checkpoint_alignment_or_bc_loss",
        ],
    }


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


def _normalise_candidate_path(daily: pd.DataFrame, *, window: str, candidate: str) -> pd.DataFrame:
    frame = daily.copy()
    if "position" not in frame.columns:
        for alias in ("router_position", "target_position", "base_position"):
            if alias in frame.columns:
                frame["position"] = frame[alias]
                break
    missing = sorted({"datetime", "position"} - set(frame.columns))
    if missing:
        raise ValueError(f"{candidate}/{window} daily path missing columns: {missing}")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime").reset_index(drop=True)
    frame["candidate"] = str(candidate)
    frame["window"] = str(window)
    frame["candidate_position"] = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
    return frame


def _position_from_aliases(frame: pd.DataFrame) -> pd.Series:
    for column in ("position", "router_position", "target_position", "base_position"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    raise ValueError("daily path missing position column or alias")


def _path_drawdown_from_daily(frame: pd.DataFrame) -> pd.Series:
    if "drawdown" in frame.columns:
        drawdown = pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.0).abs()
        if float(drawdown.max()) > 1.5:
            drawdown = drawdown / 100.0
        return drawdown.clip(lower=0.0)
    for nav_col in ("strategy_nav", "nav", "account"):
        if nav_col in frame.columns:
            nav = pd.to_numeric(frame[nav_col], errors="coerce").ffill().fillna(0.0)
            peak = nav.cummax().replace(0.0, np.nan)
            return ((peak - nav) / peak).fillna(0.0).clip(lower=0.0)
    return pd.Series(0.0, index=frame.index)


def _normalise_drawdown_intervals(intervals: pd.DataFrame | None) -> pd.DataFrame:
    if intervals is None or intervals.empty:
        return pd.DataFrame(columns=["rank", "start", "end", "max_dd"])
    result = intervals.copy()
    missing = sorted({"start", "end"} - set(result.columns))
    if missing:
        raise ValueError(f"drawdown intervals missing columns: {missing}")
    if "rank" not in result.columns:
        result["rank"] = np.arange(1, len(result) + 1)
    if "max_dd" not in result.columns:
        result["max_dd"] = 0.0
    result["start"] = pd.to_datetime(result["start"])
    result["end"] = pd.to_datetime(result["end"])
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce").fillna(0).astype(int)
    result["max_dd"] = pd.to_numeric(result["max_dd"], errors="coerce").fillna(0.0)
    return result.sort_values(["rank", "start"]).reset_index(drop=True)


def build_drawdown_control_training_targets(
    daily_path: pd.DataFrame,
    *,
    intervals: pd.DataFrame | None = None,
    candidate: str,
    window: str,
    config: StateObjectiveDrawdownControlTargetConfig = StateObjectiveDrawdownControlTargetConfig(),
) -> pd.DataFrame:
    """Build state-objective target rows from a failed path drawdown interval.

    The output is compatible with state-objective alignment and checkpoint
    scoring. It only lowers the risk target on high-exposure drawdown-control
    states, while retaining the teacher position elsewhere.
    """

    if daily_path.empty:
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
    result = daily_path.copy()
    if "datetime" not in result.columns:
        raise ValueError("daily path missing datetime column")
    result["datetime"] = pd.to_datetime(result["datetime"])
    result = result.sort_values("datetime").reset_index(drop=True)
    result["candidate"] = str(candidate)
    result["window"] = str(window)
    result["teacher_position"] = _position_from_aliases(result)
    result["path_drawdown"] = _path_drawdown_from_daily(result)
    max_path_drawdown_pct = float(result["path_drawdown"].max() * 100.0)
    window_drawdown_gap = max(0.0, max_path_drawdown_pct - float(config.target_max_dd))

    interval_rows = _normalise_drawdown_intervals(intervals)
    result["source_interval_rank"] = 0
    result["source_interval_start"] = ""
    result["source_interval_end"] = ""
    result["source_interval_max_dd"] = 0.0
    result["inside_failed_drawdown_interval"] = False
    for row in interval_rows.itertuples(index=False):
        mask = (result["datetime"] >= row.start) & (result["datetime"] <= row.end)
        mask = mask & (result["source_interval_rank"] == 0)
        if not mask.any():
            continue
        result.loc[mask, "source_interval_rank"] = int(row.rank)
        result.loc[mask, "source_interval_start"] = row.start.strftime("%Y-%m-%d")
        result.loc[mask, "source_interval_end"] = row.end.strftime("%Y-%m-%d")
        result.loc[mask, "source_interval_max_dd"] = float(row.max_dd)
        result.loc[mask, "inside_failed_drawdown_interval"] = True

    result["drawdown_control_state"] = (
        result["path_drawdown"] >= float(config.drawdown_state_threshold)
    ) | result["inside_failed_drawdown_interval"].astype(bool)
    result["is_downtrend_state"] = result["drawdown_control_state"] & (
        result["teacher_position"] > float(config.downtrend_position_cap) + float(config.min_position_delta)
    )
    result["risk_target_position"] = np.where(
        result["is_downtrend_state"],
        np.minimum(result["teacher_position"], float(config.downtrend_position_cap)),
        result["teacher_position"],
    )
    result["risk_cap_delta"] = (
        result["teacher_position"] - pd.to_numeric(result["risk_target_position"], errors="coerce")
    ).clip(lower=0.0).round(10)
    result["path_drawdown_gap"] = (
        (result["path_drawdown"] - float(config.drawdown_state_threshold)).clip(lower=0.0) * 100.0
    )
    result["window_relative_return"] = 0.0
    result["window_max_dd"] = max_path_drawdown_pct
    result["window_relative_gap"] = 0.0
    result["window_drawdown_gap"] = window_drawdown_gap
    result["walkforward_relative_weight"] = 0.0
    result["downtrend_drawdown_weight"] = np.where(
        result["is_downtrend_state"],
        result["path_drawdown_gap"] * float(config.drawdown_gap_weight)
        + np.where(result["inside_failed_drawdown_interval"], float(config.interval_weight), 0.0),
        0.0,
    )
    result["rebound_target_position"] = result["teacher_position"]
    result["rebound_participation_weight"] = 0.0
    result["cluster_drawdown_weight"] = 0.0
    result["cluster"] = ""
    result["cluster_relative_gap"] = 0.0
    result["cluster_participation_gap"] = 0.0
    result["cluster_drawdown_gap"] = 0.0
    result["is_rebound_cluster"] = False
    result["rebound_state"] = False
    result["risk_state"] = result["drawdown_control_state"]
    result["objective_weight"] = (1.0 + result["downtrend_drawdown_weight"]).clip(
        upper=float(config.max_objective_weight)
    )

    columns = [
        "datetime",
        "candidate",
        "window",
        "cluster",
        "teacher_position",
        "rebound_target_position",
        "risk_target_position",
        "risk_cap_delta",
        "objective_weight",
        "walkforward_relative_weight",
        "downtrend_drawdown_weight",
        "rebound_participation_weight",
        "cluster_drawdown_weight",
        "window_relative_return",
        "window_max_dd",
        "window_relative_gap",
        "window_drawdown_gap",
        "path_drawdown",
        "path_drawdown_gap",
        "source_interval_rank",
        "source_interval_start",
        "source_interval_end",
        "source_interval_max_dd",
        "inside_failed_drawdown_interval",
        "drawdown_control_state",
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
            "nav",
            "account",
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
        if column in result.columns
    ]
    return result[columns + passthrough]


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


def enrich_state_objective_rebound_targets(
    targets: pd.DataFrame,
    config: StateObjectiveReboundTargetEnrichmentConfig = StateObjectiveReboundTargetEnrichmentConfig(),
) -> pd.DataFrame:
    """Expand sparse rebound labels to nearby rows inside the same fold/window."""

    if targets.empty:
        result = targets.copy()
        result["rebound_enrichment_source"] = ""
        result["rebound_enrichment_distance"] = 0
        return result
    required = {
        "datetime",
        "teacher_position",
        "rebound_target_position",
        "objective_weight",
        "is_rebound_cluster",
        "is_downtrend_state",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"state-objective rebound enrichment missing columns: {sorted(missing)}")
    if int(config.radius) < 0:
        raise ValueError("rebound enrichment radius must be non-negative")

    result = targets.copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    result["is_rebound_cluster"] = result["is_rebound_cluster"].fillna(False).astype(bool)
    result["is_downtrend_state"] = result["is_downtrend_state"].fillna(False).astype(bool)
    result["objective_weight"] = pd.to_numeric(result["objective_weight"], errors="coerce").fillna(1.0)
    result["teacher_position"] = pd.to_numeric(result["teacher_position"], errors="coerce").fillna(0.0)
    result["rebound_target_position"] = pd.to_numeric(
        result["rebound_target_position"],
        errors="coerce",
    ).fillna(result["teacher_position"])
    if "rebound_participation_weight" not in result.columns:
        result["rebound_participation_weight"] = 0.0
    else:
        result["rebound_participation_weight"] = pd.to_numeric(
            result["rebound_participation_weight"],
            errors="coerce",
        ).fillna(0.0)
    if "cluster" not in result.columns:
        result["cluster"] = ""
    result["rebound_enrichment_source"] = ""
    result["rebound_enrichment_distance"] = 0

    group_cols = [col for col in config.group_cols if col in result.columns]
    grouped: Iterable[tuple[object, pd.DataFrame]]
    if group_cols:
        grouped = result.groupby(group_cols, dropna=False, sort=False)
    else:
        grouped = [(None, result)]

    radius = int(config.radius)
    for _group_key, group in grouped:
        ordered = group.sort_values("datetime")
        ordered_indices = list(ordered.index)
        rebound_positions = [
            pos
            for pos, idx in enumerate(ordered_indices)
            if bool(result.at[idx, "is_rebound_cluster"])
        ]
        if not rebound_positions:
            continue
        for rebound_pos in rebound_positions:
            rebound_idx = ordered_indices[rebound_pos]
            source_weight = float(result.at[rebound_idx, "objective_weight"])
            source_target = float(result.at[rebound_idx, "rebound_target_position"])
            source_cluster = str(result.at[rebound_idx, "cluster"])
            start = max(0, rebound_pos - radius)
            stop = min(len(ordered_indices), rebound_pos + radius + 1)
            for pos in range(start, stop):
                idx = ordered_indices[pos]
                distance = abs(pos - rebound_pos)
                if distance == 0:
                    continue
                if bool(config.exclude_downtrend) and bool(result.at[idx, "is_downtrend_state"]):
                    continue
                current_source = str(result.at[idx, "rebound_enrichment_source"])
                current_distance = int(result.at[idx, "rebound_enrichment_distance"])
                if current_source and current_distance <= distance:
                    continue
                decay = float(config.objective_weight_decay) ** max(distance, 1)
                enriched_weight = max(float(result.at[idx, "objective_weight"]), source_weight * decay)
                result.at[idx, "is_rebound_cluster"] = True
                result.at[idx, "rebound_enrichment_source"] = "neighbor_rebound"
                result.at[idx, "rebound_enrichment_distance"] = int(distance)
                if not str(result.at[idx, "cluster"]) and source_cluster:
                    result.at[idx, "cluster"] = source_cluster
                result.at[idx, "rebound_target_position"] = max(
                    float(result.at[idx, "rebound_target_position"]),
                    float(result.at[idx, "teacher_position"]),
                    float(config.min_target_position),
                    source_target * decay,
                )
                result.at[idx, "objective_weight"] = enriched_weight
                result.at[idx, "rebound_participation_weight"] = max(
                    float(result.at[idx, "rebound_participation_weight"]),
                    max(0.0, enriched_weight - 1.0),
                )

    sort_cols = [col for col in ["fold", "window", "datetime"] if col in result.columns]
    return result.sort_values(sort_cols).reset_index(drop=True) if sort_cols else result.reset_index(drop=True)


def _required_trend_target_features(config: TrendParticipationTargetConfig) -> list[str]:
    required: list[str] = []
    if config.min_anchor_gap_180 is not None:
        required.append("anchor_gap_180")
    if config.min_mom_63 is not None:
        required.append("mom_63")
    if config.min_trend_slope_63 is not None:
        required.append("trend_slope_63")
    if config.min_trend_gap_63 is not None:
        required.append("trend_gap_63")
    if config.max_vix_percentile_252 is not None:
        required.append("vix_percentile_252")
    if config.max_realized_vol_63 is not None:
        required.append("realized_vol_63")
    if config.min_drawdown_63 is not None:
        required.append("drawdown_63")
    return required


def _normalise_trend_feature_frame(features: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    frame = features.copy()
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    elif isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
        if "instrument" in frame.columns:
            frame = frame.drop(columns=["instrument"])
        if "datetime" not in frame.columns:
            raise ValueError("feature frame multi-index must include `datetime`")
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    else:
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"feature frame missing trend participation columns: {missing}")
    return frame[["datetime", *required]].sort_values("datetime").reset_index(drop=True)


def _min_feature_ready(frame: pd.DataFrame, column: str, threshold: float | None) -> pd.Series:
    if threshold is None:
        return pd.Series(True, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce") >= float(threshold)


def _max_feature_ready(frame: pd.DataFrame, column: str, threshold: float | None) -> pd.Series:
    if threshold is None:
        return pd.Series(True, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce") <= float(threshold)


def apply_trend_participation_target_gate(
    targets: pd.DataFrame,
    feature_frame: pd.DataFrame,
    config: TrendParticipationTargetConfig = TrendParticipationTargetConfig(),
) -> pd.DataFrame:
    """Lift fold-local target positions for guarded high-trend participation states."""

    if "datetime" not in targets.columns:
        raise ValueError("targets missing required column: datetime")
    required_targets = {
        "teacher_position",
        "rebound_target_position",
        "risk_target_position",
        "is_rebound_cluster",
        "is_downtrend_state",
    }
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_targets:
        raise ValueError(f"targets missing columns: {missing_targets}")

    result = targets.copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    for column in ("teacher_position", "rebound_target_position", "risk_target_position"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    for column in ("is_rebound_cluster", "is_downtrend_state"):
        result[column] = result[column].fillna(False).astype(bool)
    if "risk_target_position_original" not in result.columns:
        result["risk_target_position_original"] = result["risk_target_position"]
    else:
        result["risk_target_position_original"] = pd.to_numeric(
            result["risk_target_position_original"],
            errors="coerce",
        ).fillna(result["risk_target_position"]).clip(0.0, 1.0)
    if "objective_weight" not in result.columns:
        result["objective_weight"] = 1.0
    result["objective_weight"] = pd.to_numeric(result["objective_weight"], errors="coerce").fillna(1.0)
    if "rebound_participation_weight" not in result.columns:
        result["rebound_participation_weight"] = 0.0
    result["rebound_participation_weight"] = pd.to_numeric(
        result["rebound_participation_weight"],
        errors="coerce",
    ).fillna(0.0)

    required_features = _required_trend_target_features(config)
    features = _normalise_trend_feature_frame(feature_frame, required_features)
    result = result.drop(columns=[column for column in required_features if column in result.columns])
    result = result.merge(features, on="datetime", how="left")

    if not bool(config.enabled):
        result["trend_participation_signal_ready"] = False
        result["trend_participation_guard_ready"] = False
        result["trend_participation_target_state"] = False
        return result
    if required_features and result[required_features].notna().any(axis=1).sum() == 0:
        raise ValueError("targets and feature_frame have no overlapping non-null feature rows")

    signal_ready = (
        _min_feature_ready(result, "anchor_gap_180", config.min_anchor_gap_180)
        & _min_feature_ready(result, "mom_63", config.min_mom_63)
        & _min_feature_ready(result, "trend_slope_63", config.min_trend_slope_63)
        & _min_feature_ready(result, "trend_gap_63", config.min_trend_gap_63)
    )
    guard_ready = (
        _max_feature_ready(result, "vix_percentile_252", config.max_vix_percentile_252)
        & _max_feature_ready(result, "realized_vol_63", config.max_realized_vol_63)
        & _min_feature_ready(result, "drawdown_63", config.min_drawdown_63)
    )
    base_ready = result["teacher_position"] <= float(config.max_teacher_position)
    downtrend_ready = ~result["is_downtrend_state"] if bool(config.exclude_downtrend) else True
    target_state = signal_ready & guard_ready & base_ready & downtrend_ready

    floor_position = np.minimum(
        float(config.max_target_position),
        np.maximum(result["teacher_position"], float(config.min_target_position)),
    )
    result["trend_participation_signal_ready"] = signal_ready.astype(bool)
    result["trend_participation_guard_ready"] = guard_ready.astype(bool)
    result["trend_participation_target_state"] = target_state.astype(bool)
    result["is_rebound_cluster"] = result["is_rebound_cluster"] | target_state
    result["rebound_target_position"] = np.where(
        target_state,
        np.maximum(result["rebound_target_position"], floor_position),
        result["rebound_target_position"],
    )
    result["rebound_participation_weight"] = np.where(
        target_state,
        np.maximum(result["rebound_participation_weight"], float(config.objective_weight_bonus)),
        result["rebound_participation_weight"],
    )
    result["objective_weight"] = np.where(
        target_state,
        np.maximum(result["objective_weight"], 1.0 + float(config.objective_weight_bonus)),
        result["objective_weight"],
    )
    result["trend_participation_target_reason"] = np.where(
        target_state,
        "fold_local_high_trend_low_vol_participation",
        "",
    )
    return result.reset_index(drop=True)


def apply_bull_reentry_participation_target_gate(
    targets: pd.DataFrame,
    config: BullReentryParticipationTargetConfig = BullReentryParticipationTargetConfig(),
) -> pd.DataFrame:
    """Lift fold-local bull re-entry targets without using evaluation rows."""

    if "datetime" not in targets.columns:
        raise ValueError("targets missing required column: datetime")
    required_targets = {"teacher_position", "rebound_target_position", "risk_target_position"}
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_targets:
        raise ValueError(f"targets missing columns: {missing_targets}")

    result = targets.copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    for column in ("teacher_position", "rebound_target_position", "risk_target_position"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "risk_target_position_original" not in result.columns:
        result["risk_target_position_original"] = result["risk_target_position"]
    else:
        result["risk_target_position_original"] = pd.to_numeric(
            result["risk_target_position_original"],
            errors="coerce",
        ).fillna(result["risk_target_position"]).clip(0.0, 1.0)
    if "objective_weight" not in result.columns:
        result["objective_weight"] = 1.0
    result["objective_weight"] = pd.to_numeric(result["objective_weight"], errors="coerce").fillna(1.0)
    if "rebound_participation_weight" not in result.columns:
        result["rebound_participation_weight"] = 0.0
    result["rebound_participation_weight"] = pd.to_numeric(
        result["rebound_participation_weight"],
        errors="coerce",
    ).fillna(0.0)
    for column in ("is_downtrend_state", "drawdown_control_state", "release_gate_state"):
        if column not in result.columns:
            result[column] = False
        result[column] = result[column].fillna(False).astype(bool)

    if "target_scope" in result.columns:
        fold_local_ready = result["target_scope"].fillna("").astype(str).eq("fold_local")
    else:
        fold_local_ready = pd.Series(True, index=result.index)

    date_values = result["datetime"].dt.normalize()
    if {"train_start", "train_end"}.issubset(result.columns):
        train_start = pd.to_datetime(result["train_start"], errors="coerce").dt.normalize()
        train_end = pd.to_datetime(result["train_end"], errors="coerce").dt.normalize()
        train_ready = train_start.notna() & train_end.notna() & (date_values >= train_start) & (date_values <= train_end)
    else:
        train_ready = pd.Series(True, index=result.index)
    if {"eval_start", "eval_end"}.issubset(result.columns):
        eval_start = pd.to_datetime(result["eval_start"], errors="coerce").dt.normalize()
        eval_end = pd.to_datetime(result["eval_end"], errors="coerce").dt.normalize()
        eval_overlap = eval_start.notna() & eval_end.notna() & (date_values >= eval_start) & (date_values <= eval_end)
    else:
        eval_overlap = pd.Series(False, index=result.index)

    action_ready = (
        result["action"].fillna("").astype(str).str.upper().eq("BUY")
        if "action" in result.columns
        else pd.Series(False, index=result.index)
    )
    release_ready = result["release_gate_state"] if bool(config.include_release_gate) else pd.Series(False, index=result.index)
    downtrend_ready = ~result["is_downtrend_state"] if bool(config.exclude_downtrend) else pd.Series(True, index=result.index)
    drawdown_ready = (
        ~result["drawdown_control_state"]
        if bool(config.exclude_drawdown_control)
        else pd.Series(True, index=result.index)
    )
    signal_ready = action_ready | release_ready
    target_state = (
        bool(config.enabled)
        & fold_local_ready
        & train_ready
        & ~eval_overlap
        & signal_ready
        & downtrend_ready
        & drawdown_ready
    )

    target_position = np.minimum(
        float(config.max_target_position),
        np.maximum(result["risk_target_position"], float(config.min_target_position)),
    )
    result["bull_reentry_signal_ready"] = signal_ready.astype(bool)
    result["bull_reentry_train_window_ready"] = (train_ready & ~eval_overlap).astype(bool)
    result["bull_reentry_target_state"] = target_state.astype(bool)
    result["bull_reentry_target_position"] = np.where(
        target_state,
        target_position,
        result["risk_target_position"],
    )
    result["bull_reentry_target_lift"] = np.where(
        target_state,
        (result["bull_reentry_target_position"] - result["risk_target_position"]).clip(lower=0.0),
        0.0,
    ).round(10)
    result["risk_target_position"] = np.where(
        target_state,
        np.maximum(result["risk_target_position"], result["bull_reentry_target_position"]),
        result["risk_target_position"],
    )
    result["rebound_target_position"] = np.where(
        target_state,
        np.maximum(result["rebound_target_position"], result["bull_reentry_target_position"]),
        result["rebound_target_position"],
    )
    result["objective_weight"] = np.where(
        target_state,
        result["objective_weight"] + float(config.objective_weight_bonus),
        result["objective_weight"],
    )
    result["rebound_participation_weight"] = np.where(
        target_state,
        np.maximum(result["rebound_participation_weight"], float(config.objective_weight_bonus)),
        result["rebound_participation_weight"],
    )
    result["bull_reentry_target_reason"] = np.where(
        target_state,
        "fold_local_bull_reentry_participation",
        "",
    )
    return result.reset_index(drop=True)


def filter_state_objective_targets_for_train_window(
    targets: pd.DataFrame,
    *,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    eval_start: str | pd.Timestamp | None = None,
    eval_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter state-objective targets to a train window and reject eval overlap."""

    if "datetime" not in targets.columns:
        raise ValueError("targets missing required column: datetime")
    train_start_ts = pd.Timestamp(train_start).normalize()
    train_end_ts = pd.Timestamp(train_end).normalize()
    if train_end_ts < train_start_ts:
        raise ValueError("train_end is before train_start")
    if eval_start is not None and eval_end is not None:
        eval_start_ts = pd.Timestamp(eval_start).normalize()
        eval_end_ts = pd.Timestamp(eval_end).normalize()
        if eval_end_ts < eval_start_ts:
            raise ValueError("eval_end is before eval_start")
        if train_start_ts <= eval_end_ts and train_end_ts >= eval_start_ts:
            raise ValueError("train window overlaps eval window")

    frame = targets.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    dates = frame["datetime"].dt.normalize()
    return frame.loc[(dates >= train_start_ts) & (dates <= train_end_ts)].reset_index(drop=True)


def _fold_value(fold: Mapping[str, object], key: str, default: object = None) -> object:
    value = fold.get(key, default)
    if value is None:
        raise ValueError(f"fold missing required key: {key}")
    return value


def _filter_cluster_metrics_for_train_window(
    cluster_metrics: pd.DataFrame,
    *,
    candidate: str,
    fold: str | None = None,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
) -> pd.DataFrame:
    if cluster_metrics.empty:
        return cluster_metrics.copy()
    required = {"cluster", "start", "end", "relative_return", "max_dd", "avg_pos"}
    missing = sorted(required - set(cluster_metrics.columns))
    if missing:
        raise ValueError(f"cluster_metrics missing columns: {missing}")
    clusters = cluster_metrics.copy()
    if "candidate" in clusters.columns:
        clusters = clusters[clusters["candidate"].astype(str) == str(candidate)]
    if fold is not None and "fold" in clusters.columns:
        clusters = clusters[clusters["fold"].astype(str) == str(fold)]
    clusters["start"] = pd.to_datetime(clusters["start"])
    clusters["end"] = pd.to_datetime(clusters["end"])
    train_start_ts = pd.Timestamp(train_start).normalize()
    train_end_ts = pd.Timestamp(train_end).normalize()
    mask = (clusters["start"].dt.normalize() >= train_start_ts) & (
        clusters["end"].dt.normalize() <= train_end_ts
    )
    return clusters.loc[mask].reset_index(drop=True)


def _empty_rebound_cluster_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate",
            "fold",
            "cluster",
            "start",
            "end",
            "rows",
            "return",
            "buy_and_hold_return",
            "relative_return",
            "max_dd",
            "avg_pos",
        ]
    )


def discover_rebound_clusters_from_daily_path(
    daily: pd.DataFrame,
    *,
    candidate: str,
    fold: str,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    config: StateObjectiveReboundClusterConfig = StateObjectiveReboundClusterConfig(),
) -> pd.DataFrame:
    """Discover train-window rebound clusters from a daily teacher path."""

    if int(config.min_rows) < 1:
        raise ValueError("min_rows must be >= 1")
    if config.date_col not in daily.columns:
        raise ValueError(f"daily path missing columns: [{config.date_col!r}]")
    if config.rebound_col not in daily.columns:
        return _empty_rebound_cluster_frame()

    frame = daily.copy()
    if config.date_col != "datetime":
        frame["datetime"] = pd.to_datetime(frame[config.date_col])
    train_daily = filter_state_objective_targets_for_train_window(
        frame,
        train_start=train_start,
        train_end=train_end,
    )
    if train_daily.empty:
        return _empty_rebound_cluster_frame()
    train_daily = train_daily.sort_values("datetime").reset_index(drop=True)
    rebound_mask = train_daily[config.rebound_col].fillna(False).astype(bool)
    if not rebound_mask.any():
        return _empty_rebound_cluster_frame()

    run_id = rebound_mask.ne(rebound_mask.shift(fill_value=False)).cumsum()
    rows: list[dict[str, object]] = []
    cluster_index = 1
    for _, run in train_daily.loc[rebound_mask].groupby(run_id[rebound_mask], sort=False):
        if len(run) < int(config.min_rows):
            continue
        metrics = daily_path_metrics(run)
        rows.append(
            {
                "candidate": str(candidate),
                "fold": str(fold),
                "cluster": f"{fold}_rebound_{cluster_index}",
                "start": pd.Timestamp(metrics["start"]),
                "end": pd.Timestamp(metrics["end"]),
                "rows": int(metrics["rows"]),
                "return": _json_safe_float(metrics.get("return"), 0.0),
                "buy_and_hold_return": _json_safe_float(metrics.get("buy_and_hold_return"), 0.0),
                "relative_return": _json_safe_float(metrics.get("relative_return"), 0.0),
                "max_dd": _json_safe_float(metrics.get("max_dd"), 0.0),
                "avg_pos": _json_safe_float(metrics.get("avg_pos"), 0.0),
            }
        )
        cluster_index += 1
    if not rows:
        return _empty_rebound_cluster_frame()
    return pd.DataFrame(rows)


def build_fold_local_state_objective_training_targets(
    daily_paths: Mapping[str, pd.DataFrame],
    *,
    folds: Iterable[Mapping[str, object]],
    cluster_metrics: pd.DataFrame,
    candidate: str,
    config: StateObjectiveTrainingTargetConfig = StateObjectiveTrainingTargetConfig(),
) -> pd.DataFrame:
    """Build state-objective targets using only each fold's train window."""

    rows: list[pd.DataFrame] = []
    for fold in folds:
        name = str(_fold_value(fold, "name"))
        train_start = _fold_value(fold, "train_start")
        train_end = _fold_value(fold, "train_end")
        eval_start = _fold_value(fold, "eval_start")
        eval_end = _fold_value(fold, "eval_end")
        source_window = str(fold.get("source_window", name))
        if source_window not in daily_paths:
            raise ValueError(f"daily_paths missing source window: {source_window}")
        train_daily = filter_state_objective_targets_for_train_window(
            daily_paths[source_window],
            train_start=train_start,
            train_end=train_end,
            eval_start=eval_start,
            eval_end=eval_end,
        )
        if train_daily.empty:
            continue
        metrics = daily_path_metrics(train_daily)
        window_metrics = pd.DataFrame(
            {
                "candidate": [str(candidate)],
                "window": [name],
                "relative_return": [_json_safe_float(metrics.get("relative_return"), 0.0)],
                "max_dd": [_json_safe_float(metrics.get("max_dd"), 0.0)],
            }
        )
        train_clusters = _filter_cluster_metrics_for_train_window(
            cluster_metrics,
            candidate=str(candidate),
            fold=name,
            train_start=train_start,
            train_end=train_end,
        )
        fold_targets = build_state_objective_training_targets(
            {name: train_daily},
            window_metrics=window_metrics,
            cluster_metrics=train_clusters,
            candidate=str(candidate),
            config=config,
        )
        if fold_targets.empty:
            continue
        fold_targets["fold"] = name
        fold_targets["source_window"] = source_window
        fold_targets["target_scope"] = "fold_local"
        fold_targets["train_start"] = str(pd.Timestamp(train_start).date())
        fold_targets["train_end"] = str(pd.Timestamp(train_end).date())
        fold_targets["eval_start"] = str(pd.Timestamp(eval_start).date())
        fold_targets["eval_end"] = str(pd.Timestamp(eval_end).date())
        rows.append(fold_targets)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["fold", "datetime"]).reset_index(drop=True)


def audit_state_objective_target_scope(
    targets: pd.DataFrame,
    *,
    folds: Iterable[Mapping[str, object]],
    date_col: str = "datetime",
) -> pd.DataFrame:
    """Audit whether a target table can be used for fold-local training.

    Global OOS target tables are useful diagnostics, but they must not become
    direct training labels when they touch evaluation dates. This audit keeps
    that leakage check explicit before a notebook hands targets to BC or PPO.
    """

    if date_col not in targets.columns:
        raise ValueError(f"targets missing required column: {date_col}")
    if not folds:
        raise ValueError("folds must contain at least one fold definition")

    frame = targets.copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    if "window" not in frame.columns:
        frame["window"] = ""
    if "target_scope" not in frame.columns:
        frame["target_scope"] = ""
    frame["window"] = frame["window"].fillna("").astype(str)
    frame["target_scope"] = frame["target_scope"].fillna("").astype(str)

    fold_names = {str(_fold_value(fold, "name")) for fold in folds}
    rows: list[dict[str, object]] = []
    for (window, target_scope), group in frame.groupby(["window", "target_scope"], dropna=False, sort=True):
        dates = pd.to_datetime(group[date_col]).dt.normalize()
        target_fold_values = (
            [str(value) for value in group["fold"].dropna().unique().tolist()]
            if "fold" in group.columns
            else []
        )
        intended_fold = ""
        if len(target_fold_values) == 1:
            intended_fold = target_fold_values[0]
        elif str(window) in fold_names:
            intended_fold = str(window)
        for fold in folds:
            fold_name = str(_fold_value(fold, "name"))
            if str(target_scope) == "fold_local" and intended_fold and intended_fold != fold_name:
                continue
            train_start = pd.Timestamp(_fold_value(fold, "train_start")).normalize()
            train_end = pd.Timestamp(_fold_value(fold, "train_end")).normalize()
            eval_start = pd.Timestamp(_fold_value(fold, "eval_start")).normalize()
            eval_end = pd.Timestamp(_fold_value(fold, "eval_end")).normalize()
            if train_end < train_start:
                raise ValueError(f"{fold_name} train_end is before train_start")
            if eval_end < eval_start:
                raise ValueError(f"{fold_name} eval_end is before eval_start")

            train_overlap = (dates >= train_start) & (dates <= train_end)
            eval_overlap = (dates >= eval_start) & (dates <= eval_end)
            outside_overlap = ~(train_overlap | eval_overlap)
            train_rows = int(train_overlap.sum())
            eval_rows = int(eval_overlap.sum())
            outside_rows = int(outside_overlap.sum())
            scope_issue = ""
            if str(target_scope) != "fold_local":
                scope_issue = "global_or_unspecified_target_scope"
            elif eval_rows > 0:
                scope_issue = "eval_overlap"
            elif train_rows <= 0:
                scope_issue = "no_train_overlap"
            rows.append(
                {
                    "window": str(window),
                    "target_scope": str(target_scope),
                    "target_fold": intended_fold,
                    "fold": fold_name,
                    "rows": int(len(group)),
                    "train_overlap_rows": train_rows,
                    "eval_overlap_rows": eval_rows,
                    "outside_train_eval_rows": outside_rows,
                    "leakage_gate": scope_issue == "",
                    "scope_issue": scope_issue,
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "eval_start": str(eval_start.date()),
                    "eval_end": str(eval_end.date()),
                }
            )
    return pd.DataFrame(rows).sort_values(["leakage_gate", "scope_issue", "fold", "window"], ascending=[
        True,
        True,
        True,
        True,
    ]).reset_index(drop=True)


def summarize_state_objective_target_scope_audit(audit: pd.DataFrame) -> dict[str, object]:
    if audit.empty:
        return {
            "decision": "state_objective_targets_no_scope_rows",
            "training_allowed": False,
            "next_mechanism": "collect_state_objective_targets",
        }

    failed = audit[~audit["leakage_gate"].astype(bool)]
    training_allowed = failed.empty
    if training_allowed:
        decision = "state_objective_targets_training_ready"
        next_mechanism = "run_fold_local_state_objective_training_or_bc"
        top_issue = ""
    else:
        decision = "state_objective_targets_leakage_guard_no_training"
        next_mechanism = "regenerate_fold_local_state_objective_targets"
        top_issue = str(failed.iloc[0].get("scope_issue", ""))

    return {
        "decision": decision,
        "training_allowed": bool(training_allowed),
        "leakage_gate": bool(training_allowed),
        "audited_groups": int(len(audit)),
        "blocked_groups": int(len(failed)),
        "eval_overlap_rows": int(pd.to_numeric(audit["eval_overlap_rows"], errors="coerce").fillna(0).sum()),
        "outside_train_eval_rows": int(
            pd.to_numeric(audit["outside_train_eval_rows"], errors="coerce").fillna(0).sum()
        ),
        "scope_issues": sorted(str(value) for value in failed["scope_issue"].dropna().unique()),
        "top_scope_issue": top_issue,
        "next_mechanism": next_mechanism,
    }


def _target_position_with_priority(row: Mapping[str, object], *, priority: str) -> float:
    teacher = _json_safe_float(row.get("teacher_position"), 0.0)
    rebound = _json_safe_float(row.get("rebound_target_position"), teacher)
    risk = _json_safe_float(row.get("risk_target_position"), teacher)
    is_rebound = bool(row.get("is_rebound_cluster", False))
    is_downtrend = bool(row.get("is_downtrend_state", False))
    if priority == "risk":
        if is_downtrend:
            return risk
        if is_rebound:
            return rebound
        return teacher
    if priority == "rebound":
        if is_rebound:
            return rebound
        if is_downtrend:
            return risk
        return teacher
    raise ValueError(f"unknown target_position_priority: {priority!r}")


def audit_state_objective_target_lift(
    targets: pd.DataFrame,
    *,
    target_position_priority: str = "risk",
    min_lift: float = 1e-9,
    group_cols: tuple[str, ...] = ("fold", "window"),
    baseline_risk_position_col: str = "risk_target_position",
) -> pd.DataFrame:
    """Audit whether sparse-state target labels change the effective position target.

    A target table can be fold-local and still fail as BC/PPO training data if a
    higher-priority state cap erases the action the sparse label is meant to
    teach. This audit checks whether rebound-labeled rows actually create a
    positive target lift relative to the no-rebound baseline.
    """

    required = {
        "teacher_position",
        "rebound_target_position",
        "risk_target_position",
        "is_rebound_cluster",
        "is_downtrend_state",
    }
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"targets missing columns: {missing}")
    if min_lift < 0.0:
        raise ValueError("min_lift must be non-negative")

    frame = targets.copy()
    if baseline_risk_position_col not in frame.columns:
        raise ValueError(f"targets missing baseline risk column: {baseline_risk_position_col!r}")

    for col in ["teacher_position", "rebound_target_position", "risk_target_position", baseline_risk_position_col]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    for col in ["is_rebound_cluster", "is_downtrend_state"]:
        frame[col] = frame[col].fillna(False).astype(bool)

    frame["current_target_position"] = [
        _target_position_with_priority(row, priority=str(target_position_priority))
        for row in frame.to_dict("records")
    ]
    frame["no_rebound_position"] = np.where(
        frame["is_downtrend_state"],
        frame[baseline_risk_position_col],
        frame["teacher_position"],
    )
    frame["candidate_release_position"] = np.where(
        frame["is_downtrend_state"],
        np.maximum(frame[baseline_risk_position_col], frame["rebound_target_position"]),
        np.maximum(frame["teacher_position"], frame["rebound_target_position"]),
    )
    frame["current_effective_lift"] = frame["current_target_position"] - frame["no_rebound_position"]
    frame["candidate_release_lift"] = frame["candidate_release_position"] - frame["no_rebound_position"]

    available_group_cols = tuple(col for col in group_cols if col in frame.columns)
    if available_group_cols:
        grouped = frame.groupby(list(available_group_cols), dropna=False, sort=True)
    else:
        frame["_target_lift_group"] = "all"
        available_group_cols = ("_target_lift_group",)
        grouped = frame.groupby(list(available_group_cols), dropna=False, sort=True)

    rows: list[dict[str, object]] = []
    threshold = float(min_lift)
    for group_key, group in grouped:
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {col: str(value) for col, value in zip(available_group_cols, group_values)}
        rebound_mask = group["is_rebound_cluster"].astype(bool)
        downtrend_mask = group["is_downtrend_state"].astype(bool)
        rebound_rows = group[rebound_mask]
        current_lift_mask = rebound_mask & (group["current_effective_lift"] > threshold)
        release_lift_mask = rebound_mask & (group["candidate_release_lift"] > threshold)
        row.update(
            {
                "rows": int(len(group)),
                "rebound_rows": int(rebound_mask.sum()),
                "downtrend_rows": int(downtrend_mask.sum()),
                "rebound_downtrend_rows": int((rebound_mask & downtrend_mask).sum()),
                "current_positive_rebound_lift_rows": int(current_lift_mask.sum()),
                "candidate_release_positive_lift_rows": int(release_lift_mask.sum()),
                "mean_current_effective_lift": float(
                    group.loc[rebound_mask, "current_effective_lift"].mean()
                )
                if rebound_mask.any()
                else 0.0,
                "mean_candidate_release_lift": float(
                    group.loc[rebound_mask, "candidate_release_lift"].mean()
                )
                if rebound_mask.any()
                else 0.0,
                "max_current_effective_lift": float(rebound_rows["current_effective_lift"].max())
                if not rebound_rows.empty
                else 0.0,
                "max_candidate_release_lift": float(rebound_rows["candidate_release_lift"].max())
                if not rebound_rows.empty
                else 0.0,
                "lift_semantics_gate": bool(int(current_lift_mask.sum()) > 0),
                "release_candidate_gate": bool(int(release_lift_mask.sum()) > 0),
                "target_position_priority": str(target_position_priority),
                "min_lift": float(min_lift),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["lift_semantics_gate", "release_candidate_gate", *available_group_cols],
        ascending=[True, False, *([True] * len(available_group_cols))],
    ).reset_index(drop=True)


def apply_state_objective_sparse_release_gate(
    targets: pd.DataFrame,
    release_scores: pd.DataFrame,
    *,
    feature_set: str | None = None,
    threshold: float | None = None,
    date_col: str = "datetime",
    predicted_col: str = "predicted_release",
    min_lift: float = 0.01,
    require_rebound_label: bool = True,
) -> pd.DataFrame:
    """Lift risk targets only where a sparse release gate is active.

    The existing BC target selector gives downtrend risk caps precedence. This
    helper keeps that selector usable by preserving the original risk target as a
    baseline column, then raising ``risk_target_position`` only on gated rebound
    rows where the release target is materially above the original cap.
    """

    required_targets = {
        date_col,
        "teacher_position",
        "rebound_target_position",
        "risk_target_position",
        "is_rebound_cluster",
        "is_downtrend_state",
    }
    missing_targets = sorted(required_targets - set(targets.columns))
    if missing_targets:
        raise ValueError(f"targets missing columns: {missing_targets}")
    required_scores = {date_col, predicted_col}
    missing_scores = sorted(required_scores - set(release_scores.columns))
    if missing_scores:
        raise ValueError(f"release_scores missing columns: {missing_scores}")
    if min_lift < 0.0:
        raise ValueError("min_lift must be non-negative")

    result = targets.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    for col in ("teacher_position", "rebound_target_position", "risk_target_position"):
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    for col in ("is_rebound_cluster", "is_downtrend_state"):
        result[col] = result[col].fillna(False).astype(bool)
    if "risk_target_position_original" not in result.columns:
        result["risk_target_position_original"] = result["risk_target_position"]
    else:
        result["risk_target_position_original"] = pd.to_numeric(
            result["risk_target_position_original"],
            errors="coerce",
        ).fillna(result["risk_target_position"])

    scores = release_scores.copy()
    scores[date_col] = pd.to_datetime(scores[date_col])
    if feature_set is not None and "feature_set" in scores.columns:
        scores = scores[scores["feature_set"].astype(str) == str(feature_set)]
    if threshold is not None and "threshold" in scores.columns:
        scores = scores[np.isclose(pd.to_numeric(scores["threshold"], errors="coerce"), float(threshold))]
    scores[predicted_col] = scores[predicted_col].fillna(False).astype(bool)
    scores["_date_key"] = scores[date_col].dt.normalize()
    sort_cols = [predicted_col]
    ascending = [False]
    if "state_score" in scores.columns:
        scores["state_score"] = pd.to_numeric(scores["state_score"], errors="coerce")
        sort_cols.append("state_score")
        ascending.append(False)
    scores = scores.sort_values(sort_cols, ascending=ascending).drop_duplicates("_date_key", keep="first")

    result["_date_key"] = result[date_col].dt.normalize()
    predicted_by_date = scores.set_index("_date_key")[predicted_col].to_dict() if not scores.empty else {}
    result["release_gate_predicted"] = result["_date_key"].map(predicted_by_date).fillna(False).astype(bool)
    if "state_score" in scores.columns:
        score_by_date = scores.set_index("_date_key")["state_score"].to_dict()
        result["release_gate_state_score"] = result["_date_key"].map(score_by_date)
    else:
        result["release_gate_state_score"] = np.nan

    baseline_risk = pd.to_numeric(result["risk_target_position_original"], errors="coerce").fillna(
        result["risk_target_position"]
    )
    candidate_release = np.maximum(baseline_risk, result["rebound_target_position"])
    result["release_gate_lift"] = candidate_release - baseline_risk
    release_mask = (
        result["release_gate_predicted"]
        & result["is_downtrend_state"]
        & (result["release_gate_lift"] > float(min_lift))
    )
    if require_rebound_label:
        release_mask = release_mask & result["is_rebound_cluster"]

    result["release_gate_state"] = release_mask.astype(bool)
    result["release_gate_feature_set"] = "" if feature_set is None else str(feature_set)
    result["release_gate_threshold"] = np.nan if threshold is None else float(threshold)
    result["target_semantics_revision"] = np.where(
        result["release_gate_state"],
        "sparse_release_risk_target_lift",
        "",
    )
    result.loc[result["release_gate_state"], "risk_target_position"] = candidate_release[result["release_gate_state"]]
    return result.drop(columns=["_date_key"]).reset_index(drop=True)


def _normalise_joint_target_component(
    targets: pd.DataFrame,
    *,
    component: str,
    date_col: str,
) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()
    required = {
        date_col,
        "teacher_position",
        "rebound_target_position",
        "risk_target_position",
        "is_rebound_cluster",
        "is_downtrend_state",
    }
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"{component} targets missing columns: {missing}")

    result = targets.copy()
    result[date_col] = pd.to_datetime(result[date_col]).dt.normalize()
    if "fold" not in result.columns:
        result["fold"] = ""
    if "target_scope" not in result.columns:
        result["target_scope"] = ""
    result["fold"] = result["fold"].fillna("").astype(str)
    result["target_scope"] = result["target_scope"].fillna("").astype(str)
    for column in ("teacher_position", "rebound_target_position", "risk_target_position"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "risk_target_position_original" not in result.columns:
        result["risk_target_position_original"] = result["risk_target_position"]
    else:
        result["risk_target_position_original"] = pd.to_numeric(
            result["risk_target_position_original"],
            errors="coerce",
        ).fillna(result["risk_target_position"]).clip(0.0, 1.0)
    for column in (
        "is_rebound_cluster",
        "is_downtrend_state",
        "release_gate_state",
        "drawdown_control_state",
    ):
        if column not in result.columns:
            result[column] = False
        result[column] = result[column].fillna(False).astype(bool)
    if "objective_weight" not in result.columns:
        result["objective_weight"] = 1.0
    result["objective_weight"] = pd.to_numeric(result["objective_weight"], errors="coerce").fillna(1.0)
    result["source_component"] = str(component)
    return result


def _candidate_rows_for_fold(
    targets: pd.DataFrame,
    *,
    fold: Mapping[str, object],
    component: str,
    date_col: str,
    project_train_rows: bool,
) -> pd.DataFrame:
    fold_name = str(_fold_value(fold, "name"))
    train_start = pd.Timestamp(_fold_value(fold, "train_start")).normalize()
    train_end = pd.Timestamp(_fold_value(fold, "train_end")).normalize()
    eval_start = pd.Timestamp(_fold_value(fold, "eval_start")).normalize()
    eval_end = pd.Timestamp(_fold_value(fold, "eval_end")).normalize()
    dates = pd.to_datetime(targets[date_col]).dt.normalize()
    native = targets["fold"].astype(str) == fold_name
    train_overlap = (dates >= train_start) & (dates <= train_end)
    eval_overlap = (dates >= eval_start) & (dates <= eval_end)
    mask = (native | (bool(project_train_rows) & train_overlap)) & train_overlap & ~eval_overlap
    result = targets.loc[mask].copy()
    if result.empty:
        return result
    result["fold"] = fold_name
    result["window"] = fold_name
    result["target_scope"] = "fold_local"
    result["train_start"] = str(train_start.date())
    result["train_end"] = str(train_end.date())
    result["eval_start"] = str(eval_start.date())
    result["eval_end"] = str(eval_end.date())
    result["source_component"] = str(component)
    return result


def _collapse_joint_target_rows(group: pd.DataFrame, *, date_col: str) -> dict[str, object]:
    release_mask = group["release_gate_state"].astype(bool) & group["is_rebound_cluster"].astype(bool)
    downtrend_mask = group["is_downtrend_state"].astype(bool)
    teacher = float(group["teacher_position"].max())
    rebound = float(group["rebound_target_position"].max())
    baseline_risk = float(group["risk_target_position_original"].min())
    if release_mask.any():
        risk_target = float(
            np.maximum(
                group.loc[release_mask, "risk_target_position"],
                group.loc[release_mask, "rebound_target_position"],
            ).max()
        )
    elif downtrend_mask.any():
        risk_target = float(group.loc[downtrend_mask, "risk_target_position"].min())
    else:
        risk_target = float(group["risk_target_position"].max())

    first = group.iloc[0]
    source_components = sorted(str(value) for value in group["source_component"].dropna().unique())
    action_values = (
        group["action"].fillna("").astype(str).str.upper()
        if "action" in group.columns
        else pd.Series("", index=group.index)
    )
    if action_values.eq("BUY").any():
        action = "BUY"
    elif action_values.eq("SELL").any():
        action = "SELL"
    else:
        action = "HOLD"
    return {
        date_col: pd.Timestamp(first[date_col]),
        "candidate": str(first.get("candidate", "joint_target_contract")),
        "window": str(first.get("window", "")),
        "fold": str(first["fold"]),
        "target_scope": "fold_local",
        "train_start": str(first.get("train_start", "")),
        "train_end": str(first.get("train_end", "")),
        "eval_start": str(first.get("eval_start", "")),
        "eval_end": str(first.get("eval_end", "")),
        "teacher_position": teacher,
        "rebound_target_position": rebound,
        "risk_target_position_original": baseline_risk,
        "risk_target_position": float(np.clip(risk_target, 0.0, 1.0)),
        "action": action,
        "objective_weight": float(group["objective_weight"].max()),
        "is_rebound_cluster": bool(group["is_rebound_cluster"].any()),
        "is_downtrend_state": bool(group["is_downtrend_state"].any()),
        "release_gate_state": bool(group["release_gate_state"].any()),
        "drawdown_control_state": bool(group["drawdown_control_state"].any()),
        "source_components": ",".join(source_components),
        "source_component_count": int(len(source_components)),
        "source_row_count": int(len(group)),
    }


def build_fold_local_joint_target_contract(
    primary_targets: pd.DataFrame,
    auxiliary_targets: pd.DataFrame,
    *,
    folds: Iterable[Mapping[str, object]],
    date_col: str = "datetime",
    project_auxiliary_train_rows: bool = True,
    bull_reentry_config: BullReentryParticipationTargetConfig = BullReentryParticipationTargetConfig(),
) -> pd.DataFrame:
    """Combine fold-local sparse-release and drawdown-control target rows.

    The contract is intentionally limited to non-evaluation rows in each fold.
    Sparse release states keep their lifted risk target when they collide with a
    downtrend cap, while drawdown-control-only rows keep the lower risk target.
    """

    fold_list = list(folds)
    if not fold_list:
        raise ValueError("folds must contain at least one fold definition")
    primary = _normalise_joint_target_component(
        primary_targets,
        component="primary",
        date_col=date_col,
    )
    auxiliary = _normalise_joint_target_component(
        auxiliary_targets,
        component="auxiliary",
        date_col=date_col,
    )
    if primary.empty and auxiliary.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for fold in fold_list:
        if not primary.empty:
            rows.append(
                _candidate_rows_for_fold(
                    primary,
                    fold=fold,
                    component="primary",
                    date_col=date_col,
                    project_train_rows=False,
                )
            )
        if not auxiliary.empty:
            rows.append(
                _candidate_rows_for_fold(
                    auxiliary,
                    fold=fold,
                    component="auxiliary",
                    date_col=date_col,
                    project_train_rows=bool(project_auxiliary_train_rows),
                )
            )
    combined = pd.concat([frame for frame in rows if not frame.empty], ignore_index=True) if rows else pd.DataFrame()
    if combined.empty:
        return combined

    collapsed = [
        _collapse_joint_target_rows(group, date_col=date_col)
        for _, group in combined.groupby(["fold", date_col], dropna=False, sort=True)
    ]
    result = pd.DataFrame(collapsed).sort_values(["fold", date_col]).reset_index(drop=True)
    if bool(bull_reentry_config.enabled):
        result = apply_bull_reentry_participation_target_gate(result, bull_reentry_config)
    return result


def summarize_state_objective_target_lift_audit(audit: pd.DataFrame) -> dict[str, object]:
    if audit.empty:
        return {
            "decision": "state_objective_target_lift_no_rows",
            "training_allowed": False,
            "next_mechanism": "collect_state_objective_targets",
        }

    rebound_rows = int(pd.to_numeric(audit["rebound_rows"], errors="coerce").fillna(0).sum())
    current_lift_rows = int(
        pd.to_numeric(audit["current_positive_rebound_lift_rows"], errors="coerce").fillna(0).sum()
    )
    release_lift_rows = int(
        pd.to_numeric(audit["candidate_release_positive_lift_rows"], errors="coerce").fillna(0).sum()
    )
    failed = audit[~audit["lift_semantics_gate"].astype(bool)]
    training_allowed = bool(rebound_rows > 0 and current_lift_rows > 0 and failed.empty)
    if training_allowed:
        decision = "state_objective_target_lift_training_ready"
        next_mechanism = "run_fold_local_weighted_bc_or_ppo"
    elif release_lift_rows > 0:
        decision = "state_objective_target_lift_blocked_no_training"
        next_mechanism = "revise_state_objective_target_semantics_before_weighted_bc"
    else:
        decision = "state_objective_target_lift_no_release_candidate"
        next_mechanism = "discover_rebound_release_targets"

    return {
        "decision": decision,
        "training_allowed": training_allowed,
        "lift_semantics_gate": training_allowed,
        "audited_groups": int(len(audit)),
        "blocked_groups": int(len(failed)),
        "rebound_rows": rebound_rows,
        "current_positive_rebound_lift_rows": current_lift_rows,
        "candidate_release_positive_lift_rows": release_lift_rows,
        "rebound_downtrend_rows": int(
            pd.to_numeric(audit["rebound_downtrend_rows"], errors="coerce").fillna(0).sum()
        ),
        "max_candidate_release_lift": float(
            pd.to_numeric(audit["max_candidate_release_lift"], errors="coerce").fillna(0.0).max()
        ),
        "next_mechanism": next_mechanism,
    }


def score_state_objective_bc_readiness(
    profile_label_comparison: pd.DataFrame,
    *,
    target_scope: str,
    config: StateObjectiveBCReadinessConfig = StateObjectiveBCReadinessConfig(),
) -> pd.DataFrame:
    """Rank state-objective BC profiles by label health and target leakage scope."""

    required = {"profile", "samples", "unique_actions", "dominant_action_share", "action_entropy"}
    missing = sorted(required - set(profile_label_comparison.columns))
    if missing:
        raise ValueError(f"profile_label_comparison missing columns: {missing}")
    scored = profile_label_comparison.copy()
    scored["samples"] = pd.to_numeric(scored["samples"], errors="coerce").fillna(0).astype(int)
    scored["unique_actions"] = pd.to_numeric(scored["unique_actions"], errors="coerce").fillna(0).astype(int)
    scored["dominant_action_share"] = pd.to_numeric(
        scored["dominant_action_share"],
        errors="coerce",
    ).fillna(1.0)
    scored["action_entropy"] = pd.to_numeric(scored["action_entropy"], errors="coerce").fillna(0.0)
    scored["sample_gate"] = scored["samples"] >= int(config.min_samples)
    scored["label_distribution_gate"] = (
        (scored["unique_actions"] >= int(config.min_unique_actions))
        & (scored["dominant_action_share"] <= float(config.max_dominant_action_share))
    )
    scored["fold_local_scope_gate"] = str(target_scope) == "fold_local"
    scored["bc_readiness_gate"] = (
        scored["sample_gate"] & scored["label_distribution_gate"] & scored["fold_local_scope_gate"]
    )
    sample_loss = ((float(config.min_samples) - scored["samples"]) / max(float(config.min_samples), 1.0)).clip(
        lower=0.0
    )
    action_loss = (int(config.min_unique_actions) - scored["unique_actions"]).clip(lower=0)
    dominance_loss = (scored["dominant_action_share"] - float(config.max_dominant_action_share)).clip(lower=0.0)
    scope_loss = 0.0 if str(target_scope) == "fold_local" else 100.0
    scored["bc_readiness_loss"] = sample_loss + action_loss + dominance_loss * 10.0 + scope_loss
    scored["target_scope"] = str(target_scope)
    return scored.sort_values(
        ["bc_readiness_gate", "label_distribution_gate", "bc_readiness_loss", "dominant_action_share", "profile"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)


def summarize_state_objective_bc_readiness(scored: pd.DataFrame) -> dict[str, object]:
    if scored.empty:
        return {
            "decision": "state_objective_bc_no_profiles",
            "selected_profile": "",
            "next_mechanism": "build_profile_label_comparison",
        }
    top = scored.iloc[0]
    if bool(top.get("bc_readiness_gate", False)):
        decision = "state_objective_bc_ready_for_training"
        next_mechanism = "run_fold_local_bc_warm_start_then_walkforward_ppo"
    elif not bool(top.get("fold_local_scope_gate", False)):
        decision = "state_objective_bc_scope_blocked_no_promotion"
        next_mechanism = "generate_fold_local_state_objective_targets"
    elif not bool(top.get("label_distribution_gate", False)):
        decision = "state_objective_bc_label_degenerate_no_training"
        next_mechanism = "change_action_surface_or_target_mapping"
    else:
        decision = "state_objective_bc_sample_blocked_no_training"
        next_mechanism = "collect_more_fold_local_target_rows"
    return {
        "decision": decision,
        "selected_profile": str(top["profile"]),
        "selected_target_scope": str(top.get("target_scope", "")),
        "selected_bc_readiness_gate": bool(top.get("bc_readiness_gate", False)),
        "selected_label_distribution_gate": bool(top.get("label_distribution_gate", False)),
        "selected_fold_local_scope_gate": bool(top.get("fold_local_scope_gate", False)),
        "selected_dominant_action_share": float(top.get("dominant_action_share", 1.0)),
        "selected_unique_actions": int(top.get("unique_actions", 0)),
        "selected_samples": int(top.get("samples", 0)),
        "selected_bc_readiness_loss": float(top.get("bc_readiness_loss", 0.0)),
        "next_mechanism": next_mechanism,
    }


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


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    clean_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    clean_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    denominator = float(clean_weights.sum())
    if denominator <= 0.0:
        return float(clean_values.mean()) if len(clean_values) else 0.0
    return float((clean_values * clean_weights).sum() / denominator)


def _target_subset(targets: pd.DataFrame, *, window: str) -> pd.DataFrame:
    subset = targets[targets["window"].astype(str) == str(window)].copy()
    required = {
        "datetime",
        "window",
        "rebound_target_position",
        "risk_target_position",
        "teacher_position",
        "objective_weight",
        "is_rebound_cluster",
        "is_downtrend_state",
    }
    missing = sorted(required - set(subset.columns))
    if missing:
        raise ValueError(f"targets missing columns: {missing}")
    subset["datetime"] = pd.to_datetime(subset["datetime"])
    return subset


def _score_single_candidate_alignment(
    *,
    candidate: str,
    daily_paths: Mapping[str, pd.DataFrame],
    targets: pd.DataFrame,
    config: StateObjectiveTrainingTargetConfig,
) -> dict[str, object]:
    aligned_frames: list[pd.DataFrame] = []
    total_target_rows = int(len(targets))
    window_metrics: list[dict[str, float]] = []
    for window, daily in daily_paths.items():
        target_rows = _target_subset(targets, window=str(window))
        if target_rows.empty:
            continue
        frame = _normalise_candidate_path(daily, window=str(window), candidate=str(candidate))
        if {"strategy_nav", "buyhold_nav"}.issubset(frame.columns):
            metrics = daily_path_metrics(
                frame.rename(columns={"candidate_position": "position"})
                if "position" not in frame.columns
                else frame
            )
            relative_return = _json_safe_float(metrics.get("relative_return"), 0.0)
            max_dd = _json_safe_float(metrics.get("max_dd"), 0.0)
        else:
            relative_return = 0.0
            max_dd = 0.0
        window_metrics.append(
            {
                "relative_gap": max(0.0, float(config.target_relative_return) - relative_return),
                "drawdown_gap": max(0.0, max_dd - float(config.target_max_dd)),
                "window_weight": _window_weight(str(window), config.objective_config()),
            }
        )
        merged = target_rows.merge(
            frame[["datetime", "window", "candidate_position"]],
            on=["datetime", "window"],
            how="left",
        )
        aligned_frames.append(merged)
    if aligned_frames:
        aligned = pd.concat(aligned_frames, ignore_index=True)
    else:
        aligned = targets.copy()
        aligned["candidate_position"] = np.nan
    observed = aligned["candidate_position"].notna()
    missing_target_ratio = 1.0 - (float(observed.sum()) / float(total_target_rows or 1))
    scored_rows = aligned[observed].copy()
    if scored_rows.empty:
        position_mae = rebound_mae = risk_cap_violation = 1.0
    else:
        scored_rows["position_abs_error"] = (
            pd.to_numeric(scored_rows["candidate_position"], errors="coerce")
            - pd.to_numeric(scored_rows["teacher_position"], errors="coerce")
        ).abs()
        scored_rows["rebound_abs_error"] = np.where(
            scored_rows["is_rebound_cluster"].astype(bool),
            (
                pd.to_numeric(scored_rows["candidate_position"], errors="coerce")
                - pd.to_numeric(scored_rows["rebound_target_position"], errors="coerce")
            ).abs(),
            0.0,
        )
        scored_rows["risk_cap_excess"] = np.where(
            scored_rows["is_downtrend_state"].astype(bool),
            (
                pd.to_numeric(scored_rows["candidate_position"], errors="coerce")
                - pd.to_numeric(scored_rows["risk_target_position"], errors="coerce")
            ).clip(lower=0.0),
            0.0,
        )
        position_mae = _weighted_mean(scored_rows["position_abs_error"], scored_rows["objective_weight"])
        rebound_rows = scored_rows[scored_rows["is_rebound_cluster"].astype(bool)]
        risk_rows = scored_rows[scored_rows["is_downtrend_state"].astype(bool)]
        rebound_mae = (
            _weighted_mean(rebound_rows["rebound_abs_error"], rebound_rows["objective_weight"])
            if not rebound_rows.empty
            else 0.0
        )
        risk_cap_violation = (
            _weighted_mean(risk_rows["risk_cap_excess"], risk_rows["objective_weight"])
            if not risk_rows.empty
            else 0.0
        )
    if window_metrics:
        gate_loss = sum(
            row["window_weight"] * (row["relative_gap"] + row["drawdown_gap"] * float(config.max_dd_gap_weight))
            for row in window_metrics
        )
        relative_gap = max(row["relative_gap"] for row in window_metrics)
        drawdown_gap = max(row["drawdown_gap"] for row in window_metrics)
    else:
        gate_loss = relative_gap = drawdown_gap = 0.0
    alignment_loss = (
        position_mae
        + rebound_mae * 2.0
        + risk_cap_violation * 4.0
        + gate_loss / 100.0
        + missing_target_ratio * 10.0
    )
    return {
        "candidate": str(candidate),
        "target_rows": total_target_rows,
        "matched_rows": int(observed.sum()),
        "missing_target_ratio": float(missing_target_ratio),
        "coverage_gate": bool(missing_target_ratio <= 0.05),
        "position_tracking_mae": float(position_mae),
        "rebound_position_mae": float(rebound_mae),
        "risk_cap_violation": float(risk_cap_violation),
        "window_gate_loss": float(gate_loss),
        "max_relative_gap": float(relative_gap),
        "max_drawdown_gap": float(drawdown_gap),
        "window_gate": bool(relative_gap == 0.0 and drawdown_gap == 0.0),
        "alignment_loss": float(alignment_loss),
    }


def score_state_objective_target_alignment(
    candidate_daily_paths: Mapping[str, Mapping[str, pd.DataFrame]],
    targets: pd.DataFrame,
    *,
    config: StateObjectiveTrainingTargetConfig = StateObjectiveTrainingTargetConfig(),
) -> pd.DataFrame:
    """Rank candidates by fit to state-objective training targets and window gates."""

    if not candidate_daily_paths:
        return pd.DataFrame(
            columns=[
                "candidate",
                "target_rows",
                "matched_rows",
                "missing_target_ratio",
                "coverage_gate",
                "position_tracking_mae",
                "rebound_position_mae",
                "risk_cap_violation",
                "window_gate_loss",
                "max_relative_gap",
                "max_drawdown_gap",
                "window_gate",
                "alignment_loss",
            ]
        )
    rows = [
        _score_single_candidate_alignment(
            candidate=str(candidate),
            daily_paths=daily_paths,
            targets=targets,
            config=config,
        )
        for candidate, daily_paths in candidate_daily_paths.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["window_gate", "coverage_gate", "alignment_loss", "candidate"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def summarize_state_objective_target_alignment(scored: pd.DataFrame) -> dict[str, object]:
    if scored.empty:
        return {
            "decision": "no_candidates",
            "selected_candidate": "",
            "next_mechanism": "collect_candidate_daily_paths",
        }
    top = scored.iloc[0]
    if bool(top.get("window_gate", False)) and bool(top.get("coverage_gate", False)):
        decision = "target_alignment_promotion_review"
        next_mechanism = "rerun_full_walkforward_promotion_review"
    else:
        decision = "target_alignment_no_promotion"
        next_mechanism = "train_or_checkpoint_score_state_objective_targets"
    return {
        "decision": decision,
        "selected_candidate": str(top["candidate"]),
        "selected_alignment_loss": float(top["alignment_loss"]),
        "selected_window_gate": bool(top.get("window_gate", False)),
        "selected_coverage_gate": bool(top.get("coverage_gate", False)),
        "selected_missing_target_ratio": float(top.get("missing_target_ratio", 0.0)),
        "next_mechanism": next_mechanism,
        "required_terms": [
            "target_position_alignment_loss",
            "rebound_participation_alignment_loss",
            "downtrend_risk_cap_violation_loss",
            "walkforward_gate_gap_loss",
        ],
    }


def score_state_objective_checkpoint_candidates(
    alignment_scores: pd.DataFrame,
    *,
    config: StateObjectiveCheckpointSelectionConfig = StateObjectiveCheckpointSelectionConfig(),
) -> pd.DataFrame:
    """Rank checkpoints/candidates by state-objective loss instead of validation score alone."""

    if alignment_scores.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "checkpoint_objective_loss",
                "checkpoint_selection_score",
                "window_gate",
                "coverage_gate",
                "alignment_loss",
                "window_gate_loss",
                "missing_target_ratio",
            ]
        )
    required = {
        "candidate",
        "alignment_loss",
        "window_gate_loss",
        "missing_target_ratio",
        "window_gate",
        "coverage_gate",
    }
    missing = sorted(required - set(alignment_scores.columns))
    if missing:
        raise ValueError(f"alignment_scores missing columns: {missing}")
    scored = alignment_scores.copy()
    for column in [
        "alignment_loss",
        "window_gate_loss",
        "missing_target_ratio",
        "rebound_position_mae",
        "risk_cap_violation",
        "score",
    ]:
        if column not in scored.columns:
            scored[column] = 0.0
        scored[column] = pd.to_numeric(scored[column], errors="coerce").fillna(0.0)
    scored["checkpoint_objective_loss"] = (
        scored["alignment_loss"] * float(config.alignment_loss_weight)
        + scored["window_gate_loss"] * float(config.window_gate_loss_weight)
        + scored["missing_target_ratio"] * float(config.missing_target_penalty)
        + scored["rebound_position_mae"] * float(config.rebound_mae_weight)
        + scored["risk_cap_violation"] * float(config.risk_cap_violation_weight)
        - scored["score"] * float(config.legacy_score_reward)
    )
    scored["checkpoint_selection_score"] = -scored["checkpoint_objective_loss"]
    return scored.sort_values(
        ["window_gate", "coverage_gate", "checkpoint_objective_loss", "candidate"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def summarize_state_objective_checkpoint_selection(scored: pd.DataFrame) -> dict[str, object]:
    if scored.empty:
        return {
            "decision": "no_candidates",
            "selected_candidate": "",
            "next_mechanism": "collect_state_objective_alignment_scores",
            "required_terms": [],
        }
    top = scored.iloc[0]
    window_gate = bool(top.get("window_gate", False))
    coverage_gate = bool(top.get("coverage_gate", False))
    if window_gate and coverage_gate:
        decision = "checkpoint_objective_promotion_review"
        next_mechanism = "rerun_walkforward_promotion_review"
    else:
        decision = "checkpoint_objective_no_promotion"
        next_mechanism = "train_state_objective_checkpoint_or_bc"
    return {
        "decision": decision,
        "selected_candidate": str(top["candidate"]),
        "selected_checkpoint_objective_loss": float(top["checkpoint_objective_loss"]),
        "selected_checkpoint_selection_score": float(top["checkpoint_selection_score"])
        if "checkpoint_selection_score" in top
        else -float(top["checkpoint_objective_loss"]),
        "selected_window_gate": window_gate,
        "selected_coverage_gate": coverage_gate,
        "selected_alignment_loss": float(top.get("alignment_loss", 0.0)),
        "selected_window_gate_loss": float(top.get("window_gate_loss", 0.0)),
        "selected_missing_target_ratio": float(top.get("missing_target_ratio", 0.0)),
        "next_mechanism": next_mechanism,
        "required_terms": [
            "checkpoint_objective_alignment_loss",
            "checkpoint_objective_walkforward_gate_loss",
            "checkpoint_objective_rebound_mae",
            "checkpoint_objective_risk_cap_violation",
        ],
    }


__all__ = [
    "BullReentryParticipationTargetConfig",
    "StateObjectiveBCReadinessConfig",
    "StateObjectiveCheckpointSelectionConfig",
    "StateObjectiveDrawdownControlTargetConfig",
    "StateObjectiveReboundClusterConfig",
    "StateObjectiveReboundTargetEnrichmentConfig",
    "StateObjectiveTrainingTargetConfig",
    "ShortVolshockDeRiskTargetConfig",
    "TrendParticipationTargetConfig",
    "apply_bull_reentry_participation_target_gate",
    "apply_state_objective_sparse_release_gate",
    "apply_short_volshock_de_risk_target_gate",
    "apply_trend_participation_target_gate",
    "audit_state_objective_target_lift",
    "audit_state_objective_target_scope",
    "build_drawdown_control_training_targets",
    "build_fold_local_joint_target_contract",
    "build_fold_local_state_objective_training_targets",
    "build_state_objective_training_targets",
    "discover_rebound_clusters_from_daily_path",
    "enrich_state_objective_rebound_targets",
    "filter_state_objective_targets_for_train_window",
    "score_state_objective_bc_readiness",
    "score_state_objective_checkpoint_candidates",
    "score_state_objective_target_alignment",
    "summarize_state_objective_bc_readiness",
    "summarize_state_objective_checkpoint_selection",
    "summarize_state_objective_target_alignment",
    "summarize_state_objective_target_lift_audit",
    "summarize_state_objective_target_scope_audit",
    "summarize_state_objective_training_targets",
    "summarize_short_volshock_de_risk_targets",
]
