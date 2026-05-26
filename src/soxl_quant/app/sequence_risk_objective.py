from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceRiskFeatureSpec:
    feature: str
    direction: str
    quantile: float


@dataclass(frozen=True)
class SequenceRiskPathConfig:
    target_max_dd_pct: float = 15.0
    sequence_position_reference: float = 0.10
    exposure_weight: float = 0.35
    drawdown_weight: float = 0.40
    breach_capture_weight: float = 0.25


DEFAULT_TREND_BREAK_SPECS: tuple[SequenceRiskFeatureSpec, ...] = (
    SequenceRiskFeatureSpec("anchor_slope_180", "low", 0.25),
    SequenceRiskFeatureSpec("anchor_gap_180", "low", 0.25),
    SequenceRiskFeatureSpec("trend_gap_63", "high", 0.75),
)


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"])
        out = out.dropna(subset=["datetime"])
        out = out.sort_values("datetime", kind="stable")
        out.index = pd.DatetimeIndex(out["datetime"])
        out.index.name = None
    else:
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
        out = out[~out.index.isna()].sort_index()
        out["datetime"] = out.index
    return out


def fold_local_sequence_risk_thresholds(
    feature_frame: pd.DataFrame,
    train_dates: Sequence[Any],
    *,
    specs: Sequence[SequenceRiskFeatureSpec] = DEFAULT_TREND_BREAK_SPECS,
) -> dict[str, dict[str, float | str]]:
    """Compute sequence-risk feature thresholds from the supplied train dates only."""

    frame = _normalise_frame(feature_frame)
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(train_dates))).dropna())
    train = frame.reindex(dates).dropna(how="all")
    thresholds: dict[str, dict[str, float | str]] = {}
    for spec in specs:
        values = pd.to_numeric(train.get(spec.feature, pd.Series(dtype=float)), errors="coerce").dropna()
        threshold = float(values.quantile(float(spec.quantile))) if not values.empty else float("nan")
        thresholds[spec.feature] = {
            "direction": str(spec.direction),
            "quantile": float(spec.quantile),
            "threshold": threshold,
        }
    return thresholds


def apply_sequence_risk_mask(
    feature_frame: pd.DataFrame,
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    mode: str = "any",
) -> tuple[pd.Series, pd.Series]:
    """Apply threshold specs and return a boolean sequence-risk mask plus hit counts."""

    frame = _normalise_frame(feature_frame)
    if not thresholds or mode == "none":
        return pd.Series(False, index=frame.index), pd.Series(0, index=frame.index, dtype=int)

    hits: list[pd.Series] = []
    for feature, spec in thresholds.items():
        values = pd.to_numeric(frame.get(str(feature), pd.Series(index=frame.index, dtype=float)), errors="coerce")
        threshold = float(spec.get("threshold", float("nan")))
        direction = str(spec.get("direction", "low"))
        if not np.isfinite(threshold):
            hit = pd.Series(False, index=frame.index)
        elif direction == "high":
            hit = values >= threshold
        else:
            hit = values <= threshold
        hits.append(hit.fillna(False).astype(bool))

    if not hits:
        return pd.Series(False, index=frame.index), pd.Series(0, index=frame.index, dtype=int)
    hit_frame = pd.concat(hits, axis=1)
    hit_count = hit_frame.sum(axis=1).astype(int)
    if mode == "two_of_three":
        mask = hit_count >= 2
    elif mode == "all":
        mask = hit_count >= len(thresholds)
    else:
        mask = hit_count >= 1
    return mask.astype(bool), hit_count


def _normalise_drawdown_pct(frame: pd.DataFrame) -> pd.Series:
    if "drawdown" in frame.columns:
        drawdown = pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.0).abs()
        return drawdown * 100.0 if float(drawdown.max()) <= 1.5 else drawdown
    if "strategy_nav" in frame.columns:
        nav = pd.to_numeric(frame["strategy_nav"], errors="coerce").dropna()
    elif "nav" in frame.columns:
        nav = pd.to_numeric(frame["nav"], errors="coerce").dropna()
    elif "account" in frame.columns:
        account = pd.to_numeric(frame["account"], errors="coerce").dropna()
        nav = account / float(account.iloc[0]) if len(account) and float(account.iloc[0]) else account
    else:
        raise ValueError("daily path missing drawdown, strategy_nav, nav, or account")
    drawdown = (1.0 - nav / nav.cummax().replace(0.0, np.nan)).fillna(0.0).abs()
    return drawdown.reindex(frame.index).fillna(0.0) * 100.0


def score_sequence_risk_path(
    daily: pd.DataFrame,
    *,
    thresholds: Mapping[str, Mapping[str, Any]],
    mode: str = "any",
    config: SequenceRiskPathConfig = SequenceRiskPathConfig(),
) -> dict[str, Any]:
    """Score path-level exposure and drawdown while a fold-local sequence-risk state is active."""

    frame = _normalise_frame(daily)
    if "position" not in frame.columns:
        raise ValueError("daily path missing `position` column")
    position = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
    drawdown_pct = _normalise_drawdown_pct(frame)
    sequence_mask, hit_count = apply_sequence_risk_mask(frame, thresholds, mode=mode)
    breach_mask = drawdown_pct > float(config.target_max_dd_pct)

    sequence_rows = int(sequence_mask.sum())
    breach_rows = int(breach_mask.sum())
    sequence_breach_rows = int((sequence_mask & breach_mask).sum())
    sequence_avg_pos = float(position[sequence_mask].mean()) if sequence_rows else 0.0
    non_sequence_avg_pos = float(position[~sequence_mask].mean()) if sequence_rows < len(position) else 0.0
    sequence_max_dd_pct = float(drawdown_pct[sequence_mask].max()) if sequence_rows else 0.0
    sequence_breach_capture = sequence_breach_rows / breach_rows if breach_rows else 0.0

    exposure_loss = max(0.0, sequence_avg_pos - float(config.sequence_position_reference)) / max(
        float(config.sequence_position_reference),
        1e-9,
    )
    drawdown_loss = max(0.0, sequence_max_dd_pct - float(config.target_max_dd_pct)) / max(
        float(config.target_max_dd_pct),
        1e-9,
    )
    breach_capture_loss = sequence_breach_capture * exposure_loss
    path_penalty = (
        float(config.exposure_weight) * exposure_loss
        + float(config.drawdown_weight) * drawdown_loss
        + float(config.breach_capture_weight) * breach_capture_loss
    )

    return {
        "rows": int(len(frame)),
        "sequence_rows": sequence_rows,
        "sequence_share": float(sequence_rows / max(len(frame), 1)),
        "sequence_avg_pos": sequence_avg_pos,
        "non_sequence_avg_pos": non_sequence_avg_pos,
        "sequence_position_lift": sequence_avg_pos - non_sequence_avg_pos,
        "sequence_max_dd_pct": sequence_max_dd_pct,
        "drawdown_breach_rows": breach_rows,
        "sequence_breach_rows": sequence_breach_rows,
        "sequence_breach_capture": float(sequence_breach_capture),
        "sequence_hit_count_mean": float(hit_count.mean()) if len(hit_count) else 0.0,
        "sequence_exposure_loss": float(exposure_loss),
        "sequence_drawdown_loss": float(drawdown_loss),
        "sequence_breach_capture_loss": float(breach_capture_loss),
        "sequence_path_penalty": float(path_penalty),
    }
