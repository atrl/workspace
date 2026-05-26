from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateConditionedObjectiveConfig:
    target_relative_return: float = 30.0
    target_max_dd: float = 15.0
    cluster_relative_floor: float = -30.0
    cluster_participation_floor: float = 0.35
    relative_gap_weight: float = 1.0
    max_dd_gap_weight: float = 10.0
    cluster_relative_gap_weight: float = 1.0
    cluster_participation_gap_weight: float = 100.0
    cluster_max_dd_gap_weight: float = 5.0
    window_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "strict_oos": 0.75,
            "test_proxy": 0.75,
            "wf_2020": 1.0,
            "wf_2022": 1.25,
            "wf_2023_2024": 1.0,
        }
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


def _pct_return(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return 0.0
    return float((values.iloc[-1] / values.iloc[0] - 1.0) * 100.0)


def _max_drawdown(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(values) < 2:
        return 0.0
    peak = values.cummax()
    drawdown = values / peak - 1.0
    return abs(float(drawdown.min()) * 100.0)


def daily_path_metrics(daily: pd.DataFrame) -> dict[str, object]:
    """Compute return, relative return, drawdown, and position from a daily path."""

    frame = daily.copy()
    if "position" not in frame.columns:
        for alias in ("router_position", "target_position", "base_position"):
            if alias in frame.columns:
                frame["position"] = frame[alias]
                break
    required = {"datetime", "strategy_nav", "buyhold_nav", "position"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"daily path missing columns: {missing}")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime")
    strategy_return = _pct_return(frame["strategy_nav"])
    buyhold_return = _pct_return(frame["buyhold_nav"])
    return {
        "rows": int(len(frame)),
        "start": str(frame["datetime"].iloc[0].date()) if len(frame) else "",
        "end": str(frame["datetime"].iloc[-1].date()) if len(frame) else "",
        "return": strategy_return,
        "buy_and_hold_return": buyhold_return,
        "relative_return": strategy_return - buyhold_return,
        "max_dd": _max_drawdown(frame["strategy_nav"]),
        "avg_pos": float(pd.to_numeric(frame["position"], errors="coerce").mean()),
    }


def max_drawdown_interval_metrics(
    daily: pd.DataFrame,
    *,
    nav_col: str = "strategy_nav",
) -> dict[str, object]:
    """Return the peak-to-trough interval behind a daily path's max drawdown."""

    frame = daily.copy()
    if "position" not in frame.columns:
        for alias in ("router_position", "target_position", "base_position"):
            if alias in frame.columns:
                frame["position"] = frame[alias]
                break
    required = {"datetime", nav_col, "position"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"daily path missing columns: {missing}")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime").reset_index(drop=True)
    nav = pd.to_numeric(frame[nav_col], errors="coerce").ffill()
    if len(nav.dropna()) < 2:
        return {
            "peak_date": "",
            "trough_date": "",
            "max_dd": 0.0,
            "interval_return": 0.0,
            "interval_days": int(len(frame)),
            "interval_avg_pos": float(pd.to_numeric(frame["position"], errors="coerce").fillna(0.0).mean())
            if len(frame)
            else 0.0,
            "interval_min_pos": float(pd.to_numeric(frame["position"], errors="coerce").fillna(0.0).min())
            if len(frame)
            else 0.0,
            "interval_max_pos": float(pd.to_numeric(frame["position"], errors="coerce").fillna(0.0).max())
            if len(frame)
            else 0.0,
            "interval_trades": 0,
        }

    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    trough_idx = int(drawdown.idxmin())
    peak_idx = int(nav.loc[:trough_idx].idxmax())
    interval = frame.loc[peak_idx:trough_idx].copy()
    positions = pd.to_numeric(interval["position"], errors="coerce").fillna(0.0)
    start_nav = float(nav.iloc[peak_idx])
    trough_nav = float(nav.iloc[trough_idx])
    interval_return = (trough_nav / start_nav - 1.0) * 100.0 if start_nav else 0.0
    if "action" in interval.columns:
        actions = interval["action"].astype(str).str.upper()
        interval_trades = int(actions.isin(["BUY", "SELL"]).sum())
    else:
        interval_trades = 0
    return {
        "peak_date": str(pd.Timestamp(frame.loc[peak_idx, "datetime"]).date()),
        "trough_date": str(pd.Timestamp(frame.loc[trough_idx, "datetime"]).date()),
        "max_dd": abs(float(drawdown.iloc[trough_idx]) * 100.0),
        "interval_return": float(interval_return),
        "interval_days": int(len(interval)),
        "interval_avg_pos": float(positions.mean()) if len(positions) else 0.0,
        "interval_min_pos": float(positions.min()) if len(positions) else 0.0,
        "interval_max_pos": float(positions.max()) if len(positions) else 0.0,
        "interval_trades": interval_trades,
    }


def slice_daily_path_metrics(daily: pd.DataFrame, *, start: str, end: str) -> dict[str, object]:
    frame = daily.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    sub = frame[(frame["datetime"] >= pd.Timestamp(start)) & (frame["datetime"] <= pd.Timestamp(end))]
    if sub.empty:
        return {
            "rows": 0,
            "start": start,
            "end": end,
            "return": 0.0,
            "buy_and_hold_return": 0.0,
            "relative_return": float("-inf"),
            "max_dd": float("inf"),
            "avg_pos": 0.0,
        }
    return daily_path_metrics(sub)


def _cluster_losses(
    clusters: Mapping[str, Mapping[str, object]],
    config: StateConditionedObjectiveConfig,
) -> tuple[float, float, float, bool, str]:
    relative_loss = 0.0
    participation_loss = 0.0
    max_dd_loss = 0.0
    pass_count = 0
    worst_cluster = ""
    worst_loss = -np.inf
    for cluster, metrics in clusters.items():
        relative_return = _json_safe_float(metrics.get("relative_return"), float("-inf"))
        max_dd = _json_safe_float(metrics.get("max_dd"), float("inf"))
        avg_pos = _json_safe_float(metrics.get("avg_pos"), 0.0)
        rel_gap = max(0.0, float(config.cluster_relative_floor) - relative_return)
        pos_gap = max(0.0, float(config.cluster_participation_floor) - avg_pos)
        dd_gap = max(0.0, max_dd - float(config.target_max_dd))
        cluster_loss = (
            rel_gap * float(config.cluster_relative_gap_weight)
            + pos_gap * float(config.cluster_participation_gap_weight)
            + dd_gap * float(config.cluster_max_dd_gap_weight)
        )
        relative_loss += rel_gap * float(config.cluster_relative_gap_weight)
        participation_loss += pos_gap * float(config.cluster_participation_gap_weight)
        max_dd_loss += dd_gap * float(config.cluster_max_dd_gap_weight)
        if cluster_loss == 0.0:
            pass_count += 1
        if cluster_loss > worst_loss:
            worst_loss = cluster_loss
            worst_cluster = str(cluster)
    cluster_gate = bool(clusters) and pass_count == len(clusters)
    return relative_loss, participation_loss, max_dd_loss, cluster_gate, worst_cluster


def score_state_conditioned_objective(
    metrics_by_candidate: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    cluster_metrics_by_candidate: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
    config: StateConditionedObjectiveConfig = StateConditionedObjectiveConfig(),
) -> pd.DataFrame:
    """Rank candidates for a state-conditioned multi-objective training target."""

    rows: list[dict[str, object]] = []
    cluster_metrics_by_candidate = cluster_metrics_by_candidate or {}
    for candidate, windows in metrics_by_candidate.items():
        relative_loss = 0.0
        drawdown_loss = 0.0
        pass_count = 0
        worst_window = ""
        worst_window_loss = -np.inf
        min_relative = float("inf")
        max_dd_seen = 0.0
        mean_positions: list[float] = []
        for window, metrics in windows.items():
            relative_return = _json_safe_float(metrics.get("relative_return"), float("-inf"))
            max_dd = _json_safe_float(metrics.get("max_dd"), float("inf"))
            avg_pos = _json_safe_float(metrics.get("avg_pos"), 0.0)
            rel_gap = max(0.0, float(config.target_relative_return) - relative_return)
            dd_gap = max(0.0, max_dd - float(config.target_max_dd))
            weight = _window_weight(str(window), config)
            window_loss = weight * (
                rel_gap * float(config.relative_gap_weight)
                + dd_gap * float(config.max_dd_gap_weight)
            )
            relative_loss += weight * rel_gap * float(config.relative_gap_weight)
            drawdown_loss += weight * dd_gap * float(config.max_dd_gap_weight)
            if rel_gap == 0.0 and dd_gap == 0.0:
                pass_count += 1
            if window_loss > worst_window_loss:
                worst_window_loss = window_loss
                worst_window = str(window)
            min_relative = min(min_relative, relative_return)
            max_dd_seen = max(max_dd_seen, max_dd)
            mean_positions.append(avg_pos)
        clusters = cluster_metrics_by_candidate.get(candidate, {})
        cluster_relative_loss, participation_loss, cluster_dd_loss, cluster_gate, worst_cluster = _cluster_losses(
            clusters,
            config,
        )
        total_loss = relative_loss + drawdown_loss + cluster_relative_loss + participation_loss + cluster_dd_loss
        window_gate = bool(windows) and pass_count == len(windows)
        objective_pass = window_gate and cluster_gate
        rows.append(
            {
                "candidate": str(candidate),
                "window_count": int(len(windows)),
                "window_pass_count": int(pass_count),
                "window_gate": bool(window_gate),
                "cluster_count": int(len(clusters)),
                "cluster_gate": bool(cluster_gate),
                "objective_pass": bool(objective_pass),
                "objective_loss": float(total_loss),
                "relative_loss": float(relative_loss + cluster_relative_loss),
                "drawdown_loss": float(drawdown_loss + cluster_dd_loss),
                "participation_loss": float(participation_loss),
                "min_relative_return": float(min_relative if np.isfinite(min_relative) else 0.0),
                "max_max_dd": float(max_dd_seen),
                "mean_avg_pos": float(np.mean(mean_positions)) if mean_positions else 0.0,
                "worst_window": worst_window,
                "worst_cluster": worst_cluster,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "candidate",
                "window_count",
                "window_pass_count",
                "window_gate",
                "cluster_count",
                "cluster_gate",
                "objective_pass",
                "objective_loss",
                "relative_loss",
                "drawdown_loss",
                "participation_loss",
                "min_relative_return",
                "max_max_dd",
                "mean_avg_pos",
                "worst_window",
                "worst_cluster",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["objective_pass", "objective_loss", "candidate"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def summarize_state_conditioned_objective(scored: pd.DataFrame) -> dict[str, object]:
    if scored.empty:
        return {
            "decision": "no_candidates",
            "selected_candidate": "",
            "next_mechanism": "collect_state_objective_inputs",
            "required_terms": [],
        }
    top = scored.iloc[0]
    if bool(top["objective_pass"]):
        decision = "candidate_promotion_review"
        next_mechanism = "promote_or_replay_state_conditioned_candidate"
    else:
        decision = "no_objective_pass_candidate"
        next_mechanism = "train_state_conditioned_multi_objective"
    required_terms = [
        "walkforward_relative_return_gap_penalty",
        "walkforward_drawdown_gap_penalty",
    ]
    if float(top.get("participation_loss", 0.0)) > 0.0 or not bool(top.get("cluster_gate", False)):
        required_terms.append("high_vol_rebound_participation_floor")
    if float(top.get("drawdown_loss", 0.0)) > 0.0 or not bool(top.get("window_gate", False)):
        required_terms.append("downtrend_drawdown_state_cap")
    return {
        "decision": decision,
        "selected_candidate": str(top["candidate"]),
        "selected_objective_loss": float(top["objective_loss"]),
        "selected_window_gate": bool(top["window_gate"]),
        "selected_cluster_gate": bool(top["cluster_gate"]),
        "next_mechanism": next_mechanism,
        "required_terms": required_terms,
    }


def render_state_conditioned_objective_review(
    scored: pd.DataFrame,
    summary: Mapping[str, object],
) -> str:
    lines = [
        "# State-Conditioned Objective Screen",
        "",
        "## Decision",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- selected_candidate: `{summary.get('selected_candidate', '')}`",
        f"- next_mechanism: `{summary.get('next_mechanism', '')}`",
        "",
        "## Ranking",
        "",
        "| candidate | pass | loss | wf pass | cluster pass | rel loss | dd loss | part loss | worst wf | worst cluster |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in scored.itertuples(index=False):
        lines.append(
            "| "
            f"{row.candidate} | {bool(row.objective_pass)} | {float(row.objective_loss):.2f} | "
            f"{int(row.window_pass_count)}/{int(row.window_count)} | "
            f"{bool(row.cluster_gate)} | {float(row.relative_loss):.2f} | "
            f"{float(row.drawdown_loss):.2f} | {float(row.participation_loss):.2f} | "
            f"{row.worst_window} | {row.worst_cluster} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- A passing row is a promotion-review input, not an automatic runtime bind.",
            "- If no row passes, treat the selected row as a diagnostic target and keep no runtime promotion.",
            "- Required terms should be implemented in training-time reward/checkpoint selection before another path-level threshold search.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "StateConditionedObjectiveConfig",
    "daily_path_metrics",
    "max_drawdown_interval_metrics",
    "render_state_conditioned_objective_review",
    "score_state_conditioned_objective",
    "slice_daily_path_metrics",
    "summarize_state_conditioned_objective",
]
