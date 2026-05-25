from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExternalReference:
    label: str
    url: str
    note: str


def _date_str(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _coerce_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "strategy_nav", "buyhold_nav", "position"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily frame missing columns: {missing}")
    out = daily[["datetime", "strategy_nav", "buyhold_nav", "position"]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    for col in ("strategy_nav", "buyhold_nav", "position"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["datetime", "strategy_nav", "buyhold_nav", "position"])
    out = out.sort_values("datetime").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("daily frame needs at least two rows")
    return out


def _pct_return(values: pd.Series) -> float:
    first = float(values.iloc[0])
    last = float(values.iloc[-1])
    if first == 0.0:
        return 0.0
    return (last / first - 1.0) * 100.0


def _max_drawdown(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdown = arr / peak - 1.0
    return abs(float(np.min(drawdown)) * 100.0)


def find_poor_windows(
    daily: pd.DataFrame,
    *,
    window_size: int = 63,
    top_n: int = 12,
    min_excess_gap: float = 0.0,
    target_excess: float = 0.0,
    target_max_dd: float = 0.0,
) -> pd.DataFrame:
    """Find rolling windows where strategy excess is poor versus buy-and-hold.

    `window_size` is expressed in rows/trading days. The default score preserves
    the legacy behavior: negative excess plus local drawdown. When
    `target_excess` or `target_max_dd` are provided, the score uses gaps to
    those gates so positive-but-insufficient excess windows are still surfaced.
    """

    frame = _coerce_daily_frame(daily)
    window_size = max(int(window_size), 2)
    rows: list[dict[str, float | str | int]] = []
    for start_idx in range(0, max(len(frame) - window_size + 1, 0)):
        sub = frame.iloc[start_idx : start_idx + window_size]
        strategy_return = _pct_return(sub["strategy_nav"])
        buyhold_return = _pct_return(sub["buyhold_nav"])
        excess_return = strategy_return - buyhold_return
        max_dd = _max_drawdown(sub["strategy_nav"])
        avg_pos = float(sub["position"].mean())
        if float(target_excess) > 0.0:
            excess_gap = max(0.0, float(target_excess) - excess_return - float(min_excess_gap))
        else:
            excess_gap = max(0.0, -excess_return - float(min_excess_gap))
        if float(target_max_dd) > 0.0:
            max_dd_gap = max(0.0, max_dd - float(target_max_dd))
        else:
            max_dd_gap = max_dd
        bad_score = excess_gap + max_dd_gap
        if bad_score <= 0.0:
            continue
        rows.append(
            {
                "start": _date_str(sub["datetime"].iloc[0]),
                "end": _date_str(sub["datetime"].iloc[-1]),
                "rows": int(len(sub)),
                "strategy_return": float(strategy_return),
                "buyhold_return": float(buyhold_return),
                "excess_return": float(excess_return),
                "max_dd": float(max_dd),
                "avg_pos": avg_pos,
                "excess_gap": float(excess_gap),
                "max_dd_gap": float(max_dd_gap),
                "bad_score": float(bad_score),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "rows",
                "strategy_return",
                "buyhold_return",
                "excess_return",
                "max_dd",
                "avg_pos",
                "excess_gap",
                "max_dd_gap",
                "bad_score",
            ]
        )
    return out.sort_values(["bad_score", "start"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def _row_value(row: Mapping[str, object] | pd.Series, key: str, default: float = 0.0) -> float:
    if isinstance(row, pd.Series):
        value = row.get(key, default)
    else:
        value = row.get(key, default)
    return float(value) if pd.notna(value) else float(default)


def classify_poor_window(
    row: Mapping[str, object] | pd.Series,
    *,
    target_excess: float = 30.0,
    target_max_dd: float = 15.0,
    low_position: float = 0.25,
    high_position: float = 0.55,
    participation_position: float = 0.65,
    strong_buyhold_return: float = 30.0,
) -> str:
    """Classify a poor OOS window into an optimization failure mode."""

    buyhold_return = _row_value(row, "buyhold_return")
    excess_return = _row_value(row, "excess_return")
    max_dd = _row_value(row, "max_dd")
    avg_pos = _row_value(row, "avg_pos")
    missed_target = excess_return < float(target_excess)
    low_exposure = avg_pos <= float(low_position)
    high_exposure = avg_pos >= float(high_position)
    drawdown_failed = max_dd > float(target_max_dd)
    strong_benchmark = buyhold_return >= float(strong_buyhold_return)

    if missed_target and strong_benchmark and low_exposure and not drawdown_failed:
        return "underparticipation"
    if drawdown_failed and high_exposure:
        return "drawdown_control"
    if (
        missed_target
        and strong_benchmark
        and avg_pos <= float(participation_position)
        and max_dd <= float(target_max_dd) * 1.25
    ):
        return "underparticipation"
    if missed_target and low_exposure:
        return "underparticipation"
    if drawdown_failed:
        return "drawdown_control"
    if missed_target:
        return "selection_bias"
    return "mixed"


def _optimization_path(failure_type: str) -> str:
    return {
        "underparticipation": "reentry_participation_or_behavior_clone",
        "drawdown_control": "drawdown_cap_or_deleverage",
        "selection_bias": "checkpoint_or_model_selection",
        "mixed": "regime_router_or_joint_objective",
    }.get(failure_type, "regime_router_or_joint_objective")


def _gate_optimization_path(failure_type: str) -> str:
    return {
        "pass": "none",
        "near_miss_drawdown": "cap_calibration_or_execution_friction",
        "near_miss_relative": "checkpoint_or_model_selection",
        "underparticipation": "participation_or_reentry_objective",
        "trend_capture": "trend_participation_floor_or_bull_guard",
        "drawdown_control": "drawdown_controller_or_sequence_risk_feature",
        "joint_tradeoff": "walk_forward_weighted_multi_objective_selection",
        "selection_bias": "checkpoint_or_model_selection",
        "generalization_gap": "walk_forward_weighted_checkpoint_selection",
    }.get(failure_type, "walk_forward_weighted_multi_objective_selection")


def _empty_gate_gap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "profile",
            "window",
            "return",
            "buy_and_hold_return",
            "relative_return",
            "sharpe",
            "max_dd",
            "avg_pos",
            "relative_gap",
            "max_dd_gap",
            "relative_pass",
            "maxdd_pass",
            "gate_pass",
            "failure_type",
            "severity_score",
            "primary_optimization_path",
        ]
    )


def _empty_gate_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "profile",
            "window_count",
            "gate_pass_count",
            "failure_count",
            "dominant_failure_type",
            "worst_window",
            "max_severity_score",
            "recommended_next_mechanism",
        ]
    )


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _normalise_gate_metric(metric: Mapping[str, object]) -> dict[str, float]:
    aliases = {
        "best_oos_relative_return": "relative_return",
        "best_oos_max_dd": "max_dd",
        "selected_relative_return": "relative_return",
        "selected_max_dd": "max_dd",
    }
    fields = [
        "return",
        "buy_and_hold_return",
        "relative_return",
        "sharpe",
        "max_dd",
        "avg_pos",
        "stress_avg_pos",
        "turnover",
        "trades",
        "changed_vs_base_days",
    ]
    out: dict[str, float] = {}
    for field in fields:
        value = _optional_float(metric.get(field))
        if value is not None:
            out[field] = value
    for source, target in aliases.items():
        if target not in out:
            value = _optional_float(metric.get(source))
            if value is not None:
                out[target] = value
    if "relative_return" not in out and "return" in out and "buy_and_hold_return" in out:
        out["relative_return"] = out["return"] - out["buy_and_hold_return"]
    if "relative_return" not in out or "max_dd" not in out:
        return {}
    return out


def _add_selection_metric(
    out: dict[str, dict[str, float]],
    label: str,
    metric: object,
) -> None:
    if not isinstance(metric, Mapping):
        return
    metric_label = str(metric.get("label") or label)
    normalised = _normalise_gate_metric(metric)
    if normalised:
        out[metric_label] = normalised


def normalise_gate_metrics_from_selection(selection: Mapping[str, object]) -> dict[str, dict[str, float]]:
    """Extract gate-relevant window metrics from common runtime selection schemas."""

    out: dict[str, dict[str, float]] = {}
    metrics = selection.get("metrics")
    if isinstance(metrics, Mapping):
        for label, metric in metrics.items():
            if label == "walkforward" and isinstance(metric, Sequence) and not isinstance(metric, (str, bytes)):
                for idx, item in enumerate(metric):
                    if isinstance(item, Mapping):
                        _add_selection_metric(out, str(item.get("label") or f"walkforward_{idx}"), item)
                continue
            _add_selection_metric(out, str(label), metric)

    selected_metrics = selection.get("selected_metrics")
    if isinstance(selected_metrics, Mapping):
        for label, metric in selected_metrics.items():
            _add_selection_metric(out, str(label), metric)
    elif isinstance(selected_metrics, Sequence) and not isinstance(selected_metrics, (str, bytes)):
        for idx, metric in enumerate(selected_metrics):
            if isinstance(metric, Mapping):
                _add_selection_metric(out, str(metric.get("label") or f"selected_{idx}"), metric)

    for label in ("valid", "test_proxy", "strict_oos"):
        _add_selection_metric(out, label, selection.get(label))

    slices = selection.get("slices")
    if isinstance(slices, Mapping):
        for label, metric in slices.items():
            _add_selection_metric(out, str(label), metric)

    walk_forward = selection.get("walk_forward")
    if isinstance(walk_forward, Mapping):
        for label, metric in walk_forward.items():
            if isinstance(metric, Mapping) and isinstance(metric.get("test_proxy"), Mapping):
                _add_selection_metric(out, str(label), metric.get("test_proxy"))
            else:
                _add_selection_metric(out, str(label), metric)

    window_best_rows = selection.get("window_best_rows")
    if isinstance(window_best_rows, Sequence) and not isinstance(window_best_rows, (str, bytes)):
        for idx, metric in enumerate(window_best_rows):
            if isinstance(metric, Mapping):
                _add_selection_metric(out, str(metric.get("window") or f"window_best_{idx}"), metric)

    _add_selection_metric(out, "main", selection.get("main"))
    return out


def _classify_gate_gap(
    *,
    relative_pass: bool,
    maxdd_pass: bool,
    relative_gap: float,
    max_dd_gap: float,
    buy_and_hold_return: float,
    avg_pos: float,
    near_dd_tolerance: float,
    near_relative_tolerance: float,
    low_position: float,
    participation_position: float,
    strong_buyhold_return: float,
) -> str:
    if relative_pass and maxdd_pass:
        return "pass"
    if relative_pass and 0.0 < max_dd_gap <= near_dd_tolerance:
        return "near_miss_drawdown"
    if maxdd_pass and 0.0 < relative_gap <= near_relative_tolerance:
        return "near_miss_relative"
    if not relative_pass and not maxdd_pass:
        return "joint_tradeoff"
    if not maxdd_pass:
        return "drawdown_control"
    if buy_and_hold_return >= strong_buyhold_return and avg_pos <= low_position:
        return "underparticipation"
    if buy_and_hold_return >= strong_buyhold_return and avg_pos <= participation_position:
        return "trend_capture"
    return "selection_bias"


def attribute_gate_gaps(
    metrics_by_profile: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    target_relative_return: float = 30.0,
    target_max_dd: float = 15.0,
    near_dd_tolerance: float = 1.0,
    near_relative_tolerance: float = 5.0,
    low_position: float = 0.25,
    participation_position: float = 0.55,
    strong_buyhold_return: float = 30.0,
) -> pd.DataFrame:
    """Classify per-window promotion-gate misses across profiles.

    This is a gate-level complement to rolling poor-window diagnostics: it uses
    saved selection/backtest metrics to decide whether the next loop should
    target participation, drawdown control, or selection discipline.
    """

    rows: list[dict[str, object]] = []
    for profile, windows in metrics_by_profile.items():
        for window, metric in windows.items():
            normalised = _normalise_gate_metric(metric)
            if not normalised:
                continue
            relative_return = float(normalised["relative_return"])
            max_dd = float(normalised["max_dd"])
            buyhold = float(normalised.get("buy_and_hold_return", 0.0))
            avg_pos = float(normalised.get("avg_pos", 0.0))
            relative_gap = max(0.0, float(target_relative_return) - relative_return)
            max_dd_gap = max(0.0, max_dd - float(target_max_dd))
            relative_pass = relative_gap == 0.0
            maxdd_pass = max_dd_gap == 0.0
            failure_type = _classify_gate_gap(
                relative_pass=relative_pass,
                maxdd_pass=maxdd_pass,
                relative_gap=relative_gap,
                max_dd_gap=max_dd_gap,
                buy_and_hold_return=buyhold,
                avg_pos=avg_pos,
                near_dd_tolerance=near_dd_tolerance,
                near_relative_tolerance=near_relative_tolerance,
                low_position=low_position,
                participation_position=participation_position,
                strong_buyhold_return=strong_buyhold_return,
            )
            rows.append(
                {
                    "profile": str(profile),
                    "window": str(window),
                    "return": float(normalised.get("return", np.nan)),
                    "buy_and_hold_return": buyhold,
                    "relative_return": relative_return,
                    "sharpe": float(normalised.get("sharpe", np.nan)),
                    "max_dd": max_dd,
                    "avg_pos": avg_pos,
                    "relative_gap": float(relative_gap),
                    "max_dd_gap": float(max_dd_gap),
                    "relative_pass": bool(relative_pass),
                    "maxdd_pass": bool(maxdd_pass),
                    "gate_pass": bool(relative_pass and maxdd_pass),
                    "failure_type": failure_type,
                    "severity_score": float(relative_gap + max_dd_gap * 10.0),
                    "primary_optimization_path": _gate_optimization_path(failure_type),
                }
            )
    if not rows:
        return _empty_gate_gap_frame()
    return pd.DataFrame(rows).sort_values(
        ["severity_score", "profile", "window"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def summarize_gate_gap_attribution(gate_gaps: pd.DataFrame) -> pd.DataFrame:
    """Summarize gate-gap attribution rows into one recommendation per profile."""

    if gate_gaps.empty:
        return _empty_gate_summary_frame()
    rows: list[dict[str, object]] = []
    for profile, sub in gate_gaps.groupby("profile", sort=True):
        failures = sub[sub["gate_pass"] == False]  # noqa: E712
        if failures.empty:
            dominant = "pass"
            worst = sub.sort_values("severity_score", ascending=False).iloc[0]
        else:
            severity_by_type = failures.groupby("failure_type")["severity_score"].sum().sort_values(ascending=False)
            dominant = str(severity_by_type.index[0])
            worst = failures.sort_values("severity_score", ascending=False).iloc[0]
        rows.append(
            {
                "profile": str(profile),
                "window_count": int(len(sub)),
                "gate_pass_count": int(sub["gate_pass"].sum()),
                "failure_count": int(len(failures)),
                "dominant_failure_type": dominant,
                "worst_window": str(worst["window"]),
                "max_severity_score": float(worst["severity_score"]),
                "recommended_next_mechanism": _gate_optimization_path(dominant),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_severity_score", "profile"],
        ascending=[False, True],
    ).reset_index(drop=True)


def render_gate_gap_review(
    gate_gaps: pd.DataFrame,
    profile_summary: pd.DataFrame | None = None,
    *,
    external_references: Sequence[Mapping[str, str]] | Sequence[ExternalReference] = (),
) -> str:
    """Render a human-readable gate-gap attribution review."""

    summary = profile_summary if profile_summary is not None else summarize_gate_gap_attribution(gate_gaps)
    lines = ["# Gate-Gap Failure Attribution", ""]
    if external_references:
        lines.extend(["## External References", "", *_reference_rows(external_references), ""])
    lines.extend(
        [
            "## Profile Summary",
            "",
            "| profile | windows | failures | dominant type | worst window | severity | next mechanism |",
            "| --- | ---: | ---: | --- | --- | ---: | --- |",
        ]
    )
    if summary.empty:
        lines.append("| none | 0 | 0 | none | none | 0.00 | none |")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                "| "
                f"{row.profile} | {int(row.window_count)} | {int(row.failure_count)} | "
                f"{row.dominant_failure_type} | {row.worst_window} | "
                f"{float(row.max_severity_score):.2f} | {row.recommended_next_mechanism} |"
            )

    lines.extend(
        [
            "",
            "## Window Attribution",
            "",
            "| profile | window | rel return | max dd | avg pos | type | path |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    if gate_gaps.empty:
        lines.append("| none | none | 0.00pp | 0.00% | 0.000 | none | none |")
    else:
        for row in gate_gaps.itertuples(index=False):
            lines.append(
                "| "
                f"{row.profile} | {row.window} | {float(row.relative_return):.2f}pp | "
                f"{float(row.max_dd):.2f}% | {float(row.avg_pos):.3f} | "
                f"{row.failure_type} | {row.primary_optimization_path} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Treat joint trade-off rows as evidence against another single-coefficient reward sweep.",
            "- Use near-miss drawdown rows for cap calibration or execution friction checks before retraining.",
            "- Use underparticipation or trend-capture rows to justify participation-aware objectives or bull guards.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _empty_cluster_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cluster_id",
            "cluster_start",
            "cluster_end",
            "profile_count",
            "window_count",
            "profiles",
            "mean_excess_return",
            "worst_excess_return",
            "mean_max_dd",
            "max_max_dd",
            "mean_avg_pos",
            "mean_bad_score",
            "max_bad_score",
            "dominant_failure_type",
            "primary_optimization_path",
        ]
    )


def aggregate_poor_windows(
    windows_by_profile: Mapping[str, pd.DataFrame],
    *,
    target_excess: float = 30.0,
    target_max_dd: float = 15.0,
    overlap_gap_days: int = 0,
) -> pd.DataFrame:
    """Cluster poor windows from multiple profiles into shared failure regimes."""

    rows: list[pd.DataFrame] = []
    for profile, windows in windows_by_profile.items():
        if windows is None or windows.empty:
            continue
        frame = windows.copy()
        required = {"start", "end", "excess_return", "max_dd", "avg_pos", "bad_score"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{profile} poor_windows missing columns: {missing}")
        frame["profile"] = str(profile)
        frame["start_ts"] = pd.to_datetime(frame["start"])
        frame["end_ts"] = pd.to_datetime(frame["end"])
        frame["failure_type"] = frame.apply(
            lambda row: classify_poor_window(
                row,
                target_excess=target_excess,
                target_max_dd=target_max_dd,
            ),
            axis=1,
        )
        rows.append(frame)
    if not rows:
        return _empty_cluster_frame()

    all_windows = pd.concat(rows, ignore_index=True).sort_values(["start_ts", "end_ts"]).reset_index(drop=True)
    clusters: list[list[int]] = []
    current: list[int] = []
    current_end: pd.Timestamp | None = None
    gap = pd.Timedelta(days=max(int(overlap_gap_days), 0))
    for idx, row in all_windows.iterrows():
        start = pd.Timestamp(row["start_ts"])
        end = pd.Timestamp(row["end_ts"])
        if not current or (current_end is not None and start <= current_end + gap):
            current.append(idx)
            current_end = end if current_end is None else max(current_end, end)
        else:
            clusters.append(current)
            current = [idx]
            current_end = end
    if current:
        clusters.append(current)

    summary_rows: list[dict[str, object]] = []
    for cluster_idx, indexes in enumerate(clusters, start=1):
        sub = all_windows.loc[indexes].copy()
        failures = Counter(str(value) for value in sub["failure_type"])
        dominant_failure, _ = failures.most_common(1)[0]
        profiles = sorted(set(str(value) for value in sub["profile"]))
        summary_rows.append(
            {
                "cluster_id": f"C{cluster_idx:03d}",
                "cluster_start": _date_str(sub["start_ts"].min()),
                "cluster_end": _date_str(sub["end_ts"].max()),
                "profile_count": int(len(profiles)),
                "window_count": int(len(sub)),
                "profiles": ",".join(profiles),
                "mean_excess_return": float(sub["excess_return"].mean()),
                "worst_excess_return": float(sub["excess_return"].min()),
                "mean_max_dd": float(sub["max_dd"].mean()),
                "max_max_dd": float(sub["max_dd"].max()),
                "mean_avg_pos": float(sub["avg_pos"].mean()),
                "mean_bad_score": float(sub["bad_score"].mean()),
                "max_bad_score": float(sub["bad_score"].max()),
                "dominant_failure_type": dominant_failure,
                "primary_optimization_path": _optimization_path(dominant_failure),
            }
        )
    out = pd.DataFrame(summary_rows)
    return out.sort_values(
        ["profile_count", "max_bad_score", "cluster_start"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def aggregate_feature_differences(
    feature_differences_by_profile: Mapping[str, pd.DataFrame],
    *,
    top_n: int = 30,
) -> pd.DataFrame:
    """Aggregate feature-difference rows and rank stable cross-profile deltas."""

    frames: list[pd.DataFrame] = []
    for profile, diffs in feature_differences_by_profile.items():
        if diffs is None or diffs.empty:
            continue
        frame = diffs.copy()
        required = {"feature", "delta_z"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{profile} feature_differences missing columns: {missing}")
        frame["profile"] = str(profile)
        frame["delta_z"] = pd.to_numeric(frame["delta_z"], errors="coerce")
        if "spearman_to_excess" not in frame.columns:
            frame["spearman_to_excess"] = np.nan
        frame["spearman_to_excess"] = pd.to_numeric(frame["spearman_to_excess"], errors="coerce")
        frames.append(frame[["profile", "feature", "delta_z", "spearman_to_excess"]].dropna(subset=["delta_z"]))
    if not frames:
        return pd.DataFrame(
            columns=[
                "feature",
                "profile_count",
                "mean_delta_z",
                "mean_abs_delta_z",
                "direction_consistency",
                "mean_spearman_to_excess",
                "profiles",
                "stability_score",
            ]
        )

    joined = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for feature, sub in joined.groupby("feature"):
        signs = np.sign(sub["delta_z"].to_numpy(dtype=float))
        nonzero = signs[signs != 0]
        direction_consistency = float(abs(nonzero.sum()) / len(nonzero)) if len(nonzero) else 0.0
        profiles = sorted(set(str(value) for value in sub["profile"]))
        mean_abs_delta_z = float(sub["delta_z"].abs().mean())
        profile_count = int(len(profiles))
        stability_score = mean_abs_delta_z * direction_consistency * np.log1p(profile_count)
        rows.append(
            {
                "feature": str(feature),
                "profile_count": profile_count,
                "mean_delta_z": float(sub["delta_z"].mean()),
                "mean_abs_delta_z": mean_abs_delta_z,
                "direction_consistency": direction_consistency,
                "mean_spearman_to_excess": float(sub["spearman_to_excess"].mean()),
                "profiles": ",".join(profiles),
                "stability_score": float(stability_score),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["stability_score", "profile_count", "mean_abs_delta_z", "feature"],
        ascending=[False, False, False, True],
    ).head(top_n).reset_index(drop=True)


def _normalise_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"])
        out = out.set_index("datetime")
    if isinstance(out.index, pd.MultiIndex):
        if "datetime" in out.index.names:
            out = out.reset_index("instrument", drop=True)
        else:
            out.index = pd.to_datetime(out.index.get_level_values(-1))
    else:
        out.index = pd.to_datetime(out.index)
    out = out.select_dtypes(include=[np.number]).sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


def _window_mask(index: pd.DatetimeIndex, windows: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=index)
    for row in windows.itertuples(index=False):
        start = pd.Timestamp(getattr(row, "start"))
        end = pd.Timestamp(getattr(row, "end"))
        mask |= (index >= start) & (index <= end)
    return mask


def _spearman_to_excess(feature: pd.Series, daily: pd.DataFrame | None) -> float:
    if daily is None or "strategy_nav" not in daily.columns or "buyhold_nav" not in daily.columns:
        return float("nan")
    frame = _coerce_daily_frame(daily).set_index("datetime")
    strategy_return = frame["strategy_nav"].pct_change()
    buyhold_return = frame["buyhold_nav"].pct_change()
    excess = (strategy_return - buyhold_return).rename("daily_excess")
    joined = pd.concat([feature.rename("feature"), excess], axis=1).dropna()
    if len(joined) < 5 or joined["feature"].nunique() <= 1 or joined["daily_excess"].nunique() <= 1:
        return float("nan")
    return float(joined["feature"].corr(joined["daily_excess"], method="spearman"))


def compute_feature_differences(
    features: pd.DataFrame,
    poor_windows: pd.DataFrame,
    *,
    daily: pd.DataFrame | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compare feature distributions in poor windows against the rest of OOS."""

    feature_frame = _normalise_feature_frame(features)
    if feature_frame.empty or poor_windows.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "bad_mean",
                "reference_mean",
                "delta",
                "delta_z",
                "bad_coverage",
                "reference_coverage",
                "spearman_to_excess",
            ]
        )
    bad_mask = _window_mask(pd.DatetimeIndex(feature_frame.index), poor_windows)
    bad = feature_frame.loc[bad_mask]
    reference = feature_frame.loc[~bad_mask]
    if bad.empty or reference.empty:
        reference = feature_frame
    rows: list[dict[str, float | str]] = []
    for col in feature_frame.columns:
        bad_series = bad[col].dropna()
        ref_series = reference[col].dropna()
        if bad_series.empty or ref_series.empty:
            continue
        ref_std = float(ref_series.std(ddof=0))
        if ref_std == 0.0 or np.isnan(ref_std):
            ref_std = float(feature_frame[col].std(ddof=0))
        if ref_std == 0.0 or np.isnan(ref_std):
            continue
        bad_mean = float(bad_series.mean())
        ref_mean = float(ref_series.mean())
        delta = bad_mean - ref_mean
        rows.append(
            {
                "feature": str(col),
                "bad_mean": bad_mean,
                "reference_mean": ref_mean,
                "delta": float(delta),
                "delta_z": float(delta / ref_std),
                "bad_coverage": float(len(bad_series) / max(len(bad), 1)),
                "reference_coverage": float(len(ref_series) / max(len(reference), 1)),
                "spearman_to_excess": _spearman_to_excess(feature_frame[col], daily),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_delta_z"] = out["delta_z"].abs()
    return out.sort_values(["abs_delta_z", "feature"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def _fmt_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def _reference_rows(external_references: Sequence[Mapping[str, str]] | Sequence[ExternalReference]) -> list[str]:
    rows = []
    for ref in external_references:
        if isinstance(ref, ExternalReference):
            label, url, note = ref.label, ref.url, ref.note
        else:
            label = str(ref.get("label", "reference"))
            url = str(ref.get("url", ""))
            note = str(ref.get("note", ""))
        rows.append(f"- [{label}]({url}): {note}")
    return rows


def render_oos_diagnostic_review(
    *,
    profile: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
    target_excess: float,
    target_max_dd: float,
    windows: pd.DataFrame,
    feature_differences: pd.DataFrame,
    external_references: Sequence[Mapping[str, str]] | Sequence[ExternalReference] = (),
) -> str:
    lines = [
        f"# OOS Poor-Window Diagnostic: {profile}",
        "",
        "## Scope",
        "",
        f"- train_end: `{train_end}`",
        f"- OOS window: `{oos_start}` .. `{oos_end}`",
        f"- target_excess={target_excess:.2f}%",
        f"- target_max_dd={target_max_dd:.2f}%",
        "",
    ]
    if external_references:
        lines.extend(["## External References", "", *_reference_rows(external_references), ""])

    lines.extend(
        [
            "## Poor Windows",
            "",
            "| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if windows.empty:
        lines.append("| none | none | 0.00% | 0.00% | 0.00pp | 0.00% | 0.000 | 0.00 |")
    else:
        for row in windows.itertuples(index=False):
            lines.append(
                "| "
                f"{row.start} | {row.end} | {_fmt_pct(row.strategy_return)} | "
                f"{_fmt_pct(row.buyhold_return)} | {row.excess_return:.2f}pp | "
                f"{_fmt_pct(row.max_dd)} | {float(row.avg_pos):.3f} | {float(row.bad_score):.2f} |"
            )

    lines.extend(
        [
            "",
            "## Feature Differences",
            "",
            "| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if feature_differences.empty:
        lines.append("| none | 0.0000 | 0.0000 | 0.000 | nan |")
    else:
        for row in feature_differences.itertuples(index=False):
            corr = getattr(row, "spearman_to_excess", float("nan"))
            corr_text = "nan" if pd.isna(corr) else f"{float(corr):.3f}"
            lines.append(
                "| "
                f"{row.feature} | {float(row.bad_mean):.4f} | {float(row.reference_mean):.4f} | "
                f"{float(row.delta_z):.3f} | {corr_text} |"
            )

    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "- Treat repeated negative-excess windows as overfit candidates before retraining.",
            "- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.",
            "- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_diagnostic_artifacts(
    *,
    output_dir: Path,
    profile: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
    target_excess: float,
    target_max_dd: float,
    windows: pd.DataFrame,
    feature_differences: pd.DataFrame,
    external_references: Sequence[Mapping[str, str]] = (),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = output_dir / "poor_windows.csv"
    features_path = output_dir / "feature_differences.csv"
    review_path = output_dir / "oos_diagnostic_review.md"
    windows.to_csv(windows_path, index=False)
    feature_differences.to_csv(features_path, index=False)
    review_path.write_text(
        render_oos_diagnostic_review(
            profile=profile,
            train_end=train_end,
            oos_start=oos_start,
            oos_end=oos_end,
            target_excess=target_excess,
            target_max_dd=target_max_dd,
            windows=windows,
            feature_differences=feature_differences,
            external_references=external_references,
        ),
        encoding="utf-8",
    )
    return {"windows": windows_path, "features": features_path, "review": review_path}


def load_qlib_feature_frame(start: str, end: str) -> pd.DataFrame:
    """Load the repo qlib feature frame for a date range."""

    from src.soxl_quant.data import build_dataset, init_qlib
    from src.soxl_quant.settings import QLIB_DATA_PATH

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    frame = dataset.handler.feature_frame.droplevel("instrument").sort_index()
    return frame.loc[start:end]


def run_oos_diagnostic(
    *,
    daily_csv: Path,
    profile: str,
    output_dir: Path,
    oos_start: str = "2020-01-02",
    oos_end: str = "",
    window_size: int = 63,
    top_windows: int = 12,
    top_features: int = 30,
    target_excess: float = 30.0,
    target_max_dd: float = 15.0,
    train_end: str | None = None,
    external_references: Sequence[Mapping[str, str]] = (),
) -> dict[str, object]:
    """Run the full poor-window and feature-difference diagnostic."""

    from src.soxl_quant.settings import TRAIN_SEG

    daily = pd.read_csv(daily_csv)
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    resolved_oos_end = oos_end or str(daily["datetime"].max().date())
    daily = daily[(daily["datetime"] >= oos_start) & (daily["datetime"] <= resolved_oos_end)].copy()
    if len(daily) < window_size:
        raise ValueError(f"not enough rows for window_size={window_size}: {len(daily)}")

    windows = find_poor_windows(
        daily,
        window_size=window_size,
        top_n=top_windows,
        target_excess=target_excess,
        target_max_dd=target_max_dd,
    )
    features = load_qlib_feature_frame(oos_start, resolved_oos_end)
    feature_diffs = compute_feature_differences(features, windows, daily=daily, top_n=top_features)
    artifacts = write_diagnostic_artifacts(
        output_dir=output_dir,
        profile=profile,
        train_end=train_end or TRAIN_SEG[1],
        oos_start=oos_start,
        oos_end=resolved_oos_end,
        target_excess=target_excess,
        target_max_dd=target_max_dd,
        windows=windows,
        feature_differences=feature_diffs,
        external_references=external_references,
    )
    manifest: dict[str, object] = {
        "profile": profile,
        "daily_csv": str(daily_csv),
        "oos_start": oos_start,
        "oos_end": resolved_oos_end,
        "window_size": window_size,
        "target_excess": target_excess,
        "target_max_dd": target_max_dd,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "external_references": list(external_references),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    manifest["windows"] = int(len(windows))
    manifest["features"] = int(len(feature_diffs))
    manifest["manifest"] = str(manifest_path)
    return manifest


__all__ = [
    "ExternalReference",
    "aggregate_feature_differences",
    "aggregate_poor_windows",
    "attribute_gate_gaps",
    "classify_poor_window",
    "compute_feature_differences",
    "find_poor_windows",
    "load_qlib_feature_frame",
    "normalise_gate_metrics_from_selection",
    "render_gate_gap_review",
    "render_oos_diagnostic_review",
    "run_oos_diagnostic",
    "summarize_gate_gap_attribution",
    "write_diagnostic_artifacts",
]
