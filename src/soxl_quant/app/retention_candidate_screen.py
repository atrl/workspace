from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverlaySpec:
    name: str
    base_scale: float = 1.0
    global_cap: float = 1.0
    drawdown_lookback: int = 63
    mild_drawdown: float = -0.10
    mild_cap: float = 0.50
    severe_drawdown: float = -0.20
    severe_cap: float = 0.25


@dataclass(frozen=True)
class PathMetrics:
    return_pct: float
    max_drawdown_pct: float
    avg_position: float
    last_position: float
    rows: int


@dataclass(frozen=True)
class CandidateMetrics:
    candidate: str
    ytd_return_pct: float | None
    ytd_max_drawdown_pct: float | None
    worst_wf_max_drawdown_pct: float | None
    min_wf_return_pct: float | None
    mean_wf_return_pct: float | None
    wf_windows_present: tuple[str, ...]
    missing_wf_windows: tuple[str, ...]


def load_daily_path(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "datetime" not in frame.columns:
        if "date" in frame.columns:
            frame = frame.rename(columns={"date": "datetime"})
        else:
            raise ValueError(f"{path} missing datetime/date column")
    required = {"price", "position"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["position"] = pd.to_numeric(out["position"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["price"]).sort_values("datetime").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path} has no usable rows")
    return out


def overlay_positions(frame: pd.DataFrame, spec: OverlaySpec) -> pd.Series:
    price = pd.to_numeric(frame["price"], errors="coerce")
    base_position = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
    lookback = max(int(spec.drawdown_lookback), 1)
    rolling_peak = price.rolling(lookback, min_periods=1).max()
    price_drawdown = price / rolling_peak - 1.0
    position = (base_position * float(spec.base_scale)).clip(lower=0.0, upper=float(spec.global_cap))
    position = position.mask(price_drawdown <= float(spec.mild_drawdown), np.minimum(position, float(spec.mild_cap)))
    position = position.mask(price_drawdown <= float(spec.severe_drawdown), np.minimum(position, float(spec.severe_cap)))
    return position.clip(lower=0.0, upper=1.0)


def evaluate_position_path(frame: pd.DataFrame, spec: OverlaySpec) -> PathMetrics:
    if len(frame) < 2:
        return PathMetrics(
            return_pct=0.0,
            max_drawdown_pct=0.0,
            avg_position=float(frame["position"].mean()) if len(frame) else 0.0,
            last_position=float(frame["position"].iloc[-1]) if len(frame) else 0.0,
            rows=int(len(frame)),
        )
    price = pd.to_numeric(frame["price"], errors="coerce").astype(float)
    positions = overlay_positions(frame, spec)
    returns = price.pct_change().fillna(0.0)
    nav = (1.0 + positions.shift(1).fillna(0.0) * returns).cumprod()
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    return PathMetrics(
        return_pct=round(float((nav.iloc[-1] / nav.iloc[0] - 1.0) * 100.0), 6),
        max_drawdown_pct=round(float(abs(drawdown.min()) * 100.0), 6),
        avg_position=round(float(positions.mean()), 6),
        last_position=round(float(positions.iloc[-1]), 6),
        rows=int(len(frame)),
    )


def evaluate_export_nav_path(frame: pd.DataFrame) -> PathMetrics:
    if "strategy_nav" not in frame.columns:
        raise ValueError("export frame missing strategy_nav column")
    if len(frame) < 2:
        return PathMetrics(
            return_pct=0.0,
            max_drawdown_pct=0.0,
            avg_position=float(frame["position"].mean()) if "position" in frame.columns and len(frame) else 0.0,
            last_position=float(frame["position"].iloc[-1]) if "position" in frame.columns and len(frame) else 0.0,
            rows=int(len(frame)),
        )
    nav = pd.to_numeric(frame["strategy_nav"], errors="coerce").astype(float)
    if nav.isna().any():
        raise ValueError("strategy_nav contains non-numeric values")
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    position = pd.to_numeric(frame.get("position", pd.Series([0.0] * len(frame))), errors="coerce").fillna(0.0)
    return PathMetrics(
        return_pct=round(float((nav.iloc[-1] / nav.iloc[0] - 1.0) * 100.0), 6),
        max_drawdown_pct=round(float(abs(drawdown.min()) * 100.0), 6),
        avg_position=round(float(position.mean()), 6),
        last_position=round(float(position.iloc[-1]), 6),
        rows=int(len(frame)),
    )


def strict_gate_failures(
    metrics: CandidateMetrics,
    *,
    required_wf_windows: Sequence[str],
    max_walkforward_drawdown_pct: float,
    required_soxlbacktest_return_pct: float,
    required_best_return_pct: float,
) -> list[str]:
    failures: list[str] = []
    present = set(metrics.wf_windows_present)
    for window in required_wf_windows:
        if window not in present:
            failures.append(f"missing_wf_window:{window}")
    worst_dd = metrics.worst_wf_max_drawdown_pct
    if worst_dd is None or float(worst_dd) >= float(max_walkforward_drawdown_pct):
        failures.append("worst_wf_max_drawdown_ge_20")
    ytd_return = metrics.ytd_return_pct
    if ytd_return is None or float(ytd_return) < float(required_soxlbacktest_return_pct):
        failures.append("ytd_return_below_soxlbacktest_reference")
    if ytd_return is None or float(ytd_return) < float(required_best_return_pct):
        failures.append("ytd_return_below_current_best_alignment")
    return failures


def summarize_candidate(
    *,
    candidate: str,
    ytd_metrics: PathMetrics | None,
    wf_metrics: dict[str, PathMetrics],
    required_wf_windows: Sequence[str],
) -> CandidateMetrics:
    present = tuple(window for window in required_wf_windows if window in wf_metrics)
    missing = tuple(window for window in required_wf_windows if window not in wf_metrics)
    wf_returns = [metric.return_pct for metric in wf_metrics.values()]
    wf_drawdowns = [metric.max_drawdown_pct for metric in wf_metrics.values()]
    return CandidateMetrics(
        candidate=candidate,
        ytd_return_pct=ytd_metrics.return_pct if ytd_metrics is not None else None,
        ytd_max_drawdown_pct=ytd_metrics.max_drawdown_pct if ytd_metrics is not None else None,
        worst_wf_max_drawdown_pct=max(wf_drawdowns) if wf_drawdowns else None,
        min_wf_return_pct=min(wf_returns) if wf_returns else None,
        mean_wf_return_pct=float(np.mean(wf_returns)) if wf_returns else None,
        wf_windows_present=present,
        missing_wf_windows=missing,
    )


def candidate_to_row(
    metrics: CandidateMetrics,
    *,
    failures: Iterable[str],
    overlay: str,
    source_profile: str,
) -> dict[str, object]:
    failure_list = list(failures)
    return {
        "candidate": metrics.candidate,
        "source_profile": source_profile,
        "overlay": overlay,
        "ytd_return_pct": metrics.ytd_return_pct,
        "ytd_max_drawdown_pct": metrics.ytd_max_drawdown_pct,
        "worst_wf_max_drawdown_pct": metrics.worst_wf_max_drawdown_pct,
        "min_wf_return_pct": metrics.min_wf_return_pct,
        "mean_wf_return_pct": metrics.mean_wf_return_pct,
        "wf_windows_present": ",".join(metrics.wf_windows_present),
        "missing_wf_windows": ",".join(metrics.missing_wf_windows),
        "strict_gate_pass": not failure_list,
        "failures": ";".join(failure_list),
    }
