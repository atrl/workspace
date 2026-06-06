from __future__ import annotations

"""QuantBench-style replay gate checks for SOXL model candidates."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_WINDOWS: dict[str, tuple[str | None, str | None]] = {
    "full_replay": (None, None),
    "test_2020": ("2020-01-01", "2020-12-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
    "latest_2025_2026": ("2025-01-01", "2026-12-31"),
}


@dataclass(frozen=True)
class BacktestGateMetrics:
    profile: str
    path: str
    window: str
    rows: int
    start: str | None
    end: str | None
    strategy_return_pct: float
    buyhold_return_pct: float
    relative_return_pct: float
    max_drawdown_pct: float
    gate_pass: bool
    failures: tuple[str, ...]


def _max_drawdown_pct(nav: pd.Series) -> float:
    values = pd.to_numeric(nav, errors="coerce").dropna()
    if values.empty:
        return 0.0
    peak = values.cummax()
    drawdown = 1.0 - values / peak.replace(0.0, pd.NA)
    return float(drawdown.fillna(0.0).max() * 100.0)


def _return_pct(nav: pd.Series) -> float:
    values = pd.to_numeric(nav, errors="coerce").dropna()
    if len(values) < 2 or float(values.iloc[0]) == 0.0:
        return 0.0
    return float((values.iloc[-1] / values.iloc[0] - 1.0) * 100.0)


def slice_window(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    result = result.dropna(subset=["datetime"]).sort_values("datetime")
    if start is not None:
        result = result.loc[result["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        result = result.loc[result["datetime"] <= pd.Timestamp(end)]
    return result.reset_index(drop=True)


def compute_backtest_metrics(
    frame: pd.DataFrame,
    *,
    window: str,
    profile: str = "",
    path: str = "",
    max_drawdown_pct: float = 20.0,
) -> BacktestGateMetrics:
    if frame.empty:
        return BacktestGateMetrics(
            profile=profile,
            path=path,
            window=window,
            rows=0,
            start=None,
            end=None,
            strategy_return_pct=0.0,
            buyhold_return_pct=0.0,
            relative_return_pct=0.0,
            max_drawdown_pct=0.0,
            gate_pass=False,
            failures=("empty_window",),
        )
    ordered = frame.copy()
    ordered["datetime"] = pd.to_datetime(ordered["datetime"], errors="coerce")
    ordered = ordered.dropna(subset=["datetime"]).sort_values("datetime")
    strategy_return = _return_pct(ordered["strategy_nav"])
    buyhold_return = _return_pct(ordered["buyhold_nav"])
    relative_return = strategy_return - buyhold_return
    max_dd = _max_drawdown_pct(ordered["strategy_nav"])
    failures: list[str] = []
    if max_dd >= max_drawdown_pct:
        failures.append(f"max_drawdown_ge_{int(max_drawdown_pct)}")
    if relative_return <= 0:
        failures.append("relative_return_not_positive")
    return BacktestGateMetrics(
        profile=profile,
        path=path,
        window=window,
        rows=int(len(ordered)),
        start=ordered["datetime"].min().strftime("%Y-%m-%d"),
        end=ordered["datetime"].max().strftime("%Y-%m-%d"),
        strategy_return_pct=round(strategy_return, 6),
        buyhold_return_pct=round(buyhold_return, 6),
        relative_return_pct=round(relative_return, 6),
        max_drawdown_pct=round(max_dd, 6),
        gate_pass=not failures,
        failures=tuple(failures),
    )


def _profile_from_path(path: Path) -> str:
    name = path.name
    suffix = "_soxl_daily.csv"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def scan_model_replays(
    *,
    root: Path,
    windows: dict[str, tuple[str | None, str | None]] | None = None,
    required_windows: Sequence[str] = ("test_2022", "test_2023_2024"),
    max_drawdown_pct: float = 20.0,
) -> dict[str, object]:
    root = Path(root)
    model_dir = root / "runtime" / "model" / "card"
    window_specs = windows or DEFAULT_WINDOWS
    rows: list[dict[str, object]] = []
    by_profile: dict[str, list[BacktestGateMetrics]] = {}
    for path in sorted(model_dir.glob("*/*_soxl_daily.csv")):
        profile = _profile_from_path(path)
        frame = pd.read_csv(path)
        for window_name, (start, end) in window_specs.items():
            sliced = slice_window(frame, start, end)
            metrics = compute_backtest_metrics(
                sliced,
                window=window_name,
                profile=profile,
                path=str(path.relative_to(root)),
                max_drawdown_pct=max_drawdown_pct,
            )
            by_profile.setdefault(profile, []).append(metrics)
            payload = asdict(metrics)
            payload["failures"] = list(metrics.failures)
            rows.append(payload)
    passing_profiles: list[str] = []
    for profile, metrics_list in by_profile.items():
        metrics_by_window = {metric.window: metric for metric in metrics_list}
        if all(metrics_by_window.get(window) and metrics_by_window[window].gate_pass for window in required_windows):
            passing_profiles.append(profile)
    return {
        "summary": {
            "profiles_scanned": len(by_profile),
            "rows": len(rows),
            "required_windows": list(required_windows),
            "max_drawdown_pct": float(max_drawdown_pct),
            "passing_profiles": len(passing_profiles),
        },
        "passing_profiles": sorted(passing_profiles),
        "rows": rows,
    }


def write_scan_outputs(result: dict[str, object], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result.get("rows", [])
    pd.DataFrame(rows).to_csv(output_dir / "model_gate_scan.csv", index=False)
    import json

    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
