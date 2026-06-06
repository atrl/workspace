from __future__ import annotations

"""Fast heuristic strategy search for QuantBench-style SOXL experiments."""

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .quantbench_gates import BacktestGateMetrics, compute_backtest_metrics, slice_window


DEFAULT_HEURISTIC_WINDOWS: dict[str, tuple[str | None, str | None]] = {
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
}


@dataclass(frozen=True)
class HeuristicStrategyConfig:
    momentum_lookback: int = 20
    soxx_momentum_lookback: int = 20
    require_soxx_confirmation: bool = True
    vix_threshold: float = 24.0
    bull_position: float = 1.0
    risk_position: float = 0.35
    cash_position: float = 0.0
    drawdown_start: float = 0.08
    drawdown_full: float = 0.20
    drawdown_min_position: float = 0.0


@dataclass(frozen=True)
class HeuristicStrategyResult:
    config: HeuristicStrategyConfig
    daily: pd.DataFrame
    window_metrics: dict[str, BacktestGateMetrics]
    gate_pass: bool
    score: float


def _normalise_panel(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    if "datetime" not in frame.columns:
        if "date" not in frame.columns:
            raise ValueError("panel must contain `datetime` or `date`")
        frame["datetime"] = frame["date"]
    required = {"datetime", "soxl_close", "soxx_close", "vix"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for col in ("soxl_close", "soxx_close", "vix"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=list(required)).sort_values("datetime").reset_index(drop=True)


def _clamp_position(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def build_heuristic_position_path(panel: pd.DataFrame, config: HeuristicStrategyConfig) -> pd.DataFrame:
    """Build a causal daily position path from price momentum and risk state."""

    frame = _normalise_panel(panel)
    soxl = frame["soxl_close"].astype(float)
    soxx = frame["soxx_close"].astype(float)
    soxl_momentum = soxl / soxl.shift(max(int(config.momentum_lookback), 1)) - 1.0
    soxx_momentum = soxx / soxx.shift(max(int(config.soxx_momentum_lookback), 1)) - 1.0
    if config.require_soxx_confirmation:
        trend_ready = (soxl_momentum > 0.0) & (soxx_momentum > 0.0)
    else:
        trend_ready = soxl_momentum > 0.0
    risk_state = frame["vix"].astype(float) > float(config.vix_threshold)

    bull_position = _clamp_position(config.bull_position)
    risk_position = _clamp_position(config.risk_position)
    cash_position = _clamp_position(config.cash_position)
    raw_position = np.where(trend_ready & ~risk_state, bull_position, np.where(trend_ready, risk_position, cash_position))
    raw_position = np.nan_to_num(raw_position.astype(float), nan=cash_position)

    returns = soxl.pct_change().fillna(0.0).to_numpy(dtype=float)
    nav = 1.0
    peak_nav = 1.0
    previous_position = float(raw_position[0]) if len(raw_position) else 0.0
    capped_positions: list[float] = []
    strategy_nav: list[float] = []
    caps: list[float] = []
    drawdowns: list[float] = []

    start = max(float(config.drawdown_start), 0.0)
    full = max(float(config.drawdown_full), start)
    min_position = _clamp_position(config.drawdown_min_position)

    for idx, daily_return in enumerate(returns):
        if idx > 0:
            nav *= 1.0 + previous_position * float(daily_return)
        peak_nav = max(peak_nav, nav)
        drawdown = max(1.0 - nav / max(peak_nav, 1e-12), 0.0)
        if drawdown >= full:
            cap = min_position
        elif drawdown <= start:
            cap = 1.0
        else:
            cap = 1.0 - ((drawdown - start) / max(full - start, 1e-12)) * (1.0 - min_position)
        position = min(float(raw_position[idx]), cap)
        capped_positions.append(_clamp_position(position))
        strategy_nav.append(float(nav))
        caps.append(float(cap))
        drawdowns.append(float(drawdown))
        previous_position = float(position)

    out = pd.DataFrame(
        {
            "datetime": frame["datetime"],
            "price": soxl,
            "strategy_nav": strategy_nav,
            "buyhold_nav": soxl / float(soxl.iloc[0]),
            "position": capped_positions,
            "raw_position": raw_position,
            "drawdown_cap": caps,
            "strategy_drawdown": drawdowns,
        }
    )
    return out


def evaluate_heuristic_strategy(
    panel: pd.DataFrame,
    config: HeuristicStrategyConfig,
    *,
    windows: Mapping[str, tuple[str | None, str | None]] | None = None,
    max_drawdown_pct: float = 20.0,
) -> HeuristicStrategyResult:
    daily = build_heuristic_position_path(panel, config)
    window_specs = dict(windows or DEFAULT_HEURISTIC_WINDOWS)
    metrics: dict[str, BacktestGateMetrics] = {}
    score = 0.0
    for window_name, (start, end) in window_specs.items():
        metric = compute_backtest_metrics(
            slice_window(daily, start, end),
            window=window_name,
            profile="heuristic_strategy",
            max_drawdown_pct=max_drawdown_pct,
        )
        metrics[window_name] = metric
        score += max(0.0, metric.max_drawdown_pct - float(max_drawdown_pct)) ** 2
        score += max(0.0, -metric.relative_return_pct) ** 2
        if metric.rows == 0:
            score += 1_000_000.0
    gate_pass = bool(metrics and all(metric.gate_pass for metric in metrics.values()))
    return HeuristicStrategyResult(config=config, daily=daily, window_metrics=metrics, gate_pass=gate_pass, score=float(score))


def screen_heuristic_strategies(
    panel: pd.DataFrame,
    configs: Sequence[HeuristicStrategyConfig],
    *,
    windows: Mapping[str, tuple[str | None, str | None]] | None = None,
    max_drawdown_pct: float = 20.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, config in enumerate(configs):
        result = evaluate_heuristic_strategy(panel, config, windows=windows, max_drawdown_pct=max_drawdown_pct)
        row: dict[str, object] = {
            "config_id": int(idx),
            "gate_pass": bool(result.gate_pass),
            "score": float(result.score),
            **asdict(config),
        }
        for window_name, metric in result.window_metrics.items():
            row[f"{window_name}_rows"] = int(metric.rows)
            row[f"{window_name}_strategy_return_pct"] = float(metric.strategy_return_pct)
            row[f"{window_name}_buyhold_return_pct"] = float(metric.buyhold_return_pct)
            row[f"{window_name}_relative_return_pct"] = float(metric.relative_return_pct)
            row[f"{window_name}_max_drawdown_pct"] = float(metric.max_drawdown_pct)
            row[f"{window_name}_gate_pass"] = bool(metric.gate_pass)
            row[f"{window_name}_failures"] = ",".join(metric.failures)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["gate_pass", "score"], ascending=[False, True]).reset_index(drop=True)
    if "gate_pass" in out.columns:
        out["gate_pass"] = out["gate_pass"].astype(object)
    return out


def _position_nav_for_prices(position: np.ndarray, price: np.ndarray) -> np.ndarray:
    returns = np.zeros(len(price), dtype=float)
    if len(price) > 1:
        returns[1:] = np.divide(price[1:], price[:-1], out=np.ones(len(price) - 1), where=price[:-1] != 0.0) - 1.0
    active_position = np.r_[position[0], position[:-1]] if len(position) else np.asarray([], dtype=float)
    return np.cumprod(1.0 + active_position * returns)


def _candidate_lookback_score(position: np.ndarray, price: np.ndarray, *, max_drawdown_pct: float) -> float:
    if len(price) < 2 or len(position) < 2:
        return 1_000_000.0
    nav = _position_nav_for_prices(position, price)
    buyhold = price / price[0] if price[0] else np.ones(len(price), dtype=float)
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2000-01-01", periods=len(price), freq="D"),
            "strategy_nav": nav,
            "buyhold_nav": buyhold,
        }
    )
    metric = compute_backtest_metrics(frame, window="lookback", max_drawdown_pct=max_drawdown_pct)
    score = max(0.0, metric.max_drawdown_pct - float(max_drawdown_pct)) ** 2
    score += max(0.0, -metric.relative_return_pct) ** 2
    score -= max(0.0, metric.relative_return_pct) * 0.01
    return float(score)


def rolling_select_candidate_path(
    panel: pd.DataFrame,
    candidate_paths: Sequence[pd.DataFrame],
    *,
    lookback_days: int = 252,
    rebalance_days: int = 21,
    min_history_days: int = 63,
    max_drawdown_pct: float = 20.0,
    initial_candidate_id: int = 0,
) -> pd.DataFrame:
    """Select among candidate position paths using only past lookback performance."""

    frame = _normalise_panel(panel)
    if not candidate_paths:
        raise ValueError("candidate_paths must not be empty")
    lookback = max(int(lookback_days), 2)
    rebalance = max(int(rebalance_days), 1)
    min_history = max(int(min_history_days), 1)

    candidates: list[np.ndarray] = []
    base = frame[["datetime"]].copy()
    for candidate in candidate_paths:
        if "datetime" not in candidate.columns or "position" not in candidate.columns:
            raise ValueError("each candidate path must contain `datetime` and `position`")
        c = candidate[["datetime", "position"]].copy()
        c["datetime"] = pd.to_datetime(c["datetime"], errors="coerce")
        c["position"] = pd.to_numeric(c["position"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        merged = base.merge(c.dropna(subset=["datetime"]), on="datetime", how="left")
        candidates.append(pd.to_numeric(merged["position"], errors="coerce").fillna(0.0).to_numpy(dtype=float))

    price = frame["soxl_close"].to_numpy(dtype=float)
    selected_ids: list[int] = []
    selected_positions: list[float] = []
    current_id = max(0, min(int(initial_candidate_id), len(candidates) - 1))
    for idx in range(len(frame)):
        if idx >= min_history and (idx - min_history) % rebalance == 0:
            start = max(0, idx - lookback)
            scores = [
                _candidate_lookback_score(candidate[start:idx], price[start:idx], max_drawdown_pct=max_drawdown_pct)
                for candidate in candidates
            ]
            current_id = int(np.argmin(scores))
        selected_ids.append(current_id)
        selected_positions.append(float(candidates[current_id][idx]))

    position = np.asarray(selected_positions, dtype=float)
    strategy_nav = _position_nav_for_prices(position, price)
    out = pd.DataFrame(
        {
            "datetime": frame["datetime"],
            "price": price,
            "strategy_nav": strategy_nav,
            "buyhold_nav": price / price[0] if len(price) and price[0] else np.ones(len(price), dtype=float),
            "position": position,
            "selected_candidate_id": selected_ids,
        }
    )
    return out


def default_heuristic_config_grid() -> list[HeuristicStrategyConfig]:
    configs: list[HeuristicStrategyConfig] = []
    drawdown_budgets = (
        (0.08, 0.20, 0.0),
        (0.08, 0.20, 0.25),
        (0.99, 1.00, 0.0),
    )
    for momentum in (20, 40, 60, 120):
        for soxx_momentum in (20, 60, 120):
            for require_soxx_confirmation in (True, False):
                for vix_threshold in (18.0, 22.0, 26.0, 32.0):
                    for bull_position in (0.7, 0.85, 1.0):
                        for risk_position in (0.0, 0.2, 0.4):
                            for drawdown_start, drawdown_full, drawdown_min_position in drawdown_budgets:
                                configs.append(
                                    HeuristicStrategyConfig(
                                        momentum_lookback=momentum,
                                        soxx_momentum_lookback=soxx_momentum,
                                        require_soxx_confirmation=require_soxx_confirmation,
                                        vix_threshold=vix_threshold,
                                        bull_position=bull_position,
                                        risk_position=risk_position,
                                        cash_position=0.0,
                                        drawdown_start=drawdown_start,
                                        drawdown_full=drawdown_full,
                                        drawdown_min_position=drawdown_min_position,
                                    )
                                )
    return configs
