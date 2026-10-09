"""Diagnose and reduce the cash drag of a defensive SOXL/SOXX/cash policy.

The Stage1 production policy is deliberately defensive and often holds a large
cash weight. Two questions matter:

1. Is the cash caused by the policy's own raw target, or by a hard cap that
   flattens a more aggressive raw target? This decides whether the right fix is
   a structural overlay or an upstream change.
2. Can a bounded, pre-registered overlay that recycles part of the idle cash
   into the *unleveraged* underlying improve risk-adjusted excess over SOXX
   without giving up the drawdown protection?

The overlay is mechanical: it never inspects future returns, it only moves a
fixed fraction of the existing cash weight into SOXX. It does not change the
policy's own risk decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

ASSETS = ("soxl", "soxx", "cash")
ANNUALIZATION_SESSIONS = 252
SCOPE = "production_target_history_diagnostic_not_promotion_evidence"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class CashOverlay:
    """Move ``recycle_fraction`` of the cash weight into the underlying (SOXX).

    ``recycle_fraction=0`` is the unmodified policy. A fraction of 1.0 would
    fully remove cash. The overlay is applied to the already-legal policy
    weights, so the SOXL weight and total risk budget are unchanged.
    """

    name: str
    recycle_fraction: float

    def validate(self):
        _require(isinstance(self.name, str) and self.name, "overlay name is required")
        _require(isinstance(self.recycle_fraction, Real) and not isinstance(self.recycle_fraction, bool)
                 and np.isfinite(self.recycle_fraction)
                 and 0.0 <= float(self.recycle_fraction) <= 1.0,
                 "recycle_fraction must be in [0, 1]")
        return self


def apply_overlay(weights: np.ndarray, overlay: CashOverlay) -> np.ndarray:
    """Return legal weights with part of the cash moved into SOXX."""
    overlay.validate()
    values = np.asarray(weights, dtype=float)
    _require(values.ndim == 2 and values.shape[1] == 3, "weights must be (n, 3)")
    _require(np.isfinite(values).all() and (values >= -1e-12).all(), "weights must be finite and long-only")
    _require(np.allclose(values.sum(axis=1), 1.0, atol=1e-8), "weights must be a funded simplex per row")
    result = values.copy()
    moved = values[:, 2] * overlay.recycle_fraction
    result[:, 2] = values[:, 2] - moved
    result[:, 1] = values[:, 1] + moved
    _require(np.allclose(result.sum(axis=1), 1.0, atol=1e-8), "overlay must preserve funding")
    return result


def replay_weights(weights: pd.DataFrame, asset_returns: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """Replay a target-weight schedule with owned turnover fees.

    Weights at session t are applied to session t's returns; turnover is charged
    against the previously held (drifted) weights using the same convention as
    the production ledger.
    """
    _require(cost_bps >= 0, "cost_bps must be nonnegative")
    frame = weights.join(asset_returns, how="inner")
    columns = [f"{asset}_weight" for asset in ASSETS]
    returns = asset_returns.loc[frame.index]
    target = frame.loc[:, columns].to_numpy(float)
    ret = returns.to_numpy(float)
    rows = []
    held = np.array([0.0, 0.0, 1.0])
    for i in range(len(frame)):
        # Drift the previously held weights by this session's return.
        drifted = held * (1.0 + ret[i])
        drifted = drifted / float(drifted.sum())
        turnover = float(np.abs(target[i] - drifted).sum())
        fee = turnover * float(cost_bps) / 1e4
        net = float(target[i] @ ret[i]) - fee
        rows.append({"signal_date": frame.index[i], "strategy_return": net,
                     "turnover": turnover, "cash_weight": target[i, 2],
                     "soxx_weight": target[i, 1], "soxl_weight": target[i, 0]})
        held = target[i]
    return pd.DataFrame(rows).set_index("signal_date")


def performance(replay: pd.DataFrame, benchmark_return: pd.Series, benchmark_name: str) -> dict:
    strategy = replay["strategy_return"]
    benchmark = benchmark_return.reindex(strategy.index).dropna()
    strategy = strategy.reindex(benchmark.index)
    growth = np.cumprod(1.0 + strategy.to_numpy(float))
    bench_growth = np.cumprod(1.0 + benchmark.to_numpy(float))
    def drawdown(series):
        g = np.cumprod(1.0 + series.to_numpy(float))
        return float((g / np.maximum.accumulate(g) - 1.0).min())
    std = strategy.std(ddof=1)
    bench_std = benchmark.std(ddof=1)
    return {
        "sessions": int(len(strategy)),
        "strategy_return_pct": 100.0 * float(growth[-1] - 1.0),
        "benchmark_return_pct": 100.0 * float(bench_growth[-1] - 1.0),
        "strategy_sharpe": float(strategy.mean() / std * np.sqrt(ANNUALIZATION_SESSIONS)) if std > 0 else 0.0,
        "benchmark_sharpe": float(benchmark.mean() / bench_std * np.sqrt(ANNUALIZATION_SESSIONS)) if bench_std > 0 else 0.0,
        "strategy_max_drawdown_pct": 100.0 * drawdown(strategy),
        "benchmark_max_drawdown_pct": 100.0 * drawdown(benchmark),
        "average_cash_weight": float(replay["cash_weight"].mean()),
        "average_soxx_weight": float(replay["soxx_weight"].mean()),
        "total_turnover": float(replay["turnover"].sum()),
        "beats_return": bool(growth[-1] > bench_growth[-1]),
        "beats_sharpe": bool((strategy.mean() / std if std > 0 else 0) > (benchmark.mean() / bench_std if bench_std > 0 else 0)),
        "beats_drawdown": bool(drawdown(strategy) > drawdown(benchmark)),
        "benchmark": benchmark_name,
        "scope": SCOPE,
    }


def constrain_variants(weights: pd.DataFrame, asset_returns: pd.DataFrame, *,
                       action_beta: float, maximum_soxl_weight: float,
                       minimum_cash_weight: float, cost_bps: float) -> pd.DataFrame:
    """Re-project weights under a *changed* constraint set, then replay.

    This isolates the effect of a single structural constraint without touching
    the frozen policy signal. ``weights`` are the observed legal targets; the
    re-projection rescales the raw risk budget implied by the targets' beta.
    """
    from src.workflow.intraday_recovery_direct_allocation import project_legal_weights

    columns = [f"{asset}_weight" for asset in ASSETS]
    target = weights.loc[:, columns].to_numpy(float)
    rows = np.empty_like(target)
    for i in range(len(target)):
        soxl, soxx, _ = target[i]
        raw_beta = 3.0 * soxl + soxx
        effective_beta = min(raw_beta, float(action_beta))
        rows[i] = project_legal_weights(soxl, soxx, effective_beta,
                                        minimum_cash_weight=float(minimum_cash_weight),
                                        maximum_soxl_weight=float(maximum_soxl_weight))
    reprojected = pd.DataFrame(rows, index=weights.index, columns=columns)
    return replay_weights(reprojected, asset_returns, cost_bps=cost_bps)


def long_history_constraint_sweep(raw_targets: pd.DataFrame, asset_returns: pd.DataFrame, *,
                                  variants: dict[str, dict], cost_bps: float) -> pd.DataFrame:
    """Re-project the frozen raw upstream target under changed constraints over long history.

    ``raw_targets`` has columns ``soxl_weight``/``soxx_weight`` from the frozen
    allocation model; ``asset_returns`` has ``soxl``/``soxx`` next-open returns.
    No signal is changed: only the three frozen sizing constraints move.
    """
    from src.workflow.intraday_recovery_direct_allocation import project_legal_weights

    rows = []
    soxl_ret = asset_returns["soxl"].to_numpy(float)
    soxx_ret = asset_returns["soxx"].to_numpy(float)
    raw_sxl = raw_targets["soxl_weight"].to_numpy(float)
    raw_sxx = raw_targets["soxx_weight"].to_numpy(float)
    for name, cfg in variants.items():
        rets = np.empty(len(raw_targets))
        held = np.array([0.0, 0.0, 1.0])
        for i in range(len(raw_targets)):
            raw_beta = 3.0 * raw_sxl[i] + raw_sxx[i]
            effective = min(raw_beta, float(cfg["action_beta"]))
            w = project_legal_weights(raw_sxl[i], raw_sxx[i], effective,
                                      minimum_cash_weight=float(cfg["minimum_cash_weight"]),
                                      maximum_soxl_weight=float(cfg["maximum_soxl_weight"]))
            ar = np.array([soxl_ret[i], soxx_ret[i], 0.0])
            drifted = held * (1.0 + ar)
            drifted = drifted / drifted.sum()
            fee = float(np.abs(w - drifted).sum()) * float(cost_bps) / 1e4
            rets[i] = float(w @ ar) - fee
            held = w
        growth = np.cumprod(1.0 + rets)
        benchmark = np.cumprod(1.0 + soxx_ret)
        std = rets.std(ddof=1)
        rows.append({
            "variant": name, "cost_bps": cost_bps, **cfg,
            "strategy_return_pct": 100.0 * float(growth[-1] - 1.0),
            "benchmark_return_pct": 100.0 * float(benchmark[-1] - 1.0),
            "strategy_sharpe": float(rets.mean() / std * np.sqrt(ANNUALIZATION_SESSIONS)) if std > 0 else 0.0,
            "strategy_max_drawdown_pct": 100.0 * float((growth / np.maximum.accumulate(growth) - 1.0).min()),
            "beats_benchmark_return": bool(growth[-1] > benchmark[-1]),
        })
    return pd.DataFrame(rows)


def diagnose_cash_source(*, raw_weights: pd.DataFrame, legal_weights: pd.DataFrame,
                         overlay_cap: float = 0.0) -> pd.DataFrame:
    """Compare the raw policy target with the legal (capped) target per session.

    A large gap between raw and legal cash means a hard constraint is removing
    exposure. A small gap means the policy itself chose the cash.
    """
    columns = [f"{asset}_weight" for asset in ASSETS]
    joined = raw_weights[[f"raw_{column}" for column in columns]].join(
        legal_weights[columns], how="inner")
    rows = pd.DataFrame({
        "raw_cash": joined["raw_cash_weight"],
        "legal_cash": joined["cash_weight"],
        "cap_removed_cash": joined["cash_weight"] - joined["raw_cash_weight"],
        "raw_soxl": joined["raw_soxl_weight"],
        "legal_soxl": joined["soxl_weight"],
    })
    return rows
