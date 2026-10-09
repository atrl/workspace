"""Marginal-information diagnosis for the Stage1 action layer.

The active action layer decides, on each open slot, whether to execute the base
allocation target or to hold. Its own training label (``execute_advantage``) has a
near-zero *unconditional* mean, which says nothing about ``E[y | X]``. This module
measures the *conditional* information and the real net contribution, without
training any new model and without touching the frozen recorders.

Everything is read-only. Missing identity is reported as ``needs_data`` and never
fabricated. A counterfactual here is a one-session + next-reset proxy, not full
strategy value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION_SESSIONS = 252
SCOPE = "diagnostic_only_not_strategy_or_promotion_evidence"
MIN_DECIDABLE = 200


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


# ---------------------------------------------------------------------------
# Conditional-information metrics
# ---------------------------------------------------------------------------

def rank_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation between a prediction and a realized target."""
    p = np.asarray(prediction, dtype=float)
    y = np.asarray(target, dtype=float)
    _require(len(p) == len(y) and len(p) > 2, "prediction and target must align and be non-trivial")
    if np.ptp(p) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(pd.Series(p).corr(pd.Series(y), method="spearman"))


def auc(prediction: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney statistic."""
    p = np.asarray(prediction, dtype=float)
    pos = np.asarray(positive, dtype=bool)
    _require(len(p) == len(pos), "prediction and label must align")
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy(float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration_table(prediction: np.ndarray, target: np.ndarray, *, bins: int = 5) -> pd.DataFrame:
    """Mean realized advantage per predicted-probability quintile."""
    frame = pd.DataFrame({"p": np.asarray(prediction, float), "y": np.asarray(target, float)}).dropna()
    _require(len(frame) >= bins, "not enough rows for calibration bins")
    frame["bucket"] = pd.qcut(frame["p"], bins, labels=False, duplicates="drop")
    out = frame.groupby("bucket").agg(count=("y", "size"), mean_probability=("p", "mean"),
                                      mean_advantage=("y", "mean"))
    return out.reset_index()


def block_bootstrap_ci(values: np.ndarray, *, block: int = 20, resamples: int = 2000,
                       seed: int = 20261009, alpha: float = 0.05) -> dict:
    """Moving-block bootstrap CI of the mean of a serially dependent series."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    _require(v.size > block, "series must exceed the block length")
    rng = np.random.default_rng(seed)
    n = v.size
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(resamples, n_blocks))
    means = np.empty(resamples)
    for i in range(resamples):
        idx = (starts[i][:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[i] = v[idx].mean()
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return {
        "mean": float(v.mean()), "lower": lower, "upper": upper,
        "block": int(block), "resamples": int(resamples),
        "contains_zero": bool(lower <= 0.0 <= upper),
        "interpretation": "CI crossing zero means no evidence found, not proof of no information",
    }


# ---------------------------------------------------------------------------
# Contribution accounting
# ---------------------------------------------------------------------------

def period_metrics(returns: np.ndarray) -> dict:
    r = np.asarray(returns, dtype=float)
    _require(r.size > 0 and np.isfinite(r).all(), "returns must be finite and nonempty")
    growth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(growth)
    tail = np.sort(r)[:max(1, int(np.ceil(0.05 * r.size)))]
    std = r.std(ddof=1)
    return {
        "sessions": int(r.size),
        "return_pct": 100.0 * float(growth[-1] - 1.0),
        "max_drawdown_pct": 100.0 * float((growth / peak - 1.0).min()),
        "cvar5_pct": 100.0 * float(tail.mean()),
        "sharpe": float(r.mean() / std * np.sqrt(ANNUALIZATION_SESSIONS)) if std > 0 else 0.0,
    }


def worst_segments(action: pd.Series, no_hold: pd.Series, *, by: str = "year") -> pd.DataFrame:
    """Report the periods where the action path's drawdown/return worsened.

    ``action`` and ``no_hold`` are aligned per-session strategy returns.
    """
    frame = pd.DataFrame({"action": action, "no_hold": no_hold}).dropna()
    keys = pd.to_datetime(frame.index).year
    frame["key"] = keys
    rows = []
    for key, group in frame.groupby("key"):
        a = period_metrics(group["action"].to_numpy())
        n = period_metrics(group["no_hold"].to_numpy())
        rows.append({
            "period": int(key),
            "action_return_pct": a["return_pct"], "no_hold_return_pct": n["return_pct"],
            "action_max_drawdown_pct": a["max_drawdown_pct"],
            "no_hold_max_drawdown_pct": n["max_drawdown_pct"],
            "action_cvar5_pct": a["cvar5_pct"], "no_hold_cvar5_pct": n["cvar5_pct"],
            "return_delta_pct": a["return_pct"] - n["return_pct"],
            "max_drawdown_delta_pct": a["max_drawdown_pct"] - n["max_drawdown_pct"],
            "cvar5_delta_pct": a["cvar5_pct"] - n["cvar5_pct"],
        })
    return pd.DataFrame(rows)


def gate_checks(metrics: dict, floors: dict) -> dict:
    checks = {k: bool(np.isfinite(metrics[k]) and metrics[k] >= floor) for k, floor in floors.items()}
    return {"checks": checks, "pass": all(checks.values())}
