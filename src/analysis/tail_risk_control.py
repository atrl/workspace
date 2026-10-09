"""Bounded tail-risk control for a frozen allocation, registered as risk control.

The Stage1 production candidate is blocked by a single risk gate: its worst
rolling 5-day loss is -12.11% against a -12.00% floor. This module applies a
*risk control* -- a bounded, causal de-risking that trims exposure after a
recent loss -- to bring the worst 5-day loss inside the floor without changing
the alpha signal.

This is deliberately NOT an alpha claim: the control can only reduce exposure,
never increase it, and the caller must report both the risk improvement and the
return give-up. It is evaluated on development and then read once on outer.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

ANNUALIZATION_SESSIONS = 252
SCOPE = "risk_control_registration_not_alpha"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class TailControl:
    """Trim the effective beta target after a recent portfolio loss.

    ``trigger``: trailing ``lookback``-session compounded return at or below
    which the control engages. ``trim``: multiplicative factor applied to the
    effective beta while engaged. ``release``: trailing return at or above which
    the control releases. ``trim=1.0`` disables the control.
    """

    name: str
    trigger: float = -0.08
    release: float = -0.02
    lookback: int = 5
    trim: float = 0.5

    def validate(self):
        for value in (self.trigger, self.release, self.trim):
            _require(isinstance(value, Real) and np.isfinite(float(value)), "control params must be finite")
        _require(isinstance(self.lookback, int) and self.lookback > 1, "lookback must be > 1")
        _require(0.0 < float(self.trim) <= 1.0, "trim must be in (0, 1]")
        _require(float(self.trigger) <= float(self.release), "trigger must be <= release")
        return self


def apply_control(raw_beta: np.ndarray, strategy_returns: np.ndarray, control: TailControl) -> np.ndarray:
    """Return per-session effective-beta ceilings after applying the control.

    The control reads only realized strategy returns through session ``t-1``, so
    it is causal. When engaged it scales the raw beta by ``trim``; otherwise it
    passes the raw beta through unchanged. It never raises beta.
    """
    control.validate()
    raw = np.asarray(raw_beta, dtype=float)
    rets = np.asarray(strategy_returns, dtype=float)
    _require(len(raw) == len(rets), "raw beta and returns must align")
    caps = raw.copy()
    engaged = False
    for i in range(len(raw)):
        if i >= control.lookback:
            trailing = float(np.prod(1.0 + rets[i - control.lookback:i]) - 1.0)
            if not engaged and trailing <= control.trigger:
                engaged = True
            elif engaged and trailing >= control.release:
                engaged = False
        if engaged:
            caps[i] = raw[i] * float(control.trim)
    return caps


def replay(raw_targets: pd.DataFrame, asset_returns: pd.DataFrame, controls: dict[str, TailControl],
           *, maximum_soxl_weight: float = 0.30, minimum_cash_weight: float = 0.15,
           cost_bps: float = 10.0) -> pd.DataFrame:
    """Replay each control over the frozen raw target and report gate metrics."""
    from src.workflow.intraday_recovery_direct_allocation import project_legal_weights

    raw_sxl = raw_targets["soxl_weight"].to_numpy(float)
    raw_sxx = raw_targets["soxx_weight"].to_numpy(float)
    soxl_ret = asset_returns["soxl"].to_numpy(float)
    soxx_ret = asset_returns["soxx"].to_numpy(float)
    raw_beta = 3.0 * raw_sxl + raw_sxx
    rows = []
    for name, control in controls.items():
        control.validate()
        # First pass: an un-controlled run provides the feedback returns.
        base_rets, _, _ = _one_pass(raw_beta, raw_sxl, raw_sxx, soxl_ret, soxx_ret,
                                    maximum_soxl_weight, minimum_cash_weight, cost_bps)
        caps = apply_control(raw_beta, base_rets, control)
        rets, dd, exposure = _one_pass(caps, raw_sxl, raw_sxx, soxl_ret, soxx_ret,
                                       maximum_soxl_weight, minimum_cash_weight, cost_bps)
        rows.append(_metrics(name, rets, soxx_ret, dd, exposure, cost_bps))
    return pd.DataFrame(rows)


def _one_pass(caps, raw_sxl, raw_sxx, soxl_ret, soxx_ret, msw, mcw, cost_bps):
    from src.workflow.intraday_recovery_direct_allocation import project_legal_weights

    rets = np.empty(len(raw_sxl))
    held = np.array([0.0, 0.0, 1.0])
    exposure = np.empty(len(raw_sxl))
    for i in range(len(raw_sxl)):
        raw_beta = 3.0 * raw_sxl[i] + raw_sxx[i]
        w = project_legal_weights(raw_sxl[i], raw_sxx[i], min(raw_beta, float(caps[i])),
                                  minimum_cash_weight=mcw, maximum_soxl_weight=msw)
        ar = np.array([soxl_ret[i], soxx_ret[i], 0.0])
        drifted = held * (1.0 + ar)
        drifted = drifted / drifted.sum()
        fee = float(np.abs(w - drifted).sum()) * float(cost_bps) / 1e4
        rets[i] = float(w @ ar) - fee
        exposure[i] = float(w[0] + w[1])
        held = w
    growth = np.cumprod(1.0 + rets)
    dd = float((growth / np.maximum.accumulate(growth) - 1.0).min())
    return rets, dd, exposure


def _metrics(name, rets, soxx_ret, dd, exposure, cost_bps) -> dict:
    growth = np.cumprod(1.0 + rets)
    bench = np.cumprod(1.0 + soxx_ret)
    tail = np.sort(rets)[:max(1, int(np.ceil(0.05 * len(rets))))]
    rolling5 = pd.Series(rets).rolling(5).apply(lambda x: float(np.prod(1.0 + x) - 1.0), raw=True)
    return {
        "control": name, "cost_bps": cost_bps, "sessions": int(len(rets)),
        "return_pct": 100.0 * float(growth[-1] - 1.0),
        "benchmark_return_pct": 100.0 * float(bench[-1] - 1.0),
        "max_drawdown_pct": 100.0 * dd,
        "worst_day_pct": 100.0 * float(rets.min()),
        "cvar5_pct": 100.0 * float(tail.mean()),
        "worst_rolling_5d_pct": 100.0 * float(rolling5.min()),
        "average_exposure": float(exposure.mean()),
        "scope": SCOPE,
    }


def gate_pass(metrics: dict, *, floors=None) -> dict:
    floors = floors or {"max_drawdown_pct": -30.0, "worst_day_pct": -12.0,
                        "cvar5_pct": -8.0, "worst_rolling_5d_pct": -12.0}
    checks = {key: bool(metrics[key] >= floor) for key, floor in floors.items()}
    return {"checks": checks, "pass": all(checks.values()), "floors": floors}
