from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .quantbench_screen import QuantBenchGate, screen_quantbench_candidates, summary_record_from_payload
from .quantbench_supervised import (
    RiskAwarePositionConfig,
    fit_quantbench_supervised_models,
    prediction_rank_position_series,
    quantbench_risk_overlay_configs,
    risk_aware_position_series,
)


DEFAULT_WINDOWS: dict[str, tuple[str, str]] = {
    "test_2020": ("2020-01-01", "2020-12-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
}


@dataclass(frozen=True)
class QuantBenchPathSelectorConfig:
    """Macro-risk path selector for QuantBench-style ensemble robustness checks."""

    name: str
    defensive_path: str
    offensive_path: str
    neutral_path: str | None = None
    recovery_path: str | None = None
    stress_floor: float = 0.10
    stress_cap: float = 0.20
    neutral_floor: float = 0.0
    neutral_cap: float = 0.45
    bull_floor: float = 0.55
    bull_cap: float = 0.95
    recovery_floor: float = 0.30
    recovery_cap: float = 0.65
    stress_risk_threshold: float = 0.68
    stress_drop_threshold: float = -0.20
    stress_drawdown_threshold: float = -0.25
    bull_risk_threshold: float = 0.35
    bull_momentum_threshold: float = 0.04
    bull_trend_threshold: float = 0.02
    bull_drawdown_floor: float = -0.08
    recovery_risk_threshold: float = 0.45
    recovery_momentum_threshold: float = 0.06
    recovery_trend_threshold: float = 0.00
    recovery_drawdown_min: float = -0.25
    recovery_drawdown_max: float = -0.04
    portfolio_drawdown_trigger: float = 0.0
    portfolio_drawdown_cap: float = 0.08
    portfolio_drawdown_release: float = 0.06
    portfolio_reentry_floor: float = 0.0
    portfolio_reentry_cap: float = 0.55


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _as_position_series(series: pd.Series, name: str) -> pd.Series:
    out = series.copy()
    out.index = pd.to_datetime(out.index)
    out = out.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise ValueError(f"path `{name}` has no finite positions")
    return out.clip(0.0, 1.0).sort_index()


def _path_position_frame(paths: Mapping[str, pd.Series]) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one path is required")
    frame = pd.concat(
        {
            name: _as_position_series(series, name)
            for name, series in paths.items()
        },
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        raise ValueError("paths have no overlapping finite dates")
    return frame.sort_index()


def _risk_column(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return frame[column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(default)


def _require_path(paths: pd.DataFrame, name: str | None, role: str) -> str:
    if name is None:
        raise ValueError(f"{role} path is required")
    if name not in paths.columns:
        raise ValueError(f"{role} path `{name}` is missing from paths")
    return name


def select_quantbench_path_ensemble(
    paths: Mapping[str, pd.Series],
    risk_frame: pd.DataFrame,
    config: QuantBenchPathSelectorConfig,
) -> pd.DataFrame:
    """Select among daily position paths using laggable macro-risk state."""
    path_frame = _path_position_frame(paths)
    risk = risk_frame.copy()
    risk.index = pd.to_datetime(risk.index)
    risk = risk.sort_index().reindex(path_frame.index)

    defensive = _require_path(path_frame, config.defensive_path, "defensive")
    offensive = _require_path(path_frame, config.offensive_path, "offensive")
    neutral = config.neutral_path or defensive
    recovery = config.recovery_path or offensive
    _require_path(path_frame, neutral, "neutral")
    _require_path(path_frame, recovery, "recovery")

    regime_risk = _risk_column(risk, "regime_risk", 0.0)
    cum_drop_30 = _risk_column(risk, "cum_drop_30", 0.0)
    drawdown_63 = _risk_column(risk, "drawdown_63", 0.0)
    mom_21 = _risk_column(risk, "mom_21", 0.0)
    trend_gap_63 = _risk_column(risk, "trend_gap_63", 0.0)

    stress_mask = (
        (regime_risk >= float(config.stress_risk_threshold))
        | (cum_drop_30 <= float(config.stress_drop_threshold))
        | (drawdown_63 <= float(config.stress_drawdown_threshold))
    )
    bull_mask = (
        ~stress_mask
        & (regime_risk <= float(config.bull_risk_threshold))
        & (mom_21 >= float(config.bull_momentum_threshold))
        & (trend_gap_63 >= float(config.bull_trend_threshold))
        & (drawdown_63 >= float(config.bull_drawdown_floor))
    )
    recovery_mask = (
        ~stress_mask
        & ~bull_mask
        & (regime_risk <= float(config.recovery_risk_threshold))
        & (mom_21 >= float(config.recovery_momentum_threshold))
        & (trend_gap_63 >= float(config.recovery_trend_threshold))
        & (drawdown_63 >= float(config.recovery_drawdown_min))
        & (drawdown_63 <= float(config.recovery_drawdown_max))
    )
    selected_path = pd.Series(neutral, index=path_frame.index, dtype=object)
    selected_path.loc[stress_mask] = defensive
    selected_path.loc[bull_mask] = offensive
    selected_path.loc[recovery_mask] = recovery

    macro_regime = pd.Series("neutral", index=path_frame.index, dtype=object)
    macro_regime.loc[stress_mask] = "stress"
    macro_regime.loc[bull_mask] = "bull"
    macro_regime.loc[recovery_mask] = "recovery"

    position = path_frame[neutral].clip(float(config.neutral_floor), float(config.neutral_cap)).copy()
    position.loc[stress_mask] = path_frame.loc[stress_mask, defensive].clip(
        float(config.stress_floor),
        float(config.stress_cap),
    )
    position.loc[bull_mask] = path_frame.loc[bull_mask, offensive].clip(
        float(config.bull_floor),
        float(config.bull_cap),
    )
    position.loc[recovery_mask] = path_frame.loc[recovery_mask, recovery].clip(
        float(config.recovery_floor),
        float(config.recovery_cap),
    )

    out = pd.DataFrame(
        {
            "datetime": path_frame.index,
            "position": position.clip(0.0, 1.0).astype(float),
            "selected_path": selected_path,
            "macro_regime": macro_regime,
            "regime_risk": regime_risk,
            "cum_drop_30": cum_drop_30,
            "drawdown_63": drawdown_63,
            "mom_21": mom_21,
            "trend_gap_63": trend_gap_63,
        },
        index=path_frame.index,
    )
    for name in path_frame.columns:
        out[f"{name}_position"] = path_frame[name].astype(float)
    return out.reset_index(drop=True)


def backtest_position_path(
    close: pd.Series,
    positions: pd.Series,
    *,
    cost: float = 0.0008,
) -> dict[str, float]:
    close = close.astype(float).sort_index()
    position = positions.astype(float).reindex(close.index).ffill().fillna(0.0).clip(0.0, 1.0)
    returns = close.pct_change().fillna(0.0)
    exec_position = position.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = exec_position * returns - turnover * float(cost)
    equity = (1.0 + strategy_returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    buy_hold = close / close.iloc[0]
    return_pct = (float(equity.iloc[-1]) - 1.0) * 100.0
    bnh_return_pct = (float(buy_hold.iloc[-1]) - 1.0) * 100.0
    return {
        "return": round(return_pct, 10),
        "sharpe": (
            float(strategy_returns.mean() / strategy_returns.std(ddof=0) * np.sqrt(252))
            if len(strategy_returns) > 1 and float(strategy_returns.std(ddof=0)) > 0.0
            else 0.0
        ),
        "max_dd": round(abs(float(drawdown.min())) * 100.0, 10),
        "avg_pos": float(exec_position.mean()),
        "buy_and_hold_return": round(bnh_return_pct, 10),
        "relative_return": round(return_pct - bnh_return_pct, 10),
        "turnover": float(turnover.sum()),
        "trades": int((turnover > 1e-8).sum()),
    }


def apply_quantbench_ensemble_drawdown_guard(
    daily: pd.DataFrame,
    close: pd.Series,
    config: QuantBenchPathSelectorConfig,
    *,
    cost: float = 0.0008,
) -> pd.DataFrame:
    """Apply a no-lookahead account-equity drawdown governor to selected paths."""
    out = daily.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)
    out["raw_position"] = out["position"].astype(float).clip(0.0, 1.0)
    if float(config.portfolio_drawdown_trigger) <= 0.0:
        out["portfolio_drawdown_guard"] = False
        out["portfolio_drawdown"] = 0.0
        return out

    price = close.astype(float).copy()
    price.index = pd.to_datetime(price.index)
    returns = price.sort_index().pct_change().fillna(0.0)
    equity = 1.0
    peak = 1.0
    previous_target = 0.0
    guard_active = False
    guarded: list[float] = []
    guard_flags: list[bool] = []
    drawdowns: list[float] = []
    for row in out.itertuples(index=False):
        date = pd.Timestamp(row.datetime)
        day_return = float(returns.loc[date]) if date in returns.index else 0.0
        equity *= 1.0 + previous_target * day_return
        peak = max(peak, equity)
        drawdown = 1.0 - equity / peak if peak > 0.0 else 0.0
        if drawdown >= float(config.portfolio_drawdown_trigger):
            guard_active = True
        elif drawdown <= float(config.portfolio_drawdown_release):
            guard_active = False

        raw_target = float(row.raw_position)
        if guard_active and str(row.macro_regime) in {"bull", "recovery"} and config.portfolio_reentry_floor > 0.0:
            target = min(
                max(raw_target, float(config.portfolio_reentry_floor)),
                float(config.portfolio_reentry_cap),
            )
        elif guard_active:
            target = min(raw_target, float(config.portfolio_drawdown_cap))
        else:
            target = raw_target

        equity -= abs(target - previous_target) * float(cost)
        peak = max(peak, equity)
        guarded.append(float(np.clip(target, 0.0, 1.0)))
        guard_flags.append(bool(guard_active))
        drawdowns.append(float(drawdown))
        previous_target = target

    out["position"] = guarded
    out["portfolio_drawdown_guard"] = guard_flags
    out["portfolio_drawdown"] = drawdowns
    return out


def _payload_for_daily(
    daily: pd.DataFrame,
    close: pd.Series,
    *,
    profile: str,
    config: QuantBenchPathSelectorConfig,
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] | None,
    cost: float,
) -> dict[str, Any]:
    frame = daily.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    positions = pd.Series(frame["position"].to_numpy(dtype=float), index=frame["datetime"])
    close = close.astype(float).sort_index()
    common = close.index.intersection(positions.index)
    if len(common) < 2:
        raise ValueError(f"ensemble `{profile}` has fewer than two overlapping close rows")
    payload = {
        "profile": profile,
        "model_family": "quantbench_path_ensemble",
        "ensemble_config": asdict(config),
        "test_proxy": backtest_position_path(close.loc[common], positions.loc[common], cost=cost),
        "slices": {},
    }
    for name, (start, end) in (windows or DEFAULT_WINDOWS).items():
        window_close = close.loc[start:end]
        window_pos = positions.loc[start:end]
        if len(window_close.index.intersection(window_pos.index)) < 2:
            continue
        payload["slices"][name] = backtest_position_path(window_close, window_pos, cost=cost)
    return payload


def build_quantbench_path_ensemble_payloads(
    close: pd.Series,
    paths: Mapping[str, pd.Series],
    risk_frame: pd.DataFrame,
    *,
    configs: Iterable[QuantBenchPathSelectorConfig],
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] | None = None,
    cost: float = 0.0008,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    payloads: list[dict[str, Any]] = []
    daily_by_profile: dict[str, pd.DataFrame] = {}
    for config in tuple(configs):
        daily = select_quantbench_path_ensemble(paths, risk_frame, config)
        daily = apply_quantbench_ensemble_drawdown_guard(daily, close, config, cost=cost)
        payload = _payload_for_daily(
            daily,
            close,
            profile=config.name,
            config=config,
            windows=windows,
            cost=cost,
        )
        payloads.append(payload)
        daily_by_profile[config.name] = daily
    return payloads, daily_by_profile


def build_quantbench_path_ensemble_report(
    close: pd.Series,
    paths: Mapping[str, pd.Series],
    risk_frame: pd.DataFrame,
    *,
    card_dir: Path,
    root: Path | None = None,
    configs: Iterable[QuantBenchPathSelectorConfig],
    gate: QuantBenchGate = QuantBenchGate(),
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] | None = None,
    cost: float = 0.0008,
) -> dict[str, Any]:
    """Persist a QuantBench-compatible macro path-ensemble screen."""
    root = root.resolve() if root is not None else None
    card_dir = card_dir.resolve()
    card_dir.mkdir(parents=True, exist_ok=True)

    payloads, daily_by_profile = build_quantbench_path_ensemble_payloads(
        close,
        paths,
        risk_frame,
        configs=configs,
        windows=windows,
        cost=cost,
    )
    records = [
        summary_record_from_payload(payload, source_path=f"generated:{payload['profile']}")
        for payload in payloads
    ]
    screen_rows = screen_quantbench_candidates(records, gate=gate)
    eligible_rows = [row for row in screen_rows if row["eligible"]]

    payloads_path = card_dir / "ensemble_payloads.json"
    screen_path = card_dir / "ensemble_candidate_screen.csv"
    summary_path = card_dir / "ensemble_summary.json"
    payloads_path.write_text(json.dumps(payloads, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    _write_rows_csv(screen_path, screen_rows)
    for profile, daily in daily_by_profile.items():
        daily.to_csv(card_dir / f"{profile}_soxl_daily.csv", index=False)

    def rel(path: Path) -> str:
        return _display_path(path, root=root)

    report = {
        "paper": {
            "title": "QuantBench: Benchmarking AI Methods for Quantitative Investment",
            "arxiv_pdf": "https://arxiv.org/pdf/2504.18600",
            "mapped_claim": "seed/model ensemble robustness with macro-risk regime selection",
        },
        "data": {
            "path_count": len(paths),
            "selector_count": len(payloads),
            "start": str(pd.to_datetime(close.index).min().date()),
            "end": str(pd.to_datetime(close.index).max().date()),
        },
        "gate": gate.__dict__,
        "artifacts": {
            "summary_json": rel(summary_path),
            "ensemble_candidate_screen_csv": rel(screen_path),
            "ensemble_payloads_json": rel(payloads_path),
            "daily_csvs": [rel(card_dir / f"{profile}_soxl_daily.csv") for profile in daily_by_profile],
        },
        "screen": {
            "records": len(records),
            "eligible_count": len(eligible_rows),
            "eligible_profiles": [row["profile"] for row in eligible_rows],
            "best_rows": screen_rows[:10],
        },
        "decision": "eligible_ensemble_candidates_found"
        if eligible_rows
        else "no_ensemble_model_satisfies_quantbench_soxl_20dd_gate",
        "next_action": (
            "Promote only an eligible ensemble; otherwise inspect regime audit columns "
            "and adjust reusable path selection logic before another screen."
        ),
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return report


def _daily_index_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    if isinstance(frame.index, pd.MultiIndex) and "instrument" in frame.index.names:
        frame = frame.droplevel("instrument")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _daily_price_series(close: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(close, pd.DataFrame):
        if "price" in close.columns:
            close = close["price"]
        elif "close" in close.columns:
            close = close["close"]
        else:
            close = close.iloc[:, 0]
    series = close.copy()
    if isinstance(series.index, pd.MultiIndex) and "instrument" in series.index.names:
        series = series.droplevel("instrument")
    series.index = pd.to_datetime(series.index)
    return series.astype(float).sort_index()


def _supervised_path_name(profile_prefix: str, model_name: str, risk_config: RiskAwarePositionConfig | None = None) -> str:
    suffix = f"_{risk_config.name}" if risk_config is not None else ""
    return f"{profile_prefix}_{model_name}{suffix}"


def build_supervised_path_library(
    predictions: Mapping[str, pd.Series],
    *,
    risk_frame: pd.DataFrame,
    profile_prefix: str = "quantbench_supervised",
    risk_configs: Iterable[RiskAwarePositionConfig] = (),
    min_position: float = 0.10,
    max_position: float = 0.90,
) -> dict[str, pd.Series]:
    """Create reusable daily position paths from supervised model predictions."""
    paths: dict[str, pd.Series] = {}
    for model_name, prediction in predictions.items():
        paths[_supervised_path_name(profile_prefix, model_name)] = prediction_rank_position_series(
            prediction,
            min_position=min_position,
            max_position=max_position,
        )
        for risk_config in tuple(risk_configs):
            paths[_supervised_path_name(profile_prefix, model_name, risk_config)] = risk_aware_position_series(
                prediction.dropna(),
                risk_frame=risk_frame,
                config=risk_config,
                min_position=min_position,
                max_position=max_position,
            )
    return paths


def default_quantbench_path_selector_configs(
    profile_prefix: str = "quantbench_supervised",
) -> list[QuantBenchPathSelectorConfig]:
    """Return a bounded default screen of defensive/offensive supervised path pairs."""

    def path(model_name: str, overlay: str | None = None) -> str:
        suffix = f"_{overlay}" if overlay else ""
        return f"{profile_prefix}_{model_name}{suffix}"

    return [
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_lowdd_mlp_v1",
            defensive_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            neutral_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            offensive_path=path("mlp_mse"),
            recovery_path=path("mlp_mse"),
            stress_floor=0.10,
            stress_cap=0.16,
            neutral_cap=0.32,
            bull_floor=0.65,
            bull_cap=0.92,
            recovery_floor=0.35,
            recovery_cap=0.60,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_lowdd_ridge_v1",
            defensive_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            neutral_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            offensive_path=path("ridge_mse"),
            recovery_path=path("ridge_mse"),
            stress_floor=0.10,
            stress_cap=0.16,
            neutral_cap=0.32,
            bull_floor=0.60,
            bull_cap=0.90,
            recovery_floor=0.32,
            recovery_cap=0.58,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_extra_logistic_v1",
            defensive_path=path("extra_trees_rank", "macro_budget_lowdd_v1"),
            neutral_path=path("extra_trees_rank", "macro_budget_lowdd_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.18,
            neutral_cap=0.35,
            bull_floor=0.62,
            bull_cap=0.92,
            recovery_floor=0.35,
            recovery_cap=0.62,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_defensive_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.20,
            neutral_cap=0.40,
            bull_floor=0.62,
            bull_cap=0.92,
            recovery_floor=0.35,
            recovery_cap=0.62,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_guarded_rf_logistic_v1",
            defensive_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            neutral_path=path("random_forest_clf", "macro_budget_defensive_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
        ),
        QuantBenchPathSelectorConfig(
            name="quantbench_ensemble_guarded_hist_logistic_v1",
            defensive_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            neutral_path=path("hist_gradient_mse", "macro_budget_lowdd_v1"),
            offensive_path=path("logistic_clf"),
            recovery_path=path("logistic_clf"),
            stress_floor=0.10,
            stress_cap=0.14,
            neutral_cap=0.20,
            bull_floor=0.70,
            bull_cap=0.98,
            recovery_floor=0.50,
            recovery_cap=0.85,
            stress_risk_threshold=0.92,
            stress_drop_threshold=-0.35,
            stress_drawdown_threshold=-0.45,
            bull_risk_threshold=0.50,
            bull_momentum_threshold=0.00,
            bull_trend_threshold=-0.05,
            bull_drawdown_floor=-0.30,
            recovery_risk_threshold=0.75,
            recovery_momentum_threshold=0.00,
            recovery_trend_threshold=-0.08,
            recovery_drawdown_min=-0.45,
            recovery_drawdown_max=-0.02,
            portfolio_drawdown_trigger=0.15,
            portfolio_drawdown_cap=0.08,
            portfolio_drawdown_release=0.05,
            portfolio_reentry_floor=0.40,
            portfolio_reentry_cap=0.65,
        ),
    ]


def _config_paths_available(config: QuantBenchPathSelectorConfig, paths: Mapping[str, pd.Series]) -> bool:
    required = {
        config.defensive_path,
        config.offensive_path,
        config.neutral_path or config.defensive_path,
        config.recovery_path or config.offensive_path,
    }
    return required <= set(paths)


def build_quantbench_supervised_ensemble_report(
    features: pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    *,
    card_dir: Path,
    root: Path | None = None,
    horizon: int = 20,
    model_names: Iterable[str] | None = None,
    feature_packs: Iterable[str] = (),
    train_start: str | pd.Timestamp = "2010-03-11",
    train_end: str | pd.Timestamp = "2018-12-31",
    test_start: str | pd.Timestamp = "2020-01-01",
    test_end: str | pd.Timestamp = "2026-12-31",
    profile_prefix: str = "quantbench_supervised",
    risk_configs: Iterable[RiskAwarePositionConfig] | None = None,
    ensemble_configs: Iterable[QuantBenchPathSelectorConfig] | None = None,
    gate: QuantBenchGate = QuantBenchGate(),
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]] | None = None,
    min_position: float = 0.10,
    max_position: float = 0.90,
    cost: float = 0.0008,
) -> dict[str, Any]:
    """Train supervised paths and run the src-owned macro path ensemble screen."""
    root = root.resolve() if root is not None else None
    card_dir = card_dir.resolve()
    card_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = _daily_index_frame(features).replace([np.inf, -np.inf], np.nan)
    close_series = _daily_price_series(close)
    common_index = feature_frame.index.intersection(close_series.index)
    feature_frame = feature_frame.loc[common_index]
    close_series = close_series.loc[common_index]
    if feature_frame.empty or close_series.empty:
        raise ValueError("features and close must have overlapping rows")

    from .quantbench_objectives import build_soxl_supervised_targets

    targets = build_soxl_supervised_targets(close_series, horizons=(horizon,))
    predictions, objective_metrics = fit_quantbench_supervised_models(
        feature_frame,
        targets,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        horizon=horizon,
        model_names=model_names,
    )
    risk_config_tuple = tuple(quantbench_risk_overlay_configs() if risk_configs is None else risk_configs)
    paths = build_supervised_path_library(
        predictions,
        risk_frame=feature_frame,
        profile_prefix=profile_prefix,
        risk_configs=risk_config_tuple,
        min_position=min_position,
        max_position=max_position,
    )
    config_tuple = tuple(
        default_quantbench_path_selector_configs(profile_prefix=profile_prefix)
        if ensemble_configs is None
        else ensemble_configs
    )
    config_tuple = tuple(config for config in config_tuple if _config_paths_available(config, paths))
    if not config_tuple:
        raise ValueError("no ensemble selector configs match the generated supervised path library")

    report = build_quantbench_path_ensemble_report(
        close_series,
        paths,
        feature_frame,
        card_dir=card_dir,
        root=root,
        configs=config_tuple,
        gate=gate,
        windows=windows,
        cost=cost,
    )
    path_rows = []
    for name in sorted(paths):
        parts = name.removeprefix(f"{profile_prefix}_").split("_macro_budget_", maxsplit=1)
        path_rows.append(
            {
                "path": name,
                "model_name": parts[0],
                "risk_overlay": f"macro_budget_{parts[1]}" if len(parts) == 2 else "",
            }
        )
    path_library_path = card_dir / "ensemble_path_library.csv"
    objective_metrics_path = card_dir / "ensemble_objective_metrics.json"
    _write_rows_csv(path_library_path, path_rows)
    objective_metrics_path.write_text(
        json.dumps(objective_metrics, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    def rel(path: Path) -> str:
        return _display_path(path, root=root)

    report["data"].update(
        {
            "feature_count": int(feature_frame.shape[1]),
            "feature_packs": list(feature_packs),
            "model_names": list(predictions.keys()),
            "risk_overlays": [config.name for config in risk_config_tuple],
            "horizon": int(horizon),
            "train_window": [str(pd.Timestamp(train_start).date()), str(pd.Timestamp(train_end).date())],
            "test_window": [str(pd.Timestamp(test_start).date()), str(pd.Timestamp(test_end).date())],
        }
    )
    report["artifacts"]["ensemble_path_library_csv"] = rel(path_library_path)
    report["artifacts"]["ensemble_objective_metrics_json"] = rel(objective_metrics_path)
    summary_path = card_dir / "ensemble_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DEFAULT_WINDOWS",
    "QuantBenchPathSelectorConfig",
    "apply_quantbench_ensemble_drawdown_guard",
    "backtest_position_path",
    "build_quantbench_supervised_ensemble_report",
    "build_quantbench_path_ensemble_payloads",
    "build_quantbench_path_ensemble_report",
    "build_supervised_path_library",
    "default_quantbench_path_selector_configs",
    "select_quantbench_path_ensemble",
]
