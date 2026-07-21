from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "leveraged_etf_paper_replay_v1"
CARD_TYPE = "leveraged_etf_paper_strategy_replay_card"
OUTER_YEARS = (2023, 2024, 2025, 2026)
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
)


@dataclass(frozen=True)
class LeveragedEtfPaperReplaySpec:
    root: Path
    experiment: str = EXPERIMENT
    start: str = "2010-03-11"
    signal_start: str = "2011-01-03"
    inverse_variance_calibration_end: str = "2018-12-31"
    outer_years: tuple[int, ...] = OUTER_YEARS
    costs_bps: tuple[int, ...] = (0, 10, 25)
    refresh_provider: str = "yahoo_api"
    futu_refresh_status: str = "needs_data_opend_connection_refused"

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment


def run_leveraged_etf_paper_replay_card(*, root: Path | str | None = None) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    return _run_card(LeveragedEtfPaperReplaySpec(root=repo_root))


def _load_market_frame(spec: LeveragedEtfPaperReplaySpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = spec.provider_uri / "qlib_store_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    as_of = str(manifest["as_of"])
    raw = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.provider_uri),
        instruments=["SOXL", "SOXX"],
        fields=["$open", "$close"],
        start=spec.start,
        end=as_of,
    ).reset_index()
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["instrument"] = raw["instrument"].astype(str).str.upper()
    wide = raw.pivot(index="datetime", columns="instrument", values=["$open", "$close"]).sort_index()
    frame = pd.DataFrame(index=wide.index)
    for symbol in ("SOXL", "SOXX"):
        frame[f"{symbol.lower()}_open"] = pd.to_numeric(wide[("$open", symbol)], errors="coerce")
        frame[f"{symbol.lower()}_close"] = pd.to_numeric(wide[("$close", symbol)], errors="coerce")
    frame = frame.reset_index().rename(columns={"datetime": "signal_date"})
    frame["trade_date"] = frame["signal_date"].shift(-1)
    frame["exit_date"] = frame["signal_date"].shift(-2)
    frame["soxl_return"] = frame["soxl_open"].shift(-2) / frame["soxl_open"].shift(-1) - 1.0
    frame["soxx_return"] = frame["soxx_open"].shift(-2) / frame["soxx_open"].shift(-1) - 1.0
    frame["soxx_close_return"] = frame["soxx_close"].pct_change()
    frame = frame.loc[
        frame["signal_date"].ge(pd.Timestamp(spec.signal_start))
        & frame["trade_date"].notna()
        & frame["exit_date"].notna()
        & frame["soxl_return"].notna()
        & frame["soxx_return"].notna()
    ].reset_index(drop=True)
    overlap_audit: dict[str, Any] = {}
    for symbol in ("SOXL", "SOXX"):
        futu_path = spec.root / "data" / "raw" / "external" / f"futu_history_kline_US_{symbol}.csv"
        if not futu_path.exists():
            overlap_audit[symbol] = {"status": "needs_data", "reason": "missing_futu_history_file"}
            continue
        futu = pd.read_csv(futu_path, usecols=["date", "open", "close"])
        futu["date"] = pd.to_datetime(futu["date"])
        qlib_values = frame.loc[:, ["signal_date", f"{symbol.lower()}_open", f"{symbol.lower()}_close"]].rename(
            columns={
                "signal_date": "date",
                f"{symbol.lower()}_open": "qlib_open",
                f"{symbol.lower()}_close": "qlib_close",
            }
        )
        common = futu.merge(qlib_values, on="date", how="inner").sort_values("date").tail(22)
        relative_errors = []
        for field in ("open", "close"):
            denominator = pd.to_numeric(common[field], errors="coerce").abs().replace(0.0, np.nan)
            relative_errors.append(
                ((pd.to_numeric(common[f"qlib_{field}"], errors="coerce") - common[field]).abs() / denominator).max()
            )
        overlap_audit[symbol] = {
            "status": "ok" if not common.empty else "needs_data",
            "rows": int(len(common)),
            "last_common_date": common["date"].max().date().isoformat() if not common.empty else None,
            "max_relative_open_close_difference": float(np.nanmax(relative_errors)) if not common.empty else None,
            "reference": futu_path.resolve().as_posix(),
        }
    audit = {
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "manifest": manifest_path.resolve().as_posix(),
        "as_of": as_of,
        "source_api": ["qlib.init", "qlib.data.D.calendar", "qlib.data.D.features"],
        "fields": ["$open", "$close"],
        "instruments": ["SOXL", "SOXX"],
        "rows": int(len(frame)),
        "signal_start": frame["signal_date"].min().date().isoformat(),
        "last_signal_date": frame["signal_date"].max().date().isoformat(),
        "last_trade_date": frame["trade_date"].max().date().isoformat(),
        "last_exit_date": frame["exit_date"].max().date().isoformat(),
        "timing": "close_t_signal_then_open_t_plus_1_to_open_t_plus_2_return",
        "latest_refresh_provider": spec.refresh_provider,
        "latest_refresh_source_api": ["query1.finance.yahoo.com/v8/finance/chart"],
        "futu_refresh_status": spec.futu_refresh_status,
        "futu_yahoo_overlap_audit": overlap_audit,
    }
    return frame, audit


def _periodic_target_weight(frame: pd.DataFrame, *, target: float, period: str) -> np.ndarray:
    dates = pd.to_datetime(frame["trade_date"])
    if period == "day":
        keys: Iterable[Any] = range(len(frame))
    elif period == "week":
        iso = dates.dt.isocalendar()
        keys = list(zip(iso["year"].astype(int), iso["week"].astype(int)))
    else:
        raise ValueError(f"unsupported rebalance period: {period}")
    returns = frame["soxl_return"].to_numpy(dtype=float)
    weights = np.zeros(len(frame), dtype=float)
    previous_key: Any = None
    carried = 0.0
    for idx, key in enumerate(keys):
        if idx == 0 or key != previous_key:
            carried = float(target)
        weights[idx] = carried
        gross = 1.0 + carried * returns[idx]
        carried = carried * (1.0 + returns[idx]) / gross if gross > 0 else 0.0
        previous_key = key
    return weights


def _inverse_variance_weights(frame: pd.DataFrame, spec: LeveragedEtfPaperReplaySpec) -> tuple[np.ndarray, dict[str, Any]]:
    monthly = frame.loc[:, ["signal_date", "soxx_close_return"]].copy()
    monthly["month"] = monthly["signal_date"].dt.to_period("M")
    variance = monthly.groupby("month")["soxx_close_return"].var(ddof=1).sort_index()
    lagged = variance.shift(1)
    calibration = lagged.loc[lagged.index.to_timestamp() <= pd.Timestamp(spec.inverse_variance_calibration_end)].dropna()
    normalizer = float(calibration.median() / 3.0)
    month_weight = (normalizer / lagged).clip(lower=0.0, upper=1.0)
    weights = monthly["month"].map(month_weight).fillna(0.0).to_numpy(dtype=float)
    return weights, {
        "formula": "w_t=clip(c/previous_month_realized_variance,0,1)",
        "normalizer": normalizer,
        "normalizer_policy": "one_third_times_median_lagged_variance",
        "calibration_end": spec.inverse_variance_calibration_end,
        "selection_uses_outer_test": False,
    }


def build_paper_strategy_weights(
    frame: pd.DataFrame,
    spec: LeveragedEtfPaperReplaySpec,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    soxx_ma200 = frame["soxx_close"].rolling(200, min_periods=200).mean()
    trend = frame["soxx_close"].gt(soxx_ma200).astype(float).to_numpy()
    inv_var, inv_var_audit = _inverse_variance_weights(frame, spec)
    zeros = np.zeros(len(frame), dtype=float)
    strategies = {
        "soxl_buy_hold": (np.ones(len(frame), dtype=float), zeros.copy()),
        "soxx_buy_hold": (zeros.copy(), np.ones(len(frame), dtype=float)),
        "avellaneda_daily_1x_replication": (np.full(len(frame), 1.0 / 3.0), zeros.copy()),
        "avellaneda_weekly_1x_replication": (
            _periodic_target_weight(frame, target=1.0 / 3.0, period="week"),
            zeros.copy(),
        ),
        "gayed_ma200_2x_equivalent": ((2.0 / 3.0) * trend, zeros.copy()),
        "gayed_ma200_3x": (trend, zeros.copy()),
        "moreira_muir_lagged_inverse_variance": (inv_var, zeros.copy()),
    }
    return strategies, {"inverse_variance": inv_var_audit}


def _simulate_weights(
    frame: pd.DataFrame,
    *,
    soxl_weight: np.ndarray,
    soxx_weight: np.ndarray,
    cost_bps: int,
) -> pd.DataFrame:
    result = frame.loc[:, ["signal_date", "trade_date", "exit_date", "soxl_return", "soxx_return"]].copy()
    result["soxl_weight"] = np.asarray(soxl_weight, dtype=float)
    result["soxx_weight"] = np.asarray(soxx_weight, dtype=float)
    result["cash_weight"] = 1.0 - result["soxl_weight"] - result["soxx_weight"]
    result["turnover"] = (
        result["soxl_weight"].diff().abs().fillna(result["soxl_weight"].abs())
        + result["soxx_weight"].diff().abs().fillna(result["soxx_weight"].abs())
    )
    result["gross_return"] = (
        result["soxl_weight"] * result["soxl_return"]
        + result["soxx_weight"] * result["soxx_return"]
    )
    result["cost"] = result["turnover"] * float(cost_bps) / 10_000.0
    result["strategy_return"] = result["gross_return"] - result["cost"]
    return result


def _simulate_george_trainor_cppi(frame: pd.DataFrame, *, cost_bps: int) -> pd.DataFrame:
    result = frame.loc[:, ["signal_date", "trade_date", "exit_date", "soxl_return", "soxx_return"]].copy()
    wealth = 1.0
    floor = 0.9
    last_rebalance_close = np.nan
    current_weight = 0.0
    previous_weight = 0.0
    current_year: int | None = None
    rows: list[dict[str, float]] = []
    for row in result.itertuples(index=False):
        year = int(pd.Timestamp(row.trade_date).year)
        annual_reset = current_year is None or year != current_year
        if annual_reset:
            floor = 0.9 * wealth
            last_rebalance_close = float(frame.loc[len(rows), "soxx_close"])
            current_year = year
        current_close = float(frame.loc[len(rows), "soxx_close"])
        barrier_hit = (
            not np.isfinite(last_rebalance_close)
            or abs(current_close / last_rebalance_close - 1.0) >= 0.025
        )
        if annual_reset or barrier_hit:
            cushion = max(wealth - floor, 0.0)
            current_weight = min(max((5.0 / 3.0) * cushion / wealth, 0.0), 1.0 / 3.0) if wealth > 0 else 0.0
            last_rebalance_close = current_close
        turnover = abs(current_weight - previous_weight)
        gross_return = current_weight * float(row.soxl_return)
        cost = turnover * float(cost_bps) / 10_000.0
        strategy_return = gross_return - cost
        wealth *= 1.0 + strategy_return
        rows.append(
            {
                "soxl_weight": current_weight,
                "soxx_weight": 0.0,
                "cash_weight": 1.0 - current_weight,
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
                "strategy_return": strategy_return,
            }
        )
        previous_weight = current_weight
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _risk_metrics(result: pd.DataFrame) -> dict[str, float]:
    returns = pd.to_numeric(result["strategy_return"], errors="coerce").dropna()
    if returns.empty:
        raise ValueError("cannot calculate metrics for empty return series")
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    years = len(returns) / 252.0
    total_return = float(wealth.iloc[-1] - 1.0)
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else -1.0
    quantile = float(returns.quantile(0.05))
    cvar = float(returns.loc[returns.le(quantile)].mean())
    rolling_5 = (1.0 + returns).rolling(5).apply(np.prod, raw=True) - 1.0
    soxl = pd.to_numeric(result.loc[returns.index, "soxl_return"], errors="coerce")
    up_mask = soxl.gt(0.0)
    upside_capture = float(returns.loc[up_mask].sum() / soxl.loc[up_mask].sum()) if up_mask.any() else np.nan
    max_drawdown = float(drawdown.min())
    from qlib.contrib.evaluate import risk_analysis

    qlib_risk = risk_analysis(returns, N=252, freq=None, mode="product")["risk"]
    return {
        "return_pct": 100.0 * total_return,
        "cagr_pct": 100.0 * cagr,
        "annualized_vol_pct": 100.0 * float(returns.std(ddof=1) * np.sqrt(252.0)),
        "max_drawdown_pct": 100.0 * max_drawdown,
        "worst_day_pct": 100.0 * float(returns.min()),
        "cvar5_pct": 100.0 * cvar,
        "worst_rolling_5d_pct": 100.0 * float(rolling_5.min()),
        "avg_soxl_position_pct": 100.0 * float(result.loc[returns.index, "soxl_weight"].mean()),
        "avg_equity_position_pct": 100.0
        * float((result.loc[returns.index, "soxl_weight"] + result.loc[returns.index, "soxx_weight"]).mean()),
        "avg_effective_beta": float(
            (3.0 * result.loc[returns.index, "soxl_weight"] + result.loc[returns.index, "soxx_weight"]).mean()
        ),
        "upside_capture_pct": 100.0 * upside_capture,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "total_turnover": float(result.loc[returns.index, "turnover"].sum()),
        "qlib_annualized_return_pct": 100.0 * float(qlib_risk["annualized_return"]),
        "qlib_information_ratio": float(qlib_risk["information_ratio"]),
        "qlib_max_drawdown_pct": 100.0 * float(qlib_risk["max_drawdown"]),
        "rows": int(len(returns)),
    }


def _risk_gate(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["max_drawdown_pct"] >= -20.0
        and metrics["worst_day_pct"] >= -12.0
        and metrics["cvar5_pct"] >= -8.0
        and metrics["worst_rolling_5d_pct"] >= -12.0
    )


def _matched_beta_weights(beta: float, rows: int) -> tuple[np.ndarray, np.ndarray]:
    beta = float(np.clip(beta, 0.0, 3.0))
    if beta <= 1.0:
        return np.zeros(rows), np.full(rows, beta)
    soxl = (beta - 1.0) / 2.0
    return np.full(rows, soxl), np.full(rows, 1.0 - soxl)


def _evaluate(
    frame: pd.DataFrame,
    strategies: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    costs_bps: tuple[int, ...],
    outer_years: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_mask = frame["trade_date"].dt.year.isin(outer_years).to_numpy()
    outer = frame.loc[outer_mask].reset_index(drop=True)
    positions: list[pd.DataFrame] = []
    returns_rows: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for cost_bps in costs_bps:
        for strategy, (soxl_weight, soxx_weight) in strategies.items():
            result = _simulate_weights(
                outer,
                soxl_weight=np.asarray(soxl_weight)[outer_mask],
                soxx_weight=np.asarray(soxx_weight)[outer_mask],
                cost_bps=cost_bps,
            )
            _append_evaluation_rows(
                strategy,
                cost_bps,
                result,
                comparison_rows,
                annual_rows,
                positions,
                returns_rows,
                outer_years,
            )
        cppi = _simulate_george_trainor_cppi(outer, cost_bps=cost_bps)
        _append_evaluation_rows(
            "george_trainor_cppi_3x_2p5pct",
            cost_bps,
            cppi,
            comparison_rows,
            annual_rows,
            positions,
            returns_rows,
            outer_years,
        )
    comparison = pd.DataFrame(comparison_rows)
    annual = pd.DataFrame(annual_rows)
    position_frame = pd.concat(positions, ignore_index=True)
    return_frame = pd.concat(returns_rows, ignore_index=True)

    benchmark_rows: list[dict[str, Any]] = []
    for row in comparison.itertuples(index=False):
        strategy_result = return_frame.loc[
            return_frame["strategy"].eq(row.strategy) & return_frame["cost_bps"].eq(row.cost_bps)
        ].reset_index(drop=True)
        avg_soxl = float(strategy_result["soxl_weight"].mean())
        same = _simulate_weights(
            outer,
            soxl_weight=np.full(len(outer), avg_soxl),
            soxx_weight=np.zeros(len(outer)),
            cost_bps=int(row.cost_bps),
        )
        same_metrics = _risk_metrics(same)
        matched_soxl, matched_soxx = _matched_beta_weights(float(row.avg_effective_beta), len(outer))
        matched = _simulate_weights(
            outer,
            soxl_weight=matched_soxl,
            soxx_weight=matched_soxx,
            cost_bps=int(row.cost_bps),
        )
        matched_metrics = _risk_metrics(matched)
        benchmark_rows.append(
            {
                "strategy": row.strategy,
                "cost_bps": int(row.cost_bps),
                "same_avg_position_calmar": same_metrics["calmar"],
                "matched_beta_calmar": matched_metrics["calmar"],
                "same_avg_position_return_pct": same_metrics["return_pct"],
                "matched_beta_return_pct": matched_metrics["return_pct"],
            }
        )
    benchmarks = pd.DataFrame(benchmark_rows)
    comparison = comparison.merge(benchmarks, on=["strategy", "cost_bps"], how="left")
    year_passes = annual.groupby(["strategy", "cost_bps"])["risk_gate_pass"].sum().rename("outer_years_risk_passed")
    comparison = comparison.merge(year_passes, on=["strategy", "cost_bps"], how="left")
    comparison["risk_gate_pass"] = comparison.apply(lambda row: _risk_gate(row.to_dict()), axis=1)
    comparison["participation_gate_pass"] = comparison["avg_soxl_position_pct"].ge(15.0)
    comparison["upside_capture_gate_pass"] = comparison["upside_capture_pct"].ge(60.0)
    comparison["calmar_same_position_gate_pass"] = comparison["calmar"].gt(comparison["same_avg_position_calmar"])
    comparison["calmar_matched_beta_gate_pass"] = comparison["calmar"].gt(comparison["matched_beta_calmar"])
    comparison["outer_year_gate_pass"] = comparison["outer_years_risk_passed"].ge(3)
    comparison["all_required_gates_pass"] = comparison[
        [
            "risk_gate_pass",
            "participation_gate_pass",
            "upside_capture_gate_pass",
            "calmar_same_position_gate_pass",
            "calmar_matched_beta_gate_pass",
            "outer_year_gate_pass",
        ]
    ].all(axis=1)
    gate_columns = [
        "risk_gate_pass",
        "participation_gate_pass",
        "upside_capture_gate_pass",
        "calmar_same_position_gate_pass",
        "calmar_matched_beta_gate_pass",
        "outer_year_gate_pass",
    ]
    comparison["gate_count"] = comparison[gate_columns].sum(axis=1)
    comparison["failed_gates"] = comparison.apply(
        lambda row: ",".join(column.removesuffix("_pass") for column in gate_columns if not bool(row[column])),
        axis=1,
    )
    return comparison, annual, position_frame, return_frame


def _full_history_metrics(
    frame: pd.DataFrame,
    strategies: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    cost_bps: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, (soxl_weight, soxx_weight) in strategies.items():
        result = _simulate_weights(
            frame,
            soxl_weight=soxl_weight,
            soxx_weight=soxx_weight,
            cost_bps=cost_bps,
        )
        rows.append({"strategy": strategy, "cost_bps": cost_bps, **_risk_metrics(result)})
    cppi = _simulate_george_trainor_cppi(frame, cost_bps=cost_bps)
    rows.append({"strategy": "george_trainor_cppi_3x_2p5pct", "cost_bps": cost_bps, **_risk_metrics(cppi)})
    return pd.DataFrame(rows)


def _append_evaluation_rows(
    strategy: str,
    cost_bps: int,
    result: pd.DataFrame,
    comparison_rows: list[dict[str, Any]],
    annual_rows: list[dict[str, Any]],
    positions: list[pd.DataFrame],
    returns_rows: list[pd.DataFrame],
    outer_years: tuple[int, ...],
) -> None:
    metrics = _risk_metrics(result)
    comparison_rows.append({"strategy": strategy, "cost_bps": cost_bps, **metrics})
    stored = result.copy()
    stored["strategy"] = strategy
    stored["cost_bps"] = cost_bps
    positions.append(stored.loc[:, ["trade_date", "strategy", "cost_bps", "soxl_weight", "soxx_weight", "cash_weight"]])
    returns_rows.append(stored)
    for year in outer_years:
        year_result = result.loc[result["trade_date"].dt.year.eq(year)].copy()
        if year_result.empty:
            continue
        year_metrics = _risk_metrics(year_result)
        annual_rows.append(
            {
                "strategy": strategy,
                "cost_bps": cost_bps,
                "outer_year": year,
                **year_metrics,
                "risk_gate_pass": _risk_gate(year_metrics),
            }
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _strategy_fidelity() -> list[dict[str, Any]]:
    return [
        {
            "strategy": "avellaneda_daily_or_weekly_1x_replication",
            "paper": "Avellaneda and Zhang (2010), Path-Dependence of Leveraged ETF Returns",
            "status": "mechanism_replay",
            "implementation": "one-third SOXL plus cash, daily or weekly rebalance",
            "difference": "SOXX is an ETF proxy for the semiconductor index and cash return is zero",
        },
        {
            "strategy": "gayed_ma200_2x_equivalent",
            "paper": "Gayed (2016/2020), Leverage for the Long Run",
            "status": "mechanism_replay",
            "implementation": "SOXX close above 200-day moving average selects two-thirds SOXL; otherwise cash",
            "difference": "uses SOXX/SOXL instead of a broad-market 2x fund and Treasury bills",
        },
        {
            "strategy": "moreira_muir_lagged_inverse_variance",
            "paper": "Moreira and Muir (2017), Volatility-Managed Portfolios",
            "status": "paper_adapted",
            "implementation": "monthly SOXL weight scales with inverse previous-month SOXX variance",
            "difference": "long-only weight clipped at one; normalizer fixed from 2011-2018 and mapped to SOXL",
        },
        {
            "strategy": "george_trainor_cppi_3x_2p5pct",
            "paper": "George and Trainor (2017), Portfolio Insurance Using Leveraged ETFs",
            "status": "near_exact_mechanism_replay",
            "implementation": "90% annual floor, multiplier 5/3, one-third cap, 2.5% SOXX rebalance barrier",
            "difference": "uses observed SOXL returns, SOXX trigger, zero cash return, and explicit turnover costs",
        },
        {
            "strategy": "smart_leverage_neural_control",
            "paper": "van Staden, Forsyth and Li (2026), Smart leverage?",
            "status": "not_replicated_needs_data_and_scope",
            "implementation": "excluded from result ranking",
            "difference": "paper requires broad-market LETF plus PIT T-bills/T-bonds and bootstrap-trained neural control",
        },
    ]


def _run_card(spec: LeveragedEtfPaperReplaySpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    frame, data_audit = _load_market_frame(spec)
    strategies, strategy_audit = build_paper_strategy_weights(frame, spec)
    comparison, annual, positions, strategy_returns = _evaluate(
        frame,
        strategies,
        costs_bps=spec.costs_bps,
        outer_years=spec.outer_years,
    )
    full_history = _full_history_metrics(frame, strategies, cost_bps=10)
    required = comparison.loc[comparison["cost_bps"].eq(10)].copy()
    qualifying = required.loc[required["all_required_gates_pass"]].copy()
    promoted = not qualifying.empty
    decision = "candidate_for_validated_research" if promoted else "research_reject_no_paper_strategy_passed_all_gates"
    selected = (
        qualifying.sort_values(["calmar", "cagr_pct"], ascending=False).iloc[0].to_dict()
        if promoted
        else required.sort_values(
            ["gate_count", "outer_years_risk_passed", "max_drawdown_pct", "calmar"],
            ascending=False,
        ).iloc[0].to_dict()
    )
    sidecars = {
        "strategy_comparison": "strategy_comparison.csv",
        "annual_outer_metrics": "annual_outer_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "analysis_summary": "analysis_summary.json",
        "paper_implementation_audit": "paper_implementation_audit.json",
        "full_history_metrics": "full_history_metrics.csv",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        returns_path = tmp_dir / "strategy_returns.csv"
        positions.to_csv(positions_path, index=False)
        strategy_returns.to_csv(returns_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.provider_uri.resolve().as_posix(),
                data_as_of=data_audit["as_of"],
                execution_timing=data_audit["timing"],
                strategies=",".join(comparison["strategy"].drop_duplicates()),
                outer_years=",".join(str(year) for year in spec.outer_years),
                costs_bps=",".join(str(cost) for cost in spec.costs_bps),
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                required_cost_pass_count=float(len(qualifying)),
                selected_cagr_pct=float(selected["cagr_pct"]),
                selected_max_drawdown_pct=float(selected["max_drawdown_pct"]),
                selected_worst_rolling_5d_pct=float(selected["worst_rolling_5d_pct"]),
            )
            recorder.log_artifact(positions_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(returns_path.as_posix(), artifact_path=spec.experiment)
            summary = {
                "experiment": spec.experiment,
                "card_type": CARD_TYPE,
                "promotion_decision": decision,
                "selection_uses_outer_test": False,
                "outer_test_use": "evaluation_and_diagnostic_ranking_only_never_parameter_selection",
                "diagnostic_ranking_uses_outer_test": True,
                "data_audit": data_audit,
                "outer_years": list(spec.outer_years),
                "execution_timing": data_audit["timing"],
                "cash_return_assumption": 0.0,
                "costs_bps": list(spec.costs_bps),
                "required_cost_bps": 10,
                "stress_cost_bps": 25,
                "gates": {
                    "max_drawdown_pct_min": -20.0,
                    "worst_day_pct_min": -12.0,
                    "cvar5_pct_min": -8.0,
                    "worst_rolling_5d_pct_min": -12.0,
                    "avg_soxl_position_pct_min": 15.0,
                    "upside_capture_pct_min": 60.0,
                    "outer_years_risk_passed_min": 3,
                    "calmar_must_exceed_same_avg_soxl": True,
                    "calmar_must_exceed_matched_beta_soxl_soxx_cash": True,
                },
                "paper_implementation_audit": _strategy_fidelity(),
                "strategy_audit": strategy_audit,
                "required_cost_pass_count": int(len(qualifying)),
                "selected_or_best_diagnostic": _json_ready(selected),
                "comparison_preview_10bps": _json_ready(required.to_dict(orient="records")),
                "full_history_preview_10bps": _json_ready(full_history.to_dict(orient="records")),
                "recorder_id": recorder.id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": Path(recorder.get_local_dir()).resolve().as_posix(),
                "sidecars": sidecars,
                "artifact_policy": {
                    "policy": "slim_card_recorder_first",
                    "workflow_config": "workflow.yaml",
                    "recorder_only_artifacts": list(RECORDER_ONLY_ARTIFACTS),
                },
            }
            _write_card_files(card, summary, comparison, annual, full_history)
            for rel_path in sidecars.values():
                recorder.log_artifact((card / rel_path).as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(Path(recorder.get_local_dir()).resolve())
    return _json_ready(summary)


def _write_card_files(
    card: Path,
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    full_history: pd.DataFrame,
) -> None:
    comparison.to_csv(card / "strategy_comparison.csv", index=False)
    annual.to_csv(card / "annual_outer_metrics.csv", index=False)
    full_history.to_csv(card / "full_history_metrics.csv", index=False)
    risk_cols = [
        "strategy",
        "cost_bps",
        "max_drawdown_pct",
        "worst_day_pct",
        "cvar5_pct",
        "worst_rolling_5d_pct",
        "annualized_vol_pct",
        "risk_gate_pass",
    ]
    comparison.loc[:, risk_cols].to_csv(card / "risk_analysis.csv", index=False)
    analysis = {
        "promotion_decision": summary["promotion_decision"],
        "required_cost_pass_count": summary["required_cost_pass_count"],
        "selected_or_best_diagnostic": summary["selected_or_best_diagnostic"],
    }
    (card / "analysis_summary.json").write_text(json.dumps(_json_ready(analysis), indent=2) + "\n", encoding="utf-8")
    (card / "paper_implementation_audit.json").write_text(
        json.dumps(summary["paper_implementation_audit"], indent=2) + "\n", encoding="utf-8"
    )
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
    selected = summary["selected_or_best_diagnostic"]
    conclusion = f"""# 结论

## 跑了什么

使用 qlib 日线数据（截至 {summary['data_audit']['as_of']}）按 close(t) 生成信号、在下一交易日开盘执行，复刻或适配了 200 日均线杠杆切换、滞后一月方差缩放、3x LETF CPPI，以及 1x 动态复制基准。所有策略均报告 0/10/25bp，outer 2023–2026 完全不参与参数选择。

## 关键结果

10bp 下通过全部正式门槛的策略数为 {summary['required_cost_pass_count']}。当前最佳诊断项为 `{selected['strategy']}`：CAGR {selected['cagr_pct']:.2f}%，MaxDD {selected['max_drawdown_pct']:.2f}%，单日最差 {selected['worst_day_pct']:.2f}%，CVaR5 {selected['cvar5_pct']:.2f}%，真实滚动五日最差 {selected['worst_rolling_5d_pct']:.2f}%，SOXL 平均仓位 {selected['avg_soxl_position_pct']:.2f}%，上涨捕获 {selected['upside_capture_pct']:.2f}%。

## 年份与成本差异

逐年 outer 结果见 `annual_outer_metrics.csv`，25bp 压力结果见 `strategy_comparison.csv`。2026 为截至数据日的未完整年度，不参与任何选择。

## 晋升决定

`{summary['promotion_decision']}`。本卡是论文机制复盘，不晋升未通过全部风险、参与度、上涨捕获、双基准 Calmar 与 3/4 年度门槛的方案。

## 风险

CPPI 现金腿暂按 0 收益，缺少论文使用的 PIT 一年期国债收益；SOXX/SOXL 是半导体 ETF 映射，不等同论文的宽基指数；实际 SOXL 存在跳空、交易摩擦和杠杆衰减。`smart leverage` 的债券输入和 bootstrap 神经控制未在本卡伪复刻。

## 下一步

若继续，只应补齐 PIT T-bill/T-bond 序列后复刻 smart-leverage 数值框架，或把通过研究宽松门但未过正式门的论文规则作为模型的风险上限；不应再把底仓硬塞进不满足尾部门槛的模型。

不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    workflow = f"""experiment: {summary['experiment']}
card_type: {summary['card_type']}
provider_uri: {summary['data_audit']['provider_uri']}
recorder_uri: {summary['recorder_uri']}
recorder_id: {summary['recorder_id']}
as_of: {summary['data_audit']['as_of']}
qlib:
  market: all
  region: us
  data_api: [qlib.init, qlib.data.D.calendar, qlib.data.D.features]
execution:
  timing: {summary['execution_timing']}
  costs_bps: [0, 10, 25]
selection:
  outer_years: [2023, 2024, 2025, 2026]
  selection_uses_outer_test: false
promotion:
  decision: {summary['promotion_decision']}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  machine_entrypoint: summary.json
  recorder_only_artifacts:
    - artifacts/{summary['experiment']}/positions.csv
    - artifacts/{summary['experiment']}/strategy_returns.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
