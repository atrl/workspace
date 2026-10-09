"""Overnight futures signal card: a causal session-aligned SOXL excess test.

Uses newly collected night-session data (equity-index futures and premarket
bars). Every signal is computed from bars that end at or before the 09:30 cash
open; the traded target is the same day's 09:30-16:00 regular session. The
benchmark is SOXX over the identical regular session.

Selection uses a development split only; the held-out diagnostic split never
influences the choice. No return forecast or fitted parameter is used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.night_session_signal import (
    build_session_panel,
    evaluate_strategy,
    information_coefficient,
    load_night_session_instruments,
    switch_strategy,
)
from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.data.night_session_store import PROVIDER_URI, build_night_session_store

EXPERIMENT = "night_session_signal_20261010"
SCHEMA_VERSION = "night_session_signal_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
DATA_DIR = "data/raw/external/night_session_20261010_v1"
SIGNALS = ("nq_overnight", "es_overnight", "soxl_premarket", "soxx_premarket", "smh_premarket")


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class NightSpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")
    development_fraction: float = 0.6
    costs_bps: tuple[float, ...] = (10.0, 25.0)
    thresholds: tuple[float, ...] = (0.0, 0.001, 0.002)
    minimum_positive_years: int = 1

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment


def resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _ensure_store(spec: NightSpec) -> dict:
    """Build the qlib night-session provider and audit it through qlib D APIs."""
    provider = spec.root / PROVIDER_URI
    manifest = provider / "qlib_store_manifest.json"
    if not manifest.is_file():
        return build_night_session_store(root=spec.root, provider_uri=provider)
    return {"audit": json.loads(manifest.read_text(encoding="utf-8")), "built": False}


def _load_panel(spec: NightSpec) -> tuple[pd.DataFrame, dict]:
    """Read the night-session panel through ``qlib.data.D`` (never raw parquet)."""
    store = _ensure_store(spec)
    config = QlibProviderConfig(provider_uri=spec.root / PROVIDER_URI, region="us", market="all", freq="day")
    raw = load_qlib_feature_frame(
        config, instruments=["SOXL", "SOXX"],
        fields=["$return", "$premarket", "$nq_overnight", "$es_overnight"],
        start="2024-10-09", end="2026-10-09",
    )
    returns = raw["$return"].unstack(level=0)
    premarket = raw["$premarket"].unstack(level=0)
    nq = raw["$nq_overnight"].unstack(level=0)
    es = raw["$es_overnight"].unstack(level=0)
    panel = pd.DataFrame({
        "soxl_return": returns["SOXL"].to_numpy(float),
        "soxx_return": returns["SOXX"].to_numpy(float),
        "soxl_premarket": premarket["SOXL"].to_numpy(float),
        "soxx_premarket": premarket["SOXX"].to_numpy(float),
        "nq_overnight": nq["SOXL"].to_numpy(float),
        "es_overnight": es["SOXL"].to_numpy(float),
    }, index=pd.to_datetime(returns.index))
    panel = panel[panel.index.dayofweek < 5]
    audit = {
        "provider_uri": config.resolved_provider_uri.as_posix(),
        "source": DATA_DIR, "store": store.get("audit", {}),
        "sessions": int(len(panel)), "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "qlib_data_api": ["qlib.data.D.features"],
        "regular_session": "09:30-16:00 ET",
        "overnight_window": "prior 18:00 ET to current 09:30 ET",
        "premarket_window": "04:00-09:30 ET",
        "qlib_docs_reference": QLIB_DOCS,
    }
    return panel, audit


def run_night_session_signal(*, root: Path | str | None = None, card_dir: Path | str | None = None,
                             recorder_uri: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = NightSpec(root=spec_root)
    # The historical workflow is unconditionally blocked: it contains known
    # integrity defects (look-ahead bar, return-convention mismatch, uncharged
    # round trips, mislabeled benchmark, in-window threshold reselection) and has
    # NOT been rebuilt with correct timing/cost/qlib semantics. This is a static
    # refusal that any target path hits, so a fresh empty card_dir cannot bypass
    # it by omitting an on-disk marker.
    raise ValueError(
        "blocked_known_integrity_defects: night_session_signal run is unconditionally disabled; "
        "it is invalid_for_promotion and no success result may be produced")

    target_card = Path(card_dir) if card_dir is not None else spec.card_dir
    target_recorder = Path(recorder_uri) if recorder_uri is not None else (target_card / "qlib_recorder")
    _guard_invalid_card(target_card)
    target_card.mkdir(parents=True, exist_ok=True)

    panel, data_audit = _load_panel(spec)
    _require(len(panel) >= 60, "panel is too short")
    signals = [name for name in SIGNALS if name in panel.columns and panel[name].notna().sum() > 60]

    ic_rows = []
    for name in signals:
        ic = information_coefficient(panel[name], panel["soxl_return"], panel["soxx_return"])
        ic_rows.append({"signal": name, **ic,
                        "classification": ("tradeable_forecast" if abs(ic["ic_excess"]) >= 0.05
                                           else "weak_or_none")})
    ic_table = pd.DataFrame(ic_rows)

    split = int(len(panel) * spec.development_fraction)
    windows = {"development": panel.iloc[:split], "reused_diagnostic": panel.iloc[split:]}

    window_results: dict[str, dict] = {}
    for cost in spec.costs_bps:
        window_results[str(cost)] = {}
        for window, block in windows.items():
            rows = {}
            for name in signals:
                for defensive in ("soxx", "cash"):
                    best = None
                    for threshold in spec.thresholds:
                        replay = switch_strategy(block[name], block["soxl_return"], block["soxx_return"],
                                                 threshold=threshold, cost_bps=cost, defensive=defensive)
                        result = evaluate_strategy(replay, minimum_positive_years=spec.minimum_positive_years)
                        result["threshold"] = threshold
                        result["defensive"] = defensive
                        if best is None or result["wealth_ratio_vs_soxx"] > best["wealth_ratio_vs_soxx"]:
                            best = result
                    rows[f"{name}__{defensive}"] = best
            window_results[str(cost)][window] = rows

    primary = str(spec.costs_bps[0])
    dev = window_results[primary]["development"]
    # A candidate must beat SOXX on return AND Sharpe in development at the high
    # cost, and must keep beating SOXX on return AND Sharpe in the held-out
    # diagnostic window at the high cost. Drawdown is reported but not gated on
    # short windows, where it is too noisy to be a stable signal.
    high = str(spec.costs_bps[1])
    dev_high = window_results[high]["development"]
    diag_high = window_results[high]["reused_diagnostic"]
    eligible = sorted(name for name, res in dev_high.items()
                      if res["beats_return"] and res["beats_sharpe"])
    stable_dev = sorted(name for name, res in dev.items() if res["stable_excess"])
    selected = (max(eligible, key=lambda n: dev_high[n]["sharpe_difference"]) if eligible else
                max(dev_high, key=lambda n: dev_high[n]["sharpe_difference"]))
    promotion_eligible = bool(
        selected in eligible
        and diag_high[selected]["beats_return"] and diag_high[selected]["beats_sharpe"])

    selected_name, _, selected_defensive = selected.partition("__")
    selected_defensive = selected_defensive or "soxx"
    # Native Qlib replay of the selected strategy: qlib.backtest.backtest must
    # reproduce the research return per session before the card can claim it.
    native_audit = _native_reconciliation(spec, panel, selected_name, selected_defensive)
    _require(native_audit["status"] == "passed",
             "native Qlib reconciliation did not pass: " + json.dumps(native_audit))

    # Quarterly consistency for the selected candidate at the high cost: the
    # durable test of an overnight signal on a short sample.
    quarterly = []
    for cost in spec.costs_bps:
        replay = switch_strategy(panel[selected_name], panel["soxl_return"], panel["soxx_return"],
                                 threshold=0.0, cost_bps=cost,
                                 defensive=selected_defensive or "soxx")
        frame = replay.copy()
        index = pd.DatetimeIndex(frame.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        frame["quarter"] = index.to_period("Q").astype(str)
        for quarter, group in frame.groupby("quarter"):
            strategy_wealth = float(np.prod(1 + group["strategy"]))
            bench_wealth = float(np.prod(1 + group["benchmark"]))
            quarterly.append({"cost_bps": float(cost), "quarter": quarter,
                              "sessions": int(len(group)),
                              "wealth_ratio_vs_soxx": strategy_wealth / bench_wealth,
                              "average_exposure": float(group["exposure"].mean())})
    quarterly_wins = sum(1 for row in quarterly if row["cost_bps"] == spec.costs_bps[1]
                         and row["wealth_ratio_vs_soxx"] > 1.0)
    quarterly_total = sum(1 for row in quarterly if row["cost_bps"] == spec.costs_bps[1])
    summary = _build_summary(spec=spec, data_audit=data_audit, ic_table=ic_table,
                             window_results=window_results, stable_dev=stable_dev,
                             selected=selected, promotion_eligible=promotion_eligible,
                             quarterly=quarterly, quarterly_wins=quarterly_wins,
                             quarterly_total=quarterly_total, native_audit=native_audit)
    summary.update(_record(spec=spec, recorder_uri=target_recorder, summary=summary))
    return summary


def _native_reconciliation(spec: NightSpec, panel: pd.DataFrame, signal_name: str,
                           defensive: str) -> dict:
    """Replay the selected switch strategy through the native Qlib backtest.

    Exposure for session t is decided from the overnight signal read at t's open
    and held from t's open to t+1's open. ``qlib.backtest.backtest`` executes the
    schedule with the SimulatorExecutor; the native next-open NAV must reproduce
    the research return within tolerance.
    """
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order, TradeDecisionWO

    from src.backtest.qlib_continuous_position import QlibContinuousPosition
    from src.backtest.strategy import DirectTargetWeightStrategy
    from src.workflow.stage1_economic_action import fee_affordable_target

    provider = (spec.root / PROVIDER_URI).resolve()
    _require(provider.is_dir(), "night-session qlib provider is missing")
    cost_bps = float(spec.costs_bps[0])

    exposure = (panel[signal_name] > 0.0).astype(float)
    soxx_weight = (1.0 - exposure) if defensive == "soxx" else exposure * 0.0
    cash_weight = (1.0 - exposure) if defensive == "cash" else exposure * 0.0
    targets = pd.DataFrame({
        "soxl": exposure,
        "soxx": soxx_weight,
        "cash": cash_weight,
    })
    # Trade at the open of session t using information known at that open.
    trade_dates = pd.DatetimeIndex(panel.index)
    schedule = pd.DataFrame({
        "trade_date": trade_dates,
        "exit_date": trade_dates.to_series().shift(-1).to_numpy(),
        "target_soxl_weight": targets["soxl"].to_numpy(float),
        "target_soxx_weight": targets["soxx"].to_numpy(float),
        "target_cash_weight": targets["cash"].to_numpy(float),
        "execute_open_target": True,
    }).dropna(subset=["exit_date"]).reset_index(drop=True)
    schedule["trade_date"] = pd.to_datetime(schedule["trade_date"])
    schedule["exit_date"] = pd.to_datetime(schedule["exit_date"])

    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    from qlib.data import D
    opening = D.features(["SOXL", "SOXX"], ["$open"],
                         start_time=str(schedule["trade_date"].min().date()),
                         end_time=str(schedule["exit_date"].max().date()), freq="day")["$open"]
    opening = opening.unstack("instrument").astype(float)
    # Research open-to-open returns from the same native open prices.
    research_soxl = (opening["SOXL"].reindex(schedule["exit_date"]).to_numpy()
                     / opening["SOXL"].reindex(schedule["trade_date"]).to_numpy() - 1.0)
    research_soxx = (opening["SOXX"].reindex(schedule["exit_date"]).to_numpy()
                     / opening["SOXX"].reindex(schedule["trade_date"]).to_numpy() - 1.0)
    # Funded open-to-open accounting: before trading at session t the portfolio
    # holds the previous target grown by its real open-to-open return; fees are
    # charged on that drifted notional, matching the native Exchange fee base.
    n = len(schedule)
    asset_returns = np.column_stack([research_soxl[:n], research_soxx[:n], np.zeros(n)])
    targets_array = targets.iloc[:n].to_numpy(float)
    from src.model.post_fee_settlement import scalar_settle

    # Exact native funding convention: at each open the portfolio holds target[i-1]
    # which has drifted by that session's return; the target is funded after the
    # turnover fee, using the same solver the native Exchange implements.
    research_net = np.zeros(n)
    held = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        target = targets_array[i]
        ending, multiple, fee, turnover, _, _ = scalar_settle(
            held[0], held[1], held[2], target[0], target[1], target[2],
            asset_returns[i, 0], asset_returns[i, 1], cost_bps / 1e4)
        research_net[i] = multiple - 1.0
        held = np.asarray(ending, dtype=float)

    class NativeSwitchStrategy(DirectTargetWeightStrategy):
        def __init__(self):
            super().__init__(signal=pd.Series(dtype=float,
                          index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])),
                          risk_degree=1.0)
            self.schedule = schedule.set_index("trade_date")

        def generate_trade_decision(self, execute_result=None):
            start, end = self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step())
            row = self.schedule.loc[pd.Timestamp(start).normalize()]
            held = self.trade_position.get_stock_amount_dict()
            quote = {symbol: float(self.trade_exchange.get_deal_price(symbol, start, end, direction=Order.BUY))
                     for symbol in ("SOXL", "SOXX")}
            values = np.array([float(held.get(symbol, 0.0)) * quote[symbol] for symbol in ("SOXL", "SOXX")]
                              + [float(self.trade_position.get_cash())])
            nav = float(values.sum())
            target = row.loc[["target_soxl_weight", "target_soxx_weight", "target_cash_weight"]].to_numpy(float)
            executable = fee_affordable_target(target, pretrade_weights=values / nav, cost_bps=cost_bps)
            amounts = {symbol: nav * executable[i] / quote[symbol] for i, symbol in enumerate(("SOXL", "SOXX"))}
            orders = self.trade_exchange.generate_order_for_target_amount_position(amounts, held, start, end)
            return TradeDecisionWO(orders, self)

    first = schedule["trade_date"].iloc[0]
    last = schedule["trade_date"].iloc[-1]
    native, _ = backtest(
        start_time=first, end_time=last, strategy=NativeSwitchStrategy(),
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
        account=100_000.0, benchmark="SOXX", pos_type=QlibContinuousPosition,
        exchange_kwargs={"codes": ["SOXL", "SOXX"], "freq": "day", "deal_price": "open",
                         "limit_threshold": None, "trade_unit": None,
                         "open_cost": cost_bps / 10_000, "close_cost": cost_bps / 10_000, "min_cost": 0.0})
    _, positions = native["1day"]
    previous = 100_000.0
    errors = []
    for row in schedule.itertuples(index=False):
        position = positions[row.trade_date]
        amounts = position.get_stock_amount_dict()
        nav = float(position.get_cash()) + sum(
            float(amounts.get(symbol, 0.0)) * opening.loc[row.exit_date, symbol] for symbol in ("SOXL", "SOXX"))
        errors.append(nav / previous - 1.0)
        previous = nav
    errors = np.asarray(errors)
    maximum = float(np.max(np.abs(errors - research_net)))
    return {"status": "passed" if maximum <= 1e-6 else "failed",
            "sessions": int(len(schedule)), "cost_bps": cost_bps,
            "provider_uri": str(provider), "maximum_session_return_error": maximum,
            "tolerance": 1e-6,
            "native_api": "qlib.backtest.backtest + SimulatorExecutor + Exchange + Account + Position",
            "qlib_api_reference": "https://qlib.readthedocs.io/en/latest/reference/api.html#qlib.backtest.backtest"}


def _build_summary(*, spec, data_audit, ic_table, window_results, stable_dev,
                   selected, promotion_eligible, quarterly, quarterly_wins,
                   quarterly_total, native_audit) -> dict[str, Any]:
    comparison = []
    for cost, windows in window_results.items():
        for window, rows in windows.items():
            for name, res in rows.items():
                comparison.append({
                    "cost_bps": float(cost), "window": window, "signal": name,
                    "threshold": res["threshold"],
                    "return_pct": res["strategy"]["return_pct"],
                    "benchmark_return_pct": res["benchmark"]["return_pct"],
                    "sharpe": res["strategy"]["sharpe"], "benchmark_sharpe": res["benchmark"]["sharpe"],
                    "sharpe_difference": res["sharpe_difference"],
                    "max_drawdown_pct": res["strategy"]["max_drawdown_pct"],
                    "max_drawdown_improvement_pct": res["max_drawdown_improvement_pct"],
                    "wealth_ratio_vs_soxx": res["wealth_ratio_vs_soxx"],
                    "switches_per_year": res["switches_per_year"],
                    "positive_year_count": res["positive_year_count"], "year_count": res["year_count"],
                    "stable_excess": res["stable_excess"],
                })
    return {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "active_research",
        "decision": ("stable_overnight_excess_candidate" if promotion_eligible
                     else "no_stable_overnight_excess"),
        "native_audit": native_audit,
        "quarterly_comparison": quarterly,
        "quarterly_win_rate": (quarterly_wins / quarterly_total) if quarterly_total else 0.0,
        "quarterly_win_count": quarterly_wins,
        "quarterly_count": quarterly_total,
        "promotion_eligible": promotion_eligible,
        "objective_achieved": promotion_eligible,
        "new_data_source": DATA_DIR,
        "data_audit": data_audit,
        "development_fraction": spec.development_fraction,
        "costs_bps": list(spec.costs_bps),
        "thresholds": list(spec.thresholds),
        "signals": list(ic_table["signal"]),
        "information_coefficients": ic_table.to_dict(orient="records"),
        "selected_signal": selected,
        "qlib_data_api": ["qlib.data.D.features", "qlib.backtest.backtest"],
        "qlib_native_status": native_audit["status"],
        "stable_excess_in_development": stable_dev,
        "window_comparison": comparison,
        "return_forecast_used": False,
        "model_fits": 0,
        "causality_statement": (
            "Overnight futures change uses the prior 18:00 ET to current 09:30 ET window; premarket "
            "uses 04:00-09:30 ET. Both end at the cash open. The traded target is the same day's "
            "09:30-16:00 regular session and the benchmark is SOXX over that identical window."
        ),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                "data/session_panel.pkl",
                "analysis/information_coefficients.pkl",
                "analysis/window_results.pkl",
            ],
        },
        "limitations": [
            "The natural sample is about two years; the diagnostic split is short.",
            "Yahoo night-session history is limited to roughly two years at hourly granularity.",
            "Overnight futures prices are continuous-session closes, not a guaranteed executable fill.",
            "Turnover is high; wider spreads than 25bp would erode the edge.",
            "Overnight price moves partly reflect the prior US close, which is public information.",
        ],
    }


def _record(*, spec, recorder_uri, summary) -> dict[str, Any]:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R

    recorder_uri = Path(recorder_uri).resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / PROVIDER_URI).resolve()), region="us")
    with R.start(experiment_name=spec.experiment, recorder_name="overnight", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(
            schema_version=SCHEMA_VERSION, new_data_source=DATA_DIR,
            provider_uri=summary["data_audit"]["provider_uri"],
            selected_signal=summary["selected_signal"], costs_bps=str(list(spec.costs_bps)),
            development_fraction=spec.development_fraction,
            qlib_native_status=summary["qlib_native_status"])
        recorder.log_metrics(
            selected_dev_ratio=next((r["wealth_ratio_vs_soxx"] for r in summary["window_comparison"]
                                     if r["window"] == "development" and r["signal"] == summary["selected_signal"]
                                     and r["cost_bps"] == 10.0), 0.0),
            quarterly_win_rate=summary["quarterly_win_rate"],
            native_max_error=summary["native_audit"]["maximum_session_return_error"],
            promotion_eligible=int(summary["promotion_eligible"]))
        recorder.save_objects(summary_json=summary)
        recorder_id = recorder.id
        experiment_id = str(recorder.experiment_id)
        local_dir = recorder_uri / experiment_id / recorder_id
    return {"recorder_id": recorder_id, "experiment_id": experiment_id,
            "recorder_uri": str(recorder_uri), "recorder_local_dir": str(local_dir),
            "qlib_docs_reference": QLIB_DOCS}


def _guard_invalid_card(card_dir: Path) -> None:
    """Refuse to overwrite a card already marked invalid_for_promotion.

    The night-session card was corrected to ``invalid_for_promotion`` after a
    look-ahead defect was found. Re-running the original workflow must never
    silently restore a promotion-eligible result, so this guard fails closed.
    """
    existing = card_dir / "summary.json"
    if not existing.is_file():
        return
    try:
        prior = json.loads(existing.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    marked = str(prior.get("decision", "")).startswith("invalid_for_promotion")
    if marked or prior.get("invalid_for_promotion") is True:
        raise ValueError(
            "blocked: night-session card is marked invalid_for_promotion and must not be overwritten; "
            "use a new card identity for any corrected research")


def write_card_files(summary: dict, *, card_dir: Path) -> None:
    # The legacy writer may never record a promotion-eligible or objective-achieved
    # card: this workflow has known integrity defects and produces no success. The
    # integrity correction uses its own sidecar and does not go through here.
    if summary.get("promotion_eligible") is True or summary.get("objective_achieved") is True:
        raise ValueError(
            "blocked_known_integrity_defects: legacy night-session writer refuses to record "
            "a promotion_eligible or objective_achieved card")
    if str(summary.get("decision", "")).startswith("stable_"):
        raise ValueError(
            "blocked_known_integrity_defects: legacy writer refuses a stable_* decision")
    _guard_invalid_card(card_dir)
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(summary["information_coefficients"]).to_csv(
        card_dir / "information_coefficients.csv", index=False)
    comparison = pd.DataFrame(summary["window_comparison"])
    comparison.to_csv(card_dir / "signal_comparison.csv", index=False)
    pd.DataFrame(summary["quarterly_comparison"]).to_csv(card_dir / "quarterly_comparison.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "new_data_source": summary["new_data_source"], "recorder_uri": summary["recorder_uri"],
        "costs_bps": summary["costs_bps"], "thresholds": summary["thresholds"],
        "development_fraction": summary["development_fraction"],
        "causality_statement": summary["causality_statement"],
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    sel = summary["selected_signal"]
    rows = [r for r in summary["window_comparison"] if r["signal"] == sel]
    dev = next((r for r in rows if r["window"] == "development" and r["cost_bps"] == 10.0), {})
    diag = next((r for r in rows if r["window"] == "reused_diagnostic" and r["cost_bps"] == 25.0), {})
    ic = {r["signal"]: r for r in summary["information_coefficients"]}
    return (
        "# 夜盘信号稳定超额研究（中文结论）\n\n"
        f"- 新数据：`{summary['new_data_source']}`，Yahoo Finance 1 小时含盘前盘后（2024-10-09 至 2026-10-09），"
        f"含 SOXL/SOXX/SMH/NVDA/TSM 盘前及 ES/NQ 股指期货夜盘。\n"
        f"- 因果对齐：期货隔夜收益取前一日 18:00 ET 至当日 09:30 ET；盘前取 04:00–09:30 ET；"
        f"两者都在开盘前已知；交易标的为当日 09:30–16:00 regular session，基准为 SOXX 同一窗口。\n"
        f"- 信息系数（对 SOXL 与对 SOXL−SOXX 超额）："
        + "；".join(f"{r['signal']}(SOXL {_f(r['ic_soxl'])}, 超额 {_f(r['ic_excess'])})"
                    for r in summary["information_coefficients"]) + "。\n"
        f"- development 稳定超额信号：{summary['stable_excess_in_development'] or '无'}；选中 `{sel}`。\n"
        f"- 选中信号 development 10bp：收益 {dev.get('return_pct', 0):.2f}%（SOXX {dev.get('benchmark_return_pct', 0):.2f}%），"
        f"Sharpe 差 {dev.get('sharpe_difference', 0):+.2f}，回撤改善 {dev.get('max_drawdown_improvement_pct', 0):+.1f}pp，"
        f"年换手 {dev.get('switches_per_year', 0):.0f}。\n"
        f"- 诊断区间 25bp：收益 {diag.get('return_pct', 0):.2f}%（SOXX {diag.get('benchmark_return_pct', 0):.2f}%），"
        f"Sharpe 差 {diag.get('sharpe_difference', 0):+.2f}。\n"
        f"- qlib 原生：数据经 `data/qlib/night_session_20261010_v1` 由 `qlib.data.D.features` 读取，"
        f"策略由 `qlib.backtest.backtest`+SimulatorExecutor 执行；原生对账状态 {summary['qlib_native_status']}，"
        f"最大逐日误差 {summary['native_audit']['maximum_session_return_error']:.1e}。\n"
        f"- 季度一致性（高成本）：{summary['quarterly_win_count']}/{summary['quarterly_count']} 季度跑赢 SOXX。\n"
        f"- 是否晋升：{summary['promotion_eligible']}；决定 `{summary['decision']}`。\n"
        f"- 风险：样本仅约两年，诊断段短；期货夜盘价为连续合约收盘，不保证可成交；换手高（年约 120-240 次），"
        f"更宽价差会侵蚀边缘；隔夜价已部分反映上一美股收盘。\n"
        f"- qlib Recorder：{summary.get('recorder_id')}；qlib docs：{QLIB_DOCS}。\n"
    )


def _f(value):
    return "NA" if value is None else f"{value:+.3f}"
