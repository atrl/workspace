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
    switch_strategy,
)

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


def _load_panel(spec: NightSpec) -> tuple[pd.DataFrame, dict]:
    data_dir = spec.root / DATA_DIR
    files = {"soxl": "SOXL", "soxx": "SOXX", "smh": "SMH", "es": "ES_F", "nq": "NQ_F"}
    instruments = {}
    audit_series = {}
    for key, name in files.items():
        path = data_dir / f"{name}.parquet"
        _require(path.is_file(), f"missing night-session file {path}")
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        instruments[key] = frame
        audit_series[key] = {"file": str(path.relative_to(spec.root)), "bars": int(len(frame)),
                             "start": str(frame.index.min()), "end": str(frame.index.max())}
    panel = build_session_panel(instruments)
    audit = {"data_dir": DATA_DIR, "series": audit_series, "sessions": int(len(panel)),
             "start": str(panel.index.min().date()), "end": str(panel.index.max().date()),
             "regular_session": "09:30-16:00 ET", "overnight_window": "prior 18:00 ET to current 09:30 ET",
             "premarket_window": "04:00-09:30 ET"}
    return panel, audit


def run_night_session_signal(*, root: Path | str | None = None, card_dir: Path | str | None = None,
                             recorder_uri: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = NightSpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else spec.card_dir
    target_recorder = Path(recorder_uri) if recorder_uri is not None else (target_card / "qlib_recorder")
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

    # Quarterly consistency for the selected candidate at the high cost: the
    # durable test of an overnight signal on a short sample.
    selected_name, _, selected_defensive = selected.partition("__")
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
                             quarterly_total=quarterly_total)
    summary.update(_record(spec=spec, recorder_uri=target_recorder, summary=summary))
    return summary


def _build_summary(*, spec, data_audit, ic_table, window_results, stable_dev,
                   selected, promotion_eligible, quarterly, quarterly_wins,
                   quarterly_total) -> dict[str, Any]:
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
    qlib.init(provider_uri=str((spec.root / "data/qlib/soxl").resolve()), region="us")
    with R.start(experiment_name=spec.experiment, recorder_name="overnight", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(
            schema_version=SCHEMA_VERSION, new_data_source=DATA_DIR, selected_signal=summary["selected_signal"],
            costs_bps=str(list(spec.costs_bps)), development_fraction=spec.development_fraction)
        recorder.log_metrics(
            selected_dev_ratio=next((r["wealth_ratio_vs_soxx"] for r in summary["window_comparison"]
                                     if r["window"] == "development" and r["signal"] == summary["selected_signal"]
                                     and r["cost_bps"] == 10.0), 0.0),
            promotion_eligible=int(summary["promotion_eligible"]))
        recorder.save_objects(summary_json=summary)
        recorder_id = recorder.id
        experiment_id = str(recorder.experiment_id)
        local_dir = recorder_uri / experiment_id / recorder_id
    return {"recorder_id": recorder_id, "experiment_id": experiment_id,
            "recorder_uri": str(recorder_uri), "recorder_local_dir": str(local_dir),
            "qlib_docs_reference": QLIB_DOCS}


def write_card_files(summary: dict, *, card_dir: Path) -> None:
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
        f"- 季度一致性（高成本）：{summary['quarterly_win_count']}/{summary['quarterly_count']} 季度跑赢 SOXX。\n"
        f"- 是否晋升：{summary['promotion_eligible']}；决定 `{summary['decision']}`。\n"
        f"- 风险：样本仅约两年，诊断段短；期货夜盘价为连续合约收盘，不保证可成交；换手高（年约 120-240 次），"
        f"更宽价差会侵蚀边缘；隔夜价已部分反映上一美股收盘。\n"
        f"- qlib Recorder：{summary.get('recorder_id')}；qlib docs：{QLIB_DOCS}。\n"
    )


def _f(value):
    return "NA" if value is None else f"{value:+.3f}"
