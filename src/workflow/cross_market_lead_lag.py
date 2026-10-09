"""Cross-market lead-lag card over newly collected Asian/US semiconductor data.

Tests whether Asian semiconductor sessions (Taiwan/Korea/Japan) carry forecast
information for the next US SOXL session, with explicit same-date versus
strictly-lagged alignment. The card classifies each candidate market as
``contemporaneous_only``, ``forecast_textbook`` or ``no_signal`` and only a
strictly-lagged, cost-surviving relation can be a promotion candidate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.cross_market_lead_lag import (
    evaluate_strategy,
    information_coefficient,
    load_manifest,
    switch_strategy,
)

EXPERIMENT = "cross_market_lead_lag_20261010"
SCHEMA_VERSION = "cross_market_lead_lag_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
DATA_DIR = "data/raw/external/asia_semis_20261010_v1"
TARGET = "soxl"
ASIAN_MARKETS = ("twii", "tsmc_tw", "skhynix", "kospi", "nikkei")
US_REFERENCE = ("smh", "nvda", "tsm", "amd", "asml", "soxx")


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class LeadLagSpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")
    development: tuple[str, str] = ("2014-01-02", "2022-12-30")
    reused_diagnostic: tuple[str, str] = ("2023-01-03", "2026-10-08")
    costs_bps: tuple[float, ...] = (10.0, 25.0)
    minimum_positive_years: int = 4
    # A candidate is only "forecast_textbook" if the strictly-lagged IC clears
    # this magnitude in absolute value and its sign is consistent with a
    # tradeable direction.
    forecast_ic_floor: float = 0.03

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


def _load(spec: LeadLagSpec) -> tuple[pd.DataFrame, dict]:
    data_path = spec.root / DATA_DIR / "daily_closes_localdate.csv"
    manifest_path = spec.root / DATA_DIR / "manifest.json"
    _require(data_path.is_file() and manifest_path.is_file(), "new external data is missing")
    frame = pd.read_csv(data_path, index_col=0, parse_dates=True)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    returns = frame.pct_change(fill_method=None)
    audit = {
        "data_path": str(data_path.relative_to(spec.root)),
        "manifest_path": str(manifest_path.relative_to(spec.root)),
        "series_manifest": load_manifest(manifest).to_dict(orient="records"),
        "rows": int(len(frame)),
        "start": str(frame.index.min().date()),
        "end": str(frame.index.max().date()),
    }
    return returns, audit


def run_cross_market_lead_lag(*, root: Path | str | None = None,
                              card_dir: Path | str | None = None,
                              recorder_uri: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = LeadLagSpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else spec.card_dir
    target_recorder = Path(recorder_uri) if recorder_uri is not None else (target_card / "qlib_recorder")
    target_card.mkdir(parents=True, exist_ok=True)

    returns, data_audit = _load(spec)
    _require(TARGET in returns.columns, "target SOXL return is missing")

    ic_rows = []
    for market in list(ASIAN_MARKETS) + list(US_REFERENCE):
        if market not in returns.columns:
            continue
        same = information_coefficient(returns[market], returns[TARGET], lag=0)
        lagged = information_coefficient(returns[market], returns[TARGET], lag=1)
        ic_rows.append({
            "market": market,
            "region": "asia" if market in ASIAN_MARKETS else "us_reference",
            "same_date_ic": same["pearson_ic"],
            "same_date_ic_spearman": same["spearman_ic"],
            "lagged_ic": lagged["pearson_ic"],
            "lagged_ic_spearman": lagged["spearman_ic"],
            "observations": lagged["observations"],
        })
    ic_table = pd.DataFrame(ic_rows)

    def classify(row):
        # US reference series are near-simultaneous by construction (SOXX vs SOXL
        # same session), so only Asian markets can be a genuine cross-market
        # forecast. A forecast classification requires a strictly-lagged IC above
        # the floor; otherwise a same-date association is labelled co-movement.
        if row["region"] != "asia":
            return "us_reference_contemporaneous"
        if row["lagged_ic"] is None or not np.isfinite(row["lagged_ic"]):
            return "no_signal"
        if abs(row["lagged_ic"]) < spec.forecast_ic_floor:
            return ("contemporaneous_only"
                    if abs(row["same_date_ic"] or 0.0) >= spec.forecast_ic_floor else "no_signal")
        return "forecast_textbook"

    ic_table["classification"] = ic_table.apply(classify, axis=1)

    # Strategy evaluation for each candidate using the strictly-lagged signal.
    window_results = {}
    for cost in spec.costs_bps:
        window_results[str(cost)] = {}
        for name, span in (("development", spec.development),
                           ("reused_diagnostic", spec.reused_diagnostic)):
            mask = (returns.index >= pd.Timestamp(span[0])) & (returns.index <= pd.Timestamp(span[1]))
            block = returns.loc[mask]
            rows = {}
            # Strategy evaluation for Asian markets (potential forecast) and the
            # US reference set, always with the strictly-lagged signal.
            for market in ic_table["market"]:
                replay = switch_strategy(block[market], block[TARGET], block["soxx"],
                                         lag=1, cost_bps=cost)
                rows[market] = evaluate_strategy(replay, minimum_positive_years=spec.minimum_positive_years)
            window_results[str(cost)][name] = rows

    primary = str(spec.costs_bps[0])
    dev = window_results[primary]["development"]
    stable_dev = sorted(name for name, res in dev.items() if res["stable_excess"])
    forecast_markets = sorted(ic_table.loc[ic_table["classification"].eq("forecast_textbook"), "market"])
    promotion_eligible = bool(
        stable_dev
        and any(window_results[high]["reused_diagnostic"].get(name, {}).get("stable_excess")
                for high in (primary, str(spec.costs_bps[1])) for name in stable_dev)
    )
    summary = _build_summary(spec=spec, data_audit=data_audit, ic_table=ic_table,
                             window_results=window_results, stable_dev=stable_dev,
                             forecast_markets=forecast_markets, promotion_eligible=promotion_eligible)
    summary.update(_record(spec=spec, recorder_uri=target_recorder, summary=summary))
    return summary


def _build_summary(*, spec, data_audit, ic_table, window_results, stable_dev,
                   forecast_markets, promotion_eligible) -> dict[str, Any]:
    decision = ("forecast_lead_lag_candidate" if promotion_eligible
                else "no_tradeable_cross_market_forecast")
    comparison = []
    for cost, windows in window_results.items():
        for window, rows in windows.items():
            for market, res in rows.items():
                comparison.append({
                    "cost_bps": float(cost), "window": window, "market": market,
                    "return_pct": res["strategy"]["return_pct"],
                    "benchmark_return_pct": res["benchmark"]["return_pct"],
                    "sharpe": res["strategy"]["sharpe"], "benchmark_sharpe": res["benchmark"]["sharpe"],
                    "sharpe_difference": res["sharpe_difference"],
                    "max_drawdown_improvement_pct": res["max_drawdown_improvement_pct"],
                    "wealth_ratio_vs_soxx": res["wealth_ratio_vs_soxx"],
                    "positive_year_count": res["positive_year_count"], "year_count": res["year_count"],
                    "stable_excess": res["stable_excess"],
                })
    return {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "active_research",
        "decision": decision,
        "promotion_eligible": promotion_eligible,
        "objective_achieved": False,
        "new_data_source": DATA_DIR,
        "data_audit": data_audit,
        "target": TARGET,
        "costs_bps": list(spec.costs_bps),
        "segments": {"development": list(spec.development),
                     "reused_diagnostic": list(spec.reused_diagnostic)},
        "forecast_ic_floor": spec.forecast_ic_floor,
        "information_coefficients": ic_table.to_dict(orient="records"),
        "forecast_markets": forecast_markets,
        "stable_excess_in_development": stable_dev,
        "window_comparison": comparison,
        "return_forecast_used": False,
        "model_fits": 0,
        "causality_statement": (
            "Asian sessions close before the same exchange-local-date US open. Only the lagged "
            "alignment (signal at t-1 for the US session at t) is treated as forecast evidence. "
            "Same-date correlation is reported but classified as contemporaneous, not predictive."
        ),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                "data/returns.pkl",
                "analysis/information_coefficients.pkl",
                "analysis/window_results.pkl",
            ],
        },
        "limitations": [
            "The new data is daily closes only; no intraday Asian session timestamps are retained.",
            "A contemporaneous co-movement is not tradeable and is never counted as forecast power.",
            "No return forecast, model fit or fitted parameter is used.",
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
    with R.start(experiment_name=spec.experiment, recorder_name="lead_lag", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(
            schema_version=SCHEMA_VERSION, new_data_source=DATA_DIR,
            costs_bps=str(list(spec.costs_bps)), forecast_ic_floor=spec.forecast_ic_floor,
            decision=summary["decision"])
        recorder.log_metrics(
            forecast_market_count=len(summary["forecast_markets"]),
            stable_dev_count=len(summary["stable_excess_in_development"]),
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
    if not comparison.empty:
        comparison.to_csv(card_dir / "backtest_benchmark_comparison.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "new_data_source": summary["new_data_source"], "target": summary["target"],
        "recorder_uri": summary["recorder_uri"], "segments": summary["segments"],
        "costs_bps": summary["costs_bps"], "forecast_ic_floor": summary["forecast_ic_floor"],
        "causality_statement": summary["causality_statement"],
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    asian = [r for r in summary["information_coefficients"] if r["region"] == "asia"]
    us = [r for r in summary["information_coefficients"] if r["region"] == "us_reference"]
    fmt = lambda rows: "；".join(f"{r['market']}(同{_f(r['same_date_ic'])},滞后{_f(r['lagged_ic'])})" for r in rows)
    return (
        "# 跨市场领先-滞后研究（中文结论）\n\n"
        f"- 新数据：`{summary['new_data_source']}`，Yahoo Finance 日线收盘（2011-10-10 至 2026-10-08），"
        f"含台湾加权/台积电/韩国 SK 海力士/KOSPI/日经 与美股 SOXL/SOXX/SMH/NVDA/TSM/AMD/ASML。\n"
        f"- 方法：亚洲盘在美股同一交易所本地日开盘前收盘；只把严格滞后（t-1 信号对 t 美股）当作预测证据，"
        f"同日相关只作同期共振记录。预测 IC 门槛 |{summary['forecast_ic_floor']}|。\n"
        f"- 亚洲市场 IC：{fmt(asian)}。\n"
        f"- 美股参照 IC：{fmt(us)}。\n"
        f"- 判定为可预测（forecast_textbook）的市场：{summary['forecast_markets'] or '无'}。\n"
        f"- development 稳定超额：{summary['stable_excess_in_development'] or '无'}。\n"
        f"- 决定：`{summary['decision']}`；promotion_eligible={summary['promotion_eligible']}。\n"
        f"- 风险：仅日线收盘、未保留亚洲盘中时刻；同日共振不可交易，不计为预测能力；"
        f"无收益预测与拟合参数；本结果不构成生产或实盘证明。\n"
        f"- qlib Recorder：{summary.get('recorder_id')}；qlib docs：{QLIB_DOCS}。\n"
    )


def _f(value):
    return "NA" if value is None else f"{value:+.3f}"


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
