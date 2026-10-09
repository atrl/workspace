"""Cross-source causal signal search for a stable SOXL excess over SOXX.

Assembles every locally available causal signal family (own and SOXX reversal,
realized volatility, trend, constituent breadth/dispersion, external
semiconductor leadership) and evaluates a bounded set of exposure rules with a
single consistent, honest criterion: beat SOXX buy-and-hold on return, drawdown
and Sharpe with a majority of positive years.

Development-only selection; reused diagnostic and new forward are evaluation.
No return forecast, no fitted parameter, no future information.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.cross_source_signal_search import (
    SignalSpec,
    evaluate_signal,
    summarize_search,
)
from src.data import QlibProviderConfig, load_qlib_feature_frame

EXPERIMENT = "cross_source_signal_search_20261010"
SCHEMA_VERSION = "cross_source_signal_search_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
CONSTITUENTS = ["nvda", "tsm", "avgo", "amd", "asml", "qcom", "mu", "intc", "txn",
                "amat", "klac", "lrcx", "adi", "mchp", "nxpi", "on", "mrvl", "ter"]


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class SearchSpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")
    as_of: str = "2026-07-02"
    development: tuple[str, str] = ("2014-01-02", "2022-12-30")
    reused_diagnostic: tuple[str, str] = ("2023-01-03", "2026-07-02")
    forward: tuple[str, str] = ("2026-07-03", "2026-10-08")
    costs_bps: tuple[float, ...] = (10.0, 25.0)
    minimum_positive_years: int = 4
    thresholds: tuple[float, ...] = (0.2, 0.35, 0.5)
    directions: tuple[int, ...] = (1, -1)

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


def _load_series(spec: SearchSpec) -> dict[str, pd.Series]:
    daily = QlibProviderConfig(provider_uri=spec.root / "data/qlib/soxl", region="us", freq="day")
    g = load_qlib_feature_frame(daily, instruments=["SOXL", "SOXX"],
                                fields=["$close"], start="2010-03-11", end=spec.forward[1])
    close = g["$close"].unstack(level=0)
    close.index = pd.to_datetime(close.index)
    soxl_ret = close["SOXL"].pct_change()
    soxx_ret = close["SOXX"].pct_change()

    cons = QlibProviderConfig(provider_uri=spec.root / "data/qlib/soxx_constituents",
                              region="us", freq="day")
    cf = load_qlib_feature_frame(cons, instruments=CONSTITUENTS,
                                 fields=["$close", "$weight_pct"],
                                 start="2010-03-11", end=spec.as_of)
    c_close = cf["$close"].unstack(level=0)
    c_weight = cf["$weight_pct"].unstack(level=0)
    for frame in (c_close, c_weight):
        frame.index = pd.to_datetime(frame.index)
    c_ret = c_close.pct_change()

    cap_mkt = (c_ret * c_weight).sum(axis=1) / c_weight.sum(axis=1)
    breadth = (c_ret > 0).mean(axis=1) - 0.5
    dispersion = c_ret.std(axis=1)
    soxx_vol = soxx_ret.rolling(20).std() * np.sqrt(252)
    soxx_trend = close["SOXX"] / close["SOXX"].rolling(200).mean() - 1.0

    # Intraday 1-minute-derived features from the qlib-native soxl_intraday_t1 store.
    intraday_fields = {
        "first_hour_return": "$intraday_soxl_first_hour_return",
        "close_location": "$intraday_soxl_close_location",
        "realized_vol_1m": "$intraday_soxl_realized_vol_1m",
        "drawdown_from_open": "$intraday_soxl_drawdown_from_open",
        "amplitude": "$intraday_soxl_amplitude",
        "late_return_30m": "$intraday_soxl_late_return_30m",
        "close_vwap_deviation": "$intraday_soxl_close_vwap_deviation",
        "volume_concentration_close_30m": "$intraday_soxl_volume_concentration_close_30m",
    }
    t1 = QlibProviderConfig(provider_uri=spec.root / "data/qlib/soxl_intraday_t1", region="us", freq="day")
    intraday_raw = load_qlib_feature_frame(t1, instruments=["soxl"], fields=list(intraday_fields.values()),
                                           start="2019-01-02", end=spec.forward[1])
    intraday = intraday_raw.droplevel(0) if isinstance(intraday_raw.index, pd.MultiIndex) else intraday_raw
    intraday.index = pd.to_datetime(intraday.index)
    intraday = intraday.rename(columns={v: k for k, v in intraday_fields.items()})

    # Newly collected macro / volatility levels (FRED + CBOE), daily closes.
    macro_path = spec.root / "data/raw/external/macro_vol_20261010_v1/daily_levels.csv"
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True).sort_index()
    macro_signal = pd.DataFrame({
        "vix_level": macro["VIXCLS"],
        "vix_term_slope": macro["CBOE_VIX9D"] / macro["CBOE_VIX3M"] - 1.0,
        "credit_spread": macro["BAMLH0A0HYM2"],
        "dollar_index": macro["DTWEXBGS"],
        "ten_year_yield": macro["DGS10"],
        "semiconductor_index": macro["NASDAQSOX"],
    })

    # Align every series to the union calendar so a single boolean mask is valid.
    index = soxl_ret.index.union(cap_mkt.index).union(soxx_vol.index).union(intraday.index)
    aligned = pd.DataFrame({
        "soxl_return": soxl_ret,
        "soxx_return": soxx_ret,
        "soxl_reversal": -soxl_ret,
        "soxx_reversal": -soxx_ret,
        "leader_capmkt_reversal": -cap_mkt,
        "leader_breadth_reversal": -breadth,
        "leader_dispersion": dispersion,
        "realized_vol": soxx_vol,
        "trend": soxx_trend,
    }, index=index)
    for column in intraday.columns:
        aligned[f"intraday_{column}"] = intraday[column]
    for column in macro_signal.columns:
        aligned[f"macro_{column}"] = macro_signal[column]
    aligned = aligned.sort_index()
    return {column: aligned[column] for column in aligned.columns}


def registered_signals(series: dict[str, pd.Series]) -> list[SignalSpec]:
    """Bounded, pre-registered signal family across every local data source."""
    families = [
        ("soxl_reversal", series["soxl_reversal"], "soxl_daily"),
        ("soxx_reversal", series["soxx_reversal"], "soxx_daily"),
        ("leader_capmkt_reversal", series["leader_capmkt_reversal"], "soxx_constituents"),
        ("leader_breadth_reversal", series["leader_breadth_reversal"], "soxx_constituents"),
        ("realized_vol", series["realized_vol"], "soxx_daily"),
        ("trend", series["trend"], "soxx_daily"),
        ("intraday_early_momentum", series["intraday_first_hour_return"], "soxl_intraday_t1"),
        ("intraday_close_location", series["intraday_close_location"], "soxl_intraday_t1"),
        ("intraday_realized_vol", series["intraday_realized_vol_1m"], "soxl_intraday_t1"),
        # Newly collected FRED/CBOE macro and volatility levels.
        ("vix_level", series["macro_vix_level"], "macro_vol_20261010"),
        ("vix_term_slope", series["macro_vix_term_slope"], "macro_vol_20261010"),
        ("credit_spread", series["macro_credit_spread"], "macro_vol_20261010"),
        ("dollar_index", series["macro_dollar_index"], "macro_vol_20261010"),
        ("ten_year_yield", series["macro_ten_year_yield"], "macro_vol_20261010"),
        ("semiconductor_index", series["macro_semiconductor_index"], "macro_vol_20261010"),
    ]
    signals = []
    for name, value, source in families:
        for direction in (1, -1):
            for threshold in (0.35, 0.5):
                signals.append(SignalSpec(
                    name=f"{name}_d{direction:+d}_q{int(threshold * 100)}",
                    signal=value, direction=direction, threshold=threshold, source=source))
    return signals


def run_cross_source_signal_search(*, root: Path | str | None = None,
                                   card_dir: Path | str | None = None,
                                   recorder_uri: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = SearchSpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else spec.card_dir
    target_recorder = Path(recorder_uri) if recorder_uri is not None else (target_card / "qlib_recorder")
    target_card.mkdir(parents=True, exist_ok=True)

    series = _load_series(spec)
    candidates = registered_signals(series)
    for candidate in candidates:
        candidate.validate()

    window_results: dict[str, dict] = {}
    for cost in spec.costs_bps:
        window_results[str(cost)] = {}
        for window, span in (("development", spec.development),
                             ("reused_diagnostic", spec.reused_diagnostic),
                             ("forward", spec.forward)):
            mask = (series["soxl_return"].index >= pd.Timestamp(span[0])) & \
                   (series["soxl_return"].index <= pd.Timestamp(span[1]))
            sliced = {k: v[mask] for k, v in series.items()}
            results = {}
            for candidate in candidates:
                replay = evaluate_signal(candidate.signal[mask], sliced["soxl_return"],
                                         sliced["soxx_return"], direction=candidate.direction,
                                         threshold=candidate.threshold, cost_bps=cost)
                if replay is not None:
                    results[candidate.name] = replay
            window_results[str(cost)][window] = summarize_search(
                results, minimum_positive_years=spec.minimum_positive_years)

    dev = window_results[str(spec.costs_bps[0])]["development"]
    selected = dev[dev["stable_excess"]].head(1)
    selected_name = selected["signal"].iloc[0] if len(selected) else None

    summary = _build_summary(spec=spec, series=series, candidates=candidates,
                             window_results=window_results, selected_name=selected_name)
    summary.update(_record(spec=spec, recorder_uri=target_recorder, summary=summary))
    return summary


def _build_summary(*, spec, series, candidates, window_results, selected_name) -> dict[str, Any]:
    primary = str(spec.costs_bps[0])
    comparison = []
    for cost, windows in window_results.items():
        for window, table in windows.items():
            block = table.copy()
            block.insert(0, "cost_bps", float(cost))
            block.insert(1, "window", window)
            comparison.append(block)
    comparison = pd.concat(comparison, ignore_index=True)
    stable_signals = sorted(set(comparison.loc[comparison["stable_excess"], "signal"]))
    selected_block = {}
    if selected_name is not None:
        sel = comparison[(comparison["signal"] == selected_name) & (comparison["cost_bps"] == float(primary))]
        selected_block = sel.to_dict(orient="records")
    return {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "active_research",
        "decision": ("stable_excess_signal_found_in_family" if selected_name
                     else "no_stable_excess_across_sources"),
        "promotion_eligible": bool(selected_name),
        "objective_achieved": False,
        "as_of": spec.as_of,
        "sources": [
            "data/qlib/soxl (SOXL/SOXX daily, 2010-2026-10-08)",
            "data/qlib/soxx_constituents (30 semiconductor daily, 2010-2026-07-02)",
            "data/qlib/soxl_intraday_t1 (SOXL 1-minute-derived daily features, 2019-2026)",
            "data/raw/external/macro_vol_20261010_v1 (FRED+CBOE VIX term, credit, FX, rates, SOX index)",
            "data/raw/external/asia_semis_20261010_v1 (Asian semiconductor markets, dedicated lead-lag card)",
        ],
        "excluded_sources": [
            {"source": "data/raw/external futu capital flow intraday",
             "reason": "single session only (2026-05-22); not a time series"},
            {"source": "data/qlib/soxl_external_intraday_15min",
             "reason": "covered by the dedicated intraday_early_momentum card"},
        ],
        "costs_bps": list(spec.costs_bps),
        "segments": {"development": list(spec.development),
                     "reused_diagnostic": list(spec.reused_diagnostic),
                     "forward": list(spec.forward)},
        "candidate_count": len(candidates),
        "signal_family": [candidate.name for candidate in candidates],
        "signal_sources": sorted({candidate.source for candidate in candidates}),
        "selected_signal": selected_name,
        "selected_block": selected_block,
        "stable_signals_in_family": stable_signals,
        "return_forecast_used": False,
        "model_fits": 0,
        "window_comparison": comparison.to_dict(orient="records"),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                "data/series.pkl",
                "analysis/window_results.pkl",
                "analysis/summary_table.csv",
            ],
        },
        "limitations": [
            "The cross-source search is diagnostic; a hit must survive an independent forward window.",
            "Constituent history ends 2026-07-02, so leader signals do not cover the forward window.",
            "No return forecast or fitted parameter is used; thresholds are expanding-window quantiles.",
            "SOXL realized leverage decay is only reflected in its traded prices.",
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
    with R.start(experiment_name=spec.experiment, recorder_name="search", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(
            schema_version=SCHEMA_VERSION, as_of=spec.as_of, costs_bps=str(list(spec.costs_bps)),
            candidate_count=summary["candidate_count"],
            selected_signal=str(summary["selected_signal"]))
        recorder.log_metrics(
            stable_signal_count=len(summary["stable_signals_in_family"]),
            candidate_count=summary["candidate_count"])
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
    comparison = pd.DataFrame(summary["window_comparison"])
    comparison.to_csv(card_dir / "signal_search.csv", index=False)
    comparison[comparison["window"].eq("reused_diagnostic")].to_csv(
        card_dir / "backtest_benchmark_comparison.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "as_of": summary["as_of"], "recorder_uri": summary["recorder_uri"],
        "sources": summary["sources"], "segments": summary["segments"],
        "costs_bps": summary["costs_bps"], "selection": "development_only_stable_excess",
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    lines = [
        "# 跨数据源因果信号搜索（中文结论）\n",
        f"- 做了什么：在 qlib 原生链路上，用 {summary['candidate_count']} 条预登记因果暴露规则，"
        f"覆盖 {len(summary['signal_sources'])} 类数据源（{', '.join(summary['signal_sources'])}）；"
        f"阈值用扩展窗口分位，无收益预测、无拟合参数。\n",
        f"- 选择：仅使用 development {summary['segments']['development']} 的稳定超额（收益+回撤+Sharpe 同超）。\n",
        f"- 选中信号：{summary['selected_signal']}。\n",
        f"- development 稳定超额信号：{summary['stable_signals_in_family'] or '无'}。\n",
        f"- 决定：`{summary['decision']}`；promotion_eligible={summary['promotion_eligible']}。\n",
        f"- 风险：诊断搜索可能过拟合；命中必须经独立前向窗口验证；成分股历史止于 2026-07-02；"
        f"SOXL 杠杆衰减只由真实价格体现；本结果不构成生产或实盘证明。\n",
        f"- qlib Recorder：{summary.get('recorder_id')}；as_of={summary['as_of']}；qlib docs：{QLIB_DOCS}。\n",
    ]
    return "".join(lines)
