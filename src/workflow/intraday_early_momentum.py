"""Self-contained causal intraday early-momentum card over SOXL/SOXX/cash.

Uses only the qlib intraday provider, the observable opening 60 minutes, and the
same-session remainder. No parent model, no return forecast, no overnight risk.
Selection uses the development window only; diagnostic and forward windows are
evaluation-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.intraday_early_momentum import (
    MomentumPolicy,
    annual_wealth,
    benchmark_replay,
    build_early_session_frame,
    simulate_policy,
)
from src.analysis.stable_excess_frontier import (
    excess_versus_benchmark,
    risk_metrics,
    stable_excess_verdict,
)
from src.data import QlibProviderConfig, load_qlib_feature_frame

EXPERIMENT = "intraday_early_momentum_20261010"
SCHEMA_VERSION = "intraday_early_momentum_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class MomentumSpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")
    provider_uri: str = "data/qlib/soxl_intraday_15min"
    region: str = "us"
    market: str = "all"
    as_of: str = "2026-10-07"
    development: tuple[str, str] = ("2020-01-02", "2023-12-29")
    reused_diagnostic: tuple[str, str] = ("2024-01-02", "2026-06-30")
    forward: tuple[str, str] = ("2026-07-01", "2026-10-07")
    costs_bps: tuple[float, ...] = (10.0, 25.0)
    minimum_positive_years: int = 2

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def recorder_uri(self) -> Path:
        return self.card_dir / "qlib_recorder"


def resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def registered_candidates() -> list[MomentumPolicy]:
    """Bounded, pre-registered threshold family; no tuning after outcomes.

    Two signal sources: the target's own early move (``soxl_early``) and the
    underlying index proxy's early move (``soxx_early``). No return forecast.
    """
    policies = [MomentumPolicy(name="early_momentum_off", long_threshold=1.0, short_threshold=1.0)]
    for signal in ("soxl_early", "soxx_early"):
        tag = signal.replace("_early", "")
        for long_threshold in (0.005, 0.01, 0.02):
            policies.append(MomentumPolicy(
                name=f"{tag}_momentum_l{int(long_threshold * 1000)}",
                long_threshold=long_threshold, short_threshold=1.0, signal=signal))
    return policies


def _load_intraday(spec: MomentumSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = QlibProviderConfig(provider_uri=spec.root / spec.provider_uri, region=spec.region,
                                market=spec.market, freq="15min")
    raw = load_qlib_feature_frame(
        config, instruments=["SOXL", "SOXX"], fields=["$open", "$close"],
        start="2019-01-02", end=f"{spec.as_of} 16:00:00",
    )
    _require(not raw.empty, "intraday provider returned no bars in range")
    index = pd.DatetimeIndex(pd.to_datetime(raw.index.get_level_values("datetime")))
    if index.tz is None:
        index = index.tz_localize("America/New_York")
    else:
        index = index.tz_convert("America/New_York")

    def one(symbol):
        block = raw.xs(symbol, level=0)
        frame = pd.DataFrame({"open": block["$open"].to_numpy(float),
                              "close": block["$close"].to_numpy(float)},
                             index=pd.DatetimeIndex(pd.to_datetime(block.index)))
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("America/New_York")
        else:
            frame.index = frame.index.tz_convert("America/New_York")
        return frame
    audit = {
        "provider_uri": config.resolved_provider_uri.as_posix(),
        "market": spec.market, "region": spec.region, "freq": config.freq,
        "as_of": spec.as_of,
        "bars": int(len(raw)),
        "first_bar": str(index.min()),
        "last_bar": str(index.max()),
        "qlib_data_api": ["qlib.data.D.features"],
        "qlib_docs_reference": QLIB_DOCS,
    }
    return one("SOXL"), one("SOXX"), audit


def _segment(frame: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["signal_date"]))
    start = pd.Timestamp(window[0])
    end = pd.Timestamp(window[1]) + pd.Timedelta(days=1)
    if dates.tz is not None:
        start = start.tz_localize(dates.tz)
        end = end.tz_localize(dates.tz)
    mask = (dates >= start) & (dates < end)
    return frame.loc[mask].reset_index(drop=True)


def _evaluate(session_frame: pd.DataFrame, policy: MomentumPolicy, *, window: tuple[str, str],
              cost_bps: float, minimum_positive_years: int) -> dict:
    segment = _segment(session_frame, window)
    _require(len(segment) > 1, f"window {window} has too few sessions")
    replay = simulate_policy(segment, policy, cost_bps=cost_bps)
    benchmark = benchmark_replay(segment, cost_bps=cost_bps)
    excess = excess_versus_benchmark(replay, benchmark)
    annual = annual_wealth(replay)
    bench_annual = annual_wealth(benchmark)
    merged = annual.merge(bench_annual, on="year", suffixes=("", "_bench"))
    merged["wealth_ratio_vs_soxx"] = merged["wealth"] / merged["wealth_bench"]
    verdict = stable_excess_verdict(excess, merged, minimum_positive_years=minimum_positive_years)
    return {"excess": excess, "annual": merged, "verdict": verdict, "sessions": int(len(replay))}


def select_development_candidate(results: dict[str, dict]) -> str:
    ranked = []
    for name, result in results.items():
        if name == "early_momentum_off":
            continue
        verdict = result["verdict"]
        ranked.append((
            0 if verdict["stable_excess"] else 1,
            -verdict["positive_year_count"],
            -result["excess"]["sharpe_difference"],
            -result["excess"]["max_drawdown_improvement_pct"],
            name,
        ))
    _require(ranked, "no active momentum candidate was evaluated")
    ranked.sort()
    return ranked[0][-1]


def run_intraday_early_momentum(*, root: Path | str | None = None, card_dir: Path | str | None = None,
                                recorder_uri: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = MomentumSpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else spec.card_dir
    target_recorder = Path(recorder_uri) if recorder_uri is not None else (target_card / "qlib_recorder")
    target_card.mkdir(parents=True, exist_ok=True)

    soxl, soxx, data_audit = _load_intraday(spec)
    sessions = build_early_session_frame(soxl, soxx).frame
    candidates = registered_candidates()
    for policy in candidates:
        policy.validate()

    dev_results = {
        policy.name: _evaluate(sessions, policy, window=spec.development, cost_bps=spec.costs_bps[0],
                               minimum_positive_years=spec.minimum_positive_years)
        for policy in candidates
    }
    selected_name = select_development_candidate(dev_results)
    selected = next(policy for policy in candidates if policy.name == selected_name)

    windows = {"development": spec.development, "reused_diagnostic": spec.reused_diagnostic,
               "forward": spec.forward}
    window_results = {
        str(cost): {
            window: {policy.name: _evaluate(sessions, policy, window=span, cost_bps=cost,
                                            minimum_positive_years=spec.minimum_positive_years)
                     for policy in candidates}
            for window, span in windows.items()
        }
        for cost in spec.costs_bps
    }
    primary = str(spec.costs_bps[0])
    high = str(spec.costs_bps[1])
    promotion_eligible = bool(
        dev_results[selected_name]["verdict"]["stable_excess"]
        and window_results[primary]["reused_diagnostic"][selected_name]["verdict"]["stable_excess"]
        and window_results[high]["reused_diagnostic"][selected_name]["verdict"]["stable_excess"]
    )
    summary = _build_summary(
        spec=spec, data_audit=data_audit, selected=selected, candidates=candidates,
        dev_results=dev_results, window_results=window_results, promotion_eligible=promotion_eligible)
    summary.update(_record(spec=spec, card_dir=target_card, recorder_uri=target_recorder, summary=summary))
    return summary


def _build_summary(*, spec, data_audit, selected, candidates, dev_results, window_results,
                   promotion_eligible) -> dict[str, Any]:
    primary = str(spec.costs_bps[0])
    comparison = []
    for window, results in window_results[primary].items():
        for name, result in results.items():
            comparison.append({
                "window": window, "policy": name, "sessions": result["sessions"],
                **result["excess"]["strategy"],
                "benchmark_return_pct": result["excess"]["benchmark"]["return_pct"],
                "benchmark_max_drawdown_pct": result["excess"]["benchmark"]["max_drawdown_pct"],
                "benchmark_sharpe": result["excess"]["benchmark"]["sharpe"],
                "sharpe_difference": result["excess"]["sharpe_difference"],
                "max_drawdown_improvement_pct": result["excess"]["max_drawdown_improvement_pct"],
                "stable_excess": result["verdict"]["stable_excess"],
                "positive_year_count": result["verdict"]["positive_year_count"],
                "year_count": result["verdict"]["year_count"],
                "average_soxl_weight": result["excess"]["average_soxl_weight"],
                "average_cash_weight": result["excess"]["average_cash_weight"],
                "total_turnover": result["excess"]["total_turnover"],
            })
    annual_records = []
    for window in ("development", "reused_diagnostic", "forward"):
        annual = window_results[primary][window][selected.name]["annual"].copy()
        annual.insert(0, "window", window)
        annual.insert(1, "policy", selected.name)
        annual_records.append(annual)
    return {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "active_research",
        "decision": ("stable_excess_candidate_for_forward_shadow" if promotion_eligible
                     else "no_stable_intraday_excess_in_registered_family"),
        "promotion_eligible": promotion_eligible,
        "objective_achieved": promotion_eligible,
        "provider_uri": data_audit["provider_uri"],
        "market": spec.market, "region": spec.region, "as_of": spec.as_of,
        "data_audit": data_audit,
        "costs_bps": list(spec.costs_bps),
        "segments": {"development": list(spec.development),
                     "reused_diagnostic": list(spec.reused_diagnostic),
                     "forward": list(spec.forward)},
        "candidate_family": [policy.__dict__ for policy in candidates],
        "candidate_count": len(candidates),
        "selected_policy": selected.name,
        "selected_model": selected.__dict__,
        "return_forecast_used": False,
        "model_fits": 0,
        "overnight_position": False,
        "development_selection": {
            name: {"stable_excess": result["verdict"]["stable_excess"],
                   "positive_year_count": result["verdict"]["positive_year_count"],
                   "sharpe_difference": result["excess"]["sharpe_difference"]}
            for name, result in dev_results.items()},
        "window_comparison": comparison,
        "selected_outer": window_results[primary]["reused_diagnostic"][selected.name]["excess"],
        "selected_outer_verdict": window_results[primary]["reused_diagnostic"][selected.name]["verdict"],
        "selected_forward": window_results[primary]["forward"][selected.name]["excess"],
        "selected_forward_verdict": window_results[primary]["forward"][selected.name]["verdict"],
        "annual_metrics": pd.concat(annual_records, ignore_index=True).to_dict(orient="records"),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                "data/session_frame.pkl",
                "analysis/development_results.pkl",
                "analysis/window_results.pkl",
                "analysis/selected_replays.pkl",
            ],
        },
        "limitations": [
            "Entry at 10:30 assumes the decision close is fillable; bid/ask, queue and latency are not modeled.",
            "Only the 10:30-to-close window is traded; overnight and morning exposure are deliberately absent.",
            "The forward window is only the newly available sessions and is statistically weak.",
            "No return forecast or fitted parameter is used.",
        ],
    }


def _record(*, spec, card_dir, recorder_uri, summary) -> dict[str, Any]:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R

    recorder_uri = Path(recorder_uri).resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / "data" / "qlib" / "soxl").resolve()), region=spec.region)
    with R.start(experiment_name=spec.experiment, recorder_name="momentum", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(
            schema_version=SCHEMA_VERSION, provider_uri=summary["provider_uri"], as_of=spec.as_of,
            costs_bps=str(list(spec.costs_bps)), selected_policy=summary["selected_policy"],
            candidate_count=summary["candidate_count"])
        recorder.log_metrics(
            forward_return_pct=summary["selected_forward"]["strategy"]["return_pct"],
            forward_sharpe=summary["selected_forward"]["strategy"]["sharpe"],
            outer_return_pct=summary["selected_outer"]["strategy"]["return_pct"],
            outer_max_drawdown_pct=summary["selected_outer"]["strategy"]["max_drawdown_pct"],
            outer_sharpe=summary["selected_outer"]["strategy"]["sharpe"])
        recorder.save_objects(summary_json=summary, selected_policy=summary["selected_model"])
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
    comparison.to_csv(card_dir / "policy_comparison.csv", index=False)
    comparison[comparison["window"].eq("reused_diagnostic")].to_csv(
        card_dir / "backtest_benchmark_comparison.csv", index=False)
    pd.DataFrame(summary["annual_metrics"]).to_csv(card_dir / "annual_metrics.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "provider_uri": summary["provider_uri"], "region": summary["region"], "market": summary["market"],
        "as_of": summary["as_of"], "recorder_uri": summary["recorder_uri"],
        "segments": summary["segments"], "costs_bps": summary["costs_bps"],
        "allocation_window": "10:30_to_close", "overnight_position": False,
        "selection": "development_only_stable_excess_rank",
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    outer = summary["selected_outer"]["strategy"]
    forward = summary["selected_forward"]["strategy"]
    bench_outer = summary["selected_outer"]["benchmark"]
    bench_forward = summary["selected_forward"]["benchmark"]
    return (
        f"# 日内早盘动量稳定超额研究（中文结论）\n\n"
        f"- 做了什么：仅用 qlib 原生 15 分钟 provider，把开盘 60 分钟（09:30–10:30）可观测路径映射到当日 10:30–收盘的 "
        f"SOXL/SOXX/现金仓位；{summary['candidate_count']} 条预登记阈值规则，无收益预测、无拟合参数、无隔夜仓；"
        f"选择仅使用 development {summary['segments']['development']}。\n"
        f"- 核心指标（选中 {summary['selected_policy']}）：复用区间 {summary['segments']['reused_diagnostic']} "
        f"收益 {outer['return_pct']:.2f}%（SOXX {bench_outer['return_pct']:.2f}%），"
        f"最大回撤 {outer['max_drawdown_pct']:.2f}%（SOXX {bench_outer['max_drawdown_pct']:.2f}%），"
        f"Sharpe {outer['sharpe']:.2f}（SOXX {bench_outer['sharpe']:.2f}）。\n"
        f"- 新前向区间 {summary['segments']['forward']}：收益 {forward['return_pct']:.2f}%"
        f"（SOXX {bench_forward['return_pct']:.2f}%），回撤 {forward['max_drawdown_pct']:.2f}%，"
        f"Sharpe {forward['sharpe']:.2f}。\n"
        f"- 是否晋升：{summary['promotion_eligible']}；决定 `{summary['decision']}`。\n"
        f"- 风险：10:30 决策价可成交性未验证（无买卖价差/排队/延迟）；仅交易日内余下时段；"
        f"前向样本短；本结果不构成生产或实盘证明。\n"
        f"- 下一步：只有 development、复用诊断与高成本压力同时通过稳定超额才登记冻结参数与独立前向 shadow。\n"
        f"- qlib Recorder：{summary.get('recorder_id')}；provider_uri={summary['provider_uri']}；"
        f"as_of={summary['as_of']}；qlib docs：{QLIB_DOCS}。\n"
    )
