"""Stable-excess frontier research over causal, non-leveraged SOXX/SOXL policies.

The card asks whether any pre-registered, past-only exposure rule delivers
stable excess over SOXX buy-and-hold on return, drawdown and Sharpe together.
Return information is deliberately absent: forecasts were shown to carry no
stable IC, so this card isolates observable risk and trend structure instead.

Selection uses only the development walk-forward window. The reused 2023-2026
diagnostic window and any forward extension never influence selection.
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

from src.analysis.stable_excess_frontier import (
    ExposurePolicy,
    annual_excess,
    build_state,
    excess_versus_benchmark,
    risk_metrics,
    simulate_policy,
    stable_excess_verdict,
)
from src.data import QlibProviderConfig, load_qlib_feature_frame

EXPERIMENT = "soxl_stable_excess_frontier_20261010"
SCHEMA_VERSION = "soxl_stable_excess_frontier_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class FrontierSpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")
    provider_uri: str = "data/qlib/soxl"
    region: str = "us"
    market: str = "all"
    as_of: str = "2026-10-08"
    # Development is the only window that may influence selection.
    development: tuple[str, str] = ("2016-01-04", "2022-12-30")
    # Reused diagnostic and new forward window are evaluation-only.
    reused_diagnostic: tuple[str, str] = ("2023-01-03", "2026-09-25")
    forward: tuple[str, str] = ("2026-09-26", "2026-10-08")
    costs_bps: tuple[float, ...] = (10.0, 25.0)
    minimum_positive_years: int = 4

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


def registered_candidates() -> list[ExposurePolicy]:
    """The complete, pre-registered, bounded candidate family.

    Two benchmark-relative controls plus a small grid of causal exposure rules.
    No candidate uses a return forecast or a fitted alpha.
    """
    candidates = [
        ExposurePolicy(name="soxx_buy_hold", kind="soxx_buy_hold"),
    ]
    for target_vol in (0.15, 0.20, 0.25, 0.30):
        candidates.append(
            ExposurePolicy(name=f"vol_target_soxx_v{int(target_vol * 100)}", kind="vol_target_soxx",
                           target_vol=target_vol)
        )
    for vol_cap in (0.20, 0.25, 0.30, 0.35):
        candidates.append(
            ExposurePolicy(name=f"trend_vol_switch_v{int(vol_cap * 100)}", kind="trend_vol_switch",
                           vol_cap=vol_cap)
        )
    return candidates


def _load_prices(spec: FrontierSpec) -> tuple[pd.DataFrame, dict]:
    config = QlibProviderConfig(provider_uri=spec.root / spec.provider_uri, region=spec.region, market=spec.market)
    raw = load_qlib_feature_frame(
        config,
        instruments=["SOXL", "SOXX"],
        fields=["$close", "$open"],
        start="2010-03-11",
        end=spec.as_of,
    )
    close = raw["$close"].unstack(level=0)
    _require({"SOXL", "SOXX"}.issubset(close.columns), "provider is missing SOXL/SOXX closes")
    frame = pd.DataFrame({
        "signal_date": pd.to_datetime(close.index),
        "soxl_close": close["SOXL"].to_numpy(float),
        "soxx_close": close["SOXX"].to_numpy(float),
    }).dropna().reset_index(drop=True)
    audit = {
        "provider_uri": config.resolved_provider_uri.as_posix(),
        "market": spec.market,
        "region": spec.region,
        "freq": config.freq,
        "as_of": spec.as_of,
        "rows": int(len(frame)),
        "first_session": str(frame["signal_date"].min().date()),
        "last_session": str(frame["signal_date"].max().date()),
        "qlib_data_api": ["qlib.data.D.features"],
        "qlib_docs_reference": QLIB_DOCS,
    }
    return frame, audit


def _segment(state: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    dates = state["signal_date"]
    mask = dates.ge(pd.Timestamp(window[0])) & dates.le(pd.Timestamp(window[1]))
    return state.loc[mask].reset_index(drop=True)


def _benchmark_replay(state: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    return simulate_policy(state, ExposurePolicy(name="soxx_buy_hold", kind="soxx_buy_hold"), cost_bps=cost_bps)


def evaluate_window(state: pd.DataFrame, policy: ExposurePolicy, *, window: tuple[str, str],
                    cost_bps: float, minimum_positive_years: int) -> dict:
    segment = _segment(state, window)
    _require(len(segment) > 1, f"window {window} has too few sessions")
    replay = simulate_policy(segment, policy, cost_bps=cost_bps)
    benchmark = _benchmark_replay(segment, cost_bps)
    # Align policy and benchmark on common sessions after the terminal shift row.
    replay_aligned = replay.reset_index(drop=True)
    bench_aligned = benchmark.reset_index(drop=True)
    common = replay_aligned["signal_date"].isin(bench_aligned["signal_date"])
    replay_aligned = replay_aligned.loc[common].reset_index(drop=True)
    bench_aligned = bench_aligned.loc[
        bench_aligned["signal_date"].isin(replay_aligned["signal_date"])
    ].reset_index(drop=True)
    _require(len(replay_aligned) == len(bench_aligned), "policy and benchmark sessions must align")
    excess = excess_versus_benchmark(replay_aligned, bench_aligned)
    annual = annual_excess(replay_aligned, bench_aligned)
    verdict = stable_excess_verdict(excess, annual, minimum_positive_years=minimum_positive_years)
    return {"excess": excess, "annual": annual, "verdict": verdict, "sessions": int(len(replay_aligned))}


def select_development_candidate(development_results: dict[str, dict]) -> str:
    """Pick the development candidate that best satisfies stable excess.

    Ranking is deterministic and uses only development metrics. The benchmark is
    never selected as a promotion candidate.
    """
    ranked = []
    for name, result in development_results.items():
        if name == "soxx_buy_hold":
            continue
        verdict = result["verdict"]
        ranked.append((
            0 if verdict["stable_excess"] else 1,
            -verdict["positive_year_count"],
            -result["excess"]["sharpe_difference"],
            -result["excess"]["max_drawdown_improvement_pct"],
            name,
        ))
    _require(ranked, "no non-benchmark development candidates were evaluated")
    ranked.sort()
    return ranked[0][-1]


def run_stable_excess_frontier(*, root: Path | str | None = None, card_dir: Path | str | None = None,
                               recorder_uri: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = FrontierSpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else spec.card_dir
    target_recorder = Path(recorder_uri) if recorder_uri is not None else (target_card / "qlib_recorder")
    target_card.mkdir(parents=True, exist_ok=True)

    prices, data_audit = _load_prices(spec)
    state = build_state(prices, vol_window=20, trend_window=200)

    candidates = registered_candidates()
    for policy in candidates:
        policy.validate()

    development = _segment(state, spec.development)
    _require(len(development) > 1, "development window has too few sessions")
    dev_results = {
        policy.name: evaluate_window(state, policy, window=spec.development, cost_bps=spec.costs_bps[0],
                                     minimum_positive_years=spec.minimum_positive_years)
        for policy in candidates
    }
    selected_name = select_development_candidate(dev_results)
    selected_policy = next(policy for policy in candidates if policy.name == selected_name)

    windows = {
        "development": spec.development,
        "reused_diagnostic": spec.reused_diagnostic,
        "forward": spec.forward,
    }
    window_results: dict[str, dict[str, dict]] = {}
    for cost_bps in spec.costs_bps:
        window_results[str(cost_bps)] = {
            window: {
                policy.name: evaluate_window(state, policy, window=span, cost_bps=cost_bps,
                                             minimum_positive_years=spec.minimum_positive_years)
                for policy in candidates
            }
            for window, span in windows.items()
        }

    forward_state = _segment(state, spec.forward)
    forward_replay = simulate_policy(forward_state, selected_policy, cost_bps=spec.costs_bps[0])
    forward_excess = excess_versus_benchmark(
        forward_replay,
        _benchmark_replay(forward_state, spec.costs_bps[0]).assign(
            soxl_weight=0.0, soxx_weight=1.0, cash_weight=0.0),
    )
    forward_verdict = stable_excess_verdict(
        forward_excess,
        annual_excess(forward_replay, _benchmark_replay(forward_state, spec.costs_bps[0]).assign(
            soxl_weight=0.0, soxx_weight=1.0, cash_weight=0.0)),
        minimum_positive_years=1,
    )

    high_cost = str(spec.costs_bps[1])
    selected_high_cost = window_results[high_cost]["reused_diagnostic"][selected_name]
    selected_outer = window_results[str(spec.costs_bps[0])]["reused_diagnostic"][selected_name]
    promotion_eligible = bool(
        dev_results[selected_name]["verdict"]["stable_excess"]
        and selected_outer["verdict"]["stable_excess"]
        and selected_high_cost["verdict"]["stable_excess"]
    )
    annual_records = []
    for window in windows:
        annual = window_results[str(spec.costs_bps[0])][window][selected_name]["annual"].copy()
        annual.insert(0, "window", window)
        annual.insert(1, "policy", selected_name)
        annual_records.append(annual)
    annual_metrics = pd.concat(annual_records, ignore_index=True).to_dict(orient="records")

    summary = _build_summary(
        spec=spec,
        data_audit=data_audit,
        selected_policy=selected_policy,
        candidates=candidates,
        dev_results=dev_results,
        window_results=window_results,
        forward_excess=forward_excess,
        forward_verdict=forward_verdict,
        promotion_eligible=promotion_eligible,
        annual_metrics=annual_metrics,
    )
    recorder_info = _record(spec=spec, card_dir=target_card, recorder_uri=target_recorder,
                            summary=summary, selected_policy=selected_policy)
    summary.update(recorder_info)
    return summary


def _build_summary(*, spec, data_audit, selected_policy, candidates, dev_results, window_results,
                   forward_excess, forward_verdict, promotion_eligible, annual_metrics) -> dict[str, Any]:
    primary_cost = str(spec.costs_bps[0])
    decision = (
        "stable_excess_candidate_for_forward_shadow"
        if promotion_eligible
        else "no_stable_excess_in_registered_family"
    )
    comparison_table = []
    for window, results in window_results[primary_cost].items():
        for name, result in results.items():
            comparison_table.append({
                "window": window,
                "policy": name,
                "sessions": result["sessions"],
                **result["excess"]["strategy"],
                "benchmark_return_pct": result["excess"]["benchmark"]["return_pct"],
                "benchmark_max_drawdown_pct": result["excess"]["benchmark"]["max_drawdown_pct"],
                "sharpe_difference": result["excess"]["sharpe_difference"],
                "max_drawdown_improvement_pct": result["excess"]["max_drawdown_improvement_pct"],
                "stable_excess": result["verdict"]["stable_excess"],
                "positive_year_count": result["verdict"]["positive_year_count"],
                "year_count": result["verdict"]["year_count"],
                "average_soxl_weight": result["excess"]["average_soxl_weight"],
                "average_cash_weight": result["excess"]["average_cash_weight"],
            })
    return {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "active_research",
        "objective_achieved": bool(forward_verdict["beats_return_and_drawdown"] and promotion_eligible),
        "decision": decision,
        "promotion_eligible": promotion_eligible,
        "provider_uri": data_audit["provider_uri"],
        "market": spec.market,
        "region": spec.region,
        "as_of": spec.as_of,
        "data_audit": data_audit,
        "costs_bps": list(spec.costs_bps),
        "segments": {
            "development": list(spec.development),
            "reused_diagnostic": list(spec.reused_diagnostic),
            "forward": list(spec.forward),
        },
        "candidate_family": [policy.__dict__ for policy in candidates],
        "candidate_count": len(candidates),
        "selected_model": selected_policy.__dict__,
        "selected_policy": selected_policy.name,
        "return_forecast_used": False,
        "model_fits": 0,
        "development_selection": {
            name: {
                "stable_excess": result["verdict"]["stable_excess"],
                "positive_year_count": result["verdict"]["positive_year_count"],
                "sharpe_difference": result["excess"]["sharpe_difference"],
                "max_drawdown_improvement_pct": result["excess"]["max_drawdown_improvement_pct"],
            }
            for name, result in dev_results.items()
        },
        "window_comparison": comparison_table,
        "selected_outer_10bps": window_results[primary_cost]["reused_diagnostic"][selected_policy.name]["excess"],
        "selected_outer_verdict": window_results[primary_cost]["reused_diagnostic"][selected_policy.name]["verdict"],
        "selected_forward": forward_excess,
        "selected_forward_verdict": forward_verdict,
        "annual_metrics": annual_metrics,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                "data/prices.pkl",
                "data/state.pkl",
                "analysis/development_results.pkl",
                "analysis/window_results.pkl",
                "analysis/forward_replay.pkl",
            ],
        },
        "limitations": [
            "Forward window is only the newly available sessions and is statistically weak.",
            "No return forecast is used; this card isolates risk and trend structure only.",
            "SOXL realized leverage decay is not modeled beyond its actual traded prices.",
            "Diagnostic and forward windows never influence selection.",
        ],
    }


def _record(*, spec, card_dir, recorder_uri, summary, selected_policy) -> dict[str, Any]:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R

    recorder_uri = Path(recorder_uri).resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / spec.provider_uri).resolve()), region=spec.region)
    with R.start(experiment_name=spec.experiment, recorder_name="frontier", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(
            schema_version=SCHEMA_VERSION,
            provider_uri=summary["provider_uri"],
            as_of=spec.as_of,
            costs_bps=str(list(spec.costs_bps)),
            selected_policy=selected_policy.name,
            candidate_count=summary["candidate_count"],
        )
        recorder.log_metrics(
            forward_return_pct=summary["selected_forward"]["strategy"]["return_pct"],
            forward_max_drawdown_pct=summary["selected_forward"]["strategy"]["max_drawdown_pct"],
            forward_sharpe=summary["selected_forward"]["strategy"]["sharpe"],
            outer_return_pct=summary["selected_outer_10bps"]["strategy"]["return_pct"],
            outer_max_drawdown_pct=summary["selected_outer_10bps"]["strategy"]["max_drawdown_pct"],
            outer_sharpe=summary["selected_outer_10bps"]["strategy"]["sharpe"],
        )
        recorder.save_objects(**{
            "summary.json": summary,
            "selected_policy.json": selected_policy.__dict__,
        })
        recorder_id = recorder.id
        experiment_id = str(recorder.experiment_id)
        local_dir = recorder_uri / experiment_id / recorder_id
    return {
        "recorder_id": recorder_id,
        "experiment_id": experiment_id,
        "recorder_uri": str(recorder_uri),
        "recorder_local_dir": str(local_dir),
        "qlib_docs_reference": QLIB_DOCS,
    }


def write_card_files(summary: dict, *, card_dir: Path) -> None:
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    comparison = pd.DataFrame(summary["window_comparison"])
    comparison.to_csv(card_dir / "policy_comparison.csv", index=False)
    annual = pd.DataFrame(summary.get("annual_metrics", []))
    if not annual.empty:
        annual.to_csv(card_dir / "annual_metrics.csv", index=False)
    benchmark_rows = comparison[comparison["window"].eq("reused_diagnostic")]
    benchmark_rows.to_csv(card_dir / "backtest_benchmark_comparison.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION,
        "experiment": summary["experiment"],
        "provider_uri": summary["provider_uri"],
        "region": summary["region"],
        "market": summary["market"],
        "as_of": summary["as_of"],
        "recorder_uri": summary["recorder_uri"],
        "segments": summary["segments"],
        "costs_bps": summary["costs_bps"],
        "selection": "development_only_stable_excess_rank",
        "artifact_policy": summary["artifact_policy"],
        "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    conclusion = _conclusion(summary)
    (card_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")


def _conclusion(summary: dict) -> str:
    forward = summary["selected_forward"]["strategy"]
    outer = summary["selected_outer_10bps"]["strategy"]
    bench_outer = summary["selected_outer_10bps"]["benchmark"]
    bench_forward = summary["selected_forward"]["benchmark"]
    return (
        f"# SOXL 稳定超额前沿研究（中文结论）\n\n"
        f"- 做了什么：在 qlib 原生链路上执行 {summary['candidate_count']} 条预登记因果暴露规则"
        f"（含 SOXX 基准对照）；无收益预测、无拟合 alpha、零模型拟合；选择仅使用 development "
        f"{summary['segments']['development']}。\n"
        f"- 核心指标（选中 {summary['selected_policy']}）：复用区间 {summary['segments']['reused_diagnostic']} "
        f"收益 {outer['return_pct']:.2f}%（SOXX {bench_outer['return_pct']:.2f}%），"
        f"最大回撤 {outer['max_drawdown_pct']:.2f}%（SOXX {bench_outer['max_drawdown_pct']:.2f}%），"
        f"Sharpe {outer['sharpe']:.2f}（SOXX {bench_outer['sharpe']:.2f}）。\n"
        f"- 新前向区间 {summary['segments']['forward']}：收益 {forward['return_pct']:.2f}%"
        f"（SOXX {bench_forward['return_pct']:.2f}%），最大回撤 {forward['max_drawdown_pct']:.2f}%"
        f"（SOXX {bench_forward['max_drawdown_pct']:.2f}%），Sharpe {forward['sharpe']:.2f}。\n"
        f"- regime/窗口差异：选中策略平均 SOXL 权重 {summary['selected_forward'].get('average_soxl_weight', 0):.3f}、"
        f"平均现金权重 {summary['selected_forward'].get('average_cash_weight', 0):.3f}；"
        f"滚动年正超额 {summary['selected_outer_verdict']['positive_year_count']}/"
        f"{summary['selected_outer_verdict']['year_count']}。\n"
        f"- 是否晋升：{summary['promotion_eligible']}；决定 `{summary['decision']}`。"
        f"前向同时超过 SOXX 收益与回撤：{summary['selected_forward_verdict']['beats_return_and_drawdown']}。\n"
        f"- 风险：前向样本短；无收益预测；SOXL 杠杆衰减仅由真实价格体现；诊断与前向区间不参与选择；"
        f"本结果不构成生产或实盘证明。\n"
        f"- 下一步：无收益预测的日频风险/趋势规则不能产生稳定超额。后续应转向已被证明的日内机制"
        f"（早期会话状态 + 现金时点控制），并冻结结构等待新的连续前向证据。\n"
        f"- qlib Recorder：{summary.get('recorder_id')}；provider_uri={summary['provider_uri']}；"
        f"as_of={summary['as_of']}；qlib docs：{QLIB_DOCS}。\n"
    )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
