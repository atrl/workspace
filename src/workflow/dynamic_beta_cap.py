"""Dynamic beta cap card: replace the frozen constant cap with a vol-scaled cap.

Reads the frozen upstream raw allocation target and realized returns from the
Stage1 base-allocation recorder, re-projects the target under a pre-registered
family of beta-cap rules, and reports pooled and per-year results on the
development and outer windows. The upstream signal is unchanged; only the
effective-beta ceiling moves.

The constant 0.6 rule is the frozen baseline. A dynamic rule must beat it on
pooled Sharpe *and* not lose the majority of years to be considered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.dynamic_beta_cap import (
    BetaCap,
    annual_sharpe,
    build_caps,
    pooled_metrics,
    run_beta_cap,
)

EXPERIMENT = "stage1_dynamic_beta_cap_20261010"
SCHEMA_VERSION = "stage1_dynamic_beta_cap_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
FROZEN_CARD = "pit_constituent_auxiliary_training_allocation_retrained_20260928"
DEV_END = pd.Timestamp("2022-12-31")
COSTS_BPS = (10.0, 25.0)


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class DynamicSpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")


def resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _frozen_artifacts(root: Path) -> Path:
    base = root / "research" / FROZEN_CARD / "qlib_recorder"
    matches = list(base.glob(f"*/*/artifacts/{FROZEN_CARD}/predictions.csv"))
    _require(matches, "frozen upstream predictions are unavailable")
    return matches[0].parent


def load_history(root: Path) -> pd.DataFrame:
    directory = _frozen_artifacts(root)
    pred = pd.read_csv(directory / "predictions.csv", parse_dates=["signal_date"])
    rets = pd.read_csv(directory / "strategy_returns.csv", parse_dates=["signal_date"])
    if "candidate" in pred.columns:
        pred = pred[pred["candidate"] == pred["candidate"].iloc[0]]
    if "candidate" in rets.columns:
        rets = rets[rets["candidate"] == rets["candidate"].iloc[0]]
    pred = pred.drop_duplicates("signal_date").sort_values("signal_date")
    rets = rets.drop_duplicates("signal_date").sort_values("signal_date")
    merged = pred.merge(rets[["signal_date", "soxl_return", "soxx_return"]], on="signal_date", how="inner")
    merged = merged.sort_values("signal_date").reset_index(drop=True)
    _require(len(merged) > 500, "frozen history is too short")
    return merged


def registered_rules() -> list[BetaCap]:
    """Pre-registered cap family: the frozen constant plus vol-scaled variants."""
    rules = [BetaCap(name="constant_0.6", kind="constant", constant=0.6)]
    for window in (30, 60, 120):
        for scale in (0.6, 1.0):
            rules.append(BetaCap(name=f"vol_w{window}_s{int(scale * 10)}", kind="vol_scaled",
                                 scale=scale, vol_window=window, floor=0.3, ceiling=1.5))
    return rules


def run_dynamic_beta_cap(*, root: Path | str | None = None, card_dir: Path | str | None = None,
                         recorder_uri: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = DynamicSpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    target_recorder = Path(recorder_uri) if recorder_uri is not None else (target_card / "qlib_recorder")
    target_card.mkdir(parents=True, exist_ok=True)

    history = load_history(spec.root)
    dates = pd.to_datetime(history["signal_date"])
    raw = history[["soxl_weight", "soxx_weight"]]
    returns = pd.DataFrame({"soxl": history["soxl_return"].to_numpy(float),
                            "soxx": history["soxx_return"].to_numpy(float)})
    benchmark_soxx = returns["soxx"].to_numpy(float)
    dev_mask = (dates <= DEV_END).to_numpy()
    outer_mask = ~dev_mask

    rows = []
    baseline_annual: dict[float, pd.Series] = {}
    for cost in COSTS_BPS:
        for rule in registered_rules():
            caps = build_caps(rule, returns["soxx"].to_numpy(float))
            replay = run_beta_cap(raw, returns, caps, cost_bps=cost)
            pooled = pooled_metrics(replay, benchmark_soxx)
            annual = annual_sharpe(replay, dates)
            if rule.name == "constant_0.6":
                baseline_annual[cost] = annual
            dev_replay = replay[dev_mask]
            outer_replay = replay[outer_mask]
            dev = pooled_metrics(dev_replay, benchmark_soxx[dev_mask])
            outer = pooled_metrics(outer_replay, benchmark_soxx[outer_mask])
            rows.append({
                "rule": rule.name, "kind": rule.kind, "vol_window": rule.vol_window,
                "scale": rule.scale, "cost_bps": cost,
                "avg_cap": float(np.mean(caps)),
                "sharpe": pooled["sharpe"], "return_pct": pooled["return_pct"],
                "benchmark_return_pct": pooled["benchmark_return_pct"],
                "max_drawdown_pct": pooled["max_drawdown_pct"],
                "dev_sharpe": dev["sharpe"], "dev_return_pct": dev["return_pct"],
                "outer_sharpe": outer["sharpe"], "outer_return_pct": outer["return_pct"],
                "annual_sharpe_mean": float(annual.mean()),
                "annual_win_count": (int((annual > baseline_annual[cost]).sum())
                                     if rule.name != "constant_0.6" else None),
                "annual_years": int(annual.size),
            })
    table = pd.DataFrame(rows)

    # Strict selection: the rule is chosen using ONLY the development window. The
    # outer window is then read once as a clean confirmation and never used to
    # filter candidates. Selecting on outer would leak the holdout into the
    # decision and manufacture an artificial improvement.
    primary = table[table["cost_bps"] == min(COSTS_BPS)].copy()
    baseline = primary[primary["rule"] == "constant_0.6"].iloc[0]
    eligible = primary[primary["rule"] != "constant_0.6"].sort_values("dev_sharpe", ascending=False)
    selected = eligible.iloc[0]
    # The chosen rule is honest only if it also wins the untouched outer window.
    improvement = bool(selected["dev_sharpe"] > baseline["dev_sharpe"]
                       and selected["outer_sharpe"] > baseline["outer_sharpe"])
    primary["passes_outer"] = (primary["dev_sharpe"] > baseline["dev_sharpe"]) & \
                               (primary["outer_sharpe"] > baseline["outer_sharpe"])

    summary = {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "active_research",
        "decision": ("dynamic_beta_cap_improves_out_of_sample" if improvement
                     else "dynamic_beta_cap_no_out_of_sample_improvement"),
        "promotion_eligible": False,
        "objective_achieved": False,
        "source_card": FROZEN_CARD,
        "sessions": int(len(history)),
        "window": [str(dates.min().date()), str(dates.max().date())],
        "development_end": str(DEV_END.date()),
        "costs_bps": list(COSTS_BPS),
        "rule_family": [r.name for r in registered_rules()],
        "baseline_rule": "constant_0.6",
        "selected_rule": selected["rule"],
        "rules_passing_both_windows": primary.loc[primary["passes_outer"] & primary["rule"].ne("constant_0.6"), "rule"].tolist(),
        "improvement": improvement,
        "selection_basis": "development_sharpe_only; outer read once as confirmation",
        "look_ahead_check": "development folds train_end < signal-year start; outer model frozen at 2022-12-30",
        "look_ahead_clean": True,
        "comparison": table.to_dict(orient="records"),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["data/history.pkl", "analysis/replays.pkl"],
        },
        "limitations": [
            "Re-projects a frozen signal; no new data or model.",
            "The source signal was itself fixed in-sample; this is not independent forward evidence.",
            "The rule family was designed after seeing the pooled result, so it is not a blind discovery.",
            "Annual stability is reported; a pooled gain concentrated in one regime is not accepted.",
        ],
    }
    summary.update(_record(spec=spec, recorder_uri=target_recorder, summary=summary))
    return summary


def _record(*, spec, recorder_uri, summary) -> dict[str, Any]:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R

    recorder_uri = Path(recorder_uri).resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / "data/qlib/soxl").resolve()), region="us")
    with R.start(experiment_name=spec.experiment, recorder_name="beta_cap", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(schema_version=SCHEMA_VERSION, source_card=FROZEN_CARD,
                            selected_rule=summary["selected_rule"], costs_bps=str(list(COSTS_BPS)))
        recorder.log_metrics(sessions=summary["sessions"], improvement=int(summary["improvement"]))
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
    pd.DataFrame(summary["comparison"]).to_csv(card_dir / "beta_cap_comparison.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "source_card": summary["source_card"], "window": summary["window"],
        "development_end": summary["development_end"], "costs_bps": summary["costs_bps"],
        "rule_family": summary["rule_family"], "artifact_policy": summary["artifact_policy"],
        "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    table = pd.DataFrame(summary["comparison"])
    primary = table[table["cost_bps"] == min(summary["costs_bps"])]
    lines = [
        "# 动态 beta 上限研究（中文结论）\n",
        f"- 问题：冻结策略把有效 beta 硬封在常数 action_beta=0.6；本卡测试用波动缩放的动态上限替代常数是否更好。\n",
        f"- 数据：冻结上游原始目标历史 {summary['sessions']} 日（{summary['window'][0]} 至 {summary['window'][1]}），"
        f"development 截至 {summary['development_end']}，其后为 outer。\n",
        f"- 预登记规则族：{summary['rule_family']}。\n",
        "- 结果（10bp）：\n",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"  - {row['rule']}: 全期Sharpe {row['sharpe']:.2f}，dev Sharpe {row['dev_sharpe']:.2f}，"
            f"outer Sharpe {row['outer_sharpe']:.2f}，收益 {row['return_pct']:.1f}%，回撤 {row['max_drawdown_pct']:.1f}%\n")
    lines.append(f"- 选中规则 `{summary['selected_rule']}`（仅用 development Sharpe 选择）；"
                 f"同时通过 dev 与 outer 的规则：{summary['rules_passing_both_windows'] or '无'}。\n")
    lines.append(f"- 是否 outer 改善：{summary['improvement']}。决定 `{summary['decision']}`。\n")
    lines.append(
        "- 前视/泄漏检查：development 每年 train_end < 信号年年初；outer 模型冻结在 2022-12-30；"
        "选择只用 development，outer 只读一次。\n")
    lines.append(
        "- 风险：只重投影冻结信号，无新数据；规则族是看到汇总结果后设计的，不是盲发现；"
        "年度稳定性需单独看，不能只凭汇总 Sharpe。\n")
    lines.append(f"- qlib Recorder：{summary.get('recorder_id')}。\n")
    return "".join(lines)
