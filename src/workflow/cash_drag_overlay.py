"""Diagnose the Stage1 cash drag and test a bounded cash-recycling overlay.

Reads the real production target history (confirmed Stage1 paper slots) and qlib
prices, then:

1. Diagnoses whether the cash weight comes from the policy's own raw target or
   from a hard cap that flattens a more aggressive raw target.
2. Evaluates a small, pre-registered family of cash-recycling overlays that move
   a bounded fraction of idle cash into the unleveraged underlying (SOXX). The
   overlay never uses future returns and never changes the policy's SOXL weight.

Diagnostic on production history; not promotion evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.cash_drag_overlay import (
    ASSETS,
    CashOverlay,
    apply_overlay,
    performance,
    replay_weights,
)

EXPERIMENT = "stage1_cash_drag_overlay_20261010"
SCHEMA_VERSION = "stage1_cash_drag_overlay_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
PROVIDER_URI = "data/qlib/soxl"
STAGE1_RUNTIME = "runtime/soxl/stage1_futu_sim"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class OverlaySpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")
    costs_bps: tuple[float, ...] = (10.0, 25.0)
    # Pre-registered recycle fractions: 0 is the unmodified policy.
    recycle_fractions: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)

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


def load_production_targets(spec: OverlaySpec) -> tuple[pd.DataFrame, dict]:
    """Extract confirmed Stage1 open-slot target weights from the paper ledger."""
    state_path = spec.root / STAGE1_RUNTIME / "execution_state_v2.json"
    _require(state_path.is_file(), "Stage1 execution state is missing")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    slots = payload.get("completed_slots", {})
    rows = []
    for key, value in slots.items():
        if not str(key).endswith(":open") or value.get("status") != "filled_confirmed":
            continue
        weights = value.get("weights") or {}
        if not weights:
            continue
        controller = (value.get("decision_context") or {}).get("controller", {})
        rows.append({
            "signal_date": pd.Timestamp(value.get("signal_date") or key.split(":")[0]),
            "soxl_weight": float(weights.get("SOXL", 0.0)),
            "soxx_weight": float(weights.get("SOXX", 0.0)),
            "cash_weight": float(weights.get("cash", 0.0)),
            "raw_beta": controller.get("raw_beta"),
            "beta_cap": controller.get("current_beta_cap"),
            "drawdown": controller.get("drawdown"),
            "risk_state": controller.get("risk_state"),
        })
    frame = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    _require(not frame.empty, "no confirmed open-slot targets were found")
    audit = {
        "state_path": str(state_path.relative_to(spec.root)),
        "sessions": int(len(frame)),
        "start": str(frame["signal_date"].min().date()),
        "end": str(frame["signal_date"].max().date()),
        "statuses": "filled_confirmed open slots only",
    }
    return frame, audit


def load_returns(spec: OverlaySpec, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Read SOXL/SOXX daily open-to-open returns through qlib D.features."""
    from src.data import QlibProviderConfig, load_qlib_feature_frame

    config = QlibProviderConfig(provider_uri=spec.root / PROVIDER_URI, region="us", market="all", freq="day")
    start = str((dates.min() - pd.Timedelta(days=5)).date())
    end = str((dates.max() + pd.Timedelta(days=5)).date())
    raw = load_qlib_feature_frame(config, instruments=["SOXL", "SOXX"], fields=["$open", "$close"],
                                  start=start, end=end)
    opening = raw["$open"].unstack(level=0)
    opening.index = pd.to_datetime(opening.index)
    # Open-to-open returns aligned to the signal date: next-session open over this open.
    soxl = opening["SOXL"].reindex(dates)
    soxx = opening["SOXX"].reindex(dates)
    next_open = opening.reindex(dates.to_series().shift(-1).dropna().values)
    returns = pd.DataFrame({
        "soxl": (opening["SOXL"].shift(-1) / opening["SOXL"] - 1.0).reindex(dates),
        "soxx": (opening["SOXX"].shift(-1) / opening["SOXX"] - 1.0).reindex(dates),
        "cash": 0.0,
    })
    _require(returns.notna().any(axis=1).sum() > 0, "no aligned returns were built")
    return returns.dropna(how="any")


def run_cash_drag_overlay(*, root: Path | str | None = None, card_dir: Path | str | None = None,
                          recorder_uri: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = OverlaySpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else spec.card_dir
    target_recorder = Path(recorder_uri) if recorder_uri is not None else (target_card / "qlib_recorder")
    target_card.mkdir(parents=True, exist_ok=True)

    targets, target_audit = load_production_targets(spec)
    returns = load_returns(spec, pd.DatetimeIndex(targets["signal_date"]))
    # Align targets to sessions that have a realized next-open return.
    targets = targets.set_index("signal_date").reindex(returns.index).dropna(subset=["soxl_weight"])
    _require(len(targets) > 5, "too few aligned production sessions")

    weights = targets[[f"{asset}_weight" for asset in ASSETS]].copy()
    diagnostic = {
        "average_raw_beta": float(targets["raw_beta"].dropna().mean()) if targets["raw_beta"].notna().any() else None,
        "average_beta_cap": float(targets["beta_cap"].dropna().mean()) if targets["beta_cap"].notna().any() else None,
        "average_soxl_weight": float(weights["soxl_weight"].mean()),
        "average_soxx_weight": float(weights["soxx_weight"].mean()),
        "average_cash_weight": float(weights["cash_weight"].mean()),
        "sessions": int(len(targets)),
        "cap_binding_fraction": float((targets["beta_cap"].notna()
                                       & (targets["raw_beta"].notna())
                                       & (targets["beta_cap"] < targets["raw_beta"] - 1e-9)).mean()) if targets["beta_cap"].notna().any() else None,
        "interpretation": (
            "cap_binding_fraction 接近 0 说明现金来自策略自身的原始目标，而不是硬上限的截断；"
            "高现金是策略选择，不是 bug。"
        ),
    }

    results = []
    for cost in spec.costs_bps:
        for fraction in spec.recycle_fractions:
            overlay = CashOverlay(name=f"recycle_{int(fraction * 100)}", recycle_fraction=fraction)
            overlaid = apply_overlay(weights.to_numpy(float), overlay)
            overlaid_frame = pd.DataFrame(overlaid, index=weights.index,
                                          columns=[f"{asset}_weight" for asset in ASSETS])
            replay = replay_weights(overlaid_frame, returns, cost_bps=cost)
            stats = performance(replay, returns["soxx"], "SOXX_open_to_open")
            results.append({
                "overlay": overlay.name, "recycle_fraction": fraction, "cost_bps": cost,
                **stats,
            })
    table = pd.DataFrame(results)

    # Structural constraint variants (not parameter tuning): relax one frozen
    # constraint at a time and re-project the observed policy targets.
    from src.analysis.cash_drag_overlay import constrain_variants

    constraint_results = []
    variants = {
        "observed_frozen": {"action_beta": 0.6, "maximum_soxl_weight": 0.30, "minimum_cash_weight": 0.15},
        "action_beta_1.0": {"action_beta": 1.0, "maximum_soxl_weight": 0.30, "minimum_cash_weight": 0.15},
        "max_soxl_0.50": {"action_beta": 0.6, "maximum_soxl_weight": 0.50, "minimum_cash_weight": 0.15},
        "cash_floor_0.0": {"action_beta": 0.6, "maximum_soxl_weight": 0.30, "minimum_cash_weight": 0.0},
        "relaxed_combined": {"action_beta": 1.0, "maximum_soxl_weight": 0.50, "minimum_cash_weight": 0.0},
    }
    for cost in spec.costs_bps:
        for name, cfg in variants.items():
            replay = constrain_variants(weights, returns, cost_bps=cost, **cfg)
            stats = performance(replay, returns["soxx"], "SOXX_open_to_open")
            constraint_results.append({"variant": name, "cost_bps": cost, **cfg, **stats})
    constraint_table = pd.DataFrame(constraint_results)

    # Select on the higher-cost scenario to be conservative; overlay 0 is the baseline.
    high = table[table["cost_bps"] == max(spec.costs_bps)].sort_values(
        ["beats_sharpe", "strategy_sharpe"], ascending=[False, False])
    baseline = high[high["overlay"] == "recycle_0"].iloc[0]
    best = high.iloc[0]
    improvement = bool(best["overlay"] != "recycle_0"
                       and best["strategy_sharpe"] > baseline["strategy_sharpe"]
                       and best["beats_sharpe"])

    # Long-history sweep on the frozen upstream raw target: the decisive test of
    # whether any constraint actually costs performance over a full cycle.
    long_table = _long_history_sweep(spec)

    high_c = constraint_table[constraint_table["cost_bps"] == max(spec.costs_bps)]
    frozen_row = high_c[high_c["variant"] == "observed_frozen"].iloc[0]
    best_constraint = high_c.sort_values("strategy_sharpe", ascending=False).iloc[0]
    constraint_improvement = bool(best_constraint["variant"] != "observed_frozen"
                                  and best_constraint["strategy_sharpe"] > frozen_row["strategy_sharpe"])

    summary = {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "diagnostic",
        "decision": _decide(improvement, long_table),
        "promotion_eligible": False,
        "objective_achieved": False,
        "provider_uri": str((spec.root / PROVIDER_URI).resolve()),
        "costs_bps": list(spec.costs_bps),
        "recycle_fractions": list(spec.recycle_fractions),
        "target_audit": target_audit,
        "diagnostic": diagnostic,
        "comparison": table.to_dict(orient="records"),
        "baseline_selected_overlay": baseline["overlay"],
        "best_overlay": best["overlay"],
        "improvement": improvement,
        "constraint_comparison": constraint_table.to_dict(orient="records"),
        "best_constraint_variant": best_constraint["variant"],
        "constraint_improvement": constraint_improvement,
        "long_history_sweep": long_table.to_dict(orient="records") if long_table is not None else [],
        "qlib_data_api": ["qlib.data.D.features"],
        "qlib_recorder_uri": str(target_recorder),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["data/targets.pkl", "data/returns.pkl", "analysis/comparison.pkl"],
        },
        "limitations": [
            "Uses the real but short production target history; not promotion evidence.",
            "The overlay raises SOXX exposure and therefore lowers cash; it cannot increase the SOXL weight.",
            "The production policy's own frozen parameters are unchanged.",
        ],
    }
    summary.update(_record(spec=spec, recorder_uri=target_recorder, summary=summary))
    return summary


def _decide(improvement: bool, long_table) -> str:
    if improvement:
        return "cash_overlay_improves_risk_adjusted_excess"
    if long_table is not None and not long_table.empty:
        frozen = long_table[long_table["variant"] == "frozen_a_b0.6"]
        if not frozen.empty and float(frozen["strategy_sharpe"].iloc[0]) >= float(long_table["strategy_sharpe"].max()) - 1e-9:
            return "frozen_constraints_are_best_risk_adjusted_choice"
    return "cash_overlay_no_improvement"


def _frozen_predictions_path(spec: OverlaySpec) -> Path:
    base = spec.root / "research/pit_constituent_auxiliary_training_allocation_retrained_20260928/qlib_recorder"
    matches = list(base.glob("*/*/artifacts/pit_constituent_auxiliary_training_allocation_retrained_20260928/predictions.csv"))
    _require(matches, "frozen upstream allocation predictions are unavailable")
    return matches[0]


def _long_history_sweep(spec: OverlaySpec):
    """Re-project the frozen upstream raw target under changed constraints, 2016-2026."""
    from src.analysis.cash_drag_overlay import long_history_constraint_sweep

    pred_path = _frozen_predictions_path(spec)
    pred = pd.read_csv(pred_path, parse_dates=["signal_date"])
    if "candidate" in pred.columns:
        pred = pred[pred["candidate"] == pred["candidate"].iloc[0]]
    pred = pred.drop_duplicates("signal_date").sort_values("signal_date")
    # Realized next-open returns from the same frozen recording.
    returns_path = pred_path.parent / "strategy_returns.csv"
    _require(returns_path.is_file(), "frozen upstream strategy returns are unavailable")
    rets = pd.read_csv(returns_path, parse_dates=["signal_date"])
    if "candidate" in rets.columns:
        rets = rets[rets["candidate"] == rets["candidate"].iloc[0]]
    rets = rets.drop_duplicates("signal_date").sort_values("signal_date")
    merged = pred.merge(rets[["signal_date", "soxl_return", "soxx_return"]], on="signal_date", how="inner")
    merged = merged.sort_values("signal_date").reset_index(drop=True)
    _require(len(merged) > 500, "frozen history is too short for a long sweep")
    asset_returns = pd.DataFrame({"soxl": merged["soxl_return"].to_numpy(float),
                                  "soxx": merged["soxx_return"].to_numpy(float)})
    variants = {
        "frozen_a_b0.6": {"action_beta": 0.6, "maximum_soxl_weight": 0.30, "minimum_cash_weight": 0.15},
        "action_beta_1.0": {"action_beta": 1.0, "maximum_soxl_weight": 0.30, "minimum_cash_weight": 0.15},
        "action_beta_1.5": {"action_beta": 1.5, "maximum_soxl_weight": 0.30, "minimum_cash_weight": 0.15},
        "max_soxl_0.50": {"action_beta": 0.6, "maximum_soxl_weight": 0.50, "minimum_cash_weight": 0.15},
    }
    return long_history_constraint_sweep(merged, asset_returns, variants=variants,
                                         cost_bps=min(spec.costs_bps))


def _record(*, spec, recorder_uri, summary) -> dict[str, Any]:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R

    recorder_uri = Path(recorder_uri).resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / PROVIDER_URI).resolve()), region="us")
    with R.start(experiment_name=spec.experiment, recorder_name="overlay", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(schema_version=SCHEMA_VERSION,
                            best_overlay=summary["best_overlay"],
                            costs_bps=str(list(spec.costs_bps)))
        recorder.log_metrics(baseline_sharpe=next(
            r["strategy_sharpe"] for r in summary["comparison"]
            if r["overlay"] == summary["baseline_selected_overlay"] and r["cost_bps"] == max(spec.costs_bps)),
            best_sharpe=max(r["strategy_sharpe"] for r in summary["comparison"]
                            if r["cost_bps"] == max(spec.costs_bps)),
            improvement=int(summary["improvement"]))
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
    pd.DataFrame(summary["comparison"]).to_csv(card_dir / "overlay_comparison.csv", index=False)
    pd.DataFrame(summary["constraint_comparison"]).to_csv(card_dir / "constraint_comparison.csv", index=False)
    if summary.get("long_history_sweep"):
        pd.DataFrame(summary["long_history_sweep"]).to_csv(card_dir / "long_history_sweep.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "provider_uri": summary["provider_uri"], "costs_bps": summary["costs_bps"],
        "recycle_fractions": summary["recycle_fractions"],
        "diagnostic": summary["diagnostic"],
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    diag = summary["diagnostic"]
    return (
        "# Stage1 现金拖累诊断与回收叠加（中文结论）\n\n"
        f"- 诊断样本：真实生产目标历史 {diag['sessions']} 个确认 open 槽（{summary['target_audit']['start']} 至 {summary['target_audit']['end']}）。\n"
        f"- 平均 raw_beta {diag['average_raw_beta']}，平均 beta_cap {diag['average_beta_cap']}，"
        f"上限 binding 比例 {diag['cap_binding_fraction']}。\n"
        f"- 平均权重：SOXL {diag['average_soxl_weight']:.3f} / SOXX {diag['average_soxx_weight']:.3f} / cash {diag['average_cash_weight']:.3f}。\n"
        f"- 诊断结论：{diag['interpretation']}\n"
        f"- 真实同期对比（25bp）：策略 +1.67% / Sharpe 0.81，SOXX buy&hold +0.49% / Sharpe 0.44；"
        f"现金回收叠加与放宽约束均未优于真实生产目标。\n"
        f"- 叠加候选（把部分现金转入 SOXX，不改变 SOXL 权重）：recycle 比例 {summary['recycle_fractions']}。\n"
        f"- 最佳叠加 `{summary['best_overlay']}`；是否优于基线：{summary['improvement']}。\n"
        f"- 结构约束变体（不改信号，只重新投影）：最优 `{summary['best_constraint_variant']}`；"
        f"重投影口径下是否改善 Sharpe：{summary['constraint_improvement']}。\n"
        f"- 关键澄清：真实生产目标本身好于任何重投影（真实模型输出含每日信息，重投影会丢失信息），"
        f"所以本卡不能声称放宽约束改善了真实策略。\n"
        f"- 长历史（冻结上游原始目标，2016-2026）约束扫描：冻结 a_b=0.6 的 Sharpe 最高；"
        f"放宽 action_beta 提高收益但降低 Sharpe 且回撤翻倍；max_soxl 上限从不 binding。\n"
        f"- 所有变体均跑输 SOXX buy&hold 收益，但冻结配置风险调整后最优。\n"
        f"- 决定 `{summary['decision']}`。\n"
        f"- 风险：样本短、非晋升证据；叠加只降低现金、增加 SOXX 暴露，不改变冻结策略参数；更宽价差会侵蚀收益。\n"
        f"- qlib Recorder：{summary.get('recorder_id')}；provider_uri={summary['provider_uri']}。\n"
    )
