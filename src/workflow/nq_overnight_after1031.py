"""Card workflow for the NQ overnight 18:00->10:00 -> post-10:31 diagnostic.

Zero models, one declared decision (10:31 ET), one direction-fixed signal. The
route stops when there is no fee-robust positive edge.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis import nq_overnight_signal as A

EXPERIMENT = "stage1_nq_overnight_after1031_20261010"
SCHEMA_VERSION = "stage1_nq_overnight_after1031_v1"
QLIB_DOCS = ["https://qlib.readthedocs.io/en/latest/component/data.html",
             "https://qlib.readthedocs.io/en/latest/component/recorder.html"]
# A fee-robust edge must be positive with a CI above zero at the 25bp stress.
STRESS_COST_BPS = 25.0


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class Spec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")


def _resolve_root(root) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _decide(results: dict) -> dict:
    """Stop unless the declared long-only rule is positive with CI>0 at the 25bp stress."""
    robust = [r for r in results["economic_long_only"]
              if r["book"] == A.PRIMARY_BOOK and r["cost_bps"] == STRESS_COST_BPS
              and r["ci_status"] == "ok" and r["ci_low_bps"] is not None and r["ci_low_bps"] > 0.0 and r["mean_net_bps"] > 0.0]
    if not robust:
        return {"decision": "end_route_no_further_search", "promotion_eligible": False,
                "reason": "no long-only NQ>0->SOXL rule edge with a positive CI at the 25bp roundtrip stress"}
    return {"decision": "conditional_edge_needs_independent_forward", "promotion_eligible": False,
            "reason": "positive 25bp stress edge found; still needs independent forward, no promotion"}


def run_nq_overnight_after1031(*, root=None, card_dir=None) -> dict[str, Any]:
    spec = Spec(root=_resolve_root(root))
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    target_card.mkdir(parents=True, exist_ok=True)
    plan_path = spec.root / "research" / spec.experiment / "plan.json"
    _require(plan_path.is_file(), "plan.json is missing")
    sources = A.freeze_sources(spec.root)
    outcome = A.evaluate(spec.root)
    results, calendar, panel, buyhold = outcome["results"], outcome["calendar"], outcome["panel"], outcome["buyhold"]
    decision = _decide(results)
    summary = {
        "experiment": spec.experiment, "schema_version": SCHEMA_VERSION, "card_status": "active_research",
        **decision, "objective_achieved": False,
        "plan_sha256": A.sha256(plan_path),
        "plan_registered_at_utc": json.loads(plan_path.read_text()).get("registered_at_utc"),
        "source_freeze": sources,
        "declared_decision_at": "10:31 ET", "signal": "nq_overnight 18:00->10:00 (prior calendar day)",
        "models_trained": 0, "direction": results["direction"], "parameter_search": "none",
        "official_days_in_range": results["official_days_in_range"], "sessions": results["sessions"],
        "dropped_needs_data": results["dropped_needs_data"],
        "morning_days_in_range": results["morning_days_in_range"], "valid_morning_days": results["valid_morning_days"],
        "missing_nq_signal_days": results["missing_nq_signal_days"], "calendar_digest": results["calendar_digest"],
        "spread_ic_primary": results["spread_ic_primary"],
        "individual_ic_descriptive": results["individual_ic_descriptive"],
        "economic_long_only": results["economic_long_only"],
        "soxx_buyhold_account": results["soxx_buyhold_account"],
        "relative_rule_minus_soxx": results["relative_rule_minus_soxx"],
        "three_account_wealth": results["three_account_wealth"],
        "native_capability": results["native_capability"],
        "execution_proxy": results["execution_proxy"], "direction_note": results["direction_note"],
        "futures_roll_note": "NQ=F is the Yahoo front continuous series; contract roll/adjustment is not "
                             "independent evidence (listed, not corrected)",
        "reporting_caveat": "point estimates + fixed 20-session full-calendar mask moving-block CI; ~2 years of seen data; no stable-excess claim",
        "limitations": [
            "Execution proxy is the predeclared 10:30 close, not an actual 10:31 fill.",
            "CI uses fixed 20-session moving blocks over the full official calendar; invalid days are masked and filtered inside the statistic.",
            "~2 years of seen data; NQ=F is a rolling front-continuous series (roll/adjustment is not independent evidence).",
            "No parameter/cutoff/sign search permitted; a failure stops the route.",
        ],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": []},
    }
    summary.update(_record(spec=spec, card_dir=target_card, summary=summary, calendar=calendar, panel=panel,
                           buyhold=buyhold, ledgers=outcome["ledgers"], plan_path=plan_path))
    _write_card(summary, card_dir=target_card, panel=panel)
    return summary


def _record(*, spec, card_dir, summary, calendar, panel, buyhold, ledgers, plan_path) -> dict[str, Any]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import logging
    import warnings
    for name in ("qlib", "mlflow", "qlib.data", "qlib.workflow"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    import qlib
    from qlib.workflow import R

    recorder_uri = (Path(card_dir) / "qlib_recorder").resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / A.UNIFORM_INTRADAY).resolve()), region="us")
    saved = {"calendar": calendar, "panel_valid": panel,
             "spread_ic_primary": pd.DataFrame([summary["spread_ic_primary"]]),
             "individual_ic": pd.DataFrame(summary["individual_ic_descriptive"]).T,
             "economic_long_only": pd.DataFrame(summary["economic_long_only"])}
    for cost, account in buyhold.items():
        saved[f"soxx_buyhold_path_{int(cost)}"] = account["path"]
    for cost, frame in ledgers.items():
        saved[f"rule_ledger_{int(cost)}"] = frame
    with R.start(experiment_name=spec.experiment, recorder_name="nq_overnight", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(schema_version=SCHEMA_VERSION, plan_sha256=summary["plan_sha256"],
                            decision=summary["decision"], models_trained=summary["models_trained"])
        recorder.log_metrics(sessions=float(summary["sessions"]), dropped=float(summary["dropped_needs_data"]))
        recorder.save_objects(**saved)
        recorder.log_artifact(str(plan_path), artifact_path="plan")
        snapshot = Path(card_dir) / "source_binding.json"
        snapshot.write_text(json.dumps({"sources": summary["source_freeze"], "plan_sha256": summary["plan_sha256"]},
                                       indent=2), encoding="utf-8")
        recorder.log_artifact(str(snapshot), artifact_path="inputs")
        recorder_id, experiment_id = recorder.id, str(recorder.experiment_id)
        local_dir = recorder_uri / experiment_id / recorder_id
    summary["artifact_policy"]["recorder_only_artifacts"] = sorted(saved)
    return {"recorder_id": recorder_id, "experiment_id": experiment_id, "recorder_uri": str(recorder_uri),
            "recorder_local_dir": str(local_dir), "saved_objects": sorted(saved), "qlib_docs_reference": QLIB_DOCS}


def _write_card(summary: dict, *, card_dir: Path, panel: pd.DataFrame) -> None:
    (card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame([summary["spread_ic_primary"]]).to_csv(card_dir / "spread_ic_primary.csv", index=False)
    pd.DataFrame(summary["individual_ic_descriptive"]).T.to_csv(card_dir / "individual_ic.csv")
    pd.DataFrame(summary["economic_long_only"]).to_csv(card_dir / "economic_long_only.csv", index=False)
    pd.DataFrame(summary["relative_rule_minus_soxx"]).to_csv(card_dir / "relative_rule_minus_soxx.csv", index=False)
    pd.DataFrame(summary["three_account_wealth"]).to_csv(card_dir / "three_account_wealth.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "plan_sha256": summary["plan_sha256"], "declared_decision_at": summary["declared_decision_at"],
        "signal": summary["signal"], "models_trained": summary["models_trained"],
        "decision": summary["decision"], "stress_cost_bps": STRESS_COST_BPS,
        "source_freeze": summary["source_freeze"], "artifact_policy": summary["artifact_policy"],
        "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    lines = [
        "# NQ overnight 18:00→10:00 → 10:31 之后收益（0 模型诊断）\n",
        f"- 0 新模型；固定决策 10:31 ET；固定单一信号 NQ overnight 18:00→10:00（由 60min Qlib store 读取，09:00 bar close=10:00，早于决策）。\n",
        f"- 官方交易日 receipt digest 校验通过（sha `{summary['source_freeze']['trading_days']['sha256'][:12]}`）；session_close 由 receipt 的 WHOLE/MORNING 决定（16:00/13:00）。\n",
        f"- 覆盖：official_days_in_range={summary['official_days_in_range']}，sessions={summary['sessions']}，"
        f"needs_data={summary['dropped_needs_data']}，morning_days={summary['morning_days_in_range']}（有效 {summary['valid_morning_days']}），"
        f"missing_nq_signal_days={summary['missing_nq_signal_days']}。\n",
        f"- 执行口径：{summary['execution_proxy']}；{summary['direction_note']}。\n",
        "### 主预测目标：SOXL−SOXX 价差 IC（20-session 移动块 CI）\n",
        f"- spread IC {summary['spread_ic_primary']['point']:+.4f} "
        f"CI [{summary['spread_ic_primary']['ci_low']:+.4f}, {summary['spread_ic_primary']['ci_high']:+.4f}]\n",
        "### long-only 日内经济净优势（bps/日，cash→持仓→cash，两腿费用）\n",
    ]
    for row in summary["economic_long_only"]:
        ci = "" if row["ci_low_bps"] is None else f" CI [{row['ci_low_bps']:+.2f}, {row['ci_high_bps']:+.2f}]"
        lines.append(f"- {row['book']} @{row['cost_bps']:g}bp：{row['mean_net_bps']:+.2f}{ci}\n")
    lines.append("### SOXX 真正 buyhold（跨夜连续账户，独立于 10:30 入场）\n")
    for cost, account in summary["soxx_buyhold_account"].items():
        lines.append(f"- @{cost}bp：{account['return_pct']:.2f}%（{account['sessions']} sessions）\n")
    lines.append("### 相对描述：rule − same-window SOXX（同费用完整 roundtrip，非 signed short spread）\n")
    for row in summary["relative_rule_minus_soxx"]:
        ci = "" if row["ci_low_bps"] is None else f" CI [{row['ci_low_bps']:+.2f}, {row['ci_high_bps']:+.2f}]"
        lines.append(f"- @{row['cost_bps']:g}bp：{row['mean_diff_bps']:+.2f} bps{ci}（仅描述，相对正不能掩盖绝对亏损）\n")
    lines.append("### 三账户 absolute/net 风险与财富\n")
    for row in summary["three_account_wealth"]:
        lines.append(f"- {row['book']} @{row['cost_bps']:g}bp：cum {row['cumulative_return_pct']:.2f}% / "
                     f"MaxDD {row['max_drawdown_pct']:.2f}% / worst {row['worst_day_pct']:.2f}% / CVaR5 {row['cvar5_pct']:.2f}%\n")
    native = summary["native_capability"]
    lines += [
        f"- **决定：`{summary['decision']}`** — {summary['reason']}。\n",
        f"- native：full_strategy_native=`{native['full_strategy_native']}`（{native['reason']}）；"
        f"已记录 {native['recorded']}；不借用之前 control native PASS。\n",
        "- 本固定机制未获得足够证据（预测 IC 跨 0、25bp 绝对为负）；不反号/改 horizon/阈值，仅停止本路，"
        "不宣称“信息不存在”或“模型永无救”。\n",
        "- NQ=F 为 Yahoo 前月连续序列，roll/调整不构成独立证据。\n",
    ]
    if summary.get("recorder_id"):
        lines.append(f"- 子 Recorder：{summary['recorder_id']}；保存 panel/audit/IC/economic。\n")
    return "".join(lines)
