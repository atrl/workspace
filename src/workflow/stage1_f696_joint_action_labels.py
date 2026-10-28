"""Freeze and record legal own-state Stage1 opening action labels without fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.analysis.f696_joint_action_labels import build_labels
from src.data.minute_intake import atomic_json
from src.workflow.minute_action_value import _source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    import yaml
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        previous = json.loads((card / "summary.json").read_text())
        if previous["plan_sha256"] != digest:
            raise ValueError("blocked: fixed label plan changed")
        return previous
    source = plan["source"]
    if (_sha(root / source["card"] / "summary.json") != source["summary_sha256"]
            or _sha(root / plan["provider_uri"] / "qlib_store_manifest.json") != plan["provider_manifest_sha256"]):
        raise ValueError("blocked: matched training source identity changed")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    objects, source_audit = _source(root, source, ["inputs/covered_frame", "analysis/native/exact_f696_b472_0"])
    frame = objects["inputs/covered_frame"]
    book = objects["analysis/native/exact_f696_b472_0"]
    if (len(frame) != plan["expected_reference_rows"] or book["audit"]["status"] != "passed"
            or book["audit"]["slippage_bps"] != 0):
        raise ValueError("blocked: source account is not the frozen f696 0bp reference")
    if not pd.to_datetime(frame.feature_available_at).lt(pd.to_datetime(frame.decision_at)).all():
        raise ValueError("blocked: training information arrives after opening decision")
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": digest, "label_attempts": 0, "labels_done": False}
    if progress["plan_sha256"] != digest:
        raise ValueError("blocked: label attempt belongs to another plan")
    tracking = card / "qlib_recorder"
    kwargs = {"experiment_name": card.name, "uri": str(tracking)}
    if progress.get("recorder_id"):
        kwargs.update(recorder_id=progress["recorder_id"], resume=True)
    with R.start(**kwargs):
        recorder = R.get_recorder()
        progress.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(progress, progress_path)
        recorder.save_objects(artifact_path="inputs", plan=plan, source_audit=source_audit)
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            labels=(root / "src/analysis/f696_joint_action_labels.py").read_text())
    if progress["labels_done"]:
        labels = recorder.load_object("analysis/labels")
        audit = recorder.load_object("analysis/audit")
    else:
        if progress["label_attempts"] >= plan["budget"]["label_pass_attempts"]:
            raise ValueError("blocked: frozen label pass attempt consumed")
        progress["label_attempts"] += 1
        atomic_json(progress, progress_path)
        labels, audit = build_labels(frame, book, tariff=plan["tariff"],
                                    initial_cash=plan["initial_cash"])
        recorder.save_objects(artifact_path="analysis", labels=labels, audit=audit)
        progress["labels_done"] = True
        atomic_json(progress, progress_path)
    eligible = labels.loc[labels.eligible].copy()
    eligible["year"] = pd.DatetimeIndex(eligible.trade_date).year
    annual = eligible.groupby(["route", "year"], observed=True).local_advantage.agg(
        rows="size", positive=lambda x: int(x.gt(0).sum()), negative=lambda x: int(x.lt(0).sum()),
        mean="mean").reset_index()
    support = {}
    for route in plan["routes"]:
        group = annual.loc[annual.route.eq(route)]
        support[route] = {"rows": int(group.rows.sum()),
            "years_20_rows": int(group.rows.ge(20).sum()),
            "years_both_signs_5": int((group.positive.ge(5) & group.negative.ge(5)).sum()),
            "positive": int(group.positive.sum()), "negative": int(group.negative.sum())}
    passed = all(value["rows"] >= plan["gate"]["minimum_route_rows"]
        and value["years_20_rows"] >= plan["gate"]["minimum_years_with_20_rows"]
        and value["years_both_signs_5"] >= plan["gate"]["minimum_years_with_both_signs_5"]
        for value in support.values())
    summary = {"experiment": card.name, "plan_sha256": digest,
        "decision": "matched_joint_local_labels_pass_needs_information_screen" if passed
                    else "not_promoted_joint_local_label_support_failed",
        "objective_achieved": False, "historical_research_passed": False,
        "source_reference_recorder_id": source["recorder_id"], "label_audit": audit,
        "support": support, "gate_pass": bool(passed), "new_model_fits": 0,
        "new_native_accounts": 0, "recorder_id": recorder.id,
        "experiment_id": str(recorder.experiment_id), "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "provider_uri": str(root / plan["provider_uri"]),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/labels", "analysis/annual", "analysis/audit"]}}
    recorder.save_objects(artifact_path="analysis", annual=annual, summary=summary)
    recorder.log_metrics(eligible_local_labels=float(audit["eligible"]))
    annual.to_csv(card / "annual_support.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# f696+b472 两类开盘动作的同状态局部标签\n\n"
        "冻结完整开发期自身账户，分别在原风险上限受限层比较现金转SOXL，在非受限层比较名义beta守恒的SOXL+现金转SOXX。"
        "同一原持仓、原现金、月度Tiered费用状态在09:47分叉，下一交易日09:47复位到同一原参考目标；标签于09:49成熟。"
        "局部优势不是可直接交易的完整账户反事实。\n\n"
        f"结果：`{summary['decision']}`；支持与正负分布见 `annual_support.csv`。"
        "通过才允许另冻决策前信息门；信息门通过才可启动候选自身完整0bp账户。"
        "2016至2018分钟缺口、不同路径风险反馈和2023至2026历史复用仍是边界。\n")
    atomic_json(summary, card / "summary.json")
    return summary
