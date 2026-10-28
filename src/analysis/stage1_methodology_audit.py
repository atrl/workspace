"""Read-only provenance audit for Stage1 research estimands and training clocks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    plan_sha = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        old = json.loads((card / "summary.json").read_text())
        if old["plan_sha256"] != plan_sha:
            raise ValueError("blocked: methodology audit plan changed")
        return old
    for relpath, expected in plan["source_sha256"].items():
        if _sha(root / relpath) != expected:
            raise ValueError("blocked: methodology source changed: " + relpath)
    action = root / plan["action_card"]
    bundle = json.loads((action / "frozen_bundle_manifest.json").read_text())
    entries = {entry["path"]: entry for entry in bundle["artifacts"]}
    for relpath in plan["action_artifacts"]:
        entry = entries[relpath]
        path = action / relpath
        if _sha(path) != entry["sha256"] or path.stat().st_size != entry["size"]:
            raise ValueError("blocked: action bundle artifact differs")
    action_summary = json.loads((action / "summary.json").read_text())
    reference = json.loads((root / plan["reference_card"] / "summary.json").read_text())
    labels = json.loads((root / plan["label_card"] / "summary.json").read_text())
    tree = json.loads((root / plan["tree_card"] / "summary.json").read_text())
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    with R.uri_context(str(action / "qlib_recorder")):
        action_recorder = R.get_recorder(recorder_id=bundle["recorder_id"], experiment_id=plan["action_experiment_id"])
        models = action_recorder.load_object(plan["action_model_object"])
    if not set(plan["covered_years"]).issubset(models):
        raise ValueError("blocked: saved walk-forward action models unavailable")
    diagnostics = pd.read_csv(action / plan["action_diagnostics_artifact"])
    annual = diagnostics.loc[diagnostics.model_year.astype(str).isin([str(y) for y in plan["covered_years"]])]
    if len(annual) != len(plan["covered_years"]):
        raise ValueError("blocked: action model annual training provenance differs")
    for row in annual.itertuples():
        if int(str(row.training_end)[:4]) != int(row.model_year) - 1:
            raise ValueError("blocked: saved annual action model cutoff is not prior-year")
    final_end = action_summary["runtime_models"]["frozen_open_action_value_model"]["training_signal_end"]
    if pd.Timestamp(final_end) < pd.Timestamp(reference["source_scope"].split("_to_")[1].split("_")[0]):
        raise ValueError("blocked: frozen model training cutoff does not overlap reference")
    final = models[2023].named_steps["logisticregression"].coef_
    differences = {str(year): float(np.max(np.abs(models[year].named_steps["logisticregression"].coef_ - final)))
                   for year in plan["covered_years"]}
    if not all(value > 0 for value in differences.values()):
        raise ValueError("blocked: annual and frozen action models unexpectedly identical")
    tree_card = root / plan["tree_card"]
    with R.uri_context(str(tree_card / "qlib_recorder")):
        tree_model = R.get_recorder(recorder_id=tree["recorder_id"], experiment_id=tree["experiment_id"]).load_object("models/xgb")
    counts = tree_model.audit["label_counts"]
    if sum(counts.values()) != tree_model.audit["training_rows"]:
        raise ValueError("blocked: tree action label counts differ")
    result = {
        "decision": "methodology_correction_requires_oof_action_training_reference",
        "plan_sha256": plan_sha,
        "f696_reference_rows_with_final_b472": reference["sessions"],
        "reference_native_reconciliation_passed": reference["native_audit"]["status"] == "passed",
        "reference_has_separate_corporate_action_parity": reference["native_audit"]["actual_broker_fill_or_corporate_action_parity"],
        "frozen_b472_training_signal_end": final_end,
        "saved_prior_year_action_models": sorted(int(y) for y in plan["covered_years"]),
        "annual_vs_frozen_coefficient_max_abs": differences,
        "affected_local_label_rows": labels["label_audit"]["eligible"],
        "tree_one_step_label_counts_soxl_soxx_cash": counts,
        "tree_test_actions_soxx": json.loads((tree_card / "constraint_bridge.json").read_text())["cases"][0]["pure_soxx_predictions"],
        "tree_constrained_stop_days": {item["family"]: item["constrained_stop_days"]
            for item in json.loads((tree_card / "constraint_bridge.json").read_text())["cases"]},
        "scientific_scope": "Frozen numerical results remain valid for their recorded scaffold; the 2018-2022 b472 final-model path is in-sample, so its 617/304 local labels and downstream information screens cannot certify a strict walk-forward Stage1 receiver. The pure-action tree result does not establish continuous-allocation tree capacity or isolate individual restrictions.",
        "new_model_fits": 0, "new_accounts": 0, "new_recorders": 0,
    }
    (card / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = {"experiment": card.name, "plan_sha256": plan_sha,
        "decision": result["decision"], "objective_achieved": False,
        "historical_research_passed": False, "audit": "audit.json",
        "new_model_fits": 0, "new_native_accounts": 0,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [], "reused_native_recorders": [plan["action_card"], plan["reference_card"], plan["tree_card"]]}}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nmode: read_only\n")
    (card / "conclusion.md").write_text("# Stage1 研究方法纠正\n\n"
        f"2018–2022 的 {reference['sessions']} 日 f696 训练参考，开盘动作用的是训练信号截至 {final_end} 的最终 b472 模型。"
        "账本与 Qlib 对账通过只能证明这条已训练模型路径的会计一致；不能证明早年动作在当时可由该模型产生。"
        "原 b472 Recorder 已保存 2018–2022 逐年向前的动作模型，且其系数与最终模型不同。"
        f"在此路径上形成的 {labels['label_audit']['eligible']} 个局部标签及两次负信息门只能保留为条件诊断，"
        "不能当成严格逐年向前母本的最终信息结论。下一步需用逐年动作模型及相应恢复模型重建自身账户，"
        "核对完整价格/费用/分红范围，再冻结新标签和接收任务。\n\n"
        f"独立树模型的训练标签为 SOXL {counts['0']}、SOXX {counts['1']}、现金 {counts['2']}。"
        "它使用一步从现金出发的纯动作最优标签和硬 argmax，SOXX 中间配置的稀少具有目标几何原因；"
        "四组对照又把 SOXL 上限、现金底线及 4% 止损合并，不能隔离每项限制或代表连续权重树模型。"
        "复用2023–2026历史与只用可用点位的风险记录仍不能充当独立验证或完整分钟风险资格。\n")
    return summary
