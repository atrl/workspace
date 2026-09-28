"""A registered, finite development-only check of the fixed HOLD model."""
from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.workflow import stage1_open_action_value as action
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
    _build_teacher_training_frame,
    _fit_recovery_model,
    _load_source_frame,
)


EXPERIMENT = "stage1_open_action_value_bounded_20260928"


def run_bounded_threshold_card(*, root: Path | str, experiment: str = EXPERIMENT) -> dict[str, Any]:
    """Read the pre-registered plan, exhaust its four candidates, then stop."""
    from qlib.workflow import R

    root = Path(root).resolve()
    card = root / "research" / experiment
    plan = json.loads((card / "selection_plan.json").read_text())
    if plan["threshold_candidates"] != [0.30, 0.35, 0.40, 0.45]:
        raise ValueError("blocked: bounded selection plan changed")
    reference = root / "research" / plan["fixed_model_source_experiment"]
    source_summary = json.loads((reference / "summary.json").read_text())
    if source_summary["recorder_id"] != plan["fixed_model_source_recorder"]:
        raise ValueError("blocked: registered model recorder changed")
    model_path = Path(source_summary["recorder_local_dir"]) / "artifacts" / reference.name / "open_action_value_models.pkl"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != plan["fixed_model_sha256"]:
        raise ValueError("blocked: registered HOLD models changed")
    with model_path.open("rb") as handle:
        models = pickle.load(handle)
    spec = action.Stage1OpenActionValueSpec(root=root, experiment=experiment)
    _, selected = action._source_contract(spec)
    source_spec = IntradayRecoveryDirectAllocationSpec(
        root=root, experiment=reference.name,
        source_experiment=source_summary["source_stage1"]["allocation_source_experiment"],
    )
    frame, source_audit = _load_source_frame(source_spec)
    development = frame[frame["segment"].eq("development_walk_forward")].copy()
    teacher = _build_teacher_training_frame(
        development, spec=source_spec, teacher_beta=source_spec.teacher_recovery_beta
    )
    recovery_models = {
        int(year): _fit_recovery_model(teacher, year=int(year), c_value=float(selected["logistic_c"]))
        for year in development["signal_date"].dt.year.unique()
    }
    baseline = action._simulate(
        development, base_spec=source_spec, teacher=teacher,
        c_value=float(selected["logistic_c"]), threshold=float(selected["probability_threshold"]),
        action_beta=float(selected["action_beta"]), alpha_floor=float(selected["minimum_source_alpha_probability"]),
        first_action_year=source_spec.first_intraday_action_year, model_cache={},
    )
    fallback = action._simulate_on_policy_router(
        development, models={}, recovery_models=recovery_models, source_spec=source_spec,
        spec=spec, selected_model=selected, segment="development_walk_forward",
    )

    def metrics(replay, variant, cost):
        return action._metrics(
            action._stress(replay, cost_bps=cost), variant=variant,
            segment="development_walk_forward", cost_bps=cost,
        )

    baseline_10, baseline_25 = metrics(baseline, "baseline", 10.0), metrics(baseline, "baseline", 25.0)
    rows, replay_tables = [], []
    for threshold in plan["threshold_candidates"]:
        candidate_spec = replace(spec, execute_probability_threshold=threshold)
        replay = action._simulate_on_policy_router(
            development, models=models, recovery_models=recovery_models,
            source_spec=source_spec, spec=candidate_spec, selected_model=selected,
            segment="development_walk_forward",
        )
        candidate_10, candidate_25 = metrics(replay, "candidate", 10.0), metrics(replay, "candidate", 25.0)
        annual = action._annual_metrics(baseline, replay, segment="development_walk_forward")
        gates = action._promotion_gates(
            baseline_10=baseline_10, baseline_25=baseline_25,
            candidate_10=candidate_10, candidate_25=candidate_25,
            annual=annual, fallback=fallback, baseline_replay=baseline, spec=candidate_spec,
        )
        rows.append({
            "execute_probability_threshold": threshold,
            "return_10bps_pct": candidate_10["return_pct"],
            "return_25bps_pct": candidate_25["return_pct"],
            "max_drawdown_pct": candidate_10["max_drawdown_pct"],
            "open_trade_days": candidate_10["open_trade_days"],
            "open_trade_day_reduction_pct": 100.0 * (1.0 - candidate_10["open_trade_days"] / baseline_10["open_trade_days"]),
            "total_turnover": candidate_10["total_turnover"],
            "all_gates_pass": all(gates.values()), **gates,
        })
        replay_tables.append(replay.assign(execute_probability_threshold=threshold))
    table = pd.DataFrame(rows)
    table.to_csv(card / "bounded_model_selection.csv", index=False)
    eligible = table[table["all_gates_pass"]].sort_values(
        ["return_25bps_pct", "execute_probability_threshold"], ascending=[False, True]
    )
    chosen = None if eligible.empty else float(eligible.iloc[0]["execute_probability_threshold"])
    R.set_uri(str(card / "qlib_recorder"))
    with tempfile.TemporaryDirectory(prefix="stage1_bounded_") as tmp:
        path = Path(tmp)
        pd.concat(replay_tables, ignore_index=True).to_csv(path / "development_replays.csv", index=False)
        with R.start(experiment_name=experiment + "_development_selection"):
            recorder = R.get_recorder()
            R.log_params(candidate_count=4, fixed_model_recorder=source_summary["recorder_id"], selection_uses_outer="false")
            R.log_metrics(passing_candidates=float(len(eligible)))
            recorder.log_artifact(str(path / "development_replays.csv"), artifact_path=experiment)
            recorder.log_artifact(str(card / "bounded_model_selection.csv"), artifact_path=experiment)
            recorder.log_artifact(str(card / "selection_plan.json"), artifact_path=experiment)
            selection_recorder = {"recorder_id": recorder.id, "recorder_local_dir": recorder.get_local_dir()}
    action._remove_qlib_code_snapshot_artifacts(Path(selection_recorder["recorder_local_dir"]))
    selection = {
        "candidate_count": 4, "total_examined_threshold_count": 5,
        "selected_threshold": chosen, "development_baseline_10bps": baseline_10,
        "development_baseline_25bps": baseline_25,
        "selection_uses_outer_test": False, "all_candidates": rows,
        "source_audit": source_audit, **selection_recorder,
    }
    (card / "bounded_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    if chosen is not None:
        (card / "workflow.yaml").write_text(
            f"experiment: {experiment}\nsource_recorder_id: {source_audit['source_recorder_id']}\n"
        )
        summary = action.run_stage1_open_action_value_card(
            root=root, experiment=experiment,
            source_spec=replace(source_spec, experiment=experiment),
            execute_probability_threshold=chosen,
        )
        summary["bounded_selection"] = selection
    else:
        summary = {
            "experiment": experiment,
            "decision": "blocked_no_registered_threshold_passed_development_gates",
            "development_gate_pass": False, "production_enabled": False,
            "selection_uses_outer_test": False, "outer_evaluated": False,
            "bounded_selection": selection, **selection_recorder,
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": [f"artifacts/{experiment}/development_replays.csv"]},
        }
        (card / "workflow.yaml").write_text(f"experiment: {experiment}\nselection_plan: selection_plan.json\nselection_uses_outer_test: false\nouter_evaluated: false\n")
        (card / "selected_model.json").write_text(json.dumps({"selected_threshold": None, "production_enabled": False}, indent=2) + "\n")
        conclusion = (
            "# 中文结论\n\n"
            "固定C=0.10和原模型，仅按预注册计划验证0.30/0.35/0.40/0.45四个执行阈值；先前0.50失败记录保留。"
            "所有候选只使用2016–2022开发期、候选自己的状态和原收益/成本/风险/降频门槛。\n\n"
            "没有候选通过全部门槛，因此停止本轮有界研究，不评估这些候选的outer，不激活模拟交易。"
            "完整候选表见bounded_model_selection.csv，逐日证据进入qlib recorder " + recorder.id + "。\n"
        )
        (card / "conclusion.md").write_text(conclusion)
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
