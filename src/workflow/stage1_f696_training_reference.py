"""Exact f696 proposals with the frozen b472 controller on covered development dates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.full_path_tiered import replay_full_path
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.data.minute_intake import atomic_json
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.minute_action_value import _source
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.stage1_runtime_release import load_runtime_release


TARGETS = ("soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    from qlib.data import D
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    source = plan["sources"]
    for key, filename in (("allocation_summary_sha256", root / source["allocation_card"] / "summary.json"),
                          ("physical_summary_sha256", root / source["physical_card"] / "summary.json"),
                          ("provider_manifest_sha256", root / plan["provider_uri"] / "qlib_store_manifest.json"),
                          ("release_sha256", root / plan["release_path"])):
        if _sha(filename) != source[key]:
            raise ValueError("blocked: source identity differs: " + key)
    allocation = root / source["allocation_card"] / "qlib_recorder" / source["allocation_experiment_id"] / source["allocation_recorder_id"] / "artifacts" / source["allocation_predictions_artifact"]
    physical = root / source["physical_card"] / "qlib_recorder" / source["physical_experiment_id"] / source["physical_recorder_id"] / "artifacts/inputs/source_frame"
    if _sha(allocation) != source["allocation_predictions_sha256"] or _sha(physical) != source["physical_frame_sha256"]:
        raise ValueError("blocked: saved Qlib source object differs")
    physical_summary = json.loads((root / source["physical_card"] / "summary.json").read_text())
    if not physical_summary["nas_readback"]["verified"]:
        raise ValueError("blocked: physical source NAS readback absent")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    with R.uri_context(str(root / source["physical_card"] / "qlib_recorder")):
        recorder = R.get_recorder(experiment_id=source["physical_experiment_id"],
                                  recorder_id=source["physical_recorder_id"])
        if recorder.status != "FINISHED":
            raise ValueError("blocked: physical source Recorder unfinished")
        frame = recorder.load_object("inputs/source_frame").reset_index(drop=True)
    predictions = pd.read_csv(allocation)
    predictions = predictions.loc[predictions.segment.eq("development_walk_forward"),
                                  ["signal_date", *TARGETS]].copy()
    predictions.signal_date = pd.to_datetime(predictions.signal_date).dt.normalize()
    frame.signal_date = pd.to_datetime(frame.signal_date).dt.normalize()
    if len(predictions) != plan["expected_proposals"] or len(frame) != plan["expected_source_rows"]:
        raise ValueError("blocked: saved proposal or source frame count differs")
    if predictions.signal_date.duplicated().any() or frame.signal_date.duplicated().any():
        raise ValueError("blocked: duplicate source signal date")
    joined = frame[["signal_date"]].merge(predictions, on="signal_date", how="left", validate="one_to_one")
    covered = joined[list(TARGETS)].notna().all(axis=1)
    if (int(covered.sum()) != plan["expected_covered_rows"] or not covered.iloc[:plan["expected_covered_rows"]].all()
            or covered.iloc[plan["expected_covered_rows"]:].any()):
        raise ValueError("blocked: physical matched cohort is not the frozen prefix")
    cohort = frame.iloc[:plan["expected_covered_rows"]].copy()
    aligned = predictions.set_index("signal_date").loc[pd.DatetimeIndex(cohort.signal_date), list(TARGETS)]
    if not np.isfinite(aligned.to_numpy(float)).all() or not np.allclose(aligned[list(TARGETS[:3])].sum(axis=1), 1., atol=1e-8):
        raise ValueError("blocked: frozen f696 outputs invalid")
    for name in TARGETS:
        cohort[name] = aligned[name].to_numpy(float)
    if (str(cohort.signal_date.iloc[0].date()) != plan["first_signal"]
            or str(cohort.signal_date.iloc[-1].date()) != plan["last_signal"]):
        raise ValueError("blocked: covered cohort endpoints differ")
    minutes = pd.DatetimeIndex(D.calendar(start_time=cohort.trade_date.iloc[0],
        end_time=cohort.exit_date.iloc[-1] + pd.Timedelta(hours=10, minutes=32), freq="1min"))
    needed = pd.DatetimeIndex([day + pd.Timedelta(hours=9, minutes=47) for day in cohort.trade_date])
    if len(needed.difference(minutes)):
        raise ValueError("needs_data: Qlib 09:47 native calendar incomplete")
    release = load_runtime_release(root, root / plan["release_path"])
    action, action_audit = _source(root, source["action"],
        ["inputs/plan", "controller/source_recovery_selection"])
    if (release["action_summary"]["source_stage1"]["selected_model"]
            != action["controller/source_recovery_selection"]):
        raise ValueError("blocked: released b472 action selection differs")
    receipt = {"status": "covered_exact_f696_proposals_verified", "proposals": len(predictions),
        "physical_source_rows": len(frame), "covered_rows": len(cohort),
        "f696_earlier_uncovered_rows": len(predictions) - len(cohort),
        "physical_terminal_without_f696_rows": len(frame) - len(cohort),
        "first_signal": plan["first_signal"], "last_signal": plan["last_signal"],
        "max_proposal_alignment_error": float(np.max(np.abs(cohort[list(TARGETS)].to_numpy(float) - aligned.to_numpy(float)))),
        "new_model_fits": 0, "new_accounts": 0, "action_source_audit": action_audit}
    return {"frame": cohort, "controller": release["models"], "action_plan": action["inputs/plan"],
        "selection": action["controller/source_recovery_selection"], "receipt": receipt}


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import yaml
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    plan_hash = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        summary = json.loads((card / "summary.json").read_text())
        if summary["plan_sha256"] != plan_hash:
            raise ValueError("blocked: finalized training-reference plan changed")
        return summary
    prepared = prepare(root=root, card=card, plan=plan)
    atomic_json(prepared["receipt"], card / "preflight.json")
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": plan_hash, "account_attempts": 0, "account_completed": False}
    if progress["plan_sha256"] != plan_hash:
        raise ValueError("blocked: previous attempt belongs to another plan")
    tracking = card / "qlib_recorder"
    kwargs = {"experiment_name": card.name, "uri": str(tracking)}
    if progress.get("recorder_id"):
        kwargs.update(recorder_id=progress["recorder_id"], resume=True)
    with R.start(**kwargs):
        recorder = R.get_recorder()
        progress.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(progress, progress_path)
        recorder.save_objects(artifact_path="inputs", plan=plan, source_receipt=prepared["receipt"],
            covered_frame=prepared["frame"])
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            replay=(root / "src/backtest/full_path_tiered.py").read_text(),
            native=(root / "src/backtest/stage1_ibkr.py").read_text())
    if progress["account_completed"]:
        book = recorder.load_object("analysis/native/exact_f696_b472_0")
    else:
        if progress["account_attempts"] >= plan["budget"]["native_account_attempts"]:
            raise ValueError("blocked: frozen training reference account attempt consumed")
        progress["account_attempts"] += 1
        atomic_json(progress, progress_path)
        action_plan = prepared["action_plan"]
        replay = replay_full_path(prepared["frame"], models=None, plan=plan,
            slippage_bps=0, controller=prepared["controller"],
            recovery_spec=IntradayRecoveryDirectAllocationSpec(root=root, **action_plan["recovery_spec"]),
            action_spec=Stage1OpenActionValueSpec(root=root, **action_plan["action_spec"]),
            selection=prepared["selection"])
        book = native_tiered_stage1(prepared["frame"], replay,
            provider_uri=root / plan["provider_uri"], tariff=plan["tariff"],
            slippage_bps=0, initial_cash=plan["initial_cash"], freq="1min",
            opening_offset=pd.Timedelta(hours=9, minutes=47), position_type=QlibContinuousPosition)
        book["targets"] = replay["targets"]
        recorder.save_objects(artifact_path="analysis/native", exact_f696_b472_0=book)
        progress["account_completed"] = True
        atomic_json(progress, progress_path)
    if book["audit"]["status"] != "passed" or len(book["authority"]) != plan["expected_covered_rows"]:
        raise ValueError("blocked: exact parent training reference native account differs")
    summary = {"experiment": card.name, "plan_sha256": plan_hash,
        "decision": "exact_f696_b472_covered_training_reference_passed_needs_local_labels",
        "objective_achieved": False, "historical_research_passed": False,
        "source_scope": "2018-10-12_to_2022-12-21_signal_covered_only; missing earlier minute coverage",
        "sessions": len(book["authority"]), "native_audit": book["audit"],
        "new_model_fits": 0, "new_native_accounts": 1,
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "provider_uri": str(root / plan["provider_uri"]),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/native/*"]}}
    recorder.save_objects(artifact_path="analysis", summary=summary)
    recorder.log_metrics(training_reference_sessions=float(summary["sessions"]))
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# f696+b472 精确提案开发期物理账户\n\n"
        f"Qlib 原生账户覆盖 {summary['sessions']} 个2018-10-12至2022-12-21信号日，0bp额外滑点仍扣Tiered费用。"
        "五项f696提案逐行对齐并替换旧27baa来源，b472开盘动作与恢复模型使用自身账户状态。"
        "本卡只证明分钟源覆盖范围内的母本和账户可执行；2016至2018年早期及2022年末五日未建立同钟物理账户。"
        "没有新拟合或候选收益结论。下一步冻结同状态、共同复位的两类合法动作扣费标签；先检查跨年支持再拟合。\n")
    atomic_json(summary, card / "summary.json")
    return summary
