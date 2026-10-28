"""Point-in-time f696 plus dated b472-style action/recovery development scaffold."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.backtest.full_path_tiered import replay_full_path
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.data.minute_intake import atomic_json
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec, _build_teacher_training_frame,
    _fit_recovery_model,
)
from src.workflow.stage1_dated_controller import DatedModel, DatedStage1Controllers
from src.workflow.stage1_f696_training_reference import prepare as prepare_f696_source
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_sources(root: Path, plan: dict):
    from qlib.workflow import R

    card = root / plan["action_card"]
    bundle_path = card / "frozen_bundle_manifest.json"
    if _sha(bundle_path) != plan["action_bundle_sha256"]:
        raise ValueError("blocked: b472 frozen bundle changed")
    bundle = json.loads(bundle_path.read_text())
    entries = {item["path"]: item for item in bundle["artifacts"]}
    for rel, expected in plan["action_artifacts"].items():
        path = card / rel
        if rel not in entries or _sha(path) != expected or entries[rel]["sha256"] != expected:
            raise ValueError("blocked: b472 annual action source differs")
    with R.uri_context(str(card / "qlib_recorder")):
        recorder = R.get_recorder(experiment_id=plan["action_experiment_id"],
                                  recorder_id=bundle["recorder_id"])
        if recorder.status != "FINISHED":
            raise ValueError("blocked: b472 source Recorder unfinished")
        models = recorder.load_object(plan["action_model_object"])
    diagnostics = pd.read_csv(card / plan["action_diagnostics_artifact"])
    return models, diagnostics


def prepare(*, root: Path, card: Path, plan: dict) -> dict:
    root, card = Path(root).resolve(), Path(card).resolve()
    old_card = root / plan["old_reference_card"]
    if (_sha(old_card / "plan.json") != plan["old_reference_plan_sha256"]
            or _sha(old_card / "summary.json") != plan["old_reference_summary_sha256"]):
        raise ValueError("blocked: proposal-matched development source changed")
    source = prepare_f696_source(root=root, card=old_card,
        plan=json.loads((old_card / "plan.json").read_text()))
    frame = source["frame"].copy()
    if len(frame) != plan["expected_sessions"]:
        raise ValueError("blocked: covered development calendar differs")
    models, diagnostics = _action_sources(root, plan)
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=root,
        **source["action_plan"]["recovery_spec"])
    action_spec = Stage1OpenActionValueSpec(root=root,
        **source["action_plan"]["action_spec"])
    if source["selection"] != plan["recovery_selection"]:
        raise ValueError("blocked: frozen recovery selection differs")
    open_models = {}
    for year in plan["signal_years"]:
        row = diagnostics.loc[diagnostics.model_year.astype(str).eq(str(year))]
        if len(row) != 1 or year not in models or str(row.iloc[0].training_end) != f"{year - 1}-12-31":
            raise ValueError("blocked: dated opening model cutoff/identity differs")
        open_models[year] = DatedModel(models[year], row.iloc[0].training_end,
            f"{plan['action_recorder_id']}:open_action_value_models/{year}")
    # A new, explicitly disclosed teacher uses the same covered 09:47-to-next-
    # 09:47 price plane as the physical account. The old action-card recovery
    # teacher was not serialized and cannot be silently reconstructed from a
    # currently dirty legacy provider.
    teacher_frame = frame.copy()
    teacher_frame["segment"] = "development_walk_forward"
    teacher_frame["soxl_return"] = teacher_frame.soxl_exit_open / teacher_frame.soxl_open - 1.
    teacher_frame["soxx_return"] = teacher_frame.soxx_exit_open / teacher_frame.soxx_open - 1.
    teacher = _build_teacher_training_frame(teacher_frame, spec=recovery_spec,
        teacher_beta=recovery_spec.teacher_recovery_beta)
    if not pd.to_datetime(teacher.label_available_at, utc=True).notna().all():
        raise ValueError("blocked: recovery teacher label maturity missing")
    support = {}
    for year in plan["signal_years"]:
        cutoff = pd.Timestamp(teacher.attrs["recovery_training_cutoffs"][year])
        eligible = teacher.year.lt(year) & teacher.year.ge(2019)
        eligible &= pd.to_datetime(teacher.label_available_at, utc=True).lt(cutoff)
        sub = teacher.loc[eligible]
        support[str(year)] = {"rows": len(sub), "positive": int(sub.teacher_action_label.eq(1).sum()),
            "negative": int(sub.teacher_action_label.eq(0).sum()),
            "latest_label_available_at": str(pd.to_datetime(sub.label_available_at, utc=True).max()) if len(sub) else None,
            "cutoff": str(cutoff)}
        expected = plan["recovery_training_support"][str(year)]
        if {key: support[str(year)][key] for key in ("rows", "positive", "negative")} != expected:
            raise ValueError("blocked: OOF recovery teacher cohort differs")
    return {"frame": frame, "open_models": open_models, "teacher": teacher,
        "recovery_support": support, "recovery_spec": recovery_spec,
        "action_spec": action_spec, "selection": source["selection"],
        "source_preflight": source["receipt"]}


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import yaml
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        summary = json.loads((card / "summary.json").read_text())
        if summary["plan_sha256"] != digest:
            raise ValueError("blocked: finalized OOF reference plan changed")
        return summary
    prepared = prepare(root=root, card=card, plan=plan)
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": digest, "recovery_fit_attempts": [], "recovery_fits_done": [],
        "native_attempts": 0, "native_done": False}
    if progress["plan_sha256"] != digest:
        raise ValueError("blocked: OOF reference attempt belongs to another plan")
    tracking = card / "qlib_recorder"
    kwargs = {"experiment_name": card.name, "uri": str(tracking)}
    if progress.get("recorder_id"):
        kwargs.update(recorder_id=progress["recorder_id"], resume=True)
    with R.start(**kwargs):
        recorder = R.get_recorder()
        progress.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(progress, progress_path)
        recorder.save_objects(artifact_path="inputs", plan=plan,
            source_preflight=prepared["source_preflight"],
            recovery_support=prepared["recovery_support"],
            teacher=prepared["teacher"])
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            dated_controller=(root / "src/workflow/stage1_dated_controller.py").read_text(),
            recovery=(root / "src/workflow/intraday_recovery_direct_allocation.py").read_text())
        recovery_models = {}
        for year in plan["signal_years"]:
            if year in plan["explicit_no_recovery_model_years"]:
                recovery_models[year] = DatedModel(None, f"{year - 1}-12-31",
                    f"explicit_no_model_{year}_insufficient_prior_teacher", explicit_no_model=True)
                continue
            key = str(year)
            if key in progress["recovery_fits_done"]:
                model = recorder.load_object("models/recovery_" + key)
            else:
                if (key in progress["recovery_fit_attempts"]
                        or len(progress["recovery_fit_attempts"]) >= plan["budget"]["recovery_fit_attempts"]):
                    raise ValueError("blocked: OOF recovery fit attempt consumed")
                progress["recovery_fit_attempts"].append(key)
                atomic_json(progress, progress_path)
                model = _fit_recovery_model(prepared["teacher"], year=year,
                    c_value=float(plan["recovery_selection"]["logistic_c"]))
                if model is None:
                    raise ValueError("blocked: expected dated recovery model did not fit")
                recorder.save_objects(artifact_path="models", **{"recovery_" + key: model})
                progress["recovery_fits_done"].append(key)
                atomic_json(progress, progress_path)
            recovery_models[year] = DatedModel(model, f"{year - 1}-12-31",
                f"{recorder.id}:models/recovery_{year}")
            recorder.log_metrics(completed_recovery_fits=float(len(progress["recovery_fits_done"])))
        recorder.save_objects(artifact_path="models", **{f"open_{year}": value.model
            for year, value in prepared["open_models"].items()})
    schedule = DatedStage1Controllers(calendar=prepared["frame"][["signal_date", "trade_date"]],
        open_models=prepared["open_models"], recovery_models=recovery_models)
    schedule_audit = schedule.audit()
    if len(schedule_audit) != plan["expected_sessions"]:
        raise ValueError("blocked: annual controller schedule coverage differs")
    recorder.save_objects(artifact_path="analysis", controller_audit=schedule_audit)
    if progress["native_done"]:
        book = recorder.load_object("analysis/native/dated_f696_0")
    else:
        if progress["native_attempts"] >= plan["budget"]["native_account_attempts"]:
            raise ValueError("blocked: OOF native-account attempt consumed")
        progress["native_attempts"] += 1
        atomic_json(progress, progress_path)
        replay = replay_full_path(prepared["frame"], models=None, plan=plan,
            slippage_bps=0, controller_by_date=schedule,
            recovery_spec=prepared["recovery_spec"], action_spec=prepared["action_spec"],
            selection=prepared["selection"])
        book = native_tiered_stage1(prepared["frame"], replay,
            provider_uri=root / plan["provider_uri"], tariff=plan["tariff"],
            slippage_bps=0, initial_cash=plan["initial_cash"], freq="1min",
            opening_offset=pd.Timedelta(hours=9, minutes=47), position_type=QlibContinuousPosition)
        book["targets"] = replay["targets"]
        recorder.save_objects(artifact_path="analysis/native", dated_f696_0=book)
        progress["native_done"] = True
        atomic_json(progress, progress_path)
    if book["audit"]["status"] != "passed" or len(book["authority"]) != plan["expected_sessions"]:
        raise ValueError("blocked: dated training account/native reconciliation differs")
    summary = {"experiment": card.name, "plan_sha256": digest,
        "decision": "dated_2018_2022_training_reference_passed_needs_new_label_design",
        "objective_achieved": False, "historical_research_passed": False,
        "source_role": "new_covered_walk_forward_training_scaffold_not_frozen_live_b472_development_replay",
        "sessions": len(book["authority"]), "new_recovery_model_fits": len(progress["recovery_fits_done"]),
        "new_native_accounts": 1, "native_audit": book["audit"],
        "opening_execute_days": int(book["authority"].execute_open_target.sum()),
        "recovery_days": int(book["authority"].recovery_action.sum()),
        "recovery_support": prepared["recovery_support"],
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "provider_uri": str(root / plan["provider_uri"]),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*", "analysis/*"]}}
    recorder.save_objects(artifact_path="analysis", summary=summary)
    recorder.log_metrics(dated_training_sessions=float(summary["sessions"]))
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# f696 逐层点时开发期账户\n\n"
        f"使用f696已保存开发期提案、b472已保存的2018–2022逐年开盘模型，及本卡在同源09:47收益教师上严格按前一年成熟标签拟合的2020–2022逐年恢复模型；2018/2019明确无模型。"
        f"{summary['sessions']}个信号日完成原生Qlib 0bp Tiered自身账户，开盘执行{summary['opening_execute_days']}日、恢复{summary['recovery_days']}日。"
        "逐日模型与训练截止清单保存在Recorder `analysis/controller_audit`。"
        "这是新的点时开发训练框架，不冒充原封存最终b472模型的样本内路径，也不重写旧921个标签。"
        "价格使用既有调整价点位与原始股数费用，尚未单独认证应收分红、真实历史账单或完整分钟风险。"
        "本卡无候选特征拟合或SOXX策略通过；下一步先核对新账户局部动作可达性，再事前冻结新接收任务。\n")
    atomic_json(summary, card / "summary.json")
    return summary
