"""Fixed monthly expanding-history calibration experiment, without portfolio claims."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.model.revenue_forecast import RevenueForecast
from src.workflow.fed_capacity_information import _bootstrap
from src.workflow.minute_action_value import _source


def run_monthly_price_update(*, root, card, plan):
    import qlib
    import yaml
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    digest = hashlib.sha256((card / "plan.json").read_bytes()).hexdigest()
    if (card / "summary.json").exists():
        saved = json.loads((card / "summary.json").read_text())
        if saved["plan_sha256"] != digest:
            raise ValueError("fixed monthly plan changed")
        return saved
    qlib.init(provider_uri=plan["provider_uri"], region="us", kernels=1)
    folds = [fold["name"] for fold in plan["folds"]]
    names = ["inputs/grouped", "analysis/predictions", *["models/" + fold + "_price_calendar" for fold in folds]]
    source, source_audit = _source(root, plan["source"], names)
    grouped = source["inputs/grouped"]
    indices = None
    for fold in folds:
        index = source["analysis/predictions"][(fold, "price_calendar")]["prediction"].index
        indices = index if indices is None else indices.union(index)
    indices = indices.sort_values()
    months = pd.DatetimeIndex(indices.get_level_values("datetime")).to_period("M")
    schedule = sorted(months.unique())
    if len(schedule) - len(plan["reuse_initial_months"]) != plan["maximum_new_model_fits"]:
        raise ValueError("monthly schedule differs from frozen fit budget")
    meta = grouped.loc[indices, "meta"]
    if not pd.to_datetime(meta.feature_available_at).le(pd.to_datetime(meta.decision_at)).all():
        raise ValueError("inference input not available at decision")
    handler = DataHandlerLP.from_df(grouped)
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="fixed_monthly_price_refit", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=digest, maximum_new_fits=plan["maximum_new_model_fits"], backtests=0)
        recorder.save_objects(artifact_path="inputs", source_audit=source_audit, plan=plan, grouped=grouped)
    monthly, audits, completed, reused = [], [], 0, 0
    for month in schedule:
        selected = indices[months == month]
        cutoff = month.start_time
        dataset = DatasetH(handler=handler, segments={"train": (None, cutoff - pd.Timedelta(nanoseconds=1)),
            "test": (selected.get_level_values("datetime").min(), selected.get_level_values("datetime").max())})
        reuse = plan["reuse_initial_months"].get(str(month))
        if reuse:
            model = source["models/" + reuse + "_price_calendar"]
            if model.columns != plan["features"] or model.targets != plan["targets"] or model.cutoff != cutoff or model.alpha != plan["ridge_alpha"]:
                raise ValueError("reused initial model identity differs")
            reused += 1
        else:
            if completed >= plan["maximum_new_model_fits"]:
                raise ValueError("monthly fit budget exhausted")
            model = RevenueForecast(plan["features"], plan["targets"], cutoff, plan["ridge_alpha"]).fit(dataset)
            completed += 1
        if pd.Timestamp(model.audit["latest_label_available_at"]) >= cutoff:
            raise ValueError("monthly training used an immature outcome")
        prediction = model.predict(dataset).loc[selected]
        if reuse:
            original = source["analysis/predictions"][(reuse, "price_calendar")]["prediction"]
            common = original.index.intersection(prediction.index)
            np.testing.assert_allclose(prediction.loc[common], original.loc[common], rtol=1e-12, atol=1e-12)
        monthly.append(prediction)
        audits.append({"month": str(month), "reused": bool(reuse), **model.audit})
        recorder.save_objects(artifact_path="models", **{"monthly_" + str(month).replace("-", "_"): model})
        recorder.log_metrics(completed_new_fits=completed, reused_initial_models=reused)
    updated = pd.concat(monthly).sort_index()
    if not updated.index.equals(indices) or not np.isfinite(updated).all().all():
        raise ValueError("monthly output changed calendar or omitted predictions")
    comparisons, increments, losses, comparisons_by_fold = [], [], {}, {}
    for fold in folds:
        saved = source["analysis/predictions"][(fold, "price_calendar")]
        model = source["models/" + fold + "_price_calendar"]
        truth, valid = saved["truth"], saved["label_mature"]
        families = {"frozen": saved["prediction"], "monthly": updated.loc[truth.index]}
        normalized = {}
        for family, prediction in families.items():
            error = prediction.loc[valid] - truth.loc[valid]
            normalized[family] = ((error.to_numpy() / model.y_scaler.scale_) ** 2).mean(axis=1)
            for target in plan["targets"]:
                comparisons.append({"fold": fold, "family": family, "target": target, "scored_rows": int(valid.sum()),
                    "rmse": float(np.sqrt(np.mean(error[target] ** 2))), "bias": float(error[target].mean()),
                    "rank_ic": float(prediction.loc[valid, target].corr(truth.loc[valid, target], method="spearman"))})
        losses[fold] = normalized["frozen"] - normalized["monthly"]
        increments.append({"fold": fold, **_bootstrap(losses[fold], plan)})
        comparisons_by_fold[fold] = {"predictions": families, "truth": truth, "label_mature": valid,
            "common_normalization_scale": model.y_scaler.scale_}
    pooled = _bootstrap(np.concatenate([losses[fold] for fold in folds if fold.startswith("development")]), plan)
    increments.append({"fold": "pooled_development", **pooled})
    passed = all(row["mean_joint_normalized_loss_improvement"] > 0 for row in increments if row["fold"].startswith("development")) and pooled["lower_bound"] > 0
    summary = {"experiment": plan["experiment"], "plan_sha256": digest,
        "decision": "calibration_gate_pass_needs_full_account_test" if passed else "not_promoted_monthly_calibration_gate_failed",
        "calibration_gate_pass": bool(passed), "new_model_fits": completed, "reused_models": reused, "scheduled_months": len(schedule),
        "backtests": 0, "objective_achieved": False, "goal_remains_active": True, "production_enabled": False,
        "information_increment": increments, "historical_windows_reused": True, "normalization": "same frozen fold scale for both methods",
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id), "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id), "provider_uri": plan["provider_uri"],
        "as_of": plan["as_of"], "qlib_docs": "https://qlib.readthedocs.io/en/latest/component/model.html",
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "models/*", "analysis/*", "sources/*"]}}
    recorder.save_objects(artifact_path="analysis", updated_predictions=updated, comparisons=comparisons_by_fold,
        training_audits=audits, paired_losses=losses, metrics=pd.DataFrame(comparisons), increment=pd.DataFrame(increments), summary=summary)
    recorder.save_objects(artifact_path="sources", workflow_code=Path(__file__).read_text(), model_code=(root / "src/model/revenue_forecast.py").read_text())
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size} for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    pd.DataFrame(comparisons).to_csv(card / "forecast_metrics.csv", index=False)
    pd.DataFrame(increments).to_csv(card / "information_increment.csv", index=False)
    pd.DataFrame([{key: row[key] for key in ["month", "reused", "training_rows", "cutoff", "latest_label_available_at"]} for row in audits]).to_csv(card / "training_schedule.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": plan["provider_uri"],
        "recorder_uri": str(tracking), "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card / "summary.json")
    text = f"# 固定月度重训校准检验\n\n执行{completed}次新增拟合、复用{reused}个起始模型，覆盖{len(schedule)}个月，0组合回测。决定：`{summary['decision']}`。\n\n"
    text += "|窗口|联合预测损失改善|区块下界|区块上界|\n|---|---:|---:|---:|\n"
    for row in increments:
        text += f"|{row['fold']}|{row['mean_joint_normalized_loss_improvement']:.5f}|{row['lower_bound']:.5f}|{row['upper_bound']:.5f}|\n"
    text += "\n正数表示月度模型误差下降。两边采用相同冻结模型训练标签尺度，未利用更新模型变大的尺度制造改善。仅按月使用已成熟历史标签重训，扩展窗口、64个特征和Ridge10均固定；没有月内更新或再次选参。预测日历、标签和成熟掩码与原对照完全相同。\n\n"
    text += "历史已反复研究，区块区间不消除全部选择偏差。校准改善不是IBKR费用后的组合超额，更不是生产晋升；仍需完整账户、0/2/5bp情景、风险约束和独立前向验证。未改生产SIM，稳定超额目标保持active。固定方案失败则停止本更新规则，不调频率、窗口或参数继续搜索。\n"
    (card / "conclusion.md").write_text(text)
    return summary
