"""Preregistered forecast-information gate before another portfolio search."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.model.revenue_forecast import RevenueForecast
from src.data.guidance_information import guidance_information
from src.data.qlib_native import build_qlib_store_from_frame
from src.workflow.minute_action_value import _source


from src.workflow.fed_capacity_information import _bootstrap, evaluation_mask


def run_guidance_information(*, root, card, plan):
    import qlib
    import yaml
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    if (card / "summary.json").exists():
        previous = json.loads((card / "summary.json").read_text())
        if previous["plan_sha256"] != hashlib.sha256((card / "plan.json").read_bytes()).hexdigest():
            raise ValueError("fixed guidance plan changed")
        return previous
    qlib.init(provider_uri=str(root / "data/qlib/full_path_tiered_allocation_20261013"), region="us", kernels=1)
    prices, price_audit = _source(root, plan["price_source"], ["inputs/frame"])
    minute_days = pd.DatetimeIndex(D.calendar(freq="1min")).normalize().unique()
    qlib.init(provider_uri=str(root / plan["history_calendar"]), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day")).union(minute_days).sort_values()
    cap, cap_audit = _source(root, plan["review_source"], ["analysis/review", "inputs/source_manifest", "analysis/audit"])
    if cap["analysis/audit"]["calendar_events"] != 105:
        raise ValueError("incomplete review calendar")
    frame = prices["inputs/frame"].copy()
    state, events = guidance_information(cap["analysis/review"], cap["inputs/source_manifest"], calendar,
        active_sessions=plan["active_sessions"])
    for symbol in ["soxl", "soxx"]:
        frame["current_mark_return_"+symbol] = frame["known_"+symbol] / frame["close_"+symbol].shift(1) - 1
    fields = plan["event_controls"] + plan["guidance_features"]
    data = state[["datetime", *fields]].copy(); data["instrument"] = "GUIDANCE_INFO"
    provider = root / plan["information_provider"]
    store_manifest = build_qlib_store_from_frame(data, provider_uri=provider, freq="day", as_of=plan["as_of"],
        field_map={name:name for name in fields}, source="reviewed prospective guidance/consensus; conditional publication clock")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    native = D.features(["GUIDANCE_INFO"], ["$"+name for name in fields], freq="day").xs("GUIDANCE_INFO",level="instrument")
    if not native.index.equals(calendar):raise ValueError("information calendar differs")
    for name in fields:
        if not np.allclose(native["$"+name], state[name], rtol=2e-7, atol=1e-8, equal_nan=True):
            raise ValueError("native information readback differs")
        frame[name] = native["$"+name].reindex(frame.index)
    frame["feature_available_at"] = pd.concat([frame.feature_available_at,
        frame.trade_date + pd.Timedelta(hours=9,minutes=46)],axis=1).max(axis=1)
    future = {symbol:np.r_[frame[symbol+"_open"].to_numpy(float),float(frame[symbol+"_exit_open"].iloc[-1])] for symbol in ["soxl","soxx"]}
    future_dates = pd.DatetimeIndex([*frame.trade_date,frame.exit_date.iloc[-1]])
    labels = np.full((len(frame),2),np.nan)
    maturity = pd.Series(pd.NaT,index=frame.index,dtype="datetime64[ns]")
    for number in range(len(frame)):
        end = number + plan["horizon_sessions"]
        if end >= len(future_dates):continue
        xx = np.log(future["soxx"][end]/future["soxx"][number])
        xl = np.log(future["soxl"][end]/future["soxl"][number])
        labels[number] = [-xx,xl-xx]
        maturity.iloc[number] = future_dates[end]+pd.Timedelta(hours=9,minutes=49)
    columns = plan["base_features"] + plan["guidance_features"]
    frame["label_available_at"] = maturity
    label = pd.DataFrame(labels, index=frame.index, columns=plan["targets"])
    grouped = pd.concat({"feature": frame[columns], "label": label,
        "meta": frame[["feature_available_at", "decision_at", "label_available_at"]]}, axis=1)
    grouped.index = pd.MultiIndex.from_arrays([frame.index, ["SOXX"] * len(frame)], names=["datetime", "instrument"])
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="guidance_relative_return_information", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=hashlib.sha256((card / "plan.json").read_bytes()).hexdigest(), maximum_model_fits=6, maximum_backtests=0)
        recorder.save_objects(artifact_path="inputs", grouped=grouped, plan=plan, source_audit={"prices": price_audit, "actuals": cap_audit})
    comparisons, increments, predictions, audits, paired_losses = [], [], {}, [], {}
    for fold in plan["folds"]:
        bounds = frame.index[frame.signal_date.between(fold["start"], fold["end"])]
        dataset = DatasetH(handler=DataHandlerLP.from_df(grouped), segments={
            "train": (None, pd.Timestamp(fold["cutoff"]) - pd.Timedelta(nanoseconds=1)), "test": (bounds[0], bounds[-1])})
        per_family = {}
        for family, features in [("price_calendar", plan["base_features"]), ("guidance", columns)]:
            if len(audits) >= plan["maximum_model_fits"]:
                raise ValueError("blocked: fixed guidance fit budget exhausted")
            model = RevenueForecast(features, plan["targets"], fold["cutoff"], plan["ridge_alpha"]).fit(dataset)
            predicted = model.predict(dataset)
            truth = dataset.prepare("test", col_set="label")
            valid = evaluation_mask(truth, dataset.prepare("test", col_set="meta"), fold, plan["as_of"])
            error = predicted.loc[valid] - truth.loc[valid]
            losses = (error.to_numpy() / model.y_scaler.scale_) ** 2
            per_family[family] = losses.mean(axis=1)
            for target in plan["targets"]:
                comparisons.append({"fold": fold["name"], "family": family, "target": target, "observations": int(valid.sum()),
                    "rmse": float(np.sqrt(np.mean(error[target] ** 2))), "bias": float(error[target].mean()),
                    "rank_ic": float(predicted.loc[valid, target].corr(truth.loc[valid, target], method="spearman"))})
            predictions[(fold["name"], family)] = {"prediction": predicted, "truth": truth, "label_mature": valid}
            audits.append({"fold": fold["name"], "family": family, **model.audit})
            recorder.save_objects(artifact_path="models", **{fold["name"] + "_" + family: model})
            recorder.log_metrics(completed_model_fits=len(audits))
        improvement = per_family["price_calendar"] - per_family["guidance"]
        paired_losses[fold["name"]] = improvement
        increments.append({"fold": fold["name"], **_bootstrap(improvement, plan)})
    pooled = _bootstrap(np.concatenate([paired_losses[fold["name"]] for fold in plan["folds"] if fold["name"].startswith("development")]), plan)
    increments.append({"fold": "pooled_development", **pooled})
    passed = all(row["mean_joint_normalized_loss_improvement"] > 0 for row in increments if row["fold"].startswith("development")) and pooled["lower_bound"] > 0
    metrics, improvement = pd.DataFrame(comparisons), pd.DataFrame(increments)
    summary = {"experiment": plan["experiment"], "plan_sha256": hashlib.sha256((card / "plan.json").read_bytes()).hexdigest(),
        "decision": "conditional_information_pass_needs_portfolio_and_vintages" if passed else "not_promoted_guidance_information_gate_failed",
        "information_gate_pass": bool(passed), "historical_vintage_certified": False, "promotion_eligible": False, "source_audit": cap["analysis/audit"], "event_eligibility": events.status.value_counts().to_dict(), "qlib_docs": plan["qlib_docs"], "objective_achieved": False, "goal_remains_active": True,
        "new_model_fits": len(audits), "backtests": 0, "training_audit": audits, "information_increment": increments,
        "unmatured_terminal_rows_retained": int(label.isna().any(axis=1).sum()), "historical_windows_reused": True,
        "provider_uri": str(provider), "as_of": plan["as_of"], "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking), "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "models/*", "analysis/predictions", "analysis/paired_losses"]},
        "no_net_alpha_claim": True, "production_enabled": False}
    recorder.save_objects(artifact_path="inputs", information=state, events=events, store_manifest=store_manifest)
    recorder.save_objects(artifact_path="analysis", predictions=predictions, paired_losses=paired_losses, metrics=metrics,
        increment=improvement, summary=summary)
    recorder.save_objects(artifact_path="sources", model_code=(root / "src/model/revenue_forecast.py").read_text(),
        workflow_code=(root / "src/workflow/guidance_information.py").read_text())
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size} for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    metrics.to_csv(card / "forecast_metrics.csv", index=False); improvement.to_csv(card / "information_increment.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": str(provider), "recorder_uri": str(tracking),
        "as_of": plan["as_of"], "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    summary["training_audit_recorder_artifact"] = "analysis/summary"
    summary["training_audit"] = [{key: row[key] for key in ["fold", "family", "training_rows", "cutoff", "latest_label_available_at", "alpha"]}
        | {"feature_count": len(row["columns"])} for row in audits]
    atomic_json(summary, card / "summary.json")
    text = f"# 下一季度指引预期差信息增量诊断\n\n已执行{len(audits)}次固定线性拟合，未执行新组合回测。决定：`{summary['decision']}`。\n\n"
    text += "|窗口|联合标准化损失改善|区块下界|区块上界|\n|---|---:|---:|---:|\n"
    for row in increments:text += f"|{row['fold']}|{row['mean_joint_normalized_loss_improvement']:.4f}|{row['lower_bound']:.4f}|{row['upper_bound']:.4f}|\n"
    text += "\n正数代表相对价格、事件状态与年龄对照的预测误差下降。两个目标分别为5日cash相对SOXX、SOXL相对SOXX的对数收益，训练内标准化后等权评价；独立目标指标保留在forecast_metrics.csv。标签重叠以63日平均长度区块评估，仍不能消除历史反复研究的选择偏差。\n\n"
    text += "下一季度指引方向与数值差仅在审阅合格且无经济口径特殊事项的事件中启用20个原生交易日；未知保留NaN，仅模型用训练内均值编码，并提供相同状态控制。特征按已冻结的公告/PDF可用时间假设进入；历史原始链接与首次接收未认证，本结果为条件性诊断。严格按成熟时点训练，终端未知标签保留。信息门通过也不等于成本后超额，仍需原生完整账户和独立前向证据。若信息门失败则结束本固定来源/模型组合，不改符号或另挑窗口。目标保持active，下一步依据本信息门决定是否值得进入组合检验。\n"
    (card / "conclusion.md").write_text(text)
    return summary
