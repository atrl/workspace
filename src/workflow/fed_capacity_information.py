"""Preregistered forecast-information gate before another portfolio search."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.model.capacity_forecast import CapacityForecast
from src.workflow.minute_action_value import _source


def evaluation_mask(truth, metadata, fold, as_of):
    cutoff = (pd.Timestamp(fold["end"]) + pd.Timedelta(days=1)) if fold["name"].startswith("development") else pd.Timestamp(as_of) + pd.Timedelta(hours=16, minutes=1)
    maturity = pd.to_datetime(metadata["label_available_at"])
    return truth.notna().all(axis=1) & maturity.notna() & maturity.lt(cutoff)


def _bootstrap(values, plan):
    values = np.asarray(values, float); count = len(values)
    rng = np.random.default_rng(plan["bootstrap_seed"])
    indices = rng.integers(count, size=plan["bootstrap_samples"]); totals = values[indices].copy()
    for _ in range(1, count):
        restart = rng.random(len(indices)) < 1 / plan["bootstrap_mean_block"]
        indices = (indices + 1) % count
        indices[restart] = rng.integers(count, size=restart.sum())
        totals += values[indices]
    means = totals / count
    return {"observations": count, "mean_joint_normalized_loss_improvement": float(values.mean()),
        "lower_bound": float(np.quantile(means, plan["bootstrap_alpha"])),
        "upper_bound": float(np.quantile(means, 1 - plan["bootstrap_alpha"])),
        "method": "paired stationary bootstrap of normalized loss difference", "block_length": plan["bootstrap_mean_block"],
        "does_not_correct_all_previous_research_selection": True}


def run_capacity_information(*, root, card, plan):
    import qlib
    import yaml
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    if (card / "summary.json").exists():
        return json.loads((card / "summary.json").read_text())
    qlib.init(provider_uri=str(root / "data/qlib/full_path_tiered_allocation_20261013"), region="us", kernels=1)
    prices, price_audit = _source(root, plan["price_source"], ["inputs/frame"])
    cap, cap_audit = _source(root, plan["capacity_source"], ["analysis/information", "analysis/audit", "analysis/store_manifest"])
    if not cap["analysis/audit"]["source_complete"]:
        raise ValueError("needs_data: source qualification incomplete")
    frame = prices["inputs/frame"].copy()
    state = cap["analysis/information"].set_index("datetime").reindex(frame.index)
    provider = Path(cap["analysis/store_manifest"]["provider_uri"])
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    native = D.features(["US_SEMI_CAPACITY"], ["$known", *["$" + name for name in plan["capacity_features"]]], freq="day").xs("US_SEMI_CAPACITY", level="instrument").reindex(frame.index)
    for name in ["known", *plan["capacity_features"]]:
        if not np.allclose(native["$" + name], state[name], rtol=2e-7, atol=1e-5, equal_nan=True):
            raise ValueError("blocked: capacity cache differs from frozen Recorder")
        frame["capacity_known" if name == "known" else name] = native["$" + name]
    frame["release_age_days"] = (frame.trade_date - pd.to_datetime(state.release_date)).dt.days.astype(float)
    frame["feature_available_at"] = pd.concat([frame.feature_available_at, pd.to_datetime(state.available_at)], axis=1).max(axis=1)
    future_prices = np.r_[frame.soxx_open.to_numpy(float), float(frame.soxx_exit_open.iloc[-1])]
    future_dates = pd.DatetimeIndex([*frame.trade_date, frame.exit_date.iloc[-1]])
    labels = np.full((len(frame), 2), np.nan)
    maturity = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    horizon = plan["horizon_sessions"]
    for number in range(len(frame)):
        end = number + horizon
        if end >= len(future_prices):
            continue
        returns = future_prices[number + 1:end + 1] / future_prices[number:end] - 1
        labels[number] = [future_prices[end] / future_prices[number] - 1,
            np.log(max(float(np.mean(np.minimum(returns, 0) ** 2)), plan["variance_numerical_floor"]))]
        maturity.iloc[number] = future_dates[end] + pd.Timedelta(hours=9, minutes=49)
    columns = plan["base_features"] + plan["capacity_features"]
    frame["label_available_at"] = maturity
    label = pd.DataFrame(labels, index=frame.index, columns=plan["targets"])
    grouped = pd.concat({"feature": frame[columns], "label": label,
        "meta": frame[["feature_available_at", "decision_at", "label_available_at"]]}, axis=1)
    grouped.index = pd.MultiIndex.from_arrays([frame.index, ["SOXX"] * len(frame)], names=["datetime", "instrument"])
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="capacity_return_and_downside_information", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=hashlib.sha256((card / "plan.json").read_bytes()).hexdigest(), maximum_model_fits=6, maximum_backtests=0)
        recorder.save_objects(artifact_path="inputs", grouped=grouped, plan=plan, source_audit={"prices": price_audit, "capacity": cap_audit})
    comparisons, increments, predictions, audits, paired_losses = [], [], {}, [], {}
    for fold in plan["folds"]:
        bounds = frame.index[frame.signal_date.between(fold["start"], fold["end"])]
        dataset = DatasetH(handler=DataHandlerLP.from_df(grouped), segments={
            "train": (None, pd.Timestamp(fold["cutoff"]) - pd.Timedelta(nanoseconds=1)), "test": (bounds[0], bounds[-1])})
        per_family = {}
        for family, features in [("price_calendar", plan["base_features"]), ("capacity", columns)]:
            if len(audits) >= plan["maximum_model_fits"]:
                raise ValueError("blocked: fixed capacity fit budget exhausted")
            model = CapacityForecast(features, plan["targets"], fold["cutoff"], plan["ridge_alpha"]).fit(dataset)
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
        improvement = per_family["price_calendar"] - per_family["capacity"]
        paired_losses[fold["name"]] = improvement
        increments.append({"fold": fold["name"], **_bootstrap(improvement, plan)})
    pooled = _bootstrap(np.concatenate([paired_losses[fold["name"]] for fold in plan["folds"] if fold["name"].startswith("development")]), plan)
    increments.append({"fold": "pooled_development", **pooled})
    passed = all(row["mean_joint_normalized_loss_improvement"] > 0 for row in increments if row["fold"].startswith("development")) and pooled["lower_bound"] > 0
    metrics, improvement = pd.DataFrame(comparisons), pd.DataFrame(increments)
    summary = {"experiment": plan["experiment"], "plan_sha256": hashlib.sha256((card / "plan.json").read_bytes()).hexdigest(),
        "decision": "information_gate_pass_needs_portfolio_experiment" if passed else "not_promoted_capacity_information_gate_failed",
        "information_gate_pass": bool(passed), "objective_achieved": False, "goal_remains_active": True,
        "new_model_fits": len(audits), "backtests": 0, "training_audit": audits, "information_increment": increments,
        "unmatured_terminal_rows_retained": int(label.isna().any(axis=1).sum()), "historical_windows_reused": True,
        "provider_uri": str(provider), "as_of": plan["as_of"], "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking), "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "models/*", "analysis/predictions", "analysis/paired_losses"]},
        "no_net_alpha_claim": True, "production_enabled": False}
    recorder.save_objects(artifact_path="analysis", predictions=predictions, paired_losses=paired_losses, metrics=metrics,
        increment=improvement, summary=summary)
    recorder.save_objects(artifact_path="sources", model_code=(root / "src/model/capacity_forecast.py").read_text(),
        workflow_code=(root / "src/workflow/fed_capacity_information.py").read_text())
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size} for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    metrics.to_csv(card / "forecast_metrics.csv", index=False); improvement.to_csv(card / "information_increment.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": str(provider), "recorder_uri": str(tracking),
        "as_of": plan["as_of"], "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card / "summary.json")
    text = f"# 原始产能利用率信息增量结论\n\n已执行{len(audits)}次固定线性拟合，未执行新组合回测。决定：`{summary['decision']}`。\n\n"
    text += "|窗口|联合标准化损失改善|区块下界|区块上界|\n|---|---:|---:|---:|\n"
    for row in increments:text += f"|{row['fold']}|{row['mean_joint_normalized_loss_improvement']:.4f}|{row['lower_bound']:.4f}|{row['upper_bound']:.4f}|\n"
    text += "\n正数代表相对价格/发布日历对照的预测误差下降。两个目标分别为20日SOXX收益和未来下行方差的log，训练内标准化后等权评价；独立目标指标保留在forecast_metrics.csv。标签重叠以20日区块评估，仍不能消除历史反复研究的选择偏差。\n\n"
    text += "只用当时已经发布的同口径原始版本，严格按成熟时点训练，终端未知标签保留。信息门通过也不等于成本后超额，仍需原生完整账户和独立前向证据。若信息门失败则结束本固定来源/模型组合，不改符号或另挑窗口。目标保持active，下一步依据本信息门决定是否值得进入组合检验。\n"
    (card / "conclusion.md").write_text(text)
    return summary
