"""Bounded issuer-flow diagnostic, with retrospective-vintage limitations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.etf_share_history import parse_ishares_history, share_information
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.data.qlib_native import build_qlib_store_from_frame
from src.model.capacity_forecast import CapacityForecast
from src.workflow.fed_capacity_information import _bootstrap, evaluation_mask
from src.workflow.minute_action_value import _source


def run_etf_flow_information(*, root, card, plan):
    import qlib
    import yaml
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R

    digest = hashlib.sha256((card / "plan.json").read_bytes()).hexdigest()
    if (card / "summary.json").exists():
        previous = json.loads((card / "summary.json").read_text())
        if previous["plan_sha256"] != digest:
            raise ValueError("fixed diagnostic plan changed")
        return previous
    raw = (root / plan["issuer_workbook"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan["issuer_sha256"]:
        raise ValueError("frozen issuer workbook hash differs")
    history = parse_ishares_history(raw)
    qlib.init(provider_uri=str(root / plan["price_provider"]), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="1min")).normalize().unique().sort_values()
    prices, price_audit = _source(root, plan["price_source"], ["inputs/frame"])
    frame = prices["inputs/frame"].copy()
    state = share_information(history, calendar, plan["assumed_lag_sessions"])
    source_audit = {"raw_rows": len(history), "source_start": str(history.index.min().date()),
        "source_end": str(history.index.max().date()), "missing_shares_rows": int(history.shares.isna().sum()),
        "native_calendar_rows": len(calendar), "native_missing_shares": int(history.reindex(calendar).shares.isna().sum()),
        "share_count_basis": "issuer historical presentation; not certified original unadjusted shares",
        "split_treatment": "no extra split adjustment; ratios invariant to a constant share scale; uniform historical restatement assumed",
        "split_boundary_rows": history.loc["2024-03-05":"2024-03-08"].reset_index().astype(str).to_dict("records"),
        "source_sha256": plan["issuer_sha256"], "historical_publication_time_verified": False,
        "historical_first_receipt_verified": False, "future_revision_absence_verified": False,
        "availability": "diagnostic assumption: two native sessions after observation at09:31; not proof of historical availability",
        "soxl_share_history_used": False, "no_alpha_or_promotion_claim": True}
    provider = root / plan["information_provider"]
    fields = [*plan["flow_features"], "known"]
    store_frame = state[fields].copy(); store_frame["datetime"] = store_frame.index; store_frame["instrument"] = "SOXX_ISSUER_FLOW"
    store_manifest = build_qlib_store_from_frame(store_frame.reset_index(drop=True), provider_uri=provider, freq="day",
        as_of=plan["as_of"], field_map={name: name for name in fields}, source="frozen iShares historical workbook; conditional lag diagnostic")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    native = D.features(["SOXX_ISSUER_FLOW"], ["$" + name for name in fields], freq="day").xs("SOXX_ISSUER_FLOW", level="instrument")
    if not native.index.equals(calendar):
        raise ValueError("Qlib flow store changed native calendar")
    for name in fields:
        if not np.allclose(native["$" + name], state[name], rtol=2e-7, atol=1e-8, equal_nan=True):
            raise ValueError("Qlib flow readback mismatch")
        frame[name] = native["$" + name].reindex(frame.index)
    future_prices = np.r_[frame.soxx_open.to_numpy(float), float(frame.soxx_exit_open.iloc[-1])]
    future_dates = pd.DatetimeIndex([*frame.trade_date, frame.exit_date.iloc[-1]])
    labels = np.full((len(frame), 2), np.nan)
    maturity = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for number in range(len(frame)):
        end = number + plan["horizon_sessions"]
        if end >= len(future_prices):
            continue
        returns = future_prices[number + 1:end + 1] / future_prices[number:end] - 1
        labels[number] = [future_prices[end] / future_prices[number] - 1,
            np.log(max(float(np.mean(np.minimum(returns, 0) ** 2)), plan["variance_numerical_floor"]))]
        maturity.iloc[number] = future_dates[end] + pd.Timedelta(hours=9, minutes=49)
    label = pd.DataFrame(labels, index=frame.index, columns=plan["targets"])
    frame["label_available_at"] = maturity
    frame["feature_available_at"] = pd.concat([frame.feature_available_at,
        state.assumed_available_at.reindex(frame.index)], axis=1).max(axis=1)
    eligible = frame.known.eq(1)
    if not frame.loc[eligible, "feature_available_at"].le(frame.loc[eligible, "decision_at"]).all():
        raise ValueError("assumed source arrival exceeds decision")
    columns = plan["base_features"] + plan["flow_features"]
    grouped = pd.concat({"feature": frame[columns], "label": label,
        "meta": frame[["feature_available_at", "decision_at", "label_available_at"]]}, axis=1).loc[eligible]
    grouped.index = pd.MultiIndex.from_arrays([grouped.index, ["SOXX"] * len(grouped)], names=["datetime", "instrument"])
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="conditional_issuer_flow_information", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=digest, maximum_model_fits=6, maximum_backtests=0, historical_vintage_certified=False)
        recorder.save_objects(artifact_path="sources", workbook=raw, history=history, audit=source_audit,
            download_metadata=json.loads((root / plan["issuer_metadata"]).read_text()),
            data_code=(root / "src/data/etf_share_history.py").read_text(), workflow_code=Path(__file__).read_text())
        recorder.save_objects(artifact_path="inputs", grouped=grouped, plan=plan, information=state,
            store_manifest=store_manifest, price_audit=price_audit, model_code=(root / "src/model/capacity_forecast.py").read_text())
    comparisons, increments, audits, predictions, paired = [], [], [], {}, {}
    for fold in plan["folds"]:
        bounds = frame.index[eligible & frame.signal_date.between(fold["start"], fold["end"])]
        dataset = DatasetH(handler=DataHandlerLP.from_df(grouped), segments={
            "train": (None, pd.Timestamp(fold["cutoff"]) - pd.Timedelta(nanoseconds=1)), "test": (bounds[0], bounds[-1])})
        losses = {}
        for family, features in [("price", plan["base_features"]), ("price_plus_flow", columns)]:
            if len(audits) >= plan["maximum_model_fits"]:
                raise ValueError("fixed fit budget exhausted")
            model = CapacityForecast(features, plan["targets"], fold["cutoff"], plan["ridge_alpha"]).fit(dataset)
            prediction = model.predict(dataset); truth = dataset.prepare("test", col_set="label")
            valid = evaluation_mask(truth, dataset.prepare("test", col_set="meta"), fold, plan["as_of"])
            error = prediction.loc[valid] - truth.loc[valid]
            losses[family] = ((error.to_numpy() / model.y_scaler.scale_) ** 2).mean(axis=1)
            for target in plan["targets"]:
                comparisons.append({"fold": fold["name"], "family": family, "target": target, "observations": int(valid.sum()),
                    "rmse": float(np.sqrt(np.mean(error[target] ** 2))),
                    "rank_ic": float(prediction.loc[valid, target].corr(truth.loc[valid, target], method="spearman"))})
            predictions[(fold["name"], family)] = {"prediction": prediction, "truth": truth, "label_mature": valid}
            audits.append({"fold": fold["name"], "family": family, **model.audit})
            recorder.save_objects(artifact_path="models", **{fold["name"] + "_" + family: model})
        paired[fold["name"]] = losses["price"] - losses["price_plus_flow"]
        increments.append({"fold": fold["name"], **_bootstrap(paired[fold["name"]], plan)})
    pooled = _bootstrap(np.concatenate([values for name, values in paired.items() if name.startswith("development")]), plan)
    increments.append({"fold": "pooled_development", **pooled})
    passed = all(row["mean_joint_normalized_loss_improvement"] > 0 for row in increments if row["fold"].startswith("development")) and pooled["lower_bound"] > 0
    summary = {"experiment": plan["experiment"], "plan_sha256": digest,
        "decision": "conditional_information_pass_needs_historical_vintages" if passed else "not_promoted_flow_information_gate_failed",
        "conditional_information_gate_pass": bool(passed), "promotion_eligible": False, "objective_achieved": False,
        "goal_remains_active": True, "new_model_fits": len(audits), "backtests": 0, "training_audit": audits,
        "information_increment": increments, "source_audit": source_audit, "historical_windows_reused": True,
        "common_source_unknown_rows_excluded": int((~eligible).sum()), "terminal_unknown_labels": int(label.isna().any(axis=1).sum()),
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id), "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id), "provider_uri": str(provider),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["sources/*", "inputs/*", "models/*", "analysis/predictions", "analysis/paired_losses"]}}
    recorder.save_objects(artifact_path="analysis", predictions=predictions, paired_losses=paired, summary=summary)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    recorder.log_metrics(completed_model_fits=len(audits), registered_artifact_count=len(list(artifacts.rglob("*.pkl"))))
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size} for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    pd.DataFrame(comparisons).to_csv(card / "forecast_metrics.csv", index=False)
    pd.DataFrame(increments).to_csv(card / "information_increment.csv", index=False)
    atomic_json(source_audit, card / "source_quality.json")
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": str(provider),
        "recorder_uri": str(tracking), "as_of": plan["as_of"], "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card / "summary.json")
    text = f"# ETF份额信息增量诊断\n\n完成{len(audits)}次固定线性拟合、0次组合回测。决定：`{summary['decision']}`。\n\n"
    text += "|窗口|联合预测损失改善|区块下界|区块上界|\n|---|---:|---:|---:|\n"
    for row in increments:
        text += f"|{row['fold']}|{row['mean_joint_normalized_loss_improvement']:.5f}|{row['lower_bound']:.5f}|{row['upper_bound']:.5f}|\n"
    text += "\n正数代表误差下降。目标是5交易日SOXX收益与下行方差log，等权训练内标准化；相同样本、价格控制、严格成熟标签。历史多次研究后的窗口只作诊断，区块区间不消除选择偏差。\n\n"
    text += f"来源为冻结iShares官方历史表：{source_audit['source_start']}至{source_audit['source_end']}，{len(history)}行含休市日；原始份额缺失{source_audit['missing_shares_rows']}行。仅通过Qlib真实交易日历构造特征，缺值不填零。\n\n"
    text += "份额变化是净申赎代理，不是投资者成交量或完整资金流账本；历史份额可能按拆股重述，不再次调整，不声称原始流通股数。采用观测后两个真实交易日09:31可用的假设；原始发布日期、首次接收和后续修订均未认证，因此不能宣称PIT合格、实盘可用或稳定超额。SOXL份额未合成。\n\n"
    text += "信息门失败则结束这组固定特征/模型/周期，不改符号、不调窗口；通过则仍须取得真实历史版本或冻结前向来源，再做IBKR Tiered十万美元完整账户检验。本轮不改生产SIM，目标保持active。\n"
    (card / "conclusion.md").write_text(text)
    return summary
