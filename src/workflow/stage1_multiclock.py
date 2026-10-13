"""Frozen multi-clock prices/forecasts, full risk controller, actual Tiered books."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.backtest.stage1_multiclock import replay_multiclock, native_multiclock
from src.data.native_execution_marks import build_published_minute_view
from src.data.qlib_native import QlibProviderConfig, load_qlib_feature_frame, build_qlib_store_from_frame
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.workflow.minute_action_value import _source
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_multiclock(*, root, card, plan):
    import qlib
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    qlib.init(provider_uri=str(root / "data/qlib/full_path_tiered_allocation_20261013"), region="us", kernels=1)
    if _sha(root / "production/stage1_futu_sim_v1/runtime_release.json") != plan["release_sha256"]: raise ValueError("blocked: source release changed")
    frozen, frozen_audit = _source(root, plan["forecast_source"], ["analysis/predictions", "inputs/feature_snapshots", "models/fitted_models"])
    source, source_audit = _source(root, plan["fee_source"], ["inputs/market", "inputs/targets", "inputs/controller"])
    strategy, strategy_audit = _source(root, plan["strategy_source"], ["inputs/plan", "controller/source_recovery_selection"])
    frame = source["inputs/market"].join(source["inputs/targets"]["frozen_current"].set_axis(source["inputs/market"].index))
    predictions = {family: frozen["analysis/predictions"].loc[frozen["analysis/predictions"].family.eq(family)].reset_index()
                   for family in ("first_hour", "full_day", "full_day_context")}
    neutral, audits = [], []
    for year in range(2023, 2027):
        model = frozen["models/fitted_models"][f"first_hour_{year}"]
        original = predictions["first_hour"].loc[predictions["first_hour"].year.eq(year)].set_index(["datetime", "instrument"])
        features = frozen["inputs/feature_snapshots"].reindex(original.index)[model.features].copy()
        actual = model.pipeline.predict(features)
        if not np.allclose(actual, original[["pred_soxl", "pred_soxx"]], atol=1e-12, rtol=0):
            raise ValueError("blocked: frozen forecast readback changed")
        muted = []
        for i, name in enumerate(model.features):
            if not name.startswith("parent_") and name not in ("slot", "overnight_label"):
                features[name] = model.pipeline.named_steps["standardscaler"].mean_[i]; muted.append(name)
        data = pd.concat({"feature": features}, axis=1)
        dataset = DatasetH(handler=DataHandlerLP.from_df(data), segments={"test": (None, None)})
        predicted = model.predict(dataset).join(original[["entry_at", "label_available_at"]])
        neutral.append(predicted.reset_index())
        audits.append({"year": year, "rows": len(features), "muted_numerical_features": muted,
            "parent_slot_and_overnight_inputs_retained": True, "fits": 0})
    predictions["neutral_first_hour"] = pd.concat(neutral, ignore_index=True)
    provider = root / "data/qlib" / plan["experiment"]
    source_manifest = root / plan["source_manifest"]
    base = build_published_minute_view(source_manifest=source_manifest, source_sha256=plan["source_id"],
        provider_uri=provider / "accounting", start=plan["start"], as_of=plan["as_of"])
    adjusted = load_qlib_feature_frame(QlibProviderConfig(provider / "accounting", freq="15min"),
        instruments=["SOXL", "SOXX"], fields=["$open", "$high", "$low", "$close", "$volume"], start=plan["start"], end=plan["as_of"] + " 16:00:00")
    receipt = json.loads(source_manifest.read_text()); raw_parts = []
    for symbol in ("SOXL", "SOXX"):
        entry = next(e for e in receipt["entries"] if e["symbol"] == symbol)
        path = source_manifest.parent / entry["path"]
        if _sha(path) != entry["sha256"]: raise ValueError("blocked: raw fee quote source changed")
        part = pd.read_parquet(path, columns=["time_key", "close"])
        part["datetime"] = pd.to_datetime(part.time_key); part["instrument"] = symbol
        raw_parts.append(part.set_index(["instrument", "datetime"])["close"])
    raw = pd.concat(raw_parts).reindex(adjusted.index)
    if not np.isfinite(raw).all() or not raw.gt(0).all(): raise ValueError("needs_data: raw arrival quotes missing")
    adjusted["$raw_execution_price"] = raw
    manifest = build_qlib_store_from_frame(adjusted.reset_index(), provider_uri=provider / "execution", freq="15min", as_of=plan["as_of"],
        field_map={name: name for name in adjusted}, source="frozen NAS published accounting closes plus actual NONE close share prices")
    quotes = load_qlib_feature_frame(QlibProviderConfig(provider / "execution", freq="15min"), instruments=["SOXL", "SOXX"],
        fields=["$close", "$raw_execution_price"], start=plan["start"], end=plan["end"] + " 16:00:00")
    for family, prediction in predictions.items():
        prediction = prediction.loc[prediction.entry_at.between(plan["start"], plan["end"] + " 16:00:00")].copy()
        if prediction.entry_at.duplicated().any() or not prediction.datetime.add(pd.Timedelta(minutes=1)).lt(prediction.entry_at).all():
            raise ValueError("blocked: forecast chronology or uniqueness fails")
        predictions[family] = prediction
    return {"quotes": quotes, "daily": frame, "predictions": predictions, "controller": source["inputs/controller"],
        "recovery_spec": strategy["inputs/plan"]["recovery_spec"], "action_spec": strategy["inputs/plan"]["action_spec"],
        "selection": strategy["controller/source_recovery_selection"], "provider_uri": provider / "execution",
        "audit": {"frozen_forecasts": frozen_audit, "current_source": source_audit, "strategy": strategy_audit,
            "neutralization": audits, "accounting": base, "fee_marks": manifest, "fits": 0}, "plan_sha256": _sha(card / "plan.json")}


def daily_metrics(daily):
    from qlib.contrib.evaluate import risk_analysis
    from src.analysis.integrated_allocation import _cvar
    wealth = (1 + daily).cumprod()
    analysis = risk_analysis(daily, N=252, freq=None, mode="product")
    return {"sessions": len(daily), "return_pct": float(100 * (wealth.iloc[-1] - 1)),
        "cagr_pct": float(100 * analysis.loc["annualized_return", "risk"]),
        "max_drawdown_pct": float(100 * (wealth / wealth.cummax().clip(lower=1) - 1).min()),
        "cvar5_pct": float(100 * _cvar(daily.to_numpy())), "sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))}


def paired_log_interval(candidate, reference):
    if not candidate.index.equals(reference.index): raise ValueError("blocked: unpaired calendars")
    values = np.log1p(candidate.to_numpy()) - np.log1p(reference.to_numpy())
    rng = np.random.default_rng(20261013); indices = rng.integers(len(values), size=2000); totals = values[indices].copy()
    for _ in range(1, len(values)):
        restart = rng.random(len(indices)) < .05
        indices = (indices + 1) % len(values); indices[restart] = rng.integers(len(values), size=restart.sum())
        totals += values[indices]
    return {"annualized_log_excess": float(252 * values.mean()),
        "annualized_lower_bound": float(252 * np.quantile(totals / len(values), .025)),
        "annualized_upper_bound": float(252 * np.quantile(totals / len(values), .975)),
        "block_mean_sessions": 20, "resamples": 2000, "selection_and_refit_uncertainty_not_covered": True}


def run_multiclock(*, root, card, plan, prepared):
    import qlib
    import yaml
    from qlib.workflow import R
    if (card / "summary.json").exists(): return json.loads((card / "summary.json").read_text())
    if prepared["plan_sha256"] != _sha(card / "plan.json"): raise ValueError("blocked: prepared plan changed")
    qlib.init(provider_uri=str(prepared["provider_uri"]), region="us", kernels=1)
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="frozen_multiclock_full_controller", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=prepared["plan_sha256"], new_model_fits=0, maximum_native_cases=plan["maximum_native_cases"])
        recorder.save_objects(artifact_path="inputs", plan=plan, quotes=prepared["quotes"], daily=prepared["daily"], predictions=prepared["predictions"], audit=prepared["audit"], controller=prepared["controller"])
        recorder.save_objects(artifact_path="sources", backtest=(root / "src/backtest/stage1_multiclock.py").read_text(), workflow=(root / "src/workflow/stage1_multiclock.py").read_text())
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=root, **prepared["recovery_spec"])
    action_spec = Stage1OpenActionValueSpec(root=root, **prepared["action_spec"])
    books = {}
    for slip in plan["slippage_bps"]:
        for family in plan["families"]:
            replay = replay_multiclock(prepared["quotes"], prepared["daily"], prepared["predictions"].get(family), family=family,
                plan=plan, controller=prepared["controller"], recovery_spec=recovery_spec, action_spec=action_spec, selection=prepared["selection"], slip=slip)
            book = native_multiclock(replay, provider_uri=prepared["provider_uri"], plan=plan, slip=slip)
            books[(family, slip)] = book
            recorder.save_objects(artifact_path="analysis/native", **{f"{family}_{slip}": book})
            recorder.log_metrics(completed_native_cases=len(books))
            atomic_json({"recorder_id": recorder.id, "native_cases": len(books), "last_case": [family, slip]}, card / "progress.json")
            print("native",len(books),family,slip,flush=True)
    rows, annual, windows = [], [], []
    for (family, slip), book in books.items():
        daily, benchmark = book["daily_net"], books[("soxx", slip)]["daily_net"]
        metrics, bench = daily_metrics(daily), daily_metrics(benchmark)
        ledger = book["ledger"]; wealth = ledger.nav / plan["initial_cash"]
        rows.append({"family": family, "slippage_bps": slip, **metrics, "benchmark_return_pct": bench["return_pct"],
            "relative_wealth": (1 + metrics["return_pct"] / 100) / (1 + bench["return_pct"] / 100),
            "benchmark_cvar5_pct": bench["cvar5_pct"], "monitored_drawdown_pct": float(100 * (wealth / wealth.cummax().clip(lower=1) - 1).min()),
            "broker_fees_usd": float(ledger.broker_fees.sum()), "slippage_usd": float(ledger.slippage_cost.sum()),
            "orders": len(book["orders"]), "intraday_actions": int(ledger.action.sum()), "mandatory_repairs": int(ledger.mandatory_repair.sum()),
            "average_cash_pct": float(100 * ledger.cash_weight.mean()), "maximum_soxl_weight_pct": float(100 * ledger.soxl_weight.max())})
        for year, data in daily.groupby(daily.index.year):
            row = {"family": family, "slippage_bps": slip, "year": year, **daily_metrics(data)}
            row["benchmark_return_pct"] = daily_metrics(benchmark.reindex(data.index))["return_pct"]
            annual.append(row)
        rolling = (np.log1p(daily) - np.log1p(benchmark)).rolling(126).sum().dropna()
        windows.append({"family": family, "slippage_bps": slip, "positive_126d_fraction": float(rolling.gt(1e-8).mean())})
    table, annual, windows = pd.DataFrame(rows), pd.DataFrame(annual), pd.DataFrame(windows)
    selected = table.loc[table.family.eq(plan["selected_primary"])].set_index("slippage_bps")
    neutral = table.loc[table.family.eq("neutral_first_hour")].set_index("slippage_bps")
    intervals = [{"slippage_bps": slip, "against": against, **paired_log_interval(books[(plan["selected_primary"], slip)]["daily_net"], books[(against, slip)]["daily_net"])}
                 for slip in plan["slippage_bps"] for against in ("soxx", "neutral_first_hour")]
    gates = {"net_soxx_excess_all_costs": bool(selected.relative_wealth.gt(1 + 1e-8).all()),
        "monitored_drawdown_within_30": bool(selected.monitored_drawdown_pct.ge(-30).all()),
        "cvar_no_worse_than_soxx": bool(selected.cvar5_pct.ge(selected.benchmark_cvar5_pct).all()),
        "minute_increment_all_costs": bool((selected.return_pct - neutral.return_pct).gt(0).all()),
        "rolling_win_70pct": bool(windows.loc[windows.family.eq(plan["selected_primary"]), "positive_126d_fraction"].ge(.70).all()),
        "positive_years_at_least_three_all_costs": all(int((annual.loc[annual.family.eq(plan["selected_primary"]) & annual.slippage_bps.eq(slip), "return_pct"].to_numpy() > annual.loc[annual.family.eq(plan["selected_primary"]) & annual.slippage_bps.eq(slip), "benchmark_return_pct"].to_numpy() + 1e-6).sum()) >= 3 for slip in plan["slippage_bps"]),
        "paired_bootstrap_excess_and_increment": all(row["annualized_lower_bound"] > 0 for row in intervals),
        "qualified_independent_forward": False}
    historical_pass = all(value for key, value in gates.items() if key != "qualified_independent_forward")
    summary = {"experiment": plan["experiment"], "plan_sha256": prepared["plan_sha256"], "decision": "needs_independent_forward" if historical_pass else "not_promoted_multiclock_full_stage1_gates_failed",
        "objective_achieved": False, "goal_remains_active": True, "new_model_fits": 0, "native_cases": len(books), "gates": gates,
        "primary_family": plan["selected_primary"], "family_reselection": False, "start": plan["start"], "end": plan["end"],
        "as_of": plan["as_of"], "provider_uri": str(prepared["provider_uri"]), "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking), "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "current_release_changed": False, "production_enabled": False, "forward_qualified_sessions": 0,
        "compact_metrics": rows, "clock_limit": plan["information_clock"], "compatibility": plan["parent_compatibility"],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/native/*"]}}
    recorder.save_objects(artifact_path="analysis", comparison=table, annual=annual, windows=windows, intervals=intervals, summary=summary)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    for name, value in [("backtest_benchmark_comparison", table), ("regime_metrics", annual), ("window_metrics", windows)]: value.to_csv(card / (name + ".csv"), index=False)
    pd.DataFrame(intervals).to_csv(card / "uncertainty.csv", index=False)
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)), "sha256": _sha(p), "bytes": p.stat().st_size} for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": str(prepared["provider_uri"]), "recorder_uri": str(tracking), "freq": "15min", "as_of": plan["as_of"], "artifact_policy": summary["artifact_policy"]},sort_keys=False))
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card, recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card / "summary.json")
    text = "# 多时点完整Stage1费用检验\n\n冻结分钟模型与预测复用，新增拟合0次；18个实际Tiered/0-2-5bp账户通过Qlib逐笔费用、现金、股数及净值对账。原首小时主候选保留，其他两组不事后重选。\n\n"
    text += "|模型|滑点bp|净收益%|15分钟监测回撤%|相对SOXX财富|\n|---|---:|---:|---:|---:|\n"
    for row in table.itertuples():text += f"|{row.family}|{row.slippage_bps}|{row.return_pct:.2f}|{row.monitored_drawdown_pct:.2f}|{row.relative_wealth:.4f}|\n"
    text += f"\n门槛：{json.dumps(gates,ensure_ascii=False)}。原开盘动作、日级风险状态与恢复许可保留，盘中经济选择及中性对照共用机会集。原始生产时钟与本轮延迟close时钟不同，不能把跨时钟收益差称作alpha。\n\n"
    text += "历史仍为复用诊断，真实首收/成交/股息资格未取得；逐笔费用为当前费表投影，滑点为不利执行准备金。监测回撤使用15分钟报价，不是每秒保证。结论不晋升，目标保持active；本固定组合结束后不修改源模型、窗口、阈值或重选最佳特征组。\n"
    (card / "conclusion.md").write_text(text)
    return summary
