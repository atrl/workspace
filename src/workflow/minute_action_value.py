"""Notebook-first bounded cost-aware minute action-value experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import annual_paired_metrics, paired_metrics, stationary_bootstrap_excess
from src.backtest.minute_action_value import next_reset_advantage, replay_minute_value
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_action_marks import build_action_marks
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder, materialize_nas_evidence, materialize_nas_snapshot
from src.model.minute_action_value import MinuteActionValue
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(root, source, names):
    card = root / source["card"]
    binding = json.loads((card / "nas_binding.json").read_text())
    if binding["evidence"]["sha256"] != source["archive"]:
        raise ValueError("blocked: source NAS identity differs from frozen plan")
    summary = json.loads((card / "summary.json").read_text())
    if not Path(summary["recorder_local_dir"]).is_dir():
        materialize_nas_evidence(card / "nas_binding.json", card=card)
    return load_verified_recorder_objects(card, expected_recorder_id=source["recorder_id"], artifacts=names,
                                         expected_manifest_sha256=source["inventory_sha256"])


def prepare_minute_value(*, root, card, plan):
    import qlib
    qlib.init(provider_uri=str(root / plan["raw_provider"]), region="us", kernels=1)
    if _sha(root / "production/stage1_futu_sim_v1/runtime_release.json") != plan["release_sha256"]:
        raise ValueError("blocked: active release changed")
    names = ["inputs/market", "inputs/targets", "inputs/controller"]
    names += [f"analysis/native/tiered_{slip}_frozen_current" for slip in plan["slippage_bps"]]
    previous, previous_audit = _source(root, plan["fee_source"], names)
    strategy, strategy_audit = _source(root, plan["strategy_source"], ["inputs/plan", "controller/source_recovery_selection"])
    source = root / plan["source_manifest"]
    if not source.exists():
        materialize_nas_snapshot(root / plan["snapshot_binding"], workspace_root=root, include_derived=False)
    provider = root / "data/qlib" / plan["experiment"]
    marks, minute, mark_audit = build_action_marks(source_manifest=source, source_sha256=plan["source_id"],
        raw_provider=root / plan["raw_provider"], raw_manifest_sha256=plan["raw_manifest_sha256"], provider_uri=provider,
        start=plan["start"], as_of=plan["as_of"])
    frame = previous["inputs/market"].copy()
    frame = frame.join(previous["inputs/targets"]["frozen_current"].set_axis(frame.index))
    frame = frame.join(minute, on="trade_date")
    frame["minute_ready"] = frame.feature_available_at.notna()
    for symbol in ("SOXL", "SOXX"):
        asset = marks.xs(symbol, level="instrument")
        for slot, dates in [("open", frame.trade_date + pd.Timedelta(hours=9, minutes=30)),
                            ("action", frame.trade_date + pd.Timedelta(hours=10, minutes=32)),
                            ("exit_open", frame.exit_date + pd.Timedelta(hours=9, minutes=30))]:
            for field, prefix in [("$execution_price", ""), ("$raw_execution_price", "raw_")]:
                frame[f"{prefix}{symbol.lower()}_{slot}"] = asset[field].reindex(pd.DatetimeIndex(dates)).to_numpy(float)
    required = [f"{prefix}{symbol}_{slot}" for prefix in ("", "raw_") for symbol in ("soxl", "soxx")
                for slot in ("open", "action", "exit_open")]
    if not np.isfinite(frame[required]).all().all() or not frame[required].gt(0).all().all():
        raise ValueError("needs_data: incomplete native execution marks")
    if not frame.feature_available_at.dropna().le(frame.trade_date.loc[frame.minute_ready] + pd.Timedelta(hours=10, minutes=31)).all():
        raise ValueError("blocked: minute information not yet available at decision")
    return {"frame": frame, "controller": previous["inputs/controller"], "provider_uri": provider,
        "recovery_spec": strategy["inputs/plan"]["recovery_spec"], "action_spec": strategy["inputs/plan"]["action_spec"],
        "selection": strategy["controller/source_recovery_selection"],
        "parent_books": {slip: previous[f"analysis/native/tiered_{slip}_frozen_current"] for slip in plan["slippage_bps"]},
        "audit": {"fee_source": previous_audit, "strategy_source": strategy_audit, "marks": mark_audit,
                  "minute_ready_sessions": int(frame.minute_ready.sum()), "sessions": len(frame)},
        "plan_sha256": _sha(card / "plan.json")}


def _dataset(teacher, features, cutoff):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    data = pd.concat({"feature": teacher[features], "label": teacher[["net_advantage"]],
                      "meta": teacher[["label_available_at"]]}, axis=1)
    data.index = pd.MultiIndex.from_arrays([teacher.index, ["SOXL_SOXX_CASH"] * len(teacher)], names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(data), segments={"train": (None, pd.Timestamp(cutoff) - pd.Timedelta(nanoseconds=1)),
                                                               "test": (pd.Timestamp(cutoff), None)})


def run_minute_value(*, root, card, plan, prepared):
    import qlib
    import yaml
    from qlib.workflow import R

    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        summary = json.loads((card / "summary.json").read_text())
        if summary["plan_sha256"] != digest:
            raise ValueError("blocked: finalized experiment plan changed")
        return summary
    if prepared["plan_sha256"] != digest:
        raise ValueError("blocked: prepared data does not match frozen plan")
    qlib.init(provider_uri=str(prepared["provider_uri"]), region="us", kernels=1)
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="minute_net_action_value", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=digest, maximum_model_fits=plan["maximum_model_fits"], maximum_native_cases=plan["maximum_native_cases"])
        recorder.save_objects(artifact_path="inputs", frame=prepared["frame"], controller=prepared["controller"],
            plan=plan, audit=prepared["audit"], selection=prepared["selection"])
        recorder.save_objects(artifact_path="sources", **{name.replace("/", "__"): (root / name).read_text() for name in plan["code_sources"]})
        recorder.save_objects(artifact_path="sources", industry_selection=(card / "industry_selection.md").read_text())
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=root, **prepared["recovery_spec"])
    action_spec = Stage1OpenActionValueSpec(root=root, **prepared["action_spec"])
    kwargs = {"controller": prepared["controller"], "tariff": plan["tariff"], "recovery_spec": recovery_spec,
        "action_spec": action_spec, "selection": prepared["selection"], "initial_cash": plan["initial_cash"], "plan": plan}
    reference = replay_minute_value(prepared["frame"], slippage_bps=plan["primary_slippage_bps"], policy="current", **kwargs)
    teacher_rows = []
    for number, context in enumerate(reference["contexts"][:-1]):
        if not context["eligible"]:
            continue
        row = prepared["frame"].iloc[number]
        teacher_rows.append({"decision_at": row.trade_date + pd.Timedelta(hours=10, minutes=31), **context["features"],
            "net_advantage": next_reset_advantage(context, row, reference["contexts"][number + 1]["shared_reset_weights"]),
            "label_available_at": row.exit_date + pd.Timedelta(hours=9, minutes=31)})
    if not teacher_rows:
        raise ValueError("needs_data: no eligible action-value labels")
    teacher = pd.DataFrame(teacher_rows).set_index("decision_at").sort_index()
    minute_features = [column for column in teacher if column.startswith("minute_")]
    state_features = [column for column in teacher if column not in [*minute_features, "net_advantage", "label_available_at"]]
    feature_sets = {"state": state_features, "minute": state_features + minute_features}
    models, training = {family: {} for family in feature_sets}, []
    for year in plan["fit_years"]:
        for family, features in feature_sets.items():
            if len(training) >= plan["maximum_model_fits"]:
                raise ValueError("blocked: fit budget exhausted")
            cutoff = f"{year}-01-01"
            model = MinuteActionValue(features=features, cutoff=cutoff, alpha=plan["ridge_alpha"],
                minimum_observations=plan["minimum_training_observations"], hac_days=plan["hac_calendar_days"])
            model.fit(_dataset(teacher, features, cutoff))
            models[family][year] = model
            training.append({"family": family, "year": year, **model.audit})
            recorder.save_objects(artifact_path="models", **{f"{family}_{year}": model})
            recorder.log_metrics(completed_model_fits=len(training))
            print(f"fit {len(training)}/{plan['maximum_model_fits']} {family} {year} n={model.audit['observations']}", flush=True)
    recorder.save_objects(artifact_path="analysis", teacher=teacher, training_audit=pd.DataFrame(training))
    books, control_audits = {}, []
    for slip in plan["slippage_bps"]:
        for family in plan["policies"]:
            if len(books) >= plan["maximum_native_cases"]:
                raise ValueError("blocked: native-case budget exhausted")
            replay = reference if slip == plan["primary_slippage_bps"] and family == "current" else replay_minute_value(
                prepared["frame"], slippage_bps=slip, policy=family, value_models=models.get(family), **kwargs)
            book = native_tiered_stage1(prepared["frame"], replay, provider_uri=prepared["provider_uri"], tariff=plan["tariff"],
                slippage_bps=slip, initial_cash=plan["initial_cash"], freq="1min", opening_offset=pd.Timedelta(hours=9, minutes=30))
            book["predictions"] = replay["predictions"]
            books[(family, slip)] = book
            if family == "current":
                parent = prepared["parent_books"][slip]
                error = float(np.max(abs(book["authority"].strategy_return.to_numpy() - parent["authority"].strategy_return.to_numpy())))
                if error > 1e-7 or len(book["orders"]) != len(parent["orders"]):
                    raise ValueError("blocked: physical-clock control differs from frozen parent account")
                hold_equivalent = not book["authority"].recovery_action.any()
                if not hold_equivalent:
                    raise ValueError("blocked: current recovery no longer equals the intraday-HOLD control")
                control_audits.append({"slippage_bps": slip, "maximum_parent_return_error": error,
                    "native_orders": len(book["orders"]), "current_and_intraday_HOLD_are_identical": hold_equivalent})
            recorder.save_objects(artifact_path="analysis/native", **{f"{family}_{slip}": book})
            recorder.log_metrics(completed_native_cases=len(books))
            print(f"native {len(books)}/{plan['maximum_native_cases']} {family} slip={slip}", flush=True)
    rows, annual, uncertainty, windows = [], [], [], []
    for (family, slip), book in books.items():
        path, benchmark = book["authority"], books[("soxx", slip)]["authority"]
        pair = paired_metrics(path, benchmark)
        rows.append({"family": family, "slippage_bps": slip, **pair,
            "orders": len(book["orders"]), "recovery_actions": int(path.recovery_action.sum()),
            "broker_and_external_fees_usd": float(path.broker_and_external_fees_usd.sum()),
            "slippage_reserve_usd": float(path.slippage_reserve_usd.sum()),
            "average_cash_pct": float(100 * path.ending_cash_weight.mean())})
        yearly = annual_paired_metrics(path, benchmark)
        yearly["family"], yearly["slippage_bps"] = family, slip
        annual.append(yearly)
        logs = np.log1p(path.strategy_return.to_numpy()) - np.log1p(benchmark.strategy_return.to_numpy())
        rolling = pd.Series(logs).rolling(plan["rolling_sessions"]).sum().dropna()
        windows.append({"family": family, "slippage_bps": slip, "rolling_sessions": plan["rolling_sessions"],
            "windows": len(rolling), "positive_fraction": float(rolling.gt(1e-8).mean())})
        if family in ("state", "minute"):
            for against in ("soxx", "current", "state"):
                if against == family:
                    continue
                stats = stationary_bootstrap_excess(path, books[(against, slip)]["authority"], mean_block=20,
                    resamples=plan["bootstrap_resamples"], alpha=plan["bootstrap_alpha"], seed=plan["bootstrap_seed"])
                uncertainty.append({"family": family, "slippage_bps": slip, "against": against, **stats})
    table, annual = pd.DataFrame(rows), pd.concat(annual, ignore_index=True)
    windows, uncertainty = pd.DataFrame(windows), pd.DataFrame(uncertainty)
    minute = table.loc[table.family.eq("minute")]
    incremental = minute.set_index("slippage_bps").return_pct - table.loc[table.family.eq("state")].set_index("slippage_bps").return_pct
    gates = {"net_soxx_excess_all_cost_scenarios": bool(minute.relative_wealth.gt(1 + 1e-8).all()),
        "max_drawdown_within_30pct_all_scenarios": bool(minute.max_drawdown_pct.ge(-plan["maximum_drawdown_pct"]).all()),
        "cvar_no_worse_than_soxx": bool(minute.cvar5_pct.ge(minute.benchmark_cvar5_pct).all()),
        "minute_increment_positive_all_scenarios": bool(incremental.gt(0).all()),
        "annual_at_least_three_positive": all(int(annual.loc[annual.family.eq("minute") & annual.slippage_bps.eq(slip), "relative_wealth"].gt(1 + 1e-8).sum()) >= 3 for slip in plan["slippage_bps"]),
        "rolling_positive_at_least_70pct": bool(windows.loc[windows.family.eq("minute"), "positive_fraction"].ge(.70).all()),
        "bootstrap_soxx_and_state_lower_bound_positive": bool(uncertainty.loc[uncertainty.family.eq("minute") & uncertainty.against.isin(["soxx", "state"]), "annualized_lower_bound"].gt(0).all()),
        "independent_forward_and_real_execution_qualified": False}
    historical_pass = all(value for key, value in gates.items() if key != "independent_forward_and_real_execution_qualified")
    summary = {"experiment": plan["experiment"], "plan_sha256": digest, "decision": "needs_independent_forward" if historical_pass else "not_promoted_minute_net_value_gates_failed",
        "objective_achieved": False, "production_enabled": False, "current_release_changed": False,
        "outer_is_reused_history": True, "forward_qualified_sessions": 0, "gates": gates,
        "new_model_fits": len(training), "native_cases": len(books), "sessions": len(prepared["frame"]),
        "as_of": plan["as_of"], "initial_cash": plan["initial_cash"], "provider_uri": str(prepared["provider_uri"]),
        "region": "us", "market": "all", "data_authority": "yanxuan_nas", "recorder_id": recorder.id,
        "experiment_id": str(recorder.experiment_id), "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "control_readback": control_audits, "teacher_eligible_rows": len(teacher), "training_audit": training,
        "minute_increment_return_pct_points": incremental.to_dict(), "compact_metrics": rows,
        "scientific_limits": ["reused historical windows; no fresh holdout or certified first receipt",
            "fixed-reference counterfactual labels, not full future-controller tree labels",
            "calendar HAC mean error is a heuristic, not calibrated prediction coverage",
            "known-price roundtrip preview approximates cost-scenario score correction; actual orders charged exactly under tariff",
            "current tariff projection, assumed venue/slippage, inherited affine NAV units; actual broker fills/dividends unqualified"],
        "route_state": "end_fixed_minute_net_value_route_no_further_search",
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "models/*", "analysis/teacher", "analysis/native/*", "sources/*"]}}
    recorder.save_objects(artifact_path="analysis", comparison=table, annual=annual, windows=windows,
        uncertainty=uncertainty, summary=summary, control_readback=control_audits)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    if _sha(root / "production/stage1_futu_sim_v1/runtime_release.json") != plan["release_sha256"]:
        raise ValueError("blocked: active release changed during research")
    table.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    annual.to_csv(card / "regime_metrics.csv", index=False)
    windows.to_csv(card / "window_metrics.csv", index=False)
    uncertainty.to_csv(card / "uncertainty.csv", index=False)
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)), "sha256": _sha(p), "bytes": p.stat().st_size}
        for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": summary["provider_uri"],
        "recorder_uri": str(tracking), "region": "us", "freq": "1min", "calendar_role": "sparse_execution_events",
        "as_of": plan["as_of"], "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card / "summary.json")
    conclusion = render_conclusion(summary, table)
    (card / "conclusion.md").write_text(conclusion)
    atomic_json({"selected": None, "decision": summary["decision"], "candidate": "minute", "anchor": plan["source_release"],
        "selection_uses_outer": False, "no_deployment": True}, card / "selected_model.json")
    return summary


def render_conclusion(summary, table):
    text = "# 分钟净执行价值实验结论\n\n已执行固定的 6 次 Qlib 年度滚动训练和 12 个独立扣费账户；数据固定到 2026-10-09，覆盖 944 个持有区间。锚点为原模拟盘 f696/b472，未改模拟盘。\n\n"
    text += "已知信息截至 10:30，假定 60 秒后可用，10:31 决策并延迟到 10:32 实际分钟开盘成交。标签覆盖到次日 09:30，并对 HOLD/调整分支计入共同重置仓位的费用；未来重置目标只用于标签。\n\n"
    text += "|策略|滑点 bp|累计净收益 %|最大回撤 %|恢复次数|\n|---|---:|---:|---:|---:|\n"
    for row in table.itertuples():
        text += f"|{row.family}|{row.slippage_bps}|{row.return_pct:.2f}|{row.max_drawdown_pct:.2f}|{row.recovery_actions}|\n"
    text += f"\n决定：`{summary['decision']}`。门槛：{json.dumps(summary['gates'], ensure_ascii=False)}。分钟相对状态模型的累计收益百分点差：{summary['minute_increment_return_pct_points']}。年度和滚动差异见 regime_metrics.csv、window_metrics.csv。\n\n"
    text += "风险：历史已重复研究，没有干净样本外或合格前向证据；条件均值 HAC 误差不是未来收益覆盖保证；标签来自固定参考账户，未模拟完整未来风控树。当前 IBKR Tiered 费表为历史投影；滑点、通道、报价时点为假设，复权 NAV 未认证实际股息/成交对账。\n\n"
    text += "本固定路线结束，不反转信号或追加参数搜索。下一信息需求：若要评估交易可实现性，需真实首收时间、买卖报价、成交和佣金回执；若要继续寻找预测信息，需独立机制和足够的成熟恢复样本，不能从本窗口再挑阈值。Notebook、summary 与 NAS Recorder 是审计入口。\n"
    return text
