"""Bounded notebook workflow for state-aware, fee-aligned full-path learning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import annual_paired_metrics, paired_metrics, stationary_bootstrap_excess
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.data.full_path_prices import build_full_path_prices
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder, materialize_nas_snapshot
from src.model.full_path_tiered import FullPathTieredModel, PRICE_FIELDS
from src.workflow.minute_action_value import _source
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_full_path(*, root, card, plan):
    import qlib
    qlib.init(provider_uri=str(root / "data/qlib/minute_information_pit_allocator_20261012/core_1min"), region="us", kernels=1)
    if _sha(root / "production/stage1_futu_sim_v1/runtime_release.json") != plan["release_sha256"]:
        raise ValueError("blocked: active release differs")
    source, source_audit = _source(root, plan["allocator_source"],
        ["inputs/features", "inputs/plan", "models/core_path_outer_71"])
    action, action_audit = _source(root, plan["action_source"], ["inputs/plan", "controller/source_recovery_selection"])
    anchor, anchor_audit = _source(root, plan["fee_source"], ["inputs/market", "inputs/targets", "inputs/controller"])
    manifest = root / plan["source_manifest"]
    if not manifest.is_file():
        materialize_nas_snapshot(root / plan["snapshot_binding"], workspace_root=root, include_derived=False)
    provider = root / "data/qlib" / plan["experiment"]
    quotes, price_audit = build_full_path_prices(source_manifest=manifest, source_sha256=plan["source_id"],
        provider_uri=provider, start=plan["source_start"], as_of=plan["as_of"])
    original_features = source["inputs/features"]
    base = list(source["models/core_path_outer_71"].base_features) + list(source["inputs/plan"]["availability_features"])
    extra = list(source["inputs/plan"]["feature_groups"]["core"])
    features = original_features.loc[:, [*base, *extra]].copy()
    days = pd.DatetimeIndex(quotes.index.get_level_values("datetime").unique()).normalize().unique().sort_values()
    frame = pd.DataFrame({"signal_date": days[:-2], "trade_date": days[1:-1], "exit_date": days[2:]})
    frame.index = pd.DatetimeIndex(frame.trade_date, name="datetime")
    features = features.reindex(pd.DatetimeIndex(frame.signal_date)).set_axis(frame.index)
    if features.iloc[:, :len(base) - 2].isna().any().any():
        raise ValueError("needs_data: original daily feature incomplete on native calendar")
    frame = frame.join(features)
    for symbol in ("SOXL", "SOXX"):
        lower, asset = symbol.lower(), quotes.xs(symbol, level="instrument")
        for key, dates in [("known", frame.trade_date + pd.Timedelta(hours=9, minutes=45)),
                            ("open", frame.trade_date + pd.Timedelta(hours=9, minutes=47)),
                            ("recovery", frame.trade_date + pd.Timedelta(hours=10, minutes=32)),
                            ("exit_open", frame.exit_date + pd.Timedelta(hours=9, minutes=47))]:
            for field, prefix in [("$execution_price", ""), ("$raw_execution_price", "raw_")]:
                name = f"{prefix}known_{lower}" if key == "known" else f"{prefix}{lower}_{key}"
                frame[name] = asset[field].reindex(pd.DatetimeIndex(dates)).to_numpy(float)
        close = asset.groupby(asset.index.normalize()).last()
        frame[f"close_{lower}"] = close["$execution_price"].reindex(pd.DatetimeIndex(frame.trade_date)).to_numpy(float)
    if not np.isfinite(frame.loc[:, list(PRICE_FIELDS)]).all().all() or not frame.loc[:, list(PRICE_FIELDS)].gt(0).all().all():
        raise ValueError("needs_data: native known/fill prices incomplete")
    frame["feature_available_at"] = frame.signal_date + pd.Timedelta(hours=16, minutes=1)
    frame["decision_at"] = frame.trade_date + pd.Timedelta(hours=9, minutes=46)
    frame["label_available_at"] = frame.exit_date + pd.Timedelta(hours=9, minutes=49)
    if not frame.feature_available_at.lt(frame.decision_at).all():
        raise ValueError("blocked: previous session information not available")
    previous = anchor["inputs/market"].copy()
    previous = previous.join(anchor["inputs/targets"]["frozen_current"].set_axis(previous.index)).set_index("trade_date")
    for column in previous:
        if column not in frame:
            frame[column] = previous[column].reindex(frame.index)
    # Existing first-hour confidence input is retained only for the frozen
    # outer anchor; the new daily allocator never consumes it.
    return {"frame": frame, "columns": [*base, *extra], "base_count": len(base), "provider_uri": provider,
        "controller": anchor["inputs/controller"], "recovery_spec": action["inputs/plan"]["recovery_spec"],
        "action_spec": action["inputs/plan"]["action_spec"], "selection": action["controller/source_recovery_selection"],
        "audit": {"allocator_source": source_audit, "action_source": action_audit, "anchor_source": anchor_audit,
                  "prices": price_audit, "sessions": len(frame), "minute_unknown_sessions": int(frame.minute_core_ready.ne(1).sum())},
        "plan_sha256": _sha(card / "plan.json")}


def full_path_dataset(frame, columns, cutoff):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    joined = pd.concat({"feature": frame[columns], "price": frame.loc[:, list(PRICE_FIELDS)],
        "meta": frame[["label_available_at", "decision_at", "feature_available_at"]]}, axis=1)
    joined.index = pd.MultiIndex.from_arrays([frame.index, ["SOXL_SOXX_CASH"] * len(frame)], names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(joined), segments={"train": (None, pd.Timestamp(cutoff) - pd.Timedelta(nanoseconds=1)),
                                                                   "test": (pd.Timestamp(cutoff), None)})


def run_full_path(*, root, card, plan, prepared):
    import qlib
    import yaml
    from qlib.workflow import R
    digest = _sha(card / "plan.json")
    if (card / "summary.json").is_file():
        summary = json.loads((card / "summary.json").read_text())
        if summary["plan_sha256"] != digest:
            raise ValueError("blocked: finalized plan changed")
        return summary
    if prepared["plan_sha256"] != digest:
        raise ValueError("blocked: prepared plan changed")
    qlib.init(provider_uri=str(prepared["provider_uri"]), region="us", kernels=1)
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="full_path_economic_objective", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=digest, maximum_model_fits=plan["maximum_model_fits"], maximum_native_cases=plan["maximum_native_cases"])
        recorder.save_objects(artifact_path="inputs", frame=prepared["frame"], plan=plan, audit=prepared["audit"], controller=prepared["controller"])
        files = ["src/data/full_path_prices.py", "src/model/full_path_tiered.py", "src/backtest/full_path_tiered.py",
                 "src/workflow/full_path_tiered.py", "src/backtest/ibkr_tiered.py", "src/backtest/stage1_ibkr.py"]
        recorder.save_objects(artifact_path="sources", **{name.replace("/", "__"): (root / name).read_text() for name in files})
    models, audits = {}, []
    for fold in plan["folds"]:
        dataset = full_path_dataset(prepared["frame"], prepared["columns"], fold["cutoff"])
        for family in plan["families"]:
            models[(fold["name"], family)] = []
            for seed in plan["seeds"]:
                if len(audits) >= plan["maximum_model_fits"]:
                    raise ValueError("blocked: fit budget exhausted")
                model = FullPathTieredModel(columns=prepared["columns"], base_count=prepared["base_count"], family=family,
                    seed=seed, cutoff=fold["cutoff"], plan=plan).fit(dataset)
                models[(fold["name"], family)].append(model)
                audits.append({"fold": fold["name"], **model.audit})
                recorder.save_objects(artifact_path="models", **{f"{fold['name']}_{family}_{seed}": model})
                recorder.log_metrics(completed_model_fits=len(audits))
                atomic_json({"phase": "training", "recorder_id": recorder.id, "completed_model_fits": len(audits),
                    "last_model": [fold["name"], family, seed], "training_rows": model.audit["training_rows"]}, card / "progress.json")
                print(f"fit {len(audits)}/{plan['maximum_model_fits']} {fold['name']} {family} {seed}", flush=True)
    books, rows, annual, windows, uncertainty = {}, [], [], [], []
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=root, **prepared["recovery_spec"])
    action_spec = Stage1OpenActionValueSpec(root=root, **prepared["action_spec"])
    for fold in plan["folds"]:
        frame = prepared["frame"].loc[prepared["frame"].signal_date.between(fold["start"], fold["end"])].copy()
        cases = [(family, "ensemble", slip) for slip in plan["slippage_bps"] for family in [*plan["families"], "soxx"]]
        cases += [(family, str(seed), plan["primary_slippage_bps"]) for family in plan["families"] for seed in plan["seeds"]]
        if fold["name"] == "reused_outer":
            cases += [("anchor", "frozen", slip) for slip in plan["slippage_bps"]]
        for family, seed, slip in cases:
            if len(books) >= plan["maximum_native_cases"]:
                raise ValueError("blocked: native account budget exhausted")
            chosen = models.get((fold["name"], family))
            if chosen is not None and seed != "ensemble":
                chosen = [model for model in chosen if str(model.seed) == seed]
            replay = replay_full_path(frame, models=chosen, plan=plan, slippage_bps=slip, benchmark=family == "soxx",
                controller=prepared["controller"] if family == "anchor" else None,
                recovery_spec=recovery_spec, action_spec=action_spec, selection=prepared["selection"])
            book = native_tiered_stage1(frame, replay, provider_uri=prepared["provider_uri"], tariff=plan["tariff"],
                slippage_bps=slip, initial_cash=plan["initial_cash"], freq="1min", opening_offset=pd.Timedelta(hours=9, minutes=47))
            book["targets"] = replay["targets"]
            key = (fold["name"], family, seed, slip)
            books[key] = book
            recorder.save_objects(artifact_path="analysis/native", **{"_".join(map(str, key)): book})
            recorder.log_metrics(completed_native_cases=len(books))
            atomic_json({"phase": "native", "recorder_id": recorder.id, "completed_model_fits": len(audits),
                "completed_native_cases": len(books), "last_case": list(key)}, card / "progress.json")
            print(f"native {len(books)}/{plan['maximum_native_cases']} {key}", flush=True)
    for (fold, family, seed, slip), book in books.items():
        benchmark = books[(fold, "soxx", "ensemble", slip)]["authority"]
        path = book["authority"]
        row = {"fold": fold, "family": family, "seed": seed, "slippage_bps": slip, **paired_metrics(path, benchmark),
            "maximum_monitored_drawdown_pct": -100 * float(path.monitored_drawdown.max()),
            "average_cash_pct": 100 * float(path.ending_cash_weight.mean()), "orders": len(book["orders"]),
            "broker_and_external_fees_usd": float(path.broker_and_external_fees_usd.sum()),
            "slippage_reserve_usd": float(path.slippage_reserve_usd.sum())}
        rows.append(row)
        if seed == "ensemble" or family == "anchor":
            years = annual_paired_metrics(path, benchmark)
            years["fold"], years["family"], years["seed"], years["slippage_bps"] = fold, family, seed, slip
            annual.append(years)
            logs = np.log1p(path.strategy_return.to_numpy()) - np.log1p(benchmark.strategy_return.to_numpy())
            rolling = pd.Series(logs).rolling(126).sum().dropna()
            windows.append({"fold": fold, "family": family, "slippage_bps": slip, "positive_126d_fraction": float(rolling.gt(1e-8).mean())})
            if family in plan["families"]:
                for against in ["soxx", "daily"] if family == "minute" else ["soxx"]:
                    reference = books[(fold, against, "ensemble", slip)]["authority"]
                    stats = stationary_bootstrap_excess(path, reference, mean_block=20, resamples=2000, alpha=.025, seed=20261013)
                    uncertainty.append({"fold": fold, "family": family, "slippage_bps": slip, "against": against, **stats})
    table, annual = pd.DataFrame(rows), pd.concat(annual, ignore_index=True)
    windows, uncertainty = pd.DataFrame(windows), pd.DataFrame(uncertainty)
    candidate = table.loc[table.family.eq("minute") & table.seed.eq("ensemble")]
    gates = {"net_excess_every_fold_and_cost": bool(candidate.relative_wealth.gt(1 + 1e-8).all()),
        "drawdown_30pct_every_fold_and_cost": bool(candidate.max_drawdown_pct.ge(-30).all() & candidate.maximum_monitored_drawdown_pct.ge(-30).all()),
        "cvar_no_worse_than_soxx": bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),
        "rolling_win_fraction_at_least_70pct": bool(windows.loc[windows.family.eq("minute"), "positive_126d_fraction"].ge(.70).all()),
        "bootstrap_excess_and_minute_increment": bool(uncertainty.loc[uncertainty.family.eq("minute"), "lower_bound_positive"].all()),
        "independent_forward_qualified": False}
    historical = all(value for key, value in gates.items() if key != "independent_forward_qualified")
    summary = {"experiment": plan["experiment"], "plan_sha256": digest, "decision": "needs_independent_forward" if historical else "not_promoted_full_path_gates_failed",
        "objective_achieved": False, "goal_remains_active": True, "current_release_changed": False, "production_enabled": False,
        "new_model_fits": len(audits), "native_cases": len(books), "gates": gates, "training_audit": audits,
        "as_of": plan["as_of"], "historical_windows_reused": True, "forward_qualified_sessions": 0,
        "provider_uri": str(prepared["provider_uri"]), "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking), "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "qlib_docs_reference": plan["qlib_docs"], "compact_metrics": candidate.to_dict("records"),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "models/*", "analysis/native/*", "sources/*", "analysis/training_audit"]},
        "scientific_limits": ["reused history; no first-receipt/dividend/fill certification", "integer straight-through gradients approximate the discrete optimization",
            "training episodes reset to cash and overlap at boundaries; evaluation holdings continuous", "drawdown penalty is not a forward bound",
            "current tariff and assumed slippage projected over history; target constraints and realized holdings drift differ"],
        "next_action": "qualify frozen independent forward identity if historical gates pass; otherwise close this fixed route and qualify independent release-time semiconductor capacity data"}
    recorder.save_objects(artifact_path="analysis", comparison=table, annual=annual, windows=windows, uncertainty=uncertainty,
        summary=summary, training_audit=audits)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    for name, data in [("backtest_benchmark_comparison", table), ("regime_metrics", annual), ("window_metrics", windows), ("uncertainty", uncertainty)]:
        data.to_csv(card / (name + ".csv"), index=False)
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)), "sha256": _sha(p), "bytes": p.stat().st_size}
        for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": summary["provider_uri"],
        "recorder_uri": str(tracking), "region": "us", "freq": "1min", "calendar_role": "sparse_observed_events",
        "as_of": plan["as_of"], "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card / "summary.json")
    atomic_json({"candidate": "minute", "selected_for_production": None, "decision": summary["decision"], "selection_uses_outer": False}, card / "selected_model.json")
    text = "# 完整路径扣费优化结论\n\n"
    text += f"已执行 {len(audits)} 次严格成熟训练和 {len(books)} 个 Qlib 原生账户，完整持仓、现金与逐笔费用独立对账。输入从2018真实分钟起点扩充；所有种子保留。\n\n"
    text += "|窗口|模型|滑点bp|净收益%|最大回撤%|相对SOXX财富|\n|---|---|---:|---:|---:|---:|\n"
    for row in table.loc[table.seed.isin(["ensemble", "frozen"]) & table.slippage_bps.eq(2)].itertuples():
        text += f"|{row.fold}|{row.family}|{row.slippage_bps}|{row.return_pct:.2f}|{row.max_drawdown_pct:.2f}|{row.relative_wealth:.4f}|\n"
    text += f"\n决定：`{summary['decision']}`。门槛：{json.dumps(gates, ensure_ascii=False)}。年度/滚动与区块差异见 sidecars，所有费用和 seed 原生证据保留在 NAS Recorder。\n\n"
    text += "风险：历史复用不构成新留出期，真实首收/成交/股息资格仍缺；整数梯度、训练片段初始现金与软回撤处罚都有限制。模型中先读已知价格/过去状态，再使用未来成交价结算，没有把当前成交价作为当前 NN 输入。仓位目标约束不保证持有漂移后的约束。\n\n"
    text += f"目标保持 active；本实验结束不等于稳定超额达成。下一步：{summary['next_action']}。原模拟盘 release 不变。\n"
    (card / "conclusion.md").write_text(text)
    if _sha(root / "production/stage1_futu_sim_v1/runtime_release.json") != plan["release_sha256"]:
        raise ValueError("blocked: active release changed during full-path experiment")
    return summary
