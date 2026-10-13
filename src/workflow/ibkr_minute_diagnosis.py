"""Notebook-owned IBKR fee alignment and frozen minute-model diagnosis."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import _path_metrics, annual_paired_metrics, paired_metrics
from src.backtest.stage1_ibkr import native_tiered_stage1, replay_tiered_stage1
from src.data.ibkr_fee_marks import build_ibkr_fee_marks
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder, materialize_nas_evidence
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parent_objects(root, relative_card, expected_archive, names):
    import qlib
    from qlib.workflow import R

    card = root / relative_card
    summary = json.loads((card / "summary.json").read_text())
    binding = json.loads((card / "nas_binding.json").read_text())
    if binding["evidence"]["sha256"] != expected_archive:
        raise ValueError("blocked: parent NAS archive changed")
    inventory = json.loads((card / "verification_manifest.json").read_text())["entries"]
    with tempfile.TemporaryDirectory(prefix="soxl-ibkr-parent-") as temporary:
        cache = Path(temporary)
        materialize_nas_evidence(card / "nas_binding.json", card=cache)
        for name in names:
            relative = f"qlib_recorder/{summary['experiment_id']}/{summary['recorder_id']}/artifacts/{name}"
            matches = [e for e in inventory if e["path"] == relative]
            if len(matches) != 1 or _sha(cache / relative) != matches[0]["sha256"]:
                raise ValueError("blocked: parent frozen object changed: " + name)
        with R.uri_context(str(cache / "qlib_recorder")):
            recorder = R.get_recorder(experiment_id=summary["experiment_id"], recorder_id=summary["recorder_id"])
            return {name: recorder.load_object(name) for name in names}


def prepare_ibkr_minute_diagnosis(*, root: Path, card: Path, plan: dict):
    import qlib
    from qlib.data import D

    if _sha(root / "production/stage1_futu_sim_v1/runtime_release.json") != plan["release_sha256"]:
        raise ValueError("blocked: active release differs from the fixed fee audit")
    qlib.init(provider_uri=str(root / plan["adjusted_provider"]), region="us", kernels=1)
    parent = _parent_objects(root, plan["strategy_card"], plan["strategy_archive"],
        ["inputs/market", "inputs/native_marks", "inputs/source_targets", "inputs/plan", "controller/frozen_models",
         "controller/source_recovery_selection"])
    original = _parent_objects(root, plan["allocator_card"], plan["allocator_archive"],
        ["inputs/daily_frame", "inputs/features", "inputs/common", "inputs/timing", "analysis/outer_predictions",
         "models/core_path_outer_71"])
    provider = root / "data/qlib" / plan["experiment"]
    audit = build_ibkr_fee_marks(adjusted_provider=root / plan["adjusted_provider"], raw_provider=root / plan["raw_provider"],
        provider_uri=provider, expected_adjusted_manifest=plan["adjusted_manifest_sha256"],
        expected_raw_manifest=plan["raw_manifest_sha256"], start=plan["signal_start"], as_of=plan["as_of"])
    market, marks = parent["inputs/market"].copy(), parent["inputs/native_marks"]
    for column in marks:
        if column not in market:
            market[column] = marks[column].to_numpy()
    prices = D.features(["SOXL", "SOXX"], ["$raw_execution_price"], start_time=plan["signal_start"],
                        end_time=plan["as_of"] + " 16:00:00", freq="15min")["$raw_execution_price"].unstack("instrument")
    for symbol in ("SOXL", "SOXX"):
        for slot, offset in [("open", pd.Timedelta(hours=9, minutes=45)), ("cut", pd.Timedelta(hours=10, minutes=30))]:
            market[f"raw_{symbol.lower()}_{slot}"] = prices[symbol].reindex(pd.DatetimeIndex(market.trade_date + offset)).to_numpy(float)
    targets = parent["inputs/source_targets"]
    # These two swaps are causal diagnostics, never selectable candidates.
    for name, weight_source, confidence_source in plan["diagnostic_swaps"]:
        target = targets[weight_source].copy()
        target[["tail_probability", "alpha_probability"]] = targets[confidence_source][["tail_probability", "alpha_probability"]]
        targets[name] = target
    return {"market": market, "targets": targets, "models": parent["controller/frozen_models"],
            "recovery_selection": parent["controller/source_recovery_selection"],
            "recovery_spec": parent["inputs/plan"]["recovery_spec"], "action_spec": parent["inputs/plan"]["action_spec"],
            "provider_uri": provider, "audit": audit, "original": original, "plan_sha256": _sha(card / "plan.json")}


def diagnose_minute_heads(original, *, cutoff):
    import torch
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from src.model.minute_pit_allocator import _scaled
    from src.workflow.minute_pit_allocator import _ensemble

    model = original["models/core_path_outer_71"]
    features, common = original["inputs/features"], original["inputs/common"]
    dates = pd.DatetimeIndex(features.index[features.index >= pd.Timestamp("2023-01-03")])
    feature, shared = features.loc[dates], common.loc[dates]
    primary = torch.cat([_scaled(feature, model.base_features, model.primary_scaler), model._extra_tensor(feature)], dim=1)
    shared_x = _scaled(shared, list(model.common_scaler["mean"]), model.common_scaler)
    neutral = primary.clone()
    for number, name in enumerate(model.extra_features):
        if name not in ("minute_core_ready", "minute_peer_ready"):
            neutral[:, len(model.base_features) + number] = 0.
    with torch.no_grad():
        actual, tail, alpha = model.network(primary, shared_x)
        neutral_logits, neutral_tail, neutral_alpha = model.network(neutral, shared_x)
    architecture = {"observations": len(dates), "minute_numerical_channels": len(model.extra_features) - 2,
        "maximum_tail_logit_change_when_minute_channels_neutralized": float((tail - neutral_tail).abs().max()),
        "maximum_alpha_logit_change_when_minute_channels_neutralized": float((alpha - neutral_alpha).abs().max()),
        "mean_absolute_allocation_logit_change": float((actual - neutral_logits).abs().mean()),
        "minute_features_directly_enter_risk_or_alpha_heads": False,
        "explanation": "tail_encoder(common_features) supplies both heads; minute channels only enter allocation_head",
        "minute_information_clock": "previous completed session, next daily open; not a live minute-frequency action model"}
    labels, timing = original["inputs/daily_frame"], original["inputs/timing"]
    matured = timing.label_available_at.notna() & timing.label_available_at.lt(pd.Timestamp(cutoff))
    rows = []
    for family in ("base", "availability_control", "core_path"):
        predictions = _ensemble(original["analysis/outer_predictions"], family, [71, 172, 273])
        for year in sorted(predictions.index.year.unique()):
            keep = predictions.index[(predictions.index.year == year) & matured.reindex(predictions.index).fillna(False).to_numpy()]
            for name, column in [("tail_event_5d", "tail_probability"), ("relative_alpha_5d", "alpha_probability")]:
                outcome = labels.loc[keep, name]
                valid = outcome.notna() & predictions.loc[keep, column].notna()
                outcome = outcome.gt(0).astype(int) if name == "relative_alpha_5d" else outcome
                y, p = outcome.loc[valid].to_numpy(), predictions.loc[keep[valid], column].to_numpy()
                if not len(y):
                    continue
                rows.append({"family": family, "signal_year": int(year), "target": name, "observations": len(y),
                    "auc": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else None,
                    "brier": float(brier_score_loss(y, p)), "event_frequency": float(np.mean(y)),
                    "mean_probability": float(np.mean(p)), "role": "reused-history calibration diagnosis; no model selection"})
    return architecture, pd.DataFrame(rows)


def run_ibkr_minute_diagnosis(*, root: Path, card: Path, plan: dict, prepared: dict):
    import qlib
    import yaml
    from qlib.workflow import R

    if (card / "summary.json").exists():
        summary = json.loads((card / "summary.json").read_text())
        if summary["plan_sha256"] != _sha(card / "plan.json"):
            raise ValueError("blocked: finalized IBKR plan changed")
        return summary
    if prepared["plan_sha256"] != _sha(card / "plan.json"):
        raise ValueError("blocked: IBKR prepared inputs differ from the fixed plan")
    qlib.init(provider_uri=str(prepared["provider_uri"]), region="us", kernels=1)
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="tiered_raw_share_cost_alignment", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=_sha(card / "plan.json"), new_model_fits=0,
                            native_budget=plan["maximum_native_cases"], capital_usd=plan["initial_cash"])
        recorder.save_objects(artifact_path="inputs", market=prepared["market"], targets=prepared["targets"],
            controller=prepared["models"], plan=plan, price_audit=prepared["audit"])
        recorder.save_objects(artifact_path="sources", **{name: Path(path).read_text() for name, path in plan["source_html_cache"].items()})
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=root, **prepared["recovery_spec"])
    action_spec = Stage1OpenActionValueSpec(root=root, **prepared["action_spec"])
    books = {}
    cases = [(family, slip) for slip in plan["slippage_bps"] for family in plan["families"]]
    cases += [(name, plan["primary_slippage_bps"]) for name, _, _ in plan["diagnostic_swaps"]]
    for family, slip in cases:
        key = f"tiered_{slip}_{family}"
        if len(books) >= plan["maximum_native_cases"]:
            raise ValueError("blocked: IBKR native-case budget exhausted")
        frame = prepared["market"].copy()
        if family != "soxx":
            frame = frame.join(prepared["targets"][family].set_axis(frame.index))
        replay = replay_tiered_stage1(frame, models=prepared["models"], tariff=plan["tariff"], slippage_bps=slip,
            recovery_spec=recovery_spec, action_spec=action_spec, recovery_selection=prepared["recovery_selection"],
            initial_cash=plan["initial_cash"], benchmark=family == "soxx")
        book = native_tiered_stage1(frame, replay, provider_uri=prepared["provider_uri"], tariff=plan["tariff"],
            slippage_bps=slip, initial_cash=plan["initial_cash"])
        books[key] = book
        recorder.save_objects(artifact_path="analysis/native", **{key: book})
        recorder.log_metrics(completed_native_cases=len(books))
        print(f"native {len(books)}/{plan['maximum_native_cases']} {key}", flush=True)
    architecture, calibration = diagnose_minute_heads(prepared["original"], cutoff=plan["label_cutoff"])
    rows, annual = [], []
    for family, slip in cases:
        key = f"tiered_{slip}_{family}"
        book, reference = books[key], books[f"tiered_{slip}_soxx"]
        orders, path = book["orders"], book["authority"]
        pair = paired_metrics(path, reference["authority"])
        total_notional = float(orders.notional.sum()) if len(orders) else 0.
        cash_cost = float((orders.total - orders.slippage).sum()) if len(orders) else 0.
        rows.append({"family": family, "slippage_bps": slip, **pair,
            "orders": len(orders), "broker_and_external_fees_usd": cash_cost,
            "slippage_reserve_usd": float(orders.slippage.sum()) if len(orders) else 0.,
            "notional_weighted_broker_and_external_bps": cash_cost / total_notional * 10_000. if total_notional else 0.,
            "minimum_broker_commission_orders": int(np.isclose(orders.brokerage, plan["tariff"]["minimum_per_order"]).sum()) if len(orders) else 0,
            "average_ending_cash_pct": float(100. * path.ending_cash_weight.mean()),
            "emergency_drawdown_days": int(path.portfolio_drawdown.ge(prepared["recovery_spec"]["emergency_drawdown"]).sum()),
            "recovery_actions": int(path.recovery_action.sum()),
            "maximum_monthly_fee_counted_raw_shares": book["audit"]["maximum_monthly_fee_counted_raw_shares"],
            "native_maximum_return_error": book["audit"]["errors"]["return_error"],
            "selectable": family in plan["families"]})
        yearly = annual_paired_metrics(path, reference["authority"])
        yearly["family"], yearly["slippage_bps"] = family, slip
        annual.append(yearly)
    table, annual = pd.DataFrame(rows), pd.concat(annual, ignore_index=True)
    selected = table.loc[table.family.eq("core_path")]
    summary = {"experiment": plan["experiment"], "plan_sha256": _sha(card / "plan.json"),
        "decision": "not_promoted_fee_alignment_and_structure_diagnosis", "objective_achieved": False,
        "production_enabled": False, "new_model_fits": 0, "native_cases": len(books),
        "current_release_changed": False, "forward_qualified_sessions": 0, "outer_is_reused_history": True,
        "account": {"broker": "IBKR Pro", "plan": "Tiered", "initial_cash_usd": plan["initial_cash"],
                    "whole_share_orders": True, "other_account_monthly_share_volume_assumed": 0},
        "tariff": plan["tariff"], "fee_schedule_is_current_projection_not_historical_bill": True,
        "venue_and_slippage_are_assumptions_not_observed_fills": True,
        "as_of": plan["as_of"], "fee_quote_as_of": plan["fee_quote_as_of"],
        "provider_uri": str(prepared["provider_uri"]), "data_authority": "yanxuan_nas", "region": "us", "market": "all",
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id), "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "direct_minute_head_audit": architecture, "minute_candidate_above_soxx_all_slippage_scenarios": bool(selected.relative_wealth.gt(1.).all()),
        "next_action": "bounded direct minute risk/alpha information and horizon alignment; validate current-commission cost in training before a new fit",
        "compact_metrics": table.to_dict("records"),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/market", "inputs/targets", "analysis/native/*", "sources/*", "analysis/comparison", "analysis/calibration"]},
        "qlib_docs_reference": ["https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/strategy.html", "https://qlib.readthedocs.io/en/latest/component/recorder.html"]}
    recorder.save_objects(artifact_path="analysis", comparison=table, annual=annual, calibration=calibration,
                          direct_minute_head_audit=architecture, summary=summary)
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]),
                        native_completed=len(books), research_model_fits=0)
    if _sha(root / "production/stage1_futu_sim_v1/runtime_release.json") != plan["release_sha256"]:
        raise ValueError("blocked: release changed during fee diagnosis")
    table.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    annual.to_csv(card / "regime_metrics.csv", index=False)
    calibration.to_csv(card / "probability_calibration.csv", index=False)
    atomic_json(architecture, card / "minute_head_audit.json")
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"],
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking), "region": "us", "freq": "15min",
        "as_of": plan["as_of"], "segments": {"reused_diagnostic": [plan["signal_start"], plan["as_of"]]},
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    entries = [{"path": str(p.relative_to(card)), "sha256": _sha(p), "bytes": p.stat().st_size}
               for p in sorted(artifacts.rglob("*")) if p.is_file()]
    atomic_json({"recorder_id": recorder.id, "entries": entries}, card / "verification_manifest.json")
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card / "summary.json")
    _write_conclusion(card, summary, table)
    return summary


def _write_conclusion(card, summary, table):
    primary = table.loc[table.slippage_bps.eq(2) & table.selectable]
    lines = ["# IBKR 阶梯费率与分钟模型诊断", "",
        "使用用户确认的 IBKR Pro Tiered、10 万美元本金；逐资产整股下单。佣金用真实 NONE 价格的股数逐单计算，不能用复权记账单位直接乘每股费率。",
        "采用查询日现行价目表投影到相同历史窗口，并假设 SMART 订单按 Nasdaq 流动性移除费收费、每单一次成交、其他账户成交量为零。0/2/5bp 是单独列出的滑点现金成本情景，不是 IBKR 佣金。TAF 当前假期费率为零，CAT/清算/卖出 SEC 与佣金附加费分别列账；保留浮点美元，不声明账单美分取整或税务资格。", "",
        "| 完整策略 | Tiered + 2bp 滑点：累计净收益 | 最大回撤 | 佣金及附加费等效 bp |", "|---|---:|---:|---:|"]
    for row in primary.itertuples():
        lines.append(f"| {row.family} | {row.return_pct:.2f}% | {row.max_drawdown_pct:.2f}% | {row.notional_weighted_broker_and_external_bps:.3f} |")
    lines += ["", "全部滑点情景、两种仓位/置信度互换诊断见 backtest_benchmark_comparison.csv；互换项不能晋升。风险状态依据各账户自己的真实费用、现金和净值重算；Native Qlib 逐单费用、现金、数量、下一开盘 NAV 与独立账本核对。",
        "结构证据：将核心分钟数值通道中性化，风险和 alpha logits 的变化严格为零，而仓位 logits 改变。原共享 tail_encoder 仅读取公共日频特征；新增分钟数据只进入 allocation_head。概率年度 AUC/Brier 见 probability_calibration.csv，反映同一已复用历史，不能选新模型。",
        "限制：权益 NAV 沿用父研究的复权记账平面；整股表示的是每次交易金额对应的 raw 股数，未实现实际分红现金/税务/跨除权持股及真实 09:31/10:31 成交复核。结果是统一的当前费用投影，不是 IBKR 历史实盘账单或独立前向超额。",
        "晋升：not_promoted_fee_alignment_and_structure_diagnosis；模型未重训、现役模拟盘未切换，独立合格前向为 0。",
        "后续：先让分钟信息进入专门的风险/超额预测路径，并把信息时点、预测期限与实际动作统一；训练成本也要使用本次逐单口径。在冻结的有限实验中与原结构、可用性对照比较，禁止根据本窗口改阈值或换候选。",
        f"Recorder：{summary['recorder_id']}；数据、订单、持仓和大报告保留在 NAS。", ""]
    (card / "conclusion.md").write_text("\n".join(lines))
