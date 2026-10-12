"""Frozen-model comparison under the active Stage1 controller; never fit here."""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import validate_replay
from src.analysis.native_intraday_schedule import native_intraday_reconciliation
from src.backtest.position_accounting import settle_target_session
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder, materialize_nas_evidence, materialize_nas_snapshot
from src.data.native_execution_marks import build_execution_mark_view, build_published_minute_view, session_price_marks
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec, _load_first_hour_features
from src.workflow.minute_pit_allocator import _ensemble, _frozen_predictions, _market
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    AuxiliaryTrainingCandidate, PitAuxiliaryAllocationSpec, _smooth_weights,
)
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec, _simulate_on_policy_router
from src.workflow.stage1_runtime_release import load_runtime_release


WEIGHTS = ["soxl_weight", "soxx_weight", "cash_weight"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_stage1_minute_comparison(*, root: Path, card: Path, plan: dict) -> dict:
    """Load NAS Recorder objects and real native prices with frozen identities."""
    import qlib
    from qlib.data import D
    from qlib.workflow import R

    release_path = root / "production/stage1_futu_sim_v1/runtime_release.json"
    if _sha(release_path) != plan["runtime_release_sha256"]:
        raise ValueError("blocked: active release changed after the audit plan")
    released = load_runtime_release(root, release_path)
    if released["release"] != plan["parent_release"]:
        raise ValueError("blocked: controller identity differs from the audit plan")
    parent = root / plan["allocator_card"]
    parent_plan = json.loads((parent / "plan.json").read_text())
    if _sha(parent / "plan.json") != plan["allocator_plan_sha256"]:
        raise ValueError("blocked: allocator experiment plan changed")
    binding_path = parent / "nas_binding.json"
    binding = json.loads(binding_path.read_text())
    if binding["evidence"]["sha256"] != plan["allocator_NAS_archive_sha256"]:
        raise ValueError("blocked: allocator NAS evidence differs")
    materialize_nas_snapshot(binding, workspace_root=root)
    verification = json.loads((parent / "verification_manifest.json").read_text())
    if _sha(parent / "verification_manifest.json") != plan["allocator_verification_sha256"]:
        raise ValueError("blocked: allocator artifact inventory changed")
    names = ["inputs/daily_frame", "inputs/features", "inputs/common", "inputs/timing",
             "inputs/frozen_parent_models", "analysis/outer_predictions", "analysis/comparison", "analysis/summary"]
    checked = []
    with tempfile.TemporaryDirectory(prefix="soxl-stage1-parent-") as temporary:
        cache = Path(temporary)
        materialize_nas_evidence(binding_path, card=cache)
        for name in names:
            relative = f"qlib_recorder/{plan['allocator_experiment_id']}/{plan['allocator_recorder_id']}/artifacts/{name}"
            entries = [entry for entry in verification["entries"] if entry["path"] == relative]
            if len(entries) != 1 or _sha(cache / relative) != entries[0]["sha256"]:
                raise ValueError("blocked: NAS allocator object differs: " + name)
            checked.append(entries[0])
        # A real Qlib Recorder remains the source of inputs and predictions.
        qlib.init(provider_uri=str(root / "data/qlib" / parent_plan["experiment"] / "daily_prices"),
                  region="us", kernels=1)
        with R.uri_context(str(cache / "qlib_recorder")):
            recorder = R.get_recorder(experiment_id=plan["allocator_experiment_id"],
                                      recorder_id=plan["allocator_recorder_id"])
            objects = {name: recorder.load_object(name) for name in names}
    if objects["analysis/summary"]["selected_family"] != plan["selected_family"]:
        raise ValueError("blocked: this audit cannot select a different allocator")
    inputs = {"frame": objects["inputs/daily_frame"], "features": objects["inputs/features"],
              "common": objects["inputs/common"], "timing": objects["inputs/timing"],
              "frozen_models": objects["inputs/frozen_parent_models"]}
    source_market = _market(inputs, window="outer", plan=parent_plan)
    source_market = source_market.loc[source_market.exit_date.le(pd.Timestamp(plan["as_of"]))].reset_index(drop=True)
    providers = root / "data/qlib" / plan["experiment"]
    intraday, execution = providers / "intraday", providers / "execution_marks"
    source_manifest = root / binding["source_manifest_path"]
    source_audit = build_published_minute_view(source_manifest=source_manifest,
        source_sha256=plan["source_id"], provider_uri=intraday,
        start=plan["signal_start"], as_of=plan["as_of"])
    execution_audit = build_execution_mark_view(source_provider=intraday, provider_uri=execution,
        start=plan["signal_start"], end=plan["as_of"] + " 16:00:00", as_of=plan["as_of"])
    early = _load_first_hour_features(SimpleNamespace(intraday_provider_uri=intraday,
                                      intraday_bars_observed=plan["recovery_spec"]["intraday_bars_observed"]),
                                      end=pd.Timestamp(plan["as_of"]))
    marks = session_price_marks(provider_uri=execution, sessions=source_market)
    market = source_market.drop(columns=["soxl_return", "soxx_return"]).copy()
    market[["soxl_return", "soxx_return"]] = marks[["soxl_return", "soxx_return"]]
    early_aligned = early.reindex(pd.DatetimeIndex(market.trade_date)).reset_index(drop=True)
    if not np.isfinite(early_aligned.to_numpy(float)).all():
        raise ValueError("needs_data: a required Stage1 first-hour field is missing")
    for column in early_aligned:
        market[column] = early_aligned[column]
    for column in ["early_ret_soxl", "early_ret_soxx"]:
        if not np.allclose(market[column], marks[column], rtol=0, atol=1e-7):
            raise ValueError("blocked: first-hour features and native marks differ")
    dates = pd.DatetimeIndex(market.signal_date)
    calendar = pd.DatetimeIndex(D.calendar(freq="15min")).normalize().unique()
    required = calendar[(calendar >= market.trade_date.min()) & (calendar < market.exit_date.max())]
    if not required.equals(pd.DatetimeIndex(market.trade_date)):
        raise ValueError("blocked: audit drops a native accounting session")
    raw = {"frozen_current": _frozen_predictions(inputs, window="outer", plan=parent_plan)}
    for family in plan["families"]:
        if family not in {"frozen_current", "soxx"}:
            raw[family] = _ensemble(objects["analysis/outer_predictions"], family, parent_plan["seeds"])
    spec = PitAuxiliaryAllocationSpec(root=root, smoothing_span=parent_plan["smoothing_span"],
                                      rebalance_threshold=parent_plan["rebalance_threshold"])
    candidate = AuxiliaryTrainingCandidate(**inputs["frozen_models"]["outer_refit"]["candidate"])
    returns = market.set_index("signal_date")[["soxl_return", "soxx_return"]]
    targets = {family: _smooth_weights(prediction.reindex(dates), spec=spec, candidate=candidate,
                                       asset_returns=returns) for family, prediction in raw.items()}
    for prediction in targets.values():
        if not np.isfinite(prediction.to_numpy(float)).all():
            raise ValueError("needs_data: frozen allocator predictions omit an accounting day")
    price_delta = {symbol: float((source_market[f"{symbol}_return"] - market[f"{symbol}_return"]).abs().max())
                   for symbol in ("soxl", "soxx")}
    audit = {"parent_objects_checked": checked, "source_manifest_sha256": plan["source_id"],
             "native_source": source_audit, "native_execution": execution_audit,
             "sessions": len(market), "signal_start": str(market.signal_date.min().date()),
             "first_trade": str(market.trade_date.min().date()), "last_exit": str(market.exit_date.max().date()),
             "no_accounting_sessions_dropped": True, "unscored_tail_signal_dates":
             [str(day.date()) for day in raw["frozen_current"].index if day > dates.max()],
             "maximum_return_difference_from_allocator_only_daily_marks": price_delta,
             "source_smoothing": "released EMA/HOLD on the same virtual gross marked source holdings; not the Stage1 account",
             "source_smoothing_span": spec.smoothing_span, "source_rebalance_threshold": spec.rebalance_threshold,
             "new_model_fits": 0, "action_or_risk_parameters_changed": False,
             "all_families_share_identical_controller_and_native_prices": True,
             "historical_open_first_hour_close_marks_are_actual_broker_fills": False}
    return {"plan_sha256": _sha(card / "plan.json"),
            "market": market, "raw_predictions": raw, "source_targets": targets, "marks": marks,
            "released": released, "provider_uri": execution, "audit": audit,
            "allocator_summary": objects["analysis/summary"],
            "allocator_only_comparison": objects["analysis/comparison"]}


def _soxx_schedule(market, cost_bps):
    """One funded entry, then exact HOLD of native SOXX quantities."""
    held, equity, rows = np.array([0., 0., 1.]), 1., []
    funded = np.array([0., 1. / (1. + cost_bps / 10_000.),
                       (cost_bps / 10_000.) / (1. + cost_bps / 10_000.)])
    for number, row in enumerate(market.itertuples()):
        target = funded if number == 0 else None
        settled = settle_target_session(held, open_target_weights=target,
            asset_returns=[row.soxl_return, row.soxx_return, 0.], cost_bps=cost_bps)
        opening = funded if target is not None else held.copy()
        held = settled["ending_weights"]
        equity *= 1. + settled["net_return"]
        rows.append({"signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date,
            "strategy_return": settled["net_return"], "turnover": settled["turnover"],
            "fees": settled["fees"], "equity": equity, "cost_bps": cost_bps,
            "execute_open_target": number == 0, "execute_open_trade": number == 0, "recovery_action": False,
            **{f"open_{key}": value for key, value in zip(WEIGHTS, opening)},
            **dict(zip(WEIGHTS, opening)),
            **{f"ending_{key}": value for key, value in zip(WEIGHTS, held)}})
    return validate_replay(pd.DataFrame(rows))


def run_stage1_minute_comparison(*, root: Path, card: Path, plan: dict, prepared: dict) -> dict:
    """Replay a fixed audit once, including native Qlib orders/cash/quantities."""
    import qlib
    import yaml
    from qlib.workflow import R
    from src.analysis.minute_allocator import compare_native_books

    plan_hash = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        saved = json.loads((card / "summary.json").read_text())
        if saved["plan_sha256"] != plan_hash:
            raise ValueError("blocked: finalized full-Stage1 comparison plan differs")
        return saved
    if prepared["plan_sha256"] != plan_hash:
        raise ValueError("blocked: prepared inputs belong to a different comparison plan")
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=root, **plan["recovery_spec"])
    action_spec = Stage1OpenActionValueSpec(root=root, **plan["action_spec"])
    released, market = prepared["released"], prepared["market"]
    frozen = released["models"]
    years = sorted(market.signal_date.dt.year.unique())
    open_models = {int(year): frozen["frozen_open_action_value_model"] for year in years}
    recovery_models = {int(year): frozen["frozen_recovery_model"] for year in years}
    tracking = card / "qlib_recorder"
    qlib.init(provider_uri=str(prepared["provider_uri"]), region="us", kernels=1)
    books = {}
    with R.start(experiment_name=plan["experiment"], recorder_name="frozen_full_stage1_comparison", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=plan_hash, new_model_fits=0, maximum_native_cases=plan["maximum_native_cases"],
                            allocation_recorder=plan["allocator_recorder_id"], active_release=plan["parent_release"]["release_id"])
        recorder.save_objects(artifact_path="inputs", market=market, raw_predictions=prepared["raw_predictions"],
            source_targets=prepared["source_targets"], native_marks=prepared["marks"], input_audit=prepared["audit"], plan=plan,
            allocator_only_comparison=prepared["allocator_only_comparison"])
        recorder.save_objects(artifact_path="controller", frozen_models=frozen,
            source_recovery_selection=released["action_summary"]["source_stage1"]["selected_model"])
    recorder.set_tags(audit_status="running_native_accounts")
    for fee in plan["costs_bps"]:
        for family in plan["families"]:
            key = f"full_stage1_{fee}_{family}"
            if family == "soxx":
                frame, replay = market.copy(), _soxx_schedule(market, fee)
            else:
                frame = market.join(prepared["source_targets"][family].set_axis(market.index))
                replay = _simulate_on_policy_router(frame, models=open_models, recovery_models=recovery_models,
                    source_spec=recovery_spec, spec=replace(action_spec, transaction_cost_bps=float(fee)),
                    selected_model=released["action_summary"]["source_stage1"]["selected_model"], segment="reused_outer_diagnostic")
            native = native_intraday_reconciliation(frame, replay, provider_uri=prepared["provider_uri"],
                cost_bps=float(fee), account=float(plan["initial_cash"]), target_convention="pre_fee_account")
            authority = replay.copy()
            authority["strategy_return"] = native["reconciliation"].native_return.to_numpy(float)
            authority["equity"] = native["reconciliation"].native_nav.to_numpy(float) / plan["initial_cash"]
            books[key] = {"authority": validate_replay(authority), "independent_replay": replay, "native": native}
            recorder.save_objects(artifact_path="analysis/native", **{key: books[key]})
            recorder.log_metrics(completed_native_cases=len(books))
            print(f"native {len(books)}/{plan['maximum_native_cases']} {key}", flush=True)
    if len(books) != plan["maximum_native_cases"]:
        raise ValueError("blocked: full-Stage1 audit exceeds or omits its fixed account budget")
    comparison, annual = compare_native_books(books, window="full_stage1", costs=plan["costs_bps"],
        families=plan["families"], references=["soxx", "frozen_current", "base", "availability_control"])
    for index, row in comparison.iterrows():
        path = books[f"full_stage1_{row.cost_bps}_{row.family}"]["authority"]
        comparison.loc[index, "open_trade_days"] = int(path.execute_open_trade.sum())
        comparison.loc[index, "recovery_action_count"] = int(path.recovery_action.sum())
        comparison.loc[index, "average_target_cash_pct"] = float(100. * path.cash_weight.mean())
    selected = comparison.loc[comparison.family.eq(plan["selected_family"])]
    diagnostic_gates = {
        "net_return_above_soxx_both_costs": bool(selected.wealth_ratio_vs_soxx.gt(1.).all()),
        "observed_drawdown_within_30pct_both_costs": bool(selected.max_drawdown_pct.ge(-30.).all()),
        "return_above_refit_and_availability_controls_both_costs": bool(
            selected.wealth_ratio_vs_base.gt(1.).all() and selected.wealth_ratio_vs_availability_control.gt(1.).all()),
        "original_minute_development_gate_pass": bool(prepared["allocator_summary"]["incremental_development_gate"]),
        "independent_forward_gate_pass": False,
    }
    summary = {"experiment": plan["experiment"], "plan_sha256": plan_hash,
        "decision": "not_promoted_full_stage1_comparison_diagnostic", "production_enabled": False,
        "objective_achieved": False, "research_role": "comparison correction; no fit, tuning, or new selection",
        "selected_family": plan["selected_family"], "model_fits": 0, "native_cases": len(books),
        "comparison_layer": "complete released Stage1 allocator smoothing, open action/HOLD, risk clock, recovery and own net-fee ledger",
        "current_release_changed": False, "active_controller_functions_reused": True,
        "active_live_or_paper_fills_reproduced": False, "forward_qualified_sessions": 0,
        "outer_uses_reused_history": True, "as_of": plan["as_of"], "source_id": plan["source_id"],
        "data_authority": "yanxuan_nas", "provider_uri": str(prepared["provider_uri"]), "region": "us", "market": "all",
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id), "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "input_audit": prepared["audit"], "diagnostic_gates": diagnostic_gates,
        "old_30pct_reference_is_observed_backtest_not_guaranteed_stop": True,
        "allocator_only_parent_card": plan["allocator_card"], "parent_release": plan["parent_release"],
        "cost_and_accounting_contract": {"costs_bps": plan["costs_bps"], "each_cost_has_own_controller_cash_and_holdings": True,
            "target_convention": "pre_fee_account", "source_smoothing_is_released_virtual_gross_ledger": True,
            "cash_interest": 0, "native_share_units": "fractional adjusted-price units; no broker integer-share/dividend-cash qualification"},
        "qlib_docs_reference": ["https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/recorder.html", "https://qlib.readthedocs.io/en/latest/component/strategy.html"],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/market", "inputs/raw_predictions", "inputs/source_targets",
                "inputs/native_marks", "controller/frozen_models", "analysis/native/*", "analysis/comparison", "analysis/annual"]},
        "compact_metrics": comparison.to_dict("records")}
    recorder.save_objects(artifact_path="analysis", comparison=comparison, annual=annual, summary=summary)
    for row in comparison.itertuples():
        recorder.log_metrics(**{f"{row.family}_{row.cost_bps}_return_pct": row.return_pct,
                                 f"{row.family}_{row.cost_bps}_max_drawdown_pct": row.max_drawdown_pct})
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    recorder.set_tags(audit_status="complete")
    if _sha(root / "production/stage1_futu_sim_v1/runtime_release.json") != plan["runtime_release_sha256"]:
        raise ValueError("blocked: runtime release changed during comparison")
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    annual.to_csv(card / "regime_metrics.csv", index=False)
    pd.DataFrame(plan["legacy_registered_reference"]).to_csv(card / "legacy_reference.csv", index=False)
    prepared["allocator_only_comparison"].to_csv(card / "allocator_only_reference.csv", index=False)
    workflow = {"experiment": plan["experiment"], "provider_uri": summary["provider_uri"], "region": "us", "market": "all",
                "freq": "15min", "as_of": plan["as_of"], "recorder_uri": str(tracking),
                "segments": {"reused_outer_diagnostic": [plan["signal_start"], plan["as_of"]]},
                "model_selection": "reuse prior development selection; no selection in this audit", "artifact_policy": summary["artifact_policy"]}
    (card / "workflow.yaml").write_text(yaml.safe_dump(workflow, allow_unicode=True, sort_keys=False))
    entries = [{"path": str(p.relative_to(card)), "sha256": _sha(p), "bytes": p.stat().st_size}
               for p in sorted(artifacts.rglob("*")) if p.is_file()]
    atomic_json({"recorder_id": summary["recorder_id"], "entries": entries}, card / "verification_manifest.json")
    binding_path = card / "nas_binding.json"
    summary["nas_evidence"] = archive_nas_recorder(binding_path, workspace_root=root, card=card,
        recorder_id=summary["recorder_id"], experiment_id=summary["experiment_id"])
    atomic_json(summary, card / "summary.json")
    _write_conclusion(card, summary, comparison)
    return summary


def _write_conclusion(card, summary, comparison):
    lines = ["# 完整 Stage1 对照结论", "",
        "本次修正比较层级：先前分钟卡评估分配器单层，不能替代旧模拟盘完整策略。复用冻结分配器预测及现役动作/恢复模型，没有训练、调参或切换模拟盘。",
        f"固定历史诊断：{summary['input_audit']['first_trade']} 至 {summary['input_audit']['last_exit']}，{summary['input_audit']['sessions']} 个完整账户区间。每档费用独立持仓、净值峰值、回撤、风险时钟与 HOLD/恢复动作；Qlib 原生订单、现金、数量逐日核对。",
        "", "| 费用 | 分配器接完整 Stage1 | 净收益 | 最大回撤 | 相对 SOXX 终值 |", "|---|---|---:|---:|---:|"]
    for row in comparison.itertuples():
        lines.append(f"| {row.cost_bps}bp | {row.family} | {row.return_pct:.2f}% | {row.max_drawdown_pct:.2f}% | {row.wealth_ratio_vs_soxx:.4f} |")
    lines += ["", "旧卡开发期 10bp 最大回撤 −30.08%、25bp −31.88%；旧外层约 −26.92%/−27.32%。这些是已观察的历史结果，不是硬性保证。旧窗口、旧价格来源的数值保留在 legacy_reference.csv，不能直接与扩展窗口相减。", "",
        "旧卡 25bp 是固定 10bp 开盘/HOLD/恢复事件后重新扣费的压力测试（cost_stress_contract=fixed_target_hold_recovery_events_repriced_at_each_stage_net_nav），没有按 25bp 净值重新决定风控路径。本卡两档费用各自重算状态。两种费用口径必须区分。", "",
        "年度差异见 regime_metrics.csv。旧单模型与三 seed 重训基线的差异不能全部归因于分钟数据；core_path 必须同时与 base 和 availability_control 对照。",
        "分钟单层的历史改进需要经过完整控制器重新验证。两档费用下候选与对照的排序不同，不能据此重新选模。控制器成本敏感性见 cost_path_differences.csv，净值回撤与风险状态核对见 controller_state_diagnostics.csv。", "",
        "历史执行价格统一为 NAS 真实分钟及已公布复权事件构建的开盘/首小时结束标记。开盘及 10:30 close 是历史记账标记，不声明真实 09:31/10:31 成交、首次数据接收时点或现金分红资格。模型分钟输入仍为此前冻结的 raw NONE，未用本次复权系数重算特征。",
        f"诊断门：{json.dumps(summary['diagnostic_gates'], ensure_ascii=False)}。",
        "晋升：not_promoted_full_stage1_comparison_diagnostic；原分钟开发门已失败，独立合格前向为 0，objective_achieved=false。比较修正不能重新挑选候选或把复用历史变成新的样本外证据。",
        "下一步：结束此固定对照；保留失败路线。模型优化需要新的有效信息或预先冻结的独立前向对照，不能靠增加同一历史上的参数搜索。",
        f"Recorder：{summary['recorder_id']}。行情、大账户及模型引用留 NAS；Git 保留 notebook、瘦卡和代码。", ""]
    (card / "conclusion.md").write_text("\n".join(lines))


def supplement_stage1_cost_audit(*, root: Path, card: Path) -> dict:
    """Explain fee-sensitive controller paths from completed Recorder evidence.

    This only derives tables and metadata. It never fits or replays an account.
    """
    import qlib
    from qlib.workflow import R

    summary = json.loads((card / "summary.json").read_text())
    if summary.get("comparison_cost_audit"):
        return summary
    plan = json.loads((card / "plan.json").read_text())
    action_path = root / plan["parent_release"]["action"]["card_path"] / "summary.json"
    if _sha(action_path) != plan["parent_release"]["action"]["summary_sha256"]:
        raise ValueError("blocked: legacy fee contract identity changed")
    old = json.loads(action_path.read_text())
    qlib.init(provider_uri=summary["provider_uri"], region="us", kernels=1)
    paths, diagnostics, differences = {}, [], []
    with R.uri_context(summary["recorder_uri"]):
        recorder = R.get_recorder(experiment_id=summary["experiment_id"], recorder_id=summary["recorder_id"])
        comparison = recorder.load_object("analysis/comparison")
        for fee in plan["costs_bps"]:
            for family in plan["families"]:
                if family == "soxx":
                    continue
                path = recorder.load_object(f"analysis/native/full_stage1_{fee}_{family}")["authority"]
                previous = np.r_[1., path.equity.to_numpy(float)[:-1]]
                implied = np.maximum(0., 1. - previous / np.maximum.accumulate(previous))
                error = float(np.max(np.abs(implied - path.portfolio_drawdown.to_numpy(float))))
                if error > 1e-8:
                    raise ValueError("blocked: controller drawdown differs from its own native net NAV")
                emergency = path.portfolio_drawdown.ge(plan["recovery_spec"]["emergency_drawdown"])
                diagnostics.append({"cost_bps": fee, "family": family,
                    "own_native_NAV_drawdown_error": error, "risk_state_active_days": int(path.risk_state.ne(0).sum()),
                    "emergency_drawdown_days": int(emergency.sum()), "first_emergency_trade_date":
                    str(path.loc[emergency, "trade_date"].iloc[0].date()) if emergency.any() else None,
                    "average_effective_ending_beta": float((3. * path.ending_soxl_weight + path.ending_soxx_weight).mean()),
                    "average_ending_cash_pct": float(100. * path.ending_cash_weight.mean()),
                    "recovery_action_count": int(path.recovery_action.sum())})
                paths[(fee, family)] = path
        for family in plan["families"]:
            if family == "soxx":
                continue
            low, high = (paths[(fee, family)] for fee in plan["costs_bps"])
            if not low.trade_date.equals(high.trade_date):
                raise ValueError("blocked: cost paths have different native dates")
            different = low.risk_state.ne(high.risk_state) | low.risk_clock.ne(high.risk_clock)
            differences.append({"family": family, "costs_bps": "10/25", "different_risk_state_or_clock_days": int(different.sum()),
                "first_risk_state_or_clock_divergence_trade_date": str(low.loc[different, "trade_date"].iloc[0].date()) if different.any() else None,
                "different_open_target_event_days": int(low.execute_open_target.ne(high.execute_open_target).sum()),
                "different_beta_cap_days": int((low.current_beta_cap - high.current_beta_cap).abs().gt(1e-8).sum()),
                "source_allocator_targets_identical_across_costs": True,
                "risk_state_is_recomputed_per_cost_not_fixed_10bp_repricing": True})
        audit = {"legacy_cost_stress_contract": old["cost_stress_contract"],
                 "legacy_25bps_is_independent_controller_path": False,
                 "current_10_and_25bps_are_independent_controller_paths": True,
                 "own_controller_drawdown_checked_against_native_NAV": True,
                 "new_model_fits": 0, "new_native_cases": 0,
                 "derivation": "completed native accounts only; no parameter or model selection"}
        tables = pd.DataFrame(diagnostics), pd.DataFrame(differences)
        recorder.save_objects(artifact_path="analysis", controller_state_diagnostics=tables[0],
                              cost_path_differences=tables[1], comparison_cost_audit=audit)
        summary["comparison_cost_audit"] = audit
        summary.pop("nas_evidence", None)
        recorder.save_objects(artifact_path="analysis", summary=summary)
        artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
        recorder.log_metrics(registered_artifact_count=len([p for p in artifacts.rglob("*") if p.is_file()]))
    for table, name in zip(tables, ("controller_state_diagnostics.csv", "cost_path_differences.csv")):
        table.to_csv(card / name, index=False)
    entries = [{"path": str(p.relative_to(card)), "sha256": _sha(p), "bytes": p.stat().st_size}
               for p in sorted(artifacts.rglob("*")) if p.is_file()]
    atomic_json({"recorder_id": summary["recorder_id"], "entries": entries}, card / "verification_manifest.json")
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=summary["recorder_id"], experiment_id=summary["experiment_id"])
    atomic_json(summary, card / "summary.json")
    _write_conclusion(card, summary, comparison)
    return summary
