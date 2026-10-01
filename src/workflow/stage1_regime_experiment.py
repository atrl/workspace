"""Preregistered regime-action comparison with Qlib evidence and explicit gates."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = "stage1_regime_deployment_20261001"
SOURCE = "stage1_economic_action_20260929"
CANONICAL_ROOT = Path("/Users/rziiiii/dev/workspace")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n")


def load_frozen_inputs(root):
    """Verify the retained input identity, then read it through its Qlib Recorder."""
    import qlib
    from qlib.workflow import R

    root = Path(root)
    qlib.init(provider_uri=str(root / "data/qlib/soxl_soxx_soxs"), region="us", kernels=1)
    card = root / "research" / SOURCE
    retained = json.loads((card / "retained_inputs.json").read_text())
    source_file = card / retained["path"]
    if _sha(source_file) != retained["sha256"]:
        raise ValueError("blocked: frozen regime experiment inputs changed")
    # Existing Qlib metadata identifies the canonical retained artifact location.
    R.set_uri(str(card / "qlib_recorder"))
    recorder = R.get_recorder(experiment_id=source_file.parents[3].name, recorder_id=source_file.parents[2].name)
    inputs = recorder.load_object("inputs/prepared_inputs.pkl")
    inputs["economic_models"] = recorder.load_object("analysis/training_results.pkl")[2022]["models"]
    return inputs, {**retained, "source_card": SOURCE, "recorder_id": recorder.id}


def development_gates(comparison, annual, candidate_names):
    """No reused diagnostic observations or omitted failing years enter selection."""
    index = comparison.loc[comparison.segment.eq("development_walk_forward")].set_index(["variant", "cost_bps"])
    yearly = annual.loc[annual.segment.eq("development_walk_forward") & annual.year.ge(2017)]
    rows = []
    for name in candidate_names:
        candidate = index.loc[(name, 10.0)]
        for reference in ("active_hold_baseline", "no_hold_reference"):
            baseline = index.loc[(reference, 10.0)]
            a = yearly[yearly.variant.eq(name)].set_index("year")
            b = yearly[yearly.variant.eq(reference)].set_index("year")
            if a.empty or not a.index.equals(b.index):
                raise ValueError("blocked: annual comparison coverage differs")
            ratio = float((a.terminal_wealth / b.terminal_wealth).min())
            checks = {
                "net_return_above_reference": candidate.return_pct > baseline.return_pct,
                "stress_return_above_reference": index.loc[(name, 25.0), "return_pct"] > index.loc[(reference, 25.0), "return_pct"],
                "max_drawdown_not_worse": candidate.max_drawdown_pct >= baseline.max_drawdown_pct - 1e-10,
                "worst_day_not_worse": candidate.worst_day_pct >= baseline.worst_day_pct - 1e-10,
                "cvar5_not_worse": candidate.cvar5_pct >= baseline.cvar5_pct - 1e-10,
                "worst_5d_not_worse": candidate.worst_rolling_5d_pct >= baseline.worst_rolling_5d_pct - 1e-10,
                "open_trade_days_reduced_8pct": candidate.open_trade_days <= .92 * baseline.open_trade_days,
                "turnover_not_increased": candidate.total_turnover <= baseline.total_turnover,
                "annual_wealth_ratio_at_least_0.985": ratio >= .985,
            }
            required = ["return_pct", "max_drawdown_pct", "worst_day_pct", "cvar5_pct", "worst_rolling_5d_pct", "open_trade_days", "total_turnover"]
            if not np.isfinite([*[float(candidate[k]) for k in required], *[float(baseline[k]) for k in required], ratio]).all():
                raise ValueError("blocked: nonfinite development metrics")
            rows.append({"variant": name, "reference": reference, **{k: bool(v) for k, v in checks.items()},
                         "minimum_annual_wealth_ratio": ratio, "passed_gate_count": sum(checks.values()), "all_pass": all(checks.values())})
    return pd.DataFrame(rows)


def select_candidate(gates, comparison, annual):
    dev = comparison.loc[comparison.segment.eq("development_walk_forward") & comparison.cost_bps.eq(25)].set_index("variant")
    yearly = annual.loc[annual.segment.eq("development_walk_forward") & annual.cost_bps.eq(25) & annual.year.ge(2017)]
    ranking = []
    for name, group in gates.groupby("variant", sort=True):
        ratios = []
        a = yearly[yearly.variant.eq(name)].set_index("year")
        for ref in ("active_hold_baseline", "no_hold_reference"):
            b = yearly[yearly.variant.eq(ref)].set_index("year")
            if a.empty or not a.index.equals(b.index) or not np.isfinite(np.r_[a.terminal_wealth, b.terminal_wealth]).all() or (b.terminal_wealth <= 0).any():
                raise ValueError("blocked: incomplete cost-stressed annual selection evidence")
            ratios.append(float((a.terminal_wealth / b.terminal_wealth).min()))
        if not np.isfinite(float(dev.loc[name, "return_pct"])):
            raise ValueError("blocked: nonfinite cost-stressed return")
        ranking.append({"variant": name, "historical_all_pass": bool(group.all_pass.all()),
                        "passed_gate_count": int(group.passed_gate_count.sum()),
                        "minimum_annual_wealth_ratio": float(group.minimum_annual_wealth_ratio.min()),
                        "minimum_stress_annual_wealth_ratio": min(ratios), "stress_return_pct": float(dev.loc[name, "return_pct"])})
    ranking = pd.DataFrame(ranking)
    eligible = ranking.loc[ranking.historical_all_pass]
    if not eligible.empty:
        chosen = eligible.sort_values(["minimum_stress_annual_wealth_ratio", "stress_return_pct", "variant"], ascending=[False, False, True]).iloc[0]
    else:
        chosen = ranking.sort_values(["passed_gate_count", "minimum_annual_wealth_ratio", "stress_return_pct", "variant"], ascending=[False, False, False, True]).iloc[0]
    return chosen.to_dict(), ranking


def conditional_regime_metrics(replays, regime_features):
    """Conditional return, and drawdown only within chronological regime episodes."""
    from qlib.contrib.evaluate import risk_analysis

    names = ["regime_up_probability", "regime_range_probability", "regime_down_probability"]
    features = regime_features.copy()
    if "signal_date" not in features:
        features = features.rename_axis("signal_date").reset_index()
    features["regime"] = features[names].idxmax(axis=1).map(dict(zip(names, ("up", "range", "down"))))
    output = []
    for (variant, segment, cost), replay in replays.groupby(["variant", "segment", "cost_bps"], sort=False):
        frame = replay.drop(columns=["regime"], errors="ignore").merge(features[["signal_date", "regime"]], on="signal_date", validate="one_to_one").sort_values("trade_date")
        frame["episode"] = frame.regime.ne(frame.regime.shift()).cumsum()
        for regime, group in frame.groupby("regime"):
            episode_dd = []
            for _, episode in group.groupby("episode"):
                wealth = np.r_[1.0, np.cumprod(1.0 + episode.strategy_return.to_numpy())]
                episode_dd.append(float((wealth / np.maximum.accumulate(wealth) - 1).min()))
            output.append({"variant": variant, "segment": segment, "cost_bps": cost, "regime": regime,
                           "sessions": len(group), "episodes": group.episode.nunique(),
                           "conditional_compounded_return_pct": 100 * (np.prod(1 + group.strategy_return) - 1),
                           "mean_daily_net_return_pct": 100 * group.strategy_return.mean(),
                           "turnover": group.turnover.sum(), "open_trade_days": int(group.execute_open_trade.sum()),
                           "worst_contiguous_episode_drawdown_pct": 100 * min(episode_dd),
                           "return_is_noncontiguous_conditional_statistic": True})
    return pd.DataFrame(output)


def _train_candidate(payload):
    import qlib
    from src.workflow.stage1_regime_action import RegimeActionSpec, train_regime_models, simulate_regime_policy
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec

    root, frame, inputs, config = payload
    qlib.init(provider_uri=str(Path(root) / "data/qlib/soxl_soxx_soxs"), region="us", kernels=1)
    spec = RegimeActionSpec(**config)
    common = dict(recovery_models=inputs["recovery_models"], source_spec=IntradayRecoveryDirectAllocationSpec(root=Path(root)), selected_model=inputs["selected_model"], spec=spec)
    print(f"Training preregistered {spec.name}", flush=True)
    trained = train_regime_models(frame, update_through_year=2022, cost_bps=10, **common)
    replay10 = simulate_regime_policy(frame, models=trained["models"], cost_bps=10, **common)
    np.testing.assert_allclose(replay10.strategy_return, trained["replay"].strategy_return, atol=1e-12, rtol=0)
    replay25 = simulate_regime_policy(frame, models=trained["models"], cost_bps=25, **common)
    print(f"Finished {spec.name}; causal replay reproduced", flush=True)
    return spec.name, trained, {10.0: replay10, 25.0: replay25}


def buy_hold_reference(frame, *, symbol, cost_bps):
    """Cash-funded direct benchmark on exactly the same Qlib-derived intervals."""
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    weights = np.array([0., 0., 1.])
    desired = np.array([1., 0., 0.] if symbol == "SOXL" else [0., 1., 0.])
    rows = []
    for count, row in enumerate(frame.itertuples()):
        target = fee_affordable_target(desired, pretrade_weights=weights, cost_bps=cost_bps) if count == 0 else None
        settlement = settle_target_session(weights, open_target_weights=target, asset_returns=[row.soxl_return, row.soxx_return, 0.], cost_bps=cost_bps)
        weights = settlement["ending_weights"]
        rows.append({"signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date,
                     "strategy_return": settlement["net_return"], "turnover": settlement["turnover"],
                     "execute_open_trade": count == 0, "recovery_action": False, "cash_weight": weights[2]})
    return pd.DataFrame(rows)


def run_regime_experiment(*, root, workers=2):
    import qlib
    from qlib.contrib.evaluate import risk_analysis
    from qlib.workflow import R
    from src.analysis.stage1_regime_validation import audit_first_hour_inputs, native_open_reconciliation
    from src.workflow.stage1_regime_action import RegimeActionSpec, load_regime_features, attach_regime_features, train_regime_models, simulate_regime_policy
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec, _metrics, _simulate_on_policy_router
    from src.workflow.stage1_economic_training import simulate_economic_policy
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts

    root = Path(root).resolve()
    card = root / "research" / EXPERIMENT
    plan = json.loads((card / "selection_plan.json").read_text())
    plan_hash = _sha(card / "selection_plan.json")
    followup_path = card / "followup_plan.json"
    followup = json.loads(followup_path.read_text()) if followup_path.exists() else None
    followup_hash = _sha(followup_path) if followup else None
    candidate_configs = plan["candidates"] + (followup["candidates"] if followup else [])
    inputs, source_audit = load_frozen_inputs(root)
    if source_audit["sha256"] != plan["source_input_sha256"]:
        raise ValueError("blocked: preregistered source identity mismatch")
    frame, timing_rows, timing_audit = audit_first_hour_inputs(inputs["frame"], provider_uri=root / "data/qlib/soxl_intraday_15min")
    features = load_regime_features(root / "data/qlib/soxl_soxx_soxs", start="2011-01-01", end="2026-09-25")
    feature_audit = features.attrs["regime_source"]
    frame = attach_regime_features(frame, features)
    development = frame.loc[frame.segment.eq("development_walk_forward")].copy()
    common = dict(recovery_models=inputs["recovery_models"], source_spec=IntradayRecoveryDirectAllocationSpec(root=root), selected_model=inputs["selected_model"])
    old_action = Stage1OpenActionValueSpec(root=root)
    replays, comparisons, annual, risks, trained = [], [], [], [], {}

    def record(replay, name, segment, cost):
        replay = replay.assign(variant=name, segment=segment, cost_bps=float(cost))
        replays.append(replay)
        metric = _metrics(replay, variant=name, segment=segment, cost_bps=cost)
        comparisons.append(metric)
        for year, section in replay.groupby(pd.to_datetime(replay.signal_date).dt.year):
            annual.append({"year": int(year), **_metrics(section, variant=name, segment=segment, cost_bps=cost)})
        values = pd.Series(replay.strategy_return.to_numpy(), index=pd.to_datetime(replay.trade_date))
        native = risk_analysis(values, freq="day", mode="product")
        if not np.isclose(100 * native.loc["max_drawdown", "risk"], metric["max_drawdown_pct"], atol=1e-9):
            raise ValueError("blocked: Qlib risk analysis disagrees with path metrics")
        risks.append(native.reset_index().assign(variant=name, segment=segment, cost_bps=cost))
        return replay

    for cost in (10.0, 25.0):
        for name, models in (("active_hold_baseline", inputs["legacy_models"]), ("no_hold_reference", {})):
            print(f"Reference {name} at {cost:g}bps", flush=True)
            replay = _simulate_on_policy_router(development, models=models, spec=replace(old_action, transaction_cost_bps=cost), segment="development_walk_forward", **common)
            record(replay, name, "development_walk_forward", cost)
        record(simulate_economic_policy(development, models=inputs["economic_models"], cost_bps=cost, **common), "economic_own_state_reference", "development_walk_forward", cost)
    baseline = comparisons[0]
    reproduction = bool(np.isclose(baseline["return_pct"], 170.7449966106111, atol=1e-7))
    if not timing_audit.get("invalid_retained_recovery_rows", 0) and not reproduction:
        raise ValueError("blocked: unchanged-source active baseline did not reproduce")
    prior_results = []
    if followup:
        with R.uri_context(str(CANONICAL_ROOT / "research" / EXPERIMENT / "qlib_recorder")):
            previous = R.get_recorder(experiment_id=followup["previous_trial_experiment_id"], recorder_id=followup["previous_trial_recorder_id"])
            previous_plan_path = Path(previous.get_local_dir()) / "artifacts/inputs/selection_plan.json"
            if _sha(previous_plan_path) != plan_hash:
                raise ValueError("blocked: first-round preregistered plan changed")
            previous_models = previous.load_object("training/development_models.pkl")
            previous_replays = previous.load_object("analysis/portfolio_replays.pkl")
        for config in plan["candidates"]:
            name = config["name"]
            prior_results.append((name, previous_models[name], {cost: previous_replays.loc[previous_replays.variant.eq(name) & previous_replays.segment.eq("development_walk_forward") & previous_replays.cost_bps.eq(cost)].copy() for cost in (10., 25.)}))
    new_configs = followup["candidates"] if followup else plan["candidates"]
    payloads = [(str(root), development, {k: inputs[k] for k in ("recovery_models", "selected_model")}, config) for config in new_configs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_train_candidate, payloads))
    else:
        results = [_train_candidate(payload) for payload in payloads]
    for name, fitted, paths in [*prior_results, *results]:
        trained[name] = fitted
        for cost, path in paths.items():
            record(path, name, "development_walk_forward", cost)
    comparison_df, annual_df = pd.DataFrame(comparisons), pd.DataFrame(annual)
    gates = development_gates(comparison_df, annual_df.loc[annual_df.cost_bps.eq(10)], list(trained))
    selected, ranking = select_candidate(gates, comparison_df, annual_df)
    _json(card / "selection_seal.json", {"plan_sha256": plan_hash, "followup_plan_sha256": followup_hash, "selection_uses_outer_test": False, "selected": selected})
    print(f"Selection sealed before diagnostic evaluation: {selected}", flush=True)
    selected_spec = RegimeActionSpec(**next(c for c in candidate_configs if c["name"] == selected["variant"]))
    expanded = train_regime_models(frame, spec=selected_spec, update_through_year=2025, cost_bps=10, **common)
    outer = frame.loc[~frame.segment.eq("development_walk_forward")].copy()
    for cost in (10.0, 25.0):
        for name, models in (("active_hold_baseline", inputs["legacy_models"]), ("no_hold_reference", {})):
            record(_simulate_on_policy_router(outer, models=models, spec=replace(old_action, transaction_cost_bps=cost), segment="outer_reused_diagnostic", **common), name, "outer_reused_diagnostic", cost)
        for suffix, models in (("frozen2022", trained[selected["variant"]]["models"]), ("expanded2025", expanded["models"])):
            record(simulate_regime_policy(outer, models=models, spec=selected_spec, cost_bps=cost, **common), f"{selected['variant']}_{suffix}", "outer_reused_diagnostic", cost)
    comparison_df, annual_df = pd.DataFrame(comparisons), pd.DataFrame(annual)
    all_replays = pd.concat(replays, ignore_index=True)
    for segment, market in (("development_walk_forward", development), ("outer_reused_diagnostic", outer)):
        for symbol in ("SOXL", "SOXX"):
            for cost in (10., 25.):
                record(buy_hold_reference(market, symbol=symbol, cost_bps=cost), f"{symbol}_buy_hold", segment, cost)
    comparison_df, annual_df = pd.DataFrame(comparisons), pd.DataFrame(annual)
    all_replays = pd.concat(replays, ignore_index=True)
    regime_metrics = conditional_regime_metrics(all_replays, features)
    # Qlib engine verifies a continuous open-only interval before recovery coverage.
    native_path = all_replays.loc[all_replays.variant.eq(selected["variant"]) & all_replays.segment.eq("development_walk_forward") & all_replays.cost_bps.eq(10) & pd.to_datetime(all_replays.trade_date).lt("2018-12-31")].copy()
    native_frame = development.loc[development.signal_date.isin(native_path.signal_date)].copy()
    native = native_open_reconciliation(native_frame, native_path, provider_uri=root / "data/qlib/soxl_soxx_soxs", cost_bps=10., account=100000.)
    if _sha(card / "selection_plan.json") != plan_hash:
        raise ValueError("blocked: registered plan changed during experiment")
    if followup and _sha(followup_path) != followup_hash:
        raise ValueError("blocked: registered followup plan changed during experiment")
    canonical_card = CANONICAL_ROOT / "research" / EXPERIMENT
    uri = canonical_card / "qlib_recorder"
    uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str(root / "data/qlib/soxl_soxx_soxs"), region="us", kernels=1)
    R.set_uri(str(uri))
    with R.start(experiment_name=EXPERIMENT):
        recorder = R.get_recorder()
        R.log_params(plan_sha256=plan_hash, selection_uses_outer_test=False, source_input_sha256=source_audit["sha256"], source_as_of=plan["source_as_of"], selected=selected["variant"])
        R.log_metrics(historical_gate_pass=float(selected["historical_all_pass"]), candidate_count=len(trained), input_rows=len(frame))
        R.save_objects(artifact_path="inputs", **{"prepared_inputs.pkl": inputs, "audited_regime_frame.pkl": frame, "regime_features.pkl": features, "first_hour_audit.pkl": timing_rows})
        R.save_objects(artifact_path="training", **{"development_models.pkl": trained, "expanded_model_training.pkl": expanded})
        R.save_objects(artifact_path="analysis", **{"portfolio_replays.pkl": all_replays, "comparison.pkl": comparison_df, "annual.pkl": annual_df, "regime_metrics.pkl": regime_metrics, "development_gates.pkl": gates, "qlib_risk.pkl": pd.concat(risks, ignore_index=True)})
        R.save_objects(artifact_path="portfolio_analysis", **{"native_reconciliation.pkl": native})
        R.save_objects(artifact_path="models", **{"selected_action.pkl": expanded["final_model"], "selected_spec.pkl": selected_spec, "recovery_models.pkl": inputs["recovery_models"], "recovery_selection.pkl": inputs["selected_model"]})
        recorder.log_artifact(str(card / "selection_plan.json"), artifact_path="inputs")
        recorder.log_artifact(str(card / "selection_seal.json"), artifact_path="inputs")
        if followup:
            recorder.log_artifact(str(followup_path), artifact_path="inputs")
        recdir = Path(recorder.get_local_dir())
        pd.testing.assert_frame_equal(comparison_df, recorder.load_object("analysis/comparison.pkl"))
    _remove_qlib_code_snapshot_artifacts(recdir)
    tables = {"backtest_benchmark_comparison.csv": comparison_df, "annual_metrics.csv": annual_df, "development_gates.csv": gates, "model_selection.csv": ranking, "regime_metrics.csv": regime_metrics, "risk_analysis.csv": pd.concat(risks, ignore_index=True), "training_diagnostics.csv": expanded["diagnostics"]}
    for filename, table in tables.items():
        table.to_csv(card / filename, index=False)
    source_files = ["src/workflow/stage1_regime_action.py", "src/workflow/stage1_regime_experiment.py", "src/analysis/stage1_regime_validation.py"]
    summary = {"experiment": EXPERIMENT, "decision": "not_promoted_pending_continuous_forward_validation" if selected["historical_all_pass"] else "not_promoted_historical_gates_failed", "selected": selected,
               "source_as_of": plan["source_as_of"], "provider_uri": "data/qlib/soxl_soxx_soxs", "region": "us", "market": "all", "data_range": [str(frame.signal_date.min().date()), str(frame.signal_date.max().date())],
               "plan_sha256": plan_hash, "followup_plan_sha256": followup_hash, "previous_trial_recorder_id": followup["previous_trial_recorder_id"] if followup else None, "selection_uses_outer_test": False, "baseline_reproduction_pass": reproduction, "recorder_id": recorder.id, "recorder_uri": str(uri), "recorder_local_dir": str(recdir), "recorder_relative_dir": str(recdir.relative_to(canonical_card)),
               "allocation_source": inputs["allocation_source"], "source_audit": source_audit, "feature_audit": feature_audit, "first_hour_audit": timing_audit, "native_qlib_audit": native.get("audit", {}),
               "selected_spec": asdict(selected_spec), "final_model_training_end": "2025-12-31", "dataset_audit": expanded["dataset_audit"], "source_sha256": {path: _sha(root / path) for path in source_files},
               "model_artifacts": {key: {"path": str((recdir / f"artifacts/models/{key}.pkl").relative_to(canonical_card)), "sha256": _sha(recdir / f"artifacts/models/{key}.pkl")} for key in ("selected_action", "selected_spec", "recovery_models", "recovery_selection")},
               "sealed_source_objects": {path: _sha(recdir / "artifacts" / path) for path in ("models/selected_action.pkl", "models/selected_spec.pkl", "inputs/prepared_inputs.pkl", "training/expanded_model_training.pkl")},
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["inputs/*.pkl", "training/*.pkl", "analysis/portfolio_replays.pkl", "portfolio_analysis/native_reconciliation.pkl", "models/*.pkl"]},
               "promotion": {"historical_pass": bool(selected["historical_all_pass"]), "native_reconciliation_pass": native.get("audit", {}).get("status") == "passed", "new_continuous_forward_pass": False, "broker_release_changed": False, "real_execution": False},
               "limitations": ["All2023-2026 observations were previously reviewed; reused diagnostic only.", "Historical replay uses session open, not actual09:31 executable quote; native reconciliation does not remove this limitation.", "No first-hour recovery history before2019; missing bars do not fabricate recovery.", "Action supervision is a one-session reset-cost proxy, not a multiday controller optimum.", "Regime probabilities are smooth causal state scores, not calibrated future probabilities.", "Continuous-weight evidence does not demonstrate integer-share broker fills.", "No candidate is promoted before new continuous forward evidence."],
               "qlib_docs_reference": ["https://qlib.readthedocs.io/en/latest/component/data.html", "https://qlib.readthedocs.io/en/latest/component/recorder.html", "https://qlib.readthedocs.io/en/latest/component/strategy.html"]}
    _json(card / "summary.json", summary)
    _json(card / "selected_model.json", {"candidate": selected["variant"], "spec": asdict(selected_spec), "historical_all_pass": selected["historical_all_pass"], "production_enabled": False, "recorder_id": recorder.id})
    _json(card / "data_audit.json", {"source": source_audit, "features": feature_audit, "first_hour": timing_audit, "native": native.get("audit", {})})
    import yaml
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": EXPERIMENT, "provider_uri": summary["provider_uri"], "region": "us", "recorder_uri": str(uri), "segments": plan["evaluation"], "model_selection": "selection_plan.json", "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    lines = ["# 区间感知调仓与部署验证", "", f"完成{len(trained)}个有界共享动作模型的年度滚动训练；首轮四个，后续三个仅取消普通软目标降beta的强制执行，硬风险限制不变。两轮选择均在各自 outer 诊断前封存；这是自适应开发，不是新的独立测试。选中 {selected['variant']}，历史门全部通过={selected['historical_all_pass']}。", "", "开发期2016–2022，10bp扣费后的连续持仓路径："]
    for row in comparison_df.loc[comparison_df.segment.eq("development_walk_forward") & comparison_df.cost_bps.eq(10)].itertuples():
        lines.append(f"- {row.variant}：收益{row.return_pct:.2f}%，MaxDD {row.max_drawdown_pct:.2f}%，CVaR5 {row.cvar5_pct:.2f}%，调仓{row.open_trade_days}日，换手{row.total_turnover:.3f}。")
    lines += ["", "分上行/震荡/下行的条件收益、换手和连续区间回撤见regime_metrics.csv；不能把不连续区间收益当成可直接交易的独立策略。25bp压力从各自净持仓重新走完整路径。", "", "2023–2026仅作复用诊断；最终动作模型信息截至2025年末。历史开盘价与09:31真实报价的差异、新候选连续前向表现和整数股执行仍须单独验证。", "", f"晋升决定：{summary['decision']}。保留每一项失败门；准备独立paper观察，不据此替换原SIM或下真实订单。", "", "下一步：以冻结模型、候选自身持仓和实际观测报价累计连续前向证据。", f"QLib recorder：{recorder.id}。"]
    (card / "conclusion.md").write_text("\n".join(lines) + "\n")
    return summary
