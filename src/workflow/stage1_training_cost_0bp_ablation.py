"""One fixed training-cost ablation on the complete Stage1 raw-account chain."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path) -> dict:
    import qlib
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder

    from src.analysis.integrated_allocation import DATES, paired_metrics, stationary_bootstrap_excess, validate_replay
    from src.analysis.stage1_domain_heads import _drawdown_pct, _period_coverage
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder
    from src.model.minute_pit_allocator import MinutePitAllocator
    from src.workflow.allocator_turnover_ablation import _initial_hash
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.minute_pit_allocator import _ensemble, _prediction_dates
    from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
    from src.workflow.stage1_domain_heads import OUTPUTS, _provider, prepare
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec

    root = Path(root).resolve()
    card = root / "research/stage1_training_cost_0bp_ablation_20261023"
    plan = json.loads((card / "plan.json").read_text())
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        previous = json.loads((card / "summary.json").read_text())
        if previous["plan_sha256"] != digest:
            raise ValueError("frozen plan changed")
        return previous
    assert plan["frozen"] and plan["candidate_training_cost_bps"] == 0
    assert plan["control_training_cost_bps"] == 10 and plan["unchanged_turnover_loss_weight"] == .006
    assert plan["accounts"]["evaluation_slippage_bps_now"] == [0]
    prepared = prepare(root=root, card=card)
    assert prepared["plan_sha256"] == digest

    # The original A_0 and matched B_0 are the current raw-corporate Stage1
    # evidence, not the older minute-fee account used in turnover ablation.
    control_ref = plan["source_control"]
    control_card = root / control_ref["card"]
    control_summary = json.loads((control_card / "summary.json").read_text())
    assert control_summary["recorder_id"] == control_ref["recorder_id"]
    assert control_summary["nas_evidence"]["sha256"] == control_ref["nas_archive_sha256"]
    assert _sha(control_card / "verification_manifest.json") == control_ref["verification_manifest_sha256"]
    assert json.loads((control_card / "nas_binding.json").read_text())["evidence"]["sha256"] == control_ref["nas_archive_sha256"]
    artifact_root = control_card / "qlib_recorder" / control_ref["experiment_id"] / control_ref["recorder_id"] / "artifacts"
    for name, expected in control_ref["artifacts"].items():
        assert _sha(artifact_root / name) == expected, name
    with R.uri_context(str((control_card / "qlib_recorder").resolve())):
        control_recorder = R.get_recorder(experiment_name=control_card.name, recorder_id=control_ref["recorder_id"])
        original_book = control_recorder.load_object("native/A_active_research_replay_0")
        matched_book = control_recorder.load_object("native/B_shared_0")
        original_path = control_recorder.load_object("paths/A_active_research_replay_0")
        matched_path = control_recorder.load_object("paths/B_shared_0")
    for book, path in ((original_book, original_path), (matched_book, matched_path)):
        assert book["audit"]["status"] == "passed"
        pd.testing.assert_frame_equal(book["authority"][list(DATES)], path["replay"][list(DATES)])
    state_path = card / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "plan_sha256": digest, "fit_attempts": [], "fits_done": [],
        "path_attempts": [], "paths_done": [], "native_attempts": [], "native_done": []}
    if state["plan_sha256"] != digest:
        raise ValueError("attempt state belongs to another plan")
    tracking = card / "qlib_recorder"
    kwargs = {"experiment_name": card.name, "uri": str(tracking)}
    if state.get("recorder_id"):
        kwargs.update(recorder_id=state["recorder_id"], resume=True)
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    training_rows = []
    with R.start(**kwargs):
        recorder = R.get_recorder()
        state.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(state, state_path)
        if not state.get("sources_saved"):
            recorder.save_objects(artifact_path="inputs", plan=plan, preflight=prepared["receipt"],
                                  source_audits=prepared["source_audits"], dataset=prepared["dataset"],
                                  auxiliary_dataset=prepared["auxiliary"], original_book=original_book,
                                  matched_book=matched_book)
            recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text())
            state["sources_saved"] = True
            atomic_json(state, state_path)
        for seed in plan["training"]["seeds"]:
            key = f"candidate_{seed}"
            control = prepared["sources"]["training"][f"models/base_outer_{seed}"]
            if control.extra_features or control.spec.training_cost_bps != 10 or control.spec.turnover_loss_weight != .006:
                raise ValueError("source model recipe differs")
            spec = {**asdict(control.spec), "root": str(root), "training_cost_bps": 0}
            candidate = MinutePitAllocator(base_features=control.base_features, extra_features=[],
                feature_masks=control.feature_masks, spec=spec, seed=seed, cutoff=str(control.cutoff),
                train_end=str(control.train_end.date()), minimum_observations=control.minimum_observations)
            identity = _initial_hash(candidate)
            if identity != _initial_hash(control):
                raise ValueError("candidate initialization differs")
            saved = prepared["sources"]["training"]["analysis/outer_predictions"][("base", seed)]
            check = _prediction_dates(control.predict(prepared["dataset"]))
            pd.testing.assert_index_equal(check.index, saved.index)
            np.testing.assert_allclose(check, saved, rtol=1e-6, atol=1e-7)
            predictions[("B_shared", seed)] = check
            if key in state["fits_done"]:
                candidate = recorder.load_object("models/" + key)
            else:
                if key in state["fit_attempts"] or len(state["fit_attempts"]) >= plan["budget"]["new_model_fit_attempts_max"]:
                    raise ValueError("fit attempt consumed or fixed budget exhausted")
                state["fit_attempts"].append(key)
                atomic_json(state, state_path)
                candidate.fit(prepared["dataset"], auxiliary_dataset=prepared["auxiliary"])
                for name in ("training_rows", "auxiliary_rows", "last_mature_label_at", "epochs", "checkpoint_policy"):
                    if candidate.audit[name] != control.audit[name]:
                        raise ValueError("paired training contract differs: " + name)
                if candidate.primary_scaler != control.primary_scaler or candidate.common_scaler != control.common_scaler:
                    raise ValueError("paired scaler differs")
                recorder.save_objects(artifact_path="models", **{key: candidate})
                state["fits_done"].append(key)
                atomic_json(state, state_path)
            prediction = _prediction_dates(candidate.predict(prepared["dataset"]))
            predictions[("candidate_cost0", seed)] = prediction
            recorder.save_objects(artifact_path="predictions", **{key: prediction})
            training_rows.append({"seed": seed, "initialization_sha256": identity,
                                  "control_training_cost_bps": 10, "candidate_training_cost_bps": 0,
                                  "turnover_loss_weight_both": .006, **candidate.audit})
            recorder.log_metrics(completed_model_fits=len(state["fits_done"]))
            print("completed fit", seed, flush=True)
        dates = pd.DatetimeIndex(prepared["market"].signal_date)
        returns = prepared["market"].set_index("signal_date")[["soxl_return", "soxx_return"]]
        ensembles = {family: _ensemble(predictions, family, plan["training"]["seeds"])
                     for family in ("B_shared", "candidate_cost0")}
        targets = {family: _smooth_weights(values.reindex(dates), spec=prepared["original"].spec,
            candidate=prepared["original"].candidate, asset_returns=returns)
            for family, values in ensembles.items()}
        np.testing.assert_allclose(targets["B_shared"],
            prepared["sources"]["action"]["inputs/source_targets"]["base"], rtol=1e-6, atol=1e-7)
        recorder.save_objects(artifact_path="analysis", training=pd.DataFrame(training_rows),
                              ensembles=ensembles, smoothed_targets=targets)
        recorder.log_params(plan_sha256=digest, candidate_training_cost_bps=0,
                            control_training_cost_bps=10, added_slippage_bps=0)

    # Raw Qlib validation reinitializes the provider. Resume this same Recorder.
    recorder.start_run()
    try:
        result = _complete(root=root, card=card, plan=plan, digest=digest, prepared=prepared,
                           recorder=recorder, state=state, state_path=state_path,
                           tracking=tracking, targets=targets, predictions=predictions,
                           training_rows=training_rows, original_book=original_book,
                           matched_book=matched_book)
        recorder.end_run(status=Recorder.STATUS_FI)
        return result
    except BaseException:
        recorder.end_run(status=Recorder.STATUS_FA)
        raise


def _complete(*, root, card, plan, digest, prepared, recorder, state, state_path,
              tracking, targets, predictions, training_rows, original_book, matched_book):
    from src.analysis.integrated_allocation import DATES, paired_metrics, stationary_bootstrap_excess, validate_replay
    from src.analysis.stage1_domain_heads import _drawdown_pct, _period_coverage
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_domain_heads import OUTPUTS, _provider
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec

    provider, manifest = _provider(root, prepared)
    recorder.save_objects(artifact_path="inputs", native_provider_manifest=manifest)
    recovery = IntradayRecoveryDirectAllocationSpec(root=root, **prepared["recovery_spec"])
    action = Stage1OpenActionValueSpec(root=root, **prepared["action_spec"])
    execution = {"initial_cash": plan["accounts"]["initial_cash"], "tariff": plan["tariff"]}
    frame = prepared["frame"].copy()
    for name in OUTPUTS:
        frame[name] = targets["candidate_cost0"][name].to_numpy(copy=True)
    path_key = "candidate_cost0_0"
    if path_key in state["paths_done"]:
        path = recorder.load_object("paths/" + path_key)
    else:
        if path_key in state["path_attempts"] or len(state["path_attempts"]) >= plan["budget"]["new_raw_path_attempts_max"]:
            raise ValueError("raw path attempt consumed or budget exhausted")
        state["path_attempts"].append(path_key)
        atomic_json(state, state_path)
        path = replay_stage1_raw(frame, point_prices=prepared["points"], schedule=prepared["schedule"],
            plan=execution, controller=prepared["controller"], recovery_spec=recovery,
            action_spec=action, selection=prepared["selection"], slippage_bps=0)
        recorder.save_objects(artifact_path="paths", **{path_key: path})
        state["paths_done"].append(path_key)
        atomic_json(state, state_path)
    if path_key in state["native_done"]:
        book = recorder.load_object("native/" + path_key)
    else:
        if path_key in state["native_attempts"] or len(state["native_attempts"]) >= plan["budget"]["new_native_account_attempts_max"]:
            raise ValueError("native account attempt consumed or budget exhausted")
        state["native_attempts"].append(path_key)
        atomic_json(state, state_path)
        book = native_raw_validate(path, provider_uri=provider, point_prices=prepared["points"],
            schedule=prepared["schedule"], tariff=plan["tariff"], cost=0,
            initial_cash=plan["accounts"]["initial_cash"])
        recorder.save_objects(artifact_path="native", **{path_key: book})
        state["native_done"].append(path_key)
        atomic_json(state, state_path)
    recorder.log_metrics(completed_model_fits=len(state["fits_done"]),
                         completed_native_accounts=len(state["native_done"]))
    books = {"candidate_cost0": book, "A_active_research_replay": original_book,
             "B_shared": matched_book,
             **{name: prepared["references"][(name, 0)] for name in
                ("SOXX", "strong9097", "additional27baa")}}
    authoritative = {name: validate_replay(value["authority"]) for name, value in books.items()}
    dates = authoritative["SOXX"][list(DATES)]
    assert len(dates) == 944
    initial_cash = plan["accounts"]["initial_cash"]
    baseline_points = pd.DatetimeIndex(books["SOXX"]["points"].stamp)
    for name, value in books.items():
        assert value["audit"]["status"] == "passed"
        pd.testing.assert_frame_equal(authoritative[name][list(DATES)], dates)
        pd.testing.assert_index_equal(pd.DatetimeIndex(value["points"].stamp), baseline_points)
        np.testing.assert_allclose(value["points"].iloc[0][["cash", "receivable", "q_soxl", "q_soxx", "nav"]].to_numpy(float),
                                   [initial_cash, 0, 0, 0, initial_cash], atol=1e-7, rtol=0)
    candidate, benchmark = authoritative["candidate_cost0"], authoritative["SOXX"]
    paired = {name: paired_metrics(candidate, authoritative[name]) for name in books if name != "candidate_cost0"}
    relative = np.log1p(candidate.strategy_return) - np.log1p(benchmark.strategy_return)
    windows = pd.DataFrame({"first_trade": dates.trade_date.iloc[:-125].to_numpy(),
                            "last_exit": dates.exit_date.iloc[125:].to_numpy(),
                            "relative_log": relative.rolling(126).sum().iloc[125:].to_numpy()})
    assert len(windows) == 819
    windows["relative_return_pct"] = 100 * np.expm1(windows.relative_log)
    positive_fraction = float(windows.relative_log.gt(0).mean())
    trade_dates = pd.DatetimeIndex(dates.trade_date)
    calendar = pd.DatetimeIndex(pd.to_datetime(prepared["daily_calendar_frame"].trade_date).dropna()).unique().sort_values()
    annual_labels, annual_coverage = _period_coverage(trade_dates, calendar, "Y")
    quarter_labels, quarter_coverage = _period_coverage(trade_dates, calendar, "Q")
    periods = []
    for kind, labels, coverage in (("year", annual_labels, annual_coverage),
                                   ("quarter", quarter_labels, quarter_coverage)):
        for period in labels.unique():
            mask = labels == period
            periods.append({"kind": kind, "period": str(period), **coverage[period],
                            **paired_metrics(candidate.loc[mask], benchmark.loc[mask])})
    periods = pd.DataFrame(periods)
    complete_years = periods.loc[periods.kind.eq("year") & periods.complete_period]
    assert len(complete_years) >= 2
    marks = books["candidate_cost0"]["points"]
    stamps = pd.DatetimeIndex(marks.stamp)
    closes = pd.Series(marks.nav.to_numpy(float), index=stamps).groupby(stamps.normalize()).last().reindex(trade_dates)
    assert closes.notna().all()
    close_dd = _drawdown_pct(closes, initial_cash)
    sampled_dd = _drawdown_pct(marks.nav, initial_cash)
    bargs = plan["gates"]["paired_uncertainty"]
    bootstrap = stationary_bootstrap_excess(candidate, benchmark,
        mean_block=bargs["mean_block"], resamples=bargs["resamples"],
        alpha=bargs["alpha"], seed=bargs["seed"])
    metrics = {"0bp": paired["SOXX"], "vs_original_A0": paired["A_active_research_replay"],
               "vs_matched_B0": paired["B_shared"], "vs_strong9097": paired["strong9097"],
               "close_max_drawdown_pct": close_dd, "sampled_point_max_drawdown_pct": sampled_dd,
               "positive_126_fraction": positive_fraction, "positive_126_windows": int(windows.relative_log.gt(0).sum()),
               "required_positive_126_windows": int(np.ceil(.7 * len(windows))),
               "complete_years": len(complete_years),
               "positive_complete_years": int(complete_years.relative_wealth.gt(1).sum()),
               "raw_candidate_cash_mean": float(np.mean([predictions[("candidate_cost0", seed)].cash_weight.mean() for seed in plan["training"]["seeds"]])),
               "funded_candidate_cash_mean": float(candidate.ending_cash_weight.mean()),
               "funded_A0_cash_mean": float(authoritative["A_active_research_replay"].ending_cash_weight.mean()),
               "funded_B0_cash_mean": float(authoritative["B_shared"].ending_cash_weight.mean())}
    gates = {"SOXX_wealth_gt1": paired["SOXX"]["relative_wealth"] > 1 + 1e-8,
             "available_MaxDD_le30": close_dd >= -30 and sampled_dd >= -30,
             "CVaR5_no_worse_SOXX": paired["SOXX"]["cvar5_pct"] >= paired["SOXX"]["benchmark_cvar5_pct"],
             "complete_years_min2": len(complete_years) >= 2,
             "positive_complete_years_ge70pct": float(complete_years.relative_wealth.gt(1).mean()) >= .7,
             "positive_126_windows_ge70pct": positive_fraction >= .7,
             "SOXX_block_lower_positive": bool(bootstrap["lower_bound_positive"])}
    passed = all(gates.values())
    decision = ("0bp_historical_gates_pass_need_separate_2_5bp_and_forward" if passed
                else "not_promoted_fixed_Stage1_training_cost0_0bp_failed")
    recorder.save_objects(artifact_path="analysis", paired=paired, periods=periods,
        rolling126=windows, bootstrap=bootstrap, metrics=metrics, gates=gates)
    summary = {"experiment": plan["experiment"], "decision": decision,
               "objective_achieved": False, "plan_sha256": digest,
               "new_model_fits": len(state["fits_done"]), "new_raw_paths": len(state["paths_done"]),
               "new_native_accounts": len(state["native_done"]), "reused_native_accounts": 5,
               "source_Stage1_A0_recorder_id": plan["source_control"]["recorder_id"],
               "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
               "recorder_uri": str(tracking), "metrics": metrics, "gates": gates,
               "bootstrap_SOXX": bootstrap,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*", "predictions/*",
                                                               "paths/*", "native/*", "analysis/rolling126"]},
               "limitations": ["All 2023-2026 market history is reused development feedback.",
                               "Full-minute continuous risk and independent forward remain unqualified.",
                               "0bp pass would still require separate own-state 2/5bp paths."]}
    for name, table in (("annual_metrics", periods.loc[periods.kind.eq("year")]),
                        ("quarterly_metrics", periods.loc[periods.kind.eq("quarter")]),
                        ("rolling_summary", pd.DataFrame([{"windows": len(windows),
                            "positive_windows": metrics["positive_126_windows"],
                            "positive_fraction": positive_fraction}]))):
        table.to_csv(card / (name + ".csv"), index=False)
    recorder.save_objects(artifact_path="analysis", summary=summary)
    state["status"] = "finished"
    atomic_json(state, state_path)
    (card / "workflow.yaml").write_text(
        "experiment: " + card.name + "\nplan: plan.json\n"
        "parent: research/stage1_domain_heads_20261018\n"
        "provider_uri: " + plan["provider_output"] + "\n"
        "cost_path: 0bp_added_slippage_with_IBKR_Tiered\n")
    binding = json.loads((root / plan["sources"]["training"]["card"] / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    atomic_json(binding, card / "nas_binding.json")
    nas = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
        card=card, recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(nas["nas_path"])).split()[0]
    assert remote == nas["sha256"]
    summary["NAS_evidence"] = nas
    summary["NAS_readback"] = {"verified": True, "sha256": remote}
    atomic_json(summary, card / "summary.json")
    conclusion = (
        "# Stage1 训练 10bp 成本代理移除：完整 0bp Tiered 账户\n\n"
        "完整冻结 Stage1 A_0 原股数账户为母本；同源三种子控制与 SOXX 原股数账户分别报告。"
        "只改训练阶段 10bp 换手成本代理为0，0.006 换手正则、输入、网络、动作、风险及恢复层不变；"
        "实际账户仍照收 Tiered 费用。\n\n"
        f"{len(state['fits_done'])} 次拟合、{len(state['native_done'])} 个新原生账户。"
        f"候选/SOXX 净财富比 {paired['SOXX']['relative_wealth']:.6f}，"
        f"收盘/可用点最大回撤 {close_dd:.3f}%/{sampled_dd:.3f}%，"
        f"CVaR5 {paired['SOXX']['cvar5_pct']:.3f}% 对 SOXX {paired['SOXX']['benchmark_cvar5_pct']:.3f}%。"
        f"完整年正向 {metrics['positive_complete_years']}/{metrics['complete_years']}，"
        f"126日正向 {metrics['positive_126_windows']}/{len(windows)}，"
        f"SOXX 区块下界 {bootstrap['daily_lower_bound']:.8f}。"
        f"原始候选现金均值 {metrics['raw_candidate_cash_mean']:.3%}，"
        f"实际候选/A_0/B_0 期末现金均值 {metrics['funded_candidate_cash_mean']:.3%}/"
        f"{metrics['funded_A0_cash_mean']:.3%}/{metrics['funded_B0_cash_mean']:.3%}。"
        f"固定门 {sum(gates.values())}/{len(gates)}，决定 `{decision}`。\n\n"
        "历史 2023–2026 已复用，独立前向与完整分钟风险仍未取得；"
        "若0bp失败，不做训练成本网格或2/5bp账户。"
        f"Recorder `{recorder.id}`；NAS SHA `{remote}`。\n")
    (card / "conclusion.md").write_text(conclusion)
    return summary
