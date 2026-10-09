"""Zero-fit, frozen-forecast comparison of benchmark terminal valuation."""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import traceback

import numpy as np
import pandas as pd

from src.analysis.excess_resolution import apply_excess_resolution
from src.backtest.benchmark_relative_replay import replay_fixed_target
from src.backtest.horizon_matched_allocation import replay_horizon_allocation
from src.core.recorder_artifacts import save_compressed_object
from src.core.recorder_sources import load_verified_recorder_objects
from src.data.integrated_allocation import frame_sha256
from src.data.legacy_execution_view import load_legacy_execution_view
from src.data.oos_horizon_inputs import prepare_horizon_inputs
from src.model.benchmark_terminal_value import direct_terminal_value, transform_benchmark_terminal_inputs
from src.workflow.benchmark_evidence import evaluate_registered_paths
from src.workflow.benchmark_relative_research import _sha, _json, _same_recorder, _native_path
from src.workflow.oos_uncertainty_horizon import _source_hashes as inherited_hashes, _bundle_identity


PARENT_ARTIFACTS = ("inputs/frame.pkl", "forecast/forecasts.pkl", "analysis/paths.pkl", "models/annual_fits.pkl")
NEW_SOURCES = ("src/workflow/benchmark_terminal_experiment.py", "src/model/benchmark_terminal_value.py",
    "src/analysis/excess_resolution.py", "src/core/recorder_sources.py", "src/workflow/benchmark_evidence.py")
ASSETS = ("soxl", "soxx", "cash")


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _validate_plan(plan):
    expected = [{"name": "selected_original", "role": "frozen_forecast_control"},
                {"name": "benchmark_terminal", "role": "registered_terminal_valuation_candidate"}]
    _require(plan["schema_version"] == "benchmark_terminal_value_v1"
        and plan["registered_before_candidate_portfolio_results"] is True
        and type(plan["model_fit_budget"]) is int and plan["model_fit_budget"] == 0
        and type(plan["candidate_budget"]) is int and plan["candidate_budget"] == 1
        and plan["variants"] == expected and plan["costs_bps"] == [10., 25.]
        and type(plan["rebalance_sessions"]) is int and plan["rebalance_sessions"] == 5
        and type(plan["forecast_horizon_sessions"]) is int and plan["forecast_horizon_sessions"] == 5
        and plan["robust_radius"] == 1. and type(plan["robust_radius"]) is not bool
        and plan["benchmark"] == "SOXX" and plan["evidence_kind"] == "reused_history"
        and plan["numerical_log_excess_floor"] == 1e-8
        and plan["development_years"] == list(range(2016, 2023))
        and all(type(year) is int for year in plan["development_years"])
        and type(plan["gates"]["family_size"]) is int and plan["gates"]["family_size"] == 1
        and plan["parent"]["forecast_variant"] == "validation_selected"
        and set(plan["parent"]["artifact_sha256"]) == set(PARENT_ARTIFACTS)
        and plan["valuation"]["actual_terminal_trades"] is False,
        "registered terminal-valuation protocol differs")
    registered = pd.Timestamp(plan["registered_at_utc"])
    _require(registered.tzinfo is not None and registered <= pd.Timestamp.now(tz="UTC"),
             "invalid protocol registration time")


def _hashes(root):
    values = inherited_hashes(root)
    for name in NEW_SOURCES:
        module = importlib.import_module(name[:-3].replace("/", "."))
        _require(Path(module.__file__).resolve() == (root / name).resolve(), "imported source differs from declared root")
        values[name] = _sha(root / name)
    return values


def _load_parent(root, plan):
    parent = plan["parent"]
    return load_verified_recorder_objects(root / parent["card"], expected_recorder_id=parent["recorder_id"],
        artifacts=list(PARENT_ARTIFACTS), expected_artifact_sha256=parent["artifact_sha256"])


def _frozen_inputs(bundle, objects, plan):
    """Bind the stored model forecasts to exact source rows without fitting."""
    frame = bundle["frame"]
    try:
        pd.testing.assert_frame_equal(frame, objects["inputs/frame.pkl"], check_exact=True)
    except AssertionError as error:
        raise ValueError("blocked: reconstructed source differs from frozen parent frame") from error
    mask = (frame.index >= pd.Timestamp(plan["signal_start"])) & frame.exit_date.notna()
    market = frame.loc[mask].copy()
    forecast = objects["forecast/forecasts.pkl"][plan["parent"]["forecast_variant"]]
    rows, uncertainty = forecast["rows"], np.asarray(forecast["Vmean"], dtype=float)
    _require(not market.empty and rows.index.equals(market.index) and not rows.index.has_duplicates
             and rows.index.is_monotonic_increasing and uncertainty.shape == (len(market), 2, 2),
             "frozen forecast dates or uncertainty alignment differs")
    for saved, actual in (("signal_date", "signal_date"), ("forecast_available_at", "decision_at"),
                          ("label_available_at", "h5_available_at")):
        try:
            pd.testing.assert_series_equal(rows[saved], market[actual], check_exact=True, check_names=False)
        except AssertionError as error:
            raise ValueError("blocked: frozen forecast clocks differ from source") from error
    _require(rows.model_year.eq(rows.index.year).all(), "frozen annual forecast model year differs")
    models = objects["models/annual_fits.pkl"]
    for year in sorted(rows.index.year.unique()):
        model = models[int(year)][plan["parent"]["forecast_variant"]]
        cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        _require(model["model_year"] == year and pd.Timestamp(model["model_available_at"]) == cutoff
                 and pd.Timestamp(model["audit"]["last_label_available_at"]) < cutoff,
                 "parent annual model does not meet its frozen maturity cutoff")
    means = rows[["predicted_soxl", "predicted_soxx"]].to_numpy(float)
    sigma = np.asarray(bundle["covariance"], dtype=float)[np.flatnonzero(mask)].copy()
    _require(np.isfinite(means).all() and (means > -1).all() and np.isfinite(uncertainty).all()
             and sigma.shape == (len(market), 2, 2) and np.isfinite(sigma).all(), "incomplete frozen solver inputs")
    return market, means.copy(), uncertainty.copy(), sigma, {
        "model_fits": 0, "forecast_variant": plan["parent"]["forecast_variant"],
        "forecast_rows": len(rows), "forecast_rows_sha256": frame_sha256(rows),
        "parent_frame_exact_match": True, "prediction_clocks_exact_match": True,
        "unknown_terminal_h5_predictions_retained": int(rows.label_available_at.isna().sum()),
        "annual_model_inventory": sorted(map(int, rows.index.year.unique()))}


def verify_terminal_valuations(path, details, original_means, transformed, *, cost_bps, daily_covariances):
    """Verify solver values using independently reconstructed funded holdings."""
    q = float(transformed["wealth_scale"])
    expected = np.flatnonzero(path.optimizer_called).tolist()
    _require([item["row_number"] for item in details] == expected,
             "terminal valuation solve dates must cover each rebalance exactly once")
    records = []
    for item in details:
        i, solution = item["row_number"], item["solution"]
        row = path.iloc[i]
        _require(item["signal_date"] == row.signal_date and item["decision_at"] == row.decision_at,
                 "terminal solve dates differ from actual decisions")
        weights = row[[f"open_{asset}_weight" for asset in ASSETS]].to_numpy(float)
        funded = (1. - float(row.fees)) * weights
        _require(np.isfinite(funded).all() and (funded >= 0).all(), "funded terminal values must be canonical nonnegative holdings")
        post_error = float(np.max(np.abs(funded - np.asarray(solution["post_fee_values"], float))))
        _require(post_error <= 1e-12, "solver and independent funded terminal holdings differ")
        _require(np.array_equal(solution["expected_returns_H"], transformed["transformed_mu"][i]),
                 "terminal solver did not consume the exact transformed mean")
        _require(np.allclose(solution["mean_error_covariance_H"], transformed["transformed_covariance"][i], rtol=0, atol=1e-12)
                 and np.allclose(solution["covariance_daily"], daily_covariances[i], rtol=0, atol=1e-12),
                 "transformed mean uncertainty or unchanged daily risk covariance differs")
        direct = direct_terminal_value(funded, original_means[i], cost_bps)
        lhs = q * float(solution["objective"])
        rhs = direct - q * float(solution["robust_penalty"])
        error = abs(lhs - rhs)
        _require(np.isfinite([direct, lhs, rhs]).all() and error <= 1e-12,
                 "scaled solver objective differs from direct benchmark terminal value")
        _require(abs(float(row.fees) - float(solution["fees"])) <= 1e-12
                 and abs(float(row.fees) - float(row.turnover) * cost_bps / 10000.) <= 1e-12,
                 "realized fees differ from current funded trades")
        records.append({"signal_date": row.signal_date, "row_number": int(i), "cost_bps": float(cost_bps),
            "direct_terminal_value": direct, "scaled_objective": lhs, "scaled_robust_penalty": q * float(solution["robust_penalty"]),
            "objective_identity_error": error, "post_fee_values_error": post_error,
            "current_realized_fee": float(row.fees), "future_conversion_fee_debited": False})
    _require(len(records) == int(path.optimizer_called.sum()), "terminal valuation lacks a solve record")
    return pd.DataFrame(records)


def compare_parent_control(path, parent, *, cost_bps):
    for column in ("signal_date", "trade_date", "exit_date", "execute_open_target", "optimizer_called"):
        _require(path[column].equals(parent[column]), "original-control parent calendar or cadence differs")
    columns = ["strategy_return", "fees", "turnover", "equity",
        *[f"{prefix}_{asset}_weight" for prefix in ("pretrade", "open", "ending") for asset in ASSETS]]
    first, second = path[columns].to_numpy(float), parent[columns].to_numpy(float)
    _require(np.isfinite(first).all() and np.isfinite(second).all(), "nonfinite parent control values")
    error = float(np.max(np.abs(first - second)))
    _require(error <= 1e-10, "original frozen-forecast control differs from parent path")
    return {"cost_bps": float(cost_bps), "maximum_difference": error, "tolerance": 1e-10,
            "exact_match": bool(np.array_equal(first, second)), "passed": True}


def resolve_evidence(tables, raw, paths, plan):
    """Retain raw qualification and tighten only positive-effect classification."""
    name = "benchmark_terminal"
    years = plan["development_years"]
    dev = lambda frame: frame.loc[frame.signal_date.dt.year.isin(years)]
    resolved = apply_excess_resolution(raw[name],
        {cost: dev(paths[(name, cost)]) for cost in plan["costs_bps"]},
        {cost: dev(paths[("soxx", cost)]) for cost in plan["costs_bps"]},
        log_excess_floor=plan["numerical_log_excess_floor"])
    result = {key: value.copy(deep=True) for key, value in tables.items()}
    for filename, key in (("gates.csv", "gates"), ("rolling_summary.csv", "rolling_summary"),
                          ("bootstrap_analysis.csv", "bootstrap")):
        result["raw_" + filename] = tables[filename].copy(deep=True)
        result[filename] = resolved[key].assign(strategy=name)
    # Full strategy/benchmark annual returns stay intact; candidate-only
    # resolved classification is an explicitly separate small sidecar.
    result["resolved_annual_metrics.csv"] = resolved["annual"].assign(strategy=name)
    result["resolved_metrics.csv"] = resolved["metrics"].assign(strategy=name)
    for key in ("metrics", "annual", "bootstrap"):
        columns = [c for c in raw[name][key] if c != "lower_bound_positive"]
        pd.testing.assert_frame_equal(resolved[key][columns], raw[name][key][columns], check_exact=True)
    return result, {name: resolved}


def _conclusion(summary, tables):
    lines = ["# 基准终值计价实验", "", "本轮没有训练模型：从已封存父Recorder读取相同年度选择预测，仅改变H5规划的终值计价。两种方案独立扣除当次真实交易费用，并通过Qlib原生持仓估值核验。", "",
        "25bp结果如下；开发期为完整2016–2022信号年，后段仍是已复用历史：", ""]
    for row in tables["backtest_benchmark_comparison.csv"].query("cost_bps==25").itertuples():
        lines.append(f"- {row.window}/{row.strategy}：收益{row.return_pct:.2f}%，相对SOXX财富{row.relative_wealth:.6f}，最大回撤{row.max_drawdown_pct:.2f}%，CVaR5 {row.cvar5_pct:.2f}%，换手{row.turnover:.4f}。")
    lines += ["", f"原始符号门通过：{summary['raw_historical_gate_pass']}；排除1e-8以内累计对数超额后的门通过：{summary['historical_gate_pass']}。收益、点估计和置信区间保持原值；原始/有效正年及滚动胜率分别保留。", "",
        "终值转换仅是规划代理，不发生未来清仓交易，也没有把假设的未来转换费用再次扣入真实净值。逐次求解的q倍目标已与直接终值减歧义惩罚核对；风险协方差、费用、初始现金、预测和持有频率不变。", "",
        "年度、滚动及压力差异见小型侧表；源数据口径、16条次来源行情、连续份额、历史开盘及现金零收益假设仍有限制。估计风险预算只在调仓时约束，中间HOLD漂移单独记录。", "",
        "不晋升。所有历史已被复用，本批完成不代表稳定超额目标完成。下一步依据保留的预测、实际动作、尾部与计价差异独立诊断，再登记下一机制。", "",
        f"Recorder：{summary['recorder_id']}；父Recorder：{summary['parent_recorder_id']}；as_of={summary['as_of']}。"]
    return "\n".join(lines) + "\n"


def run_benchmark_terminal_experiment(*, source_root, evidence_root, card_dir, recorder_uri):
    import qlib
    from qlib.workflow import R

    source_root, evidence_root, card = (Path(p).resolve() for p in (source_root, evidence_root, card_dir))
    _require(not (card / "summary.json").exists(), "completed terminal card is immutable")
    plan = json.loads((card / "plan.json").read_text())
    _validate_plan(plan)
    source_hashes, plan_hash = _hashes(source_root), _sha(card / "plan.json")
    mechanism_hash = _sha(card / "mechanism.md")
    bundle = prepare_horizon_inputs(evidence_root, plan)
    provider, provider_audit = load_legacy_execution_view(evidence_root)
    _require(provider == (evidence_root / plan["source"]["execution_provider_uri"]).resolve(), "execution provider differs")
    objects, parent_audit = _load_parent(evidence_root, plan)
    market, means, mean_error, sigma, frozen_audit = _frozen_inputs(bundle, objects, plan)
    identity = _bundle_identity(bundle)
    uri = str(Path(recorder_uri).resolve())
    binding = {"plan_sha256": plan_hash, "source_sha256": source_hashes, "provider_audit": provider_audit,
        "data_audit": bundle["audit"], "bundle_identity": identity, "verified_parent": parent_audit,
        "frozen_forecast": frozen_audit, "source_root": str(source_root), "read_only_evidence_root": str(evidence_root),
        "card_dir": str(card), "recorder_uri": uri, "qlib_version": qlib.__version__, "model_fits": 0,
        "mechanism_sha256": mechanism_hash}
    _json(card / "run_binding.json", binding)
    with R.start(experiment_name=plan["experiment"], uri=uri):
        recorder = R.get_recorder()
        recorder_id = recorder.id
        recorder.log_params(plan_sha256=plan_hash, model_fit_budget=0, candidate_budget=1, evidence_kind="reused_history")
        recorder.save_objects(artifact_path="inputs", **{"plan.pkl": plan, "binding.pkl": binding, "frame.pkl": bundle["frame"]})
        recorder.save_objects(artifact_path="forecast", **{"frozen_parent.pkl": objects["forecast/forecasts.pkl"][plan["parent"]["forecast_variant"]]})
        for name in source_hashes:
            recorder.log_artifact(str(source_root / name), artifact_path="source/" + str(Path(name).parent))
    paths, native, solver_details, valuation_details, transforms = {}, {}, {}, {}, {}
    solver_rows, valuation_rows, control_rows = [], [], []
    stage = "staged_frozen_inputs"
    try:
        for cost in plan["costs_bps"]:
            stage = f"native_soxx_{cost:g}bps"
            raw = replay_fixed_target(market, target=[0., 1., 0.], cost_bps=cost, rebalance_sessions=5, buy_and_hold=True)
            paths[("soxx", cost)], native[("soxx", cost)] = _native_path(market, raw, provider, cost, plan["initial_cash"])
            for name in ("selected_original", "benchmark_terminal"):
                stage = f"solve_{name}_{cost:g}bps"
                if name == "benchmark_terminal":
                    transformed = transform_benchmark_terminal_inputs(means, mean_error, cost_bps=cost)
                    requested_mu, requested_error = transformed["transformed_mu"], transformed["transformed_covariance"]
                    transforms[cost] = transformed
                else:
                    requested_mu, requested_error = means.copy(), mean_error.copy()
                raw, audit = replay_horizon_allocation(market, requested_mu, requested_error, sigma.copy(),
                    cost_bps=cost, rebalance_sessions=5, robust_radius=plan["robust_radius"])
                solver_details[(name, cost)] = audit["solver_details"]
                solver_rows.append({"strategy": name, "cost_bps": cost, **{k: v for k, v in audit.items() if k != "solver_details"}})
                if name == "benchmark_terminal":
                    details = verify_terminal_valuations(raw, audit["solver_details"], means, transformed,
                        cost_bps=cost, daily_covariances=sigma)
                    valuation_details[cost] = details
                    valuation_rows.append({"cost_bps": cost, "solves": len(details),
                        "maximum_objective_identity_error": float(details.objective_identity_error.max()),
                        "maximum_post_fee_values_error": float(details.post_fee_values_error.max()),
                        "future_conversion_fees_debited": False, "model_fits": 0,
                        "wealth_scale": transformed["wealth_scale"],
                        "maximum_covariance_roundoff_adjustment": transformed["audit"]["maximum_covariance_roundoff_adjustment"]})
                stage = f"native_{name}_{cost:g}bps"
                paths[(name, cost)], native[(name, cost)] = _native_path(market, raw, provider, cost, plan["initial_cash"])
                if name == "selected_original":
                    control_rows.append(compare_parent_control(paths[(name, cost)],
                        objects["analysis/paths.pkl"][(plan["parent"]["forecast_variant"], cost)], cost_bps=cost))
                print(f"Native terminal valuation path passed: {name}, {cost:g}bp", flush=True)
        stage = "raw_and_resolved_evidence"
        raw_tables, raw_evidence = evaluate_registered_paths(paths, plan=plan, calendar=bundle["calendar"],
            candidate_controls={"benchmark_terminal": "selected_original"})
        tables, evidence = resolve_evidence(raw_tables, raw_evidence, paths, plan)
        tables.update({"solver_summary.csv": pd.DataFrame(solver_rows), "valuation_audit.csv": pd.DataFrame(valuation_rows),
            "parent_control_reproduction.csv": pd.DataFrame(control_rows),
            "native_audit.csv": pd.DataFrame([{"strategy": name, **value["audit"]} for (name, cost), value in native.items()])})
        stage = "end_source_identity_guard"
        fresh = prepare_horizon_inputs(evidence_root, plan)
        _, fresh_provider = load_legacy_execution_view(evidence_root)
        _, fresh_parent = _load_parent(evidence_root, plan)
        _require(fresh_provider == provider_audit and fresh_parent == parent_audit and fresh["audit"] == bundle["audit"]
            and _bundle_identity(fresh) == identity and _bundle_identity(bundle) == identity
            and _sha(card / "plan.json") == plan_hash
            and _sha(card / "mechanism.md") == mechanism_hash
            and all(_sha(source_root / name) == digest for name, digest in source_hashes.items()),
            "source, parent evidence, financial plan or running code changed")
        for name, table in tables.items():
            table.to_csv(card / name, index=False)
        summary = {"experiment": plan["experiment"], "campaign_status": "active_research", "objective_achieved": False,
            "provider_uri": plan["source"]["provider_uri"], "execution_provider_uri": plan["source"]["execution_provider_uri"],
            "as_of": plan["source"]["as_of"], "region": "us", "market": "all", "benchmark": "SOXX",
            "data_range": [str(market.signal_date.iloc[0].date()), str(market.exit_date.iloc[-1].date())],
            "recorder_id": recorder_id, "experiment_id": recorder.experiment_id, "recorder_uri": uri,
            "recorder_local_dir": recorder.get_local_dir(), "parent_recorder_id": plan["parent"]["recorder_id"],
            "parent_artifact_sha256": plan["parent"]["artifact_sha256"], "plan_sha256": plan_hash,
            "source_sha256": source_hashes, "run_binding_sha256": _sha(card / "run_binding.json"),
            "model_fits": 0, "model_selection": "none_frozen_parent_forecasts", "native_paths": len(native), "native_all_pass": True,
            "raw_historical_gate_pass": raw_evidence["benchmark_terminal"]["historical_gate_pass"],
            "historical_gate_pass": evidence["benchmark_terminal"]["historical_gate_pass"],
            "excess_resolution": evidence["benchmark_terminal"]["excess_resolution"], "promotion_eligible": False,
            "decision": "continue_research_no_forward_qualification", "evidence_kind": "reused_history",
            "actual_terminal_trades": False, "parent_control_reproduced": True,
            "solver_summary_stage": "funded ledger audit before native replay; final native verification is in native_audit.csv and native_reports.pkl.gz",
            "sidecars": [*tables, "plan.json", "run_binding.json", "mechanism.md", "conclusion.md"],
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/frame.pkl", "forecast/frozen_parent.pkl", "analysis/paths.pkl",
                    "analysis/raw_gates.pkl", "analysis/gates.pkl", "analysis/solver_details.pkl.gz",
                    "analysis/terminal_valuation_details.pkl.gz", "portfolio_analysis/native_reports.pkl.gz"]},
            "qlib_docs": plan["source_references"][1:], "limitations": plan["limitations"]}
        conclusion = _conclusion(summary, tables)
        (card / "conclusion.md").write_text(conclusion)
        stage = "archive_success"
        with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
            recorder = _same_recorder(R, recorder_id)
            recorder.save_objects(artifact_path="analysis", **{"paths.pkl": paths, "raw_gates.pkl": raw_evidence, "gates.pkl": evidence})
            summary["compressed_artifacts"] = [
                save_compressed_object(recorder, name="native_reports.pkl.gz", artifact_path="portfolio_analysis", value=native),
                save_compressed_object(recorder, name="solver_details.pkl.gz", artifact_path="analysis", value=solver_details),
                save_compressed_object(recorder, name="terminal_valuation_details.pkl.gz", artifact_path="analysis",
                    value={"decisions": valuation_details, "transforms": transforms})]
            for name in [*tables, "plan.json", "run_binding.json", "workflow.yaml", "mechanism.md", "conclusion.md"]:
                recorder.log_artifact(str(card / name), artifact_path="card")
            with tempfile.TemporaryDirectory(prefix="terminal-card-summary-") as temporary:
                staged = Path(temporary) / "summary.json"
                _json(staged, summary)
                recorder.log_artifact(str(staged), artifact_path="card")
        _json(card / "summary.json", summary)
    except Exception as error:
        failure = {"stage": stage, "exception_type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "plan_sha256": plan_hash,
                   "financial_recipe_changed": False, "summary_published": False}
        with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
            recorder = _same_recorder(R, recorder_id)
            recorder.save_objects(artifact_path="failure", **{"exception.pkl": failure, "partial_paths.pkl": paths})
            save_compressed_object(recorder, name="partial_solver_details.pkl.gz", artifact_path="failure", value=solver_details)
            if native:
                save_compressed_object(recorder, name="partial_native_reports.pkl.gz", artifact_path="failure", value=native)
            raise
    for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
        (Path(recorder.get_local_dir()) / "artifacts" / name).unlink(missing_ok=True)
    return summary, tables, conclusion
