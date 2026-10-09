"""Purged annual forecast regularization with a fixed downstream allocator."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.analysis.forecast_increment import analyze_forecast_increment
from src.backtest.benchmark_relative_replay import replay_fixed_target
from src.backtest.horizon_matched_allocation import replay_horizon_allocation
from src.core.recorder_artifacts import save_compressed_object
from src.data.convex_allocation import predict_annual
from src.data.legacy_execution_view import load_legacy_execution_view
from src.data.oos_horizon_inputs import prepare_horizon_inputs
from src.model.annual_ridge_selection import select_annual_ridge
from src.workflow.benchmark_evidence import evaluate_registered_paths
from src.workflow.benchmark_relative_research import _sha, _json, _same_recorder, _native_path
from src.workflow.oos_uncertainty_horizon import _source_hashes as inherited_hashes, _bundle_identity


NEW_SOURCES = ("src/workflow/purged_ridge_regularization.py", "src/model/annual_ridge_selection.py",
    "src/workflow/benchmark_evidence.py", "src/analysis/forecast_increment.py")


def _validate(plan):
    source_fields = {"experiment", "recorder_id", "provider_uri", "execution_provider_uri", "summary", "as_of"}
    integer_fields = ("candidate_budget", "inner_candidate_count", "rebalance_sessions", "forecast_horizon_sessions",
                      "minimum_training_rows", "minimum_validation_rows", "hac_lags")
    if (not isinstance(plan.get("source"), dict) or source_fields - set(plan["source"])
            or any(type(plan[key]) is not int for key in integer_fields)
            or type(plan["gates"]["family_size"]) is not int
            or any(type(y) is not int for y in [*plan["fit_years"], *plan["development_years"]])
            or any(isinstance(a, bool) or not np.isfinite(a) for a in plan["alphas"])):
        raise ValueError("blocked: invalid nested-selection source or parameter types")
    if (plan["schema_version"] != "purged_ridge_regularization_v1"
            or plan["registered_before_new_fits_and_portfolio_results"] is not True
            or plan["alphas"] != [10., 100., 1000.] or plan["explicit_prior_candidate"] is not True
            or plan["tie_priority"] != ["prior", "ridge_1000", "ridge_100", "ridge_10"]
            or type(plan["candidate_budget"]) is not int or plan["candidate_budget"] != 1
            or plan["inner_candidate_count"] != 4 or plan["costs_bps"] != [10., 25.]
            or plan["gates"]["family_size"] != 1 or plan["rebalance_sessions"] != 5
            or plan["forecast_horizon_sessions"] != 5 or plan["evidence_kind"] != "reused_history"
            or plan["train_start"] != "2011-01-03" or plan["minimum_training_rows"] != 252
            or plan["minimum_validation_rows"] != 126 or plan["hac_lags"] != 5
            or plan["benchmark"] != "SOXX" or isinstance(plan["robust_radius"], bool)
            or not np.isfinite(plan["robust_radius"]) or plan["robust_radius"] != 1.
            or plan["fit_years"] != list(range(2016, 2027)) or plan["development_years"] != list(range(2016, 2023))
            or [v["name"] for v in plan["variants"]] != ["fixed_ridge10", "validation_selected"]):
        raise ValueError("blocked: registered nested Ridge protocol differs")
    stamp = pd.Timestamp(plan["registered_at_utc"])
    if stamp.tzinfo is None or stamp > pd.Timestamp.now(tz="UTC"):
        raise ValueError("blocked: invalid registration clock")


def _verify_parent_control(root, plan, paths):
    """Cross-card equality is a reproduction check, never a selection score."""
    from qlib.workflow import R

    locations = [p for p in plan["cause_sources"] if "/oos_uncertainty_horizon_" in p]
    if len(locations) != 1:
        raise ValueError("blocked: exact prior fixed-Ridge control source is required")
    summary = json.loads((root / locations[0]).read_text())
    with R.uri_context(summary["recorder_uri"]):
        recorder = R.get_recorder(experiment_name=summary["experiment"], recorder_id=summary["recorder_id"])
        if recorder.client.get_run(recorder.id).info.status != "FINISHED":
            raise ValueError("blocked: prior control Recorder is not finished")
        previous = recorder.load_object("analysis/paths.pkl")
        digest = _sha(Path(recorder.get_local_dir()) / "artifacts/analysis/paths.pkl")
    rows = []
    for cost in plan["costs_bps"]:
        current, prior = paths[("fixed_ridge10", cost)], previous[("hac_h5", cost)]
        for column in ("signal_date", "trade_date", "exit_date"):
            if not current[column].equals(prior[column]):
                raise ValueError("blocked: previous fixed control calendar differs")
        columns = ["strategy_return", "turnover", "fees", *[f"open_{a}_weight" for a in ("soxl", "soxx", "cash")]]
        error = float(np.abs(current[columns].to_numpy(float) - prior[columns].to_numpy(float)).max())
        if error > 1e-10:
            raise ValueError("blocked: fixed forecast control failed prior-card reproduction")
        rows.append({"cost_bps": cost, "parent_recorder_id": recorder.id, "path_sha256": digest,
            "maximum_difference": error, "tolerance": 1e-10, "passed": True})
    return pd.DataFrame(rows)


def _hashes(root):
    hashes = inherited_hashes(root)
    for name in NEW_SOURCES:
        module = importlib.import_module(name[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (root / name).resolve():
            raise ValueError("blocked: imported nested-selection source differs")
        hashes[name] = _sha(root / name)
    return hashes


def fit_selected_forecasts(bundle, plan):
    dataset, frame = bundle["dataset"], bundle["frame"]
    dataset.calendar = bundle["calendar"].copy()
    models, inner_models, scores, audits = {}, {}, [], []
    forecasts = {name: {"rows": [], "Vmean": []} for name in ("fixed_ridge10", "validation_selected")}
    for year in plan["fit_years"]:
        selected = select_annual_ridge(dataset, frame, outer_year=year, alphas=tuple(plan["alphas"]),
            minimum_training_rows=plan["minimum_training_rows"], minimum_validation_rows=plan["minimum_validation_rows"],
            hac_lags=plan["hac_lags"])
        control, fitted = selected["control_fit"], selected["selected_fit"]
        if (control["audit"]["training_frame_sha256"] != fitted["audit"]["training_frame_sha256"]
                or not np.array_equal(control["prior_mean"], fitted["prior_mean"])):
            raise ValueError("blocked: selected/control outer training sources differ")
        models[year] = {"fixed_ridge10": control, "validation_selected": fitted}
        inner_models[year] = selected["inner_models"]
        scores.append(selected["candidate_scores"])
        audits.append(selected["audit"])
        part = frame.loc[(frame.index.year == year) & frame.exit_date.notna()]
        for name, model in models[year].items():
            forecast = predict_annual(model, part)
            if (np.shape(forecast["ridge"]) != (len(part), 2)
                    or np.shape(forecast["prior"]) != (len(part), 2)
                    or np.shape(forecast["Vmean"]) != (len(part), 2, 2)):
                raise ValueError("blocked: annual prediction/uncertainty row alignment differs")
            table = pd.DataFrame({"signal_date": part.signal_date, "forecast_available_at": part.decision_at,
                "label_available_at": part.h5_available_at, "model_year": year,
                "predicted_soxl": forecast["ridge"][:, 0], "predicted_soxx": forecast["ridge"][:, 1],
                "prior_soxl": forecast["prior"][:, 0], "prior_soxx": forecast["prior"][:, 1]}, index=part.index)
            forecasts[name]["rows"].append(table)
            forecasts[name]["Vmean"].append(forecast["Vmean"])
    for name, value in forecasts.items():
        value["rows"], value["Vmean"] = pd.concat(value["rows"]), np.concatenate(value["Vmean"])
        if (value["rows"].index.has_duplicates or not value["rows"].index.is_monotonic_increasing
                or value["Vmean"].shape != (len(value["rows"]), 2, 2)):
            raise ValueError("blocked: forecast index/uncertainty alignment differs")
    base, full = forecasts["fixed_ridge10"]["rows"], forecasts["validation_selected"]["rows"]
    common = ["signal_date", "forecast_available_at", "label_available_at", "model_year", "prior_soxl", "prior_soxx"]
    if not base[common].equals(full[common]):
        raise ValueError("blocked: fixed/selected forecast clocks or priors differ")
    info = base[common].copy()
    for asset in ("soxl", "soxx"):
        info[f"base_{asset}"] = base[f"predicted_{asset}"]
        info[f"full_{asset}"] = full[f"predicted_{asset}"]
        info[f"realized_{asset}"] = frame.loc[info.index, f"{asset}_return_h5"]
    return models, inner_models, forecasts, info, pd.concat(scores, ignore_index=True), pd.DataFrame(audits)


def _conclusion(summary, tables):
    lines = ["# 严格过去验证选择Ridge正则化", "",
        "本轮保留25价格特征、H5标签与5日执行，只改变收益头正则选择。每年用更早训练期拟合10/100/1000及历史均值预测，以前一年严格成熟验证误差选择，再用当前年开始前成熟数据重拟合。", "",
        "25bp比较如下；完整开发年2016–2022，其后仍为复用历史：", ""]
    for row in tables["backtest_benchmark_comparison.csv"].query("cost_bps==25").itertuples():
        lines.append(f"- {row.window}/{row.strategy}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD {row.max_drawdown_pct:.2f}%，CVaR5 {row.cvar5_pct:.2f}%，换手{row.turnover:.2f}。")
    lines += ["", f"历史门通过：{summary['historical_gate_pass']}；年度选择：{summary['selected_by_year']}。整体目标仍在进行，不晋升。",
        "", "历史均值预测与全仓SOXX组合是两个不同对照；若选择prior，仍须看真实优化权重、费用与收益，不能称其学到了alpha。内层两资产误差按各自内层训练方差归一，避免SOXL主导，绝不按外层收益选择。",
        "", "年度/窗口差异与所有44个内层候选分数保留于selection_scores、selection_audit、annual_metrics。预测MSE/IC及组合费用后稳定性分开评价，没有根据后验结果翻号、改alpha或改期限。",
        "", "全部历史已复用；先前年度验证仍可能不适合下一年。H5重叠、价格口径/次来源观察、历史开盘、连续份额与现金零收益限制保留。",
        "", "若选择修正仍未通过，下一步依据选择结果、预测增量与真实持仓差异继续定位信息或配置目标的问题；本批归档不关闭稳定超额目标。",
        "", f"Qlib Recorder：{summary['recorder_id']}；as_of={summary['as_of']}。"]
    return "\n".join(lines) + "\n"


def run_purged_ridge_regularization(*, source_root, evidence_root, card_dir, recorder_uri):
    import qlib
    from qlib.workflow import R

    source_root, evidence_root, card = map(lambda p: Path(p).resolve(), (source_root, evidence_root, card_dir))
    plan = json.loads((card / "plan.json").read_text())
    _validate(plan)
    if (card / "summary.json").exists():
        raise ValueError("blocked: completed card is immutable")
    hashes, plan_hash = _hashes(source_root), _sha(card / "plan.json")
    provider, provider_audit = load_legacy_execution_view(evidence_root)
    if provider != (evidence_root / plan["source"]["execution_provider_uri"]).resolve():
        raise ValueError("blocked: execution provider differs")
    bundle = prepare_horizon_inputs(evidence_root, plan)
    bundle["dataset"].calendar = bundle["calendar"].copy()
    identity = _bundle_identity(bundle)
    uri = str(Path(recorder_uri).resolve())
    binding = {"plan_sha256": plan_hash, "source_sha256": hashes, "provider_audit": provider_audit,
        "data_audit": bundle["audit"], "bundle_identity": identity, "source_root": str(source_root),
        "read_only_evidence_root": str(evidence_root), "card_dir": str(card), "recorder_uri": uri,
        "qlib_version": qlib.__version__, "cause_summaries_sha256": {p: _sha(evidence_root / p) for p in plan["cause_sources"]}}
    _json(card / "run_binding.json", binding)
    with R.start(experiment_name=plan["experiment"], uri=uri):
        recorder = R.get_recorder()
        recorder_id = recorder.id
        recorder.log_params(plan_sha256=plan_hash, candidate_budget=1, inner_candidates=4, evidence_kind="reused_history")
        recorder.save_objects(artifact_path="inputs", **{"binding.pkl": binding, "plan.pkl": plan, "frame.pkl": bundle["frame"]})
        for name in hashes:
            recorder.log_artifact(str(source_root / name), artifact_path="source/" + str(Path(name).parent))
        models, inner_models, forecasts, info, scores, audits = fit_selected_forecasts(bundle, plan)
        if _bundle_identity(bundle) != identity:
            raise ValueError("blocked: Dataset changed during nested selection")
        recorder.save_objects(artifact_path="models", **{"annual_fits.pkl": models, "inner_fits.pkl": inner_models})
        recorder.save_objects(artifact_path="forecast", **{"forecasts.pkl": forecasts, "matched_predictions.pkl": info,
            "selection_scores.pkl": scores, "selection_audit.pkl": audits})
    paths, native, details, solver_rows = {}, {}, {}, []
    try:
        statistics = analyze_forecast_increment(info,
            evaluation_cutoff=pd.Timestamp(plan["source"]["as_of"], tz="America/New_York") + pd.Timedelta(hours=17),
            development_years=tuple(plan["development_years"]), resamples=2000, mean_blocks=(20, 60))
        mask = (bundle["frame"].index >= pd.Timestamp(plan["signal_start"])) & bundle["frame"].exit_date.notna()
        market, sigma = bundle["frame"].loc[mask].copy(), bundle["covariance"][np.flatnonzero(mask)]
        for cost in plan["costs_bps"]:
            raw = replay_fixed_target(market, target=[0., 1., 0.], cost_bps=cost, rebalance_sessions=5, buy_and_hold=True)
            paths[("soxx", cost)], native[("soxx", cost)] = _native_path(market, raw, provider, cost, plan["initial_cash"])
            for name in ("fixed_ridge10", "validation_selected"):
                forecast = forecasts[name]
                loc = forecast["rows"].index.get_indexer(market.index)
                if (loc < 0).any():
                    raise ValueError("blocked: replay missing exact nested forecast dates")
                means = forecast["rows"].iloc[loc][["predicted_soxl", "predicted_soxx"]].to_numpy(float)
                raw, audit = replay_horizon_allocation(market, means, forecast["Vmean"][loc], sigma,
                    cost_bps=cost, rebalance_sessions=plan["rebalance_sessions"], robust_radius=plan["robust_radius"])
                details[(name, cost)] = audit["solver_details"]
                solver_rows.append({"strategy": name, "cost_bps": cost, **{k: v for k, v in audit.items() if k != "solver_details"}})
                paths[(name, cost)], native[(name, cost)] = _native_path(market, raw, provider, cost, plan["initial_cash"])
                print(f"Native nested Ridge path passed: {name}, {cost:g}bp", flush=True)
        tables, evidence = evaluate_registered_paths(paths, plan=plan, calendar=bundle["calendar"],
            candidate_controls={"validation_selected": "fixed_ridge10"})
        tables.update({"parent_control_reproduction.csv": _verify_parent_control(evidence_root, plan, paths),
            "selection_scores.csv": scores, "selection_audit.csv": audits,
            "forecast_metrics.csv": statistics["metrics"], "forecast_increments.csv": statistics["increments"],
            "forecast_bootstrap.csv": statistics["bootstrap"], "solver_summary.csv": pd.DataFrame(solver_rows),
            "native_audit.csv": pd.DataFrame([{"strategy": n, **v["audit"]} for (n, cost), v in native.items()])})
        _, current_provider = load_legacy_execution_view(evidence_root)
        fresh = prepare_horizon_inputs(evidence_root, plan)
        if (current_provider != provider_audit or fresh["audit"] != bundle["audit"]
                or _bundle_identity(bundle) != identity or _bundle_identity(fresh) != identity
                or _sha(card / "plan.json") != plan_hash or any(_sha(source_root / n) != h for n, h in hashes.items())
                or any(_sha(evidence_root / p) != h for p, h in binding["cause_summaries_sha256"].items())):
            raise ValueError("blocked: nested selection source or plan changed")
        for name, table in tables.items():
            table.to_csv(card / name, index=False)
        summary = {"experiment": plan["experiment"], "campaign_status": "active_research", "objective_achieved": False,
            "provider_uri": plan["source"]["provider_uri"], "execution_provider_uri": plan["source"]["execution_provider_uri"],
            "as_of": plan["source"]["as_of"], "region": "us", "market": "all", "benchmark": "SOXX",
            "data_range": [str(market.signal_date.iloc[0].date()), str(market.exit_date.iloc[-1].date())],
            "recorder_id": recorder_id, "experiment_id": recorder.experiment_id, "recorder_uri": uri,
            "recorder_local_dir": recorder.get_local_dir(), "plan_sha256": plan_hash, "source_sha256": hashes,
            "run_binding_sha256": _sha(card / "run_binding.json"), "native_paths": len(native), "native_all_pass": True,
            "model_selection": "purged_previous_year_forecast_error_then_outer_refit",
            "selected_by_year": {str(row.outer_year): row.selected_name for row in audits.itertuples()},
            "historical_gate_pass": evidence["validation_selected"]["historical_gate_pass"], "promotion_eligible": False,
            "decision": "continue_research_no_forward_qualification", "evidence_kind": "reused_history",
            "forecast_audit": statistics["audit"], "sidecars": [*tables, "plan.json", "run_binding.json", "conclusion.md"],
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/frame.pkl", "models/annual_fits.pkl", "models/inner_fits.pkl",
                    "forecast/forecasts.pkl", "forecast/matched_predictions.pkl", "analysis/paths.pkl",
                    "analysis/solver_details.pkl.gz", "portfolio_analysis/native_reports.pkl.gz"]},
            "qlib_docs": [plan["source_references"][0]], "limitations": plan["limitations"]}
        conclusion = _conclusion(summary, tables)
        (card / "conclusion.md").write_text(conclusion)
        with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
            recorder = _same_recorder(R, recorder_id)
            recorder.save_objects(artifact_path="analysis", **{"paths.pkl": paths, "gates.pkl": evidence, "forecast_statistics.pkl": statistics})
            summary["compressed_artifacts"] = [save_compressed_object(recorder, name="native_reports.pkl.gz", artifact_path="portfolio_analysis", value=native),
                save_compressed_object(recorder, name="solver_details.pkl.gz", artifact_path="analysis", value=details)]
            for name in [*tables, "plan.json", "run_binding.json", "workflow.yaml", "conclusion.md"]:
                recorder.log_artifact(str(card / name), artifact_path="card")
            with tempfile.TemporaryDirectory(prefix="qlib-card-") as temporary:
                staged = Path(temporary) / "summary.json"
                _json(staged, summary)
                recorder.log_artifact(str(staged), artifact_path="card")
        _json(card / "summary.json", summary)
    except Exception:
        with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
            _same_recorder(R, recorder_id)
            raise
    for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
        (Path(recorder.get_local_dir()) / "artifacts" / name).unlink(missing_ok=True)
    return summary, tables, conclusion
