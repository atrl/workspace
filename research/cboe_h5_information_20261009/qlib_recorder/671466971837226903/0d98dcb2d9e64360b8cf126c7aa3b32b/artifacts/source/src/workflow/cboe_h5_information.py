"""Matched-price versus lagged-Cboe H5 information experiment."""
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
from src.data.cboe_horizon_dataset import prepare_cboe_horizon_dataset
from src.data.cboe_volatility_source import FEATURES
from src.data.convex_allocation import FEATURE_COLUMNS as BASE_FEATURES
from src.data.legacy_execution_view import load_legacy_execution_view
from src.model.matched_horizon_ridge import fit_matched_horizon_ridge, predict_matched_horizon
from src.workflow.benchmark_evidence import evaluate_registered_paths
from src.workflow.benchmark_relative_research import _sha, _json, _same_recorder, _native_path
from src.workflow.oos_uncertainty_horizon import _source_hashes as parent_source_hashes, _bundle_identity


NEW_SOURCES = ("src/workflow/cboe_h5_information.py", "src/workflow/benchmark_evidence.py",
    "src/data/cboe_horizon_dataset.py", "src/data/cboe_volatility_source.py",
    "src/model/matched_horizon_ridge.py", "src/analysis/forecast_increment.py")


def _validate_plan(plan):
    required_source = {"experiment", "recorder_id", "provider_uri", "execution_provider_uri", "summary", "as_of"}
    required_external = {"provider_uri", "snapshot_dir", "store_as_of", "market_as_of", "source_start", "features", "historical_vintage_certified"}
    if (not isinstance(plan.get("source"), dict) or required_source - set(plan["source"])
            or not isinstance(plan.get("external_source"), dict) or required_external - set(plan["external_source"])
            or type(plan["candidate_budget"]) is not int or type(plan["gates"]["family_size"]) is not int
            or any(type(y) is not int for y in [*plan["fit_years"], *plan["development_years"]])):
        raise ValueError("blocked: invalid source declarations or registered parameter types")
    expected = [{"name": "base", "extra_features": False, "role": "matched_control"},
        {"name": "base_cboe3", "extra_features": True, "role": "registered_information_candidate"}]
    if (plan["schema_version"] != "cboe_h5_information_v1" or plan["variants"] != expected
            or any(type(v["extra_features"]) is not bool for v in plan["variants"])
            or plan["candidate_budget"] != 1 or plan["gates"]["family_size"] != 1
            or plan["costs_bps"] != [10., 25.] or plan["benchmark"] != "SOXX"
            or plan["rebalance_sessions"] != 5 or type(plan["rebalance_sessions"]) is not int
            or plan["forecast_horizon_sessions"] != 5 or plan["evidence_kind"] != "reused_history"
            or plan["fit_years"] != list(range(2016, 2027))
            or plan["development_years"] != list(range(2016, 2023))
            or plan["registered_before_fits_and_portfolio_results"] is not True
            or tuple(plan["external_source"]["features"]) != FEATURES
            or plan["external_source"]["market_as_of"] != plan["source"]["as_of"]
            or plan["external_source"]["historical_vintage_certified"] is not False
            or plan["external_source"]["source_start"] != "2014-02-03"
            or plan["matched_training_start"] != "2014-02-04"
            or pd.Timestamp(plan["external_source"]["store_as_of"]) < pd.Timestamp(plan["source"]["as_of"])):
        raise ValueError("blocked: registered Cboe information protocol differs")
    registered = pd.Timestamp(plan["registered_at_utc"])
    if registered.tzinfo is None or registered > pd.Timestamp.now(tz="UTC"):
        raise ValueError("blocked: invalid registration timestamp")


def _hashes(root):
    hashes = parent_source_hashes(root)
    for name in NEW_SOURCES:
        module = importlib.import_module(name[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (root / name).resolve():
            raise ValueError("blocked: imported Cboe implementation differs")
        hashes[name] = _sha(root / name)
    return hashes


def fit_matched_models(bundle, plan):
    models, forecasts, fit_rows = {}, {}, []
    for variant in plan["variants"]:
        name = variant["name"]
        columns = (*BASE_FEATURES, *FEATURES) if variant["extra_features"] else BASE_FEATURES
        models[name], tables, errors = {}, [], []
        for year in plan["fit_years"]:
            model = fit_matched_horizon_ridge(bundle["dataset"], year=year, feature_columns=columns,
                ridge_alpha=plan["ridge_alpha"], minimum_rows=plan["minimum_training_rows"],
                hac_lags=plan["hac_lags"], train_start=plan["matched_training_start"])
            if name != "base":
                reference = models["base"][year]
                for field in ("matched_rows_sha256", "label_rows_sha256", "training_dates_sha256"):
                    if model["audit"][field] != reference["audit"][field]:
                        raise ValueError("blocked: price/Cboe fits consumed different mature rows")
                if not np.array_equal(model["prior_mean"], reference["prior_mean"]):
                    raise ValueError("blocked: matched prior means differ")
            prediction = predict_matched_horizon(model, bundle["dataset"], signal_year=year)
            tables.append(prediction["predictions"])
            errors.append(prediction["Vmean"])
            models[name][year] = model
            fit_rows.append({"variant": name, **model["audit"]})
        forecasts[name] = {"rows": pd.concat(tables), "Vmean": np.concatenate(errors)}
        rows = forecasts[name]["rows"]
        if (rows.index.has_duplicates or not rows.index.is_monotonic_increasing
                or forecasts[name]["Vmean"].shape != (len(rows), 2, 2)):
            raise ValueError("blocked: inference rows and uncertainty matrices differ")
    base, full = forecasts["base"]["rows"], forecasts["base_cboe3"]["rows"]
    if not base.index.equals(full.index):
        raise ValueError("blocked: base/Cboe inference dates differ")
    common = ["signal_date", "forecast_available_at", "label_available_at", "model_year", "prior_soxl", "prior_soxx"]
    if not base[common].equals(full[common]):
        raise ValueError("blocked: base/Cboe inference clocks or prior values differ")
    info = base[["signal_date", "forecast_available_at", "label_available_at", "model_year", "prior_soxl", "prior_soxx"]].copy()
    for asset in ("soxl", "soxx"):
        info[f"base_{asset}"] = base[f"predicted_{asset}"]
        info[f"full_{asset}"] = full[f"predicted_{asset}"]
        info[f"realized_{asset}"] = bundle["frame"].loc[info.index, f"{asset}_return_h5"]
    return models, forecasts, info, pd.DataFrame(fit_rows)


def _conclusion(summary, tables):
    rows = ["# CBOE三指标的匹配H5信息增量", "",
        "BASE与BASE+CBOE3使用相同2014-02-04起的成熟训练行、Ridge10、H5期限、费用、风险和5日持有合同。三指标均取上一原生会话值，没有SKEW填补或窗口搜索。", "",
        "25bp同成本比较如下；2016–2022为完整信号年度，后段为已复用历史：", ""]
    for row in tables["backtest_benchmark_comparison.csv"].query("cost_bps==25").itertuples():
        rows.append(f"- {row.window}/{row.strategy}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD {row.max_drawdown_pct:.2f}%，CVaR5 {row.cvar5_pct:.2f}%，换手{row.turnover:.2f}。")
    rows += ["", f"历史稳定超额门通过：{summary['historical_gate_pass']}。不晋升，整体目标继续。",
        "", "预测增量见forecast_metrics/forecast_increments/forecast_bootstrap；MSE改善统一为参考误差减FULL误差。相关、误差、费用后组合是不同证据，不能用一个IC正数替代稳定净收益。",
        "", "年度与窗口差异完整保留；没有根据regime、成本或年份挑选结果。BASE使用相同缩短训练起点，所以新增信息归因仅限本次匹配对照，不能直接与2011起的旧BASE当作因果对照。",
        "", "完整官方源保存至2026-10-08，但所有本次市场观察和指标消费限于2026-09-25。首发保守边界为2014-02-03，lag1与17:00可用假设不等于历史vintage认证。全部历史复用、回溯调整价格、现金零收益、连续份额及历史开盘执行仍为限制。",
        "", "若仍未通过，将先定位新增信息的误差、实际动作与尾部损失，再单独登记下一机制；不更换风险门、不根据本轮收益追加特征或调参。",
        "", f"Qlib Recorder：{summary['recorder_id']}；价格as_of={summary['as_of']}。"]
    return "\n".join(rows) + "\n"


def run_cboe_h5_information(*, source_root, evidence_root, card_dir, recorder_uri):
    import qlib
    from qlib.workflow import R

    source_root, evidence_root, card = map(lambda p: Path(p).resolve(), (source_root, evidence_root, card_dir))
    plan = json.loads((card / "plan.json").read_text())
    _validate_plan(plan)
    if (card / "summary.json").exists():
        raise ValueError("blocked: completed card is immutable")
    hashes, plan_hash = _hashes(source_root), _sha(card / "plan.json")
    provider, provider_audit = load_legacy_execution_view(evidence_root)
    if provider != (evidence_root / plan["source"]["execution_provider_uri"]).resolve():
        raise ValueError("blocked: execution provider differs")
    bundle = prepare_cboe_horizon_dataset(evidence_root, plan)
    identity = _bundle_identity(bundle)
    uri = str(Path(recorder_uri).resolve())
    binding = {"plan_sha256": plan_hash, "source_sha256": hashes, "data_audit": bundle["audit"],
        "bundle_identity": identity, "provider_audit": provider_audit, "source_root": str(source_root),
        "read_only_evidence_root": str(evidence_root), "card_dir": str(card), "recorder_uri": uri,
        "qlib_version": qlib.__version__}
    _json(card / "run_binding.json", binding)
    with R.start(experiment_name=plan["experiment"], uri=uri):
        recorder = R.get_recorder()
        recorder_id = recorder.id
        recorder.log_params(plan_sha256=plan_hash, candidate_budget=1, evidence_kind="reused_history")
        recorder.save_objects(artifact_path="inputs", **{"binding.pkl": binding, "plan.pkl": plan, "matched_frame.pkl": bundle["frame"]})
        for name in hashes:
            recorder.log_artifact(str(source_root / name), artifact_path="source/" + str(Path(name).parent))
        models, forecasts, info, fit_rows = fit_matched_models(bundle, plan)
        if _bundle_identity(bundle) != identity:
            raise ValueError("blocked: matched Dataset changed during fitting")
        recorder.save_objects(artifact_path="models", **{"annual_fits.pkl": models})
        recorder.save_objects(artifact_path="forecast", **{"forecasts.pkl": forecasts, "matched_predictions.pkl": info})
    paths, native, details, solver_rows = {}, {}, {}, []
    try:
        statistics = analyze_forecast_increment(info,
            evaluation_cutoff=pd.Timestamp(plan["source"]["as_of"], tz="America/New_York") + pd.Timedelta(hours=17),
            development_years=tuple(plan["development_years"]), mean_blocks=(20, 60), resamples=2000)
        mask = bundle["frame"].index >= pd.Timestamp(plan["signal_start"])
        market, sigma = bundle["frame"].loc[mask].copy(), bundle["covariance"][np.flatnonzero(mask)]
        for cost in plan["costs_bps"]:
            raw = replay_fixed_target(market, target=[0., 1., 0.], cost_bps=cost, rebalance_sessions=5, buy_and_hold=True)
            paths[("soxx", cost)], native[("soxx", cost)] = _native_path(market, raw, provider, cost, plan["initial_cash"])
            for variant in plan["variants"]:
                name = variant["name"]
                predicted = forecasts[name]
                loc = predicted["rows"].index.get_indexer(market.index)
                if (loc < 0).any():
                    raise ValueError("blocked: market path lacks exact forecast dates")
                means = predicted["rows"].iloc[loc][["predicted_soxl", "predicted_soxx"]].to_numpy(float)
                raw, audit = replay_horizon_allocation(market, means, predicted["Vmean"][loc], sigma,
                    cost_bps=cost, rebalance_sessions=plan["rebalance_sessions"], robust_radius=plan["robust_radius"])
                details[(name, cost)] = audit["solver_details"]
                solver_rows.append({"strategy": name, "cost_bps": cost, **{k: v for k, v in audit.items() if k != "solver_details"}})
                paths[(name, cost)], native[(name, cost)] = _native_path(market, raw, provider, cost, plan["initial_cash"])
                print(f"Native Cboe comparison passed: {name}, {cost:g}bp", flush=True)
        tables, evidence = evaluate_registered_paths(paths, plan=plan, calendar=bundle["calendar"], candidate_controls={"base_cboe3": "base"})
        tables.update({"forecast_metrics.csv": statistics["metrics"], "forecast_increments.csv": statistics["increments"],
            "forecast_bootstrap.csv": statistics["bootstrap"], "forecast_fits.csv": fit_rows,
            "solver_summary.csv": pd.DataFrame(solver_rows),
            "native_audit.csv": pd.DataFrame([{"strategy": n, **v["audit"]} for (n, cost), v in native.items()])})
        _, final_provider = load_legacy_execution_view(evidence_root)
        fresh = prepare_cboe_horizon_dataset(evidence_root, plan)
        if (final_provider != provider_audit or fresh["audit"] != bundle["audit"]
                or _bundle_identity(bundle) != identity or _bundle_identity(fresh) != identity
                or _sha(card / "plan.json") != plan_hash or any(_sha(source_root / n) != h for n, h in hashes.items())):
            raise ValueError("blocked: Cboe execution source or plan changed")
        for name, table in tables.items():
            table.to_csv(card / name, index=False)
        summary = {"experiment": plan["experiment"], "campaign_status": "active_research", "objective_achieved": False,
            "as_of": plan["source"]["as_of"], "external_store_as_of": plan["external_source"]["store_as_of"],
            "provider_uri": plan["source"]["provider_uri"], "external_provider_uri": plan["external_source"]["provider_uri"],
            "execution_provider_uri": plan["source"]["execution_provider_uri"], "region": "us", "market": "all", "benchmark": "SOXX",
            "data_range": [str(market.signal_date.iloc[0].date()), str(market.exit_date.iloc[-1].date())],
            "recorder_id": recorder_id, "experiment_id": recorder.experiment_id, "recorder_uri": uri,
            "recorder_local_dir": recorder.get_local_dir(), "plan_sha256": plan_hash, "source_sha256": hashes,
            "run_binding_sha256": _sha(card / "run_binding.json"), "native_paths": len(native), "native_all_pass": True,
            "historical_gate_pass": evidence["base_cboe3"]["historical_gate_pass"], "promotion_eligible": False,
            "decision": "continue_research_no_forward_qualification", "evidence_kind": "reused_history",
            "model_selection": plan["model_selection"], "historical_vintage_certified": False,
            "forecast_audit": statistics["audit"], "sidecars": list(tables) + ["plan.json", "run_binding.json", "source_intake.json", "source_contract.md", "conclusion.md"],
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/matched_frame.pkl", "models/annual_fits.pkl", "forecast/forecasts.pkl",
                    "forecast/matched_predictions.pkl", "analysis/paths.pkl", "analysis/solver_details.pkl.gz", "portfolio_analysis/native_reports.pkl.gz"]},
            "qlib_docs": [plan["source_references"][0]], "limitations": plan["limitations"]}
        conclusion = _conclusion(summary, tables)
        (card / "conclusion.md").write_text(conclusion)
        with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
            recorder = _same_recorder(R, recorder_id)
            recorder.save_objects(artifact_path="analysis", **{"paths.pkl": paths, "gates.pkl": evidence, "forecast_statistics.pkl": statistics})
            summary["compressed_artifacts"] = [save_compressed_object(recorder, name="native_reports.pkl.gz", artifact_path="portfolio_analysis", value=native),
                save_compressed_object(recorder, name="solver_details.pkl.gz", artifact_path="analysis", value=details)]
            for name in [*tables, "plan.json", "run_binding.json", "source_intake.json", "source_contract.md", "workflow.yaml", "conclusion.md"]:
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
