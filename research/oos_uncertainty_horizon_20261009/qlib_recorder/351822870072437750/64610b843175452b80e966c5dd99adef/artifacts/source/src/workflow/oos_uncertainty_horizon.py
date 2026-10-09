"""Registered cadence and past-OOS ambiguity comparisons, with Qlib evidence."""
from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.analysis.benchmark_relative_protocol import evaluate_benchmark_relative
from src.analysis.expert_information_attribution import attribute_expert_information
from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics
from src.backtest.benchmark_relative_replay import replay_fixed_target
from src.backtest.horizon_matched_allocation import replay_horizon_allocation
from src.core.recorder_artifacts import save_compressed_object
from src.data.benchmark_relative_data import build_relative_dataset
from src.data.integrated_allocation import frame_sha256
from src.data.legacy_execution_view import load_legacy_execution_view
from src.data.oos_horizon_inputs import prepare_horizon_inputs, fit_annual_oos
from src.model.oos_mean_uncertainty import fit_oos_mean_error
from src.workflow.benchmark_relative_research import _sha, _json, _same_recorder, _native_path, SOURCE_FILES


ADDITIONAL_SOURCES = (
    "src/workflow/oos_uncertainty_horizon.py", "src/data/oos_horizon_inputs.py",
    "src/model/oos_mean_uncertainty.py", "src/backtest/horizon_matched_allocation.py",
    "src/core/recorder_artifacts.py", "src/analysis/expert_information_attribution.py",
    "src/data/convex_allocation.py", "src/model/convex_allocation.py",
    "src/data/conditional_variance.py", "src/workflow/benchmark_risk_excess.py",
    "src/factor/soxx_pit_tail_factors.py", "src/backtest/position_accounting.py",
    "src/model/certified_convex_allocation.py",
)


def _source_hashes(root):
    hashes = {}
    for name in dict.fromkeys((*SOURCE_FILES, *ADDITIONAL_SOURCES)):
        module = importlib.import_module(name[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (root / name).resolve():
            raise ValueError("blocked: actual imported source differs from declared root")
        hashes[name] = _sha(root / name)
    return hashes


def _validate_plan(plan):
    expected = [("hac_daily", 1, False), ("hac_h5", 5, False), ("oos_floor_h5", 5, True)]
    required_source = {"experiment", "recorder_id", "provider_uri", "execution_provider_uri", "summary", "as_of"}
    if (not isinstance(plan.get("source"), dict) or required_source - set(plan["source"])
            or any(not isinstance(plan["source"][k], str) or not plan["source"][k] for k in required_source)
            or any(type(v.get("rebalance_sessions")) is not int or type(v.get("oos_floor")) is not bool
                   for v in plan["variants"])
            or any(type(y) is not int for y in [*plan["fit_years"], *plan["development_years"]])):
        raise ValueError("blocked: protocol source or parameter types are invalid")
    actual = [(v["name"], v["rebalance_sessions"], v["oos_floor"]) for v in plan["variants"]]
    if (plan["schema_version"] != "oos_uncertainty_horizon_v1" or actual != expected
            or plan["candidate_budget"] != 2 or plan["costs_bps"] != [10., 25.]
            or plan["evidence_kind"] != "reused_history" or plan["benchmark"] != "SOXX"
            or plan["forecast_horizon_sessions"] != 5 or plan["calibration"]["bias_correct_mean"] is not False
            or plan["registered_before_fits_and_replays"] is not True
            or plan["gates"]["family_size"] != 2
            or plan["fit_years"] != list(range(2013, 2027))
            or plan["development_years"] != list(range(2016, 2023))
            or isinstance(plan["robust_radius"], bool) or not np.isfinite(plan["robust_radius"])
            or plan["robust_radius"] < 0):
        raise ValueError("blocked: unsupported registered horizon protocol")
    stamp = pd.Timestamp(plan["registered_at_utc"])
    if stamp.tzinfo is None or stamp > pd.Timestamp.now(tz="UTC"):
        raise ValueError("blocked: invalid protocol registration timestamp")


def _bundle_identity(bundle):
    prepared = bundle["dataset"].prepare("all", col_set=["feature", "label", "metadata"]).copy()
    prepared.columns = prepared.columns.get_level_values(1)
    prepared.index = pd.DatetimeIndex(prepared.index.get_level_values("datetime"))
    return {"frame_sha256": frame_sha256(bundle["frame"]), "dataset_sha256": frame_sha256(prepared),
        "covariance_sha256": hashlib.sha256(np.asarray(bundle["covariance"], dtype=np.float64).tobytes()).hexdigest(),
        "calendar_sha256": hashlib.sha256(pd.DatetimeIndex(bundle["calendar"]).asi8.tobytes()).hexdigest()}


def _verify_numerical_refit(R, plan, repair, predictions, vmean, models):
    """Numerical execution repair must not silently refit different predictors."""
    previous = R.get_recorder(experiment_name=plan["experiment"], recorder_id=repair["failed_recorder_id"])
    prior_plan = previous.load_object("inputs/plan.pkl")
    if prior_plan != plan:
        raise ValueError("blocked: numerical repair changed the financial plan")
    prior_predictions = previous.load_object("forecast/predictions.pkl")
    pd.testing.assert_frame_equal(predictions, prior_predictions, check_exact=True)
    if not np.array_equal(vmean, previous.load_object("forecast/training_hac.pkl")):
        raise ValueError("blocked: numerical repair changed forecast uncertainty")
    prior_models = previous.load_object("models/annual_fits.pkl")
    if set(prior_models) != set(models):
        raise ValueError("blocked: numerical repair changed annual model inventory")
    for year, fit in models.items():
        prior = prior_models[year]
        for field in ("feature_mean", "feature_scale", "prior_mean", "ridge_coefficients", "coefficient_covariance_hac"):
            if not np.array_equal(fit[field], prior[field]):
                raise ValueError("blocked: numerical repair changed fitted parameters")
        if fit["audit"]["training_frame_sha256"] != prior["audit"]["training_frame_sha256"]:
            raise ValueError("blocked: numerical repair changed training rows")
    return {"failed_recorder_id": previous.id, "same_financial_plan": True,
        "predictions_exact_match": True, "prediction_sha256": frame_sha256(predictions),
        "annual_parameters_exact_match": True, "training_hac_exact_match": True,
        "model_count": len(models)}


def _attribution(root, plan, execution_provider):
    from qlib.workflow import R

    source = plan["attribution_source"]
    card = root / source["card"]
    summary = json.loads((card / "summary.json").read_text())
    if summary["recorder_id"] != source["recorder_id"]:
        raise ValueError("blocked: expert attribution Recorder differs")
    with R.uri_context(summary["recorder_uri"]):
        recorder = R.get_recorder(experiment_name=summary["experiment"], recorder_id=summary["recorder_id"])
        if recorder.client.get_run(recorder.id).info.status != "FINISHED":
            raise ValueError("blocked: expert attribution source is unfinished")
        paths = recorder.load_object("analysis/paths.pkl")
        source_plan = recorder.load_object("inputs/plan.pkl")
        identity = {name: _sha(Path(recorder.get_local_dir()) / "artifacts" / name)
                    for name in ("analysis/paths.pkl", "inputs/plan.pkl")}
    _, frame, _ = build_relative_dataset(provider_uri=execution_provider,
        signal_start=source_plan["signal_start"], as_of=source_plan["as_of"])
    tables = attribute_expert_information(paths, frame,
        windows={"development": ("2016-01-01", "2022-12-31"), "reused_diagnostic": ("2023-01-01", plan["source"]["as_of"])},
        candidate_name="online_fixed_share", reference_name="fixed_mixture", benchmark_name="soxx",
        leveraged_name="leveraged", defensive_name="defensive")
    return tables, {"recorder_id": recorder.id, "artifacts_sha256": identity}


def _conclusion(plan, summary, tables):
    lines = ["# OOS均值误差与预测期限实验", "",
        "运行了统一canonical价格下的H5 Ridge＋训练HAC日调仓对照、H5持有候选、H5持有＋过去年度OOF均值误差惩罚候选。相同预测分别结算10/25bp，全部完整路径由原生Qlib核验。", "",
        "以下为25bp；2016–2022按信号年归属，后段是2023以后已复用历史，持仓连续：", ""]
    for row in tables["backtest_benchmark_comparison.csv"].query("cost_bps==25").itertuples():
        lines.append(f"- {row.window}/{row.strategy}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD {row.max_drawdown_pct:.2f}%，CVaR5 {row.cvar5_pct:.2f}%，换手{row.turnover:.2f}。")
    lines += ["", f"候选历史门：{summary['historical_gates']}。整体稳定超额目标仍在进行，不晋升。",
        "", "OOF误差项与训练内HAC分开登记；它是平均偏差与其估计误差的保守尺度，不保证条件覆盖。期限对照只改变求解频率；OOF版本与相同5日HOLD对照隔离误差惩罚。",
        "", "调仓时估计风险不高于SOXX；中间HOLD允许持仓漂移，不能声称风险预算逐日保证。年度、危机与滚动窗口结果分别见小表，实际MaxDD/CVaR门未放宽。",
        "", "旧专家归因显示其主要共用半导体beta轴，不能靠扩大在线学习率补出独立信息。本轮完整保留失败；若信息与费用/期限修正仍不足，下一阶段将独立检验有完整长历史且发布时点合格的外部信息家族。",
        "", "全部历史已被复用；价格口径、16条次来源行情、开盘成交与现金零收益假设仍有限制。下一轮先依据失败归因登记新的机制，不因本轮结束而关闭用户目标。",
        "", f"Recorder：{summary['recorder_id']}；as_of={plan['source']['as_of']}。"]
    return "\n".join(lines) + "\n"


def run_oos_uncertainty_horizon(*, source_root, evidence_root, card_dir, recorder_uri):
    import qlib
    from qlib.workflow import R

    source_root, evidence_root, card = map(lambda p: Path(p).resolve(), (source_root, evidence_root, card_dir))
    plan = json.loads((card / "plan.json").read_text())
    _validate_plan(plan)
    if (card / "summary.json").exists():
        raise ValueError("blocked: completed card is immutable")
    hashes, plan_hash = _source_hashes(source_root), _sha(card / "plan.json")
    provider, provider_audit = load_legacy_execution_view(evidence_root)
    if provider != (evidence_root / plan["source"]["execution_provider_uri"]).resolve():
        raise ValueError("blocked: audited execution provider differs")
    attribution, attribution_source = _attribution(evidence_root, plan, provider)
    bundle = prepare_horizon_inputs(evidence_root, plan)
    bundle_identity = _bundle_identity(bundle)
    uri = str(Path(recorder_uri).resolve())
    binding = {"plan_sha256": plan_hash, "source_sha256": hashes, "provider_audit": provider_audit,
        "data_audit": bundle["audit"], "bundle_identity": bundle_identity, "attribution_source": attribution_source,
        "source_root": str(source_root), "read_only_evidence_root": str(evidence_root),
        "card_dir": str(card), "recorder_uri": uri, "qlib_version": qlib.__version__,
        "numerical_repair_sha256": _sha(card / "numerical_repair.json") if (card / "numerical_repair.json").exists() else None}
    _json(card / "run_binding.json", binding)
    with R.start(experiment_name=plan["experiment"], uri=uri):
        recorder = R.get_recorder()
        recorder_id = recorder.id
        recorder.log_params(plan_sha256=plan_hash, candidate_budget=2, evidence_kind="reused_history")
        recorder.save_objects(artifact_path="inputs", **{"binding.pkl": binding, "dataset_frame.pkl": bundle["frame"], "plan.pkl": plan})
        if binding["numerical_repair_sha256"] is not None:
            recorder.log_artifact(str(card / "numerical_repair.json"), artifact_path="inputs")
        for name in hashes:
            recorder.log_artifact(str(source_root / name), artifact_path="source/" + str(Path(name).parent))
        # Fit/OOF calibration use the caller's native DatasetH, with no qlib.init.
        predictions, vmean, models, fits = fit_annual_oos(bundle, plan)
        if binding["numerical_repair_sha256"] is not None:
            repair = json.loads((card / "numerical_repair.json").read_text())
            repair_audit = _verify_numerical_refit(R, plan, repair, predictions, vmean, models)
            recorder.save_objects(artifact_path="verification", **{"numerical_refit_identity.pkl": repair_audit})
        floors, calibration_rows = {}, []
        for year in range(min(plan["development_years"]), max(plan["fit_years"]) + 1):
            floor = fit_oos_mean_error(predictions, cutoff=pd.Timestamp(f"{year}-01-01", tz="America/New_York"),
                minimum_rows=plan["calibration"]["minimum_rows"], hac_lags=plan["calibration"]["hac_lags"])
            floors[year] = floor
            calibration_rows.append({"model_year": year, **floor["audit"],
                "bias_soxl": float(floor["bias"][0]), "bias_soxx": float(floor["bias"][1]),
                "floor_soxl": float(floor["floor"][0, 0]), "floor_cross": float(floor["floor"][0, 1]),
                "floor_soxx": float(floor["floor"][1, 1])})
        recorder.save_objects(artifact_path="models", **{"annual_fits.pkl": models, "oos_floors.pkl": floors})
        recorder.save_objects(artifact_path="forecast", **{"predictions.pkl": predictions, "training_hac.pkl": vmean})
        if _bundle_identity(bundle) != bundle_identity:
            raise ValueError("blocked: source Dataset/frame/covariance changed during fitting")
    paths, native, solver_details, calibration_table = {}, {}, {}, pd.DataFrame(calibration_rows)
    try:
        frame = bundle["frame"]
        mask = (frame.index >= pd.Timestamp(plan["signal_start"])) & frame.exit_date.notna()
        market = frame.loc[mask].copy()
        indices = predictions.index.get_indexer(market.index)
        if (indices < 0).any():
            raise ValueError("blocked: replay lacks exact OOF prediction coverage")
        means = predictions.iloc[indices][["predicted_soxl", "predicted_soxx"]].to_numpy(float)
        original_error = vmean[indices]
        extended_error = original_error + np.stack([floors[int(y)]["floor"] for y in market.index.year])
        covariance = bundle["covariance"][np.flatnonzero(mask)]
        solver_rows = []
        for cost in plan["costs_bps"]:
            raw = replay_fixed_target(market, target=[0., 1., 0.], cost_bps=cost, rebalance_sessions=1, buy_and_hold=True)
            paths[("soxx", cost)], native[("soxx", cost)] = _native_path(market, raw, provider, cost, plan["initial_cash"])
            for variant in plan["variants"]:
                name = variant["name"]
                path, audit = replay_horizon_allocation(market, means,
                    extended_error if variant["oos_floor"] else original_error, covariance,
                    cost_bps=cost, rebalance_sessions=variant["rebalance_sessions"], robust_radius=plan["robust_radius"])
                solver_details[(name, cost)] = audit["solver_details"]
                solver_rows.append({"strategy": name, "cost_bps": cost, **{k: v for k, v in audit.items() if k != "solver_details"}})
                paths[(name, cost)], native[(name, cost)] = _native_path(market, path, provider, cost, plan["initial_cash"])
                print(f"Native horizon path passed: {name}, {cost:g}bp", flush=True)
        comparisons, annual, gates, rolling, boots, ablations, evidence = [], [], [], [], [], [], {}
        dev_years = plan["development_years"]
        def dev(path):
            return path.loc[path.signal_date.dt.year.isin(dev_years)]
        calendar = bundle["calendar"]
        for name in ("hac_h5", "oos_floor_h5"):
            evidence[name] = evaluate_benchmark_relative(
                {c: dev(paths[(name, c)]) for c in plan["costs_bps"]},
                {c: dev(paths[("soxx", c)]) for c in plan["costs_bps"]},
                expected_years=tuple(dev_years), expected_signal_dates=calendar[calendar.year.isin(dev_years)], **plan["gates"])
            for target, key in ((gates, "gates"), (rolling, "rolling_summary"), (boots, "bootstrap")):
                target.append(evidence[name][key].assign(strategy=name))
        for (name, cost), path in paths.items():
            benchmark = paths[("soxx", cost)]
            annual.append(annual_paired_metrics(path, benchmark).assign(strategy=name, cost_bps=cost))
            for window, selected in (("development", path.signal_date.dt.year.isin(dev_years)),
                                     ("reused_diagnostic", path.signal_date.dt.year.gt(max(dev_years)))):
                comparisons.append({"strategy": name, "cost_bps": cost, "window": window,
                    **paired_metrics(path.loc[selected], benchmark.loc[selected])})
                if name in ("hac_h5", "oos_floor_h5"):
                    control = "hac_daily" if name == "hac_h5" else "hac_h5"
                    ablations.append({"strategy": name, "control": control, "cost_bps": cost, "window": window,
                        **paired_metrics(path.loc[selected], paths[(control, cost)].loc[selected])})
        tables = {"backtest_benchmark_comparison.csv": pd.DataFrame(comparisons),
            "annual_metrics.csv": pd.concat(annual, ignore_index=True), "ablation_comparison.csv": pd.DataFrame(ablations),
            "gates.csv": pd.concat(gates, ignore_index=True), "rolling_summary.csv": pd.concat(rolling, ignore_index=True),
            "bootstrap_analysis.csv": pd.concat(boots, ignore_index=True), "solver_summary.csv": pd.DataFrame(solver_rows),
            "calibration_fits.csv": calibration_table, "forecast_fits.csv": fits,
            "native_audit.csv": pd.DataFrame([{"strategy": n, **a["audit"]} for (n, c), a in native.items()])}
        tables.update({f"expert_{key}.csv": value for key, value in attribution.items()})
        _, current_provider = load_legacy_execution_view(evidence_root)
        fresh_bundle = prepare_horizon_inputs(evidence_root, plan)
        if (current_provider != provider_audit or fresh_bundle["audit"] != bundle["audit"]
                or _bundle_identity(fresh_bundle) != bundle_identity or _bundle_identity(bundle) != bundle_identity
                or _sha(card / "plan.json") != plan_hash
                or (binding["numerical_repair_sha256"] is not None
                    and _sha(card / "numerical_repair.json") != binding["numerical_repair_sha256"])
                or any(_sha(source_root / name) != digest for name, digest in hashes.items())):
            raise ValueError("blocked: execution inputs or code changed")
        for name, table in tables.items():
            table.to_csv(card / name, index=False)
        summary = {"experiment": plan["experiment"], "campaign_status": "active_research", "objective_achieved": False,
            "provider_uri": plan["source"]["provider_uri"], "execution_provider_uri": plan["source"]["execution_provider_uri"],
            "as_of": plan["source"]["as_of"], "region": "us", "market": "all", "benchmark": "SOXX",
            "data_range": [str(market.signal_date.iloc[0].date()), str(market.exit_date.iloc[-1].date())],
            "recorder_id": recorder_id, "experiment_id": recorder.experiment_id, "recorder_uri": uri,
            "recorder_local_dir": recorder.get_local_dir(), "model_selection": "two_registered_candidates_no_post_result_selection",
            "historical_gates": {name: value["historical_gate_pass"] for name, value in evidence.items()},
            "promotion_eligible": False, "decision": "continue_research_no_forward_qualification",
            "evidence_kind": "reused_history", "native_paths": len(native), "native_all_pass": True,
            "plan_sha256": plan_hash, "source_sha256": hashes, "run_binding_sha256": _sha(card / "run_binding.json"),
            "sidecars": list(tables) + ["plan.json", "run_binding.json", "diagnosis.md", "conclusion.md"],
            "numerical_repair_sha256": binding["numerical_repair_sha256"],
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["analysis/paths.pkl", "forecast/predictions.pkl", "models/annual_fits.pkl",
                    "models/oos_floors.pkl", "portfolio_analysis/native_reports.pkl.gz", "analysis/solver_details.pkl.gz"]},
            "qlib_docs": plan["source_references"][:2], "limitations": plan["limitations"]}
        conclusion = _conclusion(plan, summary, tables)
        (card / "conclusion.md").write_text(conclusion)
        with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
            recorder = _same_recorder(R, recorder_id)
            recorder.save_objects(artifact_path="analysis", **{"paths.pkl": paths, "gates.pkl": evidence})
            summary["compressed_artifacts"] = [
                save_compressed_object(recorder, name="native_reports.pkl.gz", artifact_path="portfolio_analysis", value=native),
                save_compressed_object(recorder, name="solver_details.pkl.gz", artifact_path="analysis", value=solver_details)]
            for name in [*tables, "plan.json", "run_binding.json", "workflow.yaml", "conclusion.md", "diagnosis.md"]:
                recorder.log_artifact(str(card / name), artifact_path="card")
            with tempfile.TemporaryDirectory(prefix="qlib-card-summary-") as temporary:
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
