"""Frozen conditional-risk inputs with two matched asset representations."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import traceback

import numpy as np
import pandas as pd

from src.analysis.excess_resolution import apply_excess_resolution
from src.analysis.integrated_allocation import ASSETS, paired_metrics
from src.backtest.benchmark_relative_replay import replay_fixed_target
from src.backtest.target_schedule_replay import replay_target_schedule
from src.core.recorder_artifacts import save_compressed_object
from src.core.recorder_sources import load_verified_recorder_objects
from src.data.conditional_variance import apply_calibration, RISK_UNITS
from src.data.convex_allocation import FEATURE_COLUMNS
from src.data.integrated_allocation import frame_sha256
from src.data.legacy_execution_view import load_legacy_execution_view
from src.data.oos_horizon_inputs import prepare_horizon_inputs
from src.data.price_basis import _sha as structured_sha
from src.model.benchmark_core_risk import benchmark_core_allocation
from src.workflow.benchmark_evidence import evaluate_registered_paths
from src.workflow.benchmark_relative_research import _json, _sha, _same_recorder, _native_path
from src.workflow.oos_uncertainty_horizon import _source_hashes as inherited_hashes, _bundle_identity
from src.workflow.benchmark_risk_excess import COVARIANCE_FIELDS


PARENT_ARTIFACTS = ("inputs/raw_variances.pkl", "inputs/annual_calibrations.pkl", "inputs/primary_frame.pkl",
    "inputs/forecast_frames.pkl", "inputs/source_audit.pkl", "policy/development_inputs.pkl")
SOURCES = ("src/workflow/benchmark_core_variance.py", "src/model/benchmark_core_risk.py",
    "src/backtest/target_schedule_replay.py", "src/core/recorder_sources.py", "src/analysis/excess_resolution.py",
    "src/workflow/benchmark_evidence.py")
PARENT_VARIANTS = {"learned": "learned_vol5", "close20": "observed_close20"}


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _same_frame(first, second, message):
    try:
        pd.testing.assert_frame_equal(first, second, check_exact=True)
    except AssertionError as error:
        raise ValueError("blocked: " + message) from error


def _validate_plan(plan):
    expected = [("learned_lever_cash", "learned", "lever_cash"), ("learned_core", "learned", "core"),
        ("close20_lever_cash", "close20", "lever_cash"), ("close20_core", "close20", "core")]
    actual = [(v["name"], v["variance_key"], v["representation"]) for v in plan["variants"]]
    _require(plan["schema_version"] == "benchmark_core_variance_v1"
        and plan["registered_before_candidate_portfolio_results"] is True and actual == expected
        and all(type(plan[name]) is int for name in ("model_fit_budget", "risk_calibration_fit_budget", "candidate_budget", "rebalance_sessions"))
        and plan["model_fit_budget"] == plan["risk_calibration_fit_budget"] == 0
        and plan["candidate_budget"] == 2 and plan["rebalance_sessions"] == 1
        and plan["costs_bps"] == [10., 25.] and plan["benchmark"] == "SOXX"
        and type(plan["gates"]["family_size"]) is int and plan["gates"]["family_size"] == 2
        and plan["gates"] == {"minimum_positive_years": 6, "rolling_sessions": 252,
            "minimum_rolling_win_rate": .6, "bootstrap_resamples": 2000, "family_size": 2}
        and plan["numerical_log_excess_floor"] == 1e-8 and plan["cash_return"] == 0.
        and plan["evidence_kind"] == "reused_history" and plan["development_years"] == list(range(2016, 2023))
        and type(plan["risk_parent"]["repeat_id"]) is int and plan["risk_parent"]["repeat_id"] == 1
        and plan["risk_parent"]["manifest"] == "artifact_manifest.json"
        and set(plan["risk_parent"]["artifact_sha256"]) == set(PARENT_ARTIFACTS),
        "registered benchmark-core risk protocol differs")
    stamp = pd.Timestamp(plan["registered_at_utc"])
    _require(stamp.tzinfo is not None and stamp <= pd.Timestamp.now(tz="UTC"), "invalid registration clock")


def _hashes(root):
    result = inherited_hashes(root)
    for name in SOURCES:
        module = importlib.import_module(name[:-3].replace("/", "."))
        _require(Path(module.__file__).resolve() == (root / name).resolve(), "imported core-risk source differs")
        result[name] = _sha(root / name)
    return result


def _load_risk_parent(root, plan):
    """Explicit legacy-manifest contract; never probe an alternate filename."""
    parent = plan["risk_parent"]
    return load_verified_recorder_objects(root / parent["card"], expected_recorder_id=parent["recorder_id"],
        artifacts=list(PARENT_ARTIFACTS), expected_artifact_sha256=parent["artifact_sha256"],
        manifest_name=parent["manifest"], expected_manifest_sha256=parent["manifest_sha256"])


def reconstruct_frozen_risk_inputs(bundle, objects, plan):
    """Apply fixed repeat-one risk parameters; no mean objective or refitting."""
    frame, primary = bundle["frame"], objects["inputs/primary_frame.pkl"]
    current_source, parent_source = bundle["audit"]["qualified_source"], objects["inputs/source_audit.pkl"]
    identity_fields = ("parent_recorder_id", "parent_summary_sha256", "provider_manifest_sha256",
                       "native_frame_sha256", "as_of", "features", "source_api")
    _require(frame.index.equals(primary.index)
        and bundle["audit"]["original_frame_sha256"] == frame_sha256(primary)
        and all(name in current_source and name in parent_source and current_source[name] == parent_source[name]
                for name in identity_fields),
        "risk-parent source identity differs from qualified primary inputs")
    _same_frame(frame[list(FEATURE_COLUMNS)], primary[list(FEATURE_COLUMNS)], "risk-parent causal feature values differ")
    repeat = plan["risk_parent"]["repeat_id"]
    raw = objects["inputs/raw_variances.pkl"][repeat]
    forecast = objects["inputs/forecast_frames.pkl"][repeat]
    _require(raw.index.equals(primary.index) and forecast.index.equals(primary.index), "frozen risk/forecast calendars differ")
    _same_frame(forecast[list(FEATURE_COLUMNS)], primary[list(FEATURE_COLUMNS)], "frozen risk forecasts use different causal features")
    time_fields = ("feature_available_at", "return_available_at", "forecast_available_at", "h5_available_at", "forecast_model_available_at")
    for name in time_fields:
        _require(all(pd.isna(value) or pd.Timestamp(value).tzinfo is not None for value in forecast[name]),
                 "frozen forecast availability must have explicit timezones")
    _require(all(pd.isna(value) or pd.Timestamp(value).tzinfo is not None for value in raw.forecast_model_available_at),
             "frozen raw-risk model availability must have explicit timezone")
    records = [row for row in objects["inputs/annual_calibrations.pkl"] if row["repeat_id"] == repeat]
    years = list(range(min(plan["development_years"]), pd.Timestamp(plan["source"]["as_of"]).year + 1))
    _require(len(records) == len(years) and sorted(row["year"] for row in records) == years,
             "fixed repeat must provide exactly every registered annual calibration")
    calibrations = {row["year"]: row["calibration"] for row in records}
    calibration_rows = []
    for year, calibration in sorted(calibrations.items()):
        cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        _require(calibration["year"] == year and pd.Timestamp(calibration["cutoff"]) == cutoff
                 and calibration["risk_units"] == RISK_UNITS, "frozen calibration identity or units differ")
        common = (forecast.index >= pd.Timestamp(plan["train_start"])) & forecast.forecast_available.eq(1).to_numpy()
        for name in time_fields:
            stamps = pd.to_datetime(forecast[name], utc=True)
            common &= stamps.lt(cutoff).to_numpy()
        common_dates = forecast.index[common].strftime("%Y-%m-%d").tolist()
        audit = calibration["audit"]
        _require(common_dates == audit["common_dates"] and len(common_dates) == audit["common_rows"]
            and structured_sha(common_dates) == audit["common_dates_sha256"]
            and all(pd.Timestamp(audit[name]) < cutoff for name in ("h5_label_available_at_max", "label_available_at_max")),
            "calibration contains mismatched or post-cutoff outcome rows")
        _require(np.isfinite(calibration["B"]) and calibration["B"] > 0
            and all(calibration["c"][key] is not None and np.isfinite(calibration["c"][key])
                    and calibration["c"][key] > 0 for key in PARENT_VARIANTS),
            "registered frozen variance estimator is unavailable")
        calibration_rows.append({"year": year, "cutoff": cutoff.isoformat(), "B": calibration["B"],
            "c_learned": calibration["c"]["learned"], "c_close20": calibration["c"]["close20"],
            "common_rows": len(common_dates), "common_dates_sha256": audit["common_dates_sha256"],
            "latest_label_available_at": audit["label_available_at_max"]})
    mask = (frame.index >= pd.Timestamp(plan["signal_start"])) & frame.exit_date.notna()
    market, raw_market = frame.loc[mask].copy(), raw.loc[frame.index[mask]].copy()
    frozen_forecast = forecast.loc[market.index]
    _require(raw_market.forecast_available.eq(1).all() and frozen_forecast.forecast_available.eq(1).all()
        and np.array_equal(raw_market.future_vol_prediction.to_numpy(), frozen_forecast.future_vol_prediction.to_numpy())
        and np.isfinite(raw_market.future_vol_prediction).all() and raw_market.future_vol_prediction.gt(0).all()
        and np.array_equal(raw_market.m_learned.to_numpy(), np.square(raw_market.future_vol_prediction.to_numpy())),
        "learned variance lacks the same positive frozen forecast on every market session")
    model_times = pd.to_datetime(raw_market.forecast_model_available_at, utc=True)
    forecast_model_times = pd.to_datetime(frozen_forecast.forecast_model_available_at, utc=True)
    expected_models = pd.DatetimeIndex([pd.Timestamp(f"{year}-01-01", tz="America/New_York") for year in market.index.year]).tz_convert("UTC")
    _require(pd.DatetimeIndex(model_times).equals(expected_models)
        and pd.DatetimeIndex(forecast_model_times).equals(expected_models)
        and model_times.lt(pd.to_datetime(market.decision_at, utc=True)).all()
        and pd.to_datetime(frozen_forecast.feature_available_at, utc=True).lt(pd.to_datetime(market.decision_at, utc=True)).all(),
        "frozen risk model or features were unavailable at the new decision clock")
    packed = raw_market.loc[:, list(COVARIANCE_FIELDS)].to_numpy(float)
    shape = np.empty((len(market), 2, 2))
    shape[:, 0, 0], shape[:, 0, 1] = packed[:, 0], packed[:, 1]
    shape[:, 1, 0], shape[:, 1, 1] = packed[:, 1], packed[:, 2]
    _require(np.array_equal(shape, bundle["covariance"][np.flatnonzero(mask)]), "past-only covariance shape differs from native price source")
    inputs, applied_rows, reproduction = {}, [], []
    for variant in PARENT_VARIANTS:
        covariance, budget = np.empty_like(shape), np.empty(len(market))
        support, fallback = np.zeros(len(market), bool), np.zeros(len(market), bool)
        fit_year = np.empty(len(market), int)
        provenance = [None] * len(market)
        for year in years:
            positions = np.flatnonzero(market.index.year == year)
            _require(len(positions) > 0, "registered risk year is missing from market inputs")
            calibration = calibrations[year]
            applied = apply_calibration(calibration, raw_market.iloc[positions], variant)
            _require(not applied["fallback"].any(), "registered risk estimator fallback blocks the entire comparison")
            covariance[positions], budget[positions] = applied["covariance"], applied["benchmark_variance"]
            support[positions], fallback[positions], fit_year[positions] = applied["out_of_support"], applied["fallback"], year
            identity = {"model_year": year, "cutoff": calibration["cutoff"],
                "common_dates_sha256": calibration["audit"]["common_dates_sha256"],
                "units": RISK_UNITS, "parameter_frozen_at_cutoff": True}
            for position in positions:
                provenance[position] = identity
            applied_rows.append({"variance_key": variant, "year": year, **applied["audit"]})
        value = {"signal_dates": market.index.copy(), "covariance": covariance, "budget": budget,
            "support": support, "estimator_fallback": fallback, "calibration_year": fit_year, "budget_provenance": provenance}
        development = market.index.year.isin(plan["development_years"])
        prior = objects["policy/development_inputs.pkl"][(repeat, PARENT_VARIANTS[variant])]
        fields = ("covariance", "budget", "support", "estimator_fallback", "calibration_year")
        _require(pd.DatetimeIndex(prior["signal_dates"]).equals(market.index[development])
            and all(np.array_equal(value[name][development], prior[name]) for name in fields)
            and [provenance[i] for i in np.flatnonzero(development)] == prior["budget_provenance"],
            "reconstructed development risk inputs differ from the frozen parent")
        reproduction.append({"variance_key": variant, "repeat_id": repeat, "rows": int(development.sum()),
            "covariance_budget_support_provenance_exact": True, "diagnostic_rebuilt_from_raw_calibrations": True,
            "old_diagnostic_inputs_used": False, "mean_objective_consumed": False})
        inputs[variant] = value
    return market, inputs, {"calibrations": pd.DataFrame(calibration_rows),
        "applications": pd.DataFrame(applied_rows), "development_reproduction": pd.DataFrame(reproduction),
        "model_fits": 0, "calibration_fits": 0,
        "source_identity": {"exact_identity_fields": list(identity_fields),
            "parent_declared_provider_uri": parent_source["provider_uri"],
            "current_verified_provider_uri": current_source["provider_uri"],
            "parent_metadata_modified": False},
        "old_label_precision_preserved_in_frozen_risk_fit": True,
        "new_float64_returns_used_only_for_current_portfolio_accounting": True}


def resolve_risk_evidence(tables, raw, paths, plan, controls):
    result, resolved = {name: table.copy(deep=True) for name, table in tables.items()}, {}
    years = plan["development_years"]
    dev = lambda frame: frame.loc[frame.signal_date.dt.year.isin(years)]
    for candidate in controls:
        resolved[candidate] = apply_excess_resolution(raw[candidate],
            {c: dev(paths[(candidate, c)]) for c in plan["costs_bps"]},
            {c: dev(paths[("soxx", c)]) for c in plan["costs_bps"]}, log_excess_floor=plan["numerical_log_excess_floor"])
    for filename, field in (("gates.csv", "gates"), ("rolling_summary.csv", "rolling_summary"), ("bootstrap_analysis.csv", "bootstrap")):
        result["raw_" + filename] = tables[filename].copy(deep=True)
        result[filename] = pd.concat([value[field].assign(strategy=name) for name, value in resolved.items()], ignore_index=True)
    result["resolved_annual_metrics.csv"] = pd.concat([value["annual"].assign(strategy=name) for name, value in resolved.items()], ignore_index=True)
    return result, resolved


def _risk_weights_audit(target, realized, covariance, budgets, *, layer):
    """Separate certified targets from roundoff in settled/native holdings.

    The allowed risk difference is the actual weight perturbation's algebraic
    quadratic bound plus floating arithmetic roundoff. This does not increase
    a target budget or alter a portfolio/gate: deviations remain reported.
    """
    target, realized = np.asarray(target, float), np.asarray(realized, float)
    _require(target.shape == realized.shape and np.isfinite(realized).all()
        and (realized >= -1e-12).all() and np.allclose(realized.sum(axis=1), 1., atol=1e-12, rtol=0),
        "invalid independently realized post-fee holdings")
    change = np.abs(realized[:, :2] - target[:, :2])
    maximum = float(np.max(np.abs(realized - target)))
    _require(maximum <= 1e-10, "realized post-fee weights differ materially from the risk target")
    absolute = np.abs(covariance)
    perturbation = (2 * np.einsum("ni,nij,nj->n", change, absolute, np.abs(target[:, :2]))
                    + np.einsum("ni,nij,nj->n", change, absolute, change))
    roundoff = 64 * np.finfo(float).eps * np.einsum("ni,nij,nj->n", np.abs(realized[:, :2]), absolute, np.abs(realized[:, :2]))
    variance = np.einsum("ni,nij,nj->n", realized[:, :2], covariance, realized[:, :2])
    _require(np.isfinite(variance).all() and np.all(variance - budgets <= perturbation + roundoff),
        "realized risk excess exceeds its measured numerical weight perturbation bound")
    return {"layer": layer, "sessions": len(target), "maximum_target_weight_difference": maximum,
        "maximum_realized_variance_budget_ratio": float(np.max(variance / budgets)),
        "maximum_positive_variance_budget_excess": float(np.maximum(variance - budgets, 0).max()),
        "strict_float_variance_exceedance_rows": int((variance > budgets).sum()),
        "maximum_measured_perturbation_bound": float(np.max(perturbation + roundoff)),
        "within_measured_numerical_bound": True, "target_or_economic_budget_modified": False}


def _native_open_weights(native, path, provider):
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    quotes = D.features(["SOXL", "SOXX"], ["$open"], start_time=path.trade_date.iloc[0],
        end_time=path.trade_date.iloc[-1], freq="day")["$open"].unstack("instrument").astype(float)
    rows = []
    for date in path.trade_date:
        position = native["positions"][date]
        amounts = position.get_stock_amount_dict()
        values = np.array([float(amounts.get(symbol, 0.)) * float(quotes.loc[date, symbol])
                           for symbol in ("SOXL", "SOXX")] + [float(position.get_cash())])
        _require(np.isfinite(values).all() and values.sum() > 0, "invalid native entry NAV for risk audit")
        rows.append(values / values.sum())
    return np.vstack(rows)


def _conclusion(summary, tables):
    lines = ["# 相同冻结条件风险下的基准核心与杠杆/现金表示", "",
        "本批没有训练预测或重拟合风险校准：固定使用父研究repeat1与2016–2026全部年度参数，开发输入逐值复现；后段从原始风险表重建，没有冒用旧扩展波动诊断输入。", "",
        "25bp完整结果如下：", ""]
    for row in tables["backtest_benchmark_comparison.csv"].query("cost_bps==25").itertuples():
        lines.append(f"- {row.window}/{row.strategy}：收益{row.return_pct:.2f}%，相对SOXX财富{row.relative_wealth:.6f}，最大回撤{row.max_drawdown_pct:.2f}%，CVaR5 {row.cvar5_pct:.2f}%，换手{row.turnover:.4f}。")
    lines += ["", f"两个预登记候选的有效历史门：{summary['historical_gates']}。不晋升，整体稳定超额目标继续。",
        "", "相同风险源内core与lever_cash隔离持仓表示；learned_core与close20_core另作全成本比较。对SOXX的点估计优势不能自动证明学习波动预测有独立信息。",
        "", "风险映射只使用过去可得协方差与冻结预算，既不读取均值也不调用凸优化器；协方差仅允许数值roundoff修复。目标风险逐行核验，预算因全仓上限未用满时如实登记。日交易按各自漂移持仓独立扣费，全部由原生Qlib重放。",
        "", "年度/滚动与原始、1e-8有效正超额分类完整保留；支持域外标记不裁剪仓位。历史复用、旧风险标签精度、报价口径与16条次来源、连续份额、开盘成交及现金零收益仍有限制。下一步先审查信息增量、交易成本和尾部差异，再登记独立机制，不按本轮结果改预算或选风险源。",
        "", "冻结价格不等于已用独立分配账本核验的总回报或完整历史PIT数据；任何历史通过仍需独立来源敏感性和真实分配账本验证。本批不混入后取得的分配信息，也不据此调参。",
        "", f"Qlib Recorder：{summary['recorder_id']}；风险父Recorder：{summary['risk_parent_recorder_id']}；as_of={summary['as_of']}。"]
    return "\n".join(lines) + "\n"


def run_benchmark_core_variance(*, source_root, evidence_root, card_dir, recorder_uri):
    import qlib
    from qlib.workflow import R

    source_root, evidence_root, card = (Path(value).resolve() for value in (source_root, evidence_root, card_dir))
    _require(not (card / "summary.json").exists(), "completed core-risk card is immutable")
    plan = json.loads((card / "plan.json").read_text())
    _validate_plan(plan)
    hashes, plan_hash = _hashes(source_root), _sha(card / "plan.json")
    bundle = prepare_horizon_inputs(evidence_root, plan)
    provider, provider_audit = load_legacy_execution_view(evidence_root)
    _require(provider == (evidence_root / plan["source"]["execution_provider_uri"]).resolve(), "execution provider differs")
    objects, parent_audit = _load_risk_parent(evidence_root, plan)
    identity, uri = _bundle_identity(bundle), str(Path(recorder_uri).resolve())
    binding = {"plan_sha256": plan_hash, "source_sha256": hashes, "provider_audit": provider_audit,
        "data_audit": bundle["audit"], "bundle_identity": identity, "verified_risk_parent": parent_audit,
        "source_root": str(source_root), "read_only_evidence_root": str(evidence_root), "card_dir": str(card),
        "recorder_uri": uri, "qlib_version": qlib.__version__, "model_fits": 0, "risk_calibration_fits": 0}
    _json(card / "run_binding.json", binding)
    with R.start(experiment_name=plan["experiment"], uri=uri):
        recorder = R.get_recorder()
        rid = recorder.id
        recorder.log_params(plan_sha256=plan_hash, candidate_budget=2, model_fit_budget=0, risk_calibration_fit_budget=0)
        recorder.save_objects(artifact_path="inputs", **{"plan.pkl": plan, "binding.pkl": binding, "frame.pkl": bundle["frame"]})
        for name in hashes:
            recorder.log_artifact(str(source_root / name), artifact_path="source/" + str(Path(name).parent))
    paths, native, mappings, target_tables, replay_rows, risk_realization_rows = {}, {}, {}, {}, [], []
    stage = "reconstruct_frozen_risk_inputs"
    try:
        market, risk_inputs, risk_audit = reconstruct_frozen_risk_inputs(bundle, objects, plan)
        mapping_rows = []
        for variant in plan["variants"]:
            name, key = variant["name"], variant["variance_key"]
            mapping = benchmark_core_allocation(risk_inputs[key]["covariance"], risk_inputs[key]["budget"], representation=variant["representation"])
            mappings[name] = mapping
            target_tables[name] = pd.DataFrame(mapping["weights"], index=market.index, columns=ASSETS)
            _require(np.all(mapping["portfolio_variance"] <= risk_inputs[key]["budget"]), "risk mapping exceeds its frozen budget")
            mapping_rows.append({"strategy": name, "variance_key": key, "representation": variant["representation"],
                "sessions": len(market), "maximum_budget_used_fraction": float(mapping["budget_used_fraction"].max()),
                "budget_underused_sessions": int((~mapping["budget_targeted"]).sum()),
                "maximum_covariance_roundoff_adjustment": mapping["audit"]["maximum_covariance_roundoff_adjustment"],
                "out_of_support_sessions": int(risk_inputs[key]["support"].sum()), "estimator_fallback_sessions": 0,
                "all_risk_targets_certified": True, "mean_objective_used": False, "optimizer_solves": 0})
        for cost in plan["costs_bps"]:
            stage = f"native_soxx_{cost:g}bps"
            raw = replay_fixed_target(market, target=[0., 1., 0.], cost_bps=cost, rebalance_sessions=1, buy_and_hold=True)
            paths[("soxx", cost)], native[("soxx", cost)] = _native_path(market, raw, provider, cost, plan["initial_cash"])
            for variant in plan["variants"]:
                name, key = variant["name"], variant["variance_key"]
                stage = f"funded_and_native_{name}_{cost:g}bps"
                raw, audit = replay_target_schedule(market, target_tables[name], cost_bps=cost)
                raw["portfolio_variance"] = mappings[name]["portfolio_variance"]
                raw["risk_budget_variance"] = risk_inputs[key]["budget"]
                raw["risk_budget_used_fraction"] = mappings[name]["budget_used_fraction"]
                raw["out_of_support"] = risk_inputs[key]["support"]
                raw["estimator_fallback"] = False
                risk_realization_rows.append({"strategy": name, "cost_bps": cost, **_risk_weights_audit(
                    mappings[name]["weights"], raw[[f"settled_open_{asset}_weight" for asset in ASSETS]].to_numpy(float),
                    mappings[name]["covariance"], risk_inputs[key]["budget"], layer="independent_current_fee_settlement")})
                replay_rows.append({"strategy": name, **audit})
                paths[(name, cost)], native[(name, cost)] = _native_path(market, raw, provider, cost, plan["initial_cash"])
                risk_realization_rows.append({"strategy": name, "cost_bps": cost, **_risk_weights_audit(
                    mappings[name]["weights"], _native_open_weights(native[(name, cost)], paths[(name, cost)], provider),
                    mappings[name]["covariance"], risk_inputs[key]["budget"], layer="native_post_fee_open_positions")})
                print(f"Native core-risk path passed: {name}, {cost:g}bp", flush=True)
        controls = {"learned_core": "learned_lever_cash", "close20_core": "close20_lever_cash"}
        stage = "raw_and_resolved_evidence"
        raw_tables, raw_evidence = evaluate_registered_paths(paths, plan=plan, calendar=bundle["calendar"], candidate_controls=controls)
        tables, evidence = resolve_risk_evidence(raw_tables, raw_evidence, paths, plan, controls)
        source_comparison = []
        for cost in plan["costs_bps"]:
            left, right = paths[("learned_core", cost)], paths[("close20_core", cost)]
            for window, mask in (("development", left.signal_date.dt.year.isin(plan["development_years"])),
                                 ("reused_diagnostic", left.signal_date.dt.year.gt(max(plan["development_years"])))):
                source_comparison.append({"candidate": "learned_core", "reference": "close20_core", "window": window,
                    "cost_bps": cost, **paired_metrics(left.loc[mask], right.loc[mask])})
        tables.update({"risk_mapping_summary.csv": pd.DataFrame(mapping_rows), "replay_audit.csv": pd.DataFrame(replay_rows),
            "risk_realization_audit.csv": pd.DataFrame(risk_realization_rows),
            "calibration_identity.csv": risk_audit["calibrations"], "development_input_reproduction.csv": risk_audit["development_reproduction"],
            "variance_source_comparison.csv": pd.DataFrame(source_comparison),
            "native_audit.csv": pd.DataFrame([{"strategy": name, **value["audit"]} for (name, cost), value in native.items()])})
        stage = "end_source_identity_guard"
        fresh = prepare_horizon_inputs(evidence_root, plan)
        _, fresh_provider = load_legacy_execution_view(evidence_root)
        _, fresh_parent = _load_risk_parent(evidence_root, plan)
        _require(fresh_parent == parent_audit and fresh_provider == provider_audit and fresh["audit"] == bundle["audit"]
            and _bundle_identity(fresh) == identity and _bundle_identity(bundle) == identity
            and _sha(card / "plan.json") == plan_hash and all(_sha(source_root / name) == digest for name, digest in hashes.items()),
            "core-risk source, parent identity, plan or running implementation changed")
        for name, table in tables.items():
            table.to_csv(card / name, index=False)
        summary = {"experiment": plan["experiment"], "campaign_status": "active_research", "objective_achieved": False,
            "provider_uri": plan["source"]["provider_uri"], "execution_provider_uri": plan["source"]["execution_provider_uri"],
            "as_of": plan["source"]["as_of"], "market": "all", "region": "us", "benchmark": "SOXX",
            "data_range": [str(market.signal_date.iloc[0].date()), str(market.exit_date.iloc[-1].date())],
            "recorder_id": rid, "experiment_id": recorder.experiment_id, "recorder_uri": uri, "recorder_local_dir": recorder.get_local_dir(),
            "risk_parent_recorder_id": plan["risk_parent"]["recorder_id"], "risk_repeat_id": 1,
            "model_fits": 0, "risk_calibration_fits": 0, "optimizer_solves": 0, "model_selection": "none_fixed_two_risk_sources",
            "plan_sha256": plan_hash, "source_sha256": hashes, "run_binding_sha256": _sha(card / "run_binding.json"),
            "native_paths": len(native), "native_all_pass": True,
            "raw_historical_gates": {name: value["historical_gate_pass"] for name, value in raw_evidence.items()},
            "historical_gates": {name: value["historical_gate_pass"] for name, value in evidence.items()},
            "excess_resolution": {name: value["excess_resolution"] for name, value in evidence.items()},
            "promotion_eligible": False, "evidence_kind": "reused_history", "decision": "continue_research_no_forward_qualification",
            "source_sensitivity_completed": False, "historical_vintage_certified": False,
            "independently_verified_total_return": False,
            "risk_columns_scope": "portfolio_variance/risk_budget_used_fraction are exact target certificates; settled and native floating holdings are separately measured in risk_realization_audit.csv",
            "sidecars": [*tables, "plan.json", "run_binding.json", "conclusion.md"],
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/frame.pkl", "inputs/frozen_risk_inputs.pkl.gz", "analysis/paths.pkl",
                    "analysis/risk_mapping_details.pkl.gz", "analysis/raw_gates.pkl", "analysis/gates.pkl", "portfolio_analysis/native_reports.pkl.gz"]},
            "qlib_docs": plan["source_references"][:2], "limitations": plan["limitations"]}
        conclusion = _conclusion(summary, tables)
        (card / "conclusion.md").write_text(conclusion)
        stage = "archive_success"
        with R.start(experiment_name=plan["experiment"], recorder_id=rid, uri=uri, resume=True):
            recorder = _same_recorder(R, rid)
            recorder.save_objects(artifact_path="analysis", **{"paths.pkl": paths, "raw_gates.pkl": raw_evidence, "gates.pkl": evidence})
            summary["compressed_artifacts"] = [
                save_compressed_object(recorder, name="frozen_risk_inputs.pkl.gz", artifact_path="inputs", value={"inputs": risk_inputs, "audit": risk_audit}),
                save_compressed_object(recorder, name="risk_mapping_details.pkl.gz", artifact_path="analysis", value=mappings),
                save_compressed_object(recorder, name="native_reports.pkl.gz", artifact_path="portfolio_analysis", value=native)]
            for name in [*tables, "plan.json", "run_binding.json", "workflow.yaml", "conclusion.md"]:
                recorder.log_artifact(str(card / name), artifact_path="card")
            with tempfile.TemporaryDirectory(prefix="core-risk-card-") as temporary:
                staged = Path(temporary) / "summary.json"
                _json(staged, summary)
                recorder.log_artifact(str(staged), artifact_path="card")
        _json(card / "summary.json", summary)
    except Exception as error:
        with R.start(experiment_name=plan["experiment"], recorder_id=rid, uri=uri, resume=True):
            recorder = _same_recorder(R, rid)
            recorder.save_objects(artifact_path="failure", **{"exception.pkl": {"stage": stage,
                "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                "financial_recipe_changed": False}, "partial_paths.pkl": paths})
            save_compressed_object(recorder, name="partial_mapping.pkl.gz", artifact_path="failure", value=mappings)
            if native:
                save_compressed_object(recorder, name="partial_native.pkl.gz", artifact_path="failure", value=native)
            raise
    for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
        (Path(recorder.get_local_dir()) / "artifacts" / name).unlink(missing_ok=True)
    return summary, tables, conclusion
