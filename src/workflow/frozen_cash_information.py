"""Recorder-first diagnosis of already frozen cash forecasts; never fit a model.

The integrity loader is callable before the implementation seal. Predictive
statistics are reachable only through the sealed, single-campaign ``run``.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.data.constrained_net_excess import load_constrained_inputs
from src.data.integrated_allocation import (
    COMMON_PRIMARY_FEATURES, FORECAST_LABELS, PRIMARY_FEATURES, PRIMARY_LABELS,
    PRIMARY_METADATA, build_native_dataset, frame_sha256, prepare_mature_rows,
)

EXPERIMENT = "frozen_cash_information_20261003"
CANONICAL_ROOT = Path("/Users/rziiiii/dev/workspace")
ATOL, RTOL = 1e-14, 1e-12


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _require(condition, message):
    if not bool(condition):
        raise ValueError("blocked: " + message)


def _boolean(value):
    return isinstance(value, (bool, np.bool_))


def _no_active_recorder():
    from qlib.workflow import R
    if R.__dict__.get("_provider") is None:
        return
    experiment = R.exp_manager.active_experiment
    _require(experiment is None or experiment.active_recorder is None,
             "native source readback requires no active writable Recorder")


def _source_hashes(root):
    """Explicit used project footprint, not a third-party dependency closure."""
    root = Path(root)
    parent = json.loads((root / "research/benchmark_anchor_utility_20261002/summary.json").read_text())
    names = set(parent["source_sha256"]) | {
        "src/data/constrained_net_excess.py",
        "src/workflow/anchor_checkpoint_feasibility.py",
        "src/workflow/frozen_cash_information.py",
        "src/analysis/frozen_cash_information.py",
    }
    return {name: _sha(root / name) for name in sorted(names)}


def _read_registered_plan(card):
    card = Path(card)
    plan_path, registration_path = card / "diagnostic_plan.json", card / "registration.json"
    plan, registration = json.loads(plan_path.read_text()), json.loads(registration_path.read_text())
    _require(plan["experiment"] == EXPERIMENT and registration["plan_sha256"] == _sha(plan_path),
             "registered diagnostic plan identity differs")
    _require(plan["registered_before_new_cash_predictive_metrics"] is True
             and registration["before_predictive_analysis"] is True,
             "predictive diagnostic was not registered before calculation")
    return plan, {"plan_sha256": _sha(plan_path), "registration_sha256": _sha(registration_path)}


def _require_implementation_seal(root, card, identity):
    root, card = Path(root), Path(card)
    path = card / "implementation_seal.json"
    _require(path.is_file(), "implementation seal is required before predictive analysis")
    seal = json.loads(path.read_text())
    for key, value in identity.items():
        _require(seal.get(key) == value, f"implementation seal {key} differs")
    _require(seal.get("source_sha256") == _source_hashes(root), "sealed project source footprint differs")
    for key in ("static_contracts_passed", "data_contracts_passed", "analysis_contracts_passed"):
        _require(seal.get(key) is True, f"implementation seal {key} is not true")
    _require(seal.get("predictive_metrics_started") is False, "initial seal must precede predictive metrics")
    count = seal.get("focused_passed")
    _require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
             "focused_passed must be a positive integer count")
    tests = seal.get("test_sha256")
    required = {"tests/test_frozen_cash_information_workflow.py", "tests/test_frozen_cash_information.py"}
    _require(isinstance(tests, dict) and required.issubset(tests), "seal omits required workflow/analysis tests")
    for name, digest in tests.items():
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts, "test path escapes project")
        _require(_sha(root / relative) == digest, f"sealed test changed: {name}")
    return seal, _sha(path)


def _compatible_source_plan(plan):
    compatible = deepcopy(plan)
    contract = plan["source_readback_contract"]
    compatible["variants"] = deepcopy(contract["variants"])
    compatible["training"] = deepcopy(contract["training"])
    _require([item["name"] for item in compatible["variants"]] == ["constant", "price", "price_vol"],
             "source readback must reconstruct the exact old allocation fold inventory")
    _require(compatible["training"]["development_signal_years"] == list(range(2016, 2023))
             and compatible["training"]["minimum_rows"] == 252,
             "source readback maturity inventory differs")
    return compatible


def _times(series, name, *, missing=False):
    for value in series.loc[series.notna()]:
        _require(pd.Timestamp(value).tzinfo is not None, f"{name} lacks timezone")
    result = pd.to_datetime(series, utc=True, errors="raise")
    _require(missing or not result.isna().any(), f"{name} has missing historical timestamps")
    return result


def _native_flat(dataset):
    from qlib.data.dataset import DatasetH
    _require(isinstance(dataset, DatasetH), "forecast diagnostic requires native DatasetH")
    frame = dataset.prepare("all", col_set=["feature", "label", "metadata"]).copy()
    _require(frame.index.names == ["datetime", "instrument"]
             and frame.index.get_level_values("instrument").nunique() == 1,
             "unexpected forecast Dataset grain")
    frame.columns = [column[-1] for column in frame.columns]
    frame.index = pd.DatetimeIndex(frame.index.get_level_values("datetime"), name="datetime")
    _require(not frame.index.has_duplicates and frame.index.is_monotonic_increasing and frame.index.tz is None,
             "forecast Dataset dates must be unique ordered Qlib sessions")
    return frame


def _prepare_frozen_heads(raw, forecasts, models, saved_audit, *, as_of, years):
    """Reconstruct original joint-label fit rows and audit cached predictions.

    Only means and prediction parity are computed here; no predictive loss,
    correlation, event profitability or portfolio statistic is calculated.
    """
    dataset = build_native_dataset(raw, feature_columns=PRIMARY_FEATURES, label_columns=PRIMARY_LABELS,
        metadata_columns=PRIMARY_METADATA, as_of=as_of)
    native = _native_flat(dataset)
    expected = raw.loc[:, [*PRIMARY_FEATURES, *PRIMARY_LABELS, *PRIMARY_METADATA]]
    pd.testing.assert_frame_equal(native, expected, check_exact=True, check_freq=False)
    _require(set(models) == set(years) and len(models) == 14, "frozen 14-head inventory differs")
    _require(not saved_audit.model_year.duplicated().any()
             and set(saved_audit.model_year) == set(raw.index.year), "original forecast audit year inventory differs")
    feature_at = _times(raw.feature_available_at, "feature_available_at")
    return_at = _times(raw.return_available_at, "return_available_at", missing=True)
    forecast_at = _times(raw.forecast_available_at, "forecast_available_at", missing=True)
    observed_exit = raw.exit_date.notna()
    _require(not (observed_exit & return_at.isna()).any(), "observed H1 exit lacks maturity metadata")
    _require(not (raw.future_vol_5d.notna() & forecast_at.isna()).any(), "observed H5 target lacks maturity metadata")
    baselines, annual_audit, training_evidence = {}, [], {}
    for year in sorted(set(raw.index.year)):
        cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        eligible = (raw.index >= pd.Timestamp("2011-01-03")) & feature_at.lt(cutoff) & return_at.lt(cutoff) & forecast_at.lt(cutoff)
        strict = raw.loc[eligible, [*PRIMARY_FEATURES, *FORECAST_LABELS, "soxl_return", "soxx_return"]]
        _require(np.isfinite(strict.to_numpy(float)).all()
                 and (strict[["soxl_return", "soxx_return"]] > -1).all().all()
                 and strict.future_vol_5d.ge(0).all(), "invalid mature head input cannot be silently dropped")
        train, audit = prepare_mature_rows(dataset, decision_at=cutoff, label_columns=FORECAST_LABELS,
            availability_column="forecast_available_at")
        dates = pd.DatetimeIndex(train.index.get_level_values("datetime"))
        _require(dates.equals(raw.index[eligible]), "head Dataset silently changed exact eligible dates")
        saved = saved_audit.loc[saved_audit.model_year.eq(year)].iloc[0]
        for key, value in audit.items():
            actual = saved[key]
            if value is None:
                _require(actual is None or pd.isna(actual), f"saved head audit differs: {year}/{key}")
            else:
                _require(actual == value, f"saved head audit differs: {year}/{key}")
        ready = len(train) >= 252
        _require(_boolean(saved.available) and bool(saved.available) == ready
                 and saved.ridge_alpha == 10. and saved.minimum_training_rows == 252,
                 "saved head availability/protocol differs")
        annual_mask = raw.index.year == year
        _require(forecasts.loc[annual_mask, "forecast_available"].eq(float(ready)).all(),
                 "cached annual forecast availability differs")
        item = {"model_year": int(year), "available": ready, "training_rows": len(train),
            "first_training_signal": audit["first_signal_date"], "last_training_signal": audit["last_signal_date"],
            "joint_label_available_at_max": audit["label_available_at_max"], "cutoff": cutoff.isoformat(),
            "training_dates_sha256": hashlib.sha256(dates.asi8.tobytes()).hexdigest(),
            "training_frame_sha256": frame_sha256(train), "prediction_max_absolute_error": None,
            "cash_intercept": None, "cash_training_mean": None, "intercept_mean_absolute_error": None}
        if ready:
            model = models[int(year)]
            steps = getattr(model, "named_steps", {})
            _require(set(steps) == {"standardscaler", "ridge"}, "frozen forecast pipeline steps differ")
            scaler, ridge = steps["standardscaler"], steps["ridge"]
            _require(scaler.with_mean is True and scaler.with_std is True and ridge.fit_intercept is True
                     and ridge.alpha == 10. and model.n_features_in_ == len(COMMON_PRIMARY_FEATURES),
                     "frozen Ridge/scaler contract differs")
            _require(list(model.feature_names_in_) == list(COMMON_PRIMARY_FEATURES),
                     "frozen head feature order differs")
            x = train.loc[:, list(COMMON_PRIMARY_FEATURES)].to_numpy(float)
            y = train.loc[:, list(FORECAST_LABELS)].to_numpy(float)
            from sklearn.preprocessing._data import _is_constant_feature, _handle_zeros_in_scale
            variance = x.var(axis=0)
            scale = _handle_zeros_in_scale(np.sqrt(variance), copy=True,
                constant_mask=_is_constant_feature(variance, x.mean(axis=0), len(train)))
            _require(np.allclose(scaler.mean_, x.mean(axis=0), atol=ATOL, rtol=RTOL)
                     and np.allclose(scaler.var_, variance, atol=ATOL, rtol=RTOL)
                     and np.allclose(scaler.scale_, scale, atol=ATOL, rtol=RTOL)
                     and np.asarray(scaler.n_samples_seen_).size == 1
                     and int(scaler.n_samples_seen_) == len(train), "frozen scaler does not match strict prior rows")
            _require(np.asarray(ridge.coef_).shape == (2, len(COMMON_PRIMARY_FEATURES))
                     and np.asarray(ridge.intercept_).shape == (2,), "frozen joint-label model shape differs")
            means = y.mean(axis=0)
            _require(np.allclose(ridge.intercept_, means, atol=ATOL, rtol=RTOL),
                     "saved intercept differs from exact prior joint-mature label mean")
            prediction = np.asarray(model.predict(raw.loc[annual_mask, list(COMMON_PRIMARY_FEATURES)]), dtype=float)
            cached = forecasts.loc[annual_mask, ["cash_advantage_prediction", "future_vol_prediction"]].to_numpy(float)
            _require(prediction.shape == cached.shape
                     and np.allclose(prediction, cached, atol=ATOL, rtol=RTOL, equal_nan=True),
                     "frozen prediction readback differs")
            available_at = _times(forecasts.loc[annual_mask, "forecast_model_available_at"], "forecast_model_available_at")
            _require(available_at.eq(cutoff.tz_convert("UTC")).all(), "frozen model availability cutoff differs")
            baselines[year] = float(ridge.intercept_[0])
            paired_finite = np.isfinite(prediction) & np.isfinite(cached)
            item.update(prediction_max_absolute_error=float(np.max(np.abs(prediction[paired_finite]-cached[paired_finite]))) if paired_finite.any() else None,
                invalid_sibling_vol_prediction_rows=int((~np.isfinite(prediction[:, 1]) | (prediction[:, 1] <= 0)).sum()),
                cash_intercept=baselines[year], cash_training_mean=float(means[0]),
                intercept_mean_absolute_error=float(abs(ridge.intercept_[0]-means[0])))
            training_evidence[int(year)] = {"dates": dates, "audit": audit,
                "feature_columns": list(COMMON_PRIMARY_FEATURES), "target_columns": list(FORECAST_LABELS),
                "saved_intercept": np.asarray(ridge.intercept_).copy(), "joint_training_target_mean": means,
                "scaler_mean": np.asarray(scaler.mean_).copy(), "scaler_variance": np.asarray(scaler.var_).copy()}
        else:
            _require(year not in models and forecasts.loc[annual_mask, "forecast_model_available_at"].isna().all(),
                     "unavailable year has a model or availability stamp")
        annual_audit.append(item)
    return dataset, pd.DataFrame(annual_audit), training_evidence, baselines


def load_frozen_information_inputs(root, plan):
    """Integrity-only native readback; no loss/IC/event outcomes or model fit."""
    _no_active_recorder()
    bundle = load_constrained_inputs(root, _compatible_source_plan(plan))
    raw, forecasts = bundle["raw"], bundle["forecasts"]
    dataset, annual, training, baselines = _prepare_frozen_heads(raw, forecasts,
        bundle["forecast_models"], bundle["forecast_audit"], as_of=plan["source"]["as_of"], years=plan["forecast"]["years"])
    rows = forecasts.loc[:, [*PRIMARY_METADATA, "forecast_model_available_at", "cash_advantage_prediction",
        "cash_advantage_1d", "soxx_return", "forecast_available"]].copy()
    rows["model_year"] = rows.index.year
    rows["prior_mean_prediction"] = rows.model_year.map(baselines).astype(float)
    rows["forecast_available"] = rows.forecast_available.eq(1.)
    observed_at = pd.Timestamp(plan["source"]["as_of"] + " 09:30", tz="America/New_York")
    maturity = _times(rows.return_available_at, "return_available_at", missing=True)
    finite = np.isfinite(rows[["cash_advantage_1d", "soxx_return"]].to_numpy(float)).all(axis=1)
    known_exit = rows.exit_date.notna() & rows.exit_date.le(pd.Timestamp(plan["source"]["as_of"]))
    _require(not (known_exit & (~finite | maturity.isna() | maturity.gt(observed_at))).any(),
             "observed H1 outcome is invalid or has inconsistent maturity")
    _require(rows.loc[finite, "soxx_return"].gt(-1).all(), "nonphysical observed SOXX H1 return")
    _require(np.allclose(rows.loc[finite, "cash_advantage_1d"], -np.log1p(rows.loc[finite, "soxx_return"]),
                        atol=ATOL, rtol=RTOL), "cash target differs from frozen open-return identity")
    rows["h1_outcome_available"] = finite & maturity.notna() & maturity.le(observed_at) & known_exit
    rows["score_available"] = rows.forecast_available & rows.h1_outcome_available
    _require(np.isfinite(rows.loc[rows.score_available,
             ["cash_advantage_prediction", "prior_mean_prediction"]].to_numpy(float)).all(), "available cash forecasts are invalid")
    _require(rows.loc[~rows.forecast_available, "prior_mean_prediction"].isna().all(), "unavailable baseline must remain missing")
    audit = {**bundle["source_audit"], "diagnostic_rows": len(rows), "available_head_rows": int(rows.forecast_available.sum()),
        "unavailable_head_rows": int((~rows.forecast_available).sum()), "immature_h1_rows": int((~rows.h1_outcome_available).sum()),
        "score_available_rows": int(rows.score_available.sum()), "diagnostic_frame_sha256": frame_sha256(rows),
        "head_years": sorted(baselines), "head_count": len(baselines), "forecast_model_refits": 0,
        "baseline":"saved Ridge cash intercept after joint-training-label mean equality check",
        "evaluation_maturity":"H1 only; H5 availability is required for original training, not cash forecast evaluation",
        "predictive_metrics_computed":False}
    return {**bundle, "forecast_dataset": dataset, "diagnostic_rows": rows,
        "head_readback_audit": annual, "head_training_evidence": training, "information_source_audit": audit}


def _artifact_identity(recorder, artifact):
    base = (Path(recorder.get_local_dir()) / "artifacts").resolve()
    path = (base / artifact).resolve()
    _require(path.is_relative_to(base) and path.is_file(), "child artifact escapes or is absent")
    return {"artifact":artifact, "sha256":_sha(path), "bytes":path.stat().st_size}


def _load_completed_reference(root, card, identity, hashes, seal_sha):
    from qlib.workflow import R
    path = Path(card) / "recorder_reference.json"
    if not path.exists():
        return None
    reference = json.loads(path.read_text())
    _require(reference.get("complete") is True, "partial diagnostic Recorder blocks automatic repetition")
    for key, value in {**identity, "source_sha256":hashes, "implementation_seal_sha256":seal_sha}.items():
        _require(reference.get(key) == value, f"cached diagnostic {key} differs")
    expected_uri = str(CANONICAL_ROOT / "research" / EXPERIMENT / "qlib_recorder")
    _require(reference.get("recorder_uri") == expected_uri, "cached diagnostic Recorder URI differs")
    with R.uri_context(expected_uri):
        recorder = R.get_recorder(experiment_name=EXPERIMENT, recorder_id=reference["recorder_id"])
        _require(recorder.client.get_run(recorder.id).info.status == "FINISHED", "cached diagnostic Recorder is not FINISHED")
        artifacts = reference.get("artifacts")
        _require(isinstance(artifacts, dict) and "analysis/summary.pkl" in artifacts, "cached diagnostic lacks bound summary")
        for artifact, binding in artifacts.items():
            _require(_artifact_identity(recorder, artifact) == binding, "cached diagnostic artifact hash differs")
        summary = recorder.load_object("analysis/summary.pkl")
        for artifact, binding in artifacts.items():
            _require(_artifact_identity(recorder, artifact) == binding, "cached diagnostic changed during read")
    _require((Path(card) / "summary.json").is_file(), "completed child lacks its slim-card summary")
    _require(json.loads((Path(card) / "summary.json").read_text()) == summary,
             "card summary differs from immutable diagnostic summary")
    return summary


def run(root, card):
    """One sealed diagnostic campaign, or verified reuse of its finished R."""
    from qlib.workflow import R
    root, card = Path(root).resolve(), Path(card).resolve()
    plan, identity = _read_registered_plan(card)
    seal, seal_sha = _require_implementation_seal(root, card, identity)
    hashes = seal["source_sha256"]
    _no_active_recorder()
    cached = _load_completed_reference(root, card, identity, hashes, seal_sha)
    if cached is not None:
        return cached
    bundle = load_frozen_information_inputs(root, plan)
    _require(_source_hashes(root) == hashes, "source changed during native integrity readback")
    uri = str(CANONICAL_ROOT / "research" / EXPERIMENT / "qlib_recorder")
    reference_path = card / "recorder_reference.json"
    with R.start(experiment_name=EXPERIMENT, uri=uri):
        recorder = R.get_recorder()
        reference = {**identity, "source_sha256":hashes, "implementation_seal_sha256":seal_sha,
            "recorder_id":recorder.id, "recorder_uri":uri, "complete":False, "stage":"frozen_input_archive"}
        _write_json(reference_path, reference)
        for name in ("diagnostic_plan.json", "registration.json", "implementation_seal.json"):
            recorder.log_artifact(str(card / name), artifact_path="inputs")
        R.save_objects(artifact_path="inputs", **{
            "frozen_forecasts.pkl":bundle["forecasts"], "diagnostic_rows.pkl":bundle["diagnostic_rows"],
            "source_details.pkl":bundle["source_details"], "source_audit.pkl":bundle["information_source_audit"],
            "head_readback_audit.pkl":bundle["head_readback_audit"],
            "head_training_evidence.pkl":bundle["head_training_evidence"]})
        R.save_objects(artifact_path="models", **{"frozen_forecast_models.pkl":bundle["forecast_models"]})
        reference["stage"] = "registered_predictive_diagnostic"
        _write_json(reference_path, reference)
        from src.analysis.frozen_cash_information import analyze_information
        results = analyze_information(bundle["diagnostic_rows"], plan)
        tables = {name + ".csv": results[name] for name in
                  ("point_metrics", "calibration", "event_metrics", "bootstrap_summary")}
        _require(isinstance(tables, dict) and tables, "analysis omitted small summary tables")
        for name, table in tables.items():
            _require(Path(name).name == name and name.endswith(".csv") and isinstance(table, pd.DataFrame),
                     "analysis table path/type differs")
            table.to_csv(card / name, index=False)
        bundle["head_readback_audit"].to_csv(card / "head_readback_audit.csv", index=False)
        R.save_objects(artifact_path="analysis", **{"tables.pkl":tables, "bootstrap_draws.pkl":results["bootstrap_draws"],
            "diagnostic_audit.pkl":results["audit"]})
        summary = {"experiment":EXPERIMENT,"record_type":"frozen_forecast_information_diagnostic",
            "recorder_id":recorder.id,"recorder_uri":uri,"recorder_local_dir":recorder.get_local_dir(),
            "recorder_relative_dir":str(Path(recorder.get_local_dir()).relative_to(Path(uri))),
            **identity,"source_sha256":hashes,"implementation_seal_sha256":seal_sha,
            "source":plan["source"],"provider_uri":plan["source"]["provider_uri"],"region":"us","market":"all",
            "source_audit":bundle["information_source_audit"],"analysis_audit":results["audit"],
            "objective_achieved":False,"strategy_candidate_promotable":False,"new_forward_qualified":False,
            "historical_years_reused":True,"market_model_fits":0,"forecast_model_refits":0,
            "market_backtests":0,"new_portfolio_paths":0,"production_change":False,"real_broker_orders":False,
            "frozen_forecast_head_count":14,"sidecars":[*tables,"head_readback_audit.csv"],
            "artifact_policy":plan["artifact_policy"],"qlib_docs_reference":plan["qlib_docs"],
            "decision":"Information diagnostic only; no financial qualification or production change.",
            "source_coverage":"Explicit frozen-parent and directly used project helpers; not a complete third-party dependency closure.",
            "limitations":[plan["source_limits"],plan["event_diagnostic"]["limitations"],plan["uncertainty"]["interpretation"]]}
        _require(_source_hashes(root) == hashes, "source changed during predictive analysis")
        _require(_read_registered_plan(card)[1] == identity, "registration changed during analysis")
        R.save_objects(artifact_path="analysis", **{"summary.pkl":summary})
        _write_json(card / "summary.json", summary)
        artifacts = {}
        for path in sorted((Path(recorder.get_local_dir()) / "artifacts").rglob("*")):
            if path.is_file() and path.relative_to(Path(recorder.get_local_dir()) / "artifacts").parts[0] in {"inputs","models","analysis"}:
                name = path.relative_to(Path(recorder.get_local_dir()) / "artifacts").as_posix()
                artifacts[name] = _artifact_identity(recorder, name)
        reference.update(stage="complete", artifacts=artifacts)
    # Completion is published only after R.end_exp marks the same run FINISHED.
    with R.uri_context(uri):
        finished = R.get_recorder(experiment_name=EXPERIMENT, recorder_id=reference["recorder_id"])
        _require(finished.client.get_run(finished.id).info.status == "FINISHED", "diagnostic Recorder did not finish")
    reference["complete"] = True
    _write_json(reference_path, reference)
    return summary
