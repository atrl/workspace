from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.convex_qlib import native_open_reconciliation
from src.backtest.target_schedule_replay import replay_target_schedule
from src.data.convex_allocation import FEATURE_COLUMNS
from src.data.integrated_allocation import frame_sha256
from src.data.price_basis import _sha as structured_sha
from src.data.qlib_native import build_qlib_store_from_frame
from src.workflow.conditional_variance import prepare_policy_inputs
from src.workflow import benchmark_core_variance as workflow


@pytest.fixture
def plan():
    return json.loads((Path(__file__).resolve().parents[1] /
        "research/benchmark_core_variance_20261009/plan.json").read_text())


@pytest.fixture(scope="module")
def risk_source():
    calendar = pd.bdate_range("2014-01-02", "2026-09-25")
    calendar = calendar[calendar != pd.Timestamp("2016-01-01")]
    dates = pd.Series(calendar, index=calendar)
    primary = pd.DataFrame(.1, index=calendar, columns=FEATURE_COLUMNS)
    primary.index.name = "datetime"

    def clock(values, hour, minute=0):
        return pd.Series(pd.DatetimeIndex(values).tz_localize("America/New_York")
            + pd.Timedelta(hours=hour, minutes=minute), index=calendar)

    primary["signal_date"] = dates
    primary["trade_date"], primary["exit_date"] = dates.shift(-1), dates.shift(-2)
    primary["feature_available_at"] = clock(dates, 16)
    primary["decision_at"] = clock(primary.trade_date, 9, 31)
    primary["return_available_at"] = clock(primary.exit_date, 9, 30)
    primary["h5_available_at"] = clock(dates.shift(-6), 9, 30)
    primary["forecast_available_at"] = primary.h5_available_at
    primary["soxl_return"], primary["soxx_return"] = .01, .003
    forecast = primary.copy()
    forecast["forecast_available"] = 1.
    forecast["future_vol_prediction"] = .015 + .001 * np.sin(np.arange(len(calendar)) / 15)
    forecast["forecast_model_available_at"] = pd.DatetimeIndex([
        pd.Timestamp(f"{year}-01-01", tz="America/New_York") for year in calendar.year]).tz_convert("UTC")
    factor = 1 + .1 * np.sin(np.arange(len(calendar)) / 17)
    raw = pd.DataFrame({"cov_soxl_soxl": .0009 * factor, "cov_soxl_soxx": .00022 * factor,
        "cov_soxx_soxx": .0001 * factor, "m_expanding": .0001 * factor,
        "m_close20": .00011 * factor, "m_open20": .0001 * factor,
        "m_learned": forecast.future_vol_prediction.pow(2), "forecast_available": 1.,
        "future_vol_prediction": forecast.future_vol_prediction,
        "forecast_model_available_at": forecast.forecast_model_available_at}, index=calendar)
    raw.index.name = "datetime"
    calibrations, entries = {}, []
    for year in range(2016, 2027):
        cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        mask = np.ones(len(forecast), bool)
        for name in ("feature_available_at", "return_available_at", "forecast_available_at", "h5_available_at", "forecast_model_available_at"):
            mask &= pd.to_datetime(forecast[name], utc=True).lt(cutoff).to_numpy()
        known = forecast.loc[mask]
        common_dates = known.index.strftime("%Y-%m-%d").tolist()
        calibration = {"year": year, "cutoff": cutoff.isoformat(), "B": .00015,
            "c": {"expanding": 1., "close20": 1.1, "open20": 1., "learned": .8},
            "mu_H5": [.01, .005], "support": {key: {"min": 1e-8, "max": .01} for key in ("expanding", "close20", "open20", "learned")},
            "risk_units": workflow.RISK_UNITS,
            "audit": {"common_dates": common_dates, "common_rows": len(known),
                "common_dates_sha256": structured_sha(common_dates),
                "h5_label_available_at_max": known.h5_available_at.max().isoformat(),
                "label_available_at_max": known.forecast_available_at.max().isoformat(), "target_soxx_return_mean": .003}}
        calibrations[year] = calibration
        entries.append({"repeat_id": 1, "year": year, "segment": "development" if year <= 2022 else "reused_diagnostic", "calibration": calibration})
    frame = primary.copy()
    frame["decision_at"] = clock(dates, 16, 15)
    frame["return_available_at"] = clock(frame.exit_date, 9, 31)
    frame["h5_available_at"] = clock(dates.shift(-6), 9, 31)
    frame["forecast_available_at"] = frame.h5_available_at
    shape = np.empty((len(frame), 2, 2))
    shape[:, 0, 0], shape[:, 0, 1] = raw.cov_soxl_soxl, raw.cov_soxl_soxx
    shape[:, 1, 0], shape[:, 1, 1] = raw.cov_soxl_soxx, raw.cov_soxx_soxx
    source_audit = {"parent_recorder_id": "synthetic", "parent_summary_sha256": "a" * 64,
        "provider_manifest_sha256": "b" * 64, "native_frame_sha256": "c" * 64, "as_of": "2026-09-25",
        "features": {"synthetic": True}, "source_api": ["qlib.data.D.features"],
        "provider_uri": "/synthetic/current/provider"}
    parent_source_audit = {**source_audit, "provider_uri": "/synthetic/old-worktree/provider"}
    bundle = {"frame": frame, "calendar": calendar, "covariance": shape,
        "audit": {"original_frame_sha256": frame_sha256(primary), "qualified_source": source_audit}}
    development = primary.loc[primary.index.year.isin(range(2016, 2023))]
    old_inputs = {(1, name): prepare_policy_inputs(development, raw.loc[development.index], calibrations, key)
                  for key, name in workflow.PARENT_VARIANTS.items()}
    objects = {"inputs/primary_frame.pkl": primary, "inputs/source_audit.pkl": parent_source_audit,
        "inputs/forecast_frames.pkl": {1: forecast}, "inputs/raw_variances.pkl": {1: raw},
        "inputs/annual_calibrations.pkl": entries, "policy/development_inputs.pkl": old_inputs}
    return bundle, objects


def test_frozen_repeat_reconstructs_old_development_exactly_and_builds_all_diagnostic_years(plan, risk_source):
    bundle, objects = deepcopy(risk_source)
    before = deepcopy(objects)
    market, inputs, audit = workflow.reconstruct_frozen_risk_inputs(bundle, objects, plan)
    assert sorted(audit["calibrations"].year) == list(range(2016, 2027))
    assert audit["model_fits"] == audit["calibration_fits"] == 0
    assert audit["source_identity"]["parent_declared_provider_uri"] == "/synthetic/old-worktree/provider"
    assert audit["source_identity"]["current_verified_provider_uri"] == "/synthetic/current/provider"
    assert len(audit["applications"]) == 22
    assert audit["development_reproduction"].covariance_budget_support_provenance_exact.all()
    for key in ("learned", "close20"):
        assert len(inputs[key]["budget"]) == len(market)
        assert not inputs[key]["estimator_fallback"].any()
        assert set(inputs[key]["calibration_year"]) == set(range(2016, 2027))
        np.testing.assert_array_equal(inputs[key]["budget"], .00015)
    pd.testing.assert_frame_equal(objects["inputs/raw_variances.pkl"][1], before["inputs/raw_variances.pkl"][1], check_exact=True)
    pd.testing.assert_frame_equal(objects["inputs/primary_frame.pkl"], before["inputs/primary_frame.pkl"], check_exact=True)


def test_other_repeat_and_unconsumed_h5_means_cannot_change_risk_inputs(plan, risk_source):
    bundle, objects = deepcopy(risk_source)
    _, original, _ = workflow.reconstruct_frozen_risk_inputs(bundle, objects, plan)
    objects["inputs/raw_variances.pkl"][2] = pd.DataFrame({"bad": [np.nan]})
    objects["inputs/forecast_frames.pkl"][2] = None
    for row in objects["inputs/annual_calibrations.pkl"]:
        row["calibration"]["mu_H5"] = [np.nan, -100.]
    objects["inputs/annual_calibrations.pkl"].append({"repeat_id": 2, "year": 2016, "calibration": None})
    _, changed, _ = workflow.reconstruct_frozen_risk_inputs(bundle, objects, plan)
    for variant in original:
        for field in ("covariance", "budget", "support"):
            np.testing.assert_array_equal(original[variant][field], changed[variant][field])


@pytest.mark.parametrize("case", ["source_feature", "source_audit", "missing_year", "duplicate_year", "common_dates",
    "unmatured_calibration", "invalid_coefficient", "invalid_marginal", "unavailable_forecast", "future_model",
    "past_shape", "development_covariance", "development_provenance", "naive_forecast_clock"])
def test_invalid_risk_sources_fail_without_fallback_or_shortening(plan, risk_source, case):
    bundle, objects = deepcopy(risk_source)
    raw, forecasts = objects["inputs/raw_variances.pkl"][1], objects["inputs/forecast_frames.pkl"][1]
    day = raw.index[raw.index.year == 2025][20]
    if case == "source_feature":
        bundle["frame"].iloc[0, 0] += 1.
    elif case == "source_audit":
        objects["inputs/source_audit.pkl"] = {}
    elif case == "missing_year":
        objects["inputs/annual_calibrations.pkl"].pop()
    elif case == "duplicate_year":
        objects["inputs/annual_calibrations.pkl"].append(deepcopy(objects["inputs/annual_calibrations.pkl"][0]))
    elif case == "common_dates":
        objects["inputs/annual_calibrations.pkl"][0]["calibration"]["audit"]["common_dates"].pop()
    elif case == "unmatured_calibration":
        objects["inputs/annual_calibrations.pkl"][0]["calibration"]["audit"]["label_available_at_max"] = "2016-01-01T00:00:00-05:00"
    elif case == "invalid_coefficient":
        objects["inputs/annual_calibrations.pkl"][0]["calibration"]["c"]["learned"] = None
    elif case == "invalid_marginal":
        raw.loc[day, "m_close20"] = np.nan
    elif case == "unavailable_forecast":
        raw.loc[day, "forecast_available"] = 0.
    elif case == "future_model":
        raw.loc[day, "forecast_model_available_at"] = pd.Timestamp("2026-01-01", tz="UTC")
    elif case == "naive_forecast_clock":
        forecasts["feature_available_at"] = forecasts.feature_available_at.dt.tz_localize(None)
    elif case == "past_shape":
        raw.loc[day, "cov_soxl_soxx"] *= 1.01
    elif case == "development_covariance":
        objects["policy/development_inputs.pkl"][(1, "learned_vol5")]["covariance"][0] *= 1.01
    else:
        objects["policy/development_inputs.pkl"][(1, "observed_close20")]["budget_provenance"][0] = {}
    with pytest.raises(ValueError, match="blocked:"):
        workflow.reconstruct_frozen_risk_inputs(bundle, objects, plan)


def test_legacy_reader_is_explicit_and_pins_every_artifact_before_unpickle(plan, monkeypatch, tmp_path):
    called = []

    def verified(card, **kwargs):
        called.append(kwargs)
        assert kwargs["manifest_name"] == "artifact_manifest.json"
        assert kwargs["expected_manifest_sha256"] == plan["risk_parent"]["manifest_sha256"]
        assert kwargs["expected_artifact_sha256"] == plan["risk_parent"]["artifact_sha256"]
        assert set(kwargs["artifacts"]) == set(workflow.PARENT_ARTIFACTS)
        return {}, {"verified": True}

    monkeypatch.setattr(workflow, "load_verified_recorder_objects", verified)
    assert workflow._load_risk_parent(tmp_path, plan)[1]["verified"]
    assert len(called) == 1


def test_two_candidate_resolution_keeps_raw_tables_and_full_strategy_annual_metrics(plan, monkeypatch):
    names = ("learned_core", "close20_core")
    raw = {name: {"gates": pd.DataFrame({"cost_bps": [10., 25.], "all_pass": [True, True]}),
        "rolling_summary": pd.DataFrame({"cost_bps": [10., 25.], "positive_windows": [2, 3]}),
        "bootstrap": pd.DataFrame({"cost_bps": [10., 25.], "daily_lower_bound": [1e-12, 2e-12]}),
        "annual": pd.DataFrame({"cost_bps": [10., 25.], "log_excess_sum": [1e-12, 2e-12]})} for name in names}
    tables = {filename: pd.concat([value[key].assign(strategy=name) for name, value in raw.items()], ignore_index=True)
              for filename, key in (("gates.csv", "gates"), ("rolling_summary.csv", "rolling_summary"), ("bootstrap_analysis.csv", "bootstrap"))}
    tables["annual_metrics.csv"] = pd.DataFrame({"strategy": ["soxx", "both_controls", "both_candidates"]})
    paths = {(name, cost): pd.DataFrame({"signal_date": pd.to_datetime(["2016-01-04", "2023-01-03"])})
             for name in (*names, "soxx") for cost in (10., 25.)}
    calls = []

    def resolve(evidence, candidates, benchmarks, **kwargs):
        calls.append(evidence)
        assert kwargs["log_excess_floor"] == 1e-8
        assert all(len(frame) == 1 for frame in [*candidates.values(), *benchmarks.values()])
        value = deepcopy(evidence)
        value["gates"]["all_pass"] = False
        return value

    monkeypatch.setattr(workflow, "apply_excess_resolution", resolve)
    result, qualified = workflow.resolve_risk_evidence(tables, raw, paths, plan, {name: "control" for name in names})
    assert len(calls) == 2 and set(qualified) == set(names)
    assert not result["gates.csv"].all_pass.any() and result["raw_gates.csv"].all_pass.all()
    pd.testing.assert_frame_equal(result["annual_metrics.csv"], tables["annual_metrics.csv"], check_exact=True)
    pd.testing.assert_frame_equal(result["bootstrap_analysis.csv"], tables["bootstrap_analysis.csv"], check_exact=True)


def _small_market():
    calendar = pd.bdate_range("2020-01-02", periods=10)
    market = pd.DataFrame({"signal_date": calendar[:-3], "trade_date": calendar[1:-2], "exit_date": calendar[2:-1],
        "decision_at": calendar[:-3].tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15),
        "return_available_at": calendar[2:-1].tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31),
        "soxl_return": .03 * np.sin(np.arange(7)), "soxx_return": .01 * np.sin(np.arange(7))})
    market.index = pd.DatetimeIndex(market.signal_date, name="datetime")
    return market, calendar


def test_dynamic_three_asset_schedule_has_owned_fees_and_native_position_reconciliation(tmp_path):
    from qlib.data import D

    market, calendar = _small_market()
    raw = []
    for symbol, amplitude in (("SOXL", .03), ("SOXX", .01)):
        prices = 100. * np.cumprod(1 + amplitude * np.sin(np.arange(len(calendar))))
        for day, price in zip(calendar, prices):
            raw.append({"instrument": symbol, "datetime": day, "open": price, "close": price,
                "high": price * 1.01, "low": price * .99, "volume": 1e6})
    provider = tmp_path / "provider"
    build_qlib_store_from_frame(pd.DataFrame(raw), provider_uri=provider, as_of=str(calendar[-1].date()))
    opening = D.features(["SOXL", "SOXX"], ["$open"], start_time=calendar[0], end_time=calendar[-1])["$open"].unstack("instrument").astype(float)
    for symbol in ("SOXL", "SOXX"):
        market[symbol.lower() + "_return"] = opening[symbol].reindex(market.exit_date).to_numpy() / opening[symbol].reindex(market.trade_date).to_numpy() - 1
    targets = pd.DataFrame([[.1, .8, .1], [.2, .7, .1], [.1, .9, 0.], [0., 1., 0.], [.1, .8, .1], [.3, .6, .1], [0., .7, .3]],
                          index=market.index, columns=("soxl", "soxx", "cash"))
    path, audit = replay_target_schedule(market, targets, cost_bps=25.)
    native = native_open_reconciliation(market, path, provider_uri=provider, cost_bps=25.)
    assert native["audit"]["status"] == "passed"
    assert native["audit"]["maximum_session_return_error"] < 1e-10
    assert audit["maximum_independent_accounting_error"] < 1e-12
    native_weights = workflow._native_open_weights(native, path, provider)
    np.testing.assert_allclose(native_weights, targets.to_numpy(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(path.fees, path.turnover * .0025, rtol=0, atol=1e-12)
    changed = targets.copy()
    changed.iloc[5:] = [0., 0., 1.]
    later = replay_target_schedule(market, changed, cost_bps=25.)[0]
    pd.testing.assert_frame_equal(path.iloc[:5], later.iloc[:5], check_exact=True)
    with pytest.raises(ValueError, match="exact native"):
        replay_target_schedule(market, targets.iloc[::-1], cost_bps=25.)


def _mock_run(tmp_path, monkeypatch, plan, *, fail=False):
    import qlib.workflow
    import src.data.conditional_variance

    card = tmp_path / "card"
    card.mkdir()
    (card / "plan.json").write_text(json.dumps(plan))
    (card / "workflow.yaml").write_text("test: synthetic\n")
    market, calendar = _small_market()
    bundle = {"frame": market, "calendar": calendar, "audit": {"synthetic": True}}
    saved, contexts, native_calls = {}, [], []

    class Recorder:
        id, experiment_id = "synthetic-core", "synthetic-experiment"

        def log_params(self, **kwargs): pass
        def save_objects(self, artifact_path, **kwargs):
            saved.update({artifact_path + "/" + key: value for key, value in kwargs.items()})
        def log_artifact(self, path, artifact_path=None): assert Path(path).is_file()
        def get_local_dir(self): return str(tmp_path / "recorder")

    recorder = Recorder()

    class FakeR:
        active = False
        status = None

        @contextmanager
        def start(self, **kwargs):
            assert not self.active
            assert kwargs.get("recorder_id", recorder.id) == recorder.id
            contexts.append(kwargs)
            self.active = True
            try: yield recorder
            except Exception:
                self.status = "FAILED"
                raise
            else: self.status = "FINISHED"
            finally: self.active = False
        def get_recorder(self): return recorder

    fake = FakeR()
    monkeypatch.setattr(qlib.workflow, "R", fake)

    def prepare(*args):
        assert not fake.active
        return deepcopy(bundle)

    def parent(*args):
        assert not fake.active
        return {}, {"verified": True}

    def reconstruct(*args):
        if fail: raise ValueError("blocked: synthetic risk estimator unavailable")
        count = len(market)
        risks = {key: {"covariance": np.tile(np.array([[.0009, .00022], [.00022, .0001]]) * multiplier, (count, 1, 1)),
            "budget": np.full(count, .00015), "support": np.zeros(count, bool)} for key, multiplier in (("learned", 1.), ("close20", 1.2))}
        return market, risks, {"calibrations": pd.DataFrame({"year": [2020]}), "development_reproduction": pd.DataFrame({"exact": [True]})}

    def native(frame, path, provider, cost, account):
        assert not fake.active
        native_calls.append(cost)
        return path.copy(), {"audit": {"status": "passed", "cost_bps": cost, "sessions": len(path)}}

    fake_tables = {name: pd.DataFrame({"test": [1]}) for name in ("gates.csv", "annual_metrics.csv", "backtest_benchmark_comparison.csv")}
    fake_evidence = {name: {"historical_gate_pass": False, "excess_resolution": {"floor": 1e-8}} for name in ("learned_core", "close20_core")}
    monkeypatch.setattr(workflow, "_hashes", lambda root: {})
    monkeypatch.setattr(workflow, "prepare_horizon_inputs", prepare)
    monkeypatch.setattr(workflow, "_load_risk_parent", parent)
    monkeypatch.setattr(workflow, "load_legacy_execution_view", lambda root: ((root / plan["source"]["execution_provider_uri"]).resolve(), {"test": True}))
    monkeypatch.setattr(workflow, "_bundle_identity", lambda value: {"fixed": True})
    monkeypatch.setattr(workflow, "reconstruct_frozen_risk_inputs", reconstruct)
    monkeypatch.setattr(workflow, "_native_path", native)
    monkeypatch.setattr(workflow, "_native_open_weights", lambda native, path, provider:
        path[["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]].to_numpy(float))
    monkeypatch.setattr(workflow, "evaluate_registered_paths", lambda *a, **k: (deepcopy(fake_tables), deepcopy(fake_evidence)))
    monkeypatch.setattr(workflow, "resolve_risk_evidence", lambda *a: (deepcopy(fake_tables), deepcopy(fake_evidence)))
    monkeypatch.setattr(workflow, "paired_metrics", lambda *a: {"synthetic_only": True})
    monkeypatch.setattr(workflow, "_conclusion", lambda *a: "合成测试，不是市场结果。")
    monkeypatch.setattr(src.data.conditional_variance, "calibrate_annual", lambda *a, **k: pytest.fail("risk refit is forbidden"))
    return card, saved, contexts, native_calls, fake


def test_mocked_full_inventory_has_ten_native_paths_no_fit_and_successful_same_recorder_resume(tmp_path, monkeypatch, plan):
    card, saved, contexts, native, fake = _mock_run(tmp_path, monkeypatch, plan)
    summary, tables, conclusion = workflow.run_benchmark_core_variance(source_root=tmp_path, evidence_root=tmp_path,
        card_dir=card, recorder_uri=tmp_path / "tracking")
    assert len(native) == 10 and fake.status == "FINISHED" and not fake.active
    assert summary["model_fits"] == summary["risk_calibration_fits"] == summary["optimizer_solves"] == 0
    assert len(contexts) == 2 and contexts[-1]["resume"]
    assert (card / "summary.json").is_file() and len(summary["compressed_artifacts"]) == 3
    assert len(saved["analysis/paths.pkl"]) == 10
    assert tables["risk_mapping_summary.csv"].all_risk_targets_certified.all()
    with pytest.raises(ValueError, match="immutable"):
        workflow.run_benchmark_core_variance(source_root=tmp_path, evidence_root=tmp_path,
            card_dir=card, recorder_uri=tmp_path / "tracking")


def test_unavailable_risk_batch_is_preserved_failed_without_candidate_execution(tmp_path, monkeypatch, plan):
    card, saved, contexts, native, fake = _mock_run(tmp_path, monkeypatch, plan, fail=True)
    with pytest.raises(ValueError, match="estimator unavailable"):
        workflow.run_benchmark_core_variance(source_root=tmp_path, evidence_root=tmp_path,
            card_dir=card, recorder_uri=tmp_path / "tracking")
    assert fake.status == "FAILED" and not native
    assert saved["failure/exception.pkl"]["stage"] == "reconstruct_frozen_risk_inputs"
    assert saved["failure/exception.pkl"]["financial_recipe_changed"] is False
    assert not (card / "summary.json").exists()


def test_postfee_risk_audit_rejects_material_weight_changes_without_changing_targets():
    targets = np.array([[0., 1., 0.]])
    covariance, budgets = np.array([[[.0009, .0003], [.0003, .0001]]]), np.array([.0001])
    result = workflow._risk_weights_audit(targets, targets.copy(), covariance, budgets, layer="synthetic")
    assert result["within_measured_numerical_bound"] and not result["target_or_economic_budget_modified"]
    with pytest.raises(ValueError, match="differ materially"):
        workflow._risk_weights_audit(targets, np.array([[.01, .99, 0.]]), covariance, budgets, layer="synthetic")
