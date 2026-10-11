"""Focused mock tests for the H20 experiment workflow (no real fit)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import h20_matched_experiment as H


def _plan(**over):
    plan = {"experiment": H.EXPERIMENT, "source": {"parent_plan": "benchmark_terminal_value_20261009",
            "as_of": "2026-09-25", "region": "us", "source_sha256": {"src/analysis/h20_prefit_contract.py": "frozen"},
            "preregister_sha256": "x", "providers": {"published": "data/x", "execution": "data/y"},
            "provider_sha256": {"published": "x", "execution": "y"}, "prefit": {"recorder_id": "test"}}, "selection_years": list(range(2016, 2027)), "fixed_fit_budget": 55, "fixed_native_cases": 8,
            "candidate_budget": 1, "initial_cash": 100000.0, "minimum_training_rows": 252,
            "minimum_validation_rows": 126, "hac_lags": 5, "train_start": "2011-01-03",
            "costs_bps": [10.0, 25.0], "robust_radius": 1.0, "windows": ["development", "reused_diagnostic"],
            "development_years": list(range(2016, 2023)), "horizon_sessions": 20,
            "authorization": {"training_authorized": True, "replay_authorized": True},
            "gates": {"minimum_positive_years": 6}}
    plan.update(over)
    return plan


def test_plan_validation_and_fixed_budget(tmp_path):
    path = tmp_path / "execution_plan.json"
    path.write_text(json.dumps(_plan()))
    plan = H.load_execution_plan(path)
    assert H.fit_budget(plan) == 55
    for bad in ({"selection_years": list(range(2016, 2026))}, {"costs_bps": [10.0]}, {"robust_radius": 2.0},
                {"horizon_sessions": 5}, {"authorization": {"training_authorized": False, "replay_authorized": False}}):
        path.write_text(json.dumps(_plan(**bad)))
        with pytest.raises(ValueError):
            H.load_execution_plan(path)


def test_window_frames_only_excludes_unsettled_tail():
    days = pd.bdate_range("2015-01-02", "2024-12-31", freq="B")
    frame = pd.DataFrame({"signal_date": days, "exit_date": days + pd.tseries.offsets.BDay(2)})
    frame.loc[frame.index[-2:], "exit_date"] = pd.NaT  # unsettled terminal tail
    covariance = np.zeros((len(frame), 2, 2))
    windows, audit = H._window_frames(frame, covariance, _plan())
    assert set(windows) == {"development", "reused_diagnostic"}
    kept = pd.concat([w["frame"] for w in windows.values()])
    assert kept["exit_date"].notna().all()                       # no NaT enters a window
    assert len(kept) == int((frame["signal_date"].dt.year >= 2016).sum()) - 2   # only the 2 unsettled rows removed
    assert audit["excluded_unsettled_terminal_rows"] == 2 and audit["full_rows"] == len(frame)
    assert all(w["covariance"].shape[0] == len(w["frame"]) for w in windows.values())
    assert windows["development"]["frame"]["signal_date"].dt.year.max() == 2022
    assert windows["reused_diagnostic"]["frame"]["signal_date"].dt.year.min() == 2023


def test_assemble_registered_paths_concatenates_windows_per_cost():
    idx = pd.bdate_range("2016-01-04", periods=4)
    wp = {}
    for w in ("development", "reused_diagnostic"):
        for s in ("h20", "soxx"):
            for c in (10.0, 25.0):
                wp[(w, s, c)] = pd.DataFrame({"signal_date": idx, "equity": [1.0] * 4})
    registered = H._assemble_registered_paths(wp, _plan())
    assert set(registered) == {("h20", 10.0), ("h20", 25.0), ("soxx", 10.0), ("soxx", 25.0)}
    assert len(registered[("h20", 10.0)]) == 8  # two independent windows concatenated


def test_eight_native_cases_must_all_pass():
    keys = {f"{w}::{s}::{c}" for w in ("development", "reused_diagnostic") for s in ("h20", "soxx") for c in (10, 25)}
    good = {k: {"status": "passed"} for k in keys}
    H.require_all_native_passed(good, keys)  # 8/8 passed -> ok
    with pytest.raises(ValueError, match="pass for all 8"):
        H.require_all_native_passed({**good, "development::h20::10": {"status": "failed"}}, keys)
    with pytest.raises(ValueError, match="exactly 8"):
        H.require_all_native_passed({k: {"status": "passed"} for k in list(keys)[:7]}, keys)


def test_source_freeze_mismatch_blocked(tmp_path):
    (tmp_path / "input.txt").write_text("frozen")
    import hashlib
    sha = hashlib.sha256(b"frozen").hexdigest()
    plan = {"source": {"source_sha256": {"input.txt": sha}, "preregister_sha256": "x",
                       "providers": {}, "provider_sha256": {}}}
    out = H._source_freeze(tmp_path, tmp_path, plan)
    assert out["source_sha256"] == {"input.txt": sha}
    (tmp_path / "input.txt").write_text("changed")
    with pytest.raises(ValueError, match="differs from the execution plan"):
        H._source_freeze(tmp_path, tmp_path, plan)


def test_recorder_archive_smoke_full_api(tmp_path):
    """Actual Recorder open + resume + archive (no fit/replay); verifies id and artifacts."""
    root = Path(__file__).resolve().parents[1]
    plan = _plan()
    plan["source"]["providers"] = {"published": "data/qlib/published_futu_affine_20260925_v1",
                                   "execution": "data/qlib/legacy_soxl_execution_20260925_v1"}
    plan_path = tmp_path / "execution_plan.json"
    plan_path.write_text(json.dumps(plan))
    card = tmp_path / "card"
    card.mkdir()
    recorder_id, uri, local_dir = H._open_inputs_recorder(root, card, plan, plan_path=plan_path,
                                                          inputs={"frame_sha256": "x", "calendar_rows": 1})
    native_objects = {"development::h20::10": {
        "actual_position_reconciliation": pd.DataFrame({"trade_date": [pd.Timestamp("2016-01-04")], "native_cash": [1.0]}),
        "actual_position_audit": {"status": "passed"}, "report": {"r": 1}, "positions": {"p": 1}}}
    objects = {"native_cases.pkl": native_objects, "selections.pkl": {"2016": {"selected_name": "prior"}},
               "forecasts.pkl": {"development::h20::10": {"ridge": [0.1]}},
               "window_paths.pkl": {"development::h20::10": [1, 2]}, "final_summary.pkl": {"recorder_id": recorder_id}}
    new_id = H._archive(uri, recorder_id, plan["experiment"], objects=objects,
                        tables={"gates.csv": pd.DataFrame({"name": ["all_pass"], "value": [True]})})
    assert new_id == recorder_id  # resumed id unchanged
    from qlib.workflow import R
    artifacts_root = Path(local_dir) / "artifacts"
    for leaf in ("aggregate/native_cases.pkl", "aggregate/selections.pkl", "aggregate/forecasts.pkl",
                 "aggregate/window_paths.pkl", "aggregate/final_summary.pkl", "tables/gates.pkl", "inputs/inputs.pkl"):
        assert (artifacts_root / leaf).exists(), leaf
    with R.uri_context(uri):
        rec = R.get_recorder(recorder_id=recorder_id, experiment_name=plan["experiment"])
        assert rec.status == "FINISHED"
        loaded = rec.load_object("aggregate/final_summary.pkl")
        assert loaded["recorder_id"] == recorder_id


def test_canonical_draft_plan_is_schema_valid_but_fail_closed(tmp_path):
    root = Path(__file__).resolve().parents[1]
    draft = root / "research" / "h20_matched_forecast_execution_20261010" / "execution_plan.json"
    if not draft.exists():
        pytest.skip("canonical execution_plan.json not present")
    plan = json.loads(draft.read_text())
    denied = {**plan, "authorization": {"training_authorized": False, "replay_authorized": False}}
    denied_path = tmp_path / "denied.json"
    denied_path.write_text(json.dumps(denied))
    with pytest.raises(ValueError, match="authorized one-shot"):
        H.load_execution_plan(denied_path)
    plan["authorization"] = {"training_authorized": True, "replay_authorized": True}
    copy = tmp_path / "execution_plan_authorized.json"
    copy.write_text(json.dumps(plan))
    loaded = H.load_execution_plan(copy)  # schema valid
    assert H.fit_budget(loaded) == 55
    assert loaded["source"]["prefit"]["recorder_id"] == "5654796769704dee8d84d51186ee4960"
    assert loaded["scientific_scope"]["h5_financial_reference"] == "not_comparable"
    assert loaded["scientific_scope"]["horizon_only_causal_effect_claimed"] is False


def test_existing_attempt_blocks_before_source_reads_or_refits(tmp_path,monkeypatch):
    plan_path=tmp_path/'execution_plan.json'
    plan_path.write_text(json.dumps(_plan()))
    (tmp_path/'execution_attempt.json').write_text('{"status":"complete"}')
    def unexpected(*args,**kwargs):
        pytest.fail('A completed/failed attempt must never reach source reads or fits')
    monkeypatch.setattr(H,'_source_freeze',unexpected)
    with pytest.raises(ValueError,match='one-shot attempt already exists'):
        H.run_h20_matched_experiment(root=tmp_path,card=tmp_path,execution_plan_path=plan_path)


@pytest.mark.parametrize('change',[{'horizon_sessions':20.9},{'robust_radius':True},{'minimum_training_rows':True},
    {'minimum_validation_rows':126.5},{'hac_lags':20},{'initial_cash':200000.}])
def test_execution_protocol_drift_fails_closed(tmp_path,change):
    p=tmp_path/'plan.json';p.write_text(json.dumps(_plan(**change)))
    with pytest.raises(ValueError):
        H.load_execution_plan(p)
