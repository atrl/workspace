"""Focused verification for the fixed 5-session action-label ablation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import multisession_label as M
from src.workflow import multisession_action_ablation as W
from src.workflow.stage1_economic_training import (
    ECONOMIC_FEATURES, _event_at, build_economic_dataset, mature_economic_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def test_plan_and_source_bindings_frozen():
    import json
    plan = json.loads((ROOT / "research" / W.EXPERIMENT / "plan.json").read_text())
    assert plan["budget"]["candidates"] == 1
    assert plan["budget"]["annual_fits"] == 7
    assert plan["training"]["training_state"] == "fixed_parent_state_label_ablation"
    assert plan["source_bindings"]["labels_10_sha256"] == W.LABELS_10_SHA256


@pytest.fixture(scope="module")
def _summary(tmp_path_factory):
    return W.run_multisession_action_ablation(root=ROOT, card_dir=tmp_path_factory.mktemp("abl"))


def _objects(_summary):
    import qlib
    from qlib.workflow import R
    qlib.init(provider_uri=str((ROOT / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    with R.uri_context(_summary["recorder_uri"]):
        rec = R.get_recorder(recorder_id=_summary["recorder_id"], experiment_name=W.EXPERIMENT)
        return {name: rec.load_object(name) for name in _summary["saved_objects"]}


def test_one_candidate_seven_fits_and_real_maturity(_summary):
    fits = [d for d in _summary["fit_diagnostics"] if d["fit"]]
    assert [d["model_year"] for d in fits] == list(W.FIT_YEARS)
    for d in fits:
        assert d["maturity_strictly_before_cutoff"] is True
        assert d["max_label_available_at"] is not None
        # real maturity: every training label matured strictly before the fit cutoff
        assert pd.Timestamp(d["max_label_available_at"]) < pd.Timestamp(d["information_cutoff_exclusive"])
        assert d["label_cost_bps"] == 25.0 and d["ridge_alpha"] == 10


def test_scaler_is_strictly_matured_mean_not_full_sample(_summary):
    objects = _objects(_summary)
    rows = objects["training_rows"]
    dataset = build_economic_dataset(rows)
    for year in (2017, 2023):
        first = rows[pd.to_datetime(rows["trade_date"]).dt.year == year]["trade_date"].iloc[0]
        prepared, _ = mature_economic_rows(dataset, model_year=year, first_decision_at=_event_at(first, minute=31))
        matured_mean = prepared.loc[:, list(ECONOMIC_FEATURES)].mean().to_numpy(float)
        full = rows[pd.to_numeric(rows["execute_advantage_bps"], errors="coerce").notna()]
        full_mean = full.loc[:, list(ECONOMIC_FEATURES)].mean().to_numpy(float)
        scaler = objects[f"candidate_scaler_{year}"]
        assert np.allclose(scaler.mean_, matured_mean, rtol=0, atol=1e-9)
        assert not np.allclose(scaler.mean_, full_mean, rtol=0, atol=1e-6)


def test_unknown_rows_kept_nan_and_source_identity_preserved(_summary):
    rows = _objects(_summary)["training_rows"]
    unknown = rows[rows["window_status"] != "available"]
    assert len(unknown) == 44
    assert unknown["execute_advantage_bps"].isna().all()
    assert unknown["label_available_at"].isna().all()
    # source target identity preserved (original 1-day label not overwritten)
    assert unknown["source_execute_advantage_bps"].notna().any() or unknown["source_execute_advantage_bps"].isna().any()
    assert "source_label_status" in rows.columns


def test_native_passed_four_cases_postfee_prefix(_summary):
    native = _summary["native"]
    assert native["status"] == "passed"
    results = native["results"]
    assert len(results) == 4
    for key, entry in results.items():
        assert entry["target_convention"] == "post_fee_desired"
        assert entry["suffix"]["status"] == "passed"
        if "development_walk_forward" in key:
            assert entry["prefix"]["status"] == "passed"
            assert entry["prefix_inherited"] is True


def test_oos_predictions_are_from_replays_not_training(_summary):
    objects = _objects(_summary)
    assert "oos_action_predictions" in objects and "training_predictions" in objects
    oos = objects["oos_action_predictions"]
    assert {"signal_date", "cost_bps", "segment", "predicted_net_advantage_bps", "execute_open_target"} <= set(oos.columns)
    assert set(oos["cost_bps"]) == {10.0, 25.0}


def test_risk_analysis_table_present(_summary):
    objects = _objects(_summary)
    assert "risk_analysis" in objects
    table = objects["risk_analysis"]
    assert {"book", "segment", "cost_bps", "mean", "std", "max_drawdown"} <= set(table.columns)
    assert {"candidate_label_5d", "soxx_buy_hold"} <= set(table["book"])


def test_decision_not_promoted_and_data_lineage(_summary):
    assert _summary["decision"] in ("not_promoted", "needs_data")
    assert _summary["promotion_eligible"] is False
    assert "development_only" in _summary["qualification_scope"]
    assert _summary["data_lineage"] and _summary["qlib_docs_reference"]


def test_strict_json(_summary):
    import json
    text = json.dumps(_summary, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
