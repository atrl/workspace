from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import cost_matched_action_experiment as wf

ROOT = Path(__file__).resolve().parents[1]


def _replay(dates):
    return pd.DataFrame({
        "signal_date": pd.to_datetime(dates),
        "strategy_return": 0.0,
        "turnover": 0.0,
        "execute_open_trade": False,
        "recovery_action": False,
        "ending_soxl_weight": 0.0,
        "ending_soxx_weight": 1.0,
        "ending_cash_weight": 0.0,
    })


# --- item 3: strict reference alignment blocks ---

def test_validate_replay_blocks_on_duplicate_key():
    frame = _replay(["2020-01-02", "2020-01-02"])
    with pytest.raises(ValueError, match="duplicate"):
        wf._validate_replay(frame, label="x")


def test_validate_replay_blocks_on_nonfinite():
    frame = _replay(["2020-01-02", "2020-01-03"])
    frame.loc[0, "strategy_return"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        wf._validate_replay(frame, label="x")


def test_validate_replay_blocks_on_missing_columns():
    frame = _replay(["2020-01-02"]).drop(columns=["ending_cash_weight"])
    with pytest.raises(ValueError, match="missing columns"):
        wf._validate_replay(frame, label="x")


def test_align_blocks_on_different_session_keys():
    a = _replay(["2020-01-02", "2020-01-03"])
    b = _replay(["2020-01-02", "2020-01-06"])
    with pytest.raises(ValueError, match="session keys differ"):
        wf._align(a, b, label="x")


def test_align_accepts_identical_keys_in_any_order():
    a = _replay(["2020-01-02", "2020-01-03"])
    b = _replay(["2020-01-03", "2020-01-02"])
    ca, cb = wf._align(a, b, label="x")
    assert ca["signal_date"].tolist() == cb["signal_date"].tolist()


# --- item 1 & 4: real run: parity, gates, recorder contents ---

@pytest.fixture(scope="module")
def _summary(tmp_path_factory):
    return wf.run_cost_matched_action(root=ROOT, card_dir=tmp_path_factory.mktemp("cma"))


def test_control_parity_passes(_summary):
    assert _summary["control_parity"]["status"] == "passed"
    for check in _summary["control_parity"]["checks"]:
        assert check["returns_ok"] and check["ending_weights_ok"] and check["decisions_ok"]


def test_child_recorder_saves_models_training_and_paths(_summary):
    saved = set(_summary["saved_objects"])
    # Actual experiment objects, not only a metric table.
    for required in ("models_10", "models_25", "training_rows_10", "training_rows_25",
                     "diagnostics_10", "diagnostics_25", "candidate_replays",
                     "candidate_comparison", "reference_comparison", "annual_records"):
        assert required in saved
    assert _summary["artifact_policy"]["recorder_only_artifacts"] == sorted(saved)


def test_gates_are_computed_and_grouped(_summary):
    rows = _summary["gates"]["rows"]
    assert rows
    for row in rows:
        assert row["segment"] in wf.SEGMENT_LABELS.values()
        assert row["cost_bps"] in wf.COSTS
        assert row["reference"] in ("no_hold_reference", "active_hold_baseline")
        assert "all_pass" in row and "annual_wealth_floor_pass" in row


def test_soxx_benchmark_matches_frame(_summary):
    soxx = _summary["soxx_benchmark"]
    assert any(r["benchmark"] == "SOXX_buy_hold_open_to_open" for r in soxx)


def test_no_promotion_and_plan_locked(_summary):
    assert _summary["promotion_eligible"] is False
    assert _summary["objective_achieved"] is False
    assert _summary["plan_sha256"]
    assert _summary["source_object_sha256"]


def test_results_are_grouped_by_segment_and_cost(_summary):
    table = pd.DataFrame(_summary["results"])
    assert set(table["segment"]) == set(wf.SEGMENT_LABELS.values())
    assert set(table["cost_bps"]) == {10.0, 25.0}
    assert "cost_matched" in set(table["family"])
    assert "frozen_reference" in set(table["family"])
    assert "winner" not in _summary


def test_summary_is_strict_json_serializable(_summary):
    import json
    text = json.dumps(_summary, default=str, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_candidate_book_label_fee_matches_its_cost(_summary):
    table = pd.DataFrame(_summary["results"])
    for _, row in table[table["family"] == "cost_matched"].iterrows():
        assert row["label_cost_bps"] == row["cost_bps"] == row["evaluation_cost_bps"]
