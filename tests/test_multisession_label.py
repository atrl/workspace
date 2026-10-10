"""Verification for the multi-session label audit (diagnostic only)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import multisession_label as M
from src.workflow import multisession_label_audit as W

ROOT = Path(__file__).resolve().parents[1]


def test_all_annual_model_hashes_bound_and_carry_forward():
    for year in range(2016, 2027):
        assert f"model_control_all_state_{year}" in M.EXPECTED_OBJECT_SHA256
    assert M.CARRY_FORWARD_FROM_YEAR == {2024: 2023, 2025: 2023, 2026: 2023}
    assert M.EXPECTED_OBJECT_SHA256["model_control_all_state_2024"] == M.EXPECTED_OBJECT_SHA256["model_control_all_state_2023"]


def test_verify_observed_parity_blocks_on_mismatch():
    a = pd.DataFrame({"trade_date": pd.to_datetime(["2020-01-02"]), "observed_return": [0.01],
                      "observed_end_soxl": [0.3], "observed_end_soxx": [0.5], "observed_end_cash": [0.2]})
    good = pd.DataFrame({"trade_date": pd.to_datetime(["2020-01-02"]), "strategy_return": [0.01],
                         "ending_soxl_weight": [0.3], "ending_soxx_weight": [0.5], "ending_cash_weight": [0.2]})
    assert M.verify_observed_parity(a, good, label="t")["status"] == "exact"
    bad = good.copy(); bad["ending_cash_weight"] = [0.21]
    with pytest.raises(ValueError, match="parity failed"):
        M.verify_observed_parity(a, bad, label="t")


def test_spearman_null_on_constant_and_filter_finite():
    assert M._spearman([1.0] * 5, [1.0, 2, 3, 4, 5]) is None
    calendar = pd.DataFrame({"segment": ["s"] * 6, "routing_eligible": [True] * 6, "window_status": ["available"] * 6,
                             "predicted_net_advantage_bps": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0],
                             "label_5d_bps": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert len(M._filtered(calendar, segment="s")) == 5


def test_block_bootstrap_full_actually_resamples():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"value": rng.normal(size=200)})
    out = M.block_bootstrap_full(frame, lambda d: float(d["value"].mean()), block=20, bootstrap=500)
    assert out["status"] == "ok" and out["ci_low"] < out["point"] < out["ci_high"]


@pytest.fixture(scope="module")
def _control():
    return M.load_frozen_control(ROOT)


@pytest.fixture(scope="module")
def _summary(tmp_path_factory):
    """Run the whole workflow with all fitting entry points disabled (0-fit proof)."""
    from _pytest.monkeypatch import MonkeyPatch
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    import src.workflow.stage1_economic_training as E

    def boom(*args, **kwargs):
        raise AssertionError("no model fitting is permitted in this diagnostic round")

    patch = MonkeyPatch()
    patch.setattr(E, "train_causal_economic_models", boom)
    patch.setattr(Pipeline, "fit", boom)
    patch.setattr(Ridge, "fit", boom)
    try:
        return W.run_multisession_label_audit(root=ROOT, card_dir=tmp_path_factory.mktemp("msl"))
    finally:
        patch.undo()


def _objects(_summary):
    import qlib
    from qlib.workflow import R
    qlib.init(provider_uri=str((ROOT / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    with R.uri_context(_summary["recorder_uri"]):
        rec = R.get_recorder(recorder_id=_summary["recorder_id"], experiment_name=W.EXPERIMENT)
        return {name: rec.load_object(name) for name in _summary["saved_objects"]}


def test_full_calendar_labels_retained_and_unavailable_nan(_summary):
    objects = _objects(_summary)
    for cost in (10, 25):
        labels = objects[f"labels_{cost}"]
        assert len(labels) == 2690
        unavailable = labels[labels["window_status"] != "available"]
        assert len(unavailable) > 0
        assert unavailable["label_5d_bps"].isna().all()
        assert labels[labels["window_status"] == "available"]["label_5d_bps"].notna().all()


def test_available_maturity_is_fifth_exit_0930(_summary, _control):
    objects = _objects(_summary)
    panel = _control["panel"].reset_index(drop=True)
    exits = pd.to_datetime(panel["exit_date"]).to_numpy()
    labels = objects["labels_10"].reset_index(drop=True)
    available = labels[labels["window_status"] == "available"]
    assert len(available) > 0
    for index in available.index[:50]:
        k = panel.index.get_loc(panel.index[index])
        expected = M._event_at(exits[index + M.HORIZON - 1], minute=30)
        assert pd.Timestamp(labels.loc[index, "horizon_available_at"]) == expected


def test_branch_first_day_forced_execute_flags(_summary):
    trajectories = _objects(_summary)["trajectories_10"]
    first = trajectories[trajectories["offset"] == 0]
    ex = first[first["branch"] == "execute"]
    ho = first[first["branch"] == "hold"]
    assert len(ex) > 0 and len(ho) > 0
    assert ex["execute"].all()
    assert not ho["execute"].any()
    assert ex["trade_date"].notna().all() and ex["exit_date"].notna().all()


def test_both_parent_costs_exact_parity(_summary):
    assert _summary["models_trained"] == 0
    assert _summary["parent_costs_bps"] == [10, 25]
    for cost in (10, 25):
        for span in M.SEGMENT_LABELS.values():
            entry = _summary["observed_parent_parity"][cost][span]
            assert entry["status"] == "exact" and entry["rtol"] == 0


def test_summary_strict_json(_summary):
    import json
    text = json.dumps(_summary, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
