"""Synthetic native lifecycle regression; never opens campaign data or fits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.native_evidence import native_dataset
from src.model.path_area_action import FittedArm, feature_order
from src.workflow import path_area_action_bridge_p3 as p3


class WidthOnlyPipeline:
    n_features_in_ = 32

    def __getitem__(self, item):
        return self


def test_native_trade_date_reconciliation_and_durable_detached_model_barrier(tmp_path, monkeypatch):
    """Signal != trade date; immediate model seals survive native re-init."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder

    qlib.init(provider_uri=str(tmp_path / "unused_provider"), region="us", kernels=1)
    signal = pd.bdate_range("2020-01-02", periods=240)
    trade = signal + pd.offsets.BDay()
    values = np.random.default_rng(4).normal(size=(240, 32))
    features = pd.DataFrame(values, index=trade, columns=feature_order())
    features["prefix_mask"] = 1
    maturity = pd.Series(trade.tz_localize("America/New_York") + pd.Timedelta(hours=10), index=trade)
    delta = pd.Series(np.where(np.arange(240) % 2, .01, -.01), index=trade)
    meta = pd.DataFrame({"trade_date": trade, "label_available_at": maturity,
                         "prefix_mask": features.prefix_mask.astype("int8")}, index=trade)
    grouped = pd.concat({"feature": features, "label": delta.to_frame("local_execute_delta"), "meta": meta}, axis=1)
    grouped.index = pd.MultiIndex.from_arrays([signal, ["SOXL_SOXX_CASH"] * 240], names=["datetime", "instrument"])
    decisions = features.assign(signal_date=signal, eligible=True)
    labels = features.assign(signal_date=signal, eligible=True, label_status="mature",
                             label_available_at=maturity, execute_delta=delta)
    objects = {"local_labels/action_label_dataset": native_dataset(grouped, segments={"all": (None, None)}),
               "local_labels/action_label_frame": grouped, "local_labels/labels": labels,
               "shadow_python/decisions": decisions,
               "shadow_python/path_area_inputs": features.assign(signal_date=signal, trade_date=trade)}
    extra = objects["shadow_python/path_area_inputs"].iloc[[0]].copy()
    extra["signal_date"] = signal[0] - pd.offsets.BDay()
    extra["trade_date"] = trade[0] - pd.offsets.BDay()
    objects["shadow_python/path_area_inputs"] = pd.concat([extra, objects["shadow_python/path_area_inputs"]], ignore_index=True)
    preflight, audit = p3._training(objects)
    assert preflight.ready and preflight.row_count == preflight.pre_2022_rows == 240
    pd.testing.assert_index_equal(preflight.features.index, trade, check_names=False)
    assert audit["B_C_identical_row_ids_and_labels"]
    bad = dict(objects)
    bad["local_labels/action_label_frame"] = grouped.copy()
    bad["local_labels/action_label_frame"].iloc[0, 0] += 1
    with pytest.raises(AssertionError):
        p3._training(bad)

    card = tmp_path / "card"
    card.mkdir()
    state_path = card / "run_state.json"
    state = {"fit_attempts": [], "fit_done": [], "model_sha256": {},
             "source_attempts": [], "source_done": [], "source_objects_read": 0}
    source_card = tmp_path / "synthetic_source"
    source_card.mkdir()
    (source_card / "nas_binding.json").write_text(json.dumps({"evidence": {"sha256": "synthetic"}}))
    events = []

    def source_loader(*args, **kwargs):
        p3._no_global_writer()
        assert not (card / "qlib_recorder").exists()
        assert json.loads(state_path.read_text())["source_attempts"] == ["train"]
        events.append("source")
        return objects, {"synthetic_only": True}

    monkeypatch.setattr(p3, "load_verified_recorder_objects", source_loader)
    loaded, _ = p3._read(tmp_path, {"card": "synthetic_source", "archive": "synthetic", "recorder_id": "fake",
        "artifacts": list(objects), "inventory_sha256": "fake", "artifact_sha256": {}}, state, state_path, None, "train")
    assert loaded is objects
    with R.start(experiment_name="synthetic_P3_lifecycle", uri=str(card / "qlib_recorder")):
        rec = R.get_recorder()
    rec.start_run()

    def no_fit(preflight, arm):
        assert json.loads(state_path.read_text())["fit_attempts"][-1] == arm
        p3._no_global_writer()
        if arm == "C":
            assert (pd_path := Path(rec.get_local_dir()) / "artifacts/models/B_action_model").is_file()
            assert p3.sha(pd_path) == state["model_sha256"]["B_action_model"]
        events.append(arm)
        return FittedArm(arm, WidthOnlyPipeline(), feature_order(), 240, {0: 120, 1: 120}, 240,
                         preflight.cutoff, preflight.pre_2022_cutoff, .5, {"synthetic_no_fit": True})

    monkeypatch.setattr(p3, "fit_one_arm", no_fit)
    models = p3._fit_models(preflight, rec, state, state_path)
    assert set(models) == {"B", "C"} and state["models_sealed"] and state["evaluation_unlocked"]
    assert events == ["source", "B", "C"] and len(state["model_sha256"]) == 4
    assert json.loads(state_path.read_text())["fit_attempts"] == ["B", "C"]
    with pytest.raises(ValueError, match="attempt consumed"):
        p3._claim(state, state_path, "fit", "B", 2, rec)
    def reinitialize_without_writer():
        assert rec.status == Recorder.STATUS_FI
        qlib.init(provider_uri=str(tmp_path / "another_unused_provider"), region="us", kernels=1)
    original_recorder_id = rec.id
    for phase in ("source_like_operation", "native_like_operation"):
        p3._outside_writer(rec, reinitialize_without_writer)
        assert rec.id == original_recorder_id
        rec.save_objects(artifact_path="execution", **{phase: {"same_recorder": rec.id}})
        assert rec.load_object("execution/" + phase)["same_recorder"] == original_recorder_id
    p3._no_global_writer()
    rec.save_objects(artifact_path="execution", after_native_reinit={"same_recorder": rec.id})
    rec.end_run(status=Recorder.STATUS_FI)
    assert rec.load_object("execution/after_native_reinit")["same_recorder"] == rec.id
