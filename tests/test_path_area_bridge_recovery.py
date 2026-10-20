"""One synthetic contract for the bounded FAILED P2 checkpoint repair."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.workflow import path_area_bridge_recovery as repair


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def test_failed_checkpoint_seal_and_saved_oof_have_no_forward_or_refit(tmp_path, monkeypatch):
    card = tmp_path / "research/path_area_action_bridge_20261018/P2"
    recorder = card / "qlib_recorder" / repair.FAILED_EID / repair.FAILED_RID
    original = tmp_path / "research/path_area_action_bridge_20261018/P2_implementation_identity.json"
    _write(original, json.dumps({"sha256": repair.ORIGINAL_IMPLEMENTATION_SHA256}).encode())
    state = {"status": "failed", "recorder_id": repair.FAILED_RID,
             "plan_sha256": repair.p2.PLAN_SHA256, "new_allocation_fit_attempted": 6,
             "new_allocation_fit_completed": 6, "training_forward_calls_completed": 1800,
             "OOF_prediction_calls_completed": 15, "shadow_policy_paths": 0,
             "label_pairs": 0, "native_accounts": 0,
             "fit_barrier_sha256": "29b87bbbbf4dd643d4c85168581ed44d99ce5085d08472134c32573dfda9b6fc",
             "exception_type": "KeyError", "exception": "'tariff'"}
    barrier = {"status": "passed", "sha256": state["fit_barrier_sha256"],
               "completed_slots": [{"trade_year": year, "base_seed": seed}
                                   for year in (2018, 2019) for seed in (71, 172, 273)]}
    digests = {}
    for name, value in (("P2_execution_state.json", state), ("fit_barrier.json", barrier),
                        ("failure_summary.json", {}), ("failure_metadata_receipt.json", {})):
        digests[name] = _write(card / name, json.dumps(value).encode())
    meta_sha = _write(recorder / "meta.yaml", b"status: 4\n")
    names = (*repair.CHECKPOINT_NAMES, *(f"extra/{number}" for number in range(34 - len(repair.CHECKPOINT_NAMES))))
    assert len(names) == 34
    entries = []
    for name in names:
        payload = name.encode()
        path = recorder / "artifacts" / name
        digest = _write(path, payload)
        entries.append({"path": path.relative_to(card).as_posix(), "sha256": digest,
                        "bytes": len(payload)})
    manifest = {"schema_version": 1, "recorder_id": repair.FAILED_RID,
                "experiment_id": repair.FAILED_EID, "recorder_status": "FAILED",
                "original_implementation_sha256": repair.ORIGINAL_IMPLEMENTATION_SHA256,
                "run_metadata_path": (recorder / "meta.yaml").relative_to(card).as_posix(),
                "run_metadata_sha256": meta_sha, "entries": entries}
    digests["failure_manifest.json"] = _write(card / "failure_manifest.json",
                                               json.dumps(manifest).encode())
    monkeypatch.setattr(repair, "FAILURE_MANIFEST_SHA256", digests["failure_manifest.json"])
    monkeypatch.setattr(repair, "FAILURE_SUMMARY_SHA256", digests["failure_summary.json"])
    monkeypatch.setattr(repair, "FAILURE_METADATA_SHA256", digests["failure_metadata_receipt.json"])
    monkeypatch.setattr(repair, "FAILED_STATE_SHA256", digests["P2_execution_state.json"])
    repair_card = card / "repair_1"
    for name in ("workflow.yaml", "repair_plan.json", "repair_1.ipynb",
                 "repair_execution_authorization.json", "repair_implementation_review.json"):
        _write(repair_card / name, b"draft")
    repair._require_unused_repair_card(repair_card)
    _write(repair_card / "repair_execution_state.json", b"started")
    with pytest.raises(FileExistsError, match="no automatic retry"):
        repair._require_unused_repair_card(repair_card)
    verified, recorded = repair._read_seal(tmp_path)
    assert verified["recorder_status"] == "FAILED"
    assert set(repair.CHECKPOINT_NAMES).issubset(recorded)
    (recorder / "artifacts" / "models/new_2018_71").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="before unpickling"):
        repair._read_seal(tmp_path)

    # Stored per-seed outputs reproduce the aggregate with no model method call.
    timing_dates = pd.to_datetime([f"{year}-01-02" for year in range(2018, 2023)])
    timing = pd.DataFrame({"trade_year": list(range(2018, 2023))}, index=timing_dates)
    folds = {year: {"prediction_index": pd.DatetimeIndex([timing_dates[year - 2018]])}
             for year in range(2018, 2023)}
    checkpoint = {}
    rows = []
    for year, date in zip(range(2018, 2023), timing_dates):
        row = {"soxl_weight": .2, "soxx_weight": .3, "cash_weight": .5,
               "tail_probability": .1, "alpha_probability": .8}
        for seed in (71, 172, 273):
            checkpoint[f"OOF_seed_predictions/year_{year}_seed_{seed}"] = pd.DataFrame([row], index=[date])
        combined = (pd.DataFrame([row], index=[date]) * 3.0) / 3.0
        weights = ["soxl_weight", "soxx_weight", "cash_weight"]
        combined.loc[:, weights] = combined.loc[:, weights].div(combined.loc[:, weights].sum(axis=1), axis=0)
        rows.append(combined.iloc[0].to_dict())
    aggregate = pd.DataFrame(rows, index=timing_dates)
    aggregate.index.name = "signal_date"
    checkpoint["OOF/raw_equal_weight_predictions"] = aggregate
    assert repair._verify_saved_predictions(checkpoint, {"timing": timing,
        "fold_package": {"folds": folds}}).equals(aggregate)
    checkpoint["OOF/raw_equal_weight_predictions"] = aggregate.assign(soxl_weight=.21)
    with pytest.raises(AssertionError):
        repair._verify_saved_predictions(checkpoint, {"timing": timing,
            "fold_package": {"folds": folds}})
