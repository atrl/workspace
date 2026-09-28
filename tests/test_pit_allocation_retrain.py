import hashlib
import json

import pandas as pd
import pytest

from src.workflow.pit_allocation_retrain import PRIMARY_LABELS, _mature_training_labels, run_fixed_pit_retraining, validate_frozen_bundle


def test_training_cutoff_uses_label_realization_instead_of_signal_date():
    dates = pd.bdate_range("2022-12-01", "2023-01-20")
    frame = pd.DataFrame(1.0, index=dates, columns=PRIMARY_LABELS)
    frame["exit_date"] = pd.Series(dates, index=dates).shift(-2)
    result, audit = _mature_training_labels(frame, "2022-12-30")
    label_realization = frame["exit_date"].shift(-4)
    crosses_boundary = (dates <= "2022-12-30") & (label_realization > "2022-12-30")
    assert crosses_boundary.sum() == 6
    assert result.loc[crosses_boundary, PRIMARY_LABELS].isna().all().all()
    assert result.loc[dates > "2022-12-30", PRIMARY_LABELS].eq(1.0).all().all()
    assert audit["last_training_label_realization"] == "2022-12-30"
    assert audit["last_mature_training_signal"] == "2022-12-22"
    assert audit["excluded_cross_boundary_signal_rows"] == 6


def test_frozen_bundle_rejects_lost_or_modified_weights(tmp_path):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"verified frozen training state")
    manifest = {"artifacts": [{"path": model.name, "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}]}
    (tmp_path / "frozen_bundle_manifest.json").write_text(json.dumps(manifest))
    assert validate_frozen_bundle(tmp_path) == manifest
    model.write_bytes(b"different model")
    with pytest.raises(ValueError, match="missing or changed"):
        validate_frozen_bundle(tmp_path)
    model.unlink()
    with pytest.raises(ValueError, match="missing or changed"):
        validate_frozen_bundle(tmp_path)


def test_cached_card_cannot_be_relabelled_as_a_new_as_of(tmp_path):
    card = tmp_path / "research" / "existing_retrain"
    card.mkdir(parents=True)
    (card / "summary.json").write_text(json.dumps({"experiment": "existing_retrain", "as_of": "2026-09-24"}))
    with pytest.raises(ValueError, match="different identity"):
        run_fixed_pit_retraining(root=tmp_path, experiment="existing_retrain", as_of="2026-09-25")
