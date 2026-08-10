from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.workflow.stage1_rank_calibrated_recovery_sizing import (
    Stage1RankCalibratedRecoverySizingSpec,
    empirical_percentile,
    fixed_recovery_fraction,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_rank_calibrated_recovery_sizing_v1"


def test_empirical_percentile_uses_training_distribution_only():
    training = np.array([0.20, 0.40, 0.60, 0.80])

    assert empirical_percentile(training, 0.10) == pytest.approx(0.0)
    assert empirical_percentile(training, 0.50) == pytest.approx(0.50)
    assert empirical_percentile(training, 0.80) == pytest.approx(1.0)
    assert np.isnan(empirical_percentile(np.array([]), 0.50))


def test_fixed_recovery_fraction_has_no_tuned_shape_parameters():
    assert fixed_recovery_fraction(0.49) == pytest.approx(0.0)
    assert fixed_recovery_fraction(0.50) == pytest.approx(0.0)
    assert fixed_recovery_fraction(0.75) == pytest.approx(0.50)
    assert fixed_recovery_fraction(1.00) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fixed_recovery_fraction(0.75, activation_floor=1.0)


def test_preregistered_spec_keeps_stage1_model_and_beta_fixed(tmp_path):
    spec = Stage1RankCalibratedRecoverySizingSpec(root=tmp_path)

    assert spec.logistic_c == pytest.approx(0.03)
    assert spec.maximum_recovery_beta == pytest.approx(0.60)
    assert spec.percentile_activation_floor == pytest.approx(0.50)


def test_card_is_recorder_first_and_cannot_enable_production():
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    selected = json.loads(
        (CARD / "selected_model.json").read_text(encoding="utf-8")
    )

    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_role"] == "reused_diagnostic_only_not_model_selection"
    assert summary["production_change"] == "blocked"
    assert summary["preregistration"]["candidate_count"] == 1
    assert summary["preregistration"]["grid_search"] is False
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert selected["production_enabled"] is False
    assert not (CARD / "baseline_outer_replay.csv").exists()
    assert not (CARD / "challenger_outer_replay.csv").exists()
    assert not (CARD / "recovery_training_frame.csv").exists()


def test_notebook_exposes_qlib_dataset_recorder_and_chinese_conclusion():
    notebook = json.loads(
        (CARD / "stage1_rank_calibrated_recovery_sizing_v1.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = json.dumps(notebook, ensure_ascii=False)

    for marker in (
        "qlib.init",
        "qlib.data import D",
        "DatasetH",
        "DataHandlerLP",
        "qlib.workflow import R",
        "run_stage1_rank_calibrated_recovery_sizing_card",
        "中文结论",
    ):
        assert marker in text
