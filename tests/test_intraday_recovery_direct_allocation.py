from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.workflow import intraday_recovery_direct_allocation as recovery
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
    _resolve_source_recorder,
    apply_fixed_decision_cost_stress,
    project_legal_weights,
    select_development_candidate,
    _simulate_policy,
)


def test_recovery_training_excludes_labels_not_mature_before_first_decision(monkeypatch):
    captured = {}

    class CaptureFit:
        def fit(self, features, labels):
            captured["indices"] = features.index.tolist()
            return self

    monkeypatch.setattr(recovery, "make_pipeline", lambda *steps: CaptureFit())
    training = pd.DataFrame(
        [{feature: 0.0 for feature in recovery.FEATURE_COLUMNS} for _ in range(22)]
    )
    training["year"] = 2021
    training["teacher_action_label"] = [float(index % 2) for index in range(22)]
    training["label_available_at"] = pd.to_datetime(
        ["2021-12-31T14:30Z"] * 20
        + ["2022-01-04T15:31Z", "2022-01-05T14:30Z"],
        utc=True,
    )
    training.attrs["recovery_training_cutoffs"] = {2022: "2022-01-04T15:31Z"}

    assert recovery._fit_recovery_model(training, year=2022, c_value=0.3) is not None
    assert captured["indices"] == list(range(20))
    with pytest.raises(ValueError, match="availability timestamps"):
        recovery._fit_recovery_model(
            training.drop(columns="label_available_at"), year=2022, c_value=0.3
        )


def test_project_legal_weights_enforces_three_asset_constraints():
    weights = project_legal_weights(0.30, 0.55, 0.60)

    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] <= 0.30
    assert weights[2] >= 0.15
    assert np.all(weights >= 0.0)
    assert 3.0 * weights[0] + weights[1] <= 0.60 + 1e-12


def test_selection_ignores_outer_even_when_outer_return_is_higher():
    candidates = pd.DataFrame(
        [
            {
                "segment": "development_walk_forward",
                "selection_uses_outer_test": False,
                "risk_window_pass_count": 7,
                "positive_relative_window_count": 6,
                "relative_ratio_p25": 1.01,
                "relative_ratio_median": 1.05,
                "terminal_wealth": 5.0,
                "candidate": "development_only",
            },
            {
                "segment": "outer_diagnostic",
                "selection_uses_outer_test": True,
                "risk_window_pass_count": 7,
                "positive_relative_window_count": 7,
                "relative_ratio_p25": 2.0,
                "relative_ratio_median": 2.0,
                "terminal_wealth": 50.0,
                "candidate": "outer_leak",
            },
        ]
    )

    selected = select_development_candidate(candidates)

    assert selected["candidate"] == "development_only"


def test_fixed_decision_stress_changes_cost_not_decisions():
    gross = np.array([0.02, -0.01, 0.03])
    turnover = np.array([0.5, 0.0, 0.25])

    stressed = apply_fixed_decision_cost_stress(gross, turnover, cost_bps=25.0)

    np.testing.assert_allclose(stressed, gross - turnover * 0.0025)
    with pytest.raises(ValueError):
        apply_fixed_decision_cost_stress(gross, turnover[:2], cost_bps=25.0)


def test_source_recorder_is_pinned_by_workflow(tmp_path):
    spec = IntradayRecoveryDirectAllocationSpec(root=tmp_path)
    spec.card_dir.mkdir(parents=True)
    spec.card_dir.joinpath("workflow.yaml").write_text(
        "source_recorder_id: frozen-recorder\n",
        encoding="utf-8",
    )
    recorder = (
        spec.source_card_dir
        / "qlib_recorder"
        / "experiment-id"
        / "frozen-recorder"
    )
    recorder.mkdir(parents=True)

    resolved, recorder_id = _resolve_source_recorder(
        spec,
        source_summary={"recorder_local_dir": "/unused/latest"},
    )

    assert resolved == recorder
    assert recorder_id == "frozen-recorder"


def test_policy_rejects_invalid_source_alpha_floor():
    with pytest.raises(ValueError, match="minimum_source_alpha_probability"):
        _simulate_policy(
            pd.DataFrame(),
            spec=None,
            teacher_frame=pd.DataFrame(),
            c_value=0.3,
            threshold=0.85,
            action_beta=0.6,
            first_action_year=2020,
            model_cache={},
            cost_bps=10.0,
            minimum_source_alpha_probability=1.01,
        )


def test_source_alpha_floor_blocks_conflicting_recovery(tmp_path):
    class AlwaysRecover:
        def predict_proba(self, frame):
            return np.tile(np.array([[0.01, 0.99]]), (len(frame), 1))

    dates = pd.to_datetime(["2020-03-12", "2020-03-13"])
    frame = pd.DataFrame(
        {
            "signal_date": dates,
            "trade_date": dates,
            "exit_date": dates + pd.Timedelta(days=1),
            "segment": "development_walk_forward",
            "soxl_weight": 0.30,
            "soxx_weight": 0.55,
            "cash_weight": 0.15,
            "soxl_return": [-0.10, 0.02],
            "soxx_return": [-0.10, 0.01],
            "early_ret_soxl": [-0.05, 0.01],
            "early_ret_soxx": [-0.03, 0.005],
            "early_range_soxl": -0.06,
            "early_range_soxx": -0.04,
            "early_from_low_soxl": 0.01,
            "early_from_low_soxx": 0.01,
            "early_volume_soxl": 1_000_000.0,
            "early_volume_soxx": 1_000_000.0,
            "tail_probability": 0.50,
            "alpha_probability": 0.10,
        }
    )
    spec = IntradayRecoveryDirectAllocationSpec(
        root=tmp_path,
        stop_drawdown=0.08,
    )
    kwargs = {
        "frame": frame,
        "spec": spec,
        "teacher_frame": pd.DataFrame(),
        "c_value": 0.3,
        "threshold": 0.825,
        "action_beta": 0.6,
        "first_action_year": 2020,
        "cost_bps": 10.0,
    }

    unvetoed = _simulate_policy(
        **kwargs,
        model_cache={(2020, 0.3): AlwaysRecover()},
        minimum_source_alpha_probability=0.0,
    )
    vetoed = _simulate_policy(
        **kwargs,
        model_cache={(2020, 0.3): AlwaysRecover()},
        minimum_source_alpha_probability=0.25,
    )

    assert bool(unvetoed.iloc[1]["recovery_action"]) is True
    assert bool(vetoed.iloc[1]["recovery_action"]) is False
    assert bool(vetoed.iloc[1]["source_alpha_veto"]) is True
