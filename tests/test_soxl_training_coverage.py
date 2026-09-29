import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.soxl_training_coverage import (
    _action_coverage, _artifact_dir, analog_training_coverage, auxiliary_coverage, feature_drift, overlap_clusters,
)


def test_label_overlap_clusters_merge_assets_and_use_shared_trading_calendar():
    calendar = pd.bdate_range("2020-01-01", periods=25)
    dates = [calendar[0], calendar[0], calendar[3], calendar[7], calendar[12]]
    result = overlap_clusters(dates, calendar=calendar, horizon=5)
    assert len(result) == 4
    assert result.iloc[:3].nunique() == 1
    assert result.iloc[3] != result.iloc[2]
    with pytest.raises(ValueError, match="absent"):
        overlap_clusters([pd.Timestamp("2019-01-01")], calendar=calendar)


def test_uniform_row_draws_and_weighted_loss_mass_are_distinct():
    calendar = pd.DatetimeIndex(pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04"]))
    auxiliary = pd.DataFrame({
        "datetime": [calendar[0], calendar[0], calendar[1], calendar[2]],
        "instrument": ["A", "B", "A", "A"], "training_weight": [1.0, 3.0, 1.0, 1.0],
        "synthetic_2x_crash_label": [1.0, 1.0, 0.0, 1.0],
    })
    tables = auxiliary_coverage(auxiliary, calendar=calendar, epochs=3, batch_size=2)
    yearly = tables["auxiliary_year_coverage"].set_index("year")
    assert yearly.loc[2020, "row_draw_probability"] == .75
    assert yearly.loc[2020, "nominal_loss_weight_share"] == pytest.approx(5 / 6)
    assert yearly.loc[2020, "expected_row_draws_over_training"] == 4.5
    assert yearly.unique_dates.sum() == 3
    assert len(tables["auxiliary_crash_overlap_clusters"]) == 1
    assert tables["auxiliary_sampling_concentration"].interpretation.str.contains("not gradient influence or IID").all()


def test_feature_drift_never_refits_reference_quantiles_or_retained_scaler_on_target():
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    scaler = {"mean": {"x": train.x.mean()}, "scale": {"x": train.x.std(ddof=1)}, "median": {"x": train.x.median()}}
    normal = feature_drift(train, pd.DataFrame({"x": [49.0, 50.0]}), features=["x"], scaler=scaler, feature_group="test")
    extreme = feature_drift(train, pd.DataFrame({"x": [10000.0, np.nan]}), features=["x"], scaler=scaler, feature_group="test")
    assert normal.loc[0, "train_q01"] == extreme.loc[0, "train_q01"] == pytest.approx(.99)
    assert normal.loc[0, "train_q99"] == extreme.loc[0, "train_q99"] == pytest.approx(98.01)
    assert extreme.loc[0, "target_outside_q01_q99_fraction"] == 1.0
    assert extreme.loc[0, "target_scaler_clip_rows"] == 1
    assert extreme.loc[0, "target_missing_rows"] == 1
    assert extreme.loc[0, "target_latest_outside_q01_q99"] is None
    assert extreme.loc[0, "retained_scaler_matches_reconstructed_training"]


def test_worktree_artifact_lookup_ignores_stale_absolute_recorder_path(tmp_path):
    recorder = tmp_path / "qlib_recorder" / "experiment" / "retained"
    recorder.mkdir(parents=True)
    (tmp_path / "summary.json").write_text(json.dumps({
        "experiment": "card", "recorder_id": "retained", "recorder_local_dir": "/other/workspace/missing",
    }))
    _, artifact = _artifact_dir(tmp_path)
    assert artifact == recorder / "artifacts" / "card"


def test_pre_walk_forward_analog_is_unavailable_not_replaced_with_in_sample_returns():
    events = pd.DataFrame([{
        "event_id": "early", "retrieval_kind": "A_retrospective", "window_spec": "A44",
        "cohort": "training_2011_2022", "representative": True, "representative_rank": 1,
        "support_start": "2015-01-01", "support_end": "2015-02-28",
        "feature_start": "2015-01-01", "feature_end": "2015-02-28",
    }])
    primary = pd.DataFrame(index=pd.bdate_range("2015-01-01", "2015-02-28"))
    auxiliary = pd.DataFrame({"datetime": pd.to_datetime(["2020-01-01"])})
    returns = pd.DataFrame({"segment": ["development_walk_forward"], "cost_bps": [10],
                            "trade_date": ["2016-01-04"], "exit_date": ["2016-01-05"],
                            "cash_weight": [.5], "strategy_return": [.01]})
    result = analog_training_coverage(events, primary=primary, auxiliary=auxiliary, base_returns=returns)
    assert result.loc[0, "primary_mature_training_rows"] > 0
    assert result.loc[0, "wf_base_model_rows"] == 0
    assert result.loc[0, "wf_base_model_return_pct"] is None
    assert result.loc[0, "evaluation_status"] == "unavailable_no_retained_walk_forward_rows"


def test_action_coverage_obeys_signal_segment_as_well_as_label_maturity(tmp_path):
    training = pd.DataFrame({
        "signal_date": ["2022-12-20", "2022-12-21", "2023-01-03"],
        "label_available_at": ["2022-12-22T14:31:00Z", "2023-01-05T14:31:00Z", "2023-01-04T14:30:00Z"],
        "execute_advantage": [.1, .1, .1], "risk_state": [0, 1, 2],
        "portfolio_drawdown": [.01, .02, .03], "current_beta_cap": [.2, .2, .2],
        "target_soxl_weight": [.05, .05, .05], "target_soxx_weight": [.1, .1, .1],
    })
    training.to_csv(tmp_path / "action_value_training_frame.csv", index=False)
    (tmp_path / "dataset_audit.json").write_text(json.dumps({
        "segments": {"train_outer": ["2016-01-04", "2022-12-31"]},
        "training_cutoffs": {"train_outer": "2023-01-04T14:31:00Z"}, "features": ["risk_state"],
    }))
    training.assign(segment="development_walk_forward", trade_date=training.signal_date).to_csv(tmp_path / "portfolio_report.csv", index=False)
    table, audit = _action_coverage(tmp_path, "2022-12-01", "2022-12-31")
    assert audit["actual_open_model_training_rows"] == 1
    actual_training = table.loc[table.state_source.eq("baseline_counterfactual_training")]
    assert actual_training.risk_state.tolist() == [0]
