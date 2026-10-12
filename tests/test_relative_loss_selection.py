"""Native cohorts, causal selection, frozen-model identity, and loss geometry."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.data.convex_allocation import LABEL_COLUMNS, METADATA_COLUMNS
from src.data.generic_horizon import attach_horizon_scope
from src.data.integrated_allocation import build_native_dataset
from src.model.generic_annual_ridge_selection import qualify_annual_horizon_inputs, select_annual_horizon
from src.model.generic_horizon_estimator import predict_horizon
from src.model.relative_loss_selection import rescore_frozen_annual_candidates


FEATURES = ("common", "relative")
NAMES = ("prior", "ridge_10", "ridge_100", "ridge_1000")
PRIORITY = ("prior", "ridge_1000", "ridge_100", "ridge_10")
COMMON = dict(feature_columns=FEATURES, label_columns=LABEL_COLUMNS, availability_column="h5_available_at",
              horizon_sessions=5, minimum_training_rows=50, minimum_validation_rows=100,
              train_start="2013-01-01", hac_lags=5)


def _native(frame):
    dataset = build_native_dataset(frame, feature_columns=FEATURES, label_columns=LABEL_COLUMNS,
                                   metadata_columns=METADATA_COLUMNS, as_of="2020-01-01")
    attach_horizon_scope(dataset, 5)
    dataset.calendar = pd.bdate_range(frame.index.min(), frame.index.max() + pd.tseries.offsets.BDay(10))
    return dataset


@pytest.fixture(scope="module")
def native_parent():
    rng = np.random.default_rng(71)
    days = pd.bdate_range("2013-01-02", periods=1000)
    frame = pd.DataFrame(index=days)
    frame["signal_date"] = days
    frame["trade_date"] = days + pd.tseries.offsets.BDay(1)
    frame["exit_date"] = days + pd.tseries.offsets.BDay(2)
    frame["decision_at"] = (days + pd.Timedelta(hours=16, minutes=15)).tz_localize("America/New_York")
    frame["feature_available_at"] = (days + pd.Timedelta(hours=16)).tz_localize("America/New_York")
    frame["return_available_at"] = (frame.exit_date + pd.Timedelta(hours=9, minutes=31)).dt.tz_localize("America/New_York")
    frame["h5_available_at"] = (days + pd.tseries.offsets.BDay(6) + pd.Timedelta(hours=9, minutes=31)).tz_localize("America/New_York")
    frame["h5_exit_date"] = days + pd.tseries.offsets.BDay(6)
    frame["forecast_available_at"] = frame.return_available_at
    frame["auxiliary_available_at"] = frame.decision_at
    frame["common"], frame["relative"] = rng.normal(size=(2, len(days)))
    common_return = .04 * frame.common + rng.normal(scale=.03, size=len(days))
    frame[LABEL_COLUMNS[0]] = common_return + .01 * frame.relative + rng.normal(scale=.01, size=len(days))
    frame[LABEL_COLUMNS[1]] = common_return + rng.normal(scale=.01, size=len(days))
    dataset = _native(frame)
    parent = select_annual_horizon(dataset, frame, outer_year=2016, **COMMON)
    return frame, dataset, parent


def _rescore(dataset, frame, parent):
    return rescore_frozen_annual_candidates(dataset, frame, parent, outer_year=2016,
                                           candidate_names=NAMES, tie_priority=PRIORITY, **COMMON)


def test_relative_loss_uses_errors_difference_and_performs_no_fit(native_parent, monkeypatch):
    from src.model import generic_horizon_estimator as estimator
    frame, dataset, parent = native_parent
    def forbidden(*args, **kwargs):
        raise AssertionError("rescoring must not fit a model")
    monkeypatch.setattr(estimator, "fit_horizon_estimator", forbidden)
    monkeypatch.setattr(estimator, "fit_horizon_prior", forbidden)
    result = _rescore(dataset, frame, parent)
    q = qualify_annual_horizon_inputs(dataset, frame, outer_year=2016, **COMMON)
    query = frame.loc[q["dates"][q["validation_mask"]]]
    y = q["prepared"].loc[q["validation_mask"], "label"].to_numpy(float)
    for row in result["candidate_scores"].itertuples():
        pred = predict_horizon(parent["inner_models"][row.candidate_name], query,
                               feature_columns=FEATURES, horizon_sessions=5)["ridge"]
        expected = np.square((pred[:, 0] - pred[:, 1]) - (y[:, 0] - y[:, 1])).mean()
        assert row.relative_mse == pytest.approx(expected, rel=1e-14)
    assert result["parent_selected_name"] == parent["selected_name"]
    assert result["audit"]["model_fit_calls"] == 0
    assert result["audit"]["outer_outcomes_used_for_selection"] is False
    assert result["audit"]["portfolio_outcomes_used_for_selection"] is False


def test_future_and_unmatured_validation_labels_cannot_change_selection(native_parent):
    frame, dataset, parent = native_parent
    before = _rescore(dataset, frame, parent)
    altered = frame.copy()
    immature = pd.to_datetime(altered.h5_available_at, utc=True) >= pd.Timestamp("2016-01-01", tz="America/New_York")
    altered.loc[immature, LABEL_COLUMNS[0]] = .8
    altered.loc[immature, LABEL_COLUMNS[1]] = -.4
    after = _rescore(_native(altered), altered, parent)
    assert before["selected_name"] == after["selected_name"]
    pd.testing.assert_frame_equal(before["candidate_scores"], after["candidate_scores"], check_exact=True)


@pytest.mark.parametrize("field,value,match", [
    ("model_year", 2016, "clock"),
    ("feature_columns", ("common", "future_label"), "schema"),
])
def test_saved_candidate_with_wrong_year_or_schema_blocks(native_parent, field, value, match):
    frame, dataset, parent = native_parent
    bad = copy.deepcopy(parent)
    bad["inner_models"]["ridge_10"][field] = value
    with pytest.raises(ValueError, match=match):
        _rescore(dataset, frame, bad)


def test_candidate_training_hash_and_parent_loss_must_reproduce(native_parent):
    frame, dataset, parent = native_parent
    bad = copy.deepcopy(parent)
    bad["inner_models"]["prior"]["audit"]["training_frame_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="training bytes"):
        _rescore(dataset, frame, bad)
    bad = copy.deepcopy(parent)
    bad["candidate_scores"].loc[0, "normalized_mse"] += .2
    with pytest.raises(ValueError, match="cannot be reproduced"):
        _rescore(dataset, frame, bad)


def test_pandas_only_or_missing_candidate_is_rejected(native_parent):
    frame, dataset, parent = native_parent
    with pytest.raises(ValueError, match="DatasetH"):
        _rescore(frame, frame, parent)
    bad = copy.deepcopy(parent)
    bad["inner_models"].pop("ridge_100")
    with pytest.raises(ValueError, match="inventory"):
        _rescore(dataset, frame, bad)


def test_fixed_linear_response_coordinate_is_not_new_information(native_parent):
    frame, dataset, parent = native_parent
    model = parent["inner_models"]["ridge_10"]
    query = frame.loc[frame.index.year == 2015]
    prediction = predict_horizon(model, query, feature_columns=FEATURES, horizon_sessions=5)["ridge"]
    design = np.c_[np.ones(len(query)), (query[list(FEATURES)].to_numpy() - model["feature_mean"]) / model["feature_scale"]]
    relative_coefficients = model["ridge_coefficients"][:, 0] - model["ridge_coefficients"][:, 1]
    np.testing.assert_allclose(design @ relative_coefficients, prediction[:, 0] - prediction[:, 1], rtol=0, atol=1e-15)
