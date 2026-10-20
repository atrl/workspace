import numpy as np
import pandas as pd
import pytest

from src.data.path_area_bridge import AREA_COLUMN, CONTROL_COLUMNS
from src.model.path_area_action import (
    build_arm_features,
    feature_order,
    fit_one_arm,
    preflight_training_rows,
)
from src.workflow.stage1_open_action_value import ACTION_FEATURES


def _prepared_rows():
    # 205 strictly mature pre-2022 rows, five later eligible rows, and four
    # deliberately ineligible/invalid boundary rows. All data are synthetic.
    count = 215
    index = pd.Index(np.arange(count), name="row")
    features = pd.DataFrame(index=index, columns=feature_order(), dtype="float64")
    for position, name in enumerate(ACTION_FEATURES):
        features[name] = np.arange(count, dtype="float64") / (position + 1) + position
    features.loc[:, list(CONTROL_COLUMNS)] = np.column_stack([
        np.arange(count, dtype="float64") / (j + 1) for j in range(len(CONTROL_COLUMNS))
    ])
    features["prefix_mask"] = 1.0
    features[AREA_COLUMN] = np.arange(count, dtype="float64") / 100.0 - 0.5
    missing_prefix = np.arange(count) % 20 == 0
    path_columns = [*CONTROL_COLUMNS, AREA_COLUMN]
    features.loc[missing_prefix, path_columns] = np.nan
    features.loc[missing_prefix, "prefix_mask"] = 0.0

    labels = pd.Series(np.where(np.arange(count) % 2, 0.01, 0.0), index=index,
                       name="local_execute_delta")
    metadata = pd.DataFrame(index=index)
    metadata["eligible"] = True
    trade_dates = [pd.Timestamp("2021-01-04") + pd.Timedelta(days=i) for i in range(count)]
    metadata["trade_date"] = trade_dates
    metadata["label_available_at"] = [
        pd.Timestamp("2021-02-01", tz="America/New_York") + pd.Timedelta(days=i)
        for i in range(count)
    ]
    metadata.loc[205:214, "trade_date"] = [pd.Timestamp("2022-02-01") + pd.Timedelta(days=i)
                                                   for i in range(10)]
    metadata.loc[210, "label_available_at"] = pd.Timestamp("2023-01-01", tz="America/New_York")
    metadata.loc[211, "eligible"] = False
    features.loc[212, ACTION_FEATURES[0]] = np.nan
    labels.loc[213] = np.nan
    metadata["label_status"] = "mature"
    metadata.loc[214, "label_status"] = "unknown"
    return features, labels, metadata


def test_preflight_uses_common_mature_rows_and_fits_matched_area_arms():
    features, labels, metadata = _prepared_rows()
    preflight = preflight_training_rows(features, labels, metadata)

    assert preflight.ready
    assert preflight.row_count == 210
    assert preflight.pre_2022_rows == 205
    assert preflight.observed_area_rows > 0
    assert preflight.class_counts == {0: 105, 1: 105}
    assert preflight.target.loc[0] == 0  # Delta tie is HOLD.
    assert 210 not in preflight.features.index  # Exact cutoff is excluded.
    assert 211 not in preflight.features.index  # Original guard is retained.
    assert 212 not in preflight.features.index  # Invalid original 22 is excluded.
    assert 213 not in preflight.features.index  # Unknown label is excluded.
    assert 214 not in preflight.features.index  # Non-mature label status is excluded.

    b_features = build_arm_features(preflight.features, "B")
    c_features = build_arm_features(preflight.features, "C")
    assert b_features.index.equals(c_features.index)
    assert tuple(b_features.columns) == tuple(c_features.columns) == tuple(preflight.features.columns)
    assert b_features[AREA_COLUMN].eq(0.0).all()
    assert c_features.loc[1, AREA_COLUMN] != b_features.loc[1, AREA_COLUMN]
    assert c_features.loc[0, list(CONTROL_COLUMNS)].isna().all()
    assert c_features.loc[0, "prefix_mask"] == 0

    fitted_b = fit_one_arm(preflight, "B")
    fitted_c = fit_one_arm(preflight, "C")
    assert fitted_b.feature_order == fitted_c.feature_order == tuple(feature_order())
    assert fitted_b.row_count == fitted_c.row_count == preflight.row_count
    assert fitted_b.class_counts == fitted_c.class_counts == preflight.class_counts
    assert fitted_b.pipeline.named_steps["imputer"].transform(
        build_arm_features(preflight.features, "B")).shape[1] == 32
    assert fitted_c.pipeline.named_steps["scaler"].n_features_in_ == 32
    assert fitted_c.pipeline.named_steps["classifier"].n_features_in_ == 32
    assert fitted_b.cutoff == fitted_c.cutoff == pd.Timestamp(
        "2023-01-01 00:00:00", tz="America/New_York")

    nonnormalized = metadata.copy()
    nonnormalized.loc[0, "trade_date"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="trade_date values must be normalized"):
        preflight_training_rows(features, labels, nonnormalized)
