from __future__ import annotations

import importlib
import pickle

import numpy as np
import pandas as pd
import pytest
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


def _dataset(
    *,
    periods: int = 360,
    post_train_feature_shift: float = 0.0,
    test_label_shift: float = 0.0,
    validation_invalid_labels: int = 0,
    reversal_finite_train_targets: int | None = None,
) -> DatasetH:
    random = np.random.default_rng(20260717)
    dates = pd.bdate_range("2024-01-02", periods=periods)
    train_end = int(periods * 2 / 3) - 1
    valid_end = int(periods * 5 / 6) - 1
    index = pd.MultiIndex.from_arrays(
        [["SOXL"] * periods, dates], names=["instrument", "datetime"]
    )

    lagged_return = random.normal(0.0, 1.0, periods)
    trend = random.normal(0.0, 1.0, periods)
    volatility = random.normal(0.0, 1.0, periods)
    breadth = random.normal(0.0, 1.0, periods)
    cross_asset_momentum = random.normal(0.0, 1.0, periods)
    oversold = random.normal(0.0, 1.0, periods)
    residual = random.normal(0.0, 1.0, periods)
    daily_range = random.normal(0.0, 1.0, periods)
    volatility_gap = random.normal(0.0, 1.0, periods)

    future_return = (
        0.018 * trend
        + 0.011 * breadth
        + 0.010 * cross_asset_momentum
        - 0.014 * oversold
        - 0.009 * residual
        - 0.016 * np.maximum(volatility, 0.0)
        + random.normal(0.0, 0.006, periods)
    )
    future_downside = np.maximum(-future_return, 0.0)
    crash_cutoff = float(np.quantile(future_return[: train_end + 1], 0.10))
    future_crash = (future_return <= crash_cutoff).astype("float64")

    if reversal_finite_train_targets is not None:
        train_slice = slice(0, train_end + 1)
        return_threshold = float(np.median(lagged_return[train_slice]))
        trend_threshold = float(np.median(trend[train_slice]))
        volatility_threshold = float(np.quantile(volatility[train_slice], 2.0 / 3.0))
        stress_regime = volatility[train_slice] >= volatility_threshold
        trend_regime = (~stress_regime) & (
            (trend[train_slice] >= trend_threshold)
            | (lagged_return[train_slice] >= return_threshold)
        )
        reversal_rows = np.flatnonzero(~stress_regime & ~trend_regime)
        invalid_rows = reversal_rows[int(reversal_finite_train_targets) :]
        future_return[invalid_rows] = np.nan
        future_downside[invalid_rows] = np.nan

    if post_train_feature_shift:
        for values in (lagged_return, trend, volatility):
            values[train_end + 1 :] += post_train_feature_shift
    future_return = future_return.copy()
    future_return[valid_end + 1 :] += test_label_shift
    future_downside = future_downside.copy()
    future_downside[valid_end + 1 :] = np.maximum(
        -future_return[valid_end + 1 :], 0.0
    )
    future_crash = future_crash.copy()
    future_crash[valid_end + 1 :] = (
        future_return[valid_end + 1 :] <= crash_cutoff
    ).astype("float64")
    if validation_invalid_labels:
        valid_rows = np.arange(train_end + 1, valid_end + 1)
        invalid_rows = valid_rows[: int(validation_invalid_labels)]
        future_return[invalid_rows] = np.nan
        future_downside[invalid_rows] = np.nan

    columns = [
        ("feature", "lagged_return"),
        ("feature", "trend"),
        ("feature", "volatility"),
        ("feature", "breadth"),
        ("feature", "cross_asset_momentum"),
        ("feature", "oversold"),
        ("feature", "residual"),
        ("feature", "range"),
        ("feature", "volatility_gap"),
        ("label", "future_return_5d"),
        ("label", "future_downside_5d"),
        ("label", "future_crash_5d"),
    ]
    frame = pd.DataFrame(
        np.column_stack(
            [
                lagged_return,
                trend,
                volatility,
                breadth,
                cross_asset_momentum,
                oversold,
                residual,
                daily_range,
                volatility_gap,
                future_return,
                future_downside,
                future_crash,
            ]
        ),
        index=index,
        columns=pd.MultiIndex.from_tuples(columns),
    )
    return DatasetH(
        handler=DataHandlerLP.from_df(frame),
        segments={
            "train": (dates[0], dates[train_end]),
            "valid": (dates[train_end + 1], dates[valid_end]),
            "test": (dates[valid_end + 1], dates[-1]),
        },
    )


def _fast_spec(*, min_regime_observations: int = 40):
    module = importlib.import_module("src.model.regime_moe")
    return module.RegimeMoESpec(
        random_state=29,
        catboost_iterations=12,
        min_regime_observations=min_regime_observations,
        floor_cap_candidates=((0.15, 0.60), (0.15, 0.75)),
    )


def test_regime_moe_module_exposes_focused_workflow_api():
    module = importlib.import_module("src.model.regime_moe")

    assert module.RegimeMoESpec
    assert module.RegimeMoEResult
    assert module.FittedRegimeMoEPolicy
    assert module.run_regime_moe_models


def test_regime_moe_uses_train_thresholds_soft_gate_and_specialists():
    module = importlib.import_module("src.model.regime_moe")
    result = module.run_regime_moe_models(_dataset(), spec=_fast_spec())

    assert result.selection_evidence["route_status"] == "ok"
    assert result.selection_evidence["threshold_fit_segment"] == "train"
    assert result.selection_evidence["selection_segment"] == "valid"
    assert result.selection_evidence["test_labels_used_for_selection"] is False
    assert result.selection_evidence["test_accessed_after_freeze"] is True
    assert result.selected_candidate == "validation_selected_regime_moe"
    assert set(result.regime_occupancy) == {"trend", "reversal", "stress"}
    assert min(result.regime_occupancy.values()) >= 40

    comparison = result.model_comparison.set_index("candidate")
    assert comparison.loc["trend_ridge_expert", "status"] == "ok"
    assert comparison.loc["reversal_catboost_expert", "status"] == "ok"
    assert comparison.loc["stress_downside_expert", "status"] == "ok"
    assert comparison.loc["multinomial_soft_gate", "status"] == "ok"

    for segment in ("train", "valid", "test"):
        probabilities = result.gate_probabilities_by_segment[segment]
        assert list(probabilities.columns) == ["trend", "reversal", "stress"]
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
        assert probabilities.ge(0.0).all().all()
        assert probabilities.le(1.0).all().all()
        predictions = result.predictions_by_segment[segment]
        positions = result.positions_by_segment[segment]
        assert np.isfinite(predictions.to_numpy(dtype="float64")).all()
        assert np.isfinite(positions.to_numpy(dtype="float64")).all()
        assert ((positions == 0.0) | positions.between(result.floor, result.cap)).all()


def test_regime_thresholds_and_frozen_policy_ignore_post_train_data_and_test_labels():
    module = importlib.import_module("src.model.regime_moe")
    baseline = module.run_regime_moe_models(_dataset(), spec=_fast_spec())
    perturbed_labels = module.run_regime_moe_models(
        _dataset(test_label_shift=-3.0), spec=_fast_spec()
    )
    perturbed_features = module.run_regime_moe_models(
        _dataset(post_train_feature_shift=100.0), spec=_fast_spec()
    )

    assert perturbed_labels.regime_thresholds == baseline.regime_thresholds
    assert perturbed_labels.selection_evidence["frozen_policy"] == baseline.selection_evidence[
        "frozen_policy"
    ]
    pd.testing.assert_series_equal(
        perturbed_labels.positions_by_segment["test"],
        baseline.positions_by_segment["test"],
    )
    assert perturbed_features.regime_thresholds == baseline.regime_thresholds


def test_sparse_regime_blocks_route_deterministically_without_partial_fit():
    module = importlib.import_module("src.model.regime_moe")
    dataset = _dataset(periods=120)

    result = module.run_regime_moe_models(
        dataset, spec=_fast_spec(min_regime_observations=40)
    )

    assert result.selection_evidence["route_status"] == "blocked_sparse_regime"
    assert result.selection_evidence["blocked_reason"] == "blocked_sparse_regime"
    assert min(result.regime_occupancy.values()) < 40
    assert result.fitted_policy is None
    assert result.selected_candidate == "blocked_sparse_regime"
    assert (result.failure_rows["reason_code"] == "blocked_sparse_regime").all()
    for positions in result.positions_by_segment.values():
        assert np.isfinite(positions.to_numpy()).all()
        assert positions.eq(0.0).all()


def test_sparse_gate_uses_expert_usable_rows_not_raw_regime_occupancy():
    module = importlib.import_module("src.model.regime_moe")

    result = module.run_regime_moe_models(
        _dataset(reversal_finite_train_targets=1), spec=_fast_spec()
    )

    evidence = result.selection_evidence
    assert evidence["route_status"] == "blocked_sparse_regime"
    assert evidence["raw_regime_occupancy"]["reversal"] >= 40
    assert evidence["usable_regime_observations"]["reversal"] == 1
    assert result.regime_occupancy["reversal"] >= 40
    assert result.regime_usable_observations["reversal"] == 1
    assert result.fitted_policy is None


@pytest.mark.parametrize("invalid_count", [1, 60])
def test_incomplete_validation_return_labels_fail_closed(invalid_count):
    module = importlib.import_module("src.model.regime_moe")

    result = module.run_regime_moe_models(
        _dataset(validation_invalid_labels=invalid_count), spec=_fast_spec()
    )

    evidence = result.selection_evidence
    assert evidence["route_status"] == "blocked_incomplete_validation_labels"
    label_quality = evidence["validation_label_quality"]
    assert label_quality["reason_code"] == "incomplete_validation_labels"
    assert label_quality["invalid_count"] == invalid_count
    assert label_quality["first_offending_date"]
    assert result.fitted_policy is None
    for positions in result.positions_by_segment.values():
        assert positions.eq(0.0).all()


def test_dataset_contract_uses_dk_l_then_delayed_dk_i_test_access():
    module = importlib.import_module("src.model.regime_moe")
    native = _dataset()

    class RecordingDataset:
        def __init__(self):
            self.calls = []

        def prepare(self, segment, *, col_set, data_key):
            normalized = (col_set,) if isinstance(col_set, str) else tuple(col_set)
            self.calls.append((segment, normalized, data_key))
            return native.prepare(segment, col_set=col_set, data_key=data_key)

    dataset = RecordingDataset()
    result = module.run_regime_moe_models(dataset, spec=_fast_spec())

    assert dataset.calls == [
        ("train", ("feature", "label"), DataHandlerLP.DK_L),
        ("valid", ("feature", "label"), DataHandlerLP.DK_L),
        ("test", ("feature",), DataHandlerLP.DK_I),
    ]
    assert result.selection_evidence["test_accessed_after_freeze"] is True


def test_data_quality_fail_close_and_test_prediction_failure_are_zero_finite(monkeypatch):
    module = importlib.import_module("src.model.regime_moe")
    dataset = _dataset()
    test_index = dataset.prepare("test", col_set="feature").index
    quality = pd.Series(True, index=test_index)
    quality.iloc[2] = False
    result = module.run_regime_moe_models(
        dataset,
        spec=_fast_spec(),
        data_quality_by_segment={"test": quality},
    )
    assert result.selection_evidence["route_status"] == "blocked_data_quality"
    assert result.positions_by_segment["test"].iloc[2] == 0.0

    original = module._predict_policy_segment

    def fail_test(policy, features, quality, *, segment):
        if segment == "test":
            raise RuntimeError("frozen inference failed")
        return original(policy, features, quality, segment=segment)

    monkeypatch.setattr(module, "_predict_policy_segment", fail_test)
    failed = module.run_regime_moe_models(_dataset(), spec=_fast_spec())
    assert failed.selection_evidence["route_status"] == "blocked_prediction_failed"
    assert failed.selection_evidence["test_prediction_failure"]["reason_code"] == "prediction_failed"
    assert failed.positions_by_segment["test"].eq(0.0).all()
    assert np.isfinite(failed.predictions_by_segment["test"].to_numpy()).all()


def test_single_expert_failure_is_retained_and_fallback_keeps_route_running(monkeypatch):
    module = importlib.import_module("src.model.regime_moe")

    def fail_reversal(*args, **kwargs):
        raise RuntimeError("candidate failure")

    monkeypatch.setattr(module, "_fit_reversal_expert", fail_reversal)
    result = module.run_regime_moe_models(_dataset(), spec=_fast_spec())

    comparison = result.model_comparison.set_index("candidate")
    assert comparison.loc["reversal_catboost_expert", "status"] == "failed"
    assert comparison.loc["reversal_catboost_expert", "reason_code"] == "fit_failed"
    assert comparison.loc["reversal_train_mean_fallback", "status"] == "ok"
    assert result.selection_evidence["route_status"] == "ok_degraded_candidate_failure"
    assert np.isfinite(result.positions_by_segment["test"].to_numpy()).all()


def test_fitted_policy_pickle_round_trip_has_exact_inference_parity():
    module = importlib.import_module("src.model.regime_moe")
    dataset = _dataset()
    result = module.run_regime_moe_models(dataset, spec=_fast_spec())
    restored = pickle.loads(pickle.dumps(result.fitted_policy))
    test_features = dataset.prepare(
        "test", col_set="feature", data_key=DataHandlerLP.DK_I
    )

    actual = restored.predict(test_features, segment="test")

    pd.testing.assert_frame_equal(actual, result.predictions_by_segment["test"])


def test_config_forbids_test_selection_and_invalid_floor_cap():
    module = importlib.import_module("src.model.regime_moe")

    with pytest.raises(ValueError, match="validation segment"):
        module.RegimeMoESpec(selection_segment="test")
    with pytest.raises(ValueError, match="distinct"):
        module.RegimeMoESpec(train_segment="valid")
    with pytest.raises(ValueError, match="floor/cap"):
        module.RegimeMoESpec(floor_cap_candidates=((0.8, 0.2),))
