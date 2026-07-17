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
    periods: int = 240,
    test_label_shift: float = 0.0,
    nonfinite_test_feature: bool = False,
    nonfinite_validation_downside: bool = False,
) -> DatasetH:
    random = np.random.default_rng(20260717)
    dates = pd.bdate_range("2023-01-03", periods=periods)
    train_end = int(periods * 0.60) - 1
    valid_end = int(periods * 0.80) - 1
    index = pd.MultiIndex.from_arrays(
        [["SOXL"] * periods, dates], names=["instrument", "datetime"]
    )

    soxx_return = random.normal(0.0005, 0.012, periods)
    vix_change = random.normal(0.0, 0.04, periods)
    vix_level = np.clip(20.0 + np.cumsum(vix_change), 10.0, 55.0)
    soxl_return = 2.7 * soxx_return - 0.018 * vix_change + random.normal(0.0, 0.008, periods)
    soxl_trend = pd.Series(soxl_return).rolling(10, min_periods=1).mean().to_numpy()
    soxl_volatility = pd.Series(soxl_return).rolling(10, min_periods=2).std().fillna(0.0).to_numpy()
    future_return = (
        0.30 * soxl_trend
        + 0.06 * soxx_return
        - 0.018 * np.maximum(vix_change, 0.0)
        + random.normal(0.0, 0.004, periods)
    )
    future_downside = np.maximum(-future_return, 0.0)
    crash_cutoff = float(np.quantile(future_return[: train_end + 1], 0.10))
    future_crash = (future_return <= crash_cutoff).astype("float64")
    future_return = future_return.copy()
    future_return[valid_end + 1 :] += test_label_shift
    future_downside[valid_end + 1 :] = np.maximum(-future_return[valid_end + 1 :], 0.0)
    future_crash[valid_end + 1 :] = (
        future_return[valid_end + 1 :] <= crash_cutoff
    ).astype("float64")
    if nonfinite_test_feature:
        soxl_return = soxl_return.copy()
        soxl_return[valid_end + 3] = np.nan
    if nonfinite_validation_downside:
        future_downside = future_downside.copy()
        future_downside[train_end + 2] = np.nan

    columns = pd.MultiIndex.from_tuples(
        [
            ("feature", "soxl_return"),
            ("feature", "soxx_return"),
            ("feature", "vix_level"),
            ("feature", "vix_change"),
            ("feature", "soxl_trend"),
            ("feature", "soxl_volatility"),
            ("label", "future_return_5d"),
            ("label", "future_downside_5d"),
            ("label", "future_crash_5d"),
        ]
    )
    frame = pd.DataFrame(
        np.column_stack(
            [
                soxl_return,
                soxx_return,
                vix_level,
                vix_change,
                soxl_trend,
                soxl_volatility,
                future_return,
                future_downside,
                future_crash,
            ]
        ),
        index=index,
        columns=columns,
    )
    return DatasetH(
        handler=DataHandlerLP.from_df(frame),
        segments={
            "train": (dates[0], dates[train_end]),
            "valid": (dates[train_end + 1], dates[valid_end]),
            "test": (dates[valid_end + 1], dates[-1]),
        },
    )


def _fast_spec(**kwargs):
    module = importlib.import_module("src.model.panel_sequence")
    defaults = dict(
        random_state=31,
        hidden_width=8,
        temporal_layers=1,
        dropout=0.10,
        max_epochs=2,
        early_stopping_patience=1,
        batch_size=32,
        windows=(20, 60),
        floor_cap_candidates=((0.15, 0.60), (0.15, 0.75)),
    )
    defaults.update(kwargs)
    return module.PanelSequenceSpec(**defaults)


def test_panel_sequence_module_exposes_focused_workflow_api():
    module = importlib.import_module("src.model.panel_sequence")

    assert module.PanelSequenceSpec
    assert module.PanelSequenceResult
    assert module.FittedPanelSequencePolicy
    assert module.SequenceWindowBatch
    assert module.build_sequence_windows
    assert module.run_panel_sequence_models


def test_sequence_windows_end_at_decision_row_and_ignore_future_perturbations():
    module = importlib.import_module("src.model.panel_sequence")
    index = pd.bdate_range("2026-01-02", periods=30)
    features = pd.DataFrame(
        {"soxl_return": np.arange(30.0), "vix_level": np.arange(100.0, 130.0)},
        index=index,
    )
    targets = pd.Series(np.arange(30.0) / 100.0, index=index)

    baseline = module.build_sequence_windows(features, targets, window=20)
    perturbed = features.copy()
    perturbed.iloc[25:, :] += 10_000.0
    changed = module.build_sequence_windows(perturbed, targets, window=20)

    assert baseline.values.shape == (11, 20, 2)
    assert baseline.endpoints[0] == index[19]
    assert baseline.source_endpoints[0] == index[19]
    assert baseline.source_starts[0] == index[0]
    np.testing.assert_allclose(baseline.values[:6], changed.values[:6])
    np.testing.assert_allclose(baseline.targets, targets.iloc[19:].to_numpy())


def test_compact_candidates_are_deterministic_validation_selected_and_bounded():
    module = importlib.import_module("src.model.panel_sequence")
    first = module.run_panel_sequence_models(_dataset(), spec=_fast_spec())
    second = module.run_panel_sequence_models(_dataset(), spec=_fast_spec())

    evidence = first.selection_evidence
    assert evidence["route_status"] == "ok"
    assert evidence["selection_segment"] == "valid"
    assert evidence["test_labels_used_for_selection"] is False
    assert evidence["test_accessed_after_freeze"] is True
    assert evidence["label_horizon"] == 5
    assert evidence["purge_rows"] == 6
    assert evidence["random_state"] == 31
    assert evidence["selected_window"] in (20, 60)
    assert evidence["selected_family"] in ("tcn", "gru_alstm", "transformer")
    assert evidence["network_contract"] == {
        "hidden_width": 8,
        "temporal_layers": 1,
        "dropout": 0.1,
    }
    comparison = first.model_comparison.set_index("candidate")
    for family in ("tcn", "gru_alstm", "transformer"):
        for window in (20, 60):
            assert comparison.loc[f"{family}_w{window}", "status"] == "ok"
            assert comparison.loc[f"{family}_w{window}", "selected"] in (True, False)
    assert comparison.loc["pit_cross_sectional_panel", "status"] == "blocked"
    assert comparison.loc["pit_cross_sectional_panel", "reason_code"] == (
        "blocked_missing_pit_panel"
    )
    assert first.selected_candidate == second.selected_candidate
    pd.testing.assert_series_equal(
        first.positions_by_segment["test"], second.positions_by_segment["test"]
    )
    for segment in ("train", "valid", "test"):
        positions = first.positions_by_segment[segment]
        assert np.isfinite(positions.to_numpy()).all()
        assert ((positions == 0.0) | positions.between(first.floor, first.cap)).all()


def test_test_labels_do_not_change_frozen_policy_or_test_predictions():
    module = importlib.import_module("src.model.panel_sequence")
    baseline = module.run_panel_sequence_models(_dataset(), spec=_fast_spec())
    perturbed = module.run_panel_sequence_models(
        _dataset(test_label_shift=-3.0), spec=_fast_spec()
    )

    assert baseline.selected_candidate == perturbed.selected_candidate
    assert baseline.selection_evidence["frozen_policy"] == perturbed.selection_evidence[
        "frozen_policy"
    ]
    pd.testing.assert_frame_equal(
        baseline.predictions_by_segment["test"],
        perturbed.predictions_by_segment["test"],
    )


def test_dataset_contract_reads_test_dk_i_only_after_policy_freeze():
    module = importlib.import_module("src.model.panel_sequence")
    native = _dataset()

    class RecordingDataset:
        def __init__(self):
            self.calls = []

        def prepare(self, segment, *, col_set, data_key):
            normalized = (col_set,) if isinstance(col_set, str) else tuple(col_set)
            self.calls.append((segment, normalized, data_key))
            return native.prepare(segment, col_set=col_set, data_key=data_key)

    dataset = RecordingDataset()
    result = module.run_panel_sequence_models(dataset, spec=_fast_spec())

    assert dataset.calls == [
        ("train", ("feature", "label"), DataHandlerLP.DK_L),
        ("valid", ("feature", "label"), DataHandlerLP.DK_L),
        ("test", ("feature",), DataHandlerLP.DK_I),
    ]
    assert result.selection_evidence["test_accessed_after_freeze"] is True


def test_missing_pit_panel_is_separate_deterministic_block_and_valid_pit_is_eligible():
    module = importlib.import_module("src.model.panel_sequence")
    blocked = module.run_panel_sequence_models(_dataset(), spec=_fast_spec())
    assert blocked.mode_evidence["diagnostic_multivariate"]["status"] == "ok"
    assert blocked.mode_evidence["constituent_cross_sectional"]["status"] == (
        "blocked_missing_pit_panel"
    )
    assert blocked.blocked_mode_positions["constituent_cross_sectional"].eq(0.0).all()

    dates = pd.bdate_range("2025-01-02", periods=3)
    index = pd.MultiIndex.from_product(
        [dates, ["NVDA", "AMD"]], names=["datetime", "instrument"]
    )
    pit = pd.DataFrame(
        {
            "pit_member": True,
            "feature_1": np.arange(len(index), dtype="float64"),
        },
        index=index,
    )
    pit.attrs["point_in_time_membership"] = True
    eligible = module.run_panel_sequence_models(
        _dataset(), spec=_fast_spec(), pit_panel=pit
    )
    assert eligible.mode_evidence["constituent_cross_sectional"]["status"] == "eligible"
    assert eligible.model_comparison.set_index("candidate").loc[
        "pit_cross_sectional_panel", "status"
    ] == "eligible"


def test_nonfinite_test_feature_and_external_quality_fail_closed_to_finite_zero():
    module = importlib.import_module("src.model.panel_sequence")
    dataset = _dataset(nonfinite_test_feature=True)
    test_index = dataset.prepare("test", col_set="feature").index
    supplied = pd.Series(True, index=test_index)
    supplied.iloc[7] = False

    result = module.run_panel_sequence_models(
        dataset,
        spec=_fast_spec(),
        data_quality_by_segment={"test": supplied},
    )

    assert result.selection_evidence["route_status"] == "blocked_data_quality"
    assert result.selection_evidence["data_quality_evidence"]["test"]["invalid_count"] >= 2
    assert np.isfinite(result.positions_by_segment["test"].to_numpy()).all()
    assert result.positions_by_segment["test"].iloc[3] == 0.0
    assert result.positions_by_segment["test"].iloc[7] == 0.0


def test_incomplete_shared_validation_labels_block_before_model_selection():
    module = importlib.import_module("src.model.panel_sequence")

    result = module.run_panel_sequence_models(
        _dataset(nonfinite_validation_downside=True), spec=_fast_spec()
    )

    assert result.selection_evidence["route_status"] == "blocked_data_quality"
    assert result.fitted_policy is None
    assert result.positions_by_segment["test"].eq(0.0).all()


def test_frozen_test_inference_failure_has_explicit_fail_closed_status(monkeypatch):
    module = importlib.import_module("src.model.panel_sequence")
    original = module._predict_policy_segment

    def fail_test(policy, features, quality, *, segment):
        if segment == "test":
            raise RuntimeError("frozen inference failed")
        return original(policy, features, quality, segment=segment)

    monkeypatch.setattr(module, "_predict_policy_segment", fail_test)
    result = module.run_panel_sequence_models(_dataset(), spec=_fast_spec())

    assert result.selection_evidence["route_status"] == "blocked_prediction_failed"
    assert result.selection_evidence["test_prediction_failure"]["reason_code"] == (
        "prediction_failed"
    )
    assert result.positions_by_segment["test"].eq(0.0).all()
    assert np.isfinite(result.predictions_by_segment["test"].to_numpy()).all()


def test_one_backend_failure_is_retained_and_other_candidates_continue(monkeypatch):
    module = importlib.import_module("src.model.panel_sequence")
    original = module._fit_torch_candidate

    def fail_tcn(*args, **kwargs):
        if kwargs["family"] == "tcn":
            raise RuntimeError("tcn backend failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_fit_torch_candidate", fail_tcn)
    result = module.run_panel_sequence_models(_dataset(), spec=_fast_spec())
    comparison = result.model_comparison.set_index("candidate")

    assert comparison.loc["tcn_w20", "status"] == "failed"
    assert comparison.loc["tcn_w20", "reason_code"] == "fit_failed"
    assert comparison.loc["gru_alstm_w20", "status"] == "ok"
    assert result.selection_evidence["route_status"] == "ok_degraded_candidate_failure"
    assert np.isfinite(result.positions_by_segment["test"].to_numpy()).all()


def test_all_backend_failures_return_zero_finite_blocked_result(monkeypatch):
    module = importlib.import_module("src.model.panel_sequence")

    def fail_all(*args, **kwargs):
        raise ModuleNotFoundError("torch unavailable")

    monkeypatch.setattr(module, "_fit_torch_candidate", fail_all)
    result = module.run_panel_sequence_models(_dataset(), spec=_fast_spec())

    assert result.selection_evidence["route_status"] == "blocked_model_backend_unavailable"
    assert result.fitted_policy is None
    assert result.selected_candidate == "blocked_model_backend_unavailable"
    assert (result.failure_rows["reason_code"] == "backend_unavailable").all()
    for positions in result.positions_by_segment.values():
        assert np.isfinite(positions.to_numpy()).all()
        assert positions.eq(0.0).all()


def test_nonfinite_candidate_predictions_are_retained_and_all_fail_closed(monkeypatch):
    module = importlib.import_module("src.model.panel_sequence")

    def nonfinite(self, values):
        return np.full(len(values), np.nan, dtype="float64")

    monkeypatch.setattr(module._FittedTorchSequenceEstimator, "predict", nonfinite)
    result = module.run_panel_sequence_models(_dataset(), spec=_fast_spec())

    assert result.selection_evidence["route_status"] == (
        "blocked_no_valid_sequence_candidate"
    )
    assert result.fitted_policy is None
    assert (result.failure_rows["reason_code"] == "fit_failed").all()
    assert result.positions_by_segment["test"].eq(0.0).all()


def test_fitted_policy_pickle_round_trip_has_inference_parity():
    module = importlib.import_module("src.model.panel_sequence")
    dataset = _dataset()
    result = module.run_panel_sequence_models(dataset, spec=_fast_spec())
    restored = pickle.loads(pickle.dumps(result.fitted_policy))
    test_features = dataset.prepare(
        "test", col_set="feature", data_key=DataHandlerLP.DK_I
    )

    actual = restored.predict(test_features, segment="test")

    pd.testing.assert_frame_equal(actual, result.predictions_by_segment["test"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"selection_segment": "test"}, "validation segment"),
        ({"windows": (21, 60)}, "20 and 60"),
        ({"hidden_width": 64}, "at most 32"),
        ({"temporal_layers": 3}, "at most two"),
        ({"dropout": 0.05}, "at least 0.10"),
        ({"label_horizon": 10}, "fixed at five"),
    ],
)
def test_sequence_spec_enforces_frozen_compact_contract(kwargs, message):
    module = importlib.import_module("src.model.panel_sequence")

    with pytest.raises(ValueError, match=message):
        module.PanelSequenceSpec(**kwargs)
