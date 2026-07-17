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
    missing_source: str | None = None,
    boundary_nan_labels: bool = False,
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
    if boundary_nan_labels:
        for values in (future_return, future_downside, future_crash):
            values[train_end - 5 : train_end + 1] = np.nan
            values[valid_end - 5 : valid_end + 1] = np.nan

    feature_values = {
        "soxl_return": soxl_return,
        "soxx_return": soxx_return,
        "vix_level": vix_level,
        "vix_change": vix_change,
        "soxl_trend": soxl_trend,
        "soxl_volatility": soxl_volatility,
    }
    if missing_source is not None:
        feature_values = {
            name: values
            for name, values in feature_values.items()
            if not name.startswith(f"{missing_source.lower()}_")
        }
    columns = pd.MultiIndex.from_tuples(
        [
            *[("feature", name) for name in feature_values],
            ("label", "future_return_5d"),
            ("label", "future_downside_5d"),
            ("label", "future_crash_5d"),
        ]
    )
    frame = pd.DataFrame(
        np.column_stack(
            [
                *feature_values.values(),
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
    assert result.selection_evidence["event_trace"][-2:] == (
        "candidate_selection_frozen",
        "test_dk_i_accessed",
    )


def test_test_prepare_requires_an_observable_frozen_decision(monkeypatch):
    module = importlib.import_module("src.model.panel_sequence")
    original = module._prepare_inference_segment
    observed = []

    def require_frozen(dataset, segment, *, frozen_decision, event_trace):
        assert isinstance(frozen_decision, module._FrozenRouteDecision)
        assert frozen_decision.frozen is True
        assert event_trace[-1] == "candidate_selection_frozen"
        observed.append((frozen_decision.selected_candidate, frozen_decision.policy is not None))
        return original(
            dataset,
            segment,
            frozen_decision=frozen_decision,
            event_trace=event_trace,
        )

    monkeypatch.setattr(module, "_prepare_inference_segment", require_frozen)
    success = module.run_panel_sequence_models(_dataset(), spec=_fast_spec())
    blocked = module.run_panel_sequence_models(
        _dataset(missing_source="VIX"), spec=_fast_spec()
    )

    assert observed[0][1] is True
    assert observed[1] == ("blocked_missing_required_source", False)
    assert success.selection_evidence["test_accessed_after_freeze"] is True
    assert blocked.selection_evidence["test_accessed_after_freeze"] is True


def test_missing_pit_panel_is_separate_deterministic_block_and_aligned_pit_is_eligible():
    module = importlib.import_module("src.model.panel_sequence")
    dataset = _dataset()
    blocked = module.run_panel_sequence_models(dataset, spec=_fast_spec())
    assert blocked.mode_evidence["diagnostic_multivariate"]["status"] == "ok"
    assert blocked.mode_evidence["constituent_cross_sectional"]["status"] == (
        "blocked_missing_pit_panel"
    )
    assert blocked.blocked_mode_positions["constituent_cross_sectional"].eq(0.0).all()

    dates = pd.bdate_range("2023-09-01", periods=3)
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
    pit.attrs.update(
        {
            "point_in_time_membership": True,
            "source": "synthetic_pit_fixture",
            "as_of": "2023-12-04",
            "effective_date_field": "datetime",
            "provider_uri": "memory://qlib-fixture",
            "manifest_hash": "fixture-manifest-sha256",
        }
    )
    eligible = module.run_panel_sequence_models(
        dataset,
        spec=_fast_spec(),
        pit_panel=pit,
        qlib_lineage={
            "provider_uri": "memory://qlib-fixture",
            "manifest_hash": "fixture-manifest-sha256",
        },
    )
    assert eligible.mode_evidence["constituent_cross_sectional"]["status"] == "eligible"
    assert eligible.mode_evidence["constituent_cross_sectional"]["provenance"] == {
        "point_in_time_membership": True,
        "source": "synthetic_pit_fixture",
        "as_of": "2023-12-04",
        "effective_date_field": "datetime",
        "provider_uri": "memory://qlib-fixture",
        "manifest_hash": "fixture-manifest-sha256",
    }
    assert eligible.model_comparison.set_index("candidate").loc[
        "pit_cross_sectional_panel", "status"
    ] == "eligible"


def test_structurally_aligned_pit_without_explicit_provenance_is_blocked():
    module = importlib.import_module("src.model.panel_sequence")
    dates = pd.bdate_range("2023-09-01", periods=3)
    index = pd.MultiIndex.from_product(
        [dates, ["NVDA", "AMD"]], names=["datetime", "instrument"]
    )
    pit = pd.DataFrame(
        {"pit_member": True, "feature_1": np.arange(len(index), dtype=float)},
        index=index,
    )

    result = module.run_panel_sequence_models(
        _dataset(), spec=_fast_spec(), pit_panel=pit
    )

    evidence = result.mode_evidence["constituent_cross_sectional"]
    assert evidence["status"] == "blocked_missing_pit_panel"
    assert "missing_point_in_time_provenance" in evidence["validation_errors"]
    assert evidence["provenance"]["point_in_time_membership"] is False
    assert result.model_comparison.set_index("candidate").loc[
        "pit_cross_sectional_panel", "reason_code"
    ] == "blocked_missing_pit_panel"


@pytest.mark.parametrize(
    "missing_field",
    ("source", "as_of", "effective_date_field", "provider_uri", "manifest_hash"),
)
def test_pit_provenance_metadata_and_qlib_lineage_must_be_verifiable(missing_field):
    module = importlib.import_module("src.model.panel_sequence")
    dates = pd.bdate_range("2023-09-01", periods=3)
    index = pd.MultiIndex.from_product(
        [dates, ["NVDA", "AMD"]], names=["datetime", "instrument"]
    )
    pit = pd.DataFrame(
        {"pit_member": True, "feature_1": np.arange(len(index), dtype=float)},
        index=index,
    )
    metadata = {
        "point_in_time_membership": True,
        "source": "fixture",
        "as_of": "2023-12-04",
        "effective_date_field": "datetime",
        "provider_uri": "memory://qlib-fixture",
        "manifest_hash": "fixture-manifest-sha256",
    }
    metadata[missing_field] = ""
    pit.attrs.update(metadata)

    result = module.run_panel_sequence_models(
        _dataset(),
        spec=_fast_spec(),
        pit_panel=pit,
        qlib_lineage={
            "provider_uri": "memory://qlib-fixture",
            "manifest_hash": "fixture-manifest-sha256",
        },
    )

    evidence = result.mode_evidence["constituent_cross_sectional"]
    assert evidence["status"] == "blocked_missing_pit_panel"
    assert f"missing_provenance_{missing_field}" in evidence["validation_errors"]


@pytest.mark.parametrize(
    ("defect", "expected_error"),
    (
        ("lineage_mismatch", "provenance_manifest_hash_mismatch"),
        ("early_as_of", "provenance_as_of_precedes_panel"),
        ("missing_effective_mapping", "effective_date_field_not_found"),
    ),
)
def test_pit_provenance_values_must_match_lineage_and_effective_dates(
    defect, expected_error
):
    module = importlib.import_module("src.model.panel_sequence")
    dates = pd.bdate_range("2023-09-01", periods=3)
    index = pd.MultiIndex.from_product(
        [dates, ["NVDA", "AMD"]], names=["datetime", "instrument"]
    )
    pit = pd.DataFrame(
        {"pit_member": True, "feature_1": np.arange(len(index), dtype=float)},
        index=index,
    )
    pit.attrs.update(
        {
            "point_in_time_membership": True,
            "source": "fixture",
            "as_of": "2023-12-04",
            "effective_date_field": "datetime",
            "provider_uri": "memory://qlib-fixture",
            "manifest_hash": "fixture-manifest-sha256",
        }
    )
    if defect == "lineage_mismatch":
        pit.attrs["manifest_hash"] = "wrong-hash"
    elif defect == "early_as_of":
        pit.attrs["as_of"] = "2023-08-01"
    else:
        pit.attrs["effective_date_field"] = "effective_date"
    result = module.run_panel_sequence_models(
        _dataset(),
        spec=_fast_spec(),
        pit_panel=pit,
        qlib_lineage={
            "provider_uri": "memory://qlib-fixture",
            "manifest_hash": "fixture-manifest-sha256",
        },
    )

    evidence = result.mode_evidence["constituent_cross_sectional"]
    assert evidence["status"] == "blocked_missing_pit_panel"
    assert expected_error in evidence["validation_errors"]


@pytest.mark.parametrize(
    "defect",
    ("no_overlap", "duplicate", "nonfinite", "no_active", "no_feature"),
)
def test_pit_panel_requires_real_unique_finite_active_qlib_alignment(defect):
    module = importlib.import_module("src.model.panel_sequence")
    dataset = _dataset()
    dates = (
        pd.bdate_range("2025-01-02", periods=3)
        if defect == "no_overlap"
        else pd.bdate_range("2023-09-01", periods=3)
    )
    index = pd.MultiIndex.from_product(
        [dates, ["NVDA", "AMD"]], names=["datetime", "instrument"]
    )
    pit = pd.DataFrame(
        {"pit_member": defect != "no_active", "feature_1": np.arange(len(index), dtype=float)},
        index=index,
    )
    if defect == "duplicate":
        pit = pd.concat([pit, pit.iloc[[0]]])
    elif defect == "nonfinite":
        pit.iloc[0, pit.columns.get_loc("feature_1")] = np.nan
    elif defect == "no_feature":
        pit = pit[["pit_member"]]
    result = module.run_panel_sequence_models(dataset, spec=_fast_spec(), pit_panel=pit)

    evidence = result.mode_evidence["constituent_cross_sectional"]
    row = result.model_comparison.set_index("candidate").loc["pit_cross_sectional_panel"]
    assert evidence["status"] == "blocked_missing_pit_panel"
    assert evidence["validation_status"].startswith("invalid_pit_panel")
    assert evidence["validation_errors"]
    assert row["status"] == "blocked"
    assert row["reason_code"] == "blocked_missing_pit_panel"
    assert result.blocked_mode_positions["constituent_cross_sectional"].eq(0.0).all()


@pytest.mark.parametrize("missing_source", ("SOXL", "SOXX", "VIX"))
def test_missing_diagnostic_source_blocks_route_with_zero_finite_positions(missing_source):
    module = importlib.import_module("src.model.panel_sequence")

    result = module.run_panel_sequence_models(
        _dataset(missing_source=missing_source), spec=_fast_spec()
    )

    evidence = result.selection_evidence
    assert evidence["route_status"] == "blocked_missing_required_source"
    assert evidence["source_evidence"][missing_source]["available"] is False
    assert evidence["source_evidence"][missing_source]["matched_columns"] == ()
    assert result.fitted_policy is None
    assert result.selected_candidate == "blocked_missing_required_source"
    for positions in result.positions_by_segment.values():
        assert np.isfinite(positions.to_numpy()).all()
        assert positions.eq(0.0).all()


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


def test_boundary_purge_label_nans_are_excluded_but_internal_nans_still_block():
    module = importlib.import_module("src.model.panel_sequence")

    boundary = module.run_panel_sequence_models(
        _dataset(boundary_nan_labels=True), spec=_fast_spec()
    )
    internal = module.run_panel_sequence_models(
        _dataset(nonfinite_validation_downside=True), spec=_fast_spec()
    )

    assert boundary.selection_evidence["route_status"] == "ok"
    for segment in ("train", "valid"):
        evidence = boundary.selection_evidence["label_completeness_evidence"][segment]
        assert evidence["boundary_excluded_rows"] == 6
        assert evidence["boundary_invalid_rows"] == 6
        assert evidence["internal_invalid_rows"] == 0
        assert evidence["labels_complete"] is True
    assert internal.selection_evidence["route_status"] == "blocked_data_quality"
    assert internal.selection_evidence["label_completeness_evidence"]["valid"][
        "internal_invalid_rows"
    ] == 1


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


def test_segment_datetime_order_overlap_and_public_policy_inference_are_guarded():
    module = importlib.import_module("src.model.panel_sequence")
    native = _dataset()

    class MutatingDataset:
        def __init__(self, mutation):
            self.mutation = mutation

        def prepare(self, segment, *, col_set, data_key):
            frame = native.prepare(segment, col_set=col_set, data_key=data_key)
            return self.mutation(segment, frame)

    with pytest.raises(ValueError, match="strictly increasing"):
        module.run_panel_sequence_models(
            MutatingDataset(
                lambda segment, frame: frame.iloc[::-1] if segment == "train" else frame
            ),
            spec=_fast_spec(),
        )

    with pytest.raises(ValueError, match="unique"):
        module.run_panel_sequence_models(
            MutatingDataset(
                lambda segment, frame: pd.concat([frame.iloc[:1], frame])
                if segment == "train"
                else frame
            ),
            spec=_fast_spec(),
        )

    train_tail = native.prepare(
        "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    ).tail(1)
    with pytest.raises(ValueError, match="must not overlap"):
        module.run_panel_sequence_models(
            MutatingDataset(
                lambda segment, frame: pd.concat([train_tail, frame])
                if segment == "valid"
                else frame
            ),
            spec=_fast_spec(),
        )

    result = module.run_panel_sequence_models(native, spec=_fast_spec())
    test_features = native.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    with pytest.raises(ValueError, match="strictly increasing"):
        result.fitted_policy.predict(test_features.iloc[::-1])
    duplicate = pd.concat([test_features.iloc[:1], test_features])
    with pytest.raises(ValueError, match="unique"):
        result.fitted_policy.predict(duplicate)


def test_test_segment_must_follow_validation_without_overlap():
    module = importlib.import_module("src.model.panel_sequence")
    native = _dataset()
    valid_features = native.prepare(
        "valid", col_set="feature", data_key=DataHandlerLP.DK_I
    )

    class OverlappingTestDataset:
        def prepare(self, segment, *, col_set, data_key):
            if segment == "test":
                return valid_features.tail(3)
            return native.prepare(segment, col_set=col_set, data_key=data_key)

    with pytest.raises(ValueError, match="must not overlap"):
        module.run_panel_sequence_models(OverlappingTestDataset(), spec=_fast_spec())


def test_sequence_training_restores_numpy_and_torch_process_state():
    torch = pytest.importorskip("torch")
    module = importlib.import_module("src.model.panel_sequence")
    np.random.seed(9182)
    torch.manual_seed(8173)
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    threads_before = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()

    module.run_panel_sequence_models(_dataset(), spec=_fast_spec())

    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert torch.get_num_threads() == threads_before
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only_before


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
