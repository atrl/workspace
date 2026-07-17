from __future__ import annotations

import builtins
import importlib

import numpy as np
import pandas as pd
import pytest
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


def _dataset(*, test_label_shift: float = 0.0, one_class_crash: bool = False) -> DatasetH:
    random = np.random.default_rng(20260717)
    dates = pd.bdate_range("2025-01-02", periods=120)
    index = pd.MultiIndex.from_arrays(
        [["SOXL"] * len(dates), dates],
        names=["instrument", "datetime"],
    )
    market = random.normal(0.0, 1.0, len(dates))
    stress = np.sin(np.linspace(0.0, 8.0 * np.pi, len(dates)))
    residual = random.normal(0.0, 0.6, len(dates))
    future_return = 0.025 * market - 0.018 * stress + 0.006 * residual
    future_downside = np.maximum(-future_return, 0.0)
    crash_cutoff = float(np.quantile(future_return[:72], 0.20))
    future_crash = (future_return <= crash_cutoff).astype("float64")
    if one_class_crash:
        future_crash[:] = 0.0
    future_return = future_return.copy()
    future_downside = future_downside.copy()
    future_crash = future_crash.copy()
    future_return[96:] += test_label_shift
    future_downside[96:] = np.maximum(-future_return[96:], 0.0)
    future_crash[96:] = (future_return[96:] <= crash_cutoff).astype("float64")

    frame = pd.DataFrame(
        np.column_stack(
            [
                market,
                stress,
                residual,
                market * stress,
                future_return,
                future_downside,
                future_crash,
            ]
        ),
        index=index,
        columns=pd.MultiIndex.from_tuples(
            [
                ("feature", "market"),
                ("feature", "stress"),
                ("feature", "residual"),
                ("feature", "interaction"),
                ("label", "future_return_5d"),
                ("label", "future_downside_5d"),
                ("label", "future_crash_5d"),
            ]
        ),
    )
    handler = DataHandlerLP.from_df(frame)
    return DatasetH(
        handler=handler,
        segments={
            "train": (dates[0], dates[71]),
            "valid": (dates[72], dates[95]),
            "test": (dates[96], dates[-1]),
        },
    )


def _fast_config():
    module = importlib.import_module("src.model.tail_budget")
    return module.TailBudgetSpec(
        random_state=17,
        ridge_alphas=(0.5,),
        lightgbm_rounds=12,
        catboost_iterations=16,
        densemble_epochs=8,
        blend_weights=(0.25, 0.50, 0.75),
        floor_cap_candidates=((0.15, 0.60), (0.15, 0.75)),
    )


def test_tail_budget_module_exposes_focused_workflow_api():
    module = importlib.import_module("src.model.tail_budget")

    assert module.TailBudgetSpec
    assert module.TailBudgetResult
    assert module.run_tail_budget_models


def test_tail_budget_fits_qlib_components_and_emits_finite_bounded_positions():
    module = importlib.import_module("src.model.tail_budget")
    dataset = _dataset()
    test_index = dataset.prepare("test", col_set="feature").index
    quality = pd.Series(True, index=test_index)
    quality.iloc[3] = False

    result = module.run_tail_budget_models(
        dataset,
        spec=_fast_config(),
        data_quality_by_segment={"test": quality},
    )

    comparison = result.model_comparison.set_index("candidate")
    assert {
        "ridge_expected",
        "lightgbm_expected",
        "catboost_downside",
        "qlib_densemble_downside",
        "catboost_crash",
        "validation_blend_downside",
    }.issubset(comparison.index)
    assert comparison.loc["ridge_expected", "status"] == "ok"
    assert comparison.loc["lightgbm_expected", "status"] == "ok"
    assert comparison.loc["catboost_downside", "status"] == "ok"
    assert comparison.loc["qlib_densemble_downside", "status"] == "ok"
    assert comparison.loc["catboost_crash", "status"] == "ok"
    assert result.selected_candidate in set(comparison.index)
    assert result.selection_evidence["selection_segment"] == "valid"
    assert result.selection_evidence["test_labels_used_for_selection"] is False
    assert result.selection_evidence["test_accessed_after_freeze"] is True
    assert result.selection_evidence["densemble_prediction_api"] == "DEnsembleModel.predict(dataset, segment)"
    assert set(result.predictions_by_segment) == {"train", "valid", "test"}
    assert set(result.positions_by_segment) == {"train", "valid", "test"}

    for segment in ("train", "valid", "test"):
        predictions = result.predictions_by_segment[segment]
        positions = result.positions_by_segment[segment]
        assert predictions.index.equals(dataset.prepare(segment, col_set="feature").index)
        assert positions.index.equals(predictions.index)
        assert np.isfinite(predictions.to_numpy(dtype="float64")).all()
        assert np.isfinite(positions.to_numpy(dtype="float64")).all()
        assert ((positions == 0.0) | positions.between(result.floor, result.cap)).all()
    assert result.positions_by_segment["test"].iloc[3] == 0.0


def test_tail_budget_selection_and_frozen_policy_do_not_use_test_labels():
    module = importlib.import_module("src.model.tail_budget")
    baseline = module.run_tail_budget_models(_dataset(), spec=_fast_config())
    perturbed = module.run_tail_budget_models(
        _dataset(test_label_shift=-2.0),
        spec=_fast_config(),
    )

    assert perturbed.selected_candidate == baseline.selected_candidate
    assert perturbed.floor == baseline.floor
    assert perturbed.cap == baseline.cap
    assert perturbed.selection_evidence["frozen_policy"] == baseline.selection_evidence["frozen_policy"]
    np.testing.assert_allclose(
        perturbed.positions_by_segment["test"],
        baseline.positions_by_segment["test"],
        rtol=0.0,
        atol=1e-12,
    )


def test_failed_optional_candidate_is_retained_without_aborting_route():
    module = importlib.import_module("src.model.tail_budget")

    result = module.run_tail_budget_models(
        _dataset(one_class_crash=True),
        spec=_fast_config(),
    )

    failures = result.failure_rows.set_index("candidate")
    assert "catboost_crash" in failures.index
    assert failures.loc["catboost_crash", "status"] == "failed"
    assert failures.loc["catboost_crash", "reason_code"] == "fit_failed"
    comparison = result.model_comparison.set_index("candidate")
    assert comparison.loc["validation_blend_downside", "status"] == "ok"
    assert comparison.loc["validation_selected_blend", "status"] == "ok"
    assert result.selection_evidence["utility_best_blend"]["crash_candidate"] == "train_crash_prior"
    assert result.selected_candidate
    assert np.isfinite(result.positions_by_segment["test"].to_numpy()).all()


def test_tail_budget_config_rejects_test_segment_as_selection_source():
    module = importlib.import_module("src.model.tail_budget")

    with pytest.raises(ValueError, match="validation segment"):
        module.TailBudgetSpec(selection_segment="test")


def test_tail_budget_accepts_flat_dataset_frames_and_delays_test_prepare():
    module = importlib.import_module("src.model.tail_budget")
    native = _dataset()

    class FlatDataset:
        def __init__(self):
            self.calls = []

        def prepare(self, segment, *, col_set, data_key):
            self.calls.append((segment, tuple(col_set), data_key))
            frame = native.prepare(segment, col_set=["feature", "label"], data_key=data_key).copy()
            frame.columns = frame.columns.get_level_values(1)
            return frame

    dataset = FlatDataset()
    result = module.run_tail_budget_models(dataset, spec=_fast_config())

    assert [call[0] for call in dataset.calls] == ["train", "valid", "test"]
    assert all(call[1] == ("feature", "label") for call in dataset.calls)
    assert result.selection_evidence["test_accessed_after_freeze"] is True


def test_blend_weights_are_frozen_from_end_to_end_validation_utility():
    module = importlib.import_module("src.model.tail_budget")

    result = module.run_tail_budget_models(_dataset(), spec=_fast_config())

    evidence = result.selection_evidence
    evaluations = evidence["blend_grid_evaluations"]
    utility_best = max(
        evaluations,
        key=lambda row: (
            row["validation_utility"],
            -row["expected_weight"],
            -row["downside_weight"],
            -row["floor"],
            -row["cap"],
        ),
    )
    assert evidence["blend_selection_metric"] == "validation_allocation_utility"
    assert evidence["utility_best_blend"] == utility_best
    comparison = result.model_comparison.set_index("candidate")
    assert comparison.loc["validation_selected_blend", "validation_score"] == pytest.approx(
        utility_best["validation_utility"]
    )
    assert comparison.loc["validation_blend_expected", "blend_weight"] == utility_best["expected_weight"]
    assert comparison.loc["validation_blend_downside", "blend_weight"] == utility_best["downside_weight"]
    assert "validation_blend_crash" not in comparison.index
    assert all("crash_weight" not in row for row in evaluations)
    assert evidence["test_labels_used_for_selection"] is False


def test_unavailable_base_models_emit_all_declared_blend_failure_rows(monkeypatch):
    module = importlib.import_module("src.model.tail_budget")
    real_import = builtins.__import__

    def without_catboost(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "catboost":
            raise ModuleNotFoundError("catboost intentionally unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_catboost)
    result = module.run_tail_budget_models(_dataset(), spec=_fast_config())

    comparison = result.model_comparison.set_index("candidate")
    assert {
        "ridge_expected",
        "lightgbm_expected",
        "catboost_downside",
        "qlib_densemble_downside",
        "catboost_crash",
        "validation_blend_expected",
        "validation_blend_downside",
        "validation_selected_blend",
    }.issubset(comparison.index)
    assert comparison.loc["catboost_downside", "status"] == "failed"
    assert comparison.loc["catboost_crash", "status"] == "failed"
    assert comparison.loc["validation_blend_expected", "status"] == "ok"
    for candidate in ("validation_blend_downside", "validation_selected_blend"):
        assert comparison.loc[candidate, "status"] == "unavailable"
        assert comparison.loc[candidate, "reason_code"] == "base_candidate_unavailable"
    assert result.selected_candidate
    assert np.isfinite(result.positions_by_segment["test"].to_numpy()).all()


def test_non_finite_train_prediction_fails_only_that_candidate(monkeypatch):
    module = importlib.import_module("src.model.tail_budget")
    from lightgbm import LGBMRegressor

    original_predict = LGBMRegressor.predict

    def non_finite_predict(self, features, *args, **kwargs):
        values = np.asarray(original_predict(self, features, *args, **kwargs), dtype="float64")
        values[2] = np.nan
        return values

    monkeypatch.setattr(LGBMRegressor, "predict", non_finite_predict)
    result = module.run_tail_budget_models(_dataset(), spec=_fast_config())

    comparison = result.model_comparison.set_index("candidate")
    failed = comparison.loc["lightgbm_expected"]
    assert failed["status"] == "failed"
    assert failed["reason_code"] == "non_finite_prediction"
    assert failed["segment"] == "train"
    assert failed["first_offending_date"].startswith("2025-01-06")
    assert comparison.loc["ridge_expected", "status"] == "ok"
    assert comparison.loc["validation_blend_downside", "status"] == "ok"
    assert np.isfinite(result.positions_by_segment["test"].to_numpy()).all()


def test_non_finite_validation_prediction_fails_only_that_candidate(monkeypatch):
    module = importlib.import_module("src.model.tail_budget")
    from catboost import CatBoostRegressor

    original_predict = CatBoostRegressor.predict

    def non_finite_valid_predict(self, features, *args, **kwargs):
        values = np.asarray(original_predict(self, features, *args, **kwargs), dtype="float64")
        if len(features) == 24:
            values[4] = np.inf
        return values

    monkeypatch.setattr(CatBoostRegressor, "predict", non_finite_valid_predict)
    result = module.run_tail_budget_models(_dataset(), spec=_fast_config())

    comparison = result.model_comparison.set_index("candidate")
    failed = comparison.loc["catboost_downside"]
    assert failed["status"] == "failed"
    assert failed["reason_code"] == "non_finite_prediction"
    assert failed["segment"] == "valid"
    assert failed["first_offending_date"].startswith("2025-04-18")
    assert comparison.loc["qlib_densemble_downside", "status"] == "ok"
    assert np.isfinite(result.positions_by_segment["test"].to_numpy()).all()


def test_non_finite_frozen_test_prediction_returns_blocked_fail_closed_result(monkeypatch):
    module = importlib.import_module("src.model.tail_budget")
    original_predict = module._predict_model

    def non_finite_on_test(model, features, *, segment=None):
        values = np.asarray(original_predict(model, features), dtype="float64")
        if segment == "test":
            values[0] = np.nan
        return values

    monkeypatch.setattr(module, "_predict_model", non_finite_on_test)
    result = module.run_tail_budget_models(_dataset(), spec=_fast_config())

    evidence = result.selection_evidence
    assert evidence["route_status"] == "blocked_non_finite_prediction"
    assert evidence["test_prediction_failure"]["reason_code"] == "non_finite_prediction"
    assert evidence["test_prediction_failure"]["segment"] == "test"
    assert evidence["test_prediction_failure"]["first_offending_date"].startswith("2025-05-16")
    failed = result.model_comparison.set_index("candidate").loc["frozen_test_prediction"]
    assert failed["status"] == "failed"
    assert failed["reason_code"] == "non_finite_prediction"
    assert (result.positions_by_segment["test"] == 0.0).all()
    assert np.isfinite(result.predictions_by_segment["test"].to_numpy()).all()
