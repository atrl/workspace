from dataclasses import asdict
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.integrated_allocation import PRIMARY_FEATURES, PRIMARY_LABELS
from src.model.benchmark_risk_allocation import AllocationSpec, BenchmarkRiskAllocationPolicy
from src.workflow import benchmark_risk_excess as workflow


def _prices():
    dates = pd.bdate_range("2010-01-04", periods=350)
    rng = np.random.default_rng(29)
    common = rng.normal(0, .012, len(dates))
    returns = np.column_stack((3 * common + rng.normal(0, .003, len(dates)), common))
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates,
                        columns=pd.MultiIndex.from_tuples([("open", "SOXL"), ("open", "SOXX")]))


def test_covariance_uses_only_observed_prefix_and_matches_sample_covariance():
    prices = _prices()
    first = workflow.expanding_covariance(prices)
    changed = prices.copy()
    changed.iloc[170:, 0] *= 2.
    second = workflow.expanding_covariance(changed)
    np.testing.assert_array_equal(first.iloc[:170].to_numpy(), second.iloc[:170].to_numpy())
    sample = prices.pct_change(fill_method=None).iloc[1:170].cov().to_numpy()
    np.testing.assert_allclose(first.iloc[169].to_numpy(), [sample[0, 0], sample[0, 1], sample[1, 1]], rtol=1e-12)
    assert first.iloc[:2].isna().all().all()
    assert np.isfinite(first.iloc[2:].to_numpy()).all()
    assert first.attrs["source"]["source_price_sha256"] != second.attrs["source"]["source_price_sha256"]
    determinant = first.iloc[2:, 0] * first.iloc[2:, 2] - first.iloc[2:, 1] ** 2
    assert determinant.min() >= -1e-20


def test_covariance_attachment_is_by_date_and_named_fields_not_column_position():
    covariance = workflow.expanding_covariance(_prices()).iloc[5:10]
    frame = pd.DataFrame({"value": range(5)}, index=covariance.index)
    correct = workflow.attach_covariance(frame, covariance)
    reordered = workflow.attach_covariance(frame, covariance.loc[:, list(reversed(workflow.COVARIANCE_FIELDS))])
    pd.testing.assert_frame_equal(correct, reordered)
    with pytest.raises(ValueError, match="lacks observed covariance"):
        workflow.attach_covariance(frame, covariance.iloc[1:])


def _audit_inputs():
    dates = pd.bdate_range("2020-01-02", periods=3)
    frame = pd.DataFrame({"signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
                          "exit_date": dates + pd.offsets.BDay(2),
                          "cov_soxl_soxl": .0009, "cov_soxl_soxx": .0003, "cov_soxx_soxx": .0001})
    replay = frame.loc[:, ["signal_date", "trade_date", "exit_date"]].copy()
    replay["soxl_weight"], replay["soxx_weight"], replay["cash_weight"] = .2, .3, .5
    return replay, frame


def test_budget_audit_rejects_overexposure_and_reports_constant_control_correctly():
    replay, frame = _audit_inputs()
    audit = workflow.covariance_audit(replay, frame)
    assert audit["maximum_predicted_variance_ratio"] == pytest.approx(.81)
    replay["soxl_weight"], replay["soxx_weight"], replay["cash_weight"] = .8, .1, .1
    with pytest.raises(ValueError, match="variance exceeds"):
        workflow.covariance_audit(replay, frame)


def test_budget_audit_cannot_silently_pair_different_dates_or_non_psd_covariance():
    replay, frame = _audit_inputs()
    with pytest.raises(ValueError, match="dates|coverage|ordered|aligned"):
        workflow.covariance_audit(replay, frame.iloc[::-1])
    frame["cov_soxl_soxx"] = -.01
    with pytest.raises(ValueError, match="PSD|semidefinite|covariance"):
        workflow.covariance_audit(replay, frame)


def test_constant_control_ignores_market_and_holdings_but_still_obeys_shared_covariance():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = BenchmarkRiskAllocationPolicy(4, common_feature_indices=(0, 1, 2, 3), market_conditioning=False).double()
    features = torch.tensor([[1., 2., 3., 4., .0009, .0003, .0001],
                             [-100., 0., 100., 90., .0009, .0003, .0001]], dtype=torch.float64)
    holdings = torch.tensor([[0., 0., 1.], [.7, .2, .1]], dtype=torch.float64)
    with torch.no_grad():
        targets = model(features, holdings)
        raw = model._raw_target(features, holdings)
        changed = features.clone()
        changed[1, -3:] = torch.tensor([.0036, .0006, .0001], dtype=torch.float64)
        mapped = model(changed, holdings)
    torch.testing.assert_close(targets[0], targets[1], rtol=0, atol=0)
    torch.testing.assert_close(raw[0], raw[1], rtol=0, atol=0)
    assert not torch.allclose(mapped[0], mapped[1])
    assert not model.market_conditioning


def test_annual_risk_fold_excludes_unmatured_outcomes():
    dates = pd.bdate_range("2016-01-04", "2017-12-29")
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(len(dates), len(PRIMARY_FEATURES))), index=dates, columns=PRIMARY_FEATURES)
    frame.index.name = "datetime"
    for name in PRIMARY_LABELS:
        frame[name] = .001
    frame["signal_date"], frame["trade_date"], frame["exit_date"] = dates, dates + pd.offsets.BDay(1), dates + pd.offsets.BDay(2)
    for name, offset, hour in [("feature_available_at", 0, 16), ("decision_at", 1, 10),
                              ("return_available_at", 2, 10), ("forecast_available_at", 6, 10),
                              ("auxiliary_available_at", 5, 16)]:
        frame[name] = (dates + pd.offsets.BDay(offset)).tz_localize("America/New_York") + pd.Timedelta(hours=hour)
    frame.loc[:, list(workflow.COVARIANCE_FIELDS)] = np.array([.0009, .0003, .0001])
    variant = {"name": "base_budget", "pit_auxiliary": False}
    first = workflow.prepare_fold(frame, None, year=2017, variant=variant, as_of="2017-12-29")
    cutoff = pd.Timestamp("2017-01-01", tz="America/New_York")
    frame.loc[frame.return_available_at.ge(cutoff), ["soxl_return", "soxx_return"]] = .5
    second = workflow.prepare_fold(frame, None, year=2017, variant=variant, as_of="2017-12-29")
    np.testing.assert_array_equal(first["returns"], second["returns"])
    assert pd.Timestamp(first["audit"]["primary"]["label_available_at_max"]) < cutoff
    assert first["features"].shape[1] == len(PRIMARY_FEATURES) + 3


def test_cached_fit_cannot_reuse_changed_auxiliary_values(tmp_path, monkeypatch):
    from src.model import benchmark_risk_allocation as model_module

    calls = []
    spec = AllocationSpec(auxiliary_coefficient=.05)
    variant = {"name": "pit_budget", "market_conditioning": True}
    fold = {"features": np.zeros((3, 6)), "returns": np.zeros((3, 2)),
            "auxiliary": {"features": np.ones((2, 3)), "targets": np.ones((2, 2)), "sample_weight": np.ones(2)},
            "audit": {"primary": {"label_available_at_max": "2021-12-30T14:30:00Z"}}}

    def fitted(features, returns, **kwargs):
        calls.append(kwargs["auxiliary"]["targets"].copy())
        digest = lambda array: hashlib.sha256(np.asarray(array, dtype=np.float64).tobytes()).hexdigest()
        metadata = {"spec": asdict(spec), "seed": 17, "market_conditioning": True,
                    "training_rows": len(features), "auxiliary_rows": 2, "history": [{"loss": 0.}],
                    "training_feature_sha256": digest(features), "training_return_sha256": digest(returns),
                    "training_benchmark_sha256": digest(returns[:, 1]),
                    "auxiliary_feature_sha256": digest(kwargs["auxiliary"]["features"]),
                    "auxiliary_target_sha256": digest(kwargs["auxiliary"]["targets"]),
                    "auxiliary_weight_sha256": digest(kwargs["auxiliary"]["sample_weight"])}
        return SimpleNamespace(training_metadata=metadata), metadata

    monkeypatch.setattr(model_module, "fit_policy", fitted)
    payload = (variant, 17, {2022: fold}, spec, str(tmp_path))
    workflow._fit_job(payload)
    workflow._fit_job(payload)
    assert len(calls) == 1
    fold["auxiliary"]["targets"][0, 0] = 9.
    try:
        workflow._fit_job(payload)
    except ValueError as error:
        assert any(word in str(error) for word in ("auxiliary", "PIT", "fold", "cache"))
    else:
        assert len(calls) == 2, "Changed auxiliary labels silently reused a cached fit"
