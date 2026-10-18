from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from src.data.finra_allocation_features import (
    FINRA_BINARY_FEATURES, FINRA_FEATURES, FINRA_NUMERIC_FEATURES, build_finra_allocation_features,
)
from src.model.finra_allocator import FinraAllocator, initialization_contract
from src.model.minute_pit_allocator import MinutePitAllocator
from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationSpec


def source_dataset():
    rows = []
    for date, values in [
        ("2018-08-01", ((.6, 100., 1.), (.4, 200., 1.))),
        ("2018-08-02", ((.8, 300., 1.), (np.nan, np.nan, 0.))),
        ("2018-08-03", ((.3, 400., 1.), (.8, 500., 1.))),
        ("2026-02-23", ((.7, 600.5, 1.), (.2, 700.25, 1.))),
    ]:
        for symbol, value in zip(("SOXL", "SOXX"), values):
            rows.append((pd.Timestamp(date), symbol, *value))
    frame = pd.DataFrame(rows, columns=["datetime", "instrument", "short_ratio", "total_volume", "source_ready"])
    frame = frame.set_index(["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(pd.concat({"feature": frame}, axis=1)),
                    segments={"all": ("2018-08-01", "2026-02-23")})


def models():
    args = dict(base_features=[f"original_{i}" for i in range(39)],
                spec=asdict(PitAuxiliaryAllocationSpec(root=Path("."))), seed=71,
                cutoff="2023-01-01T00:00:00-05:00", train_end="2022-12-30", minimum_observations=3)
    return (MinutePitAllocator(**args, extra_features=[], feature_masks={}),
            FinraAllocator(**args, mode="neutral"), FinraAllocator(**args, mode="observed"))


def test_native_source_same_date_no_fill_full_calendar_and_unexecutable_tail():
    source = source_dataset().prepare("all", col_set="feature")
    dates = pd.DatetimeIndex(["2011-01-03", "2018-08-01", "2018-08-02", "2018-08-03", "2026-02-23"], name="datetime")
    clocks = pd.Series(pd.to_datetime(["2011-01-04 09:29", "2018-08-02 09:29", "2018-08-03 09:29",
                                      "2018-08-06 09:29", None]).tz_localize("America/New_York"), index=dates)
    features, timing = build_finra_allocation_features(source, signal_index=dates, decision_at=clocks)
    assert features.index.equals(dates) and timing.index.equals(dates)
    assert features.finra_both_ready.tolist() == [0., 1., 0., 1., 1.]
    assert features.loc[[dates[0], dates[2]], list(FINRA_NUMERIC_FEATURES)].isna().all().all()
    np.testing.assert_allclose(features.finra_short_mix.iloc[[1, 3, 4]], [.2, -.5, .5])
    assert timing.source_date.tolist() == dates.tolist()
    assert timing.assumed_available_at.dt.hour.eq(18).all()
    assert pd.isna(timing.decision_at.iloc[-1]) and not timing.decision_executable.iloc[-1]
    assert timing.decision_executable.iloc[:-1].all()

    changed = source.copy()
    changed.loc[(pd.Timestamp("2018-08-03"), "SOXL"), ["short_ratio", "total_volume"]] = [.99, 99999.]
    later, _ = build_finra_allocation_features(changed, signal_index=dates, decision_at=clocks)
    pd.testing.assert_frame_equal(features.iloc[:3], later.iloc[:3])
    bad_clock = clocks.copy()
    bad_clock.iloc[1] = pd.Timestamp("2018-08-01 09:29", tz="America/New_York")
    with pytest.raises(ValueError, match="strictly before"):
        build_finra_allocation_features(source, signal_index=dates, decision_at=bad_clock)
    with pytest.raises(ValueError, match="Decimal-derived float64"):
        build_finra_allocation_features(source.astype(np.float32), signal_index=dates, decision_at=clocks)


def test_scaler_uses_only_ready_supplied_train_and_neutral_differs_only_in_q():
    _, neutral, observed = models()
    dates = pd.bdate_range("2022-12-19", periods=5)
    # The fourth row is unknown even if a bad upstream numeric payload is
    # finite. The fifth is prediction-only and never enters the supplied train.
    features = pd.DataFrame({
        "finra_both_ready": [1., 1., 1., 0., 1.], "chicago_regime": 1., "fractional_regime": 0.,
        "log1p_SOXL_CNMS_TotalVolume": [2., 4., 6., 1e6, 1e9],
        "log1p_SOXX_CNMS_TotalVolume": [1., 2., 3., 1e6, -1e9],
        "finra_short_mix": [-.2, 0., .2, 1e6, .8],
    }, index=dates)
    for model in (neutral, observed):
        model._fit_extra_scaler(features.iloc[:4])
    assert neutral.extra_scaler == observed.extra_scaler
    assert observed.extra_scaler[FINRA_NUMERIC_FEATURES[0]] == {
        "mean": 4., "scale": 2., "observations": 3, "active": True,
    }
    assert observed.extra_scaler["finra_short_mix"]["scale"] == pytest.approx(.2)
    b, c = neutral._extra_tensor(features), observed._extra_tensor(features)
    assert torch.equal(b[:, :-1], c[:, :-1]) and b[:, -1].eq(0).all()
    np.testing.assert_allclose(c[:, -1], [-1., 0., 1., 0., 4.])
    np.testing.assert_allclose(c[3, 3:], [0., 0., 0.])
    np.testing.assert_allclose(c[4, 3:5], [8., -8.])
    np.testing.assert_array_equal(c[:, :3], features.loc[:, list(FINRA_BINARY_FEATURES)].to_numpy())
    assert features.iloc[3].finra_short_mix == 1e6  # No canonical-value mutation.
    insufficient = FinraAllocator(mode="observed", base_features=observed.base_features,
        spec=asdict(observed.spec), seed=71, cutoff=str(observed.cutoff),
        train_end=str(observed.train_end.date()), minimum_observations=4)
    insufficient._fit_extra_scaler(features.iloc[:4])
    assert insufficient._extra_tensor(features)[:, 3:].eq(0).all()
    bad = features.copy()
    bad.loc[dates[0], "finra_both_ready"] = .5
    with pytest.raises(ValueError, match="explicit binary"):
        observed._extra_tensor(bad)


def test_added_columns_keep_original_tensors_and_exact_paired_initial_outputs():
    original, neutral, observed = models()
    audit = initialization_contract(original, neutral, observed)
    assert audit["paired_initial_outputs_identical"]
    assert audit["original_vs_extended_output_max_absolute_error"] < 1e-6
    assert audit["paired_parameters"] - audit["original_parameters"] == len(FINRA_FEATURES) * original.spec.hidden_units
    # The adapter inherits the original shared auxiliary route and objective;
    # it does not create domain-specific heads or alter their tensors.
    assert FinraAllocator.auxiliary_logits is MinutePitAllocator.auxiliary_logits
    assert FinraAllocator.fit is MinutePitAllocator.fit
