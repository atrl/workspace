from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from src.data.integrated_allocation import (
    AUXILIARY_TARGETS, COMMON_FEATURES, FORECAST_FEATURES, FORECAST_LABELS, PRIMARY_FEATURES,
    RETURN_LABELS, IntegratedAllocationDataSpec, _native_observations,
    load_integrated_data, prepare_mature_rows, primary_frame_from_qlib,
)
from src.data.qlib_native import build_qlib_store_from_frame


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("integrated_native")
    calendar = pd.bdate_range("2010-06-01", periods=330)
    number = np.arange(len(calendar))

    def market(symbol, scale):
        close = scale * np.exp(.001 * number + .03 * np.sin(number / 7) + .01 * np.cos(number / 11))
        opening = close * (1 + .003 * np.sin(number / 3))
        return pd.DataFrame({"instrument": symbol, "datetime": calendar, "open": opening,
            "high": np.maximum(opening, close) * 1.01, "low": np.minimum(opening, close) * .99,
            "close": close, "volume": 100000 + number * 100 + 5000 * np.sin(number / 5)})

    primary = pd.concat([market("SOXL", 20), market("SOXX", 100)], ignore_index=True)
    build_qlib_store_from_frame(primary, provider_uri=root / "data/qlib/soxl", as_of=str(calendar[-1].date()))
    build_qlib_store_from_frame(primary.drop(columns="volume"), provider_uri=root / "data/qlib/soxl_soxx_soxs", as_of=str(calendar[-1].date()))
    auxiliary = []
    for symbol in ("AAA", "BBB", "ACTUAL2X"):
        frame = market(symbol, 30)
        frame["adj_close"] = frame.close
        frame["pit_available"] = (number >= 180).astype(float)
        frame["pit_member"] = 1.
        if symbol == "BBB":
            frame.loc[number >= 260, "pit_member"] = 0.
        frame["pit_weight_pct"] = 10.
        frame["instrument_type"] = float(symbol == "ACTUAL2X")
        frame["leverage_multiple"] = 2. if symbol == "ACTUAL2X" else 1.
        frame["is_actual_2x"] = float(symbol == "ACTUAL2X")
        auxiliary.append(frame)
    build_qlib_store_from_frame(pd.concat(auxiliary, ignore_index=True),
        provider_uri=root / "data/qlib/soxx_pit_leveraged_training_v1", as_of=str(calendar[-1].date()))
    spec = IntegratedAllocationDataSpec(root=root, as_of=str(calendar[-1].date()), train_start=str(calendar[90].date()))
    return root, calendar, spec, load_integrated_data(spec)


def test_loader_reads_native_qlib_and_preserves_neutral_oof_slots(frozen):
    _, calendar, spec, bundle = frozen
    assert isinstance(bundle["primary_dataset"], DatasetH)
    assert isinstance(bundle["primary_dataset"].handler, DataHandlerLP)
    assert isinstance(bundle["auxiliary_dataset"], DatasetH)
    assert len(bundle["primary_features"]) == 28
    assert bundle["common_feature_indices"] == tuple(range(16))
    frame = bundle["primary_frame"]
    assert frame.index[0] == pd.Timestamp(spec.train_start) and frame.index[-1] == calendar[-1]
    assert frame.loc[:, list(FORECAST_FEATURES)].eq(0).all().all()
    assert frame.soxl_return.tail(2).isna().all() and frame.future_vol_5d.tail(6).isna().all()
    assert bundle["audit"]["pre_pit_union_backfill_used"] is False
    assert bundle["audit"]["primary_frame_sha256"]
    assert "qlib.data.D.features" in bundle["audit"]["source_api"]
    assert not set(PRIMARY_FEATURES).intersection({"soxl_return", "cash_advantage_1d", "future_vol_5d"})


def test_labels_have_exact_entry_exit_and_completion_times(frozen):
    _, calendar, _, bundle = frozen
    frame, prices = bundle["primary_frame"], bundle["execution_prices"]
    pos = 200
    day, entry, exit_day = calendar[pos:pos + 3]
    row = frame.loc[day]
    expected = prices.loc[exit_day, ("open", "SOXX")] / prices.loc[entry, ("open", "SOXX")] - 1
    assert row.soxx_return == pytest.approx(expected)
    assert row.cash_advantage_1d == pytest.approx(-np.log1p(expected))
    forward = [prices.loc[calendar[pos + step + 2], ("open", "SOXX")] /
               prices.loc[calendar[pos + step + 1], ("open", "SOXX")] - 1 for step in range(5)]
    assert row.future_vol_5d == pytest.approx(np.std(forward, ddof=1))
    assert row.return_available_at == pd.Timestamp(f"{exit_day.date()} 09:30:00", tz="America/New_York")
    assert row.forecast_available_at == pd.Timestamp(f"{calendar[pos + 6].date()} 09:30:00", tz="America/New_York")
    assert row.feature_available_at < row.decision_at < row.return_available_at


def test_future_prices_change_labels_but_not_earlier_features(frozen):
    root, calendar, spec, bundle = frozen
    raw, _, _ = _native_observations(root / spec.primary_provider, fields=("$open", "$high", "$low", "$close", "$volume"),
                                    instruments=("SOXL", "SOXX"), as_of=spec.as_of)
    execution, _, _ = _native_observations(root / spec.execution_provider, fields=("$open", "$close"),
                                          instruments=("SOXL", "SOXX"), as_of=spec.as_of)
    cutoff = calendar[240]
    raw_changed, execution_changed = raw.copy(), execution.copy()
    future = raw_changed.index.get_level_values("datetime") > cutoff
    raw_changed.loc[future, ["$open", "$high", "$low", "$close"]] *= 1.5
    execution_changed.loc[execution_changed.index.get_level_values("datetime") > cutoff] *= 1.5
    changed, _ = primary_frame_from_qlib(raw_changed, execution_changed, calendar, train_start=spec.train_start)
    pd.testing.assert_frame_equal(bundle["primary_frame"].loc[:cutoff, list(PRIMARY_FEATURES)],
                                  changed.loc[:cutoff, list(PRIMARY_FEATURES)], check_freq=False)
    assert changed.loc[calendar[239], "soxx_return"] != bundle["primary_frame"].loc[calendar[239], "soxx_return"]


def test_auxiliary_membership_is_lagged_and_2x_union_rows_never_enter(frozen):
    _, calendar, _, bundle = frozen
    auxiliary = bundle["auxiliary_frame"]
    assert set(auxiliary.index.get_level_values("instrument")) == {"AAA", "BBB"}
    assert auxiliary.index.get_level_values("datetime").min() == calendar[181]
    assert (calendar[260], "BBB") in auxiliary.index
    assert (calendar[261], "BBB") not in auxiliary.index
    assert auxiliary.training_scope.eq("lagged_published_pit_member").all()
    assert (auxiliary.membership_observed_at < auxiliary.feature_available_at).all()
    assert auxiliary.groupby(level="datetime").sample_weight.sum().eq(1.).all()
    assert list(auxiliary.columns[:16]) == list(COMMON_FEATURES)


def test_maturity_is_strict_and_does_not_use_shorter_target_availability(frozen):
    _, calendar, _, bundle = frozen
    cutoff = pd.Timestamp(f"{calendar[250].date()} 09:30:00", tz="America/New_York")
    primary, audit = prepare_mature_rows(bundle["primary_dataset"], decision_at=cutoff,
        label_columns=RETURN_LABELS, availability_column="return_available_at", start="2010-01-01")
    assert pd.to_datetime(primary.return_available_at, utc=True).max() < cutoff
    # Even a mistaken short declared horizon cannot leak a future five-day label.
    forecast, forecast_audit = prepare_mature_rows(bundle["primary_dataset"], decision_at=cutoff,
        label_columns=FORECAST_LABELS, availability_column="return_available_at", start="2010-01-01")
    assert pd.to_datetime(forecast.forecast_available_at, utc=True).max() < cutoff
    assert len(primary) - len(forecast) == 4
    assert "forecast_available_at" in forecast_audit["required_availability_columns"]
    assert audit["excluded_unmatured"] > 0


def test_pre_pit_training_is_explicitly_empty_and_date_weights_remain_normalized(frozen):
    _, calendar, _, bundle = frozen
    early = pd.Timestamp(f"{calendar[180].date()} 00:00:00", tz="America/New_York")
    rows, audit = prepare_mature_rows(bundle["auxiliary_dataset"], decision_at=early,
        label_columns=AUXILIARY_TARGETS, availability_column="auxiliary_available_at", start="2010-01-01")
    assert rows.empty and audit["rows"] == 0 and audit["label_available_at_max"] is None
    later = pd.Timestamp(f"{calendar[300].date()} 16:00:00", tz="America/New_York")
    rows, _ = prepare_mature_rows(bundle["auxiliary_dataset"], decision_at=later,
        label_columns=AUXILIARY_TARGETS, availability_column="auxiliary_available_at", start="2010-01-01")
    assert rows.groupby(level="datetime").sample_weight.sum().eq(1.).all()
    assert rows.auxiliary_available_at.max() < later


def test_newer_mutable_provider_cannot_silently_be_truncated(frozen):
    _, calendar, spec, _ = frozen
    with pytest.raises(ValueError, match="beyond the frozen as_of"):
        load_integrated_data(replace(spec, as_of=str(calendar[-2].date())))


def test_missing_provider_and_naive_cutoff_fail_closed(frozen, tmp_path):
    _, _, spec, bundle = frozen
    with pytest.raises(ValueError, match="audited Qlib provider is unavailable"):
        load_integrated_data(replace(spec, root=tmp_path))
    with pytest.raises(ValueError, match="explicit timezone"):
        prepare_mature_rows(bundle["primary_dataset"], decision_at="2011-01-01", label_columns=RETURN_LABELS,
                            availability_column="return_available_at")
