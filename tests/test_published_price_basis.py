from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.price_basis import PriceBasisError, load_qlib_price_calendar, validate_price_basis
from src.data.price_basis import _sha
from src.data.published_price_basis import build_published_basis
from src.data.qlib_native import build_qlib_store_from_frame


@pytest.fixture
def inputs():
    dates = ["2010-03-11", "2010-09-22", "2011-01-03", "2011-01-04"]
    primary = pd.DataFrame([
        {"code": f"US.{symbol}", "time_key": f"{day} 00:00:00",
         "open": 100., "high": 110., "low": 90., "close": 105., "volume": 1000.}
        for symbol in ("SOXL", "SOXX") for day in dates
        if (symbol, day) != ("SOXL", dates[0])
    ])
    # Quote bars already reflect a later 4-for-1 split, including volume.
    secondary = pd.DataFrame([{"date": dates[0], "open": 25., "high": 27.5,
                               "low": 22.5, "close": 26.25, "volume": 4000., "adjclose": -999.}])
    rehab = pd.DataFrame([
        {"ex_div_date": "2010-09-22", "forward_adj_factorA": 1., "forward_adj_factorB": -0.00771},
        {"ex_div_date": "2011-01-03", "forward_adj_factorA": .25, "forward_adj_factorB": 0.,
         "split_base": 1., "split_ert": 4., "split_ratio": .25},
    ])
    evidence = {symbol: {"before_sha256": "a" * 64, "after_sha256": "a" * 64,
                          "before_observed_at_utc": "2011-01-05T00:00:00Z",
                          "after_observed_at_utc": "2011-01-05T02:00:00Z"} for symbol in ("SOXL", "SOXX")}
    common = {"retrieval_id": "synthetic", "retrieved_at_utc": "2011-01-05T01:00:00Z",
              "session": "RTH", "extended_time": False, "timezone": "America/New_York",
              "raw_artifacts": [{"name": "synthetic", "sha256": "b" * 64}], "rehabilitation": evidence}
    primary_id = common | {"source": "Futu", "adjustment": "NONE", "requested_start": dates[0], "requested_end": dates[-1]}
    secondary_id = common | {"source": "Yahoo", "adjustment": "SPLIT_ONLY", "instrument": "SOXL",
                             "requested_start": dates[0], "requested_end": "2010-03-12"}
    return primary, secondary, {symbol: rehab.copy() for symbol in ("SOXL", "SOXX")}, primary_id, secondary_id


def test_chronological_affine_all_fields_exact_split_volume_and_source_identity(inputs):
    frame, audit = build_published_basis(*inputs, as_of="2011-01-04")
    assert len(frame) == 8
    before = frame.loc[(frame.instrument == "SOXX") & (frame.datetime == "2010-03-11")].iloc[0]
    yahoo = frame.loc[(frame.instrument == "SOXL") & (frame.datetime == "2010-03-11")].iloc[0]
    for field, raw in (("open", 100), ("high", 110), ("low", 90), ("close", 105)):
        assert before[field] == pytest.approx((raw - .00771) * .25)
        assert yahoo[field] == before[field]
    assert before.affine_A == .25
    assert before.affine_B == -.00771 * .25
    assert before.volume == yahoo.volume == 4000
    assert yahoo.source_code == "SOXL"
    assert before.source_code == "US.SOXX"
    assert yahoo.source_identity_sha256 != before.source_identity_sha256
    same_day = frame.loc[(frame.instrument == "SOXX") & (frame.datetime == "2011-01-03")].iloc[0]
    assert same_day.open == 100  # event is effective on this date, not reapplied
    assert same_day.volume == 1000
    assert audit["secondary_whole_bar_rows"] == 1
    assert audit["input_rows_discarded"] == 0
    assert not audit["sdk_qfq_equivalence_claimed"]


def test_generic_qualification_accepts_declared_published_basis(inputs, tmp_path):
    frame, _ = build_published_basis(*inputs, as_of="2011-01-04")
    reference = tmp_path / "reference"
    build_qlib_store_from_frame(frame, provider_uri=reference, as_of="2011-01-04",
                               field_map={name: name for name in ("open", "high", "low", "close", "volume")})
    calendar = load_qlib_price_calendar(reference, "2010-03-11", "2011-01-04")
    audit = validate_price_basis(frame, calendar, "2011-01-04")
    assert audit["qualification_pass"], audit
    modified = frame.copy(deep=True)
    modified.loc[0, "affine_A"] = .3
    assert not validate_price_basis(modified, calendar, "2011-01-04")["qualification_pass"]
    # Even individually well-declared source identities cannot mix two bases.
    mixed = frame.copy(deep=True)
    mixed.attrs = copy.deepcopy(frame.attrs)
    primary_key = mixed.iloc[0].source_identity_sha256
    sdk_identity = dict(mixed.attrs["source_identities"][primary_key], adjustment="QFQ")
    mixed.attrs["source_identities"][_sha(sdk_identity)] = sdk_identity
    mixed.loc[0, "source_identity_sha256"] = _sha(sdk_identity)
    mixed_audit = validate_price_basis(mixed, calendar, "2011-01-04")
    assert not mixed_audit["basis_consistency_pass"]
    assert not mixed_audit["qualification_pass"]


def test_secondary_cannot_replace_or_influence_primary(inputs):
    primary, secondary, rehab, primary_id, secondary_id = inputs
    duplicate = secondary.copy()
    duplicate["date"] = "2010-09-22"
    with pytest.raises(PriceBasisError, match="cannot replace available primary"):
        build_published_basis(primary, duplicate, rehab, primary_id, secondary_id, "2011-01-04")
    base, _ = build_published_basis(*inputs, as_of="2011-01-04")
    secondary.loc[0, "close"] = 24.9
    changed, _ = build_published_basis(*inputs, as_of="2011-01-04")
    assert np.array_equal(base.iloc[:-1][["open", "high", "low", "close", "volume"]], changed.iloc[:-1][["open", "high", "low", "close", "volume"]])
    assert base.iloc[-1].close != changed.iloc[-1].close


def test_invalid_coefficients_or_inexact_split_contract_fail_closed(inputs):
    primary, secondary, rehab, primary_id, secondary_id = inputs
    altered = copy.deepcopy(rehab)
    altered["SOXL"].loc[0, "forward_adj_factorA"] = -1
    with pytest.raises(PriceBasisError, match="A must be positive"):
        build_published_basis(primary, secondary, altered, primary_id, secondary_id, "2011-01-04")
    altered = copy.deepcopy(rehab)
    altered["SOXL"] = altered["SOXL"].drop(columns=["split_base", "split_ert"])
    with pytest.raises(PriceBasisError, match="lacks exact published"):
        build_published_basis(primary, secondary, altered, primary_id, secondary_id, "2011-01-04")


def test_already_adjusted_input_identity_cannot_be_applied_twice(inputs):
    primary, secondary, rehab, primary_id, secondary_id = inputs
    primary_id = primary_id | {"adjustment": "published_futu_affine_v1", "original_adjustment": "NONE"}
    with pytest.raises(PriceBasisError, match="must declare NONE source OHLC"):
        build_published_basis(primary, secondary, rehab, primary_id, secondary_id, "2011-01-04")


def test_future_market_rows_rejected_and_future_split_not_silently_undone(inputs):
    primary, secondary, rehab, primary_id, secondary_id = inputs
    future = primary.copy()
    future.loc[0, "time_key"] = "2011-01-05 00:00:00"
    with pytest.raises(PriceBasisError, match="after as_of cannot be truncated"):
        build_published_basis(future, secondary, rehab, primary_id, secondary_id, "2011-01-04")
    altered = copy.deepcopy(rehab)
    altered["SOXL"].loc[1, "ex_div_date"] = "2011-01-05"
    with pytest.raises(PriceBasisError, match="post-as_of split"):
        build_published_basis(primary, secondary, altered, primary_id, secondary_id, "2011-01-04")


def test_rounded_split_ratio_does_not_override_exact_counts_and_join_is_supported(inputs):
    primary, secondary, rehab, primary_id, secondary_id = inputs
    altered = copy.deepcopy(rehab)
    altered["SOXX"].loc[1, ["split_base", "split_ert", "split_ratio"]] = [np.nan, np.nan, 7.001]
    altered["SOXX"].loc[1, ["join_base", "join_ert"]] = [7., 1.]
    frame, _ = build_published_basis(primary, secondary, altered, primary_id, secondary_id, "2011-01-04")
    row = frame.loc[(frame.instrument == "SOXX") & (frame.datetime == "2010-03-11")].iloc[0]
    assert row.split_price_factor == 7
    assert row.volume == pytest.approx(1000 / 7)


def test_invalid_observed_source_bars_abort_before_any_transformation(inputs):
    inputs[0].loc[0, "volume"] = 0
    inputs[1].loc[0, "high"] = 1
    with pytest.raises(PriceBasisError, match="invalid observed source OHLCV before affine"):
        build_published_basis(*inputs, as_of="2011-01-04")


def test_positive_affine_intercept_cannot_heal_invalid_raw_prices(inputs):
    primary, secondary, rehab, primary_id, secondary_id = inputs
    secondary.loc[0, ["open", "high", "low", "close"]] = [-4., -2., -5., -3.]
    rehab["SOXL"].loc[1, "forward_adj_factorB"] = 100.
    with pytest.raises(PriceBasisError, match="invalid observed source OHLCV before affine"):
        build_published_basis(primary, secondary, rehab, primary_id, secondary_id, "2011-01-04")


@pytest.mark.parametrize("source,key,contradiction", [
    ("Yahoo", "volume_basis", "raw_shares"),
    ("Futu", "volume_basis", "split_adjusted_shares"),
    ("Yahoo", "original_volume_basis", "raw_shares"),
    ("Futu", "volume_unit", "lots"),
])
def test_explicit_wrong_raw_volume_basis_cannot_be_relabelled(inputs, source, key, contradiction):
    primary, secondary, rehab, primary_id, secondary_id = inputs
    identity = primary_id if source == "Futu" else secondary_id
    identity[key] = contradiction
    with pytest.raises(PriceBasisError, match="contradicts observed"):
        build_published_basis(primary, secondary, rehab, primary_id, secondary_id, "2011-01-04")
