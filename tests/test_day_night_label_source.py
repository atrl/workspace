"""One synthetic-only source boundary, native persistence and label contract."""
import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.day_night_labels import (CLOCKS, TARGETS, basis_certificate, original_target_digest,
    assert_reference_unchanged, build_h5_mapping, validate_prices, h1_vintage_diagnostic,
    common_log_plane, construct_labels, binary64_environment, _bound)
from src.workflow.day_night_label_source import (native_dataset, prepare_all_groups, bounded_native_read, _init)


def test_fixed_source_basis_calendar_bound_and_native_contract(tmp_path):
    # Synthetic provider includes later rows deliberately excluded from price IO.
    # Index headers and calendars are metadata; no real provider is inspected.
    calendar = pd.bdate_range("2020-01-02", periods=16)
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    (provider / "calendars/day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n")
    (provider / "instruments/all.txt").write_text("\n".join(f"{s}\t{calendar[0].date()}\t{calendar[-1].date()}" for s in ("SOXL", "SOXX")) + "\n")
    boundary = calendar[5]
    expected = {s: {"early_segment_before": str(boundary.date()), "k_open": ko, "k_close": kc}
                for s, ko, kc in (("SOXL", 2.0, 4.0), ("SOXX", 3.0, 1.5))}
    metadata = {"symbol_audits": {s: {"nasdaq_start": c["early_segment_before"],
                 "stitch_scales": {"open": c["k_open"], "close": c["k_close"]}} for s, c in expected.items()}}
    stored = {}
    for n, symbol in enumerate(("SOXL", "SOXX")):
        # A varying common distribution factor is retained in N, cancels in D.
        factor = 0.8 + np.arange(16) * .004
        raw_open = 30 + n * 15 + np.arange(16) * (1 + .1 * n)
        raw_close = raw_open * (1.02 - .005 * n)
        early = calendar < boundary
        ko = np.where(early, expected[symbol]["k_open"], 1.)
        kc = np.where(early, expected[symbol]["k_close"], 1.)
        folder = provider / "features" / symbol.lower(); folder.mkdir(parents=True)
        for name, values in (("open", raw_open * factor * ko), ("close", raw_close * factor * kc)):
            array = values.astype(np.float32); stored[(symbol, name)] = array.copy()
            np.concatenate([np.array([0], dtype=np.float32), array]).astype("<f4").tofile(folder / (name + ".day.bin"))
    _init(provider)
    counts = []
    native_calendar, prices, read_audit = bounded_native_read(provider,
        start=str(calendar[0].date()), end=str(calendar[11].date()), before_call=counts.append)
    assert counts == ["D_calendar_calls", "D_features_calls"]
    assert len(read_audit["raw_field_slices"]) == 4
    assert all(x["end_index"] == 11 for x in read_audit["raw_field_slices"])
    assert prices.index.get_level_values("datetime").max() == calendar[11]
    assert len(native_calendar) == 16
    # Nonconsecutive signal cohort must map on all sessions, not shift rows.
    selected = [0, 3, 5]
    reference = pd.DataFrame(index=pd.DatetimeIndex(calendar[selected], name="signal_date"))
    for name, offset, minute in ((CLOCKS[0], 1, 29), (CLOCKS[1], 2, 31), (CLOCKS[2], 6, 31)):
        reference[name] = pd.DatetimeIndex(calendar[np.array(selected) + offset]).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=minute)
    for n, target in enumerate(TARGETS):
        reference[target] = np.array([.01 + .1 * n, -.02, .03], dtype=np.float32)
    before = original_target_digest(reference)
    mapping = build_h5_mapping(native_calendar, reference, expected_rows=3)
    assert mapping.loc[calendar[3], "d0"] == calendar[4]
    assert mapping.loc[calendar[3], "d5"] == calendar[9]
    validate_prices(prices, mapping)
    paired, diagnostic = h1_vintage_diagnostic(prices, mapping, reference)
    assert any(row["max_absolute_difference"] > 0 for row in diagnostic)
    assert all(row["acceptance_threshold"] is None and row["diagnostic_only"] for row in diagnostic)
    assert paired.shape == (3, 6)
    certificate = basis_certificate(metadata, expected)
    original_prices = prices.copy(deep=True)
    plane = common_log_plane(prices, certificate)
    for symbol in ("SOXL", "SOXX"):
        for date in calendar[:12]:
            opening = prices.at[(date, symbol), "$open"]
            closing = prices.at[(date, symbol), "$close"]
            assert plane.at[(date, symbol), "ell_open"] == math.log(float(opening))
            correction = math.log(expected[symbol]["k_open"]) - math.log(expected[symbol]["k_close"])
            actual = math.log(float(closing)) + correction if date < boundary else math.log(float(closing))
            assert plane.at[(date, symbol), "ell_close"] == actual
    labels, audit = construct_labels(plane, mapping)
    assert len(labels) == 3 and len(audit) == 9 and audit.passed.all()
    assert set(audit.leaf_count) == {22, 44}
    assert np.isfinite(audit.bound).all() and (audit.residual.abs() <= audit.bound).all()
    assert (labels.Z == labels.N_relative - labels.D_relative).all()
    # Corrected closes cancel only in the total; the fixed stored opens stay exact.
    assert np.allclose(labels.S, labels.S_endpoint, rtol=0, atol=audit.bound.max())
    pd.testing.assert_frame_equal(prices, original_prices)
    assert_reference_unchanged(reference, before)
    assert binary64_environment()["gradual_underflow"]
    with pytest.raises((KeyError, ValueError)):
        basis_certificate({"symbol_audits": {}}, expected)
    bad = json.loads(json.dumps(metadata)); bad["symbol_audits"]["SOXL"]["stitch_scales"]["close"] = 1.00000001
    with pytest.raises(ValueError, match="basis metadata"):
        basis_certificate(bad, expected)
    bad_reference = reference.copy(); bad_reference.iloc[0, 0] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="clock mismatch"):
        build_h5_mapping(native_calendar, bad_reference, expected_rows=3)
    changed = reference.copy(); changed.iloc[0, 3] += np.float32(.1)
    with pytest.raises(ValueError, match="mutated"):
        assert_reference_unchanged(changed, before)
    missing = prices.copy(); missing.loc[(mapping.iloc[0].d2, "SOXL"), "$close"] = np.nan
    with pytest.raises(ValueError, match="H5 close"):
        validate_prices(missing, mapping)
    with pytest.raises(ValueError, match="nonfinite/overflow"):
        _bound([float(np.finfo(float).max)] * 44)
    # Native serialization preserves groups/order and exact timestamps. No setup_data.
    frame = pd.concat({"price": prices}, axis=1)
    dataset = native_dataset(frame)
    restored = pickle.loads(pickle.dumps(dataset))
    pd.testing.assert_frame_equal(prepare_all_groups(restored), frame)
    assert all(values.dtype == np.float32 for _, values in prepare_all_groups(restored).items())
