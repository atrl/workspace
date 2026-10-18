from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.finra_short_mix_schema import (
    CACHE_FIELDS, CNMSSchemaError, HEADER, parse_cnms_file, source_frame,
)


def _body(rows):
    return ("|".join(HEADER) + "\r\n" + "\r\n".join(rows) + f"\r\n{len(rows)}\r\n").encode("ascii")


def test_retained_official_samples_full_file_contract():
    receipt_path = Path("research/stage1_finra_short_mix_20261018/probe_receipts.json")
    if not receipt_path.exists():
        pytest.skip("retained official probe receipts are not available in this checkout")
    samples = json.loads(receipt_path.read_text())["samples"]
    if not all(Path(sample["raw_path"]).exists() for sample in samples):
        pytest.skip("retained official probe bytes are unavailable; no network fallback")
    for sample in samples:
        body = Path(sample["raw_path"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == sample["sha256"]
        parsed = parse_cnms_file(body, sample["trade_date"])
        assert parsed["record_count"] == int(sample["trailer"])
        assert parsed["missing_symbols"] == []
        assert ["|".join(row[name] for name in HEADER) for row in parsed["rows"]] == sample["selected_rows"]
        json.dumps(parsed)  # Original decimal spellings must survive durable JSON receipts.
        if sample["trade_date"] == "20261009":
            assert parsed["schema"]["maximum_observed_decimal_places"] == 6
            assert parsed["rows"][0]["ShortVolume"] == "16217885.634043"
            assert Decimal(parsed["rows"][0]["ShortExemptVolume"]) == Decimal("2390308.327559")


def test_rejects_invalid_non_target_records_and_schema_drift():
    valid = "20261005|SOXL|2|1|5|B,Q,N"
    bad_rows = [
        "20261006|OTHER|2|1|5|Q",  # A bad unselected record rejects the entire file.
        "20261005|OTHER|2|3|5|Q",
        "20261005|OTHER|6|1|5|Q",
        "20261005|OTHER|2.0000001|1|5|Q",
        "20261005|OTHER|NaN|1|5|Q",
        "20261005|OTHER|2|1|5|O",  # ORF is outside the CNMS scope.
        "20261005|OTHER|2|1|5|Q,Q",
        "20261005|OTHER|2|1|5|Q|extra",
    ]
    for bad in bad_rows:
        with pytest.raises(CNMSSchemaError):
            parse_cnms_file(_body([valid, bad]), "20261005")
    duplicate = "20261005|OTHER|2|1|5|D"
    with pytest.raises(CNMSSchemaError, match="duplicate"):
        parse_cnms_file(_body([valid, duplicate, duplicate]), "20261005")
    with pytest.raises(CNMSSchemaError, match="footer"):
        parse_cnms_file(_body([valid]).replace(b"\r\n1\r\n", b"\r\n2\r\n"), "20261005")
    with pytest.raises(CNMSSchemaError, match="header"):
        parse_cnms_file(_body([valid]).replace(b"ShortExemptVolume", b"ShortExempt"), "20261005")


def test_exact_quantities_missing_zero_and_qlib_derived_readback(tmp_path):
    first = parse_cnms_file(_body([
        "20261005|SOXL|100.123456|20.123456|200.123456|B,Q,N",
        "20261005|SOXX|20|5|100.000000|D",
    ]), "20261005")
    zero = parse_cnms_file(_body(["20261006|SOXL|0|0|0|Q"]), "20261006")
    frame = source_frame({"20261005": first, "20261006": zero, "20261007": None})
    assert first["rows"][1]["TotalVolume"] == "100.000000"
    selected = frame.set_index(["instrument", "datetime"]).sort_index()
    original = selected.loc[("SOXL", pd.Timestamp("2026-10-05"))]
    assert original.short_ratio == pytest.approx(float(Decimal("100.123456") / Decimal("200.123456")))
    assert original.short_ratio != pytest.approx(float(Decimal("120.246912") / Decimal("200.123456")))
    assert np.isnan(selected.loc[("SOXL", pd.Timestamp("2026-10-06")), "short_ratio"])
    assert selected.loc[("SOXL", pd.Timestamp("2026-10-06")), "total_volume"] == 0
    assert np.isnan(selected.loc[("SOXX", pd.Timestamp("2026-10-06")), "total_volume"])
    assert selected.loc[(slice(None), slice("2026-10-06", None)), "source_ready"].eq(0).all()

    import qlib
    from qlib.data import D
    from src.data import build_qlib_store_from_frame

    provider = tmp_path / "source_provider"
    build_qlib_store_from_frame(frame, provider_uri=provider, freq="day", region="us", as_of="2026-10-07",
                               field_map={name: name for name in CACHE_FIELDS},
                               source="synthetic CNMS parser regression; exact strings retained separately")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    observed = D.features(["SOXL", "SOXX"], ["$" + name for name in CACHE_FIELDS],
                          start_time="2026-10-05", end_time="2026-10-07", freq="day")
    expected = selected.sort_index().loc[:, list(CACHE_FIELDS)].astype("float32")
    pd.testing.assert_index_equal(observed.index, expected.index)
    np.testing.assert_allclose(observed.to_numpy(), expected.to_numpy(), rtol=0, atol=0, equal_nan=True)
    # Qlib stores float32.  A decimal share quantity is deliberately not
    # certified lossless even though it round-trips through that cache exactly.
    cached = float(observed.loc[("SOXL", pd.Timestamp("2026-10-05")), "$short_volume"])
    assert Decimal.from_float(cached) != Decimal(first["rows"][0]["ShortVolume"])
