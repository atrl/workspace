import pytest

from src.data.cboe_volume_forward_capture import _rows


CSV = b"Trade Date,Underlying,Product Type,Exchange,Volume\n2026/10/20,SOXX,S,CBOE,7\n2026/10/20,SOXX,S,C2,3\n"


def test_forward_capture_preserves_exchange_day_identity():
    assert _rows(CSV, "SOXX", {"CBOE", "C2"}) == {"2026-10-20": {"CBOE": 7, "C2": 3}}
    with pytest.raises(ValueError, match="duplicate"):
        _rows(CSV + b"2026/10/20,SOXX,S,CBOE,1\n", "SOXX", {"CBOE", "C2"})
    with pytest.raises(ValueError, match="scope"):
        _rows(CSV, "SOXL", {"CBOE", "C2"})
