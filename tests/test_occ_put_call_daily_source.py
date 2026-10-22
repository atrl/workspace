import pytest

from src.data.occ_put_call_daily_source import parse_daily


SAMPLE = (b"quantity,underlying,symbol,actype,porc,exchange,actdate\r\n"
          b"12,SOXX,SOXX,C,C,CBOE,10/20/2026,\r\n"
          b"8,SOXX,2SOXX,M,P,C2,10/20/2026,\r\n")


def test_occ_daily_parser_preserves_class_and_adjusted_root():
    parsed = parse_daily(SAMPLE, "SOXX", "2026-10-20")
    assert parsed["call_quantity"] == 12 and parsed["put_quantity"] == 8
    assert parsed["roots"] == ["2SOXX", "SOXX"]
    with pytest.raises(ValueError, match="identity"):
        parse_daily(SAMPLE, "SOXL", "2026-10-20")
    with pytest.raises(ValueError, match="header"):
        parse_daily(b"Report date cannot be prior to  2 years\r\n", "SOXX", "2023-01-03")
