import numpy as np
import pandas as pd
import pytest

from src.data.price_basis import PriceBasisError, _sha
from src.data.soxs_published_close import compose_events, transform_soxs_close


def _calendar():
    payload = {"source_api": "qlib.data.D.calendar", "freq": "day", "as_of": "2020-01-06",
               "requested_start": "2020-01-02", "dates": ["2020-01-02", "2020-01-03", "2020-01-06"]}
    return payload | {"calendar_sha256": _sha(payload)}


def _sources():
    futu = pd.DataFrame({"code": ["US.SOXS", "US.SOXS"], "time_key": ["2020-01-02", "2020-01-06"], "close": [10., 19.]})
    yahoo = pd.DataFrame({"date": _calendar()["dates"], "close": [200., 20., 190.], "volume": [0, 0, 0]})
    for frame in (futu, yahoo):
        for field in ("open", "high", "low"): frame[field] = frame.close
    events = pd.DataFrame({"date": pd.to_datetime(["2020-01-06"]), "a": [2.], "b": [-1.], "split": [2.]})
    return futu, yahoo, events


def test_affine_close_uses_raw_primary_and_undoes_secondary_exact_splits():
    futu, yahoo, events = _sources()
    frame, audit = transform_soxs_close(futu, yahoo, events, _calendar(), "2020-01-06")
    assert frame.close.tolist() == [19., 19., 19.]
    assert frame.source.tolist() == ["Futu_NONE", "Yahoo_SPLIT_ONLY", "Futu_NONE"]
    assert audit["primary_rows"] == 2 and audit["secondary_rows"] == 1
    assert not audit["volume_used"]
    assert "volume" not in frame


@pytest.mark.parametrize("failure", ["missing", "duplicate", "future", "nonpositive", "physical_ohlc"])
def test_close_qualification_fail_closed(failure):
    futu, yahoo, events = _sources()
    if failure == "missing": yahoo = yahoo.iloc[[0, 2]]
    if failure == "duplicate": futu = pd.concat([futu, futu.iloc[[0]]])
    if failure == "future":
        yahoo.loc[3] = yahoo.iloc[0]
        yahoo.loc[3, "date"] = "2020-01-07"
    if failure == "nonpositive": yahoo.loc[1, "close"] = np.nan
    if failure == "physical_ohlc": yahoo.loc[1, "low"] = 21.
    with pytest.raises(PriceBasisError):
        transform_soxs_close(futu, yahoo, events, _calendar(), "2020-01-06")


def test_supplemental_split_must_match_vendor_and_be_before_futu_events():
    rehab = pd.DataFrame({"ex_div_date": ["2020-01-06"], "forward_adj_factorA": [2.], "forward_adj_factorB": [0.],
                          "join_base": [2.], "join_ert": [1.]})
    supplement = [{"date": "2020-01-03", "a": 5., "b": 0., "split": 5., "source_url": "https://issuer.example/event", "evidence_file": "event.html"}]
    splits = {"a": {"date": int(pd.Timestamp("2020-01-03T14:30:00Z").timestamp()), "numerator": 1., "denominator": 5.},
              "b": {"date": int(pd.Timestamp("2020-01-06T14:30:00Z").timestamp()), "numerator": 1., "denominator": 2.}}
    events = compose_events(rehab, supplement, splits, "2020-01-06")
    assert events.split.tolist() == [5., 2.]
    splits["a"]["denominator"] = 4.
    with pytest.raises(PriceBasisError, match="split table"):
        compose_events(rehab, supplement, splits, "2020-01-06")
