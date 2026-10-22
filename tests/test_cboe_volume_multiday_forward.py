"""Focused guards for the bounded Cboe forward source observer."""
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from src.data.cboe_volume_multiday_forward import _capture, bind_third


ROOT = Path(__file__).resolve().parents[1]
CARD = Path("research/cboe_total_options_activity_source_20261022")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture(tmp_path):
    card = tmp_path / CARD
    card.mkdir(parents=True)
    plan = json.loads((ROOT / CARD / "multiday_forward_plan.json").read_text())
    (card / "multiday_forward_plan.json").write_text(json.dumps(plan))
    for name in ("forward_publication_plan.json", "forward_transport_addendum.json",
                 "forward_transport_identity_clarification.json"):
        (card / name).write_bytes((ROOT / CARD / name).read_bytes())
    parser = tmp_path / plan["parser_module"]
    parser.parent.mkdir(parents=True)
    parser.write_bytes((ROOT / plan["parser_module"]).read_bytes())
    observer = tmp_path / plan["observer_module"]
    observer.write_bytes((ROOT / plan["observer_module"]).read_bytes())
    addendum = json.loads((card / "forward_transport_addendum.json").read_text())
    for relative in addendum["prior_receipts_sha256"]:
        destination = card / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / CARD / relative).read_bytes())
    second = card / max(addendum["prior_receipts_sha256"])
    receipt = json.loads(second.read_text())
    receipt["index"] = 3
    receipt["observation_id"] = "test-third"
    for row in receipt["records"]:
        row["requested_at_utc"] = "2026-10-22T12:45:01+00:00"
    third = card / "forward_observations/test-third.json"
    third.write_text(json.dumps(receipt))
    binding = bind_third(tmp_path)
    assert binding["third_values"] == {"SOXX": 7425, "SOXL": 53614}
    return card


class _Response:
    status = 200
    headers = {"Content-Type": "text/csv", "Date": "Fri, 23 Oct 2026 04:10:01 GMT"}

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _limit):
        return self.body


def test_frozen_window_blocks_early_fetch(tmp_path):
    card = _fixture(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("network called before the frozen window")

    with pytest.raises(ValueError, match="before frozen window"):
        _capture(tmp_path, "2026-10-22", "post_midnight",
                 now=dt.datetime(2026, 10, 23, 4, 0, tzinfo=dt.timezone.utc),
                 fetch=forbidden, clock=lambda: dt.datetime(2026, 10, 23, 4, 0, tzinfo=dt.timezone.utc))
    assert not (card / "multiday_forward_observations/2026-10-22").exists()


def test_first_day_preserves_raw_and_predecessor_check(tmp_path):
    card = _fixture(tmp_path)
    requests = []

    def fetch(request, timeout):
        assert timeout == 30
        symbol = parse_qs(urlparse(request.full_url).query)["symbol"][0]
        requests.append(symbol)
        previous, current = {"SOXX": (7425, 8100), "SOXL": (53614, 55000)}[symbol]
        body = ("Trade Date,Underlying,Product Type,Exchange,Volume\n"
                f"2026/10/21,{symbol},S,CBOE,{previous}\n"
                f"2026/10/22,{symbol},S,CBOE,{current}\n").encode()
        return _Response(body)

    receipt = _capture(tmp_path, "2026-10-22", "post_midnight",
                       now=dt.datetime(2026, 10, 23, 4, 10, tzinfo=dt.timezone.utc),
                       fetch=fetch, clock=lambda: dt.datetime(2026, 10, 23, 4, 10, tzinfo=dt.timezone.utc))
    assert requests == ["SOXX", "SOXL"]
    assert receipt["status"] == "observed" and receipt["http_GETs"] == 2
    assert all(row["target_date_present"] and row["predecessor_unchanged"] for row in receipt["records"])
    for row in receipt["records"]:
        assert _sha(tmp_path / row["relative_path"]) == row["sha256"]
    assert (card / "multiday_forward_observations/2026-10-22/post_midnight.json").is_file()
    with pytest.raises(ValueError, match="already attempted"):
        _capture(tmp_path, "2026-10-22", "post_midnight",
                 now=dt.datetime(2026, 10, 23, 4, 11, tzinfo=dt.timezone.utc),
                 fetch=fetch, clock=lambda: dt.datetime(2026, 10, 23, 4, 11, tzinfo=dt.timezone.utc))


def test_second_symbol_does_not_fetch_after_window_closes(tmp_path):
    _fixture(tmp_path)
    requests = []
    moments = iter([dt.datetime(2026, 10, 23, 4, 54, tzinfo=dt.timezone.utc),
                    dt.datetime(2026, 10, 23, 4, 56, tzinfo=dt.timezone.utc)])

    def fetch(request, timeout):
        requests.append(parse_qs(urlparse(request.full_url).query)["symbol"][0])
        return _Response(("Trade Date,Underlying,Product Type,Exchange,Volume\n"
                          "2026/10/21,SOXX,S,CBOE,7425\n"
                          "2026/10/22,SOXX,S,CBOE,8100\n").encode())

    receipt = _capture(tmp_path, "2026-10-22", "post_midnight",
                       now=dt.datetime(2026, 10, 23, 4, 54, tzinfo=dt.timezone.utc),
                       fetch=fetch, clock=lambda: next(moments))
    assert requests == ["SOXX"]
    assert receipt["status"] == "partial_failure" and receipt["http_GETs"] == 1
    assert "outside frozen window" in receipt["records"][1]["error"]
