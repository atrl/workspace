"""Prospective issuer bytes must be bounded by time and source identity."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from src.data.tsmc_press_forward_capture import _capture, _press_links


ROOT = Path(__file__).resolve().parents[1]
CARD = Path("research/tsmc_monthly_revenue_release_source_20261022")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture(tmp_path):
    card = tmp_path / CARD
    card.mkdir(parents=True)
    plan = json.loads((ROOT / CARD / "future_2026_10_press_capture_plan.json").read_text())
    (card / "future_2026_10_press_capture_plan.json").write_text(json.dumps(plan))
    for name in ("future_2026_10_revenue_calendar_probe.json", "alternate_issuer_press_review.json"):
        (card / name).write_bytes((ROOT / CARD / name).read_bytes())
    code = tmp_path / plan["implementation_module"]
    code.parent.mkdir(parents=True)
    code.write_bytes((ROOT / plan["implementation_module"]).read_bytes())
    binding = {"frozen": True, "plan_sha256": _sha(card / "future_2026_10_press_capture_plan.json"),
               "implementation_sha256": _sha(code)}
    (card / plan["implementation_binding"]).write_text(json.dumps(binding))
    return card


class _Response:
    status = 200

    def __init__(self, body, url, content_type="text/html"):
        self.body = body
        self.url = url
        self.headers = {"Content-Type": content_type, "Date": "Tue, 10 Nov 2026 05:40:01 GMT"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _limit):
        return self.body


def test_date_prefix_listing_selector_and_early_gate(tmp_path):
    card = _fixture(tmp_path)
    listing = b'<a href="/english/news/3350">2026/11/10 TSMC October 2026 Revenue Report</a>'
    assert _press_links(listing, "TSMC October 2026 Revenue Report") == ["https://pr.tsmc.com/english/news/3350"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("network called before frozen time")

    moment = dt.datetime(2026, 11, 10, 5, 30, tzinfo=dt.timezone.utc)
    with pytest.raises(ValueError, match="before frozen window"):
        _capture(tmp_path, "after_scheduled", now=moment, fetch=forbidden, clock=lambda: moment)
    assert not (card / "future_2026_10_press").exists()


def test_exact_article_and_pdf_receipt_with_calendar_failure(tmp_path):
    card = _fixture(tmp_path)
    listing = b'<a href="/english/news/3350">2026/11/10 TSMC October 2026 Revenue Report</a>'
    article = (b'<h1>TSMC October 2026 Revenue Report</h1><p>Issued on: 2026/11/10</p>'
               b'<a href="/system/files/newspdf/attachment/october.pdf">October report</a>')
    calls = []

    def fetch(request, timeout):
        assert timeout == 30
        url = request.full_url
        calls.append(url)
        if "financial-calendar" in url:
            raise ValueError("calendar temporarily unavailable")
        if "latest-news" in url:
            return _Response(listing, url)
        if "/english/news/3350" in url:
            return _Response(article, url)
        return _Response(b"%PDF-1.4\nexact source", url, "application/pdf")

    moment = dt.datetime(2026, 11, 10, 5, 40, tzinfo=dt.timezone.utc)
    receipt = _capture(tmp_path, "after_scheduled", now=moment, fetch=fetch, clock=lambda: moment)
    assert receipt["status"] == "source_observed_calendar_unverified"
    assert receipt["http_GETs"] == len(calls) == 4
    for row in receipt["records"][1:]:
        assert _sha(tmp_path / row["relative_path"]) == row["sha256"]
    assert (card / "future_2026_10_press/after_scheduled.json").is_file()
