"""A future Korean release is captured only inside its frozen source window."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from src.data.korea_export_forward_capture import _capture


ROOT = Path(__file__).resolve().parents[1]
CARD = Path("research/korea_semiconductor_exports_korean_vintage_20261022")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture(tmp_path):
    card = tmp_path / CARD
    card.mkdir(parents=True)
    plan = json.loads((ROOT / CARD / "forward_oct_2026_publication_plan.json").read_text())
    (card / "forward_oct_2026_publication_plan.json").write_text(json.dumps(plan))
    (card / "korean_national_article_manifest.json").write_bytes(
        (ROOT / CARD / "korean_national_article_manifest.json").read_bytes())
    for name in (plan["source_listing_parser"], plan["source_article_parser"], plan["implementation_module"]):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    binding = {"frozen": True, "plan_sha256": _sha(card / "forward_oct_2026_publication_plan.json"),
               "implementation_sha256": _sha(tmp_path / plan["implementation_module"])}
    (card / plan["implementation_binding"]).write_text(json.dumps(binding))
    return card


class _Response:
    status = 200
    headers = {"Content-Type": "text/html", "Date": "Sun, 01 Nov 2026 02:10:01 GMT"}

    def __init__(self, body, content_type="text/html"):
        self.body = body
        self.headers = {**self.headers, "Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _limit):
        return self.body


def _listing():
    rows = ['<tr><td><a href="/kor/article/ATCL3f49a5a8c/999/view">'
            '2026년 10월 수출입 동향</a></td><td>2026-11-01</td></tr>']
    rows += [f'<tr><td><a href="/kor/article/ATCL3f49a5a8c/{1000 + i}/view">'
             f'Other release {i}</a></td><td>2026-10-31</td></tr>' for i in range(4)]
    return ('<table>' + ''.join(rows) + '</table>').encode()


def test_before_publication_window_never_fetches(tmp_path):
    card = _fixture(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("network called before window")

    moment = dt.datetime(2026, 11, 1, 2, 0, tzinfo=dt.timezone.utc)
    with pytest.raises(ValueError, match="before frozen Korean source window"):
        _capture(tmp_path, "after_hypothesis", now=moment, fetch=forbidden, clock=lambda: moment)
    assert not (card / "forward_oct_2026").exists()


def test_source_bytes_and_article_identity_are_receipted(tmp_path):
    card = _fixture(tmp_path)
    article = (b'<div class="detail-tit"><span>2026\xeb\x85\x84 10\xec\x9b\x94 '
               b'\xec\x88\x98\xec\xb6\x9c\xec\x9e\x85 \xeb\x8f\x99\xed\x96\xa5</span></div>'
               b'<li><em>\xeb\x93\xb1\xeb\xa1\x9d\xec\x9d\xbc</em><span>2026-11-01</span></li>'
               b'<a href="/attach/down/a/b/c">October.pdf [1 KB]</a>')
    calls = []

    def fetch(request, timeout):
        assert timeout == 30
        url = request.full_url
        calls.append(url)
        if "/attach/down/" in url:
            return _Response(b"%PDF-1.4\nsource bytes", "application/pdf")
        if "/999/view" in url:
            return _Response(article)
        return _Response(_listing())

    moment = dt.datetime(2026, 11, 1, 2, 10, tzinfo=dt.timezone.utc)
    receipt = _capture(tmp_path, "after_hypothesis", now=moment, fetch=fetch, clock=lambda: moment)
    assert receipt["status"] == "observed" and receipt["http_GETs"] == len(calls) == 4
    assert receipt["matching_article_ids"] == ["999"]
    for row in receipt["records"]:
        assert _sha(tmp_path / row["relative_path"]) == row["sha256"]
    assert (card / "forward_oct_2026/after_hypothesis.json").is_file()
