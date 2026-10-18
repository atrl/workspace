from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from src.workflow.finra_short_mix_source_intake import BudgetedHTTP, month_links


def test_redirect_counts_global_budget_and_successful_cache_is_not_fetched(tmp_path):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(302 if self.path == "/redirect" else 200)
            if self.path == "/redirect":
                self.send_header("Location", "/body")
            self.end_headers()
            self.wfile.write(b"body")
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = BudgetedHTTP(tmp_path, maximum=2, retry_maximum=0, gap=.25,
                              headers={}, allowed_hosts={"127.0.0.1"}, local_test=True)
        base = "http://127.0.0.1:" + str(server.server_port)
        body, receipt = client.get(base + "/redirect", "daily")
        assert body == b"body" and receipt["final_url"] == base + "/body"
        assert len(client.state["attempts"]) == 2
        starts = [row["launch_monotonic"] for row in client.state["attempts"]]
        assert starts[1] - starts[0] >= .25
        assert client.get(base + "/redirect", "daily")[0] == body
        assert client.get(base + "/body", "daily")[0] == body
        assert calls == ["/redirect", "/body"]
        with pytest.raises(ValueError, match="budget exhausted"):
            client.get(base + "/another", "daily")
        assert len(calls) == 2
        concurrent = BudgetedHTTP(tmp_path / "concurrent", maximum=3, retry_maximum=0, gap=.25,
                                 headers={}, allowed_hosts={"127.0.0.1"}, local_test=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda suffix: concurrent.get(base + suffix, "daily"), ["/a", "/b", "/c"]))
        starts = [row["launch_monotonic"] for row in concurrent.state["attempts"]]
        assert len(starts) == 3 and min(b - a for a, b in zip(starts, starts[1:])) >= .25
        assert len(calls) == 5
        nofollow = BudgetedHTTP(tmp_path / "nofollow", maximum=2, retry_maximum=0, gap=.25,
            headers={}, allowed_hosts={"127.0.0.1"}, local_test=True, follow_redirects=False)
        body, receipt = nofollow.get(base + "/redirect", "daily")
        assert body is None and receipt["unresolved_redirect"] and len(nofollow.state["attempts"]) == 1
        assert len(calls) == 6 and calls[-1] == "/redirect"
        nofollow.canonical_daily_only = True
        nofollow._allowed(base + "/equity/regsho/daily/CNMSshvol20261009.txt")
        with pytest.raises(ValueError, match="canonical direct daily"):
            nofollow._allowed(base + "/alternate")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_month_index_must_verify_selected_filters_and_CNMS_scope():
    template = '''<select name="custom_month[month]"><option value="10" selected>October</option></select>
    <select name="custom_year[year]"><option value="0" selected>2026</option></select>
    <h2>FINRA Consolidated NMS</h2><ul><li><a href="https://cdn.finra.org/equity/regsho/daily/CNMSshvol20261009.txt">9th</a></li>
    <li><a href="https://cdn.finra.org/equity/regsho/daily/CNMSshvol20261009_updated.txt">9th Updated</a></li></ul>
    <h2>FINRA/NYSE TRF</h2><a href="https://cdn.finra.org/equity/regsho/daily/NYXshvol20261009.txt">9th</a>'''
    result = month_links(template.encode(), 2026, 10)
    assert result["valid"] and len(result["links"]) == 2
    assert [row["updated"] for row in result["links"]] == [False, True]
    assert not month_links(template.encode(), 2020, 3)["valid"]
    assert not month_links(template.replace("20261009", "20260930").encode(), 2026, 10)["valid"]
    assert not month_links(b"<h2>FINRA Consolidated NMS</h2>", 2026, 10)["valid"]


def test_429_stops_host_without_retry_and_cache_only_branch_makes_no_request(tmp_path):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(429)
            self.send_header("Retry-After", "120")
            self.end_headers()
            self.wfile.write(b"<html><title>Just a moment...</title></html>")
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = "http://127.0.0.1:" + str(server.server_port)
        client = BudgetedHTTP(tmp_path, maximum=10, retry_maximum=16, gap=.25,
                              headers={}, allowed_hosts={"127.0.0.1"}, local_test=True)
        assert client.get(base + "/first", "daily")[0] is None
        assert client.get(base + "/second", "daily")[1]["no_http_request"]
        assert len(client.state["attempts"]) == 1 and not client.state["retried"]
        assert client.state["paused_hosts"]["127.0.0.1"]["not_before_utc"] is not None
        cached = BudgetedHTTP(tmp_path / "cacheonly", maximum=10, retry_maximum=16, gap=.25,
            headers={}, allowed_hosts={"127.0.0.1"}, local_test=True, cache_only_hosts={"127.0.0.1"})
        assert cached.get(base + "/third", "month_index")[1]["no_http_request"]
        assert calls == ["/first"] and len(cached.state["attempts"]) == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
