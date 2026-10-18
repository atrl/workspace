"""Budgeted official FINRA source intake. No financial labels or model fitting."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import tarfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src.data.minute_intake import atomic_json
from src.data.finra_short_mix_schema import CACHE_FIELDS, parse_cnms_file, source_frame

NAME = "stage1_finra_short_mix_20261018"
CODE_FILES = ["src/workflow/finra_short_mix_source_intake.py", "src/data/finra_short_mix_schema.py",
              "src/data/qlib_native.py", "src/data/minute_intake.py", "src/data/nas_market_data.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def code_identity(root):
    return {name: sha(Path(root) / name) for name in CODE_FILES}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class BudgetedHTTP:
    """One atomic global attempt counter and launch clock, including redirects."""

    def __init__(self, work, *, maximum, retry_maximum, gap, headers, timeout=30,
                 allowed_hosts=("www.finra.org", "finra.org", "cdn.finra.org"), local_test=False):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "objects").mkdir(exist_ok=True)
        self.path = self.work / "http_state.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "attempts": [], "cache": {}, "failures": {}, "retried": [], "maximum": maximum}
        self.state.setdefault("failures", {})
        if self.state["maximum"] != maximum:
            raise ValueError("HTTP budget changed on resume")
        self.maximum, self.retry_maximum = int(maximum), int(retry_maximum)
        self.gap, self.headers, self.timeout = float(gap), dict(headers), timeout
        self.allowed_hosts, self.local_test = set(allowed_hosts), local_test
        self.lock, self.last_launch = threading.Lock(), 0.
        self.opener = build_opener(NoRedirect())

    def _allowed(self, url):
        parsed = urlsplit(url)
        scheme_ok = parsed.scheme == "https" or (self.local_test and parsed.scheme == "http")
        if not scheme_ok or parsed.hostname not in self.allowed_hosts or parsed.username or parsed.password:
            raise ValueError("unapproved HTTP/redirect destination")
        if not self.local_test and parsed.port not in (None, 443):
            raise ValueError("unapproved destination port")
        if re.search(r"/(?:login|signin|authorize)(?:/|$)", parsed.path, re.I):
            raise ValueError("source requires authentication; no login attempted")

    def _save(self):
        atomic_json(self.state, self.path)

    def _body(self, receipt):
        path = self.work / receipt["object"]
        if not path.is_file() or sha(path) != receipt["sha256"]:
            raise ValueError("successful cached source missing/changed; no silent refetch")
        return path.read_bytes()

    def import_probe(self, receipt):
        """Four already retained real HTTP requests consume the same daily budget."""
        url = receipt["url"]
        with self.lock:
            if url in self.state["cache"]:
                self._body(self.state["cache"][url])
                return
            if len(self.state["attempts"]) >= self.maximum:
                raise ValueError("HTTP budget exhausted importing probes")
            source = Path(receipt["raw_path"])
            if sha(source) != receipt["sha256"]:
                raise ValueError("retained readiness bytes changed")
            target = self.work / "objects" / (receipt["sha256"] + ".bin")
            target.write_bytes(source.read_bytes())
            item = {**receipt, "category": "daily", "state": "complete", "from_retained_probe": True,
                    "attempt_id": len(self.state["attempts"]), "object": str(target.relative_to(self.work)),
                    "original_url": url, "final_url": url}
            self.state["attempts"].append(item)
            self.state["cache"][url] = item
            self._save()

    def _request(self, url, category, original_url):
        self._allowed(url)
        response, open_error = None, None
        with self.lock:
            if len(self.state["attempts"]) >= self.maximum:
                raise ValueError("global HTTP hard budget exhausted")
            remaining = self.last_launch + self.gap - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            self.last_launch = time.monotonic()
            number = len(self.state["attempts"])
            self.state["attempts"].append({"attempt_id": number, "url": url, "original_url": original_url,
                "category": category, "state": "started", "request_started_at": now(),
                "launch_monotonic": self.last_launch})
            self._save()  # durable reservation BEFORE any HTTP request
            # Keep the launch lock through response headers. Body download may
            # overlap; actual opener calls cannot be squeezed together by a
            # delayed checkpoint write or thread scheduling.
            self.last_launch = time.monotonic()
            self.state["attempts"][number]["launch_monotonic"] = self.last_launch
            try:
                response = self.opener.open(Request(url, headers=self.headers), timeout=self.timeout)
            except HTTPError as error:
                response = error
            except (URLError, TimeoutError, OSError) as error:
                open_error = error
        try:
            if open_error is not None:
                raise open_error
            body = response.read()
            status = response.code
            digest = hashlib.sha256(body).hexdigest()
            target = self.work / "objects" / (digest + ".bin")
            target.write_bytes(body)
            result = {"state": "complete", "status": status, "received_at": now(), "bytes": len(body),
                "sha256": digest, "object": str(target.relative_to(self.work)),
                "headers": {name: response.headers.get(name) for name in ["Date", "Last-Modified", "Content-Type", "ETag", "Location", "Retry-After"]}}
        except (URLError, TimeoutError, OSError) as error:
            result = {"state": "failed", "status": None, "received_at": now(),
                      "error": type(error).__name__ + ": " + str(error)}
        finally:
            if response is not None:
                response.close()
        with self.lock:
            self.state["attempts"][number].update(result)
            self._save()
            return dict(self.state["attempts"][number])

    def get(self, original_url, category):
        with self.lock:
            cached = self.state["cache"].get(original_url)
            if cached is not None:
                return self._body(cached), dict(cached)
            if original_url in self.state["failures"]:
                return None, dict(self.state["failures"][original_url])
            unresolved = [row for row in self.state["attempts"]
                          if row.get("original_url") == original_url and row["state"] == "started"]
            if unresolved:
                raise ValueError("interrupted source attempt requires explicit review; do not silently repeat HTTP")
        url, visited, retry = original_url, set(), False
        while True:
            if url in visited or len(visited) >= 5:
                raise ValueError("redirect loop or five-hop limit reached")
            visited.add(url)
            receipt = self._request(url, "retry" if retry else category if url == original_url else "redirect", original_url)
            status = receipt["status"]
            if status in [301, 302, 303, 307, 308]:
                destination = receipt["headers"].get("Location")
                if not destination:
                    return None, receipt
                url = urljoin(url, destination)
                self._allowed(url)
                retry = False
                continue
            if status == 200:
                receipt.update(final_url=url, original_url=original_url)
                with self.lock:
                    self.state["cache"][original_url] = receipt
                    self.state["cache"][url] = receipt
                    self._save()
                return self._body(receipt), receipt
            retriable = status is None or status == 429 or (isinstance(status, int) and 500 <= status <= 599)
            with self.lock:
                allowed_retry = (retriable and original_url not in self.state["retried"]
                                 and len(self.state["retried"]) < self.retry_maximum)
                if allowed_retry:
                    self.state["retried"].append(original_url)
                    self._save()
            if not allowed_retry:
                with self.lock:
                    self.state["failures"][original_url] = receipt
                    self._save()
                return None, receipt
            # One fixed retry for this logical URL, counted again by _request.
            time.sleep(1.)
            visited.discard(url)
            retry = True


def month_links(body, year, month):
    """Fail unknown when filters or the Consolidated NMS section are unverifiable."""
    soup = BeautifulSoup(body, "html.parser")
    for name, value in [("custom_year[year]", year), ("custom_month[month]", month)]:
        select = soup.find("select", attrs={"name": name})
        selected = select.find("option", selected=True) if select else None
        if selected is None or not str(selected.get("value", "")).isdigit() or int(selected["value"]) != value:
            return {"valid": False, "reason": "requested_month_filter_not_verified", "links": []}
    heading = next((h for h in soup.find_all("h2") if h.get_text(" ", strip=True).casefold() == "finra consolidated nms"), None)
    if heading is None:
        return {"valid": False, "reason": "CNMS_section_missing", "links": []}
    links = []
    for node in heading.next_elements:
        if getattr(node, "name", None) == "h2":
            break
        if getattr(node, "name", None) != "a" or not node.get("href"):
            continue
        url = urljoin("https://www.finra.org", node["href"])
        parsed = urlsplit(url)
        match = re.search(r"CNMSshvol(\d{8})", parsed.path)
        if not match:
            continue
        if parsed.hostname != "cdn.finra.org" or not parsed.path.startswith("/equity/regsho/daily/"):
            return {"valid": False, "reason": "unexpected_CNMS_destination", "links": []}
        day = match.group(1)
        if day[:6] != f"{year:04}{month:02}":
            return {"valid": False, "reason": "CNMS_links_outside_requested_month", "links": []}
        parent = node.find_parent("li") or node.parent
        context = parent.get_text(" ", strip=True)
        links.append({"trade_date": day, "url": url, "updated": bool(re.search(r"\bUpdated\b", context, re.I)), "link_context": context})
    return {"valid": bool(links), "reason": "verified" if links else "no_CNMS_links_not_proof_of_no_revisions", "links": links}


@contextmanager
def process_lock(work):
    path = Path(work) / "intake.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _approval(root, card, plan_sha):
    review = json.loads((card / "design_review.json").read_text())
    execution = json.loads((card / "source_execution_approval.json").read_text())
    if review.get("plan_sha256") != plan_sha or review.get("verdict") != "APPROVE_SOURCE_INTAKE_ONLY":
        raise ValueError("source-only design approval missing or changed")
    if execution.get("plan_sha256") != plan_sha or execution.get("approved") is not True:
        raise ValueError("explicit source execution approval missing")
    if execution.get("code_sha256") != code_identity(root):
        raise ValueError("source intake implementation changed after approval")


def preflight(*, root, card):
    """Validate request scope and four retained source samples without HTTP."""
    root, card = Path(root), Path(card)
    plan = json.loads((card / "plan.json").read_text())
    inventory = json.loads((card / "source_inventory.json").read_text())
    if sha(card / "probe_receipts.json") != inventory["probe_receipts_sha256"]:
        raise ValueError("probe receipt identity changed")
    calendar = (card / "source_calendar.txt").read_text().splitlines()
    if len(calendar) != 2059 or len(set(calendar)) != len(calendar) or calendar != sorted(calendar):
        raise ValueError("request calendar is not the frozen ordered unique domain")
    if sha(card / "source_calendar.txt") != plan["calendar_text_sha256"]:
        raise ValueError("request calendar hash differs")
    if sha(root / plan["calendar"]["calendar_file"]) != plan["calendar"]["calendar_file_sha256"]:
        raise ValueError("native source calendar changed")
    request = plan["requests"]
    arithmetic = (request["daily_unique_count"] + request["month_index_count"] + request["official_document_count"]
                  + request["maximum_extra_updated_files"] + request["maximum_network_retries"])
    if arithmetic != request["maximum_total_HTTP_requests"] or arithmetic != 2242:
        raise ValueError("source HTTP budget arithmetic differs")
    samples = []
    for receipt in json.loads((card / "probe_receipts.json").read_text())["samples"]:
        path = Path(receipt["raw_path"])
        if sha(path) != receipt["sha256"]:
            raise ValueError("retained raw source bytes changed")
        parsed = parse_cnms_file(path.read_bytes(), receipt["trade_date"])
        samples.append({"trade_date": receipt["trade_date"], "sha256": receipt["sha256"],
                        "file_records": parsed["record_count"], "target_rows": len(parsed["rows"]),
                        "historical_original_vintage_certified": False})
    result = {"plan_sha256": sha(card / "plan.json"), "calendar_rows": len(calendar),
        "calendar_hash_verified": True, "HTTP_hard_limit": arithmetic, "retained_probe_requests": len(samples),
        "remaining_HTTP_max": arithmetic - len(samples), "samples": samples,
        "raw_authority": "original bytes and decimal strings", "float32_role": "derived cache with measured quantization",
        "bulk_HTTP_started": False, "new_model_fits": 0, "return_labels_computed": 0, "new_accounts": 0,
        "code_sha256": code_identity(root)}
    atomic_json(result, card / "source_preflight.json")
    return result


def run(*, root, card):
    """One approved source collection; never calls a financial label/model/account API."""
    root, card = Path(root), Path(card)
    plan_sha = sha(card / "plan.json")
    _approval(root, card, plan_sha)
    plan = json.loads((card / "plan.json").read_text())
    summary_path = card / "summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old.get("stage") == "source_intake_completed":
            if old.get("plan_sha256") != plan_sha:
                raise ValueError("completed source belongs to another plan")
            return old
    calendar = (card / "source_calendar.txt").read_text().splitlines()
    if sha(card / "source_calendar.txt") != plan["calendar_text_sha256"] or len(calendar) != 2059:
        raise ValueError("frozen request calendar changed")
    work = Path(plan["artifacts"]["temporary_download_root"])
    with process_lock(work):
        try:
            return _collect(root, card, plan, plan_sha, calendar, work)
        except BaseException as error:
            state_path = work / "http_state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {"attempts": []}
            atomic_json({"plan_sha256": plan_sha, "stage": "source_intake_failed_cache_preserved",
                "error": type(error).__name__ + ": " + str(error), "HTTP_attempts": len(state["attempts"]),
                "model_fits": 0, "accounts": 0}, card / "source_run_state.json")
            raise


def _collect(root, card, plan, plan_sha, calendar, work):
    requests = plan["requests"]
    client = BudgetedHTTP(work, maximum=requests["maximum_total_HTTP_requests"],
        retry_maximum=requests["maximum_network_retries"], gap=requests["minimum_global_request_gap_seconds"],
        headers=requests["headers"], timeout=requests["timeout_seconds"])
    for probe in json.loads((card / "probe_receipts.json").read_text())["samples"]:
        client.import_probe(probe)
    months = list(pd.period_range(requests["month_first"], requests["month_last"], freq="M"))
    metadata = {}
    for index_number, period in enumerate(months, 1):
        url = requests["month_index_url"] + "?" + urlencode({"custom_month[month]": period.month, "custom_year[year]": period.year})
        body, receipt = client.get(url, "month_index")
        parsed = month_links(body, period.year, period.month) if body is not None else {"valid": False, "reason": "HTTP_failure", "links": []}
        metadata[str(period)] = {**parsed, "receipt": receipt}
        if index_number % 10 == 0 or index_number == len(months):
            atomic_json({"plan_sha256": plan_sha, "stage": "collecting_month_metadata",
                "month_indexes_processed": index_number, "valid_month_indexes": sum(x["valid"] for x in metadata.values()),
                "HTTP_attempts": len(client.state["attempts"]), "model_fits": 0, "accounts": 0}, card / "source_run_state.json")
    documents = {}
    for url in requests["official_document_urls"]:
        _, receipt = client.get(url, "official_document")
        documents[url] = receipt
    updated = sorted({(row["trade_date"], row["url"]) for month in metadata.values() if month["valid"]
                      for row in month["links"] if row["updated"] and pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d") in calendar})
    if len(updated) > requests["maximum_extra_updated_files"]:
        atomic_json({"plan_sha256": plan_sha, "decision": "needs_source_amendment_updated_budget", "updated_links": len(updated),
                     "HTTP_attempts": len(client.state["attempts"]), "new_fits": 0}, card / "source_run_state.json")
        raise ValueError("explicit updated-file inventory exceeds frozen source budget")
    update_receipts = {}
    for day, url in updated:
        body, receipt = client.get(url, "updated")
        try:
            updated_schema = parse_cnms_file(body, day) if body is not None else None
            updated_error = None
        except ValueError as error:
            updated_schema, updated_error = None, str(error)
        update_receipts[url] = {"receipt": receipt, "trade_date": day,
            "schema": updated_schema, "schema_error": updated_error}
    parsed_by_date, receipts, errors = {}, {}, {}

    def daily(day):
        compact = day.replace("-", "")
        url = requests["daily_base_url"] + "CNMSshvol" + compact + ".txt"
        month = metadata[day[:7]]
        originals = {row["url"] for row in month["links"]
                     if row["trade_date"] == compact and not row["updated"]} if month["valid"] else set()
        if len(originals) == 1:
            chosen = next(iter(originals))
            if compact in requests["retained_probe_dates"] and chosen != url:
                raise ValueError("retained probe is not listed original; source budget/identity amendment required")
            url = chosen
        body, receipt = client.get(url, "daily")
        if body is None:
            return day, None, receipt, "HTTP_failure"
        try:
            parsed = parse_cnms_file(body, compact)
            return day, parsed, receipt, None
        except ValueError as error:
            return day, None, receipt, str(error)

    with ThreadPoolExecutor(max_workers=requests["workers"]) as pool:
        for number, (day, parsed, receipt, error) in enumerate(pool.map(daily, calendar), 1):
            parsed_by_date[day], receipts[day] = parsed, receipt
            if error:
                errors[day] = error
            if number % 50 == 0 or number == len(calendar):
                atomic_json({"plan_sha256": plan_sha, "stage": "collecting_source_only", "daily_processed": number,
                    "daily_valid": number - len(errors), "HTTP_attempts": len(client.state["attempts"]),
                    "HTTP_hard_limit": client.maximum, "model_fits": 0, "accounts": 0}, card / "source_run_state.json")
                print("source progress", number, "of", len(calendar), "HTTP", len(client.state["attempts"]), flush=True)
    revision_rows = []
    for day in calendar:
        month = metadata[day[:7]]
        links = [row for row in month["links"] if row["trade_date"] == day.replace("-", "")] if month["valid"] else []
        status = "unknown" if not month["valid"] or not links else "updated_link_present" if any(row["updated"] for row in links) else "original_link_observed_not_vintage_certified"
        revision_rows.append({"trade_date": day, "revision_metadata": status, "month_index_valid": month["valid"],
                              "original_bytes_certified": False, "links": links})
    panel = source_frame(parsed_by_date)
    return _archive(root, card, plan, plan_sha, work, client, panel, parsed_by_date,
                    receipts, metadata, documents, update_receipts, revision_rows, errors)


def _archive(root, card, plan, plan_sha, work, client, panel, parsed, receipts, metadata, documents, updates, revisions, errors):
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    from src.data import build_qlib_store_from_frame
    from src.data.nas_market_data import archive_nas_recorder
    provider = root / "data/qlib" / (NAME + "_source")
    manifest = build_qlib_store_from_frame(panel, provider_uri=provider, freq="day", region="us", as_of="2026-10-09",
        field_map={name: name for name in CACHE_FIELDS}, source="Official FINRA CNMS source fields; float32 derivative, no market prices or return labels")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    observed = D.features(["SOXL", "SOXX"], ["$" + name for name in CACHE_FIELDS], start_time="2018-08-01", end_time="2026-10-09")
    expected = panel.set_index(["instrument", "datetime"])[list(CACHE_FIELDS)].sort_index()
    observed = observed.rename(columns=lambda name: name.removeprefix("$")).reindex(expected.index)
    np.testing.assert_allclose(observed.to_numpy(), expected.to_numpy(dtype=np.float32), rtol=0, atol=0, equal_nan=True)
    quantization = []
    for name in CACHE_FIELDS:
        left, right = expected[name].to_numpy(float), observed[name].to_numpy(float)
        valid = np.isfinite(left) & np.isfinite(right)
        error = abs(left[valid] - right[valid])
        denominator = np.maximum(abs(left[valid]), 1e-30)
        quantization.append({"field": name, "finite_rows": int(valid.sum()), "max_absolute_error": float(error.max()) if len(error) else None,
                             "max_relative_error": float((error / denominator).max()) if len(error) else None,
                             "authority": "rawbytes and decimal strings, not float32 cache"})
    dataset_frame = expected.reset_index().set_index(["datetime", "instrument"]).sort_index()
    dataset = DatasetH(handler=DataHandlerLP.from_df(pd.concat({"feature": dataset_frame}, axis=1)), segments={"all": ("2018-08-01", "2026-10-09")})
    bundle = work / "raw_sources.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in sorted((work / "objects").iterdir()):
            archive.add(path, arcname="objects/" + path.name)
        archive.add(client.path, arcname="http_state.json")
    valid_documents = sum(receipt.get("status") == 200 for receipt in documents.values())
    raw_daily_complete = not errors and len(parsed) == 2059
    revision_unknown = sum(row["revision_metadata"] == "unknown" for row in revisions)
    revision_complete = all(month["valid"] for month in metadata.values()) and revision_unknown == 0
    source_ready = raw_daily_complete and valid_documents == 4
    coverage = panel.assign(year=pd.to_datetime(panel.datetime).dt.year).groupby(["year", "instrument"]).agg(
        requested_days=("source_ready", "size"), ready_days=("source_ready", "sum")).reset_index()
    coverage["ready_fraction"] = coverage.ready_days / coverage.requested_days
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path="inputs", plan=plan, calendar_binding=plan["calendar"], source_inventory=json.loads((card / "source_inventory.json").read_text()))
        rec.save_objects(artifact_path="source", parsed_exact=parsed, panel_float64=panel, dataset=dataset,
            daily_receipts=receipts, monthly_metadata=metadata, official_documents=documents,
            updated_files=updates, revision_inventory=revisions, failures=errors, HTTP_state=client.state,
            provider_manifest=manifest, quantization=pd.DataFrame(quantization))
        rec.save_objects(local_path=str(bundle), artifact_path="source")
        rec.save_objects(artifact_path="sources", **{name.replace("/", "__"): (root / name).read_text() for name in CODE_FILES})
        summary = {"experiment": NAME, "stage": "source_intake_completed", "plan_sha256": plan_sha,
            "decision": ("source_snapshot_ready_for_separate_model_design" if revision_complete else "raw_snapshot_ready_revision_metadata_partial") if source_ready else "needs_data_incomplete_source_snapshot",
            "source_snapshot_complete": bool(source_ready and revision_complete),
            "raw_daily_complete": raw_daily_complete, "revision_metadata_complete": revision_complete,
            "original_vintage_complete": False, "requested_daily_files": 2059, "valid_daily_files": 2059 - len(errors),
            "valid_month_indexes": sum(month["valid"] for month in metadata.values()),
            "valid_official_documents": valid_documents,
            "revision_metadata_unknown_days": revision_unknown,
            "explicit_updated_urls": len(updates), "actual_HTTP_attempts": len(client.state["attempts"]),
            "HTTP_hard_limit": client.maximum, "new_model_fits": 0, "return_labels_computed": 0,
            "new_accounts": 0, "historical_PIT_qualified": False, "independent_forward_qualified": False,
            "objective_achieved": False, "recorder_id": rec.id, "experiment_id": str(rec.experiment_id),
            "recorder_uri": str(tracking), "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id),
            "provider_uri": str(provider), "source_bundle_sha256": sha(bundle),
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": plan["artifacts"]["recorder_only"]}}
        rec.save_objects(artifact_path="analysis", summary=summary, coverage=coverage)
        artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
        rec.log_metrics(registered_artifact_count=sum(path.is_file() for path in artifacts.rglob("*")),
                        valid_source_days=summary["valid_daily_files"], HTTP_requests=summary["actual_HTTP_attempts"])
    atomic_json({"recorder_id": rec.id, "entries": [{"path": str(path.relative_to(card)), "sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted(artifacts.rglob("*")) if path.is_file()]}, card / "verification_manifest.json")
    binding = json.loads((root / "research/minute_information_pit_allocator_20261012/nas_binding.json").read_text())
    binding.pop("evidence", None)
    # This is a standalone source archive, never a replacement of sync.current.
    binding["source_id"] = "finra-cnms-20180801-20261009-" + summary["source_bundle_sha256"]
    atomic_json(binding, card / "nas_binding.json")
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=rec.id, experiment_id=str(rec.experiment_id))
    coverage.to_csv(card / "coverage_by_year.csv", index=False)
    pd.DataFrame(quantization).to_csv(card / "source_quantization.csv", index=False)
    atomic_json(summary, card / "summary.json")
    atomic_json({"stage": "completed", "plan_sha256": plan_sha, "actual_HTTP_attempts": summary["actual_HTTP_attempts"],
                 "recorder_id": rec.id, "model_fits": 0, "accounts": 0}, card / "source_run_state.json")
    (card / "conclusion.md").write_text("# FINRA来源资格结果\n\n" +
        f"固定2059个Qlib日期，合法文件{summary['valid_daily_files']}个，月索引验证{summary['valid_month_indexes']}/99；实际HTTP请求{summary['actual_HTTP_attempts']}/{client.maximum}。" +
        f"修订元数据未知{summary['revision_metadata_unknown_days']}日，不能解释为没有修订。\n\n" +
        "原始字节和精确decimal字符串为NAS来源权威；Qlibfloat32仅为派生缓存，量化误差已单列。ShortVolume包含ShortExemptVolume，不等于shortinterest、净库存、情绪或作者operationalshorting指标。\n\n" +
        "历史真实首次接收与最初vintage未认证。官方18:00发布上界只能支持另行声明的滞后研究假设，不能授予PIT或前向资格。\n\n" +
        f"决定：{summary['decision']}。0标签、0拟合、0账户，生产未改，稳定超额目标未完成。下一阶段必须另行冻结并评审model_plan，不自动启动训练。Recorder：`{rec.id}`。\n")
    return summary
