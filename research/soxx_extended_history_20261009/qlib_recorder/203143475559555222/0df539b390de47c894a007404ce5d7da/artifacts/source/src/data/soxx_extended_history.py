"""Explicit SOXX-only capture and immutable Qlib source qualification.

Capture is a separate caller action. Notebook qualification is offline and
never starts OpenD, authenticates, subscribes, or changes an existing provider.
The older dual-instrument qualification helpers retain their sealed contract.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import logging.handlers
from pathlib import Path
import pickle
import socket

import numpy as np
import pandas as pd

from src.core.recorder_sources import _local, _relative, _sha
from src.data.price_basis import FIELDS, price_frame_sha256
from src.data.published_price_basis import build_published_basis, _events
from src.data.qlib_native import build_qlib_store_from_frame


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _tree_hashes(directory):
    root = Path(directory).resolve()
    _require(root.is_dir(), "source directory is absent")
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _capture_config(plan):
    config = plan["capture"]
    _require(config["code"] == "US.SOXX" and config["instrument"] == "SOXX"
             and config["autype"] == "NONE" and config["ktype"] == "K_DAY"
             and config["session"] == "RTH" and config["extended_time"] is False
             and config["max_count"] == 1000 and config["fields"] == "KL_FIELD.ALL",
             "capture must be the declared unadjusted SOXX daily contract")
    _require(config["requested_start"] <= config["requested_end"] == config["as_of"],
             "capture start/end/as_of differ")
    return config


@contextlib.contextmanager
def _futu_log_route(directory):
    """Route this process's SDK handlers before importing Futu; never change HOME."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = logging.handlers.TimedRotatingFileHandler
    def routed(filename, *args, **kwargs):
        if "com.futunn.FutuOpenD" in str(filename):
            filename = str(directory / Path(filename).name)
        return original(filename, *args, **kwargs)
    with (directory / "capture_output.log").open("a") as stream, \
            contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        logging.handlers.TimedRotatingFileHandler = routed
        try:
            yield
        finally:
            logging.handlers.TimedRotatingFileHandler = original


def capture_soxx_history(root, plan, *, quote_context_factory=None, sdk=None):
    """Explicitly archive one before/pages/after sequence; no implicit retry.

    Factory/SDK injection is for synthetic contract tests. Production uses only
    the already listening local endpoint. Each decoded original SDK response
    is retained with a truthful request receipt, including failed responses.
    """
    root, config = Path(root).resolve(), _capture_config(plan)
    destination = root / config["raw_dir"]
    _require(not destination.exists(), "capture destination already exists; do not overwrite receipts")
    destination.mkdir(parents=True)
    manifest = {"schema_version": "single_soxx_futu_capture_v1", "status": "running",
        "started_at_utc": _utc(), "request_contract": config, "read_only": True,
        "raw_responses": [], "sdk_logs_outside_repo": config["sdk_logs_dir"]}
    _write_json(destination / "manifest.json", manifest)
    context = None
    def request(name, method, params, **actual):
        before = _utc()
        response = method(**actual)
        after = _utc()
        target = destination / (name + ".pkl")
        target.write_bytes(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
        receipt = {"name": name, "params": params, "requested_at_utc": before,
            "received_at_utc": after, "ret": int(response[0]), "response_path": target.name,
            "response_sha256": _sha(target), "rows": len(response[1]) if isinstance(response[1], pd.DataFrame) else None}
        manifest["raw_responses"].append(receipt)
        _write_json(destination / "manifest.json", manifest)
        if response[0] != sdk.RET_OK:
            text = str(response[1]).lower()
            manifest["status"] = "needs auth" if any(term in text for term in
                ("login", "permission", "quota", "登录", "权限", "额度")) else "needs data"
            raise ValueError("historical request failed; no permission or login workaround")
        return response
    try:
        with _futu_log_route(config["sdk_logs_dir"]):
            if quote_context_factory is None:
                with socket.create_connection((config["host"], config["port"]), timeout=2):
                    pass
                import futu as sdk
                from futu.common.ft_logger import logger
                logger._console_level = logging.CRITICAL
                logger.console_logger.setLevel(logging.CRITICAL)
                sdk.SysConfig.set_all_thread_daemon(True)
                quote_context_factory = sdk.OpenQuoteContext
                manifest["sdk_version"] = importlib.metadata.version("futu-api")
            else:
                _require(sdk is not None, "injected capture requires an explicit synthetic SDK")
                manifest["sdk_version"] = "synthetic"
            context = quote_context_factory(host=config["host"], port=config["port"])
            request("SOXX_rehab_before", context.get_rehab, {"code": config["code"]}, code=config["code"])
            page, page_number = None, 0
            while True:
                page_number += 1
                _require(page_number <= config["max_pages"], "capture exceeded its fixed page limit")
                params = {key: config[key] for key in
                    ("code", "requested_start", "requested_end", "ktype", "autype", "fields", "session", "extended_time", "max_count")}
                params.update(page_number=page_number, has_prior_page_key=page is not None)
                result = request(f"SOXX_history_page_{page_number}", context.request_history_kline, params,
                    code=config["code"], start=config["requested_start"], end=config["requested_end"],
                    ktype=sdk.KLType.K_DAY, autype=sdk.AuType.NONE, fields=[sdk.KL_FIELD.ALL],
                    max_count=config["max_count"], page_req_key=page,
                    extended_time=False, session=sdk.Session.RTH)
                _require(len(result) == 3 and isinstance(result[1], pd.DataFrame), "invalid historical response tuple")
                page = result[2]
                if page is None:
                    break
            request("SOXX_rehab_after", context.get_rehab, {"code": config["code"]}, code=config["code"])
            manifest["status"] = "captured_unqualified"
    except Exception as error:
        if manifest["status"] == "running":
            manifest["status"] = "needs data"
        manifest["failure"] = {"exception_type": type(error).__name__,
            "message": "Capture incomplete; original receipts retained; no automatic retry or service changes."}
    finally:
        if context is not None:
            try:
                with _futu_log_route(config["sdk_logs_dir"]):
                    context.close()
            except Exception as error:
                manifest["close_error_type"] = type(error).__name__
                manifest["status"] = "needs data"
        manifest["finished_at_utc"] = _utc()
        _write_json(destination / "manifest.json", manifest)
    return manifest


def verify_soxx_capture(directory, config, calendar):
    """Offline receipt, data and calendar audit; do not sanitize source rows."""
    directory, calendar = Path(directory).resolve(), pd.DatetimeIndex(calendar)
    manifest = json.loads((directory / "manifest.json").read_text())
    _require(manifest["status"] == "captured_unqualified" and manifest["read_only"] is True
             and manifest["request_contract"] == config, "capture manifest/request contract differs or failed")
    receipts = manifest["raw_responses"]
    names = [r["name"] for r in receipts]
    page_count = len(receipts) - 2
    expected = ["SOXX_rehab_before", *[f"SOXX_history_page_{i}" for i in range(1, page_count + 1)], "SOXX_rehab_after"]
    _require(page_count >= 1 and page_count <= config["max_pages"] and names == expected,
             "missing, duplicated or reordered capture receipts")
    responses, last_received = [], None
    for receipt in receipts:
        filename = _relative(receipt["response_path"], "capture response")
        _require(len(filename.parts) == 1, "response path must be a simple filename")
        path = directory / str(filename)
        _require(not path.is_symlink() and _sha(path) == receipt["response_sha256"], "raw response identity differs")
        before, after = pd.Timestamp(receipt["requested_at_utc"]), pd.Timestamp(receipt["received_at_utc"])
        _require(before.tzinfo is not None and after.tzinfo is not None and before <= after
                 and (last_received is None or last_received <= before), "capture times do not bracket sequential requests")
        last_received = after
        response = pickle.loads(path.read_bytes())
        _require(receipt["ret"] == response[0] == 0 and isinstance(response[1], pd.DataFrame)
                 and receipt["rows"] == len(response[1]), "raw response status/type/count differs")
        responses.append(response)
    for receipt in (receipts[0], receipts[-1]):
        _require(receipt["params"] == {"code": "US.SOXX"}, "rehab request is not SOXX")
    _require(len(responses[0]) == len(responses[-1]) == 2, "rehab response requires the original pair")
    pd.testing.assert_frame_equal(responses[0][1], responses[-1][1], check_exact=True)
    _events(responses[0][1])
    chunks = []
    for i, (receipt, response) in enumerate(zip(receipts[1:-1], responses[1:-1]), 1):
        params = {key: config[key] for key in
            ("code", "requested_start", "requested_end", "ktype", "autype", "fields", "session", "extended_time", "max_count")}
        params.update(page_number=i, has_prior_page_key=i > 1)
        _require(receipt["params"] == params and len(response) == 3
                 and bool(response[2]) == (i < page_count), "pagination request/continuation differs")
        _require(0 < len(response[1]) <= config["max_count"] and response[1].code.eq("US.SOXX").all(),
                 "empty, oversized or wrong-symbol history page")
        chunks.append(response[1])
    raw = pd.concat(chunks, ignore_index=True)
    required = {"code", "time_key", *FIELDS}
    _require(required.issubset(raw.columns), "raw SOXX fields are incomplete")
    dates = pd.DatetimeIndex(pd.to_datetime(raw.time_key))
    values = raw[list(FIELDS)].to_numpy(float)
    _require(dates.tz is None and dates.equals(dates.normalize()) and not dates.has_duplicates
             and dates.is_monotonic_increasing, "duplicate/unordered/non-session raw dates")
    _require(len(raw) == config["expected_rows"] and dates.equals(calendar)
             and str(dates[0].date()) == config["requested_start"]
             and str(dates[-1].date()) == config["requested_end"], "full declared native calendar coverage differs")
    _require(np.isfinite(values).all() and (values > 0).all(), "nonpositive or nonfinite raw OHLCV")
    _require(((raw.low <= raw.open) & (raw.open <= raw.high)
              & (raw.low <= raw.close) & (raw.close <= raw.high)).all(), "nonphysical observed OHLC bars")
    identity = {"source": "Futu", "adjustment": "NONE", "session": "RTH", "extended_time": False,
        "timezone": "America/New_York", "volume_unit": "shares", "volume_basis": "raw_shares",
        "requested_start": config["requested_start"], "requested_end": config["requested_end"],
        "retrieved_at_utc": manifest["finished_at_utc"], "sdk_version": manifest["sdk_version"],
        "raw_artifacts": [{"name": item["response_path"], "sha256": item["response_sha256"]} for item in receipts],
        "rehabilitation": {"SOXX": {"before_sha256": receipts[0]["response_sha256"],
            "after_sha256": receipts[-1]["response_sha256"], "before_observed_at_utc": receipts[0]["received_at_utc"],
            "after_observed_at_utc": receipts[-1]["received_at_utc"]}}}
    return {"frame": raw, "rehab": responses[0][1], "identity": identity, "manifest": manifest,
        "audit": {"rows": len(raw), "pages": page_count, "raw_frame_sha256": price_frame_sha256(raw),
            "rehab_frame_sha256": price_frame_sha256(responses[0][1]), "manifest_sha256": _sha(directory / "manifest.json"),
            "invalid_rows_dropped": 0, "missing_rows_filled": 0, "read_only_capture": True}}


def load_parent_evidence(root, plan):
    """Read the parent's original artifact_manifest contract, without migrating it."""
    import qlib
    from qlib.workflow import R
    from qlib.data import D
    root, spec = Path(root).resolve(), plan["parent"]
    card = (root / spec["summary"]).parent
    summary_path, manifest_path = card / "summary.json", card / "artifact_manifest.json"
    summary, manifest = json.loads(summary_path.read_text()), json.loads(manifest_path.read_text())
    _require(summary["recorder_id"] == manifest["recorder_id"] == spec["recorder_id"], "parent Recorder identity differs")
    _require(summary["data_as_of"] == spec["as_of"] and summary["provider_uri"] == spec["provider_uri"]
             and summary.get("source_qualification_pass") is True, "parent source scope or qualification differs")
    uri, local = _local(summary["recorder_uri"], "parent URI"), _local(summary["recorder_local_dir"], "parent local dir")
    _require(uri == (root / spec["recorder_uri"]).resolve() and local.parent.name == spec["experiment_id"]
             and local.name == spec["recorder_id"] and local.is_relative_to(uri), "parent local path binding differs")
    identities = {}
    for name, digest in spec["artifact_sha256"].items():
        path = local / "artifacts" / str(_relative(name, "parent artifact"))
        _require(path.resolve().is_relative_to(local / "artifacts") and not path.is_symlink(), "parent artifact escaped")
        key = path.relative_to(root).as_posix()
        entries = [entry for entry in manifest["entries"] if entry["path"] == key]
        _require(len(entries) == 1 and entries[0]["sha256"] == digest == _sha(path)
                 and entries[0]["bytes"] == path.stat().st_size, "parent explicit/archive artifact SHA differs")
        identities[name] = {"sha256": digest, "path": str(path)}
    qlib.init(provider_uri=str(root / spec["provider_uri"]), region="us", kernels=1)
    native = D.features(["SOXX"], ["$" + field for field in FIELDS], start_time=spec["overlap_start"], end_time=spec["as_of"], freq="day")
    _require(R.exp_manager.active_experiment is None or R.exp_manager.active_experiment.active_recorder is None,
             "parent source load cannot occur during an active writable Recorder")
    with R.uri_context(str(uri)):
        recorder = R.get_recorder(recorder_id=spec["recorder_id"], experiment_id=spec["experiment_id"])
        _require(recorder.id == spec["recorder_id"] and recorder.status == "FINISHED"
                 and Path(recorder.get_local_dir()).resolve() == local, "returned parent Recorder differs or is unfinished")
        objects = {name: recorder.load_object(name) for name in identities}
    _require(all(_sha(item["path"]) == item["sha256"] for item in identities.values()), "parent changed while reading")
    raw = objects["inputs/verified_raw_inputs.pkl"]
    return {"objects": objects, "raw_none": raw["none"]["frame"], "rehab": raw["none"]["rehabilitation"],
        "qualified": objects["data/qualified_frame.pkl"], "native": native,
        "audit": {"recorder_id": recorder.id, "artifacts": identities, "read_only": True,
            "metadata_contract": "legacy artifact_manifest.json; incompatible with new verification_manifest loader; no synthetic migration",
            "summary_sha256": _sha(summary_path), "artifact_manifest_sha256": _sha(manifest_path),
            "provider_file_sha256": _tree_hashes(root / spec["provider_uri"]),
            "source_api": ["qlib.workflow.R.get_recorder", "Recorder.load_object", "qlib.data.D.features"]}}


def load_reference_calendar(root, plan):
    import qlib
    from qlib.data import D
    spec, root = plan["calendar"], Path(root).resolve()
    provider = root / spec["provider_uri"]
    _require(_sha(provider / "calendars/day.txt") == spec["calendar_file_sha256"]
             and _sha(provider / "qlib_store_manifest.json") == spec["store_manifest_sha256"], "reference calendar provider changed")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    dates = pd.DatetimeIndex(D.calendar(freq="day", future=False))
    dates = dates[(dates >= plan["capture"]["requested_start"]) & (dates <= spec["market_as_of"])]
    digest = hashlib.sha256("\n".join(dates.strftime("%Y-%m-%d")).encode()).hexdigest()
    _require(digest == spec["selected_dates_sha256"] and len(dates) == plan["capture"]["expected_rows"], "native calendar identity differs")
    return dates


def qualify_soxx_extension(capture, parent, calendar, plan):
    """Reuse the published operator, with a new explicit single-symbol audit."""
    raw, spec, as_of = capture["frame"], plan["parent"], plan["capture"]["as_of"]
    dates = pd.DatetimeIndex(pd.to_datetime(raw.time_key))
    _require(dates.equals(pd.DatetimeIndex(calendar)), "capture calendar differs before transform")
    overlap = dates >= pd.Timestamp(spec["overlap_start"])
    old = parent["raw_none"].loc[parent["raw_none"].code.eq("US.SOXX")].copy()
    old_dates = pd.DatetimeIndex(pd.to_datetime(old.time_key))
    _require(int(overlap.sum()) == spec["overlap_rows"] and dates[overlap].equals(old_dates)
             and np.array_equal(raw.loc[overlap, list(FIELDS)].to_numpy(float), old[list(FIELDS)].to_numpy(float)),
             "new NONE overlap differs from sealed parent bars")
    used = ["event_date", "forward_adj_factorA", "forward_adj_factorB", "exact_split_price_factor"]
    new_events, old_events = _events(capture["rehab"]), _events(parent["rehab"]["SOXX"])
    new_used = new_events.loc[new_events.event_date <= pd.Timestamp(as_of), used].reset_index(drop=True)
    old_used = old_events.loc[old_events.event_date <= pd.Timestamp(as_of), used].reset_index(drop=True)
    pd.testing.assert_frame_equal(new_used, old_used, check_exact=True)
    secondary = pd.DataFrame(columns=["instrument", "date", *FIELDS])
    secondary_identity = {"source": "Yahoo", "adjustment": "SPLIT_ONLY", "volume_unit": "shares",
        "volume_basis": "split_adjusted_shares", "role": "empty compatibility input; no Yahoo quotes fetched or used"}
    tables = {"SOXX": capture["rehab"], "SOXL": parent["rehab"]["SOXL"]}
    frame, transform = build_published_basis(raw, secondary, tables, capture["identity"], secondary_identity, as_of)
    _require(frame.instrument.eq("SOXX").all() and len(frame) == len(calendar)
             and frame.datetime.equals(pd.Series(calendar, name="datetime")), "single-asset transformed coverage differs")
    old_qualified = parent["qualified"].loc[parent["qualified"].instrument.eq("SOXX")]
    tail = frame.loc[frame.datetime >= pd.Timestamp(spec["overlap_start"])]
    _require(pd.DatetimeIndex(tail.datetime).equals(pd.DatetimeIndex(old_qualified.datetime))
             and np.array_equal(tail[list(FIELDS)].to_numpy(float), old_qualified[list(FIELDS)].to_numpy(float)),
             "published overlap changed from sealed parent")
    _require(np.isfinite(frame[list(FIELDS)].to_numpy(float)).all()
             and (frame[list(FIELDS)].to_numpy(float) > 0).all()
             and ((frame.low <= frame.open) & (frame.open <= frame.high)
                  & (frame.low <= frame.close) & (frame.close <= frame.high)).all(), "invalid transformed OHLCV")
    _require(int((frame.datetime < pd.Timestamp(spec["overlap_start"])).sum()) == plan["capture"]["expected_early_rows"],
             "early extension count differs")
    audit = {"qualification_pass": True, "instrument_universe": ["SOXX"], "as_of": as_of,
        "rows": len(frame), "early_rows": int((frame.datetime < pd.Timestamp(spec["overlap_start"])).sum()),
        "parent_overlap_rows": len(tail), "raw_overlap_exact": True, "published_overlap_exact": True,
        "formula_fields_match_parent": True, "source_frame_sha256": price_frame_sha256(frame),
        "new_rehab_frame_sha256": price_frame_sha256(capture["rehab"]),
        "unused_parent_SOXL_rehab_for_transform_schema_only": True, "secondary_rows": 0,
        "new_before_after_rehab_exact": True, "price_basis": "published_futu_affine_v1",
        "total_return_certified": False, "historical_PIT_certified": False,
        "model_or_portfolio_evidence": False, "transformation": transform}
    return frame, audit


def build_and_verify_soxx_store(frame, *, provider_uri, as_of, calendar, parent_native, overlap_start, qualification):
    import qlib
    from qlib.data import D
    destination = Path(provider_uri).resolve()
    _require(qualification.get("qualification_pass") is True
             and qualification["source_frame_sha256"] == price_frame_sha256(frame), "input qualification differs")
    existing = destination.exists()
    before = _tree_hashes(destination) if existing else None
    if existing:
        marker_path = destination / "soxx_extension_qualification.json"
        _require(marker_path.is_file(), "existing provider has no completed extension qualification")
        marker = json.loads(marker_path.read_text())
        _require(marker.get("qualification_pass") is True
                 and marker["source_frame_sha256"] == qualification["source_frame_sha256"]
                 and marker["as_of"] == as_of, "immutable provider belongs to another source identity")
    else:
        build_qlib_store_from_frame(frame, provider_uri=destination, as_of=as_of, market="all", region="us",
            field_map={field: field for field in FIELDS}, source="qualified_single_SOXX_published_affine_extension")
    qlib.init(provider_uri=str(destination), region="us", kernels=1)
    dates = pd.DatetimeIndex(D.calendar(freq="day", future=False))
    _require(dates.equals(pd.DatetimeIndex(calendar)) and D.list_instruments(D.instruments("all"), as_list=True) == ["SOXX"],
             "new native calendar/universe differs")
    native = D.features(["SOXX"], ["$" + field for field in FIELDS], start_time=dates[0], end_time=as_of, freq="day")
    expected = frame.set_index(["instrument", "datetime"])[list(FIELDS)].sort_index().astype(np.float32)
    expected.columns = ["$" + field for field in FIELDS]
    _require(native.index.equals(expected.index) and np.array_equal(native.to_numpy(), expected.to_numpy()),
             "native SOXX values differ from source float32 values")
    tail = native.loc[native.index.get_level_values("datetime") >= pd.Timestamp(overlap_start)]
    _require(tail.index.equals(parent_native.index) and tail.columns.equals(parent_native.columns)
             and np.array_equal(tail.to_numpy(), parent_native.to_numpy()), "2010+ native SOXX values changed")
    if existing:
        _require(_tree_hashes(destination) == before, "immutable reexecution modified provider files")
    else:
        _write_json(destination / "soxx_extension_qualification.json", qualification)
    return {"status": "passed", "rows": len(native), "fields": list(native.columns),
        "parent_native_overlap_rows": len(tail), "parent_native_overlap_bitwise_equal": True,
        "expected_float32_bitwise_equal": True, "immutable_provider_reused": existing,
        "source_api": ["qlib.init", "D.calendar", "D.instruments", "D.list_instruments", "D.features"],
        "provider_file_sha256": _tree_hashes(destination), "frame_sha256": price_frame_sha256(native)}


def run_soxx_history_qualification(root, card_dir):
    """Offline notebook entrypoint; capture must already have succeeded."""
    import qlib
    from qlib.workflow import R
    root, card = Path(root).resolve(), Path(card_dir).resolve()
    plan_path = card / "plan.json"
    plan, plan_hash = json.loads(plan_path.read_text()), _sha(plan_path)
    parent = load_parent_evidence(root, plan)
    calendar = load_reference_calendar(root, plan)
    capture = verify_soxx_capture(root / plan["capture"]["raw_dir"], _capture_config(plan), calendar)
    frame, qualification = qualify_soxx_extension(capture, parent, calendar, plan)
    uri = str((root / plan["outputs"]["recorder_uri"]).resolve())
    binding = {"plan_sha256": plan_hash, "parent": parent["audit"], "capture": capture["audit"],
        "source_sha256": {name: _sha(root / name) for name in ["src/data/soxx_extended_history.py",
            "src/data/published_price_basis.py", "src/data/price_basis.py", "src/data/qlib_native.py", "src/core/recorder_sources.py"]}}
    with R.start(experiment_name=plan["experiment"], uri=uri):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", **{"parent_bindings.pkl": binding, "verified_capture.pkl": capture})
        for name in binding["source_sha256"]:
            recorder.log_artifact(str(root / name), artifact_path="source/" + str(Path(name).parent))
        recorder_id, experiment_id = recorder.id, recorder.experiment_id
    try:
        readback = build_and_verify_soxx_store(frame, provider_uri=root / plan["outputs"]["provider_uri"],
            as_of=plan["capture"]["as_of"], calendar=calendar, parent_native=parent["native"],
            overlap_start=plan["parent"]["overlap_start"], qualification=qualification)
    except Exception:
        with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
            raise
    _require(_sha(plan_path) == plan_hash and all(_sha(root / name) == digest for name, digest in binding["source_sha256"].items()),
             "plan or implementation changed during qualification")
    _require(_tree_hashes(root / plan["parent"]["provider_uri"]) == parent["audit"]["provider_file_sha256"]
             and all(_sha(item["path"]) == item["sha256"] for item in parent["audit"]["artifacts"].values()),
             "parent evidence changed during qualification")
    summary = {"experiment": plan["experiment"], "record_type": "SOXX_only_data_qualification",
        "provider_uri": plan["outputs"]["provider_uri"], "market": "all", "region": "us",
        "as_of": plan["capture"]["as_of"], "data_range": [plan["capture"]["requested_start"], plan["capture"]["as_of"]],
        "rows": len(frame), "new_early_rows": qualification["early_rows"], "qualification_pass": True,
        "parent_overlap_rows": readback["parent_native_overlap_rows"], "parent_native_overlap_bitwise_equal": True,
        "recorder_id": recorder_id, "experiment_id": experiment_id, "recorder_uri": uri,
        "plan_sha256": plan_hash, "source_sha256": binding["source_sha256"], "model_selection": {"performed": False},
        "promotion_eligible": False, "objective_achieved": False, "campaign_status": "active_research",
        "decision": "qualified_underlying_history_no_investment_or_forward_evidence",
        "artifact_policy": plan["artifact_policy"], "qlib_docs_reference": plan["official_references"][:2],
        "limitations": plan["limitations"]}
    conclusion = ("# SOXX独立长历史资格\n\n"
        f"已用原生Qlib建立{len(frame)}日单SOXX数据，其中新增{qualification['early_rows']}日早期观察。"
        f"原有{readback['parent_native_overlap_rows']}日SOXX native OHLCV逐位不变。\n\n"
        "未拟合、未回测、未生成SOXL早期数据。2001年至2006-10-09仍缺数据；回溯affine不是总收益或历史PIT认证。"
        "数据资格不构成稳定超额、晋升或前向证据。下一步须单独登记underlying-only特征研究。\n\n"
        f"Recorder：{recorder_id}。\n")
    with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
        recorder = R.get_recorder()
        _require(recorder.id == recorder_id, "resumed Recorder identity differs")
        summary["recorder_local_dir"] = recorder.get_local_dir()
        recorder.save_objects(artifact_path="data", **{"transformed_frame.pkl": frame,
            "qualification.pkl": qualification, "native_readback.pkl": readback})
        recorder.log_metrics(rows=len(frame), added_early_rows=qualification["early_rows"], native_overlap_exact=1.)
        _write_json(card / "summary.json", summary)
        _write_json(card / "qualification.json", qualification)
        _write_json(card / "native_readback.json", readback)
        _write_json(card / "run_binding.json", binding)
        (card / "conclusion.md").write_text(conclusion)
        for name in ["plan.json", "summary.json", "qualification.json", "native_readback.json", "run_binding.json", "conclusion.md", "workflow.yaml"]:
            recorder.log_artifact(str(card / name), artifact_path="card")
    for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
        (Path(recorder.get_local_dir()) / "artifacts" / name).unlink(missing_ok=True)
    return summary
