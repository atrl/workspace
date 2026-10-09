"""Offline ETF event-source qualification with immutable Qlib evidence.

This module neither downloads data nor builds prices or runs an investment
strategy. The plan supplies all event, precision and calendar parameters.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.core.recorder_sources import _relative, _sha, load_verified_recorder_objects


QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def assert_run_available(output_card):
    """Call before any qlib.init, including the notebook's explicit init."""
    _require(not (Path(output_card) / "summary.json").exists(),
             "completed card cannot be overwritten")
    from qlib.workflow import R
    try:
        active = R.exp_manager.active_experiment
    except AttributeError:
        return
    _require(active is None or active.active_recorder is None,
             "active writable Recorder must finish before qlib.init")


def read_frozen_plan(plan_path, *, expected_plan_sha256):
    payload = Path(plan_path).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == expected_plan_sha256,
             "plan SHA-256 differs from the reviewed notebook contract")
    plan = json.loads(payload)
    _require(plan["plan_state"] == "registered_after_root_review_before_qualification",
             "plan has not passed root review")
    _require(plan["source_archive"]["network_in_notebook"] is False,
             "qualification must remain offline")
    return plan


def _archive_inventory(directory):
    root = Path(directory)
    _require(root.is_dir() and not root.is_symlink(), "archive is not an independent directory")
    entries = {}
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "source archive cannot contain symbolic links")
        if path.is_file():
            entries[path.relative_to(root).as_posix()] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    return entries


def copy_immutable_sources(source_archive, destination):
    """Validate the whole frozen inventory before copying; never replace it."""
    destination = Path(destination)
    expected = {}
    for entry in source_archive["files"]:
        name = str(_relative(entry["archive_relative_path"], "archive source path"))
        _require(name not in expected, "duplicate frozen archive path")
        expected[name] = {"sha256": entry["sha256"], "bytes": entry["bytes"]}
    _require(bool(expected), "empty frozen source inventory")
    if destination.exists():
        _require(_archive_inventory(destination) == expected, "existing immutable source archive differs")
        return {"reused": True, "files": expected, "file_count": len(expected)}
    for entry in source_archive["files"]:
        source = Path(entry["staged_input"])
        _require(source.is_file() and not source.is_symlink(), "staged source is missing or symbolic")
        _require(source.stat().st_size == entry["bytes"] and _sha(source) == entry["sha256"],
                 "staged source hash or byte count differs: " + entry["archive_relative_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".event-sources-", dir=destination.parent))
    try:
        for entry in source_archive["files"]:
            target = temporary / entry["archive_relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry["staged_input"], target)
        _require(_archive_inventory(temporary) == expected, "source copy differs from frozen bytes")
        _require(not destination.exists(), "source destination appeared during copy")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"reused": False, "files": expected, "file_count": len(expected)}


def audit_source_receipts(plan, archive):
    """Retain failed requests and unknown timestamps without upgrading them."""
    entries, receipts = plan["source_archive"]["files"], []
    by_digest = {}
    for entry in entries:
        by_digest.setdefault(entry["sha256"], []).append(entry)
    successful_bodies = set()
    for entry in entries:
        if "receipt" not in entry["capture_kind"]:
            continue
        receipt = json.loads((Path(archive) / entry["archive_relative_path"]).read_bytes())
        status = receipt.get("status_code", receipt.get("status"))
        success = isinstance(status, int) and 200 <= status < 300
        if entry["capture_kind"] == "failed_http_receipt":
            _require(isinstance(status, int) and status >= 400, "failed receipt has a successful status")
        if entry["capture_kind"] in ("http_receipt", "http_receipt_partial"):
            _require(success, "valid source receipt did not succeed")
        digest = receipt.get("body_sha256", receipt.get("sha256"))
        body_entries = []
        if digest is not None:
            body_entries = by_digest.get(digest, [])
            _require(bool(body_entries), "receipt body digest is absent from frozen archive")
            count = receipt.get("body_bytes", receipt.get("bytes"))
            _require(all(item["bytes"] == count for item in body_entries), "receipt byte count differs")
            if success:
                _require(all(not item["capture_kind"].startswith("failed_") for item in body_entries),
                         "successful receipt refers to a failed body")
                successful_bodies.add(digest)
        else:
            _require(not success and bool(receipt.get("error")), "receipt has no retained body identity")
        receipts.append({
            "archive_relative_path": entry["archive_relative_path"], "status_code": status,
            "valid_response_body": success and digest is not None,
            "body_sha256": digest, "body_paths": [item["archive_relative_path"] for item in body_entries],
            "request_started_at_utc": receipt.get("request_started_at_utc", receipt.get("started_utc")),
            "response_received_at_utc": receipt.get("response_received_at_utc", receipt.get("completed_utc",
                                              receipt.get("retrieved_at_utc"))),
            "error": receipt.get("error"),
        })
    for entry in entries:
        if entry["capture_kind"] in ("http_response_body", "archived_original_issuer_http_response_body"):
            _require(entry["sha256"] in successful_bodies, "valid body lacks a successful frozen receipt")
    return {"receipts": receipts, "source_versions": plan["source_versions"],
            "dom_request_started_at_utc": None, "dom_exact_response_received_at_utc": None,
            "failed_body_count": sum(x["capture_kind"] == "failed_http_response_body" for x in entries),
            "receipt_count": len(receipts), "failed_receipts_retained": True}


def compare_extraction_reference(parsed, reference_bytes, *, source_fields):
    """The frozen CSV is a check, never a replacement for issuer decimals."""
    reference = pd.read_csv(io.BytesIO(reference_bytes), dtype=str, keep_default_na=False)
    _require(list(parsed.columns) == list(reference.columns) == list(source_fields),
             "extraction reference schema differs")
    _require(len(parsed) == len(reference), "extraction reference row count differs")
    date_fields = {"Record Date", "Ex-Date", "Payable Date"}
    for row, (actual, expected) in enumerate(zip(parsed.to_dict("records"), reference.to_dict("records"))):
        for field in source_fields:
            left, right = actual[field], expected[field]
            _require(isinstance(left, str) and isinstance(right, str), "source tokens must remain strings")
            if field in date_fields:
                same = pd.Timestamp(left).date() == pd.Timestamp(right).date()
            else:
                try:
                    lhs, rhs = Decimal(left), Decimal(right)
                    same = lhs.is_finite() and rhs.is_finite() and lhs == rhs
                except InvalidOperation:
                    same = False
            _require(same, f"extraction reference differs at row {row}, field {field}")
    return {"rows": len(parsed), "ordered_rows_equal": True, "date_equivalence": "calendar_date",
            "amount_equivalence": "Decimal_exact", "source_strings_preserved": True}


def load_parent_bindings(plan, evidence_root):
    """Explicitly read each new/legacy manifest without rewriting either card."""
    objects, audits = {}, {}
    for name, contract in plan["parent_references"].items():
        card = Path(evidence_root) / _relative(contract["card"], "parent card")
        _require(_sha(card / "summary.json") == contract["summary_sha256"], "parent summary hash differs")
        manifest_hash = contract.get("manifest_sha256", contract.get("verification_manifest_sha256"))
        loaded, audit = load_verified_recorder_objects(
            card, expected_recorder_id=contract["recorder_id"],
            artifacts=list(contract["artifact_sha256"]),
            expected_artifact_sha256=contract["artifact_sha256"],
            manifest_name=contract["manifest_name"], expected_manifest_sha256=manifest_hash)
        _require(audit["summary_sha256"] == contract["summary_sha256"]
                 and audit["experiment_id"] == str(contract["experiment_id"])
                 and Path(audit["recorder_uri"]).resolve() == Path(contract["recorder_uri"]).resolve(),
                 "parent Recorder identity or summary differs")
        objects[name], audits[name] = loaded, audit
    return objects, audits


def verify_registered_event_examples(cash, splits, plan, historical):
    """Bind canonical outputs back to the reviewed terminal and source facts."""
    expected = plan["time_contract"]["terminal_unpaid_example"]
    matching = cash.loc[cash.instrument.eq(expected["instrument"]) & cash.ex_date.eq(expected["ex_date"])]
    _require(len(matching) == 1, "terminal unpaid event is missing or duplicated")
    row = matching.iloc[0]
    _require(expected["market_as_of"] == plan["market_as_of"]
             and row["pay_date"] == expected["pay_date"] > plan["market_as_of"]
             and Decimal(row["original_total_decimal"]) == Decimal(expected["amount_per_original_share_decimal"])
             and row["payable_status_at_market_as_of"] == "unpaid_receivable_not_spendable",
             "terminal event amount, timing or receivable classification differs")
    event = cash.loc[cash.instrument.eq("SOXL") & cash.ex_date.eq(historical["economic_ex_date"])]
    _require(len(event) == 1, "historical issuer canonical event is missing or duplicated")
    row = event.iloc[0]
    _require(row["record_date"] == historical["record_date"] and row["pay_date"] == historical["pay_date"]
             and Decimal(row["original_total_decimal"]) == Decimal(historical["original_share_rate_decimal"]),
             "canonical historical event differs from archived original rate or dates")
    fields = ["instrument", "record_date", "payable_after_close_date", "effective_trading_date"]
    def identity(item):
        return (*[str(item[key]) for key in fields], Decimal(str(item["new_shares_per_old_share"])))
    _require(len(splits) == len(plan["split_events"])
             and sorted(map(identity, splits.to_dict("records"))) == sorted(map(identity, plan["split_events"])),
             "split count, ratio or effective dates differ from the frozen plan")
    return {"terminal_unpaid_example_verified": True, "historical_canonical_event_verified": True,
            "all_declared_split_identities_verified": True}


def _source_identity(source_root):
    result = {}
    for module_name in ("src.workflow.etf_corporate_actions", "src.data.etf_corporate_actions",
                        "src.core.recorder_sources"):
        module = importlib.import_module(module_name)
        relative = module_name.replace(".", "/") + ".py"
        expected = Path(source_root) / relative
        _require(Path(module.__file__).resolve() == expected.resolve(), "imported implementation belongs to another source root")
        result[relative] = _sha(expected)
    return result


def _assert_bound_files(plan_path, plan_hash, source_root, source_hashes, archive, inventory, parents):
    _require(_sha(plan_path) == plan_hash, "plan changed during qualification")
    _require(all(_sha(Path(source_root) / name) == digest for name, digest in source_hashes.items()),
             "implementation changed during qualification")
    _require(_archive_inventory(archive) == inventory, "immutable source archive changed during qualification")
    for audit in parents.values():
        card = Path(audit["recorder_uri"]).parent
        manifest = audit.get("manifest_name", "verification_manifest.json")
        digest = audit.get("manifest_sha256", audit.get("verification_manifest_sha256"))
        _require(_sha(card / "summary.json") == audit["summary_sha256"] and _sha(card / manifest) == digest,
                 "parent metadata changed during qualification")
        _require(all(_sha(item["local_path"]) == item["sha256"] for item in audit["artifacts"].values()),
                 "parent object changed during qualification")


def _atomic_publish(card, summary, manifest):
    """Summary is the completion marker and is the final atomic publication."""
    card = Path(card)
    _require(not (card / "summary.json").exists(), "completed card appeared during archival")
    for name, value in (("verification_manifest.json", manifest), ("summary.json", summary)):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=card, prefix="." + name + "-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(_json_bytes(value))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, card / name)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def archive_and_publish_completion(card, summary, *, recorder_context, recorder_loader,
                                   write_artifacts, verify_bound_inputs):
    """No completion marker survives a failed archive, exit, or readback."""
    card = Path(card)
    _require(not (card / "summary.json").exists(), "completed card cannot be overwritten")
    with recorder_context:
        recorder = recorder_loader()
        summary.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id),
                       recorder_local_dir=recorder.get_local_dir())
        write_artifacts(recorder, summary)
        # Qlib's automatic working-tree dumps are not the scoped source record.
        artifact_root = Path(recorder.get_local_dir()) / "artifacts"
        for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
            (artifact_root / name).unlink(missing_ok=True)
        inventory = _archive_inventory(artifact_root)
        recorder.save_objects(artifact_path="validation", **{"artifact_hashes.pkl": inventory})
        summary["qlib_artifacts"] = sorted([*inventory, "validation/artifact_hashes.pkl", "card/summary.json"])
        with tempfile.TemporaryDirectory(prefix="event-summary-") as directory:
            staged = Path(directory) / "summary.json"
            staged.write_bytes(_json_bytes(summary))
            recorder.log_artifact(str(staged), artifact_path="card")
        final_inventory = _archive_inventory(artifact_root)
    info = recorder.client.get_run(recorder.id).info
    _require(info.status == "FINISHED" and info.run_id == summary["recorder_id"],
             "Recorder did not finish successfully")
    _require(_archive_inventory(artifact_root) == final_inventory, "Recorder bytes changed after completion")
    _require(recorder.load_object("validation/artifact_hashes.pkl") == inventory,
             "Recorder artifact-hash readback differs")
    _require(all(final_inventory[name] == identity for name, identity in inventory.items()),
             "Recorder artifact changed during final archival")
    verify_bound_inputs()
    entries = [{"path": (artifact_root / name).relative_to(card).as_posix(), **identity,
                "role": "finished_recorder_evidence"} for name, identity in sorted(final_inventory.items())]
    for name in ("plan.json", "workflow.yaml", "qualification.json", "source_differences_summary.json", "conclusion.md"):
        path = card / name
        entries.append({"path": name, "sha256": _sha(path), "bytes": path.stat().st_size, "role": "slim_card"})
    encoded = _json_bytes(summary)
    entries.append({"path": "summary.json", "sha256": hashlib.sha256(encoded).hexdigest(),
                    "bytes": len(encoded), "role": "completion_marker"})
    manifest = {"schema_version": "etf_event_verification_manifest_v1", "recorder_id": recorder.id,
                "entries": entries, "notebook_final_output_hash": "added_after_notebook_execution",
                "recorder_status": "FINISHED", "all_recorded_artifact_hashes_read_back": True}
    _atomic_publish(card, summary, manifest)
    return summary


def run_etf_corporate_actions_qualification(plan_path, *, output_card, source_root, evidence_root,
                                           expected_plan_sha256):
    """The notebook calls this once after the reviewed source inventory exists."""
    assert_run_available(output_card)
    plan = read_frozen_plan(plan_path, expected_plan_sha256=expected_plan_sha256)
    source_root, evidence_root, card = map(lambda value: Path(value).resolve(), (source_root, evidence_root, output_card))
    _require(card.is_dir() and Path(plan_path).resolve() == card / "plan.json", "plan must be card-local")
    source_hashes = _source_identity(source_root)
    from src.data.etf_corporate_actions import (
        parse_ishares_distribution_workbook, parse_direxion_distribution_table,
        qualify_etf_corporate_actions, compare_futu_cash_events, verify_historical_2014_source,
    )
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    calendar_contract = plan["calendar_reference"]
    provider = evidence_root / _relative(calendar_contract["provider_uri"], "calendar provider")
    qlib.init(provider_uri=str(provider), region=calendar_contract["region"], kernels=1)
    parents, parent_audits = load_parent_bindings(plan, evidence_root)
    calendar_file = provider / "calendars" / (calendar_contract["freq"] + ".txt")
    _require(_sha(calendar_file) == plan["parent_references"]["native_calendar"]["calendar_file_sha256"],
             "native calendar bytes differ from parent contract")
    native_calendar = pd.DatetimeIndex(D.calendar(freq=calendar_contract["freq"], future=False))
    _require(len(native_calendar) == calendar_contract["expected_rows"]
             and native_calendar.is_monotonic_increasing and not native_calendar.has_duplicates
             and str(native_calendar[0].date()) == calendar_contract["start"]
             and str(native_calendar[-1].date()) == calendar_contract["end"], "native calendar range or count differs")
    readback = parents["native_calendar"]["data/native_readback.pkl"]
    _require(readback["rows"] == len(native_calendar)
             and readback["provider_file_sha256"]["calendars/day.txt"] == _sha(calendar_file),
             "parent calendar readback differs")
    provider_hashes = _archive_inventory(provider)
    archive = source_root / _relative(plan["source_archive"]["destination"], "source archive destination")
    source_copy = copy_immutable_sources(plan["source_archive"], archive)
    receipts = audit_source_receipts(plan, archive)
    rows = {
        "SOXX": parse_ishares_distribution_workbook((archive / "soxx/issuer_fund_download.bin").read_bytes()),
        "SOXL": parse_direxion_distribution_table((archive / plan["sources"]["SOXL"]["source_file"]).read_bytes()),
    }
    reference = compare_extraction_reference(rows["SOXX"],
        (archive / "soxx/issuer_distributions_extracted.csv").read_bytes(),
        source_fields=plan["sources"]["SOXX"]["source_fields"])
    historical = verify_historical_2014_source(
        (archive / "soxl/independent_2014/issuer_distributions_20141224.html").read_bytes(),
        (archive / "soxl/independent_2014/direxioninvestments.com_cdx.response").read_bytes(),
        contract=plan["sources"]["SOXL"]["historical2014_source"])
    historical_rows = rows["SOXL"].loc[rows["SOXL"]["Ex Date"].eq(historical["economic_ex_date"])]
    _require(len(historical_rows) == 1, "archived issuer event is absent or duplicated in current table")
    historical_row = historical_rows.iloc[0]
    _require(historical_row["Record Date"] == historical["record_date"]
             and historical_row["Pay Date"] == historical["pay_date"]
             and Decimal(historical_row["Income Dividend"]) == Decimal(historical["original_share_rate_decimal"]),
             "archived issuer rate or dates differ from the frozen current issuer row")
    cash, splits, qualification = qualify_etf_corporate_actions(rows, plan["split_events"],
        native_calendar=native_calendar, market_as_of=plan["market_as_of"], source_contract=plan)
    examples = verify_registered_event_examples(cash, splits, plan, historical)
    frozen = parents["futu_comparison_only"]["inputs/verified_raw_inputs.pkl"]
    differences, difference_audit = compare_futu_cash_events(cash, frozen["none"]["rehabilitation"], source_contract=plan)
    qualification = {**qualification, "extraction_reference": reference,
                     "registered_event_examples": examples, "historical_2014_source": historical, "receipt_summary": {
                         key: receipts[key] for key in ("receipt_count", "failed_body_count", "failed_receipts_retained")}}
    binding = {"plan_sha256": expected_plan_sha256, "source_sha256": source_hashes,
               "parent_bindings": parent_audits, "source_archive": source_copy,
               "provider_file_identity": provider_hashes}
    summary = {"experiment": plan["experiment"], "record_type": "ETF_event_source_qualification",
        "provider_uri": str(provider), "market": "all", "region": calendar_contract["region"],
        "as_of": plan["market_as_of"], "data_range": [calendar_contract["start"], calendar_contract["end"]],
        "native_calendar_rows": len(native_calendar), "cash_event_rows": len(cash), "split_event_rows": len(splits),
        "source_file_count": source_copy["file_count"], "source_archive": plan["source_archive"]["destination"],
        "qualification_pass": True, "decision": plan["decision_semantics"]["success"],
        "plan_sha256": expected_plan_sha256, "source_sha256": source_hashes,
        "recorder_uri": str(card / "qlib_recorder"), "model_selection": {"performed": False},
        "model_fits": 0, "strategy_replays": 0, "return_evaluations": 0, "broker_api_calls": 0,
        "price_provider_changed": False, "promotion_eligible": False, "historical_vintage_certified": False,
        "benchmark": "not_evaluated_data_qualification_only", "artifact_policy": plan["artifact_policy"],
        "sidecars": ["qualification.json", "source_differences_summary.json", "conclusion.md", "workflow.yaml"],
        "qlib_docs_reference": QLIB_DOCS, "limitations": plan["decision_semantics"]["never_claim"]}
    def verify_bound_inputs():
        _assert_bound_files(plan_path, expected_plan_sha256, source_root, source_hashes,
                            archive, source_copy["files"], parent_audits)
        _require(_archive_inventory(provider) == provider_hashes, "calendar provider changed during qualification")
    def write_artifacts(recorder, current_summary):
        recorder.save_objects(artifact_path="inputs", **{"source_contract.pkl": plan, "parent_bindings.pkl": binding,
                                                         "native_calendar.pkl": native_calendar})
        recorder.save_objects(artifact_path="data", **{"source_rows.pkl": rows, "cash_events.pkl": cash,
                                                       "split_events.pkl": splits})
        recorder.save_objects(artifact_path="analysis", **{"source_differences.pkl": differences,
            "qualification.pkl": qualification, "source_difference_audit.pkl": difference_audit})
        recorder.save_objects(artifact_path="validation", **{"source_receipts.pkl": receipts,
            "parent_readback.pkl": parents["native_calendar"]})
        for entry in plan["source_archive"]["files"]:
            relative = Path(entry["archive_relative_path"])
            recorder.log_artifact(str(archive / relative), artifact_path="sources/" + relative.parent.as_posix())
        for name in source_hashes:
            recorder.log_artifact(str(source_root / name), artifact_path="source/" + Path(name).parent.as_posix())
        recorder.log_params(plan_sha256=expected_plan_sha256, market_as_of=plan["market_as_of"],
                            task_kind=plan["task_kind"], historical_vintage_certified=False)
        recorder.log_metrics(cash_event_rows=len(cash), split_event_rows=len(splits), native_calendar_rows=len(native_calendar))
        (card / "qualification.json").write_bytes(_json_bytes(qualification))
        (card / "source_differences_summary.json").write_bytes(_json_bytes(difference_audit))
        conclusion = ("# ETF 公司行动数据资格\n\n"
            f"已离线核验 {len(cash)} 条现金事件、{len(splits)} 条拆股及 {len(native_calendar)} 日原生 Qlib 参考日历；"
            f"原始来源 {source_copy['file_count']} 文件逐字节封存。\n\n"
            "SOXX 较早事件保留为参考日历外，零金额事件保留；窗口差异属于数据覆盖，不是收益表现。"
            "SOXX 2017 漏项、SOXL 2014 份额冲突与 2019 精度差保留在 Recorder。"
            f"发行人存档直接证明 {historical['economic_ex_date']} 每原始份额 {historical['original_share_rate_decimal']}，"
            "但抓取晚于除息日，不能证明事前获知。"
            "期末未支付分红仍是应收，不能当可支出现金。\n\n"
            "本卡未拟合、未回测、未改变旧价格，无晋升或稳定超额结论。"
            "下一步须另卡登记真实份额、应收及支付现金账本，保留到账时点与税费未知项。\n\n"
            f"Recorder：{recorder.id}。\n")
        (card / "conclusion.md").write_text(conclusion)
        for name in ("plan.json", "workflow.yaml", "qualification.json", "source_differences_summary.json", "conclusion.md"):
            recorder.log_artifact(str(card / name), artifact_path="card")
        verify_bound_inputs()
    return archive_and_publish_completion(card, summary,
        recorder_context=R.start(experiment_name=plan["experiment"], uri=str(card / "qlib_recorder")),
        recorder_loader=R.get_recorder, write_artifacts=write_artifacts, verify_bound_inputs=verify_bound_inputs)
