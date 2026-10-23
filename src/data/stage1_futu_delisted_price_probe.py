"""Bounded Futu transport probe for three old SOXX constituent symbols."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(root: Path) -> dict:
    import futu
    from futu import AuType, KLType, OpenQuoteContext, RET_OK

    from src.data.nas_market_data import _settings, _ssh, nas_api_call

    root = Path(root).resolve()
    card = root / "research/stage1_futu_delisted_price_probe_20261023"
    plan = json.loads((card / "plan.json").read_text())
    assert plan["frozen"] and len(plan["request"]["fixed_probes"]) == 3
    raw_dir = root / "data/raw/external/stage1_futu_delisted_price_probe_20261023"
    receipt_path = raw_dir / "request_receipt.json"
    if receipt_path.exists() or (card / "summary.json").exists():
        raise ValueError("fixed Futu delisted-symbol probe already attempted")
    raw_dir.mkdir(parents=True, exist_ok=True)
    request = plan["request"]
    receipt = {"status": "attempted", "plan_sha256": _sha((card / "plan.json").read_bytes()),
               "SDK_version": futu.__version__,
               "request_common": {key: request[key] for key in ("api", "host", "port", "ktype", "autype", "max_count")},
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "probes": []}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    raw_files = []
    quote = OpenQuoteContext(host=request["host"], port=request["port"])
    try:
        for index, item in enumerate(request["fixed_probes"], start=1):
            ret, data, next_key = quote.request_history_kline(
                item["code"], start=item["start"], end=item["end"],
                ktype=KLType.K_DAY, autype=AuType.NONE,
                max_count=request["max_count"], page_req_key=None)
            if ret == RET_OK and isinstance(data, pd.DataFrame):
                payload = data.to_csv(index=False, lineterminator="\n").encode()
                rows = len(data)
                first = str(pd.to_datetime(data.time_key).min().date()) if rows else None
                last = str(pd.to_datetime(data.time_key).max().date()) if rows else None
                extension = "csv"
            else:
                payload = str(data).encode()
                rows, first, last, extension = 0, None, None, "txt"
            path = raw_dir / (item["code"].removeprefix("US.") + "_response." + extension)
            path.write_bytes(payload)
            raw_files.append(path)
            receipt["probes"].append({"ordinal": index, "code": item["code"],
                "start": item["start"], "end": item["end"], "role": item["role"],
                "ret_code": int(ret), "rows": rows, "first_date": first, "last_date": last,
                "continuation": next_key is not None,
                "response_file": str(path.relative_to(root)), "response_sha256": _sha(payload),
                "response_bytes": len(payload)})
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    finally:
        quote.close()
    receipt["status"] = "captured"
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    members = [*raw_files, receipt_path]
    identities = [{"path": str(path.relative_to(root)), "sha256": _sha(path.read_bytes()),
                   "bytes": path.stat().st_size} for path in members]
    settings = _settings(root)
    with tempfile.TemporaryDirectory(prefix="futu-delisted-source-") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for path in members:
                handle.add(path, arcname=str(path.relative_to(root)))
        digest = _sha(archive.read_bytes())
        response = nas_api_call(origin=settings["origin"], token=settings["token"],
            method="PUT", path="/api/research/artifacts/" + digest,
            upload=archive, executor_id="soxl-minute-research")
        evidence = response["artifact"]
        if evidence["sha256"] != digest or evidence["size_bytes"] != archive.stat().st_size:
            raise ValueError("NAS delisted source receipt mismatch")
        remote = settings["artifact_root"].rstrip("/") + "/" + digest[:2] + "/" + digest[2:4] + "/" + digest
        if _ssh(settings["ssh"], "sha256sum " + shlex.quote(remote)).split()[0] != digest:
            raise ValueError("NAS delisted source remote hash mismatch")
        restored = Path(temporary) / "restored.tar.gz"
        with restored.open("wb") as target:
            process = subprocess.run(["ssh", "-p", str(settings["ssh"]["port"]),
                "-o", "BatchMode=yes", settings["ssh"]["host"],
                "cat " + shlex.quote(remote)], stdout=target,
                stderr=subprocess.PIPE, timeout=300)
        if process.returncode or _sha(restored.read_bytes()) != digest:
            raise ValueError("NAS delisted source archive restore failed")
        with tarfile.open(restored, "r:gz") as handle:
            actual = {item.name: handle.extractfile(item).read()
                      for item in handle.getmembers() if item.isfile()}
        for item in identities:
            if item["path"] not in actual or _sha(actual[item["path"]]) != item["sha256"]:
                raise ValueError("NAS delisted source scoped member restore failed")
    archive_receipt = {"status": "verified", "NAS_archive_sha256": digest,
                       "NAS_archive_bytes": evidence["size_bytes"], "NAS_path": remote,
                       "source_files": identities, "scoped_restore_all_member_SHA256_parity": True}
    (card / "nas_source_archive.json").write_text(json.dumps(archive_receipt, indent=2) + "\n")
    positive = [probe["code"] for probe in receipt["probes"] if probe["rows"] > 0]
    decision = ("positive_old_symbol_transport_needs_security_identity_and_full_history"
                if positive else "futu_exact_old_symbol_transport_unavailable_no_price_factor")
    result = {"experiment": card.name, "decision": decision,
              "plan_sha256": receipt["plan_sha256"], "request_receipt": "data/raw/external/stage1_futu_delisted_price_probe_20261023/request_receipt.json",
              "fixed_requests": len(receipt["probes"]), "positive_symbols": positive,
              "probes": receipt["probes"], "raw_NAS_archive_sha256": digest,
              "NAS_scoped_restored": True, "future_SOXX_outcome_reads": 0,
              "model_fits": 0, "native_accounts": 0, "purchases": 0}
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = {"experiment": card.name, "decision": decision,
               "objective_achieved": False, "plan_sha256": result["plan_sha256"],
               "result": "result.json", "new_model_fits": 0, "new_native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": []},
               "limitations": ["A ticker response cannot establish original CUSIP identity or full 2011-2022 basket coverage.",
                               "No Qlib factor, future label, model or account was constructed."]}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nentrypoint: " + card.name + ".ipynb\n"
        "phase: completed_read_only_source_transport\n")
    lines = ["# Futu 旧 SOXX 成分股价格：固定只读传输探针", "",
        "对原篮子缺价的 SNDK、XLNX、MXIM 各执行一次预定日期的 Futu 不复权日 K 线请求。"
        "全部响应或错误字符串和请求回执原字节按哈希备份 NAS、远端校验并逐件限定恢复；"
        "没有未来 SOXX 收益、因子、拟合、账户、下单或购买。", ""]
    for probe in receipt["probes"]:
        lines.append(f"- `{probe['code']}` {probe['start']}–{probe['end']}: 返回码 {probe['ret_code']}，{probe['rows']} 行。")
    lines.extend(["", f"决定 `{decision}`。", "",
        "即便某符号有历史行，也必须逐只核对原证券 CUSIP、全部篮子日期、拆分/股息/退市处理、首次可用时点和 NAS 原件，"
        "不得用收购方价格替代或把缺失成分重归一。SOXX 0bp/2/5bp及独立前向门仍未通过。"])
    (card / "conclusion.md").write_text("\n".join(lines) + "\n")
    return summary
