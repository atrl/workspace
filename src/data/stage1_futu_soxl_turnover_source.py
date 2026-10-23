"""Read-only Futu SOXL daily turnover capture for one frozen source card."""

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


def turnover_field_quality(raw: pd.DataFrame) -> dict:
    """Treat a zero reported rate with real trading as an unavailable field."""
    rate = pd.to_numeric(raw.turnover_rate, errors="coerce")
    traded = pd.to_numeric(raw.volume, errors="coerce").gt(0) & pd.to_numeric(raw.turnover, errors="coerce").gt(0)
    invalid_zero = rate.eq(0) & traded
    return {"zero_rate_with_positive_trading": int(invalid_zero.sum()),
            "usable_rate": rate.where(~invalid_zero).where(rate.gt(0))}


def capture(root: Path) -> dict:
    import futu
    from futu import AuType, KLType, OpenQuoteContext, RET_OK

    root = Path(root).resolve()
    card = root / "research/stage1_futu_soxl_turnover_source_20261023"
    plan = json.loads((card / "plan.json").read_text())
    request = plan["request"]
    raw_dir = root / "data/raw/external/stage1_futu_soxl_turnover_20261023"
    raw_path = raw_dir / "SOXL_NONE_day.csv"
    receipt_path = raw_dir / "request_receipt.json"
    if raw_path.exists() or receipt_path.exists():
        raise ValueError("source capture already attempted; inspect saved bytes before retry")
    assert plan["frozen"] and request["code"] == "US.SOXL"
    assert request["ktype"] == "KLType.K_DAY" and request["autype"] == "AuType.NONE"
    raw_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "attempted", "SDK_version": futu.__version__, "request": request,
               "plan_sha256": _sha((card / "plan.json").read_bytes()),
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "pages": []}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    frames = []
    key = None
    context = OpenQuoteContext(host=request["host"], port=request["port"])
    try:
        for page in range(1, request["maximum_pages"] + 1):
            key_in_hash = _sha(str(key).encode()) if key is not None else None
            ret, frame, next_key = context.request_history_kline(
                request["code"], start=request["start"], end=request["end"],
                ktype=KLType.K_DAY, autype=AuType.NONE,
                max_count=request["max_count"], page_req_key=key)
            if ret != RET_OK or not isinstance(frame, pd.DataFrame):
                raise ValueError("Futu page " + str(page) + " failed: " + str(frame)[:240])
            if frame.empty or not frame.code.eq(request["code"]).all():
                raise ValueError("Futu SOXL page empty or wrong symbol")
            page_bytes = frame.to_csv(index=False, lineterminator="\n").encode()
            frames.append(frame)
            receipt["pages"].append({"page": page, "rows": len(frame),
                                     "serialized_csv_sha256": _sha(page_bytes),
                                     "bytes": len(page_bytes), "continuation": next_key is not None,
                                     "page_req_key_sha256": key_in_hash})
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            key = next_key
            if key is None:
                break
        if key is not None:
            raise ValueError("SOXL daily history exceeded frozen page budget")
        raw = pd.concat(frames, ignore_index=True)
        raw["time_key"] = pd.to_datetime(raw.time_key)
        raw = raw.sort_values("time_key").reset_index(drop=True)
        if raw.time_key.duplicated().any() or not raw.time_key.is_monotonic_increasing:
            raise ValueError("duplicate or unsorted SOXL daily dates")
        if "turnover_rate" not in raw or not pd.to_numeric(raw.turnover_rate, errors="coerce").notna().all():
            raise ValueError("SOXL turnover_rate missing or nonnumeric")
        data = raw.to_csv(index=False, lineterminator="\n").encode()
        raw_path.write_bytes(data)
        receipt.update(status="captured", rows=len(raw),
                       first_date=str(raw.time_key.min().date()),
                       last_date=str(raw.time_key.max().date()),
                       columns=list(raw.columns),
                       data_file=str(raw_path.relative_to(root)),
                       data_sha256=_sha(data), data_bytes=len(data),
                       finished_at_utc=datetime.now(timezone.utc).isoformat())
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        return receipt
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__,
                       error=str(error)[:500], finished_at_utc=datetime.now(timezone.utc).isoformat())
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        raise
    finally:
        context.close()


def archive_raw(root: Path) -> dict:
    from src.data.nas_market_data import _settings, _ssh, nas_api_call

    root = Path(root).resolve()
    card = root / "research/stage1_futu_soxl_turnover_source_20261023"
    receipt_path = root / "data/raw/external/stage1_futu_soxl_turnover_20261023/request_receipt.json"
    raw_path = root / "data/raw/external/stage1_futu_soxl_turnover_20261023/SOXL_NONE_day.csv"
    output = card / "nas_raw_archive.json"
    if output.exists():
        raise ValueError("SOXL raw source already archived; verify saved receipt")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "captured" and _sha(raw_path.read_bytes()) == receipt["data_sha256"]
    members = [raw_path, receipt_path]
    member_info = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                    "sha256": _sha(p.read_bytes())} for p in members]
    settings = _settings(root)
    with tempfile.TemporaryDirectory(prefix="soxl-turnover-source-") as temporary:
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
            raise ValueError("NAS SOXL source receipt does not match uploaded bytes")
        remote = settings["artifact_root"].rstrip("/") + "/" + digest[:2] + "/" + digest[2:4] + "/" + digest
        if _ssh(settings["ssh"], "sha256sum " + shlex.quote(remote)).split()[0] != digest:
            raise ValueError("NAS SOXL source remote hash mismatch")
        restored = Path(temporary) / "restored.tar.gz"
        with restored.open("wb") as target:
            process = subprocess.run(["ssh", "-p", str(settings["ssh"]["port"]),
                                      "-o", "BatchMode=yes", settings["ssh"]["host"],
                                      "cat " + shlex.quote(remote)], stdout=target,
                                     stderr=subprocess.PIPE, timeout=300)
        if process.returncode or _sha(restored.read_bytes()) != digest:
            raise ValueError("NAS SOXL source scoped archive restore failed")
        with tarfile.open(restored, "r:gz") as source:
            actual = {member.name: source.extractfile(member).read()
                      for member in source.getmembers() if member.isfile()}
        for item in member_info:
            if _sha(actual[item["path"]]) != item["sha256"] or len(actual[item["path"]]) != item["bytes"]:
                raise ValueError("NAS SOXL source scoped member restore failed")
    result = {"status": "verified", "NAS_archive_sha256": digest,
              "NAS_archive_bytes": evidence["size_bytes"], "NAS_path": remote,
              "source_files": member_info, "scoped_restore_all_member_SHA256_parity": True}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def audit_qlib(root: Path) -> dict:
    import numpy as np
    import qlib
    from qlib.data import D
    from qlib.workflow import R

    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder
    from src.data.qlib_native import build_qlib_store_from_frame

    root = Path(root).resolve()
    card = root / "research/stage1_futu_soxl_turnover_source_20261023"
    if (card / "summary.json").exists():
        raise ValueError("SOXL source coverage audit already complete")
    plan = json.loads((card / "plan.json").read_text())
    raw_dir = root / "data/raw/external/stage1_futu_soxl_turnover_20261023"
    raw_path = raw_dir / "SOXL_NONE_day.csv"
    receipt = json.loads((raw_dir / "request_receipt.json").read_text())
    nas = json.loads((card / "nas_raw_archive.json").read_text())
    assert receipt["status"] == nas["status"].replace("verified", "captured")
    assert nas["scoped_restore_all_member_SHA256_parity"]
    assert _sha(raw_path.read_bytes()) == receipt["data_sha256"]
    assert receipt["request"]["autype"] == "AuType.NONE"
    assert len(receipt["pages"]) <= plan["budget"]["Futu_pages_max"]
    raw = pd.read_csv(raw_path)
    assert len(raw) == receipt["rows"] and raw.code.eq("US.SOXL").all()
    date = pd.DatetimeIndex(pd.to_datetime(raw.time_key).dt.normalize())
    assert date.is_unique and date.is_monotonic_increasing
    turnover = pd.to_numeric(raw.turnover_rate, errors="coerce").to_numpy(float)
    assert np.isfinite(turnover).all() and (turnover >= 0).all()
    quality = turnover_field_quality(raw)
    provider = root / "data/qlib/stage1_futu_soxl_turnover_20261023"
    source_frame = pd.DataFrame({"datetime": date, "instrument": "SOXL",
                                 "turnover_rate": turnover})
    manifest = build_qlib_store_from_frame(
        source_frame, provider_uri=provider, freq="day", region="us",
        field_map={"$turnover_rate": "turnover_rate"}, as_of=plan["request"]["end"],
        source="Futu request_history_kline US.SOXL K_DAY AuType.NONE; raw NAS SHA256=" + nas["NAS_archive_sha256"])
    qlib.init(provider_uri=str(provider.resolve()), region="us", kernels=1)
    own_calendar = pd.DatetimeIndex(D.calendar(start_time="2011-01-03", end_time=plan["request"]["end"], freq="day"))
    values = D.features(["SOXL"], ["$turnover_rate"], start_time="2011-01-03",
                        end_time=plan["request"]["end"], freq="day")
    native = values["$turnover_rate"].droplevel("instrument").sort_index()
    assert native.index.equals(own_calendar)
    np.testing.assert_allclose(native.to_numpy(float),
                               source_frame.set_index("datetime").turnover_rate.reindex(native.index).to_numpy(float),
                               rtol=1e-6, atol=1e-8)
    qlib.init(provider_uri=str((root / "data/qlib/minute_information_pit_allocator_20261012/daily_prices").resolve()),
              region="us", kernels=1)
    parent_calendar = pd.DatetimeIndex(D.calendar(start_time="2011-01-03", end_time=plan["request"]["end"], freq="day"))
    known = native.reindex(parent_calendar)
    train = known.loc["2011-01-03":"2022-12-30"]
    held = known.loc["2023-01-03":plan["request"]["end"]]
    usable = pd.Series(quality["usable_rate"].to_numpy(float), index=date).reindex(parent_calendar)
    usable_train = usable.loc["2011-01-03":"2022-12-30"]
    gates = {"train_support": int(usable_train.notna().sum()) >= plan["source_gate"]["Stage1_training_dates_known_min"],
             "held_support": int(held.notna().sum()) >= plan["source_gate"]["Stage1_reused_held_dates_known_min"],
             "no_duplicates": bool(date.is_unique),
             "finite_nonnegative": bool(np.isfinite(turnover).all() and (turnover >= 0).all()),
             "NAS_restored": bool(nas["scoped_restore_all_member_SHA256_parity"]),
             "no_zero_rate_with_positive_trading": quality["zero_rate_with_positive_trading"] == 0}
    decision = "conditional_source_coverage_pass_first_public_unknown" if all(gates.values()) else "needs_source_coverage_no_fit"
    result = {"experiment": card.name, "decision": decision,
              "plan_sha256": _sha((card / "plan.json").read_bytes()),
              "Futu_SDK_version": receipt["SDK_version"],
              "raw_sha256": receipt["data_sha256"], "raw_bytes": receipt["data_bytes"],
              "raw_NAS_archive_sha256": nas["NAS_archive_sha256"],
              "Qlib_provider_uri": str(provider),
              "Qlib_provider_manifest_sha256": _sha((provider / "qlib_store_manifest.json").read_bytes()),
              "Qlib_source_calendar_rows": len(own_calendar),
              "Stage1_parent_calendar_rows": len(parent_calendar),
              "train_calendar": len(train), "train_known": int(train.notna().sum()),
              "train_usable_known": int(usable_train.notna().sum()),
              "held_calendar": len(held), "held_known": int(held.notna().sum()),
              "zero_rate_with_positive_trading_rows": quality["zero_rate_with_positive_trading"],
              "missing_Stage1_dates": [str(x.date()) for x in known.index[known.isna()]],
              "turnover_rate_min": float(native.min()),
              "turnover_rate_median": float(native.median()),
              "turnover_rate_max": float(native.max()),
              "gates": gates, "new_market_requests": len(receipt["pages"]),
              "new_model_fits": 0, "new_native_accounts": 0,
              "future_return_reads": 0, "historical_first_public_qualified": False}
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, request=receipt,
                              nas_raw_archive=nas, Qlib_provider_manifest=manifest,
                              raw_source_bytes=raw_path.read_bytes())
        recorder.save_objects(artifact_path="analysis", native_turnover_rate=native,
                              Stage1_calendar_coverage=known, result=result)
        R.log_params(source="Futu_US_SOXL_K_DAY_AuType_NONE", news_data=False)
        R.log_metrics(train_known=result["train_known"], held_known=result["held_known"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    result.update(recorder_id=rid, experiment_id=eid)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    binding = json.loads((root / "research/minute_information_pit_allocator_20261012/nas_binding.json").read_text())
    binding.pop("evidence", None)
    (card / "nas_binding.json").write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n")
    archived = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                    card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(archived["nas_path"])).split()[0]
    assert remote == archived["sha256"]
    summary = {"experiment": card.name, "decision": decision, "objective_achieved": False,
               "plan_sha256": result["plan_sha256"], "result": "result.json",
               "recorder_id": rid, "experiment_id": eid,
               "NAS_evidence": archived, "NAS_readback": {"verified": True, "sha256": remote},
               "new_model_fits": 0, "new_native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/raw_source_bytes", "analysis/native_turnover_rate",
                                                               "analysis/Stage1_calendar_coverage"]},
               "limitations": ["Historical first-public SOXL turnover values and revisions are not certified.",
                               "Coverage alone is not incremental prediction or SOXX excess.",
                               "No relative-turnover feature has been fit or funded."]}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nentrypoint: " + card.name + ".ipynb\n"
        "provider_uri: data/qlib/stage1_futu_soxl_turnover_20261023\nphase: source_only_no_factor_fit\n")
    (card / "conclusion.md").write_text(
        "# Stage1 Futu SOXL 日换手率：同口径来源与覆盖门\n\n"
        "Futu `US.SOXL` 完成日 K 线、AuType.NONE 只读取得换手率。原始表和请求回执均按 SHA-256 备份 NAS 并逐件恢复；"
        "原生 Qlib 来源与原 Stage1 信号日历核对。没有新闻、未来收益、拟合或资金账户。\n\n"
        f"2011–2022 训练 {result['train_known']}/{result['train_calendar']} 日、2023–2026 复用评价 "
        f"{result['held_known']}/{result['held_calendar']} 日有值；缺日 {result['missing_Stage1_dates']}。"
        f"来源门 {sum(gates.values())}/{len(gates)}，决定 `{decision}`。\n\n"
        "历史首次公开版本、修订与每日收盘后可用时间仍未认证。此卡仅为后续 SOXL 相对 SOXX 换手假设补全同字段来源；"
        "已关闭的 SOXX 单边换手异常不能改名重开。下一步先按原39项和 SOXX 换手做无结果去重，"
        "再另冻接收任务与信息门；通过后才可能考虑原 f696 父本0bp Tiered完整账户。\n\n"
        f"原始 NAS SHA `{nas['NAS_archive_sha256']}`；Qlib Recorder `{rid}` NAS SHA `{remote}`。"
        "SOXX 超额、2/5bp及独立前向仍未实现。\n")
    return summary
