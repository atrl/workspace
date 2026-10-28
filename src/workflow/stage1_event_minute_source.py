"""Seal full event-time Qlib prices against the frozen Stage1 execution plane."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

from src.data.minute_intake import atomic_json
from src.data.stage1_event_minute_source import build_event_minute_provider
from src.workflow.minute_action_value import _source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    import yaml
    from qlib.data import D
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        summary = json.loads((card / "summary.json").read_text())
        if summary["plan_sha256"] != digest:
            raise ValueError("blocked: event source plan changed")
        return summary
    source_path = root / plan["source_manifest"]
    if _sha(source_path) != plan["source_sha256"]:
        raise ValueError("blocked: event source manifest differs")
    binding = json.loads((root / plan["nas_binding"]).read_text())
    if not binding["verified"] or binding["source_id"] != plan["source_sha256"]:
        raise ValueError("blocked: event source NAS snapshot receipt differs")
    provider = root / plan["provider_uri"]
    if (provider / "qlib_store_manifest.json").exists():
        manifest = json.loads((provider / "qlib_store_manifest.json").read_text())
        if manifest.get("source_manifest_sha256") != plan["source_sha256"]:
            raise ValueError("blocked: preexisting event provider has another source")
    else:
        manifest = build_event_minute_provider(source_manifest=source_path,
            source_sha256=plan["source_sha256"], provider_uri=provider,
            as_of=plan["source_as_of"], workspace_root=root)
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    frozen, source_audit = _source(root, plan["price_source"], ["inputs/frame"])
    frame = frozen["inputs/frame"]
    if len(frame) != plan["expected_frame_rows"]:
        raise ValueError("blocked: frozen full-path calendar differs")
    fields = ["$open", "$raw_open", "$close", "$raw_close", "$price_valid"]
    observed = D.features(["SOXL", "SOXX"], fields,
        start_time=plan["source_start"] + " 09:31:00",
        end_time=plan["source_as_of"] + " 16:00:00", freq="1min")
    if len(observed) != plan["expected_minute_rows"]:
        raise ValueError("blocked: event Qlib minute row count differs")
    audit_rows = []
    schedule = xc.get_calendar("XNYS", start=frame.trade_date.min(),
                               end=frame.exit_date.max()).schedule
    close_stamps = pd.Series([pd.Timestamp(x).tz_convert("America/New_York").tz_localize(None)
        for x in schedule.close], index=pd.DatetimeIndex(schedule.index).tz_localize(None))
    for symbol in ("SOXL", "SOXX"):
        source = observed.xs(symbol, level="instrument")
        lower = symbol.lower()
        for slot, dates, field, name in (
            ("known", frame.trade_date + pd.Timedelta(hours=9, minutes=45), "$close", f"known_{lower}"),
            ("open", frame.trade_date + pd.Timedelta(hours=9, minutes=48), "$open", f"{lower}_open"),
            ("recovery", frame.trade_date + pd.Timedelta(hours=10, minutes=33), "$open", f"{lower}_recovery"),
            ("exit_open", frame.exit_date + pd.Timedelta(hours=9, minutes=48), "$open", f"{lower}_exit_open"),
            ("close", frame.trade_date.map(close_stamps), "$close", f"close_{lower}"),
        ):
            for prefix in (("", "raw_") if slot != "close" else ("",)):
                qfield = "$raw_" + field.removeprefix("$") if prefix else field
                column = "raw_" + name if prefix else name
                actual = source[qfield].reindex(pd.DatetimeIndex(dates)).to_numpy(float)
                expected = frame[column].to_numpy(float)
                if not np.array_equal(actual, expected):
                    raise ValueError("blocked: event-minute price plane differs: " + column)
                audit_rows.append({"symbol": symbol, "slot": slot,
                    "plane": prefix or "adjusted", "rows": len(actual), "max_abs_error": 0.0})
    valid = observed["$price_valid"]
    if int(valid.ne(1).sum()) != plan["expected_invalid_bars"]:
        raise ValueError("blocked: event-minute source validity mask differs")
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=card.name, uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan,
            provider_manifest=manifest, source_audit=source_audit)
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            builder=(root / "src/data/stage1_event_minute_source.py").read_text())
        recorder.save_objects(artifact_path="analysis", anchor_audit=pd.DataFrame(audit_rows))
        recorder.log_metrics(exact_price_anchor_rows=float(sum(x["rows"] for x in audit_rows)))
    summary = {"experiment": card.name, "plan_sha256": digest,
        "decision": "event_minute_provider_source_parity_pass_needs_label_information",
        "objective_achieved": False, "historical_research_passed": False,
        "provider_uri": str(provider), "provider_manifest_sha256": _sha(provider / "qlib_store_manifest.json"),
        "source_sha256": plan["source_sha256"], "minute_rows": len(observed),
        "invalid_bars": int(valid.ne(1).sum()),
        "exact_anchor_rows": int(sum(x["rows"] for x in audit_rows)),
        "historical_first_receipt_qualified": False,
        "new_model_fits": 0, "new_native_accounts": 0,
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/*"]}}
    pd.DataFrame(audit_rows).to_csv(card / "anchor_audit.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": str(provider), "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# Stage1 事件驱动全分钟来源\n\n"
        f"现有NAS原始NONE分钟与公布复权事件构建新Qlib全分钟双价格平面，共{len(observed)}行。"
        f"与原执行价格框架{summary['exact_anchor_rows']}个调整价及原价锚点精确相同；"
        f"源无效bar {summary['invalid_bars']}条，不填补。"
        "右边界完成分钟在60秒研究假定后可用，另延迟到后续bar开盘成交；历史真实首收未认证。"
        "本卡只证明价格来源和时钟映射，0拟合、0新账户、0策略收益。\n")
    atomic_json(summary, card / "summary.json")
    return summary
