"""Notebook-owned qualification of original Fed semiconductor vintages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fed_capacity_vintages import capacity_information_state, collect_capacity_vintages
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.data.qlib_native import build_qlib_store_from_frame


def run_capacity_source(*, root, card, plan):
    import qlib
    import yaml
    from qlib.data import D
    from qlib.workflow import R
    if (card / "summary.json").is_file():
        previous = json.loads((card / "summary.json").read_text())
        current_hash = hashlib.sha256((card / "plan.json").read_bytes()).hexdigest()
        if previous["plan_sha256"] == current_hash:
            return previous
        if previous["decision"] != "needs_source_repair":
            raise ValueError("blocked: qualified source plan changed")
        attempts_path = card / "qualification_attempts.json"
        attempts = json.loads(attempts_path.read_text()) if attempts_path.is_file() else []
        if not any(attempt["recorder_id"] == previous["recorder_id"] for attempt in attempts):
            attempts.append({"recorder_id": previous["recorder_id"], "plan_sha256": previous["plan_sha256"],
                "decision": previous["decision"], "nas_evidence": previous["nas_evidence"],
                "reason": "preserve initial parsing failure and original payloads; source scope and ASA page markers repaired"})
            atomic_json(attempts, attempts_path)
    def progress(value):
        atomic_json(value, card / "acquisition_progress.json")
    raw = root / plan["canonical_raw_cache"]
    manifest = collect_capacity_vintages(directory=raw, as_of=plan["as_of"], start=plan["source_start"],
        maximum_documents=plan["acquisition_budget_documents"], workers=plan["download_workers"], progress=progress)
    qlib.init(provider_uri=str(root / plan["price_calendar"]), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="1min")).normalize().unique().sort_values()
    state, events = capacity_information_state(manifest, calendar)
    store = root / plan["information_provider"]
    fields = ["known", *plan["information_features"]]
    data = state[["datetime", *fields]].copy(); data["instrument"] = "US_SEMI_CAPACITY"
    store_manifest = build_qlib_store_from_frame(data, provider_uri=store, freq="day", as_of=plan["as_of"],
        field_map={field: field for field in fields}, source="original Fed G17 NAICS3344 publication vintages, not trading prices")
    store_manifest.update(information_instrument=True, source_is_trading_prices=False, revised_history_backfilled=False,
        availability="next supported native trading session09:31 after release; pre-calendar vintages known by first supported date",
        public_clock_declared_by_government=True, first_receipt_certified=False)
    (store / "qlib_store_manifest.json").write_text(json.dumps(store_manifest, indent=2) + "\n")
    qlib.init(provider_uri=str(store), region="us", kernels=1)
    observed = D.features(["US_SEMI_CAPACITY"], ["$" + field for field in fields], freq="day").xs("US_SEMI_CAPACITY", level="instrument")
    if not pd.DatetimeIndex(D.calendar(freq="day")).equals(calendar) or not observed.index.equals(calendar):
        raise ValueError("blocked: information store changed the full price calendar")
    for field in fields:
        if not np.allclose(observed["$" + field].to_numpy(), state[field].to_numpy(), rtol=2e-7, atol=1e-5, equal_nan=True):
            raise ValueError("blocked: Qlib information readback changed source values")
    known = state.known.eq(1)
    if not state.loc[known, "available_at"].le(state.loc[known, "datetime"] + pd.Timedelta(hours=9, minutes=46)).all():
        raise ValueError("blocked: source available after decision")
    scoped = events.loc[events.source_scope_qualified.eq(True)]
    audit = {"raw_documents": len(manifest["entries"]), "expected_documents": manifest["expected_documents"],
        "target_scope_documents": len(scoped), "legacy_scope_documents_excluded": len(events) - len(scoped),
        "first_target_scope_publication": scoped.release_date.min(),
        "source_complete": manifest["source_complete"], "failures": manifest["failures"], "native_calendar_rows": len(calendar),
        "known_rows": int(known.sum()), "unknown_rows": int((~known).sum()), "earliest_publication": events.release_date.min(),
        "latest_publication": events.release_date.max(), "first_receipt_qualified": False,
        "original_published_values_used": True, "same_vintage_monthly_changes": True, "latest_fred_revisions_used": False,
        "historical_published_clock_verified": True, "next_supported_session_guard": True,
        "economic_scope": "US manufacturing NAICS3344; not global chip sales or direct ETF fair value"}
    qualified = manifest["source_complete"] and known.mean() >= .99
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=plan["experiment"], recorder_name="original_g17_capacity_source", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.log_params(plan_sha256=hashlib.sha256((card / "plan.json").read_bytes()).hexdigest(), model_fits=0, backtests=0)
        recorder.save_objects(artifact_path="sources", manifest=manifest, raw_index=(raw / "release_index.html").read_bytes(),
            raw_supplements={entry["path"]: (raw / entry["path"]).read_bytes() for entry in [*manifest["entries"], *manifest["failures"]] if "path" in entry},
            acquisition_code=(root / "src/data/fed_capacity_vintages.py").read_text(), workflow_code=(root / "src/workflow/fed_capacity_source.py").read_text())
        recorder.save_objects(artifact_path="analysis", information=state, events=events, audit=audit, store_manifest=store_manifest)
        recorder.log_metrics(qualified_source_documents=len(manifest["entries"]), source_unknown_rows=int((~known).sum()),
            registered_artifact_count=9)
    summary = {"experiment": plan["experiment"], "plan_sha256": hashlib.sha256((card / "plan.json").read_bytes()).hexdigest(),
        "decision": "source_qualified_for_bounded_information_experiment" if qualified else "needs_source_repair",
        "forecast_or_alpha_claim": False, "objective_achieved": False, "goal_remains_active": True,
        "model_fits": 0, "backtests": 0, "as_of": plan["as_of"], "source_audit": audit,
        "provider_uri": str(store), "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking), "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["sources/raw_supplements", "sources/raw_index", "analysis/information", "analysis/events"]},
        "first_receipt_or_deployment_qualified": False, "latest_revision_as_old_information_used": False,
        "next_action": "freeze a distinct, parsimonious information-increment experiment after source gates; source qualification is not alpha"}
    artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
    atomic_json({"recorder_id": recorder.id, "entries": [{"path": str(p.relative_to(card)),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(artifacts.rglob("*")) if p.is_file()]}, card / "verification_manifest.json")
    minute_binding = json.loads((root / "research/minute_information_pit_allocator_20261012/nas_binding.json").read_text())
    atomic_json({"schema_version": "nas_research_source_v1", "authority": "yanxuan_nas", "ssh": minute_binding["ssh"],
        "source_id": hashlib.sha256((raw / "source_manifest.json").read_bytes()).hexdigest(),
        "raw_authority": "Recorder sources/raw_supplements immutable byte archive on NAS; local raw cache disposable"}, card / "nas_binding.json")
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
        recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
    atomic_json(summary, card / "summary.json")
    events.drop(columns=["published_months", "published_values"], errors="ignore").to_csv(card / "publication_inventory.csv", index=False)
    atomic_json(audit, card / "source_quality.json")
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "provider_uri": str(store),
        "recorder_uri": str(tracking), "as_of": plan["as_of"], "region": "us", "freq": "day",
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    text = "# 半导体产能利用率原始来源资格结论\n\n"
    text += f"已从原始 Fed G17 日期化补充表恢复 {audit['raw_documents']}/{audit['expected_documents']} 个发布版本，发布日期覆盖 {audit['earliest_publication']} 至 {audit['latest_publication']}。完整 Qlib 价格日历 {len(calendar)} 天，已知信息 {int(known.sum())} 天，未知 {(~known).sum()} 天。\n\n"
    text += f"决定：`{summary['decision']}`。没有训练、预测或回测，不能据此声称产生超额。原始发布版本按 Table2 NAICS3344 和容量百分比核对，月份和发布时间分开；差值只用同一版本已经发布的两个相邻月份，不回填最新 FRED 修订值。\n\n"
    text += "原文声明09:15发布，采用发布日之后的下一个受支持真实交易日09:31可用；早于价格日历的发布只被认定在首个支持日期已知，不声称最早接收时点。US国内制造业指标不等于全球芯片销售，经济预测能力需新实验自证。\n\n"
    text += "原始字节、事件状态、信息store及读取审计保存在 NAS Recorder，Git只保存瘦卡和引用。真实首收资格仍未取得，不能直接晋升。目标保持active：下一步先冻结简洁的独立信息增量检验，保留同数据/成本/风控对照与失败，不追加本轮已关闭NN路线的参数搜索。\n"
    (card / "conclusion.md").write_text(text)
    return summary
