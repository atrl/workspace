"""Existing-source endpoint and event preflight; never computes a ROD signal or return."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.full_path_prices import _source_frames
from src.data.close_rod_events import audit_event_coverage
from src.data.minute_intake import atomic_json

NAME = "stage1_close_rod_20261018_source_preflight"
FIELDS = ["open", "high", "low", "close", "price_valid", "is_final"]
CODE_FILES = ["src/workflow/close_rod_source_preflight.py", "src/data/close_rod_events.py",
              "src/data/full_path_prices.py", "src/data/minute_price_basis.py",
              "src/data/qlib_native.py", "src/core/recorder_sources.py", "src/data/nas_market_data.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root):
    return {name: sha(Path(root) / name) for name in CODE_FILES}


def _source(root, binding):
    if binding.get("archive_sha256"):
        nas = json.loads((root / binding["card"] / "nas_binding.json").read_text())
        if nas["evidence"]["sha256"] != binding["archive_sha256"]:
            raise ValueError("frozen Recorder archive identity changed")
    return load_verified_recorder_objects(root / binding["card"],
        expected_recorder_id=binding["recorder_id"], artifacts=binding["artifacts"],
        expected_manifest_sha256=binding["manifest_sha256"],
        expected_artifact_sha256=binding["artifact_sha256"])


def _sessions(root, plan):
    import qlib
    from qlib.data import D
    import exchange_calendars as xc
    calendar = plan["calendar"]
    if sha(root / calendar["file"]) != calendar["sha256"]:
        raise ValueError("frozen native calendar changed")
    if xc.__version__ != calendar["exchange_calendars_version"]:
        raise ValueError("session schedule implementation changed")
    qlib.init(provider_uri=str(root / calendar["provider"]), region="us", kernels=1)
    days = pd.DatetimeIndex(D.calendar(start_time=calendar["first_previous_session"],
                                      end_time=plan["as_of"], freq="day"))
    schedule = xc.get_calendar("XNYS", start=days[0], end=days[-1]).schedule
    schedule.index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    pd.testing.assert_index_equal(days, schedule.index, check_names=False)
    account = days[(days >= pd.Timestamp(calendar["first_trade"])) & (days <= pd.Timestamp(calendar["last_trade"]))]
    if len(account) != 944 or days[0] != pd.Timestamp("2023-01-03"):
        raise ValueError("frozen account/previous-session calendar differs")
    records = []
    for day in account:
        previous = days[days.get_loc(day) - 1]
        row, prior = schedule.loc[day], schedule.loc[previous]
        opened = pd.Timestamp(row.open).tz_convert("America/New_York").tz_localize(None)
        closed = pd.Timestamp(row.close).tz_convert("America/New_York").tz_localize(None)
        prev_close = pd.Timestamp(prior.close).tz_convert("America/New_York").tz_localize(None)
        minutes = int((closed - opened).total_seconds() / 60)
        if minutes not in [390, 210] or opened != day + pd.Timedelta("9h30min"):
            raise ValueError("unsupported session structure")
        records.append({"trade_date": day, "previous_session": previous, "open_at": opened,
                        "actual_close_at": closed, "previous_actual_close_at": prev_close,
                        "expected_minutes": minutes, "regular_session": minutes == 390})
    return pd.DataFrame(records).set_index("trade_date"), days


def _requirements(sessions, terminal):
    rows = []
    def add(day, slot, bar_end, price_stamp, field):
        for symbol in ["SOXL", "SOXX"]:
            rows.append({"trade_date": day, "slot": slot, "instrument": symbol,
                         "bar_end": bar_end, "price_stamp": price_stamp, "field": field})
    for day, row in sessions.iterrows():
        add(day, "previous_actual_close", row.previous_actual_close_at, row.previous_actual_close_at, "close")
        for slot, source_time, actual_time, field in [
            ("base_known", "9h45min", "9h45min", "close"),
            ("base_open", "9h48min", "9h47min", "open"),
            ("base_recovery", "10h33min", "10h32min", "open")]:
            add(day, slot, day + pd.Timedelta(source_time), day + pd.Timedelta(actual_time), field)
        add(day, "base_actual_close", row.actual_close_at, row.actual_close_at, "close")
        if row.regular_session:
            for slot, source_time, actual_time, field in [
                ("late_known", "15h30min", "15h30min", "close"),
                ("late_entry", "15h33min", "15h32min", "open"),
                ("late_exit", "16h", "15h59min", "open")]:
                add(day, slot, day + pd.Timedelta(source_time), day + pd.Timedelta(actual_time), field)
    for slot, source_time, actual_time, field in [
        ("terminal_known", "9h45min", "9h45min", "close"),
        ("terminal_open", "9h48min", "9h47min", "open")]:
        add(terminal, slot, terminal + pd.Timedelta(source_time), terminal + pd.Timedelta(actual_time), field)
    return pd.DataFrame(rows)


def _native_endpoints(root, plan, requirements, frames):
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from src.data import build_qlib_store_from_frame
    pieces, statuses = [], []
    for symbol in ["SOXL", "SOXX"]:
        raw = frames[symbol].copy()
        raw["datetime"] = raw.timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
        if raw.datetime.duplicated().any():
            raise ValueError("duplicate source timestamps")
        raw = raw.set_index("datetime")
        required = requirements.loc[requirements.instrument.eq(symbol)]
        unique = pd.DatetimeIndex(required.bar_end.unique()).sort_values()
        picked = raw.reindex(unique)
        for record in required.to_dict("records"):
            row = picked.loc[record["bar_end"]]
            value = row[record["field"]]
            valid = bool(pd.notna(value) and np.isfinite(value) and value > 0
                         and row.get("price_valid") == True and row.get("is_final") == True)
            statuses.append({**record, "present": record["bar_end"] in raw.index,
                             "valid": valid, "raw_float64_price": float(value) if pd.notna(value) else np.nan})
        chosen = picked.loc[:, FIELDS].astype("float64")
        chosen["instrument"] = symbol
        pieces.append(chosen.rename_axis("datetime").reset_index())
    panel = pd.concat(pieces, ignore_index=True).sort_values(["datetime", "instrument"]).reset_index(drop=True)
    status = pd.DataFrame(statuses)
    # Missing future execution marks fail the whole endpoint dependency. They
    # are never turned into an ex-post per-day no-trade selector.
    if not status.valid.all():
        return panel, status, None, None, pd.DataFrame()
    provider = root / plan["provider_output"]
    manifest = build_qlib_store_from_frame(panel, provider_uri=provider, freq="1min", region="us",
        as_of=plan["as_of"], field_map={name: name for name in FIELDS},
        source="Hash-verified raw NONE endpoint bars only; no signal/return expressions")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    observed = D.features(["SOXL", "SOXX"], ["$" + name for name in FIELDS],
        start_time=panel.datetime.min(), end_time=panel.datetime.max(), freq="1min")
    expected = panel.set_index(["datetime", "instrument"])[FIELDS].sort_index()
    observed = observed.rename(columns=lambda name: name.removeprefix("$")).reorder_levels(["datetime", "instrument"]).sort_index()
    # Sparse union calendars contain rows absent for the other instrument; both
    # frozen instruments share this endpoint set, so exact alignment is required.
    pd.testing.assert_index_equal(observed.index, expected.index)
    np.testing.assert_allclose(observed.to_numpy(), expected.to_numpy(np.float32), rtol=0, atol=0, equal_nan=True)
    precision = []
    for name in FIELDS:
        left, right = expected[name].to_numpy(float), observed[name].to_numpy(float)
        precision.append({"field": name, "rows": len(left), "max_absolute_representation_error": float(abs(left-right).max()),
                          "max_relative_representation_error": float((abs(left-right)/np.maximum(abs(left),1e-30)).max())})
    dataset = DatasetH(handler=DataHandlerLP.from_df(pd.concat({"feature": expected}, axis=1)),
                       segments={"all": (panel.datetime.min(), panel.datetime.max())})
    dataset.config(dump_all=True, recursive=True)
    pd.testing.assert_frame_equal(dataset.prepare("all", col_set="feature"), expected)
    return panel, status, manifest, dataset, pd.DataFrame(precision)


def run(*, root, card):
    import qlib
    from qlib.workflow import R
    from src.data.nas_market_data import archive_nas_recorder
    root, card = Path(root), Path(card)
    plan_path = card / "source_preflight_plan.json"
    plan, digest = json.loads(plan_path.read_text()), sha(plan_path)
    review = json.loads((card / "design_preflight_review.json").read_text())
    if (sha(card / "design_preflight_review.json") != plan["review_sha256"]
            or review.get("verdict") != "APPROVE_SOURCE_ENDPOINT_EVENT_PREFLIGHT_ONLY"
            or review.get("source_only_preflight_eligible") is not True
            or review.get("policy_execution_approved") is not False):
        raise ValueError("source-only preflight approval missing or changed")
    if sha(card / "experiment_design_draft.json") != plan["corrected_design_sha256"]:
        raise ValueError("corrected no-signal design boundary changed")
    if sha(card / "source_contract_draft.json") != plan["source_contract_sha256"]:
        raise ValueError("source contract changed")
    if (sha(card / "experiment_design_draft_before_preflight_corrections.json")
            != review["reviewed_design_draft_sha256"]):
        raise ValueError("original reviewed draft identity changed")
    identity = json.loads((card / "source_preflight_implementation_identity.json").read_text())
    if identity["plan_sha256"] != digest or identity["code_sha256"] != code_identity(root):
        raise ValueError("source preflight code/plan identity differs")
    output = card / "source_preflight"
    output.mkdir(exist_ok=True)
    if (output / "summary.json").exists():
        prior = json.loads((output / "summary.json").read_text())
        if prior["plan_sha256"] != digest:
            raise ValueError("completed source preflight belongs to another plan")
        return prior
    atomic_json({"phase": "running_source_only", "plan_sha256": digest,
                 "signals": 0, "labels": 0, "fits": 0, "accounts": 0}, card / "source_preflight_run_state.json")
    sessions, calendar = _sessions(root, plan)
    sources, audits = {}, {}
    for role, binding in plan["sources"].items():
        sources[role], audits[role] = _source(root, binding)
    source_path = root / plan["minute_manifest"]["path"]
    if sha(source_path) != plan["minute_manifest"]["sha256"]:
        raise ValueError("frozen NONE manifest changed")
    manifest = json.loads(source_path.read_text())
    frames, rehab = _source_frames(source_path, manifest, as_of=plan["as_of"], workspace_root=root)
    corp = sources["corporate"]
    event_mask, event_audit, differences = audit_event_coverage(corp["data/cash_events.pkl"],
        corp["data/split_events.pkl"], rehab, sessions=sessions.index,
        source_contract=corp["inputs/source_contract.pkl"], rehab_source_manifest=manifest,
        original_qualified_through="2026-09-25", as_of=plan["as_of"])
    requirements = _requirements(sessions, pd.Timestamp(plan["as_of"]))
    panel, endpoints, provider_manifest, dataset, precision = _native_endpoints(root, plan, requirements, frames)
    original = sources["anchors"]["inputs/raw_points"]
    original_checks = []
    for symbol in ["SOXL", "SOXX"]:
        selected = endpoints.loc[endpoints.instrument.eq(symbol) & endpoints.slot.str.startswith(("base_", "terminal_"))]
        prices = pd.Series(selected.raw_float64_price.to_numpy(np.float32).astype(float), index=pd.DatetimeIndex(selected.price_stamp)).sort_index()
        pd.testing.assert_index_equal(prices.index, original.index, check_names=False)
        exact = np.array_equal(prices.to_numpy(), original[symbol].to_numpy())
        original_checks.append({"instrument": symbol, "original_points": len(prices), "exact_float32_anchor_equality": bool(exact)})
    coverage = endpoints.groupby(["instrument", "slot"], as_index=False).agg(
        required=("valid", "size"), valid=("valid", "sum"), present=("present", "sum"))
    source_ready = bool(endpoints.valid.all() and all(x["exact_float32_anchor_equality"] for x in original_checks))
    summary = {"experiment": NAME, "plan_sha256": digest, "stage": "source_only_preflight_completed",
        "decision": "source_dependencies_observed_for_separate_strategy_review" if source_ready else "needs_data_endpoint_dependency",
        "endpoint_gate_pass": source_ready, "account_sessions": len(sessions),
        "regular_sessions": int(sessions.regular_session.sum()), "half_days": int((~sessions.regular_session).sum()),
        "first_previous_session": str(sessions.previous_session.iloc[0].date()),
        "first_previous_close": str(sessions.previous_actual_close_at.iloc[0]),
        "prior_half_day_closes_used": int(sessions.previous_actual_close_at.dt.hour.eq(13).sum()),
        "endpoint_rows": len(endpoints), "invalid_endpoint_rows": int((~endpoints.valid).sum()),
        "native_float64_rows": len(panel), "original_anchor_checks": original_checks,
        "event_audit": event_audit,
        "source_regular_non_event_sessions": int((sessions.regular_session & event_mask.event_source_ready).sum()),
        "new_market_source_HTTP": 0, "new_ROD_values_or_signs": 0, "new_return_labels": 0,
        "new_model_fits": 0, "new_policy_paths": 0, "new_native_accounts": 0,
        "reference_passive_revaluations": 0, "historical_PIT_qualified": False,
        "independent_forward_qualified": False, "strategy_design_approved": False,
        "policy_execution_approved": False, "objective_achieved": False,
        "provider_uri": str(root / plan["provider_output"]), "as_of": plan["as_of"],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "source/*", "analysis/endpoint_rows", "analysis/event_mask", "sources/*"]}}
    tracking = output / "qlib_recorder"
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        summary.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id), recorder_uri=str(tracking),
                       recorder_local_dir=str(tracking / str(rec.experiment_id) / rec.id))
        rec.save_objects(artifact_path="inputs", plan=plan, review=review, source_audits=audits,
                         native_session_calendar=sessions, original_raw_points=original,
                         minute_manifest=manifest, frozen_reference_identities=plan["reference_identities"])
        rec.save_objects(artifact_path="source", panel_float64=panel, dataset=dataset,
                         provider_manifest=provider_manifest, issuer_contract=corp["inputs/source_contract.pkl"])
        rec.save_objects(artifact_path="analysis", summary=summary, endpoint_rows=endpoints, coverage=coverage,
                         precision=precision, event_mask=event_mask, event_audit=event_audit, event_differences=differences)
        rec.save_objects(artifact_path="sources", **{name.replace("/", "__"): (root/name).read_text() for name in CODE_FILES})
        artifacts = Path(summary["recorder_local_dir"]) / "artifacts"
        rec.log_metrics(registered_artifact_count=sum(path.is_file() for path in artifacts.rglob("*")),
                        source_endpoint_rows=len(endpoints), new_signals=0, new_accounts=0)
    atomic_json({"recorder_id": rec.id, "entries": [{"path": str(path.relative_to(output)), "sha256": sha(path),
                 "bytes": path.stat().st_size} for path in sorted(artifacts.rglob("*")) if path.is_file()]}, output/"verification_manifest.json")
    transport = json.loads((root/plan["nas_transport_binding"]).read_text())
    binding = {"schema_version": "recorder_evidence_binding_v1", "authority": "yanxuan_nas", "ssh": transport["ssh"],
               "source_id": "close-rod-source-preflight-"+digest, "as_of": plan["as_of"],
               "source_plan": str(plan_path.relative_to(root)), "verification_manifest": "verification_manifest.json"}
    atomic_json(binding, output/"nas_binding.json")
    summary["nas_evidence"] = archive_nas_recorder(output/"nas_binding.json", workspace_root=root, card=output,
        recorder_id=rec.id, experiment_id=str(rec.experiment_id))
    for name, table in [("endpoint_coverage", coverage), ("price_representation_error", precision), ("event_differences", differences)]:
        if len(table) > 500:
            raise ValueError("unexpectedly large source summary table")
        table.to_csv(card/(name+".csv"), index=False)
    atomic_json(summary, output/"summary.json")
    atomic_json(summary, card/"summary.json")
    (output/"workflow.yaml").write_text("experiment: "+NAME+"\nentrypoint: src.workflow.close_rod_source_preflight.run\nphase: source_only_preflight_completed\nartifact_policy: slim_card_recorder_first\n")
    (card/"workflow.yaml").write_text("experiment: stage1_close_rod_20261018\nphase: source_preflight_completed_strategy_review_pending\nsource_card: source_preflight\npolicy_execution_authorized: false\nartifact_policy: slim_card_recorder_first\n")
    conclusion = (f"来源预检完成：{len(sessions)}账户日，其中{summary['regular_sessions']}正常日、{summary['half_days']}半日。"
        f"必要端点{len(endpoints)}行，缺失或无效{summary['invalid_endpoint_rows']}行；两资产各3778个原账户价格锚点精确核对。"
        f"首个上一交易日为{summary['first_previous_session']}，实际收盘{summary['first_previous_close']}；前一半日13:00收盘按原日历使用。\n\n"
        f"事件源当前快照覆盖至{event_audit['qualified_through']}，尾段支持={event_audit['event_tail_supported']}。"
        "事件及未知覆盖日双方均不允许新增尾盘lot，完整原账户日保留；这不构成历史PIT的无事件证明。\n\n"
        "仅校验原始价格和Qlib数值表示误差，未比较两价格生成方向，未计算ROD/尾盘收益/标签，0fit/0policy/0account/0新市场数据HTTP。"
        "float64价格是未来获批后的严格符号权威；Qlibfloat32仅派生缓存。\n\n"
        f"决定：{summary['decision']}。来源与NAS身份见source_preflight/summary.json；正式策略设计独评和另行执行GO仍必需。稳定超额目标未完成。\n")
    (card/"conclusion.md").write_text(conclusion)
    atomic_json({"phase": "completed_source_only", "plan_sha256": digest, "recorder_id": rec.id,
                 "signals": 0, "labels": 0, "fits": 0, "accounts": 0}, card/"source_preflight_run_state.json")
    return summary
