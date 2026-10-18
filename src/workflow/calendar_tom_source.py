"""Cache-only, venue-separated calendar knowledge metadata. No TOM feature values."""
from pathlib import Path
import hashlib
import json
import os

import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json

NAME = "calendar_tom_source_20261018"
VENUES = ("NYSE", "NYSE_ARCA", "NASDAQ")
CLOCKS = (("original_0929", 9, 29), ("research_0946", 9, 46))
CODE_FILES = ["src/workflow/calendar_tom_source.py", "src/data/calendar_knowledge.py",
    "src/core/recorder_sources.py", "src/data/minute_intake.py", "src/data/nas_market_data.py",
    "tests/test_calendar_knowledge.py", "tests/test_calendar_tom_source.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root):
    return {name:sha(Path(root)/name) for name in CODE_FILES}


def _events_at(ledger, venue, clock):
    if pd.isna(clock):return ledger.iloc[:0]
    return ledger.loc[ledger.venue.eq(venue) & ledger.knowledge_at.notna() & ledger.knowledge_at.le(clock)]


def _planned_state(known, date):
    """Conditional ordinary schedule metadata; not actual-open confirmation."""
    base=known.loc[known.assertion_kind.eq("weekday_baseline") & known.effective_from.le(date)
                   & known.effective_until.ge(date)]
    if base.empty:return "unknown", (), False
    newest=base.knowledge_at.max();base=base.loc[base.knowledge_at.eq(newest)]
    if base.document_id.nunique()!=1:return "unknown", tuple(sorted(base.document_id.unique())), True
    document=base.iloc[0].document_id
    documents={document}
    ordinary=known.loc[known.document_id.eq(document) & known.session_date.eq(date)
                       & known.assertion_kind.isin(["planned_closed","early_close"])]
    state="closed" if ordinary.assertion_kind.eq("planned_closed").any() else "early_close" if ordinary.assertion_kind.eq("early_close").any() else "weekend" if date.dayofweek>=5 else "planned_weekday_open"
    if ordinary.assertion_kind.nunique()>1:return "unknown",tuple(sorted(documents)),True
    exceptions=known.loc[known.session_date.eq(date) & known.assertion_kind.isin(["temporary_closed","reopening"])]
    if not exceptions.empty:
        latest=exceptions.loc[exceptions.knowledge_at.eq(exceptions.knowledge_at.max())]
        documents.update(latest.document_id)
        if latest.assertion_kind.nunique()>1:return "unknown",tuple(sorted(documents)),True
        state="closed" if latest.assertion_kind.iloc[0]=="temporary_closed" else "planned_reopening"
    return state,tuple(sorted(documents)),False


def _horizon(known, day, state_cache=None):
    """Does already announced ordinary metadata cover month end and next open?"""
    if pd.isna(day):return False,pd.NaT,pd.NaT,[],False,"unknown"
    end=day+pd.offsets.MonthEnd(0)
    documents=set();conflict=False
    cache={} if state_cache is None else state_cache
    identity=tuple(known.index)
    def lookup(date):
        key=(identity,date)
        if key not in cache:cache[key]=_planned_state(known,date)
        return cache[key]
    current,doc,bad=lookup(day)
    documents.update(doc)
    conflict |= bad
    for date in pd.date_range(day,end,freq="D"):
        state,doc,bad=lookup(date)
        conflict |= bad
        documents.update(doc)
        if state=="unknown":return False,pd.NaT,pd.NaT,sorted(documents),conflict,current
    for date in pd.date_range(end+pd.Timedelta(days=1),end+pd.Timedelta(days=14),freq="D"):
        state,doc,bad=lookup(date)
        conflict |= bad
        documents.update(doc)
        if state=="unknown":return False,end,pd.NaT,sorted(documents),conflict,current
        if state in ("planned_weekday_open","early_close","planned_reopening"):
            return not conflict,end,date,sorted(documents),conflict,current
    return False,end,pd.NaT,sorted(documents),conflict,current


def build_date_metadata(timing, calendar, ledger, *, as_of, amendment_gaps=()):
    """Metadata only: no ranks, TOM flags, return data, or policy decisions."""
    calendar=pd.DatetimeIndex(calendar)
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing or calendar.tz is not None:
        raise ValueError("invalid observed native calendar")
    if not calendar.equals(calendar.normalize()) or calendar[-1]>pd.Timestamp(as_of):
        raise ValueError("observed calendar beyond as_of or non-date values")
    if timing.index.has_duplicates or not timing.index.is_monotonic_increasing:
        raise ValueError("ambiguous original signal index")
    required=["decision_at","label_available_at","return_available_at"]
    if set(timing.columns)!=set(required):raise ValueError("only frozen timestamp columns may be consumed")
    if not pd.DatetimeIndex(timing.index).isin(calendar).all():raise ValueError("signal index absent from native calendar")
    for column in required:
        if not isinstance(timing[column].dtype,pd.DatetimeTZDtype):raise ValueError("timestamp dtype or timezone lost")
    rows=[];state_cache={}
    for signal,source in timing.iterrows():
        original=source.decision_at
        day=pd.NaT if pd.isna(original) else original.tz_convert("America/New_York").tz_localize(None).normalize()
        if pd.notna(day) and original.tz_convert("America/New_York").time()!=pd.Timestamp("09:29").time():
            raise ValueError("original decision clock differs")
        within=pd.notna(day) and day<=pd.Timestamp(as_of)
        month_start=pd.NaT if pd.isna(day) else day.replace(day=1)
        prefix_complete=bool(within and calendar[0]<=month_start)
        past=calendar[calendar<day] if within else calendar[:0]
        last_past=pd.NaT if past.empty else past[-1]
        observed=bool(within and day in calendar)
        account_eligible=bool(pd.Timestamp("2023-01-03")<=signal<=pd.Timestamp("2026-10-07") and within and day<=pd.Timestamp("2026-10-08"))
        for venue in VENUES:
            for label,hour,minute in CLOCKS:
                clock=original if label=="original_0929" else pd.NaT if pd.isna(day) else day.tz_localize("America/New_York")+pd.Timedelta(hours=hour,minutes=minute)
                known=_events_at(ledger,venue,clock)
                ready,through,next_open,documents,conflict,current=_horizon(known,day if within else pd.NaT,state_cache)
                selected=known.loc[known.document_id.isin(documents)]
                extra=known.loc[known.assertion_kind.isin(["temporary_closed","reopening"])]
                if pd.notna(day):extra=extra.loc[extra.session_date.between(month_start,day+pd.offsets.MonthEnd(0))]
                else:extra=extra.iloc[:0]
                gap_ids=[gap["id"] for gap in amendment_gaps if venue in gap["venues"] and pd.notna(day)
                         and day.year in gap["affected_calendar_years"] and pd.notna(clock)
                         and clock>=pd.Timestamp(gap["unknown_from"])]
                reasons=["historical_first_vintage_unverified","receiver_venue_not_selected","extraordinary_notice_corpus_not_exhaustive"]
                reasons.extend(gap_ids)
                if not ready:reasons.append("ordinary_horizon_unknown_or_conflicting")
                if not prefix_complete:reasons.append("actual_prefix_not_covered")
                if not observed:reasons.append("current_session_or_decision_unknown")
                if venue in ("NYSE","NYSE_ARCA"):reasons.append("NYSE_Arca_exception_announcement_gaps")
                if label=="original_0929":reasons.append("current_session_not_yet_open_at_original_clock")
                else:reasons.append("current_open_has_no_archival_intraday_confirmation")
                rows.append({"signal_date":signal,"instrument":venue+"__"+label,"venue":venue,"clock":label,
                    "original_decision_at":original,"research_decision_at_0946":pd.NaT if pd.isna(day) else day.tz_localize("America/New_York")+pd.Timedelta(hours=9,minutes=46),
                    "audited_clock_at":clock,"mapped_trade_date":day,"original_label_available_at":source.label_available_at,
                    "original_return_available_at":source.return_available_at,"source_as_of":pd.Timestamp(as_of),
                    "actual_prefix_coverage_start":month_start,"actual_prefix_last_past_open":last_past,
                    "actual_prefix_complete":prefix_complete,"past_prefix_evidence_class":"historical_actual_session_snapshot_assumption_not_arrival_audit",
                    "current_date_in_observed_calendar":observed,"current_open_first_receipt_at":pd.NaT,
                    "current_open_first_receipt_verified":False,
                    "current_open_conditional_0946_assumption":bool(label=="research_0946" and observed),
                    "ordinary_horizon_complete":bool(ready),"announced_coverage_through":through,"next_planned_open_after_month_end":next_open,
                    "current_announced_state":current,"known_document_ids":"|".join(documents),
                    "known_extraordinary_document_ids":"|".join(sorted(extra.document_id.unique())),
                    "calendar_amendment_gap_present":bool(gap_ids),
                    "knowledge_max_at":pd.NaT if selected.empty else selected.knowledge_at.max(),
                    "conflict_present":bool(conflict),"extraordinary_notice_completeness":"unknown",
                    "receiving_venue_status":"unselected_unknown","historical_PIT_qualified":False,
                    "qualification_status":"conditional_snapshot" if ready and prefix_complete and observed and not conflict and not gap_ids else "unknown",
                    "unknown_reasons":"|".join(reasons),"account_eligible":account_eligible,
                    "tail_reason":"unknown_next_decision" if pd.isna(original) else "terminal_or_outside_account_interval" if signal>=pd.Timestamp("2026-10-08") else ""})
    frame=pd.DataFrame(rows).set_index(["signal_date","instrument"]).rename_axis(index=["datetime","instrument"]).sort_index()
    for name in ("original_decision_at","research_decision_at_0946","audited_clock_at","original_label_available_at",
                 "original_return_available_at","current_open_first_receipt_at","knowledge_max_at"):
        frame[name]=pd.to_datetime(frame[name],utc=True).dt.tz_convert("America/New_York")
    return frame


def calendar_discrepancies(calendar,ledger,*,start,as_of):
    """Already-known ordinary plans versus final observed dates, no relabeling."""
    rows=[]
    for day in pd.date_range(start,as_of,freq="D"):
        for venue in VENUES:
            clock=day.tz_localize("America/New_York")+pd.Timedelta(hours=9,minutes=29)
            known=_events_at(ledger,venue,clock)
            state,documents,conflict=_planned_state(known,day)
            if state=="unknown":continue
            expected=state in ("planned_weekday_open","early_close","planned_reopening")
            observed=day in calendar
            if expected!=observed or conflict:
                rows.append({"date":day,"venue":venue,"clock":"original_0929","audited_clock_at":clock,
                    "known_document_ids":"|".join(documents),
                    "knowledge_max_at":known.loc[known.document_id.isin(documents),"knowledge_at"].max(),
                    "announced_state":state,"observed_calendar_contains_date":observed,
                    "difference":"planned_open_not_in_observed_snapshot" if expected else "planned_closed_but_observed_snapshot_open",
                    "conflict_present":conflict,"knowledge_qualification_changed":False,
                    "role":"diagnostic against common observed snapshot; not proof of venue equivalence or permission for retrospective corrections"})
    return pd.DataFrame(rows,columns=["date","venue","clock","audited_clock_at","known_document_ids","knowledge_max_at","announced_state",
        "observed_calendar_contains_date","difference","conflict_present","knowledge_qualification_changed","role"])


def metadata_dataset(frame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    native=pd.concat({"meta":frame},axis=1)
    dataset=DatasetH(handler=DataHandlerLP.from_df(native),segments={"all":(None,None)})
    dataset.config(dump_all=True,recursive=True)
    return dataset


def _approval(root,card,plan):
    if os.environ.get("CALENDAR_TOM_SOURCE_EXECUTE")!="1" or not plan.get("frozen_before_source_execution"):
        raise ValueError("source qualification disabled")
    review=json.loads((card/"source_implementation_review.json").read_text())
    approval=json.loads((card/"source_execution_approval.json").read_text())
    digest=sha(card/"source_plan.json")
    if review.get("verdict")!="APPROVE_SOURCE_IMPLEMENTATION" or review.get("source_plan_sha256")!=digest:
        raise ValueError("implementation review absent or stale")
    if (approval.get("approved") is not True or approval.get("source_plan_sha256")!=digest
            or approval.get("implementation_review_sha256")!=sha(card/"source_implementation_review.json")
            or approval.get("code_sha256")!=code_identity(root)):
        raise ValueError("owner source-only approval absent or stale")
    for name,item in plan["frozen_files"].items():
        if sha(card/item["path"])!=item["sha256"]:raise ValueError("frozen source file differs: "+name)
    return {"source_plan_sha256":digest,"source_execution_approval_sha256":sha(card/"source_execution_approval.json"),
            "implementation_review_sha256":sha(card/"source_implementation_review.json"),"code_sha256":code_identity(root)}


def _cache_documents(card,plan):
    documents=[]
    for name in ("inventory","extension"):
        documents.extend(json.loads((card/plan["frozen_files"][name]["path"]).read_text())["documents"])
    if len(documents)!=17 or len({doc["id"] for doc in documents})!=17:
        raise ValueError("frozen17 document receipts differ")
    bodies={}
    for doc in documents:
        capture=doc["native_capture"]
        if capture.get("sha256") is None:
            if capture.get("temporary_raw_path") is not None:raise ValueError("failed raw capture cannot have a body")
            continue
        path=Path(capture["temporary_raw_path"])
        body=path.read_bytes()
        if hashlib.sha256(body).hexdigest()!=capture["sha256"] or len(body)!=capture["bytes"]:
            raise ValueError("cached source bytes changed: "+doc["id"])
        bodies[doc["id"]]=body
    if len(bodies)!=12:raise ValueError("exactly12 successful caches required; no refetch")
    return documents,bodies


def _mapping_check(timing,calendar):
    outer=timing.loc["2023-01-03":"2026-10-09"]
    if len(outer)!=946:raise ValueError("all946 outer timing rows must remain")
    account=outer.loc[:"2026-10-07"]
    trades=account.decision_at.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    expected=calendar[(calendar>=pd.Timestamp("2023-01-04")) & (calendar<=pd.Timestamp("2026-10-08"))]
    if (len(account)!=944 or trades.isna().any() or trades.iloc[0]!=pd.Timestamp("2023-01-04")
            or trades.iloc[-1]!=pd.Timestamp("2026-10-08") or not pd.DatetimeIndex(trades).equals(expected)):
        raise ValueError("frozen944 account mapping differs")
    if list(outer.index[-2:])!=[pd.Timestamp("2026-10-08"),pd.Timestamp("2026-10-09")]:
        raise ValueError("two non-account signal tails differ")
    return {"outer_signal_rows":946,"account_signal_rows":944,"tail_signal_rows":2,
        "tail_original_decision_at":{str(day.date()):None if pd.isna(value) else str(value) for day,value in outer.decision_at.iloc[-2:].items()},
        "training_cohort_3014_verified":False,"training_cohort_boundary":"Later label-aware model gate; source stage does not rebuild maturity/finite-return eligibility."}


def _coverage(frame):
    values=frame.reset_index();values["year"]=values.datetime.dt.year;values["month"]=values.datetime.dt.strftime("%Y-%m")
    def summarize(keys):
        return values.groupby(keys,dropna=False).agg(rows=("qualification_status","size"),
            ordinary_horizon_complete=("ordinary_horizon_complete","sum"),prefix_complete=("actual_prefix_complete","sum"),
            conditional_snapshot=("qualification_status",lambda x:int(x.eq("conditional_snapshot").sum())),
            unknown=("qualification_status",lambda x:int(x.eq("unknown").sum())),
            conflicts=("conflict_present","sum"),account_rows=("account_eligible","sum"),
            historical_PIT_rows=("historical_PIT_qualified","sum")).reset_index()
    return summarize(["venue","clock","year"]),summarize(["venue","clock","month"])


def run(*,root,card):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.calendar_knowledge import build_knowledge_ledger
    from src.data.nas_market_data import archive_nas_recorder
    root,card=Path(root),Path(card)
    plan=json.loads((card/"source_plan.json").read_text());approval=_approval(root,card,plan)
    state_path=card/"source_run_state.json"
    if state_path.exists():raise ValueError("source attempt already exists; owner repair review required before any repeat")
    state={"source_plan_sha256":approval["source_plan_sha256"],"status":"started","new_source_HTTP":0,
        "candidate_feature_values":0,"new_labels":0,"new_fits":0,"new_accounts":0}
    atomic_json(state,state_path)
    local=plan["local_sources"]["actual_daily_calendar"]
    tracking=card/"qlib_recorder"
    try:
        if sha(root/local["calendar_file"])!=local["calendar_sha256"]:raise ValueError("observed calendar source changed")
        qlib.init(provider_uri=str(root/local["provider_uri"]),region="us",kernels=1)
        # Verified read-only source access rejects an active writable Recorder.
        # Load and slice only the permitted timestamps before starting this run.
        source=plan["local_sources"]["original_timing"]
        if json.loads((root/source["card"]/"nas_binding.json").read_text())["evidence"]["sha256"]!=source["archive_sha256"]:
            raise ValueError("timing source archive identity differs")
        objects,source_audit=load_verified_recorder_objects(root/source["card"],expected_recorder_id=source["recorder_id"],
            artifacts=["inputs/timing"],expected_manifest_sha256=source["verification_manifest_sha256"],
            expected_artifact_sha256={"inputs/timing":source["artifact_sha256"]})
        timing=objects["inputs/timing"].loc[plan["bounds"]["signal_rows_start"]:plan["bounds"]["signal_rows_end"]].copy()
        with R.start(experiment_name=NAME,uri=str(tracking)):
            rec=R.get_recorder();state.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(state,state_path)
            rec.save_objects(artifact_path="inputs",plan=plan,approval=approval)
            documents,bodies=_cache_documents(card,plan)
            specs=json.loads((card/plan["frozen_files"]["parser_specs"]["path"]).read_text())
            if "document_specs" in specs:specs=specs["document_specs"]
            parsed=build_knowledge_ledger(documents,bodies,document_specs=specs)
            calendar=pd.DatetimeIndex(D.calendar(start_time=plan["bounds"]["observed_calendar_snapshot_start"],end_time=plan["bounds"]["historical_snapshot_as_of"],freq="day"))
            mapping=_mapping_check(timing,calendar)
            frame=build_date_metadata(timing,calendar,parsed["ledger"],as_of=plan["bounds"]["historical_snapshot_as_of"],
                amendment_gaps=plan["calendar_amendment_gaps"])
            dataset=metadata_dataset(frame)
            pd.testing.assert_frame_equal(dataset.prepare("all",col_set="meta"),frame)
            rec.save_objects(artifact_path="source/raw_documents",**bodies)
            rec.save_objects(artifact_path="source",documents=documents,knowledge_ledger=parsed["ledger"],
                document_status=parsed["document_status"],source_gaps=parsed["source_gaps"],decision_metadata=frame,
                metadata_dataset=dataset,actual_calendar=calendar,original_timing=timing,timing_source_audit=source_audit)
            restored=rec.load_object("source/metadata_dataset")
            pd.testing.assert_frame_equal(restored.prepare("all",col_set="meta"),frame)
            annual,monthly=_coverage(frame)
            discrepancies=calendar_discrepancies(calendar,parsed["ledger"],start=plan["bounds"]["signal_rows_start"],
                as_of=plan["bounds"]["historical_snapshot_as_of"])
            annual.to_csv(card/"coverage_by_year.csv",index=False)
            parsed["document_status"].to_csv(card/"document_status.csv",index=False)
            rec.save_objects(artifact_path="analysis",coverage_by_year=annual,coverage_by_month=monthly,mapping=mapping,
                discrepancies=discrepancies)
            rec.save_objects(artifact_path="sources",**{name.replace("/","__"):(root/name).read_text() for name in CODE_FILES})
            summary={"experiment":NAME,"decision":"source_complete_conditional_snapshot_needs_qualification_review",
                "source_plan_sha256":approval["source_plan_sha256"],"status":"completed_pending_independent_source_result_review",
                "recorder_id":rec.id,"experiment_id":str(rec.experiment_id),"recorder_uri":str(tracking),
                "recorder_local_dir":str(tracking/str(rec.experiment_id)/rec.id),"provider_uri":str(root/local["provider_uri"]),
                "region":"us","as_of":plan["bounds"]["historical_snapshot_as_of"],
                "captured_documents":12,"attempt_receipts_preserved":17,"failed_raw_document_gaps":5,
                "explicit_historical_capture_HTTP_attempts":17,"new_source_HTTP":0,"metadata_rows":len(frame),
                "signal_rows":len(timing),"venues":list(VENUES),"clocks":[x[0] for x in CLOCKS],
                "knowledge_assertions":len(parsed["ledger"]),"calendar_discrepancy_rows":len(discrepancies),
                "mapping":mapping,"dataset_direct_prepare_roundtrip":"passed",
                "historical_PIT_qualified":False,"independent_forward_qualified":False,"receiving_venue_selected":False,
                "original_training_clock_compatibility":"unresolved","feature_generation_approved":False,
                "candidate_features_generated":0,"return_labels_read_or_computed":0,"new_fits":0,"new_accounts":0,
                "scope_boundary":"Per-venue ordinary-schedule conditional coverage only; extraordinary corpus, current-open receipt and historical-vintage gaps remain. NoTOMproxy calculated.",
                "source_gaps":parsed["source_gaps"],"known_amendment_gaps":plan["calendar_amendment_gaps"],
                "ordinary_horizon_complete_meaning":plan["implementation_contract"]["ordinary_horizon_complete_meaning"],
                "artifact_policy":plan["artifact_policy"],
                "qlib_docs":["https://qlib.readthedocs.io/en/latest/component/data.html","https://qlib.readthedocs.io/en/latest/component/recorder.html"]}
            rec.save_objects(artifact_path="analysis",summary=summary)
            artifacts=Path(summary["recorder_local_dir"])/"artifacts"
            rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob("*")),metadata_rows=len(frame),source_HTTP_requests=0)
        atomic_json({"recorder_id":rec.id,"entries":[{"path":str(x.relative_to(card)),"sha256":sha(x),"bytes":x.stat().st_size}
            for x in sorted(artifacts.rglob("*")) if x.is_file()]},card/"verification_manifest.json")
        transport=json.loads((root/source["card"]/"nas_binding.json").read_text())
        atomic_json({"schema_version":"recorder_evidence_binding_v1","authority":"yanxuan_nas","ssh":transport["ssh"],
            "source_id":"calendar-tom-source-"+approval["source_plan_sha256"],"as_of":summary["as_of"],
            "source_plan":"source_plan.json"},card/"nas_binding.json")
        summary["nas_evidence"]=archive_nas_recorder(card/"nas_binding.json",workspace_root=root,card=card,
            recorder_id=rec.id,experiment_id=str(rec.experiment_id))
        atomic_json(summary,card/"summary.json")
        state.update(status="completed",native_metadata_rows=len(frame));atomic_json(state,state_path)
        (card/"conclusion.md").write_text("# 交易日历来源资格：固定cache-only结果\n\n读取12份已获准缓存，保留17次历史请求与5个缺失原文receipt；本轮0新增来源HTTP、0价格/收益读取、0TOM特征、0拟合或账户。\n\n"+
            f"完成{len(frame)}条venue×clock原生元数据，{len(parsed['ledger'])}条来源断言；946 outer和944 account日期映射保留，两个非账户尾行未补造可执行日期。\n\n"+
            "年度/月份知识覆盖和缺口留Recorder，小表为coverage_by_year.csv、document_status.csv。NYSE/Arca/Nasdaq分开；ordinary日程覆盖仅conditional_snapshot，不代表完整临时闭市、历史PIT或当前开市首收证明。09:29与09:46不互换，原训练标签兼容性及接收venue仍未决定。\n\n"+
            "来源结果待独立评审；不晋升、不因年历数量或source工作流通过而启动模型。下一步先接受来源资格边界，再另行冻结任何feature/接收器计划。\n\nRecorder：`"+rec.id+"`。\n")
        return summary
    except BaseException as error:
        state.update(status="failed",exception_type=type(error).__name__,exception=str(error));atomic_json(state,state_path)
        raise
