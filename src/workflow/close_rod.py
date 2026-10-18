"""Frozen model-free close-only recovery research; real signs require execution approval."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json

NAME = "stage1_close_rod_20261018"
ARMS = {"B_close_permission": False, "C_close_ROD_positive": True}
CODE_FILES = ["src/workflow/close_rod.py", "src/backtest/close_rod_raw.py",
    "src/backtest/close_rod_reference.py", "src/analysis/close_rod.py",
    "src/backtest/native_raw_corporate.py", "src/backtest/raw_corporate_policy.py",
    "src/backtest/qlib_continuous_position.py",
    "src/backtest/ibkr_tiered.py", "src/workflow/stage1_open_action_value.py",
    "src/workflow/intraday_recovery_direct_allocation.py", "src/analysis/stage1_domain_heads.py",
    "src/analysis/integrated_allocation.py", "src/core/recorder_sources.py",
    "src/data/qlib_native.py", "src/data/nas_market_data.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root):
    return {name: sha(Path(root)/name) for name in CODE_FILES}


def _source(root, binding):
    card = root/binding["card"]
    if binding.get("archive"):
        if json.loads((card/"nas_binding.json").read_text())["evidence"]["sha256"] != binding["archive"]:
            raise ValueError("frozen source archive differs")
    return load_verified_recorder_objects(card, expected_recorder_id=binding["recorder_id"],
        artifacts=binding["artifacts"], expected_manifest_sha256=binding["inventory_sha256"],
        expected_artifact_sha256=binding["artifact_sha256"])


def _reference_bytes(root, binding):
    """Check pinned reference bytes without loading or valuing portfolio returns."""
    card = root/binding["card"]
    if sha(card/"verification_manifest.json") != binding["inventory_sha256"]:
        raise ValueError("reference manifest differs")
    summary = json.loads((card/"summary.json").read_text())
    if summary["recorder_id"] != binding["recorder_id"]:
        raise ValueError("reference Recorder differs")
    if json.loads((card/"nas_binding.json").read_text())["evidence"]["sha256"] != binding["archive"]:
        raise ValueError("reference archive differs")
    for name, digest in binding["artifact_sha256"].items():
        if sha(Path(summary["recorder_local_dir"])/"artifacts"/name) != digest:
            raise ValueError("reference native bytes changed: "+name)
    return {"recorder_id":binding["recorder_id"],"artifact_count":len(binding["artifacts"]),
            "byte_identity_verified":True,"portfolio_objects_loaded":False}


def _provider(root, plan, source, original_points, guard):
    """Select quotes only; no comparison of two economic prices or account valuation."""
    import qlib
    from qlib.data import D
    from src.data import build_qlib_store_from_frame
    raw = source["source/dataset"].prepare("all", col_set="feature")
    if any(dtype != np.dtype("float64") for dtype in raw.dtypes):
        raise ValueError("ROD sign authority must remain native float64")
    endpoints = source["analysis/endpoint_rows"]
    late = endpoints.loc[endpoints.slot.isin(["late_known","late_entry","late_exit"])].copy()
    # Query native source prices as F32, then promote for account arithmetic.
    cache = root/plan["quote_source_provider"]
    source_rows = raw.reset_index()
    source_manifest = build_qlib_store_from_frame(source_rows, provider_uri=cache, freq="1min", region="us",
        as_of=plan["as_of"], field_map={name:name for name in raw.columns}, source="Pinned ROD source Dataset; source price projection only")
    qlib.init(provider_uri=str(cache),region="us",kernels=1)
    observed = D.features(["SOXL","SOXX"],["$open","$close"],start_time=raw.index.get_level_values("datetime").min(),
        end_time=raw.index.get_level_values("datetime").max(),freq="1min")
    additions=[]
    for row in late.itertuples():
        value=float(observed.loc[(row.instrument,row.bar_end),"$"+row.field])
        expected=np.float32(raw.loc[(row.bar_end,row.instrument),row.field])
        if value != expected:
            raise ValueError("new quote native F32 projection differs")
        additions.append({"stamp":row.price_stamp,"instrument":row.instrument,"price":value})
    extra=pd.DataFrame(additions).pivot(index="stamp",columns="instrument",values="price")[["SOXL","SOXX"]]
    points=pd.concat([original_points,extra]).sort_index()
    if points.index.has_duplicates or len(points)!=6586:
        raise ValueError("expanded matched grid must contain exactly6586 points")
    pd.testing.assert_frame_equal(points.loc[original_points.index],original_points,check_names=False,check_exact=True)
    guard_prices=guard["$raw_execution_price"].unstack("instrument")[["SOXL","SOXX"]]
    if guard_prices.index[0]<=points.index[-1]:
        raise ValueError("native guard is not strictly after terminal")
    guarded=pd.concat([points,guard_prices]).sort_index()
    fields=["open","high","low","close","execution_price","raw_execution_price"]
    rows=[{"datetime":stamp,"instrument":symbol,**{field:float(value) for field in fields}}
          for symbol in ["SOXL","SOXX"] for stamp,value in guarded[symbol].items()]
    provider=root/plan["provider_output"]
    provider_manifest=build_qlib_store_from_frame(pd.DataFrame(rows),provider_uri=provider,freq="1min",region="us",
        as_of=plan["as_of"],field_map={name:name for name in fields},source="Unchanged oldF32 anchors plus native-F32 late marks; no returns")
    qlib.init(provider_uri=str(provider),region="us",kernels=1)
    checked=D.features(["SOXL","SOXX"],["$raw_execution_price"],start_time=guarded.index[0],
        end_time=guarded.index[-1],freq="1min")["$raw_execution_price"].unstack("instrument")[["SOXL","SOXX"]]
    pd.testing.assert_frame_equal(checked.astype(float),guarded.astype(float),check_names=False,check_exact=True)
    return points,raw,{"source_projection":source_manifest,"execution":provider_manifest},provider


def prepare(*,root,card):
    """No real ROD/sign, path, opportunity count, return or passive valuation."""
    import qlib
    from qlib.data import D
    root,card=Path(root),Path(card)
    plan=json.loads((card/"plan.json").read_text());digest=sha(card/"plan.json")
    if not plan.get("frozen_before_execution"):
        raise ValueError("complete strategy plan is not frozen")
    review_path=root/plan["source_result_review"]["path"]
    if sha(review_path)!=plan["source_result_review"]["sha256"]:
        raise ValueError("source result review changed")
    review=json.loads(review_path.read_text())
    if review.get("verdict")!="ACCEPT_SOURCE_PREFLIGHT_FOR_STRATEGY_FREEZE" or review.get("strategy_freeze_eligible") is not True:
        raise ValueError("source is not approved for strategy freeze")
    if review["source_recorder_id"]!=plan["sources"]["late_source"]["recorder_id"] or review["archive_sha256"]!=plan["sources"]["late_source"]["archive"]:
        raise ValueError("source review identity differs")
    if sha(root/plan["active_release"]["path"])!=plan["active_release"]["sha256"]:
        raise ValueError("active frozen release changed")
    calendar=plan["calendar"]
    if sha(root/calendar["file"])!=calendar["sha256"]:
        raise ValueError("frozen native daily calendar changed")
    qlib.init(provider_uri=str(root/calendar["provider"]),region="us",kernels=1)
    daily=pd.DataFrame({"trade_date":pd.DatetimeIndex(D.calendar(freq="day"))})
    base,base_audit=_source(root,plan["sources"]["base"])
    source,source_audit=_source(root,plan["sources"]["late_source"])
    if base["inputs/plan"]["active_release"]!=plan["active_release"]:
        raise ValueError("base account is not frozen activeA")
    if base["inputs/plan"]["tariff"]!=plan["tariff"]:
        raise ValueError("fee projection changed")
    frame=base["inputs/frame"].copy()
    sessions=source["inputs/native_session_calendar"].copy()
    dates=pd.DatetimeIndex(frame.trade_date)
    if not dates.equals(sessions.index) or len(frame)!=944:
        raise ValueError("account/source sessions differ")
    if not source["analysis/summary"]["endpoint_gate_pass"]:
        raise ValueError("source endpoint gate is not passed")
    points,raw,manifest,provider=_provider(root,plan,source,base["inputs/raw_points"],base["inputs/guard"])
    late=sessions[["actual_close_at","regular_session"]].copy()
    late["source_ready"]=sessions.regular_session & source["analysis/event_mask"].event_source_ready
    # Preserve inputs separately. No sign or cross-price arithmetic here.
    pairs=pd.DataFrame(index=dates,columns=["previous_close_float64","close_1530_float64"],dtype=float)
    for day,row in sessions.iterrows():
        pairs.loc[day,"previous_close_float64"]=raw.loc[(row.previous_actual_close_at,"SOXX"),"close"]
        if row.regular_session:
            pairs.loc[day,"close_1530_float64"]=raw.loc[(day+pd.Timedelta(minutes=930),"SOXX"),"close"]
    reference_audits={role:_reference_bytes(root,binding) for role,binding in plan["reference_sources"].items()}
    receipt={"plan_sha256":digest,"account_sessions":944,"regular_sessions":int(sessions.regular_session.sum()),
        "half_days":int((~sessions.regular_session).sum()),"source_regular_non_event_sessions":int(late.source_ready.sum()),
        "old_anchor_points":3778,"expanded_grid_points":len(points),"old_anchor_values_exact":True,
        "new_quote_plane":"native D.features float32 promoted tofloat64","signal_plane":"verified native float64; not compared",
        "reference_objects_byte_verified":12,"reference_objects_loaded_or_valued":0,
        "new_ROD_values_or_signs":0,"new_policy_paths":0,"new_native_accounts":0,"new_model_fits":0,
        "source_audits":{"base":base_audit,"late_source":source_audit},"reference_audits":reference_audits}
    atomic_json(receipt,card/"strategy_preflight.json")
    return {"plan":plan,"plan_sha256":digest,"base":base,"source":source,"frame":frame,"late":late,
        "signal_prices_float64":pairs,"points":points,"provider":provider,"provider_manifest":manifest,
        "daily_calendar_frame":daily,"receipt":receipt,"source_audits":{"base":base_audit,"late_source":source_audit}}


def _approval(root,card,digest):
    review_path,go_path=card/"design_review.json",card/"execution_approval.json"
    if not review_path.exists() or not go_path.exists():
        raise ValueError("sign calculation and policy accounts require strategy review plus ownerGO")
    review,go=json.loads(review_path.read_text()),json.loads(go_path.read_text())
    if review.get("plan_sha256")!=digest or review.get("verdict") not in {"APPROVE_DESIGN","APPROVE_STRATEGY_DESIGN"}:
        raise ValueError("strategy design review identity differs")
    if go.get("approved") is not True or go.get("plan_sha256")!=digest or go.get("code_sha256")!=code_identity(root):
        raise ValueError("strategy execution/code approval differs")
    return {"review_sha256":sha(review_path),"execution_approval_sha256":sha(go_path),"code_sha256":go["code_sha256"]}


def _signals(root,plan,prepared):
    """Called only after full strategy execution approval; no tail return is computed."""
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from src.data import build_qlib_store_from_frame
    late=prepared["late"].copy();pairs=prepared["signal_prices_float64"]
    known=late.source_ready.astype(bool)
    if not np.isfinite(pairs.loc[known].to_numpy()).all() or not pairs.loc[known].gt(0).all().all():
        raise ValueError("known source price pair invalid")
    signs=pd.DataFrame({"known":known.astype(float),"rod_positive":0.},index=late.index)
    signs.loc[known,"rod_positive"]=(pairs.loc[known,"close_1530_float64"]>pairs.loc[known,"previous_close_float64"]).astype(float)
    rows=signs.rename_axis("datetime").reset_index();rows["instrument"]="CLOSE_ROD"
    provider=root/plan["signal_provider"]
    manifest=build_qlib_store_from_frame(rows,provider_uri=provider,freq="day",region="us",as_of=plan["as_of"],
        field_map={"known":"known","rod_positive":"rod_positive"},source="Authorized strictnativefloat64 sign; binary cache only")
    qlib.init(provider_uri=str(provider),region="us",kernels=1)
    observed=D.features(["CLOSE_ROD"],["$known","$rod_positive"],start_time=late.index[0],end_time=late.index[-1]).droplevel("instrument")
    np.testing.assert_array_equal(observed.to_numpy(),signs.to_numpy())
    late["rod_known"]=observed["$known"].to_numpy()==1
    late["rod_positive"]=observed["$rod_positive"].to_numpy()==1
    native=signs.copy();native.index=pd.MultiIndex.from_arrays([signs.index,["CLOSE_ROD"]*len(signs)],names=["datetime","instrument"])
    dataset=DatasetH(handler=DataHandlerLP.from_df(pd.concat({"feature":native},axis=1)),segments={"all":(late.index[0],late.index[-1])})
    dataset.config(dump_all=True,recursive=True)
    return late,dataset,manifest


def run(*,root,card,prepared=None):
    import qlib
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    root,card=Path(root),Path(card);digest=sha(card/"plan.json")
    approval=_approval(root,card,digest)
    p=prepare(root=root,card=card) if prepared is None else prepared
    if p["plan_sha256"]!=digest:raise ValueError("prepared plan differs")
    state_path=card/"run_state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else {"plan_sha256":digest,
        "path_attempts":[],"paths_done":[],"native_attempts":[],"native_done":[],"passive_attempts":[],"passive_done":[]}
    if state["plan_sha256"]!=digest:raise ValueError("state belongs toanotherplan")
    if state.get("completed"):return json.loads((card/"summary.json").read_text())
    rawrefs={}
    for role,binding in p["plan"]["reference_sources"].items():rawrefs[role]=_source(root,binding)[0]
    late,signal_dataset,signal_manifest=_signals(root,p["plan"],p)
    qlib.init(provider_uri=str(p["provider"]),region="us",kernels=1)
    tracking=card/"qlib_recorder";kwargs={"experiment_name":NAME,"uri":str(tracking)}
    if state.get("recorder_id"):kwargs.update(recorder_id=state["recorder_id"],resume=True)
    with R.start(**kwargs):
        rec=R.get_recorder();state.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(state,state_path)
        if not state.get("sources_saved"):
            rec.save_objects(artifact_path="inputs",plan=p["plan"],approval=approval,source_audits=p["source_audits"],
                preflight=p["receipt"],frame=p["frame"],late_inputs=late,signal_prices_float64=p["signal_prices_float64"],
                signal_dataset=signal_dataset,signal_manifest=signal_manifest,point_prices=p["points"],
                provider_manifest=p["provider_manifest"],daily_calendar_frame=p["daily_calendar_frame"])
            rec.save_objects(artifact_path="sources",**{name.replace("/","__"):(root/name).read_text() for name in CODE_FILES})
            state["sources_saved"]=True;atomic_json(state,state_path)
    rec.start_run()
    try:return _accounts(root,card,p,late,rawrefs,rec,state,state_path,tracking)
    except BaseException:
        rec.end_run(status=Recorder.STATUS_FA)
        raise


def _accounts(root,card,p,late,rawrefs,rec,state,state_path,tracking):
    from qlib.workflow.recorder import Recorder
    from src.backtest.close_rod_raw import replay_close_permission_raw
    from src.backtest.close_rod_reference import passive_expand_native_book
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    from src.analysis.close_rod import analyze_accounts
    from src.data.nas_market_data import archive_nas_recorder
    plan,base=p["plan"],p["base"];books={};activity=[]
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**base["inputs/recovery_spec"])
    action=Stage1OpenActionValueSpec(root=root,**base["inputs/action_spec"])
    for family,gate in ARMS.items():
        for cost in plan["accounts"]["slippage_bps"]:
            key=f"{family}_{cost}"
            if key in state["paths_done"]:path=rec.load_object("paths/"+key)
            else:
                if key in state["path_attempts"] or len(state["path_attempts"])>=6:raise ValueError("path attempt consumed")
                state["path_attempts"].append(key);atomic_json(state,state_path)
                path=replay_close_permission_raw(p["frame"],point_prices=p["points"],schedule=base["inputs/schedule"],
                    plan={"initial_cash":plan["accounts"]["initial_cash"],"tariff":plan["tariff"]},controller=base["inputs/controller"],
                    recovery_spec=recovery,action_spec=action,selection=base["inputs/selection"],slippage_bps=cost,
                    late_inputs=late,require_ROD=gate)
                rec.save_objects(artifact_path="paths",**{key:path});state["paths_done"].append(key);atomic_json(state,state_path)
            if key in state["native_done"]:book=rec.load_object("native/"+key)
            else:
                if key in state["native_attempts"] or len(state["native_attempts"])>=6:raise ValueError("native attempt consumed")
                state["native_attempts"].append(key);atomic_json(state,state_path)
                book=native_raw_validate(path,provider_uri=p["provider"],point_prices=p["points"],schedule=base["inputs/schedule"],
                    tariff=plan["tariff"],cost=cost,initial_cash=plan["accounts"]["initial_cash"])
                book["late_events"]=path["late_events"]
                rec.save_objects(artifact_path="native",**{key:book});state["native_done"].append(key);atomic_json(state,state_path)
            books[(family,cost)]=book;events=path["late_events"]
            activity.append({"family":family,"slippage_bps":cost,"late_trade_days":int(events.late_action.sum()),
                "all_incremental_lots_closed":bool(events.lot_closed.all()),"sum_roundtrip_cash_change_usd":float(events.roundtrip_pnl_usd.sum())})
            rec.log_metrics(completed_native_accounts=len(state["native_done"]));print("completed native",key,flush=True)
    for item in plan["reference_books"]:
        family,cost=item["family"],item["cost"];key=f"{family}_{cost}";original=rawrefs[item["source_role"]][item["artifact"]]
        if key in state["passive_done"]:saved=rec.load_object("passive/"+key)
        else:
            if key in state["passive_attempts"] or len(state["passive_attempts"])>=12:raise ValueError("passive valuation attempt consumed")
            state["passive_attempts"].append(key);atomic_json(state,state_path)
            expanded=passive_expand_native_book(original,point_prices=p["points"],source_identity={
                "original_point_prices":base["inputs/raw_points"],"recorder_id":plan["reference_sources"][item["source_role"]]["recorder_id"],
                "artifact_sha256":plan["reference_sources"][item["source_role"]]["artifact_sha256"][item["artifact"]],
                "source_manifest_sha256":plan["sources"]["late_source"]["inventory_sha256"]})
            saved={name:expanded[name] for name in ["points","passive_added_points","original_native_audit","passive_expansion_audit","evidence_roles"]}
            rec.save_objects(artifact_path="passive",**{key:saved});state["passive_done"].append(key);atomic_json(state,state_path)
        books[(family,cost)]={**original,**saved}
    results=analyze_accounts(books,p["daily_calendar_frame"],plan)
    for name,value in results.items():
        if isinstance(value,pd.DataFrame):
            if name in {"metrics","paired_comparison","annual_metrics","quarterly_metrics","rolling_summary","uncertainty"}:
                if len(value)>500:raise ValueError("large root sidecar forbidden")
                value.to_csv(card/(name+".csv"),index=False)
            rec.save_objects(artifact_path="analysis",**{name:value})
        elif name=="rolling":rec.save_objects(artifact_path="analysis/rolling",**value)
    activity=pd.DataFrame(activity);activity.to_csv(card/"late_activity.csv",index=False);rec.save_objects(artifact_path="analysis",late_activity=activity)
    economic=all(value for name,value in results["gates"].items() if name not in ["qualified_independent_forward","fresh_fullminute_risk_qualified"])
    no_op=not activity.late_trade_days.gt(0).any()
    summary={"experiment":NAME,"plan_sha256":p["plan_sha256"],"decision":"not_promoted_no_executable_lots_fixed_ROD_v1" if no_op else
        "historical_economic_pass_needs_risk_and_independent_forward" if economic else "not_promoted_fixed_close_ROD_v1",
        "gates":results["gates"],"coarse_economic_gates_pass":economic,"no_executable_lots":no_op,
        "objective_achieved":False,"new_fits":0,"new_policy_paths":6,"new_native_accounts":6,
        "reused_native_reference_accounts":12,"passive_reference_revaluations":12,"total_comparison_accounts":18,
        "reference_added_marks_native_validated":False,"sessions_each":944,"points_each":6586,
        "recorder_id":rec.id,"experiment_id":str(rec.experiment_id),"recorder_uri":str(tracking),
        "recorder_local_dir":str(tracking/str(rec.experiment_id)/rec.id),"provider_uri":str(p["provider"]),
        "as_of":plan["as_of"],"historical_PIT_qualified":False,"independent_forward_qualified":False,
        "history_status":"all_history_reused_not_independent","production_changed":False,
        "artifact_policy":{"policy":"slim_card_recorder_first","workflow_config":"workflow.yaml",
            "recorder_only_artifacts":["inputs/*","sources/*","paths/*","native/*","passive/*","analysis/rolling/*"]}}
    rec.save_objects(artifact_path="analysis",summary=summary,gates=results["gates"])
    artifacts=Path(summary["recorder_local_dir"])/"artifacts";rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob("*")))
    rec.end_run(status=Recorder.STATUS_FI)
    atomic_json({"recorder_id":rec.id,"entries":[{"path":str(x.relative_to(card)),"sha256":sha(x),"bytes":x.stat().st_size} for x in sorted(artifacts.rglob("*")) if x.is_file()]},card/"verification_manifest.json")
    transport=json.loads((root/plan["sources"]["base"]["card"]/"nas_binding.json").read_text())
    atomic_json({"schema_version":"recorder_evidence_binding_v1","authority":"yanxuan_nas","ssh":transport["ssh"],
        "source_id":"close-rod-strategy-"+p["plan_sha256"],"as_of":plan["as_of"],"source_plan":"plan.json"},card/"nas_binding.json")
    summary["nas_evidence"]=archive_nas_recorder(card/"nas_binding.json",workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    atomic_json(summary,card/"summary.json");state["completed"]=True;atomic_json(state,state_path)
    lines=["# 尾盘ROD新增受限恢复许可：固定结果","","原f696+b472晨间链保留；B/C共享新增尾盘许可，C仅多严格float64价格方向门。0拟合、6新原生账户；12旧native参考只被动增加估值点，不能称作12次新native验证。",
        "","|方案|滑点bp|净收益%|收盘DD%|全采样点DD%|","|---|---:|---:|---:|---:|"]
    for row in results["metrics"].itertuples():lines.append(f"|{row.family}|{row.slippage_bps}|{row.return_pct:.3f}|{row.close_max_drawdown_pct:.3f}|{row.sampled_point_max_drawdown_pct:.3f}|")
    lines += ["","决定："+summary["decision"]+"。失败或未获得的门："+", ".join(name for name,value in results["gates"].items() if not value)+"。",
        "","全部年/季度（含partial）、126日滚动及12项置信下界均保留，不择窗；费用情景各自NAV/风险/费用状态独立。新增整数lot按15:59精确退出，不重置财富。来源为条件历史snapshot，假设延迟并非实盘首收或auction成交证明，重复历史不是前向独立证据。",
        "","若没有可执行整数lot，应解释为机会集无信息，不能声称检验出ROD无效；固定版本失败则关闭，不改符号、窗口、caps、时钟或退出参数救援。执行者不自签promotion，等待独立结果评估。",
        "","Recorder：`"+rec.id+"`。来源结论独立保存在source_preflight/summary.json；稳定超额目标未完成。"]
    (card/"conclusion.md").write_text("\n".join(lines)+"\n")
    (card/"workflow.yaml").write_text("experiment: "+NAME+"\nentrypoint: src.workflow.close_rod.run\nphase: completed\nartifact_policy: slim_card_recorder_first\n")
    return summary
