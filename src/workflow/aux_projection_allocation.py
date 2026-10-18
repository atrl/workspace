"""Three C-only fits with training-only data access and pinned original B reuse."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.model.aux_projection_allocator import AuxProjectionAllocator
from src.workflow.aux_projection_preflight import (
    _read_training_source, _prepare_training, _software, state_sha, SHARED, PRIVATE,
)
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES, _standardize_pair

NAME = "stage1_aux_projection_allocation_20261018"
CANDIDATE = "C_aux_projection"
SEEDS = [71, 172, 273]
DATES = ["signal_date", "trade_date", "exit_date"]
OUTPUTS = ["soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability"]
REQUIRED = ("A_active_research_replay", "B_shared", "SOXX", "strong9097")
FAMILIES = (*REQUIRED, CANDIDATE, "additional27baa")
CODE_FILES = ["src/workflow/aux_projection_allocation.py", "src/model/aux_projection_allocator.py",
    "src/workflow/aux_projection_preflight.py", "src/model/minute_pit_allocator.py",
    "src/workflow/minute_pit_allocator.py", "src/workflow/pit_constituent_auxiliary_training_allocation.py",
    "src/backtest/stage1_domain_heads_raw.py", "src/backtest/native_raw_corporate.py",
    "src/backtest/qlib_continuous_position.py", "src/backtest/raw_corporate_policy.py",
    "src/backtest/ibkr_tiered.py", "src/workflow/stage1_open_action_value.py",
    "src/workflow/intraday_recovery_direct_allocation.py", "src/analysis/stage1_domain_heads.py",
    "src/analysis/integrated_allocation.py", "src/core/recorder_sources.py", "src/data/qlib_native.py",
    "src/data/nas_market_data.py", "src/workflow/stage1_domain_heads.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root):
    return {name: sha(Path(root)/name) for name in CODE_FILES}


def verify_binding(root, source):
    """Full-object byte verification only; never deserializes evaluation content."""
    card=Path(root)/source["card"]
    if sha(card/"verification_manifest.json")!=source["inventory_sha256"]:
        raise ValueError("source manifest identity differs")
    summary=json.loads((card/"summary.json").read_text())
    if summary["recorder_id"]!=source["recorder_id"]:
        raise ValueError("source Recorder identity differs")
    if json.loads((card/"nas_binding.json").read_text())["evidence"]["sha256"]!=source["archive"]:
        raise ValueError("source archive identity differs")
    for name,digest in source["artifact_sha256"].items():
        if sha(Path(summary["recorder_local_dir"])/"artifacts"/name)!=digest:
            raise ValueError("source artifact changed: "+name)
    return {"recorder_id":source["recorder_id"],"artifacts":list(source["artifacts"]),
            "bytes_verified":True,"objects_deserialized":False}


def read_binding(root, source):
    verify_binding(root,source)
    return load_verified_recorder_objects(Path(root)/source["card"],expected_recorder_id=source["recorder_id"],
        artifacts=source["artifacts"],expected_manifest_sha256=source["inventory_sha256"],
        expected_artifact_sha256=source["artifact_sha256"])


def _new_model(original, root):
    candidate=AuxProjectionAllocator(base_features=original.base_features,extra_features=[],feature_masks=original.feature_masks,
        spec={**asdict(original.spec),"root":str(root)},seed=original.seed,cutoff=str(original.cutoff),
        train_end=str(original.train_end.date()),minimum_observations=original.minimum_observations)
    source_spec={key:value for key,value in asdict(original.spec).items() if key!="root"}
    candidate_spec={key:value for key,value in asdict(candidate.spec).items() if key!="root"}
    if candidate_spec!=source_spec or asdict(candidate.candidate)!=asdict(original.candidate):
        raise ValueError("complete original spec/objective candidate contract differs")
    for name in ("base_features","extra_features","feature_masks","cutoff","train_end","seed","minimum_observations"):
        if getattr(candidate,name)!=getattr(original,name):raise ValueError("constructor contract differs: "+name)
    return candidate


def _reviews(root,plan):
    reviews={}
    for name,item in plan["reviews"].items():
        path=root/item["path"]
        if sha(path)!=item["sha256"]:raise ValueError("review identity changed: "+name)
        reviews[name]=json.loads(path.read_text())
    if reviews["mechanical"]["verdict"]!="ACCEPT_MECHANICAL_RESULT_ONLY":raise ValueError("mechanical result not accepted")
    if reviews["model_implementation"]["verdict"]!="ACCEPT_MODEL_IMPLEMENTATION_ONLY":
        raise ValueError("model implementation not accepted")
    if sha(root/"src/model/aux_projection_allocator.py")!=reviews["model_implementation"]["model_code_sha256"]:
        raise ValueError("reviewed model implementation changed")
    parity=reviews["B_reproduction"]["training_checks"]
    if not parity["model_scalers_match_frozen_source"] or [r["seed"] for r in parity["B_parity"]]!=SEEDS:
        raise ValueError("B reproduction coverage differs")
    if any(r["network_tensor_max_abs_diff"]!=0 or r["prediction_max_abs_diff"]!=0 for r in parity["B_parity"]):
        raise ValueError("B sealed reproduction not exact; no fallback fit permitted")
    return reviews


def prepare(*,root,card,write_receipt=True):
    """Train-only native reconstruction, exact tensors/scalers/backend; no gradients."""
    import qlib
    root,card=Path(root),Path(card)
    plan=json.loads((card/"plan.json").read_text());digest=sha(card/"plan.json")
    if not plan.get("frozen_before_execution"):raise ValueError("formal plan not frozen")
    reviews=_reviews(root,plan)
    torch.set_num_threads(1)
    if _software()!=plan["software"]:raise ValueError("frozen computation backend differs")
    if sha(root/plan["active_release"]["path"])!=plan["active_release"]["sha256"]:
        raise ValueError("active release differs")
    qlib.init(provider_uri=str(root/plan["provider_init"]),region="us",kernels=1)
    training_plan={"source":plan["training_source"],"train_end":plan["training"]["train_end"],"cutoff":plan["training"]["cutoff"]}
    verify_binding(root,plan["training_source"])
    objects,source_audits=_read_training_source(root,training_plan)
    dataset,auxiliary,tensors,training_audit=_prepare_training(objects,training_plan)
    for item in (dataset,auxiliary):item.config(dump_all=True,recursive=True)
    mechanical,mechanical_audit=read_binding(root,plan["mechanical_source"])
    expected=mechanical["inputs/tensors"];expected_audit=mechanical["inputs/training_audit"]
    if training_audit["tensor_sha256"]!=expected_audit["tensor_sha256"]:
        raise ValueError("training tensor SHA differs from sealed mechanics")
    for name,value in tensors.items():
        if not torch.equal(value,expected[name]):raise ValueError("training tensor values differ: "+name)
    train=dataset.prepare("train",col_set=["feature","common"])
    original=objects["models/base_outer_71"]
    primary,_,p_scaler=_standardize_pair(train["feature"],train["feature"],original.base_features)
    common,_,c_scaler=_standardize_pair(train["common"],train["common"],COMMON_FEATURES)
    p_dict={key:value.to_dict() for key,value in p_scaler.items()}
    c_dict={key:value.to_dict() for key,value in c_scaler.items()}
    if not torch.equal(primary,expected["primary"]) or not torch.equal(common,expected["common"]):
        raise ValueError("recomputed standardize_pair tensors differ before optimization")
    initial=[]
    for seed in SEEDS:
        old=objects[f"models/base_outer_{seed}"]
        if p_dict!=old.primary_scaler or c_dict!=old.common_scaler:
            raise ValueError("recomputed mature-training scalers differ from source")
        left=old._initialize();rng=torch.get_rng_state().clone()
        candidate=_new_model(old,root);candidate.network=candidate._initialize()
        candidate._validate_projection_contract(3014,20741)
        if not torch.equal(rng,torch.get_rng_state()):raise ValueError("initialization RNG differs")
        for name,value in left.state_dict().items():
            if not torch.equal(value,candidate.network.state_dict()[name]):raise ValueError("initial tensor differs: "+name)
        initial.append({"seed":seed,"initial_state_sha256":state_sha(left),"tensors_and_rng_exact":True,
                        "parameters":973,"shared_parameters":154,"private_parameters":819,"forward_or_gradient_calls":0,
                        "complete_spec_except_root_candidate_and_constructor_equal":True})
    evaluation_bytes={name:verify_binding(root,source) for name,source in plan["deferred_evaluation_sources"].items()}
    receipt={"plan_sha256":digest,"training_audit":training_audit,"initialization":initial,
        "standardize_pair_scalers_equal_all_three_sealed_sources":True,
        "standardize_pair_primary_common_equal_archived_tensors":True,"all_training_tensors_elementwise_and_SHA_equal":True,
        "primary_scaler_sha256":hashlib.sha256(json.dumps(p_dict,sort_keys=True).encode()).hexdigest(),
        "common_scaler_sha256":hashlib.sha256(json.dumps(c_dict,sort_keys=True).encode()).hexdigest(),
        "backend_exact":True,"evaluation_byte_checks":evaluation_bytes,
        "evaluation_artifacts_deserialized":0,"evaluation_datasets_created":0,"new_model_fits":0,
        "new_forward_or_gradient_probes":0,"optimizer_steps":0,"new_accounts":0,"B_new_fits":0,
        "source_audits":source_audits,"mechanical_audit":mechanical_audit}
    if write_receipt:
        if (card/"execution_approval.json").exists():
            raise ValueError("approved preflight is immutable; use write_receipt=False")
        atomic_json(receipt,card/"preflight.json")
    return {"plan":plan,"plan_sha256":digest,"dataset":dataset,"auxiliary":auxiliary,
        "tensors":tensors,"originals":objects,"training_audit":training_audit,"receipt":receipt,"reviews":reviews}


def _approval(root, card, digest):
    if os.environ.get("AUX_PROJECTION_EXECUTE") != "1":
        raise ValueError("formal execution is disabled")
    review_path, approval_path = card/"design_review.json", card/"execution_approval.json"
    review = json.loads(review_path.read_text())
    approval = json.loads(approval_path.read_text())
    if review.get("verdict") != "APPROVE_DESIGN" or review.get("plan_sha256") != digest:
        raise ValueError("independent formal design approval missing or stale")
    if (approval.get("approved") is not True or approval.get("plan_sha256") != digest
            or approval.get("design_review_sha256") != sha(review_path)
            or approval.get("preflight_sha256") != sha(card/"preflight.json")
            or approval.get("constructor_preflight_sha256") != sha(card/"constructor_preflight.json")
            or approval.get("code_sha256") != code_identity(root)):
        raise ValueError("owner approval, source preflight or code identity differs")
    return {"design_review_sha256":sha(review_path), "execution_approval_sha256":sha(approval_path),
            "code_sha256":code_identity(root)}


def _claim(state, state_path, phase, key, maximum):
    attempts=state[phase+"_attempts"]
    if key in attempts or len(attempts)>=maximum:
        raise ValueError(phase+" attempt consumed; no silent retry or fallback")
    attempts.append(key)
    atomic_json(state,state_path)


def _failure_model(rec, model, key, error):
    traces=getattr(model,"training_trace",[])
    receipt={"exception_type":type(error).__name__,"exception":str(error),
        "numerical_audit":getattr(error,"audit",None),
        "completed_optimizer_steps":sum(row.get("status")!="failed_before_step" for row in traces),
        "actual_optimizer_state_available":hasattr(model,"optimizer_state_dict"),
        "restart_or_resume_authorized":False}
    objects={"receipt":receipt,"training_trace":traces,
             "gradient_trace":getattr(model,"gradient_trace",[])}
    if getattr(model,"network",None) is not None:
        objects["network_state"]={name:value.detach().clone() for name,value in model.network.state_dict().items()}
    if hasattr(model,"optimizer_state_dict"):
        objects["actual_optimizer_state"]=model.optimizer_state_dict
    rec.save_objects(artifact_path="failed_fits/"+key,**objects)
    return receipt


def _fixed_models(rec, state, artifact_root):
    expected=[f"{CANDIDATE}_{seed}" for seed in SEEDS]
    if state["fits_done"]!=expected or set(state.get("fixed_model_sha256",{}))!=set(expected):
        raise ValueError("evaluation remains locked until all three C models are finalized")
    for key in expected:
        if sha(artifact_root/"models"/key)!=state["fixed_model_sha256"][key]:
            raise ValueError("final C model bytes changed after fit: "+key)
    return {seed:rec.load_object(f"models/{CANDIDATE}_{seed}") for seed in SEEDS}


def _evaluation(root, plan, models):
    """Called only after the durable three-model barrier; never during prepare/fit."""
    from src.model.minute_pit_allocator import TARGETS
    from src.workflow.minute_pit_allocator import _dataset, _prediction_dates, _ensemble
    from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
    from src.analysis.stage1_domain_heads import information_metrics
    if set(models)!=set(SEEDS):raise ValueError("incomplete fixed C ensemble")
    sources,audits={},{}
    for name,source in plan["deferred_evaluation_sources"].items():
        sources[name],audits[name]=read_binding(root,source)
    source=sources["evaluation_data"]
    frame=source["inputs/daily_frame"]
    bounds=tuple(plan["prediction_bounds"])
    index=frame.loc[bounds[0]:bounds[1]].index
    dataset=_dataset(source["inputs/features"].loc[index],frame.loc[index,list(TARGETS)],
        source["inputs/common"].reindex(index),source["inputs/timing"].reindex(index),{"predict":bounds})
    dataset.config(dump_all=True,recursive=True)
    predictions={}
    for seed in SEEDS:
        predictions[(CANDIDATE,seed)]=_prediction_dates(models[seed].predict(dataset))
        predictions[("B_shared",seed)]=sources["B_reuse"][f"predictions/B_shared_{seed}"]
        pd.testing.assert_frame_equal(predictions[(CANDIDATE,seed)].iloc[:0],
                                      predictions[("B_shared",seed)].iloc[:0])
        pd.testing.assert_index_equal(predictions[(CANDIDATE,seed)].index,
                                      predictions[("B_shared",seed)].index)
    ensembles={family:_ensemble(predictions,family,SEEDS) for family in ("B_shared",CANDIDATE)}
    account=sources["account_inputs"]
    current=account["inputs/frame"].reset_index(drop=True).copy()
    market=account["inputs/market"].reset_index(drop=True)
    pd.testing.assert_frame_equal(current[DATES],market[DATES])
    if len(current)!=944 or len(account["inputs/raw_points"])!=3778:
        raise ValueError("original account calendar/grid differs")
    dates=pd.DatetimeIndex(current.signal_date)
    decision=source["inputs/timing"].reindex(dates).decision_at
    if decision.isna().any() or not pd.DatetimeIndex(decision.dt.tz_localize(None).dt.normalize()).equals(pd.DatetimeIndex(current.trade_date)):
        raise ValueError("non-executable prediction tail entered the account")
    values=ensembles[CANDIDATE].reindex(dates)
    if not np.isfinite(values.to_numpy(float)).all():raise ValueError("missing C account predictions")
    original=models[SEEDS[0]]
    target=_smooth_weights(values,spec=original.spec,candidate=original.candidate,
        asset_returns=market.set_index("signal_date")[["soxl_return","soxx_return"]])
    points,schedule=account["inputs/raw_points"],account["inputs/schedule"]
    strong=sources["strong_native"]
    pd.testing.assert_frame_equal(points,strong["inputs/raw_points"])
    if plan["deferred_evaluation_sources"]["account_inputs"]["artifact_sha256"]["inputs/schedule"] != plan["deferred_evaluation_sources"]["strong_native"]["artifact_sha256"]["inputs/schedule"]:
        raise ValueError("reference corporate schedule bytes differ")
    for source_plan in (account["inputs/plan"],strong["inputs/plan"]):
        source_accounts=source_plan.get("accounts",source_plan)
        if (source_plan["tariff"]!=plan["tariff"] or source_accounts["initial_cash"]!=plan["accounts"]["initial_cash"]
                or source_accounts["slippage_bps"]!=[0,2,5]):
            raise ValueError("reference tariff/cost/capital contract differs")
    references={}
    for item in plan["reference_books"]:
        book=sources[item["source_role"]][item["artifact"]]
        pd.testing.assert_frame_equal(book["authority"][DATES].reset_index(drop=True),current[DATES])
        if not pd.DatetimeIndex(book["points"].stamp).equals(points.index) or book["audit"]["status"]!="passed":
            raise ValueError("reference grid or native evidence differs")
        np.testing.assert_allclose(book["points"].iloc[0].nav,plan["accounts"]["initial_cash"],rtol=0,atol=1e-8)
        references[(item["family"],item["cost"])]=book
    if len(references)!=15:raise ValueError("exactly fifteen sealed references required")
    info=information_metrics(predictions,ensembles,dataset,as_of=plan["accounts"]["terminal"])
    prepared={"plan":plan,"frame":current,"market":market,"points":points,"schedule":schedule,
        "guard":account["inputs/guard"],"controller":account["inputs/controller"],
        "selection":account["inputs/selection"],"recovery_spec":account["inputs/recovery_spec"],
        "action_spec":account["inputs/action_spec"],"references":references,"daily_calendar_frame":frame}
    return prepared,dataset,predictions,ensembles,target,info,audits


def run(*,root,card,prepared=None):
    """One fixed execution; all fits precede any evaluation-object deserialization."""
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    root,card=Path(root),Path(card)
    digest=sha(card/"plan.json")
    approval=_approval(root,card,digest)
    p=prepared if prepared is not None else prepare(root=root,card=card,write_receipt=False)
    plan=p["plan"]
    if p["plan_sha256"]!=digest or _software()!=plan["software"]:
        raise ValueError("prepared plan/backend differs")
    if (plan["maximum_model_fits"],plan["maximum_new_native_accounts"],plan["reused_native_accounts"])!=(3,3,15):
        raise ValueError("fixed execution budget differs")
    state_path=card/"run_state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else {
        "plan_sha256":digest,"fit_attempts":[],"fits_done":[],"path_attempts":[],"paths_done":[],
        "native_attempts":[],"native_done":[],"fixed_model_sha256":{},"B_fit_attempts":0}
    if state["plan_sha256"]!=digest:raise ValueError("attempt ledger plan differs")
    if state.get("completed"):return json.loads((card/"summary.json").read_text())
    if state.get("failed"):raise ValueError("failed execution requires scoped owner repair review; no automatic retry")
    tracking=card/"qlib_recorder"
    kwargs={"experiment_name":NAME,"uri":str(tracking)}
    if state.get("recorder_id"):kwargs.update(recorder_id=state["recorder_id"],resume=True)
    with R.start(**kwargs):
        rec=R.get_recorder()
        state.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id),phase="training")
        atomic_json(state,state_path)
        artifact_root=tracking/str(rec.experiment_id)/rec.id/"artifacts"
        if not state.get("training_inputs_saved"):
            rec.save_objects(artifact_path="inputs",plan=plan,preflight=p["receipt"],approval=approval,
                training_dataset=p["dataset"],auxiliary_dataset=p["auxiliary"],training_audit=p["training_audit"],
                source_audits=p["receipt"]["source_audits"],reviews=p["reviews"])
            rec.save_objects(artifact_path="sources",**{name.replace("/","__"):(root/name).read_text() for name in CODE_FILES})
            state["training_inputs_saved"]=True
            atomic_json(state,state_path)
        for seed in SEEDS:
            key=f"{CANDIDATE}_{seed}"
            if key in state["fits_done"]:continue
            _claim(state,state_path,"fit",key,3)
            model=_new_model(p["originals"][f"models/base_outer_{seed}"],root)
            try:
                model.fit(p["dataset"],auxiliary_dataset=p["auxiliary"])
                original=p["originals"][f"models/base_outer_{seed}"]
                for name in ("training_rows","auxiliary_rows","last_mature_label_at","epochs","checkpoint_policy"):
                    if model.audit[name]!=original.audit[name]:raise ValueError("training contract differs: "+name)
                if model.primary_scaler!=original.primary_scaler or model.common_scaler!=original.common_scaler:
                    raise ValueError("fit standardization differs from preflight")
                if model.audit["completed_optimizer_steps"]!=300:raise ValueError("incomplete final300 model")
                rec.save_objects(artifact_path="models",**{key:model})
                rec.save_objects(artifact_path="optimizer",**{key:model.optimizer_state_dict})
                rec.save_objects(artifact_path="training_trace",**{key:model.training_trace})
                rec.save_objects(artifact_path="gradients",**{key:model.gradient_trace})
                state["fixed_model_sha256"][key]=sha(artifact_root/"models"/key)
                state["fits_done"].append(key)
                atomic_json(state,state_path)
                rec.log_metrics(completed_C_fits=len(state["fits_done"]))
                print("completed C fit",key,flush=True)
            except BaseException as error:
                state.update(failed=True,failed_phase="training",failure=_failure_model(rec,model,key,error))
                atomic_json(state,state_path)
                raise
        models=_fixed_models(rec,state,artifact_root)
        state.update(all_models_fixed=True,evaluation_unlocked=True,phase="models_fixed_before_evaluation")
        rec.save_objects(artifact_path="inputs",evaluation_unlock={"fixed_model_sha256":state["fixed_model_sha256"],
            "completed_fits":list(state["fits_done"]),"evaluation_objects_read_before_barrier":0})
        atomic_json(state,state_path)
    # No active global R.start while source reads or native validators reinitialize Qlib.
    # Continue the same native Recorder through its documented lower-level lifecycle.
    rec.start_run()
    try:
        if state.get("evaluation_done"):
            e=rec.load_object("evaluation/prepared")
            target=rec.load_object("analysis/smoothed_target")
            info=rec.load_object("analysis/classification")
        else:
            state["phase"]="evaluation"
            atomic_json(state,state_path)
            e,dataset,predictions,ensembles,target,info,audits=_evaluation(root,plan,models)
            rec.save_objects(artifact_path="evaluation",prepared=e,dataset=dataset,source_audits=audits)
            rec.save_objects(artifact_path="predictions",**{f"{family}_{seed}":value for (family,seed),value in predictions.items()})
            rec.save_objects(artifact_path="analysis",ensembles=ensembles,smoothed_target=target,classification=info)
            state["evaluation_done"]=True
            atomic_json(state,state_path)
        training=pd.DataFrame([{"family":CANDIDATE,"seed":seed,**models[seed].audit} for seed in SEEDS])
        return _accounts(root,card,plan,digest,rec,state,state_path,e,target,info,training,tracking)
    except BaseException as error:
        state.update(failed=True,failed_phase=state["phase"],failure={"type":type(error).__name__,"message":str(error)})
        atomic_json(state,state_path)
        rec.save_objects(artifact_path="execution",failure=state)
        rec.end_run(status=Recorder.STATUS_FA)
        raise


def _accounts(root,card,plan,digest,rec,state,state_path,e,target,info,training,tracking):
    from qlib.workflow.recorder import Recorder
    from src.workflow.stage1_domain_heads import _provider
    from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    from src.analysis.stage1_domain_heads import analyze_paired_accounts
    from src.data.nas_market_data import archive_nas_recorder
    provider,manifest=_provider(root,e)
    rec.save_objects(artifact_path="inputs",native_provider_manifest=manifest)
    recovery=IntradayRecoveryDirectAllocationSpec(root=root,**e["recovery_spec"])
    action=Stage1OpenActionValueSpec(root=root,**e["action_spec"])
    frame=e["frame"].copy()
    for name in OUTPUTS:frame[name]=target[name].to_numpy(copy=True)
    execution={"initial_cash":plan["accounts"]["initial_cash"],"tariff":plan["tariff"]}
    books=dict(e["references"])
    for cost in (0,2,5):
        key=f"{CANDIDATE}_{cost}"
        state["phase"]="path"
        if key in state["paths_done"]:path=rec.load_object("paths/"+key)
        else:
            _claim(state,state_path,"path",key,3)
            path=replay_stage1_raw(frame,point_prices=e["points"],schedule=e["schedule"],plan=execution,
                controller=e["controller"],recovery_spec=recovery,action_spec=action,selection=e["selection"],slippage_bps=cost)
            rec.save_objects(artifact_path="paths",**{key:path})
            state["paths_done"].append(key)
            atomic_json(state,state_path)
        state["phase"]="native"
        if key in state["native_done"]:book=rec.load_object("native/"+key)
        else:
            _claim(state,state_path,"native",key,3)
            book=native_raw_validate(path,provider_uri=provider,point_prices=e["points"],schedule=e["schedule"],
                tariff=plan["tariff"],cost=cost,initial_cash=plan["accounts"]["initial_cash"])
            rec.save_objects(artifact_path="native",**{key:book})
            state["native_done"].append(key)
            atomic_json(state,state_path)
        books[(CANDIDATE,cost)]=book
        rec.log_metrics(completed_native_accounts=len(state["native_done"]))
        print("completed C native account",key,flush=True)
    state["phase"]="analysis"
    if state.get("analysis_done"):results=rec.load_object("analysis/results")
    else:
        if state.get("analysis_attempted"):raise ValueError("analysis attempt consumed; bootstrap cannot be silently redrawn")
        state["analysis_attempted"]=True
        atomic_json(state,state_path)
        results=analyze_paired_accounts(books,e["daily_calendar_frame"],plan,candidate_family=CANDIDATE,
            required_comparators=REQUIRED,expected_families=FAMILIES)
        rec.save_objects(artifact_path="analysis",results=results)
        state["analysis_done"]=True
        atomic_json(state,state_path)
    small={"metrics","paired_comparison","annual_metrics","quarterly_metrics","rolling_summary","uncertainty"}
    for name,value in results.items():
        if name in small:
            if not isinstance(value,pd.DataFrame) or len(value)>500:raise ValueError("invalid slim summary: "+name)
            value.to_csv(card/(name+".csv"),index=False)
        if name=="rolling":rec.save_objects(artifact_path="analysis/rolling",**value)
        else:rec.save_objects(artifact_path="analysis",**{name:value})
    training.to_csv(card/"training_summary.csv",index=False)
    info.to_csv(card/"information_metrics.csv",index=False)
    rec.save_objects(artifact_path="analysis",training=training)
    economic=all(value for key,value in results["gates"].items()
        if key not in ("qualified_independent_forward","fresh_fullminute_risk_qualified"))
    summary={"experiment":NAME,"plan_sha256":digest,"decision":
        "historical_economic_pass_needs_risk_and_independent_forward" if economic else "not_promoted_fixed_aux_projection",
        "objective_achieved":False,"promotion_approved":False,"gates":results["gates"],"coarse_economic_gates_pass":economic,
        "new_fits":3,"B_new_fits":0,"new_native_accounts":3,"reused_native_accounts":15,"total_comparison_accounts":18,
        "sessions_each":944,"native_points_each":3778,"fixed_model_sha256":state["fixed_model_sha256"],
        "recorder_id":rec.id,"experiment_id":str(rec.experiment_id),"recorder_uri":str(tracking),
        "recorder_local_dir":str(tracking/str(rec.experiment_id)/rec.id),"provider_uri":str(provider),
        "as_of":plan["as_of"],"history_status":"reused_history_not_independent_validation", "production_changed":False,
        "source_http_requests":0,"artifact_policy":plan["artifact_policy"],"independent_evaluation":"pending"}
    rec.save_objects(artifact_path="analysis",summary=summary)
    rec.save_objects(artifact_path="execution",receipt={**state,"execution_finished_before_archive":True})
    artifact_root=Path(summary["recorder_local_dir"])/"artifacts"
    rec.log_metrics(registered_artifact_count=sum(path.is_file() for path in artifact_root.rglob("*")))
    rec.end_run(status=Recorder.STATUS_FI)
    atomic_json({"recorder_id":rec.id,"entries":[{"path":str(path.relative_to(card)),"sha256":sha(path),
        "bytes":path.stat().st_size} for path in sorted(artifact_root.rglob("*")) if path.is_file()]},card/"verification_manifest.json")
    old=json.loads((root/plan["training_source"]["card"]/"nas_binding.json").read_text())
    atomic_json({"schema_version":"recorder_evidence_binding_v1","authority":old["authority"],"ssh":old["ssh"],
        "source_id":"aux-projection-allocation-"+digest,"as_of":plan["as_of"],"source_plan":"plan.json"},card/"nas_binding.json")
    summary["nas_evidence"]=archive_nas_recorder(card/"nas_binding.json",workspace_root=root,card=card,
        recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    atomic_json(summary,card/"summary.json")
    _conclusion(card,summary,results["metrics"])
    state.update(completed=True,phase="completed",new_fits=3,new_native_accounts=3,reused_native_accounts=15)
    atomic_json(state,state_path)
    return summary


def _conclusion(card,summary,metrics):
    lines=["# 原辅助梯度的非对称投影：固定实验", "",
        "仅完成C的3个seed、各300轮拟合；B直接继承已精确复现的共享头模型/预测/账户，没有重新拟合或补跑。原3014主样本、20741辅助样本、39+16输入、标签、损失权重和初始化均固定。只有辅助梯度与复合主任务梯度严格冲突时，投影共享154参数上的辅助分量；其余梯度按冻结合同保留。",
        "", "三个C模型全部保存并固定SHA后才读取评价集。完成C的0/2/5bp三个独立费用原生账户，并复用A/B/SOXX/9097/27baa十五个原始3778点账户；不包含FINRA或尾盘ROD。A是冻结现役模型在同源研究时钟下的重放，非实际券商执行复现。",
        "", "|方案|滑点bp|累计净收益%|收盘最大回撤%|CVaR5%|", "|---|---:|---:|---:|---:|"]
    for row in metrics.itertuples():
        lines.append(f"|{row.family}|{row.slippage_bps}|{row.return_pct:.3f}|{row.close_max_drawdown_pct:.3f}|{row.cvar5_pct:.3f}|")
    lines.extend(["", "决定：**"+summary["decision"]+"**，等待独立结果评审，整体稳定超额目标尚未达成。",
        "失败或未取得的门："+", ".join(key for key,value in summary["gates"].items() if not value)+"。",
        "", "完整/不完整年度、全部季度和126交易日滚动窗口见相应小表；全部3seed分类及训练梯度保留，不择窗、不按分类分数取消账户。原历史被多次研究复用，重叠窗口和多seed不构成独立市场验证。",
        "", "全局裁剪系数与AdamW实际更新仍可能变化；投影的局部方向性质不证明配置损失保护、有限步收益改善或控制器因果。采样风险不等于全分钟资格，历史来源和纸面公司行动也不等于历史PIT/真实结算。生产未改。",
        "", ("下一步仅完成独立评审及原定风险/前向资格。" if summary["coarse_economic_gates_pass"] else
        "关闭本固定投影版本，不调整角度、权重、裁剪、seed、epoch、窗口或控制器救援；下一机制由campaign另行冻结。"),
        "", "Recorder：`"+summary["recorder_id"]+"`；完整证据与NAS身份见summary.json、verification_manifest.json、nas_binding.json。"])
    (card/"conclusion.md").write_text("\n".join(lines)+"\n")
