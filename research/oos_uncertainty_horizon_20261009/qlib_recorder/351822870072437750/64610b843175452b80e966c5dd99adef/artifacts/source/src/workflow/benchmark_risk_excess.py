"""Continue stable-excess research with a causal benchmark-risk budget."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from src.workflow.stable_excess_integration import (
    CANONICAL_ROOT, SEEDS, annual_forecasts, benchmark_path, choose_family,
    freeze_forecast_features, json_write, load_frozen_bundle, replay_models, sha, training_spec,
)

EXPERIMENT="benchmark_risk_excess_20261002"
SOURCE="stable_excess_integration_20261002"
COVARIANCE_FIELDS=("cov_soxl_soxl","cov_soxl_soxx","cov_soxx_soxx")


def expanding_covariance(prices):
    """Every covariance row ends at the signal-day open, already observed."""
    opened=pd.DataFrame({s:prices[("open",s)].astype(float) for s in ("SOXL","SOXX")})
    if opened.index.has_duplicates or not opened.index.is_monotonic_increasing or not np.isfinite(opened.to_numpy()).all() or (opened<=0).any().any():
        raise ValueError("blocked: covariance source requires ordered complete observed opens")
    returns=opened.pct_change(fill_method=None)
    first=returns.SOXL.expanding(min_periods=2).var(ddof=1)
    second=returns.SOXX.expanding(min_periods=2).var(ddof=1)
    cross=returns.SOXL.expanding(min_periods=2).cov(returns.SOXX,ddof=1)
    limit=np.sqrt(first*second)
    violation=cross.abs()-limit
    if (violation>1e-12*np.maximum(limit,1e-15)).any():
        raise ValueError("blocked: expanding sample covariance is materially non-PSD")
    cross=cross.clip(lower=-limit,upper=limit)
    result=pd.DataFrame(dict(zip(COVARIANCE_FIELDS,(first,cross,second))),index=prices.index)
    result.attrs["source"]={"estimator":"expanding sample covariance ddof1", "price_mark":"open",
        "last_observation":"signal-day open, before signal close and next-open decision",
        "window_search":False,"future_return_labels_used":False,
        "roundoff_psd_repairs":int((violation>0).sum()),
        "source_price_sha256":hashlib.sha256(pd.util.hash_pandas_object(opened,index=True).values.tobytes()).hexdigest()}
    return result


def attach_covariance(frame,covariance):
    result=frame.copy()
    if covariance.index.has_duplicates or set(COVARIANCE_FIELDS)-set(covariance.columns):
        raise ValueError("blocked: primary decision lacks observed covariance")
    values=covariance.loc[:,list(COVARIANCE_FIELDS)].reindex(frame.index)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("blocked: primary decision lacks observed covariance")
    result.loc[:,list(COVARIANCE_FIELDS)]=values.to_numpy()
    return result


def prepare_fold(frame,auxiliary_dataset,*,year,variant,as_of):
    from src.data.integrated_allocation import (PRIMARY_FEATURES,PRIMARY_LABELS,PRIMARY_METADATA,RETURN_LABELS,
        COMMON_FEATURES,AUXILIARY_TARGETS,build_native_dataset,prepare_mature_rows)
    fields=(*PRIMARY_FEATURES,*COVARIANCE_FIELDS)
    dataset=build_native_dataset(frame,feature_columns=fields,label_columns=PRIMARY_LABELS,metadata_columns=PRIMARY_METADATA,as_of=as_of)
    cutoff=pd.Timestamp(f"{year}-01-01",tz="America/New_York")
    train,audit=prepare_mature_rows(dataset,decision_at=cutoff,label_columns=RETURN_LABELS,availability_column="return_available_at")
    if len(train)<252 or train.index.get_level_values("datetime").max().year>=year:
        raise ValueError("blocked: invalid annual risk-budget training maturity")
    auxiliary=None;aux_audit={"rows":0,"available":False,"reason":"not_requested"}
    if variant["pit_auxiliary"]:
        aux,aux_audit=prepare_mature_rows(auxiliary_dataset,decision_at=cutoff,label_columns=AUXILIARY_TARGETS,availability_column="auxiliary_available_at")
        auxiliary={"features":aux.loc[:,list(COMMON_FEATURES)].to_numpy(dtype=float),"targets":aux.loc[:,list(AUXILIARY_TARGETS)].to_numpy(dtype=float),
            "sample_weight":aux.sample_weight.to_numpy(dtype=float),"common_feature_indices":tuple(range(16))}
        aux_audit["available"]=not aux.empty
    return {"features":train.loc[:,list(fields)].to_numpy(dtype=float),"returns":train.loc[:,list(RETURN_LABELS)].to_numpy(dtype=float),
            "auxiliary":auxiliary,"audit":{"variant":variant["name"],"model_year":year,"primary":audit,"auxiliary":aux_audit}}


def _fit_job(payload):
    from src.model.benchmark_risk_allocation import fit_policy
    variant,seed,folds,spec,cache_root=payload
    models={};diagnostics=[]
    for year,fold in folds.items():
        before=time.monotonic()
        digest=hashlib.sha256(fold["features"].tobytes()+fold["returns"].tobytes()+json.dumps(fold["audit"],sort_keys=True).encode())
        if fold["auxiliary"] is not None:
            for field in ("features","targets","sample_weight"):
                value=np.asarray(fold["auxiliary"][field],dtype=np.float64)
                digest.update(str(value.shape).encode());digest.update(value.tobytes())
        fingerprint=digest.hexdigest()
        path=Path(cache_root)/f"{variant['name']}_{seed}_{year}_{fingerprint[:12]}.pkl"
        if path.exists():
            with path.open("rb") as file:model,metadata=pickle.load(file)
            if metadata["spec"]!=asdict(spec) or metadata["seed"]!=seed or metadata["market_conditioning"]!=variant["market_conditioning"]:
                raise ValueError("blocked: cached risk-budget training identity differs")
            if metadata.get("fold_fingerprint")!=fingerprint:
                raise ValueError("blocked: cached risk-budget fold values differ")
        else:
            model,metadata=fit_policy(fold["features"],fold["returns"],spec=spec,seed=seed,auxiliary=fold["auxiliary"],
                common_feature_indices=tuple(range(16)),market_conditioning=variant["market_conditioning"])
            metadata["fold_fingerprint"]=fingerprint
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix(".tmp")
            with temporary.open("wb") as file:pickle.dump((model,metadata),file,protocol=4)
            temporary.replace(path)
        model.fit_audit=fold["audit"];models[year]=model
        diagnostics.append({"variant":variant["name"],"seed":seed,"model_year":year,"seconds":time.monotonic()-before,
            "training_rows":metadata["training_rows"],"auxiliary_rows":metadata["auxiliary_rows"],"training_loss":metadata["history"][-1]["loss"],
            "label_available_at_max":fold["audit"]["primary"]["label_available_at_max"]})
        print(f"Risk-budget fit {variant['name']} seed={seed} year={year} seconds={time.monotonic()-before:.1f}",flush=True)
    return variant["name"],seed,models,pd.DataFrame(diagnostics)


def covariance_audit(replay,frame):
    dates=["signal_date","trade_date","exit_date"]
    if len(replay)!=len(frame) or not replay[dates].reset_index(drop=True).equals(frame[dates].reset_index(drop=True)):
        raise ValueError("blocked: risk audit dates are not aligned")
    weights=replay[["soxl_weight","soxx_weight","cash_weight"]].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights<0).any() or not np.allclose(weights.sum(1),1,atol=1e-10,rtol=0):
        raise ValueError("blocked: invalid risk-audit allocation weights")
    w=replay[["soxl_weight","soxx_weight"]].to_numpy(dtype=float)
    covariance=frame.loc[:,list(COVARIANCE_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(covariance).all() or (covariance[:,[0,2]]<0).any() or (covariance[:,1]**2>covariance[:,0]*covariance[:,2]+1e-12*np.maximum(covariance[:,0]*covariance[:,2],1e-30)).any():
        raise ValueError("blocked: invalid or non-PSD covariance in risk audit")
    variance=w[:,0]**2*covariance[:,0]+2*w[:,0]*w[:,1]*covariance[:,1]+w[:,1]**2*covariance[:,2]
    reference=covariance[:,2]
    if not np.isfinite(variance).all() or (variance>reference+1e-12*np.maximum(reference,1e-15)).any():
        raise ValueError("blocked: predicted portfolio variance exceeds the declared benchmark budget")
    ratio=np.divide(variance,reference,out=np.zeros_like(variance),where=reference>0)
    return {"status":"passed","maximum_predicted_variance_ratio":float(ratio.max()),"mean_predicted_variance_ratio":float(ratio.mean()),
        "mean_nominal_beta":float((3*w[:,0]+w[:,1]).mean()),"mean_soxl_pct":float(100*w[:,0].mean()),
        "mean_soxx_pct":float(100*w[:,1].mean()),"mean_cash_pct":float(100*replay.cash_weight.mean()),
        "interpretation":"causal estimated risk budget, not a realized tail-risk guarantee"}


def run_experiment(*,root,workers=3):
    import qlib
    from qlib.workflow import R
    from src.data.integrated_allocation import PRIMARY_FEATURES
    from src.analysis.integrated_allocation import StabilityGateSpec,stable_excess_gates,paired_metrics,annual_paired_metrics,native_allocation_reconciliation
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts
    root=Path(root).resolve();card=root/"research"/EXPERIMENT
    plan=json.loads((card/"selection_plan.json").read_text());plan_hash=sha(card/"selection_plan.json")
    source_plan=json.loads((root/"research"/SOURCE/"selection_plan.json").read_text())
    source_summary=json.loads((root/"research"/SOURCE/"summary.json").read_text())
    if source_summary["recorder_id"]!=plan["source_recorder_id"]:
        raise ValueError("blocked: continued research source identity changed")
    bundle=load_frozen_bundle(root,source_plan)
    raw=bundle["primary_frame"];forecasts,forecast_audit,forecast_models=annual_forecasts(raw,as_of=plan["source_as_of"])
    covariance=expanding_covariance(bundle["execution_prices"])
    base=attach_covariance(raw,covariance);augmented=attach_covariance(forecasts,covariance)
    fields=(*PRIMARY_FEATURES,*COVARIANCE_FIELDS)
    valid=raw[["soxl_return","soxx_return"]].notna().all(axis=1)
    dev_mask=valid & raw.index.year.isin(plan["training"]["development_signal_years"])
    outer_mask=valid & raw.index.year.isin(plan["training"]["diagnostic_signal_years"])
    development=base.loc[dev_mask];outer=base.loc[outer_mask]
    source_files=("src/model/benchmark_risk_allocation.py","src/workflow/benchmark_risk_excess.py","src/model/integrated_allocation.py",
        "src/data/integrated_allocation.py","src/analysis/integrated_allocation.py","src/workflow/stable_excess_integration.py")
    source_hashes={path:sha(root/path) for path in source_files}
    identity=hashlib.sha256(json.dumps({"plan":plan_hash,"source":source_hashes,"data":bundle["audit"]},sort_keys=True).encode()).hexdigest()
    cache=Path('/tmp')/f"soxl-benchmark-risk-{identity[:20]}"
    jobs=[];fold_audits=[]
    for variant in plan["variants"]:
        frame=augmented if variant["forecast_features"] else base
        folds={year:prepare_fold(frame,bundle["auxiliary_dataset"],year=year,variant=variant,as_of=plan["source_as_of"])
               for year in plan["training"]["development_signal_years"]}
        fold_audits.extend(f["audit"] for f in folds.values())
        for seed in SEEDS:jobs.append((variant,seed,folds,training_spec(plan,variant),str(cache)))
    models={v["name"]:{} for v in plan["variants"]};training_rows=[]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(_fit_job,job) for job in jobs]
        for future in as_completed(futures):
            name,seed,fitted,diagnostics=future.result();models[name][seed]=fitted;training_rows.append(diagnostics)
    benchmarks={cost:benchmark_path(development,cost_bps=cost) for cost in plan["evaluation"]["cost_bps"]}
    replays=[];seed_rows=[];risks=[];evidence={};paths_by_variant={}
    gate_spec=replace(StabilityGateSpec(),family_size=plan["evaluation"]["family_size_adjustment"])
    for variant in plan["variants"]:
        name=variant["name"];market=(augmented if variant["forecast_features"] else base).loc[dev_mask];paths={}
        for cost,benchmark in benchmarks.items():
            path=replay_models(market,models[name],cost_bps=cost,feature_columns=fields)
            risks.append({"variant":name,"seed":"ensemble","segment":"development","cost_bps":cost,**covariance_audit(path,market)})
            paths[cost]=path;replays.append(path.assign(variant=name,seed="ensemble",segment="development"))
            for seed in SEEDS:
                path=replay_models(market,models[name],cost_bps=cost,feature_columns=fields,seeds=(seed,))
                seed_rows.append({"variant":name,"seed":seed,"cost_bps":cost,**paired_metrics(path,benchmark)})
                risks.append({"variant":name,"seed":str(seed),"segment":"development","cost_bps":cost,**covariance_audit(path,market)})
                replays.append(path.assign(variant=name,seed=str(seed),segment="development"))
        evidence[name]=stable_excess_gates(paths,benchmarks,spec=gate_spec,seed_table=pd.DataFrame(seed_rows).loc[lambda x:x.variant.eq(name)])
        paths_by_variant[name]=paths
    selected,ranking=choose_family(evidence)
    json_write(card/"selection_seal.json",{"plan_sha256":plan_hash,"selected":selected,"selection_uses_diagnostic_signal_years":False,
        "development_last_exit":str(development.exit_date.max().date()),"campaign_status":"active_research"})
    print(f"Phase2 development sealed: {selected}",flush=True)
    # Quantify information gain against the registered no-signal control.
    control_rows=[]
    for variant in plan["variants"]:
        for cost in benchmarks:
            control_rows.append({"variant":variant["name"],"cost_bps":cost,"reference":"constant_budget",
                **paired_metrics(paths_by_variant[variant["name"]][cost],paths_by_variant["constant_budget"][cost])})
    chosen=next(v for v in plan["variants"] if v["name"]==selected["variant"])
    chosen_frame=augmented if chosen["forecast_features"] else base
    folds={year:prepare_fold(chosen_frame,bundle["auxiliary_dataset"],year=year,variant=chosen,as_of=plan["source_as_of"])
           for year in plan["training"]["diagnostic_signal_years"]}
    fold_audits.extend(f["audit"] for f in folds.values());outer_models={}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(_fit_job,(chosen,seed,folds,training_spec(plan,chosen),str(cache))) for seed in SEEDS]
        for future in as_completed(futures):
            _,seed,fitted,diagnostics=future.result();outer_models[seed]=fitted;training_rows.append(diagnostics)
    comparisons=[result["metrics"].assign(variant=name,segment="development") for name,result in evidence.items()]
    annuals=[result["annual"].assign(variant=name,segment="development") for name,result in evidence.items()]
    for cost in benchmarks:
        reference=benchmark_path(outer,cost_bps=cost)
        for mode,fitted in (("annual_refit",outer_models),("frozen2022",{seed:{2023:m[2023]} for seed,m in outer_models.items()})):
            market=chosen_frame.loc[outer_mask]
            if mode=="frozen2022" and chosen["forecast_features"]:
                market=attach_covariance(freeze_forecast_features(raw.loc[outer_mask],forecast_models.get(2023)),covariance)
            path=replay_models(market,fitted,cost_bps=cost,feature_columns=fields);name=f"{chosen['name']}_{mode}"
            comparisons.append(pd.DataFrame([{"variant":name,"segment":"reused_diagnostic","cost_bps":cost,**paired_metrics(path,reference)}]))
            annuals.append(annual_paired_metrics(path,reference).assign(variant=name,segment="reused_diagnostic",cost_bps=cost))
            replays.append(path.assign(variant=name,seed="ensemble",segment="reused_diagnostic"))
            risks.append({"variant":name,"seed":"ensemble","segment":"reused_diagnostic","cost_bps":cost,**covariance_audit(path,market)})
        replays.append(reference.assign(variant="SOXX_buy_hold",seed="benchmark",segment="reused_diagnostic"))
        replays.append(benchmarks[cost].assign(variant="SOXX_buy_hold",seed="benchmark",segment="development"))
    native={cost:native_allocation_reconciliation(development.reset_index(drop=True),paths_by_variant[chosen["name"]][cost],
        provider_uri=root/"data/qlib"/SOURCE,cost_bps=cost) for cost in benchmarks}
    comparison=pd.concat(comparisons,ignore_index=True);annual=pd.concat(annuals,ignore_index=True)
    gates=pd.concat([r["gates"].assign(variant=name) for name,r in evidence.items()],ignore_index=True)
    bootstrap=pd.concat([r["bootstrap"].assign(variant=name) for name,r in evidence.items()],ignore_index=True)
    if sha(card/"selection_plan.json")!=plan_hash or any(sha(root/p)!=v for p,v in source_hashes.items()):
        raise ValueError("blocked: phase2 plan or implementation changed during execution")
    uri=CANONICAL_ROOT/"research"/EXPERIMENT/"qlib_recorder";uri.mkdir(parents=True,exist_ok=True)
    qlib.init(provider_uri=str(root/"data/qlib"/SOURCE),region="us",kernels=1);R.set_uri(str(uri))
    with R.start(experiment_name=EXPERIMENT):
        recorder=R.get_recorder();R.log_params(plan_sha256=plan_hash,source_recorder_id=plan["source_recorder_id"],campaign_status="active_research")
        R.log_metrics(historical_qualification_pass=float(selected["historical_qualification_pass"]),registered_variants=4)
        R.save_objects(artifact_path="inputs",**{"primary_frame.pkl":base,"forecast_frame.pkl":augmented,"covariance.pkl":covariance,
            "data_audit.pkl":bundle["audit"],"fold_audits.pkl":fold_audits})
        R.save_objects(artifact_path="models",**{"development.pkl":models,"selected_annual_refits.pkl":outer_models,"forecast_models.pkl":forecast_models})
        R.save_objects(artifact_path="analysis",**{"portfolio_replays.pkl":pd.concat(replays,ignore_index=True),"comparison.pkl":comparison,
            "annual.pkl":annual,"gates.pkl":gates,"bootstrap.pkl":bootstrap,"seed_metrics.pkl":pd.DataFrame(seed_rows),
            "risk_budget_audit.pkl":pd.DataFrame(risks),"constant_control.pkl":pd.DataFrame(control_rows)})
        R.save_objects(artifact_path="portfolio_analysis",**{"native_reconciliation.pkl":native})
        recorder.log_artifact(str(card/"selection_plan.json"),artifact_path="inputs");recorder.log_artifact(str(card/"selection_seal.json"),artifact_path="inputs")
        recdir=Path(recorder.get_local_dir());pd.testing.assert_frame_equal(comparison,recorder.load_object("analysis/comparison.pkl"))
    _remove_qlib_code_snapshot_artifacts(recdir)
    for filename,table in {"model_selection.csv":ranking,"backtest_benchmark_comparison.csv":comparison,"annual_metrics.csv":annual,
        "development_gates.csv":gates,"bootstrap_analysis.csv":bootstrap,"seed_metrics.csv":pd.DataFrame(seed_rows),
        "risk_budget_audit.csv":pd.DataFrame(risks),"constant_control_comparison.csv":pd.DataFrame(control_rows),
        "training_diagnostics.csv":pd.concat(training_rows,ignore_index=True),"forecast_training_audit.csv":forecast_audit}.items():
        table.to_csv(card/filename,index=False)
    summary={"experiment":EXPERIMENT,"campaign_status":"active_research","objective_achieved":False,
        "decision":"historical_candidate_requires_new_forward" if selected["historical_qualification_pass"] else "continue_research_no_historical_qualification",
        "selected":selected,"plan_sha256":plan_hash,"source_as_of":plan["source_as_of"],"source_recorder_id":plan["source_recorder_id"],
        "provider_uri":f"data/qlib/{SOURCE}","region":"us","market":"all","covariance_audit":covariance.attrs["source"],
        "source_sha256":source_hashes,"historical_qualification_pass":bool(selected["historical_qualification_pass"]),
        "new_forward_qualified":False,"selection_uses_diagnostic_signal_years":False,"development_last_exit":str(development.exit_date.max().date()),
        "recorder_id":recorder.id,"recorder_uri":str(uri),"recorder_local_dir":str(recdir),"recorder_relative_dir":str(recdir.relative_to(CANONICAL_ROOT/"research"/EXPERIMENT)),
        "native_qlib_audit":{str(cost):v["audit"] for cost,v in native.items()},"production_change":False,"real_broker_orders":False,
        "artifact_policy":{"policy":"slim_card_recorder_first","workflow_config":"workflow.yaml","recorder_only_artifacts":["inputs/*.pkl","models/*.pkl","analysis/portfolio_replays.pkl","portfolio_analysis/*.pkl"]},
        "limitations":["Risk budget is an estimated variance ceiling; it does not guarantee realized drawdown/CVaR.","All historical windows have been reviewed; adaptive research and seed repetition do not create new independent evidence.","Covariance uses expanding prior observed opens, not futurelabels; its regime lag remains a model assumption.","A failed batch leaves the overall stable-excess objective active."]}
    json_write(card/"summary.json",summary)
    import yaml
    (card/"workflow.yaml").write_text(yaml.safe_dump({"experiment":EXPERIMENT,"provider_uri":summary["provider_uri"],"region":"us",
        "recorder_uri":str(uri),"segments":{"development_signal_years":plan["training"]["development_signal_years"],"reused_diagnostic_signal_years":plan["training"]["diagnostic_signal_years"]},
        "model_selection":"selection_plan.json","artifact_policy":summary["artifact_policy"]},sort_keys=False))
    lines=["# 基准风险预算实验结论","",f"这是稳定超额目标的继续研究；本轮完成4个版本、3个种子，历史资格通过={selected['historical_qualification_pass']}，总体目标尚未完成。",
        "","使用截至信号日已观察开盘收益的扩展协方差；模型学习风险预算使用比例与SOXL/SOXX组成，三维协方差只进入可行权重映射，不进入神经输入。训练与推理采用同一映射和净费结算。","","开发期2016–2022信号年、10bp："]
    for row in comparison[comparison.segment.eq("development")&comparison.cost_bps.eq(10)].itertuples():
        lines.append(f"- {row.variant}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD{row.max_drawdown_pct:.2f}%，CVaR5{row.cvar5_pct:.2f}%。")
    lines += ["","无市场信号对照、年度表现、两档费用、全部种子及失败门均已保留。开发期末次结算2023-01-04开盘；之后outer信号未参与排名。实际波动、尾部和稳定性门不因预测风险满足预算而豁免。","",
        f"本轮决定：{summary['decision']}。没有切换原SIM或提交委托；下一步由本轮结果及独立预测信息检验决定，不把单轮结束视为总体目标完成。",f"QLib recorder：{recorder.id}。"]
    (card/"conclusion.md").write_text("\n".join(lines)+"\n")
    return summary
