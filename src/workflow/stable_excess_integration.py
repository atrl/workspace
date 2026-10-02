"""Bounded, recorder-first integration of prior continuous-allocation research."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

EXPERIMENT = "stable_excess_integration_20261002"
CANONICAL_ROOT = Path("/Users/rziiiii/dev/workspace")
SEEDS = (17, 41, 71)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n")


def annual_forecasts(frame, *, as_of):
    """Causal OOF forecasts; a future training outcome never becomes a feature."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.data.integrated_allocation import (build_native_dataset, prepare_mature_rows, PRIMARY_FEATURES,
        PRIMARY_LABELS, PRIMARY_METADATA, COMMON_PRIMARY_FEATURES, FORECAST_LABELS)

    dataset = build_native_dataset(frame, feature_columns=PRIMARY_FEATURES, label_columns=PRIMARY_LABELS,
        metadata_columns=PRIMARY_METADATA, as_of=as_of)
    augmented = frame.copy()
    diagnostics, models = [], {}
    for year in sorted(frame.index.year.unique()):
        cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        training, audit = prepare_mature_rows(dataset, decision_at=cutoff, label_columns=FORECAST_LABELS,
            availability_column="forecast_available_at")
        mask = frame.index.year == year
        model = None
        if len(training) >= 252:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
            model.fit(training.loc[:, list(COMMON_PRIMARY_FEATURES)], training.loc[:, list(FORECAST_LABELS)])
            predictions = model.predict(frame.loc[mask, list(COMMON_PRIMARY_FEATURES)])
            if not np.isfinite(predictions).all():
                raise ValueError("blocked: nonfinite OOF cash/volatility forecast")
            augmented.loc[mask, ["cash_advantage_prediction", "future_vol_prediction"]] = predictions
            augmented.loc[mask, "forecast_available"] = 1.
            models[int(year)] = model
        diagnostics.append({"model_year": int(year), "available": model is not None, "ridge_alpha": 10.,
                            "minimum_training_rows": 252, **audit})
    return augmented, pd.DataFrame(diagnostics), models


def prepare_fold(frame, auxiliary_dataset, *, year, variant, as_of):
    from src.data.integrated_allocation import (build_native_dataset, prepare_mature_rows, PRIMARY_FEATURES,
        PRIMARY_LABELS, PRIMARY_METADATA, RETURN_LABELS, COMMON_FEATURES, AUXILIARY_TARGETS)

    dataset = build_native_dataset(frame, feature_columns=PRIMARY_FEATURES, label_columns=PRIMARY_LABELS,
        metadata_columns=PRIMARY_METADATA, as_of=as_of)
    cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    training, audit = prepare_mature_rows(dataset, decision_at=cutoff, label_columns=RETURN_LABELS,
        availability_column="return_available_at")
    if len(training) < 252 or training.index.get_level_values("datetime").max().year >= year:
        raise ValueError("blocked: primary annual allocation training coverage or maturity is invalid")
    auxiliary = None
    auxiliary_audit = {"rows": 0, "available": False, "reason": "not_requested"}
    if variant["pit_auxiliary"]:
        aux, auxiliary_audit = prepare_mature_rows(auxiliary_dataset, decision_at=cutoff,
            label_columns=AUXILIARY_TARGETS, availability_column="auxiliary_available_at")
        auxiliary_audit["available"] = not aux.empty
        auxiliary = {"features": aux.loc[:, list(COMMON_FEATURES)].to_numpy(dtype=float),
                     "targets": aux.loc[:, list(AUXILIARY_TARGETS)].to_numpy(dtype=float),
                     "common_feature_indices": tuple(range(len(COMMON_FEATURES))),
                     "sample_weight": aux.sample_weight.to_numpy(dtype=float)}
    return {"features": training.loc[:, list(PRIMARY_FEATURES)].to_numpy(dtype=float),
            "returns": training.loc[:, list(RETURN_LABELS)].to_numpy(dtype=float), "auxiliary": auxiliary,
            "audit": {"model_year": year, "variant": variant["name"], "primary": audit, "auxiliary": auxiliary_audit}}


def training_spec(plan, variant):
    from src.model.integrated_allocation import AllocationSpec
    p = plan["training"]
    return AllocationSpec(hidden_units=p["hidden_units"], epochs=p["epochs"], learning_rate=p["learning_rate"],
        weight_decay=p["weight_decay"], block_length=p["block_length"], warmup_sessions=p["warmup_sessions"],
        batch_blocks=p["batch_blocks"], downside_coefficient=p["objective_downside_coefficient"],
        auxiliary_coefficient=p["pit_auxiliary_coefficient"] if variant["pit_auxiliary"] else 0.,
        training_cost_bps=p["training_cost_bps"], gradient_clip=p["gradient_clip"])


def load_frozen_bundle(root, plan):
    """After first execution reuse immutable Recorder rows, not a later live store."""
    import qlib
    from qlib.workflow import R
    from src.data.integrated_allocation import (load_integrated_data,IntegratedAllocationDataSpec,build_native_dataset,
        PRIMARY_FEATURES,PRIMARY_LABELS,PRIMARY_METADATA,COMMON_FEATURES,AUXILIARY_TARGETS,AUXILIARY_METADATA,frame_sha256)

    card=root/"research"/EXPERIMENT
    summary_path=card/"summary.json"
    if not summary_path.exists():
        return load_integrated_data(IntegratedAllocationDataSpec(root=root,as_of=plan["data_as_of"]))
    summary=json.loads(summary_path.read_text())
    if summary["plan_sha256"]!=sha(card/"selection_plan.json"):
        raise ValueError("blocked: retained integration plan differs")
    qlib.init(provider_uri=str(root/"data/qlib"/EXPERIMENT),region="us",kernels=1)
    with R.uri_context(summary["recorder_uri"]):
        recorder=R.get_recorder(experiment_name=EXPERIMENT,recorder_id=summary["recorder_id"])
        raw=recorder.load_object("inputs/primary_frame.pkl")
        aux=recorder.load_object("inputs/auxiliary_frame.pkl")
        prices=recorder.load_object("inputs/execution_prices.pkl")
        audit=recorder.load_object("inputs/source_audit.pkl")
    if audit != summary["data_audit"]:
        raise ValueError("blocked: Recorder source audit differs from the sealed card summary")
    for data,key in ((raw,"primary_frame_sha256"),(aux,"auxiliary_frame_sha256"),(prices,"execution_prices_sha256")):
        if frame_sha256(data)!=audit[key]:
            raise ValueError("blocked: retained integration frame hash differs")
    return {"primary_frame":raw,"auxiliary_frame":aux,"execution_prices":prices,"audit":audit,
        "auxiliary_dataset":build_native_dataset(aux,feature_columns=COMMON_FEATURES,label_columns=AUXILIARY_TARGETS,
            metadata_columns=AUXILIARY_METADATA,as_of=plan["data_as_of"])}


def pin_execution_provider(root, bundle, plan):
    from src.data.qlib_native import build_qlib_store_from_frame, QlibProviderConfig, init_qlib_provider
    from qlib.data import D
    prices=bundle["execution_prices"]
    provider=root/"data/qlib"/EXPERIMENT
    manifest=provider/"qlib_store_manifest.json"
    if manifest.exists():
        stored=json.loads(manifest.read_text())
        if stored.get("source")!=f"Qlib frozen execution frame {bundle['audit']['execution_prices_sha256']}" or stored.get("as_of")!=plan["data_as_of"]:
            raise ValueError("blocked: immutable execution snapshot identity differs")
        init_qlib_provider(QlibProviderConfig(provider_uri=provider))
        actual=D.features(["SOXL","SOXX"],["$open","$close"],start_time=prices.index.min(),end_time=prices.index.max(),freq="day")
        actual.columns=["open","close"]
        actual=actual.unstack("instrument").sort_index()
        if not actual.index.equals(prices.index) or not actual.columns.equals(prices.columns) or not np.array_equal(actual.to_numpy(),prices.to_numpy()):
            raise ValueError("blocked: frozen execution provider values differ from recorded prices")
        return provider,stored
    rows=[]
    for symbol in ("SOXL","SOXX"):
        rows.append(pd.DataFrame({"datetime":prices.index,"instrument":symbol,
            "open":prices[("open",symbol)].to_numpy(),"close":prices[("close",symbol)].to_numpy()}))
    stored=build_qlib_store_from_frame(pd.concat(rows,ignore_index=True),provider_uri=provider,region="us",market="all",
        as_of=plan["data_as_of"],source=f"Qlib frozen execution frame {bundle['audit']['execution_prices_sha256']}")
    return provider,stored


def _fit_seed_job(payload):
    import torch
    from src.model.integrated_allocation import fit_policy

    torch.set_num_threads(1)
    variant, seed, folds, spec, cache_root = payload
    models, diagnostics = {}, []
    for year, fold in folds.items():
        cache = Path(cache_root) / f"{variant}_{seed}_{year}.pkl"
        before = time.monotonic()
        if cache.exists():
            with cache.open("rb") as handle:
                model, metadata = pickle.load(handle)
            if metadata["spec"] != asdict(spec) or metadata["seed"] != seed:
                raise ValueError("blocked: cache model differs from registered training identity")
            for field,key in (("features","training_feature_sha256"),("returns","training_return_sha256")):
                if metadata[key]!=hashlib.sha256(np.asarray(fold[field],dtype=np.float64).tobytes()).hexdigest():
                    raise ValueError("blocked: cached model trained on different fold values")
            if metadata.get("fold_audit_sha256")!=hashlib.sha256(json.dumps(fold["audit"],sort_keys=True).encode()).hexdigest():
                raise ValueError("blocked: cached model trained under different label maturity")
            if fold["auxiliary"] is not None and len(fold["auxiliary"]["features"]):
                for field,key in (("features","auxiliary_feature_sha256"),("targets","auxiliary_target_sha256"),("sample_weight","auxiliary_weight_sha256")):
                    expected=hashlib.sha256(np.asarray(fold["auxiliary"][field],dtype=np.float64).tobytes()).hexdigest()
                    if metadata.get(key)!=expected:
                        raise ValueError("blocked: cached model trained on different PIT auxiliary values")
        else:
            model, metadata = fit_policy(fold["features"], fold["returns"], spec=spec, seed=seed,
                auxiliary=fold["auxiliary"], common_feature_indices=tuple(range(16)))
            metadata["fold_audit_sha256"]=hashlib.sha256(json.dumps(fold["audit"],sort_keys=True).encode()).hexdigest()
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump((model, metadata), handle, protocol=4)
            temporary.replace(cache)
        model.fit_audit = fold["audit"]
        models[year] = model
        diagnostics.append({"variant": variant, "seed": seed, "model_year": year,
                            "elapsed_seconds": time.monotonic()-before,
                            "training_rows": metadata["training_rows"], "auxiliary_rows": metadata["auxiliary_rows"],
                            "final_training_loss": metadata["history"][-1]["loss"],
                            "label_available_at_max": fold["audit"]["primary"]["label_available_at_max"],
                            "training_signal_end": fold["audit"]["primary"]["last_signal_date"]})
        print(f"Fit complete {variant} seed={seed} year={year} rows={metadata['training_rows']} seconds={time.monotonic()-before:.1f}", flush=True)
    return variant, seed, models, pd.DataFrame(diagnostics)


def replay_models(frame, models, *, cost_bps, feature_columns, seeds=SEEDS):
    """A model/seed ensemble shares actual holdings, never averages path returns."""
    import torch
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    current = np.array([0., 0., 1.])
    equity = 1.
    rows = []
    for date, market in frame.iterrows():
        targets = []
        for seed in seeds:
            years = [year for year in models[seed] if year <= date.year]
            if not years:
                raise ValueError("blocked: missing causal policy for a requested decision year")
            model = models[seed][max(years)]
            parameter = next(model.parameters())
            with torch.no_grad():
                value = model(torch.as_tensor(market.loc[list(feature_columns)].to_numpy(dtype=float), dtype=parameter.dtype)[None,:],
                    torch.as_tensor(current, dtype=parameter.dtype)[None,:]).detach().cpu().numpy()[0].astype(float)
            if not np.isfinite(value).all() or value.min() < 0:
                raise ValueError("blocked: invalid continuous model target")
            targets.append(value/value.sum())
        target = np.mean(targets, axis=0)
        target /= target.sum()
        execution = fee_affordable_target(target, pretrade_weights=current, cost_bps=cost_bps)
        settled = settle_target_session(current, open_target_weights=execution,
            asset_returns=[market.soxl_return, market.soxx_return, 0.], cost_bps=cost_bps)
        equity *= settled["net_equity"]
        rows.append({"signal_date": market.signal_date, "trade_date": market.trade_date, "exit_date": market.exit_date,
            "strategy_return": settled["net_return"], "turnover": settled["turnover"], "fees": settled["fees"],
            "cost_bps": cost_bps, "equity": equity, "execute_open_target": True,
            "execute_open_trade": settled["turnover"] > 1e-12,
            **{f"{asset}_weight": float(target[i]) for i,asset in enumerate(("soxl","soxx","cash"))},
            **{f"pretrade_{asset}_weight": float(current[i]) for i,asset in enumerate(("soxl","soxx","cash"))},
            **{f"ending_{asset}_weight": float(settled["ending_weights"][i]) for i,asset in enumerate(("soxl","soxx","cash"))}})
        current = settled["ending_weights"]
    return pd.DataFrame(rows)


def benchmark_path(frame, *, cost_bps, stock_weight=1.):
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    current = np.array([0.,0.,1.]); equity=1.; rows=[]
    target = np.array([0., stock_weight, 1-stock_weight])
    for number,(_,market) in enumerate(frame.iterrows()):
        execution = fee_affordable_target(target, pretrade_weights=current, cost_bps=cost_bps) if number == 0 else None
        settled=settle_target_session(current, open_target_weights=execution,
            asset_returns=[market.soxl_return,market.soxx_return,0.],cost_bps=cost_bps)
        equity*=settled["net_equity"]
        rows.append({"signal_date":market.signal_date,"trade_date":market.trade_date,"exit_date":market.exit_date,
            "strategy_return":settled["net_return"],"turnover":settled["turnover"],"fees":settled["fees"],
            "cost_bps":cost_bps,"equity":equity,"execute_open_target":number==0,
            **{f"{asset}_weight":float(settled["open_net_weights"][i]) for i,asset in enumerate(("soxl","soxx","cash"))}})
        current=settled["ending_weights"]
    return pd.DataFrame(rows)


def choose_family(evidence):
    rows=[]
    for name,result in evidence.items():
        gates=result["gates"]
        columns=[column for column in gates if gates[column].dtype==bool and column not in {"all_pass","seed_evidence_complete"}]
        rows.append({"variant":name,"historical_qualification_pass":result["historical_qualification_pass"],
                     "passed_gates":int(gates[columns].to_numpy().sum()),
                     "worst_annualized_lower_bound":float(result["bootstrap"].annualized_lower_bound.min()),
                     "worst_relative_wealth":float(result["metrics"].relative_wealth.min()),
                     "positive_years_10bps":int(gates.loc[gates.cost_bps.eq(10),"positive_years"].iloc[0])})
    ranked=pd.DataFrame(rows).sort_values(["historical_qualification_pass","passed_gates","worst_annualized_lower_bound","variant"],ascending=[False,False,False,True])
    return ranked.iloc[0].to_dict(),ranked


def freeze_forecast_features(raw, forecast_model):
    from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES
    result=raw.copy()
    if forecast_model is None:
        raise ValueError("blocked: frozen allocator comparison lacks its matching frozen forecast model")
    prediction=forecast_model.predict(result.loc[:,list(COMMON_PRIMARY_FEATURES)])
    if not np.isfinite(prediction).all():
        raise ValueError("blocked: nonfinite frozen-system forecast")
    result.loc[:,["cash_advantage_prediction","future_vol_prediction"]]=prediction
    result.loc[:,"forecast_available"]=1.
    return result


def run_experiment(*, root, workers=3):
    import qlib
    from qlib.workflow import R
    from src.data.integrated_allocation import load_integrated_data,IntegratedAllocationDataSpec,PRIMARY_FEATURES
    from src.analysis.integrated_allocation import stable_excess_gates,paired_metrics,annual_paired_metrics,native_allocation_reconciliation
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts

    root=Path(root).resolve();card=root/"research"/EXPERIMENT
    plan=json.loads((card/"selection_plan.json").read_text());plan_hash=sha(card/"selection_plan.json")
    bundle=load_frozen_bundle(root,plan)
    execution_provider,execution_manifest=pin_execution_provider(root,bundle,plan)
    if list(PRIMARY_FEATURES)!=plan["features"]["primary_ordered"]:
        raise ValueError("blocked: actual features differ from preregistration")
    raw=bundle["primary_frame"]
    forecasts,forecast_audit,forecast_models=annual_forecasts(raw,as_of=plan["data_as_of"])
    valid=raw.loc[:,["soxl_return","soxx_return"]].notna().all(axis=1)
    dev_mask=valid & raw.index.year.isin(plan["training"]["development_years"])
    diag_mask=valid & raw.index.year.isin(plan["training"]["diagnostic_years"])
    development=raw.loc[dev_mask].copy();diagnostic=raw.loc[diag_mask].copy()
    source_paths=("src/model/integrated_allocation.py","src/data/integrated_allocation.py","src/analysis/integrated_allocation.py","src/workflow/stable_excess_integration.py")
    source_hashes={p:sha(root/p) for p in source_paths}
    identity=hashlib.sha256(json.dumps({"plan":plan_hash,"source_hashes":source_hashes,"inputs":bundle["audit"]},sort_keys=True,default=str).encode()).hexdigest()
    cache=Path("/tmp")/f"soxl-stable-excess-{identity[:20]}"
    print(f"Frozen data {bundle['audit']['primary_rows']} primary rows; cache identity {identity}",flush=True)
    payloads=[]; fold_audits=[]
    for variant in plan["variants"]:
        source=forecasts if variant["forecast_features"] else raw
        folds={year:prepare_fold(source,bundle["auxiliary_dataset"],year=year,variant=variant,as_of=plan["data_as_of"]) for year in plan["training"]["development_years"]}
        fold_audits.extend(fold["audit"] for fold in folds.values())
        for seed in SEEDS:
            payloads.append((variant["name"],seed,folds,training_spec(plan,variant),str(cache)))
    fitted={v["name"]:{} for v in plan["variants"]};training_diagnostics=[]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(_fit_seed_job,payload) for payload in payloads]
        for future in as_completed(futures):
            name,seed,models,diagnostics=future.result();fitted[name][seed]=models;training_diagnostics.append(diagnostics)
    benchmarks={cost:benchmark_path(development,cost_bps=cost) for cost in plan["evaluation"]["cost_bps"]}
    evidence={};replays=[];seed_rows=[]
    for variant in plan["variants"]:
        name=variant["name"];market=(forecasts if variant["forecast_features"] else raw).loc[dev_mask]
        paths={}
        for cost,benchmark in benchmarks.items():
            paths[cost]=replay_models(market,fitted[name],cost_bps=cost,feature_columns=PRIMARY_FEATURES)
            replays.append(paths[cost].assign(variant=name,seed="ensemble",segment="development",cost_bps=cost))
            for seed in SEEDS:
                replay=replay_models(market,fitted[name],cost_bps=cost,feature_columns=PRIMARY_FEATURES,seeds=(seed,))
                metric=paired_metrics(replay,benchmark)
                seed_rows.append({"variant":name,"seed":seed,"cost_bps":cost,**metric})
                replays.append(replay.assign(variant=name,seed=str(seed),segment="development"))
        evidence[name]=stable_excess_gates(paths,benchmarks,seed_table=pd.DataFrame(seed_rows).loc[lambda x:x.variant.eq(name)])
    selected,ranking=choose_family(evidence)
    seal={"plan_sha256":plan_hash,"selected":selected,"selection_uses_post2022_signals":False,
          "development_final_outcome_observed_at":str(development.return_available_at.max()),"cache_identity":identity}
    json_write(card/"selection_seal.json",seal)
    print(f"Development selection sealed: {selected}",flush=True)
    chosen=next(v for v in plan["variants"] if v["name"]==selected["variant"])
    chosen_frame=forecasts if chosen["forecast_features"] else raw
    outer_folds={year:prepare_fold(chosen_frame,bundle["auxiliary_dataset"],year=year,variant=chosen,as_of=plan["data_as_of"]) for year in plan["training"]["diagnostic_years"]}
    fold_audits.extend(fold["audit"] for fold in outer_folds.values())
    outer_models={}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(_fit_seed_job,(chosen["name"],seed,outer_folds,training_spec(plan,chosen),str(cache))) for seed in SEEDS]
        for future in as_completed(futures):
            _,seed,models,diagnostics=future.result();outer_models[seed]=models;training_diagnostics.append(diagnostics)
    outer_rows=[];outer_annual=[];static_rows=[]
    for cost in plan["evaluation"]["cost_bps"]:
        reference=benchmark_path(diagnostic,cost_bps=cost)
        for mode,models in (("annual_refit",outer_models),("frozen2022",{seed:{2023:models[2023]} for seed,models in outer_models.items()})):
            replay_frame=chosen_frame.loc[diag_mask]
            if mode=="frozen2022" and chosen["forecast_features"]:
                replay_frame=freeze_forecast_features(raw.loc[diag_mask],forecast_models.get(2023))
            replay=replay_models(replay_frame,models,cost_bps=cost,feature_columns=PRIMARY_FEATURES)
            name=f"{chosen['name']}_{mode}"
            replays.append(replay.assign(variant=name,seed="ensemble",segment="reused_diagnostic"))
            outer_rows.append({"variant":name,"segment":"reused_diagnostic","cost_bps":cost,**paired_metrics(replay,reference)})
            outer_annual.append(annual_paired_metrics(replay,reference).assign(variant=name,segment="reused_diagnostic",cost_bps=cost))
        for segment,market,benchmark in (("development",development,benchmarks[cost]),("reused_diagnostic",diagnostic,reference)):
            for weight in (.25,.5,.75,1.):
                static=benchmark_path(market,cost_bps=cost,stock_weight=weight)
                static_rows.append({"variant":f"SOXX_cash_initial_{weight:g}","segment":segment,"cost_bps":cost,**paired_metrics(static,benchmark)})
            replays.append(benchmark.assign(variant="SOXX_buy_hold",seed="benchmark",segment=segment))
    all_paths=pd.concat(replays,ignore_index=True)
    native={}
    for cost in plan["evaluation"]["cost_bps"]:
        path=all_paths.loc[all_paths.variant.eq(chosen["name"]) & all_paths.seed.eq("ensemble") & all_paths.segment.eq("development") & all_paths.cost_bps.eq(cost)].copy()
        native[cost]=native_allocation_reconciliation(development.reset_index(drop=True),path,provider_uri=execution_provider,cost_bps=cost)
    if sha(card/"selection_plan.json")!=plan_hash or any(sha(root/p)!=digest for p,digest in source_hashes.items()):
        raise ValueError("blocked: source or preregistration changed during the integrated experiment")
    comparison=pd.concat([result["metrics"].assign(variant=name,segment="development") for name,result in evidence.items()]+[pd.DataFrame(outer_rows),pd.DataFrame(static_rows)],ignore_index=True)
    annual=pd.concat([result["annual"].assign(variant=name,segment="development") for name,result in evidence.items()]+outer_annual,ignore_index=True)
    gates=pd.concat([result["gates"].assign(variant=name) for name,result in evidence.items()],ignore_index=True)
    bootstrap=pd.concat([result["bootstrap"].assign(variant=name) for name,result in evidence.items()],ignore_index=True)
    uri=CANONICAL_ROOT/"research"/EXPERIMENT/"qlib_recorder";uri.mkdir(parents=True,exist_ok=True)
    qlib.init(provider_uri=str(execution_provider),region="us",kernels=1);R.set_uri(str(uri))
    with R.start(experiment_name=EXPERIMENT):
        recorder=R.get_recorder()
        R.log_params(plan_sha256=plan_hash,source_as_of=plan["data_as_of"],benchmark="SOXX",selected_variant=chosen["name"],selection_uses_outer_test=False)
        R.log_metrics(historical_qualification_pass=float(selected["historical_qualification_pass"]),candidate_count=3,seeds_per_candidate=3)
        R.save_objects(artifact_path="inputs",**{"primary_frame.pkl":raw,"auxiliary_frame.pkl":bundle["auxiliary_frame"],"execution_prices.pkl":bundle["execution_prices"],"forecast_frame.pkl":forecasts,"source_audit.pkl":bundle["audit"],"fold_audits.pkl":fold_audits})
        R.save_objects(artifact_path="models",**{"development.pkl":fitted,"selected_annual_refits.pkl":outer_models,"forecast_models.pkl":forecast_models})
        R.save_objects(artifact_path="analysis",**{"portfolio_replays.pkl":all_paths,"comparison.pkl":comparison,"annual_metrics.pkl":annual,"gates.pkl":gates,"bootstrap.pkl":bootstrap,"seed_metrics.pkl":pd.DataFrame(seed_rows),"training_diagnostics.pkl":pd.concat(training_diagnostics,ignore_index=True)})
        R.save_objects(artifact_path="portfolio_analysis",**{"native_reconciliation.pkl":native})
        recorder.log_artifact(str(card/"selection_plan.json"),artifact_path="inputs");recorder.log_artifact(str(card/"selection_seal.json"),artifact_path="inputs")
        recdir=Path(recorder.get_local_dir());pd.testing.assert_frame_equal(comparison,recorder.load_object("analysis/comparison.pkl"))
    _remove_qlib_code_snapshot_artifacts(recdir)
    for filename,table in {"backtest_benchmark_comparison.csv":comparison,"annual_metrics.csv":annual,"development_gates.csv":gates,"model_selection.csv":ranking,"bootstrap_analysis.csv":bootstrap,"seed_metrics.csv":pd.DataFrame(seed_rows),"training_diagnostics.csv":pd.concat(training_diagnostics,ignore_index=True),"forecast_training_audit.csv":forecast_audit}.items():
        table.to_csv(card/filename,index=False)
    summary={"experiment":EXPERIMENT,"decision":"historical_candidate_pending_new_forward" if selected["historical_qualification_pass"] else "not_promoted_no_stable_excess_evidence",
        "selected":selected,"plan_sha256":plan_hash,"source_as_of":plan["data_as_of"],"provider_uri":f"data/qlib/{EXPERIMENT}","execution_snapshot_manifest":execution_manifest,"region":"us","market":"all",
        "data_audit":bundle["audit"],"recorder_id":recorder.id,"recorder_uri":str(uri),"recorder_local_dir":str(recdir),"recorder_relative_dir":str(recdir.relative_to(CANONICAL_ROOT/"research"/EXPERIMENT)),
        "source_sha256":source_hashes,"seeds":list(SEEDS),"model_selection_uses_outer_test":False,"research_adaptive_history":True,
        "development_period":{"first_signal":str(development.signal_date.min().date()),"last_signal":str(development.signal_date.max().date()),"last_exit":str(development.exit_date.max().date()),"annual_basis":"signal_year"},
        "diagnostic_frozen_system":"allocator and any cash/volatility forecast heads both fixed at their2023fit with pre2023mature training labels",
        "historical_qualification_pass":bool(selected["historical_qualification_pass"]),"new_forward_qualified":False,"production_change":False,
        "native_qlib_audit":{str(cost):value["audit"] for cost,value in native.items()},
        "artifact_policy":{"policy":"slim_card_recorder_first","workflow_config":"workflow.yaml","recorder_only_artifacts":["inputs/*.pkl","models/*.pkl","analysis/portfolio_replays.pkl","portfolio_analysis/native_reconciliation.pkl"]},
        "limitations":["All2016-2026 history has participated in earlier research; repeated historical qualification is not new forward evidence.","Block bootstrap does not include all previous model/feature search uncertainty; family adjustment covers only this registered family.","Training uses initially-cash chronological84-session blocks with21warmup; evaluation uses continuous own holdings.","PIT auxiliary starts2019; earlier folds intentionally receive no auxiliary data.","Historical open execution is not verified09:31 fills; no real orders or existingSIM replacement.","Cash returns assumed zero; no promise of future stable excess."],
        "qlib_docs_reference":["https://qlib.readthedocs.io/en/latest/component/data.html","https://qlib.readthedocs.io/en/latest/component/workflow.html","https://qlib.readthedocs.io/en/latest/component/strategy.html"]}
    json_write(card/"summary.json",summary);json_write(card/"data_audit.json",bundle["audit"])
    import yaml
    (card/"workflow.yaml").write_text(yaml.safe_dump({"experiment":EXPERIMENT,"provider_uri":summary["provider_uri"],"region":"us","recorder_uri":str(uri),"segments":{"development":["2016-01-01","2022-12-31"],"reused_diagnostic":["2023-01-01",plan["data_as_of"]]},"model_selection":"selection_plan.json","artifact_policy":summary["artifact_policy"]},sort_keys=False))
    lines=["# 稳定净超额整合实验结论","",f"完成3个同结构候选、每个3个种子的年度成熟标签训练；无旧止损计时器、现金下限、SOXL上限、EMA或HOLD开关。选中{chosen['name']}；历史稳定性门全部通过={selected['historical_qualification_pass']}。","","开发期2016–2022，各模型为同一真实持仓上平均三个种子目标的组合，以下为10bp："]
    for row in comparison.loc[comparison.segment.eq("development") & comparison.cost_bps.eq(10) & comparison.variant.isin(fitted)].itertuples():
        lines.append(f"- {row.variant}：净收益{row.return_pct:.2f}%，相对SOXX终值比{row.relative_wealth:.4f}，MaxDD{row.max_drawdown_pct:.2f}%，CVaR5{row.cvar5_pct:.2f}%。")
    lines += ["","年度相对财富见annual_metrics.csv；20/60交易日相关块、三候选调整后的区间见bootstrap_analysis.csv；所有种子和失败门均保留。25bp重新计算各自费用、持仓与后续决策。","","开发期按2016–2022信号年划分，末次收益结算到2023-01-04开盘；此后outer持有区间未参与本次排名。2023–2026信号仅作已查看历史的复用诊断，不能用于追加候选或选择种子。冻结2022对照同时冻结配置器和现金/波动预测头。原生Qlib回测按两档成本核对完整开发期目标与现金结算。","",f"决定：{summary['decision']}。本次没有更换原SIM或提交委托；没有把软件验证或历史拟合当成稳定前向超额。","","下一步：若历史门通过，冻结模型收集全新同价独立前向证据；否则保留失败结果，依据失败来源评估是否有新的独立信息，而不继续未登记调参。",f"QLib recorder：{recorder.id}。"]
    (card/"conclusion.md").write_text("\n".join(lines)+"\n")
    return summary
