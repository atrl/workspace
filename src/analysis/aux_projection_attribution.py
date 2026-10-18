"""Bounded, read-only attribution of one sealed auxiliary projection experiment."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(card, rid, names, manifest_sha):
    manifest=json.loads((card/"verification_manifest.json").read_text())
    digests={name:next(item["sha256"] for item in manifest["entries"]
        if item["path"].endswith("/artifacts/"+name)) for name in names}
    return load_verified_recorder_objects(card,expected_recorder_id=rid,artifacts=names,
        expected_manifest_sha256=manifest_sha,expected_artifact_sha256=digests)


def _difference(left,right):
    # Equality of recorded values, not a new signal/threshold or causal test.
    return ~(left.eq(right) | (left.isna() & right.isna()))


def _outputs(left,right,kind):
    pd.testing.assert_index_equal(left.index,right.index)
    columns=["soxl_weight","soxx_weight","cash_weight","tail_probability","alpha_probability"]
    rows=[]
    for name in columns:
        b,c=left[name].to_numpy(float),right[name].to_numpy(float)
        if not np.isfinite(b).all() or not np.isfinite(c).all():raise ValueError("invalid saved outputs")
        delta=c-b
        rows.append({"kind":kind,"output":name,"rows":len(b),"B_mean":float(b.mean()),"C_mean":float(c.mean()),
            "mean_C_minus_B":float(delta.mean()),"mean_absolute_difference":float(np.abs(delta).mean()),
            "RMSE":float(np.sqrt(np.mean(delta**2))),"max_absolute_difference":float(np.abs(delta).max()),
            "exactly_equal_rows":int((delta==0).sum())})
    return rows


def run(*,root,card):
    import qlib
    root,card=Path(root),Path(card)
    scope=json.loads((card/"failure_attribution_scope.json").read_text())
    plan=json.loads((card/"plan.json").read_text())
    if (card/"failure_attribution.json").exists():raise ValueError("fixed attribution already complete; no repeat")
    before={name:_sha(card/name) for name in ("verification_manifest.json","nas_binding.json")}
    if json.loads((card/"nas_binding.json").read_text())["evidence"]["sha256"]!=scope["archive_sha256"]:
        raise ValueError("sealed archive differs")
    qlib.init(provider_uri=str(root/plan["provider_init"]),region="us",kernels=1)
    names=[*[f"training_trace/C_aux_projection_{s}" for s in (71,172,273)],
        *[f"models/C_aux_projection_{s}" for s in (71,172,273)],
        *[f"native/C_aux_projection_{cost}" for cost in (0,2,5)],
        "analysis/ensembles","analysis/classification","analysis/smoothed_target","evaluation/prepared",
        "analysis/annual_metrics","analysis/quarterly_metrics"]
    own,own_audit=_read(card,scope["recorder_id"],names,scope["manifest_sha256"])
    bsource=plan["deferred_evaluation_sources"]["B_reuse"]
    bnames=[f"models/B_shared_{s}" for s in (71,172,273)]+["analysis/smoothed_targets"]
    baseline,b_audit=_read(root/bsource["card"],bsource["recorder_id"],bnames,bsource["inventory_sha256"])
    if b_audit["artifacts"]["analysis/smoothed_targets"]["sha256"]!=scope["additional_existing_B_target"]["sha256"]:
        raise ValueError("existing B target identity differs")
    traces,steps,directions,weights=[],[],[],[]
    for seed in (71,172,273):
        data=pd.DataFrame(own[f"training_trace/C_aux_projection_{seed}"])
        if list(data.epoch)!=list(range(1,301)) or not data.status.eq("completed").all():
            raise ValueError("not all300 completed training steps are present")
        conflict=data.strict_projection_triggered.astype(bool)
        epochs=data.loc[conflict,"epoch"]
        steps.append({"seed":seed,"epochs":len(data),"projection_steps":int(conflict.sum()),
            "first_projection_epoch":None if epochs.empty else int(epochs.min()),
            "last_projection_epoch":None if epochs.empty else int(epochs.max()),
            "full_parity_all_pass":bool(data.full_parity_pass.all()),"shared_parity_all_pass":bool(data.shared_parity_pass.all()),
            "full_parity_max_error":float(data.full_parity_error.max()),"shared_parity_max_error":float(data.shared_parity_error.max()),
            "global_clip_active_steps":int(data.actual_global_clip_coefficient.lt(1).sum()),
            "minimum_actual_clip_coefficient":float(data.actual_global_clip_coefficient.min()),
            "maximum_actual_preclip_norm":float(data.actual_global_preclip_norm.max()),
            "maximum_original_combined_norm_on_C_path":float(data.original_combined_preclip_norm_audit64.max()),
            "private_original_combined_unchanged_all":bool(data.private_original_combined_unchanged.all())})
        for objective in ("P","allocation"):
            for subset in ("full","shared"):
                values=data[f"{objective}_{subset}_actual_delta_linear_loss_change"]
                directions.append({"seed":seed,"objective":objective,"coordinates":subset,"epochs":len(values),
                    "negative_linear_estimate_count":int(values.lt(0).sum()),"positive_linear_estimate_count":int(values.gt(0).sum()),
                    "exact_zero_count":int(values.eq(0).sum()),"minimum":float(values.min()),"maximum":float(values.max()),
                    "mean":float(values.mean()),"median":float(values.median()),
                    "meaning":"negative is a first-order loss-decrease estimate only, not finite-step or return guarantee"})
        b=baseline[f"models/B_shared_{seed}"].network.state_dict()
        c=own[f"models/C_aux_projection_{seed}"].network.state_dict()
        if list(b)!=list(c):raise ValueError("saved parameter schema differs")
        for name in b:
            diff=(c[name].double()-b[name].double()).numpy()
            weights.append({"seed":seed,"parameter":name,"elements":int(b[name].numel()),
                "exact_tensor_equality":bool(torch.equal(b[name],c[name])),
                "max_absolute_difference":float(np.abs(diff).max()),"L2_difference":float(np.linalg.norm(diff))})
    output_rows=_outputs(own["analysis/ensembles"]["B_shared"],own["analysis/ensembles"]["C_aux_projection"],"raw_equal_seed_ensemble")
    output_rows+=_outputs(baseline["analysis/smoothed_targets"]["B_shared"],own["analysis/smoothed_target"],"saved_smoothed_target")
    prepared=own["evaluation/prepared"]
    books={key:value for key,value in prepared["references"].items() if key[0] in ("A_active_research_replay","B_shared")}
    books.update({("C_aux_projection",cost):own[f"native/C_aux_projection_{cost}"] for cost in (0,2,5)})
    exposure,divergence=[],[]
    for (family,cost),book in books.items():
        authority=book["authority"]
        exposure.append({"family":family,"slippage_bps":cost,"sessions":len(authority),
            "mean_ending_soxl_weight":float(authority.ending_soxl_weight.mean()),
            "mean_ending_soxx_weight":float(authority.ending_soxx_weight.mean()),
            "mean_ending_cash_like_weight":float(authority.ending_cash_weight.mean()),
            "mean_ending_beta":float((3*authority.ending_soxl_weight+authority.ending_soxx_weight).mean()),
            "mean_spendable_cash_usd":float(authority.ending_spendable_cash.mean()),
            "risk_nonzero_days":int(authority.risk_state.ne(0).sum()),
            "open_execute_days":int(authority.execute_open_target.sum()),
            "recovery_action_days":int(authority.recovery_action.sum())})
    pairs=[(("C_aux_projection",cost),(family,cost)) for cost in (0,2,5) for family in ("A_active_research_replay","B_shared")]
    pairs += [(("C_aux_projection",a),("C_aux_projection",b)) for a,b in ((0,2),(0,5),(2,5))]
    for left,right in pairs:
        a,b=books[left]["authority"].reset_index(drop=True),books[right]["authority"].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[["signal_date","trade_date","exit_date"]],b[["signal_date","trade_date","exit_date"]])
        for field in ("risk_state","risk_clock","execute_open_target","recovery_action","ending_cash_weight",
                      "ending_soxl_weight","ending_soxx_weight","ending_spendable_cash"):
            if field not in a or field not in b:
                divergence.append({"left":str(left),"right":str(right),"field":field,"available":False})
                continue
            diff=_difference(a[field],b[field]);changed=a.loc[diff,"trade_date"]
            numeric=a[field].astype(float)-b[field].astype(float)
            divergence.append({"left":str(left),"right":str(right),"field":field,"available":True,
                "different_sessions":int(diff.sum()),"first_difference":None if changed.empty else str(changed.iloc[0].date()),
                "mean_difference":float(numeric.mean()),"maximum_absolute_difference":float(numeric.abs().max()),
                "evidence":"saved native authority; each cost/account has its own endogenous state"})
    periods=[]
    for kind in ("annual","quarterly"):
        data=own[f"analysis/{kind}_metrics"]
        for cost in (0,2,5):
            candidate=data.loc[data.family.eq("C_aux_projection") & data.slippage_bps.eq(cost)].set_index("period")
            for family in ("B_shared","A_active_research_replay","SOXX"):
                comparison=data.loc[data.family.eq(family) & data.slippage_bps.eq(cost)].set_index("period")
                pd.testing.assert_index_equal(candidate.index,comparison.index)
                if not candidate.complete_period.equals(comparison.complete_period):raise ValueError("period coverage differs")
                for period,row in candidate.iterrows():
                    other=comparison.loc[period];ratio=float(row.terminal_wealth/other.terminal_wealth)
                    periods.append({"kind":kind,"period":str(period),"complete_period":bool(row.complete_period),
                        "slippage_bps":cost,"comparator":family,"C_return_pct":float(row.return_pct),
                        "comparator_return_pct":float(other.return_pct),"relative_wealth":ratio,"C_wins":ratio>1+1e-8})
    period_frame=pd.DataFrame(periods);win_rows=[]
    for key,group in period_frame.groupby(["kind","slippage_bps","comparator"]):
        for coverage in ("all","complete","partial"):
            selected=group if coverage=="all" else group.loc[group.complete_period.eq(coverage=="complete")]
            win_rows.append(dict(zip(("kind","slippage_bps","comparator"),key))|
                {"coverage":coverage,"periods":len(selected),"wins":int(selected.C_wins.sum()),"loss_or_tie":int((~selected.C_wins).sum())})
    tables={"gradient_steps":pd.DataFrame(steps),"actual_step_directions":pd.DataFrame(directions),
        "model_tensor_difference":pd.DataFrame(weights),"output_difference":pd.DataFrame(output_rows),
        "account_exposure":pd.DataFrame(exposure),"account_divergence":pd.DataFrame(divergence),
        "period_comparison":period_frame,"period_wins":pd.DataFrame(win_rows)}
    for name,table in tables.items():
        if len(table)>500:raise ValueError("attribution table exceeds slim scope")
        table.to_csv(card/("attribution_"+name+".csv"),index=False)
    all_weights=tables["model_tensor_difference"]
    result={"scope_sha256":_sha(card/"failure_attribution_scope.json"),"source_recorder_id":scope["recorder_id"],
        "source_archive_sha256":scope["archive_sha256"],"source_artifacts":own_audit["artifacts"],
        "B_source_artifacts":b_audit["artifacts"],"training_steps":steps,"actual_step_directions":directions,
        "all_parameter_equal_by_seed":{str(s):bool(all_weights.loc[all_weights.seed.eq(s),"exact_tensor_equality"].all()) for s in (71,172,273)},
        "existing_classification":own["analysis/classification"].to_dict(orient="records"),
        "small_tables":{name:"attribution_"+name+".csv" for name in tables},
        "new_fit_forward_autograd_replay_CI_source_NAS_reads":0,
        "limits":["C-path gradient/clip/delta diagnostics are not a reconstructed B optimizer counterfactual.",
                  "First-order delta signs do not guarantee finite-step primary/allocation loss or future return.",
                  "Account/control state differences are endogenous path descriptions, not controller or fees-only causal decompositions.",
                  "No threshold/window/seed/anchor rescue; fixed version remains closed and reused history is not independent."],
        "sealed_manifest_and_NAS_binding_unchanged":before=={name:_sha(card/name) for name in before}}
    if not result["sealed_manifest_and_NAS_binding_unchanged"]:raise ValueError("sealed source mutated")
    atomic_json(result,card/"failure_attribution.json")
    return {"steps":tables["gradient_steps"],"parameter_exact":result["all_parameter_equal_by_seed"],
            "output_difference":tables["output_difference"],"exposure":tables["account_exposure"],
            "period_wins":tables["period_wins"],"sealed_unchanged":True}
