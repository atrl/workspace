"""Bounded original Stage1 rolling-window objective research workflow."""
from __future__ import annotations

import torch
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.minute_intake import atomic_json
from src.model.minute_pit_allocator import TARGETS, _flat_index, _scaled
from src.model.stage1_rolling126_shortfall import (
    Stage1Rolling126ShortfallAllocator, WINDOW, rolling_soxx_shortfall,
)
from src.workflow.minute_pit_allocator import _auxiliary_dataset, _primary_dataset
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES, _project_legal_weights
from src.workflow.stage1_domain_heads import _read_source


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_from_original(original, root: Path) -> Stage1Rolling126ShortfallAllocator:
    return Stage1Rolling126ShortfallAllocator(
        base_features=original.base_features, extra_features=[], feature_masks={},
        spec={**asdict(original.spec), "root": str(root)}, seed=original.seed,
        cutoff=str(original.cutoff), train_end=str(original.train_end.date()),
        minimum_observations=original.minimum_observations,
    )


def prepare_training(*, root: Path, card: Path) -> dict:
    """Verify the old Qlib source and all train-only mechanics without fitting."""
    import qlib

    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / "plan.json").read_text())
    parent_path = root / plan["parent"]["stage1_plan"]
    if (sha(parent_path) != plan["parent"]["stage1_plan_sha256"]
            or sha(parent_path.parent / "summary.json") != plan["parent"]["stage1_summary_sha256"]
            or sha(root / plan["model"]["adapter"]) != plan["model"]["adapter_sha256"]):
        raise ValueError("blocked: frozen Stage1 parent or objective adapter changed")
    for name, key in [("A_0bp_parity_receipt", "A_0bp_parity_receipt_sha256"),
                      ("account_source_preflight_receipt", "account_source_preflight_sha256")]:
        if sha(root / plan["parent"][name]) != plan["parent"][key]:
            raise ValueError("blocked: prior original Stage1 parity/source evidence changed")
    parity = json.loads((root / plan["parent"]["A_0bp_parity_receipt"]).read_text())
    if (parity["status"] != "original_Stage1_0bp_reference_replay_parity_passed"
            or parity["funded_sessions"] != 944
            or parity["max_absolute_nav_usd_difference"] != 0):
        raise ValueError("blocked: original Stage1 A 0bp reference parity invalid")
    parent = json.loads(parent_path.read_text())
    qlib.init(provider_uri=str(root / parent["provider_init"]), region="us", kernels=1)
    source, audit = _read_source(root, parent["sources"]["training"])
    originals = {seed: source[f"models/base_outer_{seed}"] for seed in plan["model"]["seeds"]}
    first = originals[plan["model"]["seeds"][0]]
    if first.extra_features or len(first.base_features) != 39:
        raise ValueError("blocked: source is not the original 39-input Stage1")
    inputs = {"frame": source["inputs/daily_frame"], "features": source["inputs/features"],
              "common": source["inputs/common"], "timing": source["inputs/timing"],
              "spec": first.spec}
    dataset = _primary_dataset(inputs, train_end=str(first.train_end.date()),
                               predict_bounds=source["inputs/plan"]["outer_window"])
    auxiliary = _auxiliary_dataset(source["inputs/auxiliary"]["outer"],
                                   str(first.train_end.date()))
    rows = _flat_index(dataset.prepare("train", col_set=["feature", "label", "common", "meta"]))
    maturity = pd.to_datetime(rows[("meta", "label_available_at")])
    eligible = (maturity.notna() & maturity.lt(first.cutoff)
                & (rows.index <= first.train_end)
                & np.isfinite(rows["label"][list(TARGETS)].to_numpy(float)).all(axis=1))
    mature = rows.loc[eligible]
    if (len(mature) != 3014 or len(auxiliary.prepare("train")) != 20741
            or len(mature["common"].columns) != 16
            or len(mature) < WINDOW):
        raise ValueError("blocked: original Stage1 mature training cohorts differ")
    calendar_spec = plan["model"]["training_calendar"]
    calendar_provider = root / calendar_spec["provider"]
    if sha(calendar_provider / "qlib_store_manifest.json") != calendar_spec["manifest_sha256"]:
        raise ValueError("blocked: training daily calendar provider changed")
    qlib.init(provider_uri=str(calendar_provider), region="us", kernels=1)
    from qlib.data import D
    native_sessions = pd.DatetimeIndex(D.calendar(start_time=mature.index.min(),
                                                   end_time=mature.index.max(), freq="day"))
    if not pd.DatetimeIndex(mature.index).equals(native_sessions):
        raise ValueError("blocked: mature Stage1 training rows are not consecutive native Qlib sessions")
    primary_x = _scaled(mature["feature"], first.base_features, first.primary_scaler)
    common_x = _scaled(mature["common"], list(COMMON_FEATURES), first.common_scaler)
    if not torch.isfinite(primary_x).all() or not torch.isfinite(common_x).all():
        raise ValueError("blocked: original Stage1 scaled training inputs are not finite")
    returns = torch.tensor(mature["label"][list(TARGETS[:2])].to_numpy(np.float32))
    tensor_proofs = []
    for seed, original in originals.items():
        candidate = candidate_from_original(original, root)
        if candidate.base_features != original.base_features or candidate.spec != original.spec:
            raise ValueError("blocked: rolling candidate changed Stage1 specification")
        old, new = original._initialize(), candidate._initialize()
        if not all(torch.equal(value, new.state_dict()[name])
                   for name, value in old.state_dict().items()):
            raise ValueError("blocked: original Stage1 and rolling candidate initial tensors differ")
        logits, _, _ = new(primary_x, common_x)
        weights = _project_legal_weights(logits,
            maximum_soxl_weight=candidate.spec.maximum_soxl_weight,
            minimum_cash_weight=candidate.spec.minimum_cash_weight)
        objective = rolling_soxx_shortfall(weights, returns,
                                           cost_bps=candidate.spec.training_cost_bps)
        if not torch.isfinite(objective) or objective <= 0:
            raise ValueError("blocked: initialized Stage1 has no rolling SOXX shortfall")
        objective.backward()
        head_gradients = [parameter.grad for parameter in new.allocation_head.parameters()]
        if any(gradient is None or not torch.isfinite(gradient).all()
               for gradient in head_gradients):
            raise ValueError("blocked: initialized Stage1 allocation head gradient is not finite")
        head_gradient_abs_sum = float(sum(gradient.abs().sum() for gradient in head_gradients))
        if head_gradient_abs_sum <= 0:
            raise ValueError("blocked: rolling objective has no allocation head gradient")
        tensor_proofs.append({"seed": seed, "initial_parameter_tensors_identical": True,
                              "original_parameter_count": sum(x.numel() for x in old.parameters()),
                              "candidate_parameter_count": sum(x.numel() for x in new.parameters()),
                              "initialized_rolling_shortfall": float(objective.detach()),
                              "allocation_head_gradient_abs_sum": head_gradient_abs_sum,
                              "backward_without_optimizer_step": True})
    proxy = torch.tensor([[0.0, 0.85, 0.15]] * len(returns), dtype=torch.float32,
                         requires_grad=True)
    shortfall = rolling_soxx_shortfall(proxy, returns, cost_bps=first.spec.training_cost_bps)
    if not torch.isfinite(shortfall) or shortfall <= 0:
        raise ValueError("blocked: real training path has no rolling SOXX shortfall")
    shortfall.backward()
    if proxy.grad is None or not torch.isfinite(proxy.grad).all() or proxy.grad.abs().sum() <= 0:
        raise ValueError("blocked: rolling objective has no finite train-path gradient")
    receipt = {
        "status": "verified_no_fit_original_Stage1_training_and_rolling_gradient",
        "plan_sha256": sha(card / "plan.json"),
        "source_recorder_id": parent["sources"]["training"]["recorder_id"],
        "source_audit": audit,
        "mature_primary_rows": len(mature), "auxiliary_rows": 20741,
        "first_mature_signal": str(mature.index.min().date()),
        "last_mature_signal": str(mature.index.max().date()),
        "last_label_available_at": str(maturity.loc[eligible].max()),
        "rolling_window_sessions": WINDOW,
        "rolling_windows_in_training": len(mature) - WINDOW + 1,
        "training_calendar_provider": calendar_spec["provider"],
        "training_calendar_manifest_sha256": calendar_spec["manifest_sha256"],
        "native_qlib_consecutive_sessions": len(native_sessions),
        "cash_15pct_proxy_training_shortfall": float(shortfall.detach()),
        "cash_15pct_proxy_gradient_abs_sum": float(proxy.grad.abs().sum()),
        "three_seed_initialization": tensor_proofs,
        "new_model_fits": 0, "new_native_accounts": 0,
        "evaluation_account_objects_loaded": 0,
    }
    atomic_json(receipt, card / "training_preflight_result.json")
    return {"plan": plan, "dataset": dataset, "auxiliary": auxiliary,
            "originals": originals, "inputs": inputs, "receipt": receipt,
            "reference_prediction_indexes": {
                seed: source["analysis/outer_predictions"][("base", seed)].index
                for seed in plan["model"]["seeds"]}}


def _require_approval(root: Path, card: Path, plan: dict) -> dict:
    import sklearn

    if sklearn.__version__ != plan["runtime"]["saved_action_classifier_scikit_learn"]:
        raise ValueError("needs_runtime: original Stage1 classifier version differs")
    review_path, permit_path = card / "design_review_final.json", card / "execution_approval.json"
    if not review_path.is_file() or not permit_path.is_file():
        raise ValueError("blocked: independent design review and exact execution permit missing")
    review, permit = json.loads(review_path.read_text()), json.loads(permit_path.read_text())
    digest = sha(card / "plan.json")
    if (review.get("verdict") != "APPROVE_DESIGN"
            or review.get("plan_sha256") != digest
            or permit.get("approved") is not True
            or permit.get("plan_sha256") != digest
            or permit.get("review_sha256") != sha(review_path)):
        raise ValueError("blocked: design approval does not bind the frozen plan")
    required = ["src/workflow/stage1_rolling126_shortfall.py",
                "src/model/stage1_rolling126_shortfall.py",
                plan["account_replay"]["helper_module"],
                "src/backtest/stage1_domain_heads_raw.py",
                "src/backtest/native_raw_corporate.py",
                "src/workflow/pit_constituent_auxiliary_training_allocation.py"]
    hashes = permit.get("code_sha256", {})
    if not all(name in hashes and sha(root / name) == hashes[name] for name in required):
        raise ValueError("blocked: independently reviewed execution code differs")
    replay = plan["account_replay"]
    pinned = {replay["helper_module"]: replay["helper_module_sha256"],
              "src/backtest/stage1_domain_heads_raw.py": replay["raw_path_module_sha256"],
              "src/backtest/native_raw_corporate.py": replay["native_validator_module_sha256"]}
    if not all(sha(root / name) == digest for name, digest in pinned.items()):
        raise ValueError("blocked: frozen Stage1 account replay implementation changed")
    return {"review_sha256": sha(review_path), "approval_sha256": sha(permit_path),
            "code_sha256": hashes}


def _funded_date_digest(frame: pd.DataFrame) -> str:
    dates = frame[["signal_date", "trade_date", "exit_date"]]
    payload = "\n".join("|".join(str(value.date()) for value in row)
                        for row in dates.itertuples(index=False, name=None)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _analyze_account(*, account: pd.DataFrame, reference_soxx: pd.DataFrame,
                     reference_stage1: pd.DataFrame, sampled_drawdown: float,
                     plan: dict) -> dict:
    from src.analysis.integrated_allocation import paired_metrics, stationary_bootstrap_excess

    relative = paired_metrics(account, reference_soxx)
    increment = paired_metrics(account, reference_stage1)
    annual_rows, quarter_rows = [], []
    trade_dates = pd.to_datetime(account.trade_date)
    for kind, frequency, output in (("year", "Y", annual_rows),
                                    ("quarter", "Q", quarter_rows)):
        periods = trade_dates.dt.to_period(frequency).astype(str)
        for period in periods.unique():
            take = periods.eq(period)
            output.append({"kind": kind, "period": period, "sessions": int(take.sum()),
                           "complete_period": period not in (periods.iloc[0], periods.iloc[-1]),
                           **paired_metrics(account.loc[take], reference_soxx.loc[take])})
    annual, quarter = pd.DataFrame(annual_rows), pd.DataFrame(quarter_rows)
    if set(annual.loc[annual.complete_period, "period"]) != {"2024", "2025"}:
        raise ValueError("blocked: frozen complete-year coverage differs")
    log_excess = (np.log1p(account.strategy_return.to_numpy(float))
                  - np.log1p(reference_soxx.strategy_return.to_numpy(float)))
    rolling_log = pd.Series(log_excess).rolling(126).sum().iloc[125:].reset_index(drop=True)
    rolling = pd.DataFrame({"first_trade": account.trade_date.iloc[:len(rolling_log)].to_numpy(),
                            "last_exit": account.exit_date.iloc[125:].to_numpy(),
                            "relative_log": rolling_log,
                            "relative_return_pct": 100 * np.expm1(rolling_log)})
    if len(rolling) != 819 or not np.isfinite(rolling.relative_log).all():
        raise ValueError("blocked: complete contiguous 126-session windows differ")
    bootstrap = plan["hard_screen_0bp"]["paired_stationary_bootstrap"]
    uncertainty = stationary_bootstrap_excess(
        account, reference_soxx, mean_block=bootstrap["mean_block"],
        resamples=bootstrap["draws"], seed=bootstrap["seed"],
        alpha=bootstrap["one_sided_alpha"])
    complete_years = annual.loc[annual.complete_period]
    gates = {
        "SOXX_terminal_net_wealth": bool(relative["relative_wealth"] > 1 + 1e-8),
        "available_drawdown_30pct": bool(relative["max_drawdown_pct"] >= -30
                                        and sampled_drawdown >= -30),
        "CVaR5_no_worse_SOXX": bool(relative["cvar5_pct"] >= relative["benchmark_cvar5_pct"]),
        "complete_years_70pct": bool(len(complete_years) >= 2
                                     and complete_years.relative_wealth.gt(1 + 1e-8).mean() >= .7),
        "rolling_126_70pct": bool(rolling.relative_log.gt(0).mean() >= .7),
        "SOXX_paired_lower_positive": bool(uncertainty["daily_lower_bound"] > 0),
        "full_minute_risk_qualified": False,
        "independent_forward_qualified": False,
    }
    historical = all(value for name, value in gates.items()
                     if name not in {"full_minute_risk_qualified", "independent_forward_qualified"})
    return {"SOXX_metrics": relative, "Stage1_increment_metrics": increment,
            "annual": annual, "quarter": quarter, "rolling": rolling,
            "rolling_summary": {"windows": len(rolling),
                                "positive_fraction": float(rolling.relative_log.gt(0).mean()),
                                "minimum_relative_return_pct": float(rolling.relative_return_pct.min())},
            "uncertainty": uncertainty, "gates": gates,
            "historical_0bp_gate_pass": bool(historical)}


def run(*, root: Path, card: Path) -> dict:
    """One frozen three-fit, one-account Stage1 0bp experiment."""
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    from src.workflow.minute_pit_allocator import _ensemble, _prediction_dates
    from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
    from src.workflow.stage1_nvda_guidance_receiver import prepare_account_sources
    from src.workflow.stage1_domain_heads import _provider
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
    from src.backtest.native_raw_corporate import native_raw_validate

    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / "plan.json").read_text())
    approval = _require_approval(root, card, plan)
    state_path = card / "run_state.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("analysis_done"):
            return resume_nas_archive(root=root, card=card)
        if any(previous.get(key) for key in ("target_attempted", "provider_attempted",
                                             "path_attempted", "native_attempted",
                                             "bootstrap_attempted")):
            raise ValueError("blocked: fixed account or analysis attempt already consumed")
    preflight = json.loads((card / "training_preflight_result.json").read_text())
    if (preflight["status"] != "verified_no_fit_original_Stage1_training_and_rolling_gradient"
            or preflight["plan_sha256"] != sha(card / "plan.json")
            or preflight["mature_primary_rows"] != 3014
            or preflight["auxiliary_rows"] != 20741):
        raise ValueError("blocked: plan-bound Stage1 no-fit source preflight differs")
    prepared = prepare_training(root=root, card=card)
    tracking = card / "qlib_recorder"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
             "plan_sha256": sha(card / "plan.json"), "fit_attempts": [],
             "fits_done": [], "prediction_attempts": [], "predictions_done": [],
             "provider_attempted": False, "target_attempted": False,
             "path_attempted": False, "native_attempted": False,
             "bootstrap_attempted": False, "new_native_accounts": 0,
             "artifact_sha256": {"models": {}, "predictions": {}}}
    if state["plan_sha256"] != sha(card / "plan.json"):
        raise ValueError("blocked: run state belongs to another frozen design")
    predictions = {}
    recorder_args = {"experiment_name": plan["experiment"], "uri": str(tracking)}
    if state.get("recorder_id"):
        recorder_args.update(recorder_id=state["recorder_id"], resume=True)
    with R.start(**recorder_args):
        recorder = R.get_recorder()
        if state.get("recorder_id") and (state["recorder_id"] != recorder.id
                                         or state["experiment_id"] != str(recorder.experiment_id)):
            raise ValueError("blocked: resumed Qlib Recorder identity changed")
        state.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(state, state_path)
        artifact_root = tracking / state["experiment_id"] / state["recorder_id"] / "artifacts"
        if not state.get("sources_saved"):
            recorder.save_objects(artifact_path="inputs", plan=plan, preflight=prepared["receipt"],
                                  approval=approval, dataset=prepared["dataset"],
                                  auxiliary_dataset=prepared["auxiliary"])
            recorder.save_objects(artifact_path="sources",
                                  workflow=(root / "src/workflow/stage1_rolling126_shortfall.py").read_text(),
                                  adapter=(root / plan["model"]["adapter"]).read_text())
            state["sources_saved"] = True
            atomic_json(state, state_path)
        for seed in plan["model"]["seeds"]:
            key = "B_rolling126_" + str(seed)
            original = prepared["originals"][seed]
            if key in state["fits_done"]:
                if sha(artifact_root / "models" / key) != state["artifact_sha256"]["models"][key]:
                    raise ValueError("blocked: resumed fitted model bytes changed")
                model = recorder.load_object("models/" + key)
            else:
                if (key in state["fit_attempts"] or
                        len(state["fit_attempts"]) >= plan["budget"]["new_model_fit_attempts_max"]):
                    raise ValueError("blocked: frozen fit attempt consumed")
                model = candidate_from_original(original, root)
                state["fit_attempts"].append(key)
                atomic_json(state, state_path)
                model.fit(prepared["dataset"], auxiliary_dataset=prepared["auxiliary"])
                if (model.audit["training_rows"] != 3014
                        or model.audit["auxiliary_rows"] != 20741
                        or model.audit["epochs"] != plan["model"]["epochs_per_seed"]
                        or model.primary_scaler != original.primary_scaler
                        or model.common_scaler != original.common_scaler):
                    raise ValueError("blocked: rolling model training cohort or original Stage1 scalers differ")
                recorder.save_objects(artifact_path="models", **{key: model})
                state["artifact_sha256"]["models"][key] = sha(artifact_root / "models" / key)
                state["fits_done"].append(key)
                atomic_json(state, state_path)
            if key in state["predictions_done"]:
                if sha(artifact_root / "predictions" / key) != state["artifact_sha256"]["predictions"][key]:
                    raise ValueError("blocked: resumed prediction bytes changed")
                prediction = recorder.load_object("predictions/" + key)
            else:
                if key in state["prediction_attempts"]:
                    raise ValueError("blocked: frozen prediction attempt consumed")
                state["prediction_attempts"].append(key)
                atomic_json(state, state_path)
                prediction = _prediction_dates(model.predict(prepared["dataset"]))
                pd.testing.assert_index_equal(prediction.index,
                    prepared["reference_prediction_indexes"][seed], check_names=False)
                recorder.save_objects(artifact_path="predictions", **{key: prediction})
                state["artifact_sha256"]["predictions"][key] = sha(artifact_root / "predictions" / key)
                state["predictions_done"].append(key)
                atomic_json(state, state_path)
            recorder.log_metrics(completed_fits=len(state["fits_done"]),
                                 completed_predictions=len(state["predictions_done"]))
            predictions[("B_rolling126", seed)] = prediction
    if len(state["fits_done"]) != 3 or len(state["predictions_done"]) != 3:
        raise ValueError("blocked: all three models must be sealed before economic reads")
    artifacts = tracking / state["experiment_id"] / state["recorder_id"] / "artifacts"
    model_seal = {"status": "three_models_and_predictions_sealed",
                  "plan_sha256": state["plan_sha256"],
                  "recorder_id": state["recorder_id"],
                  "artifact_sha256": {kind: {key: sha(artifacts / kind / key)
                                                for key in state["fits_done"]}
                                      for kind in ("models", "predictions")},
                  "new_native_accounts": 0}
    atomic_json(model_seal, card / "model_seal.json")
    helper = plan["account_replay"]
    helper_card = root / helper["helper_card"]
    before_receipt = sha(root / plan["parent"]["account_source_preflight_receipt"])
    account = prepare_account_sources(root=root, card=helper_card)
    if (sha(root / plan["parent"]["account_source_preflight_receipt"]) != before_receipt
            or before_receipt != plan["parent"]["account_source_preflight_sha256"]):
        raise ValueError("blocked: borrowed original Stage1 account-source proof changed")
    dates = pd.DatetimeIndex(account["market"].signal_date)
    values = _ensemble(predictions, "B_rolling126", plan["model"]["seeds"])
    if len(values) != 946 or not dates.isin(values.index).all():
        raise ValueError("blocked: original prediction calendar differs")
    returns = account["market"].set_index("signal_date")[["soxl_return", "soxx_return"]]
    original = prepared["originals"][plan["model"]["seeds"][0]]
    state["target_attempted"] = True
    atomic_json(state, state_path)
    target = _smooth_weights(values.reindex(dates), spec=original.spec,
                             candidate=original.candidate, asset_returns=returns)
    if len(target) != plan["economics_first_stage"]["funded_sessions"]:
        raise ValueError("blocked: funded Stage1 target length differs")
    recorder.start_run()
    try:
        recorder.save_objects(artifact_path="targets", B_rolling126=target)
        state["provider_attempted"] = True
        atomic_json(state, state_path)
        provider, provider_manifest = _provider(root, {"points": account["points"],
            "guard": account["guard"], "plan": {"provider_output": helper["provider_output"]}})
        recorder.save_objects(artifact_path="inputs", native_provider_manifest=provider_manifest)
        frame = account["frame"].copy(deep=True)
        outputs = ["soxl_weight", "soxx_weight", "cash_weight",
                   "tail_probability", "alpha_probability"]
        pd.testing.assert_index_equal(target.index, dates, check_names=False)
        for name in outputs:
            frame[name] = target[name].to_numpy(copy=True)
        recovery = IntradayRecoveryDirectAllocationSpec(root=root, **account["recovery_spec"])
        action = Stage1OpenActionValueSpec(root=root, **account["action_spec"])
        execution = {"initial_cash": plan["economics_first_stage"]["initial_cash_usd"],
                     "tariff": account["parent"]["tariff"]}
        state["path_attempted"] = True
        atomic_json(state, state_path)
        path = replay_stage1_raw(frame, point_prices=account["points"],
            schedule=account["schedule"], plan=execution, controller=account["controller"],
            recovery_spec=recovery, action_spec=action, selection=account["selection"],
            slippage_bps=0)
        if (_funded_date_digest(path["replay"]) != helper["funded_date_tuple_sha256"]
                or len(path["points"]) != 3778):
            raise ValueError("blocked: candidate account calendar differs from original Stage1")
        recorder.save_objects(artifact_path="paths", B_rolling126_0=path)
        state["native_attempted"] = True
        atomic_json(state, state_path)
        book = native_raw_validate(path, provider_uri=provider,
            point_prices=account["points"], schedule=account["schedule"],
            tariff=account["parent"]["tariff"], cost=0,
            initial_cash=execution["initial_cash"])
        if book["audit"]["status"] != "passed":
            raise ValueError("blocked: rolling candidate failed independent native validation")
        recorder.save_objects(artifact_path="native", B_rolling126_0=book)
        state["new_native_accounts"] = 1
        atomic_json(state, state_path)
        state["bootstrap_attempted"] = True
        atomic_json(state, state_path)
        result = _analyze_account(account=book["authority"],
            reference_soxx=account["references"]["SOXX"]["authority"],
            reference_stage1=account["references"]["Stage1_A"]["authority"],
            sampled_drawdown=-100 * float(path["replay"].monitored_drawdown.max()),
            plan=plan)
        for name in ("annual", "quarter"):
            result[name].to_csv(card / (name + "_metrics.csv"), index=False)
            recorder.save_objects(artifact_path="analysis", **{name: result[name]})
        recorder.save_objects(artifact_path="analysis/rolling", all_windows=result["rolling"])
        recorder.save_objects(artifact_path="analysis", metrics=result["SOXX_metrics"],
            increment=result["Stage1_increment_metrics"], gates=result["gates"],
            uncertainty=result["uncertainty"], rolling_summary=result["rolling_summary"])
        recorder.log_metrics(completed_fits=3, completed_native_accounts=1,
            SOXX_relative_wealth=result["SOXX_metrics"]["relative_wealth"],
            historical_0bp_pass=int(result["historical_0bp_gate_pass"]))
        recorder.end_run(status=Recorder.STATUS_FI)
    except BaseException:
        recorder.end_run(status=Recorder.STATUS_FA)
        raise
    summary = {
        "experiment": plan["experiment"], "plan_sha256": state["plan_sha256"],
        "decision": ("historical_0bp_pass_needs_other_cost_risk_and_forward"
                     if result["historical_0bp_gate_pass"] else
                     "not_promoted_fixed_Stage1_rolling126_objective_0bp"),
        "objective_achieved": False, "historical_0bp_gate_pass": result["historical_0bp_gate_pass"],
        "SOXX_metrics": result["SOXX_metrics"],
        "Stage1_increment_metrics": result["Stage1_increment_metrics"],
        "rolling_summary": result["rolling_summary"],
        "uncertainty": result["uncertainty"], "gates": result["gates"],
        "new_model_fits": 3, "new_native_accounts": 1, "new_bootstrap_runs": 1,
        "recorder_id": state["recorder_id"], "experiment_id": state["experiment_id"],
        "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / state["experiment_id"] / state["recorder_id"]),
        "provider_uri": str(provider), "reused_market_history": True,
        "independent_forward_qualified_days": 0,
        "artifact_policy": {"policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*",
                "predictions/*", "targets/*", "paths/*", "native/*", "analysis/rolling/*"]},
        "limitations": ["Original training objective uses a differentiable weight proxy, not the later own-state risk controller",
                        "2023-2026 history has been repeatedly used and is development-only",
                        "Available 3778 marks do not qualify full-minute maximum drawdown",
                        "2/5bp native accounts and independent forward evidence absent"],
    }
    atomic_json(summary, card / "summary.json")
    atomic_json({"status": summary["decision"], "plan_sha256": state["plan_sha256"],
                 "recorder_id": state["recorder_id"],
                 "SOXX_relative_wealth": result["SOXX_metrics"]["relative_wealth"],
                 "Stage1_relative_wealth": result["Stage1_increment_metrics"]["relative_wealth"],
                 "rolling_positive_fraction": result["rolling_summary"]["positive_fraction"],
                 "SOXX_bootstrap_lower": result["uncertainty"]["daily_lower_bound"],
                 "gates": result["gates"], "new_model_fits": 3,
                 "new_native_accounts": 1, "independent_forward_qualified_days": 0},
                card / "result_review.json")
    (card / "conclusion.md").write_text(
        "# 原 Stage1 的126日相对短缺损失：固定0bp结果\n\n"
        "完成原39/16输入、三seed各300轮与一本独立原价Qlib账户；冻结动作、风险、恢复、Tiered费用不变。\n\n"
        f"候选相对SOXX期末财富 {result['SOXX_metrics']['relative_wealth']:.6f}；"
        f"相对原Stage1 A期末财富 {result['Stage1_increment_metrics']['relative_wealth']:.6f}；"
        f"126日正超额窗口 {100*result['rolling_summary']['positive_fraction']:.2f}%；"
        f"SOXX配对区块下界 {result['uncertainty']['daily_lower_bound']:.9f}。"
        "完整年度、季度、819个重叠窗口和全部失败门保存在原生Recorder及小表。\n\n"
        f"决定：`{summary['decision']}`。训练滚动损失是可微代理，不能替代完整自身风险账户；"
        "现有价格点不是全分钟风险认证，已复用历史不是独立前向。"
        + ("下一步另行冻结2/5bp自身账户和前向来源。\n" if result["historical_0bp_gate_pass"]
           else "关闭这个固定目标版本，不调系数、窗口、风险阈值或成本挽救；总SOXX目标继续。\n"))
    artifact_files = sorted(x for x in artifacts.rglob("*") if x.is_file())
    atomic_json({"recorder_id": state["recorder_id"],
                 "entries": [{"path": str(x.relative_to(card)), "sha256": sha(x),
                              "bytes": x.stat().st_size} for x in artifact_files]},
                card / "verification_manifest.json")
    state["analysis_done"] = True
    atomic_json(state, state_path)
    return resume_nas_archive(root=root, card=card)


def resume_nas_archive(*, root: Path, card: Path) -> dict:
    """Verify completed Recorder bytes, then archive without economic reruns."""
    import shlex
    from src.data.nas_market_data import _ssh, archive_nas_recorder

    root, card = Path(root).resolve(), Path(card).resolve()
    state_path = card / "run_state.json"
    state = json.loads(state_path.read_text())
    summary = json.loads((card / "summary.json").read_text())
    if (not state.get("analysis_done") or state["plan_sha256"] != sha(card / "plan.json")
            or summary["plan_sha256"] != state["plan_sha256"]
            or state["recorder_id"] != summary["recorder_id"]
            or len(state["fits_done"]) != 3 or state["new_native_accounts"] != 1):
        raise ValueError("blocked: completed analysis checkpoint differs")
    manifest = json.loads((card / "verification_manifest.json").read_text())
    if manifest["recorder_id"] != summary["recorder_id"]:
        raise ValueError("blocked: completed Recorder manifest differs")
    for entry in manifest["entries"]:
        path = card / entry["path"]
        if sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("blocked: completed Recorder bytes differ")
    binding_path = card / "nas_binding.json"
    if not binding_path.exists():
        parent_binding = root / "research/stage1_nvda_guidance_receiver_20261022/nas_binding.json"
        binding = json.loads(parent_binding.read_text())
        if binding.get("verified") is not True or binding.get("authority") != "yanxuan_nas":
            raise ValueError("blocked: parent NAS source binding is unverified")
        binding.pop("evidence", None)
        binding["research_card"] = str(card.relative_to(root))
        atomic_json(binding, binding_path)
    if "nas_evidence" not in summary:
        summary["nas_evidence"] = archive_nas_recorder(binding_path,
            workspace_root=root, card=card, recorder_id=summary["recorder_id"],
            experiment_id=summary["experiment_id"])
        atomic_json(summary, card / "summary.json")
    evidence = summary["nas_evidence"]
    binding = json.loads(binding_path.read_text())
    if binding.get("evidence", {}).get("sha256") != evidence["sha256"]:
        raise ValueError("blocked: NAS archive binding differs from summary")
    remote_sha = _ssh(binding["ssh"], "sha256sum " + shlex.quote(evidence["nas_path"])).split()[0]
    if remote_sha != evidence["sha256"]:
        raise ValueError("blocked: NAS archive remote byte readback differs")
    summary["nas_readback"] = {"sha256": remote_sha, "bytes": evidence["size_bytes"],
                                "verified": True}
    atomic_json(summary, card / "summary.json")
    state["completed"] = True
    atomic_json(state, state_path)
    return summary
