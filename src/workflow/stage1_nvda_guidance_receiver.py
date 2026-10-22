"""Bounded original-Stage1 guidance receiver preparation; no implicit fit."""
from __future__ import annotations

import torch  # Load the repository's numerical runtime before pandas/OpenMP users.
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.minute_intake import atomic_json
from src.data.nvda_own_guidance_realization import FEATURE, FLAGS, align_realizations
from src.model.minute_pit_allocator import TARGETS, _flat_index
from src.model.stage1_nvda_guidance import (
    Stage1NvdaGuidanceAllocator, initialization_contract,
)
from src.workflow.minute_pit_allocator import _auxiliary_dataset, _primary_dataset
from src.workflow.stage1_domain_heads import _read_source
from src.workflow.stage1_runtime_release import load_runtime_release


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _guidance_source(plan: dict) -> dict:
    value = plan["source"]
    return {
        "card": value["card"],
        "recorder_id": value["recorder_id"],
        "archive": value["NAS_archive_sha256"],
        "inventory_sha256": value["verification_manifest_sha256"],
        "artifacts": ["analysis/pairs"],
        "artifact_sha256": {"analysis/pairs": value["pairs_artifact_sha256"]},
    }


def _paired_models(original, root: Path):
    args = dict(base_features=original.base_features,
                spec={**asdict(original.spec), "root": str(root)},
                seed=original.seed, cutoff=str(original.cutoff),
                train_end=str(original.train_end.date()),
                minimum_observations=original.minimum_observations)
    return (Stage1NvdaGuidanceAllocator(**args, mode="neutral"),
            Stage1NvdaGuidanceAllocator(**args, mode="observed"))


def prepare_training(*, root: Path, card: Path) -> dict:
    """Verify the frozen training inputs and return a DatasetH without fitting.

    The pinned source container includes later rows for calendar alignment. Only
    the pre-2023 mature-label training segment is prepared and inspected here;
    evaluation accounts, prices and scores are not loaded by this function.
    """
    import qlib

    root, card = Path(root).resolve(), Path(card).resolve()
    plan_path = card / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["execution_authorized"]:
        raise ValueError("blocked: preparation expects an unapproved design")
    parent_path = root / plan["parent"]["plan"]
    if _sha(parent_path) != plan["parent"]["plan_sha256"]:
        raise ValueError("blocked: frozen Stage1 parent plan differs")
    if _sha(root / plan["parent"]["summary"]) != plan["parent"]["summary_sha256"]:
        raise ValueError("blocked: frozen Stage1 parent summary differs")
    if _sha(root / plan["source"]["alignment_module"]) != plan["source"]["alignment_module_sha256"]:
        raise ValueError("blocked: guidance formula/timing module differs")
    if _sha(root / plan["model"]["adapter"]) != plan["model"]["adapter_sha256"]:
        raise ValueError("blocked: guidance model adapter differs")
    binding = plan["calendar_binding"]
    binding_path = root / binding["result"]
    if _sha(binding_path) != binding["result_sha256"]:
        raise ValueError("blocked: funded Stage1 calendar binding differs")
    bound = json.loads(binding_path.read_text())
    if (bound["status"] != "exact_944_signal_trade_exit_binding_verified"
            or bound["funded_date_tuple_sha256"] != binding["funded_date_tuple_sha256"]
            or bound["funded_prediction_rows"] != plan["economics_first_stage"]["funded_sessions"]):
        raise ValueError("blocked: funded signal-to-fill binding invalid")
    parent = json.loads(parent_path.read_text())
    qlib.init(provider_uri=str(root / parent["provider_init"]), region="us", kernels=1)
    source, train_audit = _read_source(root, parent["sources"]["training"])
    guidance, guidance_audit = _read_source(root, _guidance_source(plan))
    originals = {seed: source[f"models/base_outer_{seed}"] for seed in plan["model"]["seeds"]}
    first = originals[plan["model"]["seeds"][0]]
    if first.extra_features or len(first.base_features) != plan["model"]["base_primary_features"]:
        raise ValueError("blocked: source is not the original Stage1 allocator")
    frame = source["inputs/daily_frame"]
    base_features = source["inputs/features"]
    timing = source["inputs/timing"]
    realized, trace, events = align_realizations(guidance["analysis/pairs"], timing.decision_at)
    pd.testing.assert_index_equal(base_features.index, realized.index, check_names=False)
    if set([*FLAGS, FEATURE]).intersection(base_features.columns):
        raise ValueError("blocked: proposed guidance coordinate already exists")
    added = base_features.join(realized)
    inputs = {"frame": frame, "features": added,
              "common": source["inputs/common"], "timing": timing,
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
    actual = rows.loc[eligible]
    if (len(actual) != plan["model"]["primary_training_rows_expected"]
            or len(auxiliary.prepare("train")) != plan["model"]["auxiliary_training_rows_expected"]
            or len(actual["common"].columns) != plan["model"]["shared_common_features"]
            or int(actual["feature"][FLAGS[1]].sum())
            != plan["model"]["source_comparable_primary_rows_expected"]):
        raise ValueError("blocked: matched Stage1 training domain differs")
    initialization = []
    for seed, original in originals.items():
        if original.base_features != first.base_features or original.spec != first.spec:
            raise ValueError("blocked: original Stage1 seed specification differs")
        neutral, observed = _paired_models(original, root)
        proof = initialization_contract(original, neutral, observed)
        neutral._fit_extra_scaler(actual["feature"])
        observed._fit_extra_scaler(actual["feature"])
        if neutral.extra_scaler != observed.extra_scaler:
            raise ValueError("blocked: paired guidance train-only scalers differ")
        initialization.append({"seed": seed, **proof,
                               "eligible_numeric_scaler_rows": neutral.extra_scaler[FEATURE]["observations"]})
    receipt = {
        "status": "verified_no_fit_training_dataset",
        "plan_sha256": _sha(plan_path),
        "source_recorder_ids": {"training": parent["sources"]["training"]["recorder_id"],
                                "guidance": plan["source"]["recorder_id"]},
        "source_artifact_hashes_verified": True,
        "primary_mature_rows": len(actual),
        "auxiliary_rows": plan["model"]["auxiliary_training_rows_expected"],
        "comparable_primary_rows": int(actual["feature"][FLAGS[1]].sum()),
        "last_mature_label_at": str(maturity.loc[eligible].max()),
        "three_seed_initialization": initialization,
        "source_event_rows": len(events),
        "new_model_fits": 0, "new_native_accounts": 0,
        "evaluation_account_objects_loaded": 0,
        "source_audits": {"training": train_audit, "guidance": guidance_audit},
    }
    atomic_json(receipt, card / "training_dataset_preflight.json")
    return {"plan": plan, "dataset": dataset, "auxiliary": auxiliary,
            "originals": originals, "guidance_trace": trace,
            "reference_prediction_indexes": {
                seed: source["analysis/outer_predictions"][("base", seed)].index
                for seed in plan["model"]["seeds"]},
            "receipt": receipt}


def fit_and_seal_models(*, root: Path, card: Path) -> dict:
    """Use the one frozen Qlib Recorder; never fit after an attempt fails."""
    from qlib.workflow import R
    from src.workflow.minute_pit_allocator import _prediction_dates

    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / "plan.json").read_text())
    approval = _require_execution_approval(root, card, plan)
    _check_prefit_receipts(card, plan)
    prepared = prepare_training(root=root, card=card)
    state_path = card / "run_state.json"
    digest = _sha(card / "plan.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "plan_sha256": digest, "fit_attempts": [], "fits_done": [],
        "prediction_attempts": [], "predictions_done": [],
        "path_attempts": [], "paths_done": [],
        "native_attempts": [], "native_done": []}
    if state["plan_sha256"] != digest:
        raise ValueError("blocked: run state belongs to a different frozen plan")
    tracking = card / "qlib_recorder"
    kwargs = {"experiment_name": plan["experiment"], "uri": str(tracking)}
    if state.get("recorder_id"):
        kwargs.update(recorder_id=state["recorder_id"], resume=True)
    with R.start(**kwargs):
        recorder = R.get_recorder()
        state.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(state, state_path)
        if not state.get("sources_saved"):
            recorder.save_objects(artifact_path="inputs", plan=plan,
                preparation=prepared["receipt"], approval=approval,
                dataset=prepared["dataset"], auxiliary_dataset=prepared["auxiliary"])
            recorder.save_objects(artifact_path="sources",
                workflow=(root / "src/workflow/stage1_nvda_guidance_receiver.py").read_text(),
                adapter=(root / plan["model"]["adapter"]).read_text())
            state["sources_saved"] = True
            atomic_json(state, state_path)
        for family, mode in [("B_flags_only", "neutral"),
                             ("C_realization", "observed")]:
            for seed in plan["model"]["seeds"]:
                key = f"{family}_{seed}"
                original = prepared["originals"][seed]
                if key in state["fits_done"]:
                    model = recorder.load_object("models/" + key)
                else:
                    if (key in state["fit_attempts"]
                            or len(state["fit_attempts"]) >= plan["budget"]["model_fit_attempts_max"]):
                        raise ValueError("blocked: model fit attempt consumed")
                    state["fit_attempts"].append(key)
                    atomic_json(state, state_path)
                    model = _paired_models(original, root)[0 if mode == "neutral" else 1]
                    model.fit(prepared["dataset"], auxiliary_dataset=prepared["auxiliary"])
                    if (model.audit["training_rows"] != plan["model"]["primary_training_rows_expected"]
                            or model.audit["auxiliary_rows"] != plan["model"]["auxiliary_training_rows_expected"]
                            or model.extra_scaler[FEATURE]["observations"]
                            != plan["model"]["source_comparable_primary_rows_expected"]):
                        raise ValueError("blocked: fitted model training cohort differs")
                    if (model.primary_scaler != original.primary_scaler
                            or model.common_scaler != original.common_scaler):
                        raise ValueError("blocked: original Stage1 scaler changed")
                    recorder.save_objects(artifact_path="models", **{key: model})
                    state["fits_done"].append(key)
                    atomic_json(state, state_path)
                if key not in state["predictions_done"]:
                    if key in state["prediction_attempts"]:
                        raise ValueError("blocked: prediction attempt consumed")
                    state["prediction_attempts"].append(key)
                    atomic_json(state, state_path)
                    prediction = _prediction_dates(model.predict(prepared["dataset"]))
                    pd.testing.assert_index_equal(
                        prediction.index, prepared["reference_prediction_indexes"][seed],
                        check_names=False)
                    recorder.save_objects(artifact_path="predictions", **{key: prediction})
                    state["predictions_done"].append(key)
                    atomic_json(state, state_path)
                recorder.log_metrics(completed_fits=len(state["fits_done"]),
                                     completed_predictions=len(state["predictions_done"]))
    if len(state["fits_done"]) != 6 or len(state["predictions_done"]) != 6:
        raise ValueError("blocked: incomplete model barrier")
    artifact_root = tracking / state["experiment_id"] / state["recorder_id"] / "artifacts"
    sealed = {kind: {key: _sha(artifact_root / kind / key)
                     for key in state["fits_done"]}
              for kind in ["models", "predictions"]}
    seal = {"status": "six_models_and_predictions_sealed",
            "plan_sha256": digest, "recorder_id": state["recorder_id"],
            "experiment_id": state["experiment_id"], "completed_fits": 6,
            "source_preparation_sha256": _sha(card / "training_dataset_preflight.json"),
            "artifact_sha256": sealed,
            "target_sha256": {}, "new_native_accounts": 0}
    atomic_json(seal, card / "model_seal.json")
    return seal


def _subset_source(source: dict, names: list[str]) -> dict:
    if not set(names).issubset(source["artifacts"]):
        raise ValueError("blocked: parent source artifact inventory differs")
    return {**source, "artifacts": names,
            "artifact_sha256": {name: source["artifact_sha256"][name] for name in names}}


def prepare_account_sources(*, root: Path, card: Path) -> dict:
    """Bind immutable 0bp Stage1 price, controller and reference identities.

    This preflight reads reference accounts only for dates and native audit
    identities. It does not create a candidate path, fit a model or score one.
    """
    import qlib
    import sklearn

    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / "plan.json").read_text())
    runtime = plan["runtime"]
    if sklearn.__version__ != runtime["saved_action_classifier_scikit_learn"]:
        raise ValueError("needs_runtime: saved Stage1 action classifier scikit-learn version differs")
    if _sha(root / runtime["sim_runtime_requirements"]) != runtime["sim_runtime_requirements_sha256"]:
        raise ValueError("blocked: declared simulation runtime dependency identity differs")
    parent_path = root / plan["parent"]["plan"]
    if _sha(parent_path) != plan["parent"]["plan_sha256"]:
        raise ValueError("blocked: parent plan changed")
    parent = json.loads(parent_path.read_text())
    identities = plan["account_source_identities"]
    release = root / parent["active_release"]["path"]
    if (_sha(release) != parent["active_release"]["sha256"]
            or _sha(release) != identities["active_release_sha256"]):
        raise ValueError("blocked: frozen Stage1 runtime release changed")
    qlib.init(provider_uri=str(root / parent["provider_init"]), region="us", kernels=1)
    roles = {
        "action": ["inputs/plan", "inputs/market", "inputs/source_targets",
                   "controller/source_recovery_selection"],
        "controller": ["inputs/controller", "inputs/plan"],
        "raw": ["inputs/frame", "inputs/plan", "inputs/raw_points",
                "inputs/schedule", "inputs/repair_1_calendar_guard",
                "native/raw_SOXX_0", "native/raw_strong9097_0",
                "native/raw_candidate27baa_0"],
    }
    objects, audits = {}, {}
    for role, names in roles.items():
        source = _subset_source(parent["sources"][role], names)
        if (source["recorder_id"] != identities[role + "_recorder_id"]
                or source["inventory_sha256"] != identities[role + "_manifest_sha256"]):
            raise ValueError("blocked: exact " + role + " source identity differs")
        objects[role], audits[role] = _read_source(root, source)
    action, controller, raw = objects["action"], objects["controller"], objects["raw"]
    for name, digest in {
        "controller/source_recovery_selection": identities["action_selection_artifact_sha256"],
    }.items():
        if parent["sources"]["action"]["artifact_sha256"][name] != digest:
            raise ValueError("blocked: action source selection differs")
    for role, name, digest in [
        ("controller", "inputs/controller", identities["controller_artifact_sha256"]),
        ("raw", "inputs/raw_points", identities["raw_points_artifact_sha256"]),
        ("raw", "inputs/schedule", identities["raw_schedule_artifact_sha256"]),
        ("raw", "inputs/repair_1_calendar_guard", identities["raw_guard_artifact_sha256"]),
    ]:
        if parent["sources"][role]["artifact_sha256"][name] != digest:
            raise ValueError("blocked: " + role + " artifact identity differs")
    loaded = load_runtime_release(root, release)
    if (loaded["action_summary"]["source_stage1"]["selected_model"]
            != action["controller/source_recovery_selection"]):
        raise ValueError("blocked: frozen action/recovery selection differs")
    if controller["inputs/plan"]["tariff"] != parent["tariff"]:
        raise ValueError("blocked: controller Tiered fee schedule differs")
    if (raw["inputs/plan"]["tariff"] != parent["tariff"]
            or raw["inputs/plan"]["initial_cash"] != plan["economics_first_stage"]["initial_cash_usd"]):
        raise ValueError("blocked: raw-share cashbook and tariff differ")
    frame = raw["inputs/frame"].reset_index(drop=True)
    market = action["inputs/market"].reset_index(drop=True)
    dates = ["signal_date", "trade_date", "exit_date"]
    pd.testing.assert_frame_equal(frame[dates], market[dates])
    if len(frame) != plan["economics_first_stage"]["funded_sessions"]:
        raise ValueError("blocked: funded source calendar differs")
    serialized = "\n".join("|".join(str(v.date()) for v in row)
                           for row in frame[dates].itertuples(index=False, name=None)) + "\n"
    if (hashlib.sha256(serialized.encode()).hexdigest()
            != plan["calendar_binding"]["funded_date_tuple_sha256"]):
        raise ValueError("blocked: raw funded signal/trade/exit tuples differ")
    points = raw["inputs/raw_points"]
    if len(points) != 4 * len(frame) + 2 or not points.gt(0).all().all():
        raise ValueError("blocked: raw point grid differs")
    references = {}
    for role, name in [("SOXX", "native/raw_SOXX_0"),
                       ("strong9097", "native/raw_strong9097_0"),
                       ("additional27baa", "native/raw_candidate27baa_0")]:
        book = raw[name]
        pd.testing.assert_frame_equal(book["authority"][dates].reset_index(drop=True), frame[dates])
        if book["audit"]["status"] != "passed":
            raise ValueError("blocked: native source account failed: " + role)
        references[role] = book
    stage1_source = {
        "card": "research/stage1_domain_heads_20261018",
        "recorder_id": plan["parent"]["account_recorder_id"],
        "archive": json.loads((root / "research/stage1_domain_heads_20261018/summary.json").read_text())["nas_evidence"]["sha256"],
        "inventory_sha256": identities["Stage1_A_0_verification_manifest_sha256"],
        "artifacts": ["native/A_active_research_replay_0"],
        "artifact_sha256": {"native/A_active_research_replay_0": identities["Stage1_A_0_native_artifact_sha256"]},
    }
    old, stage1_audit = _read_source(root, stage1_source)
    book = old["native/A_active_research_replay_0"]
    pd.testing.assert_frame_equal(book["authority"][dates].reset_index(drop=True), frame[dates])
    if book["audit"]["status"] != "passed":
        raise ValueError("blocked: frozen Stage1 native A account failed")
    references["Stage1_A"] = book
    receipt = {
        "status": "verified_no_new_account_source_binding",
        "plan_sha256": _sha(card / "plan.json"),
        "source_recorder_ids": {role: parent["sources"][role]["recorder_id"] for role in roles},
        "reference_recorder_id": stage1_source["recorder_id"],
        "funded_sessions": len(frame), "raw_price_points": len(points),
        "funded_date_tuple_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "Tiered_tariff_and_release_matched": True,
        "action_selection_matched": True,
        "native_reference_accounts_verified": list(references),
        "new_model_fits": 0, "new_native_accounts": 0,
        "reference_account_returns_used": 0,
    }
    atomic_json(receipt, card / "account_source_preflight.json")
    return {"plan": plan, "parent": parent, "frame": frame, "market": market,
            "points": points, "schedule": raw["inputs/schedule"],
            "guard": raw["inputs/repair_1_calendar_guard"],
            "controller": loaded["models"],
            "selection": action["controller/source_recovery_selection"],
        "source_targets": action["inputs/source_targets"]["frozen_current"],
            "recovery_spec": action["inputs/plan"]["recovery_spec"],
            "action_spec": action["inputs/plan"]["action_spec"],
            "references": references, "source_audits": {**audits, "Stage1_A": stage1_audit},
            "receipt": receipt}


def verify_baseline_parity(*, root: Path, card: Path) -> dict:
    """Spend one frozen reference replay to check the executable Stage1 clock."""
    from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec

    root, card = Path(root).resolve(), Path(card).resolve()
    plan_path = card / "baseline_parity_plan.json"
    parity_plan = json.loads(plan_path.read_text())
    if parity_plan["status"] != "frozen_original_Stage1_0bp_runtime_parity_check":
        raise ValueError("blocked: baseline parity plan is not frozen")
    state_path = card / "baseline_parity_run_state.json"
    if state_path.exists():
        raise ValueError("blocked: single baseline parity replay attempt already consumed")
    prepared = prepare_account_sources(root=root, card=card)
    plan = prepared["plan"]
    if plan["economics_first_stage"]["slippage_bps"] != [0]:
        raise ValueError("blocked: parity is only the frozen 0bp account")
    frame = prepared["frame"].copy()
    columns = ["soxl_weight", "soxx_weight", "cash_weight",
               "tail_probability", "alpha_probability"]
    targets = prepared["source_targets"]
    pd.testing.assert_index_equal(targets.index, pd.DatetimeIndex(frame.signal_date), check_names=False)
    if list(targets) != columns:
        raise ValueError("blocked: source Stage1 target schema differs")
    for name in columns:
        frame[name] = targets[name].to_numpy(copy=True)
    atomic_json({"status": "attempt_started", "plan_sha256": _sha(plan_path),
                 "new_model_fits": 0, "new_candidate_accounts": 0}, state_path)
    replay = replay_stage1_raw(
        frame, point_prices=prepared["points"], schedule=prepared["schedule"],
        plan={"initial_cash": plan["economics_first_stage"]["initial_cash_usd"],
              "tariff": prepared["parent"]["tariff"]},
        controller=prepared["controller"],
        recovery_spec=IntradayRecoveryDirectAllocationSpec(root=root, **prepared["recovery_spec"]),
        action_spec=Stage1OpenActionValueSpec(root=root, **prepared["action_spec"]),
        selection=prepared["selection"], slippage_bps=0)
    old = prepared["references"]["Stage1_A"]
    left, right = replay["replay"], old["authority"]
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-10, atol=1e-7)
    pd.testing.assert_frame_equal(replay["orders"], old["orders"],
                                  check_exact=False, rtol=1e-10, atol=1e-7)
    pd.testing.assert_frame_equal(replay["points"], old["independent_points"],
                                  check_exact=False, rtol=1e-10, atol=1e-7)
    receipt = {
        "status": "original_Stage1_0bp_reference_replay_parity_passed",
        "plan_sha256": _sha(plan_path),
        "source_plan_sha256": _sha(card / "plan.json"),
        "funded_sessions": len(left), "orders": len(replay["orders"]),
        "raw_price_marks": len(replay["points"]),
        "max_absolute_return_difference": float((left.strategy_return-right.strategy_return).abs().max()),
        "max_absolute_nav_usd_difference": float((left.native_expected_nav-right.native_expected_nav).abs().max()),
        "max_absolute_cash_usd_difference": float((left.ending_cash-right.ending_cash).abs().max()),
        "all_action_and_risk_state_rows_matched": True,
        "scikit_learn_runtime": plan["runtime"]["saved_action_classifier_scikit_learn"],
        "new_model_fits": 0, "new_candidate_accounts": 0,
        "independent_market_evidence": False,
    }
    atomic_json(receipt, card / "baseline_parity_result.json")
    atomic_json({"status": "completed", "plan_sha256": _sha(plan_path),
                 "result": "baseline_parity_result.json"}, state_path)
    return receipt


def _require_execution_approval(root: Path, card: Path, plan: dict) -> dict:
    """No model fit or B/C account is allowed by a provisional review."""
    import sklearn

    if sklearn.__version__ != plan["runtime"]["saved_action_classifier_scikit_learn"]:
        raise ValueError("needs_runtime: approved Stage1 simulation requires saved scikit-learn version")
    review_path = card / "design_review_final.json"
    approval_path = card / "execution_approval.json"
    if not review_path.is_file() or not approval_path.is_file():
        raise ValueError("blocked: independent final design review and execution approval missing")
    review = json.loads(review_path.read_text())
    approval = json.loads(approval_path.read_text())
    digest = _sha(card / "plan.json")
    if (review.get("verdict") != "APPROVE_DESIGN"
            or review.get("plan_sha256") != digest
            or approval.get("approved") is not True
            or approval.get("plan_sha256") != digest):
        raise ValueError("blocked: approval does not match the frozen Stage1 plan")
    paths = approval.get("code_sha256", {})
    required = ["src/workflow/stage1_nvda_guidance_receiver.py",
                "src/model/stage1_nvda_guidance.py",
                "src/backtest/stage1_domain_heads_raw.py",
                "src/backtest/native_raw_corporate.py"]
    if not all(name in paths and _sha(root / name) == paths[name] for name in required):
        raise ValueError("blocked: approved executable code identity differs")
    return {"review_sha256": _sha(review_path),
            "execution_approval_sha256": _sha(approval_path),
            "code_sha256": paths}


def _check_prefit_receipts(card: Path, plan: dict) -> None:
    """Bind source/account/A parity evidence before opening a writable Recorder."""
    digest = _sha(card / "plan.json")
    training = json.loads((card / "training_dataset_preflight.json").read_text())
    account = json.loads((card / "account_source_preflight.json").read_text())
    parity = json.loads((card / "baseline_parity_result.json").read_text())
    change = json.loads((card / "plan_change_equivalence.json").read_text())
    if (training.get("status") != "verified_no_fit_training_dataset"
            or account.get("status") != "verified_no_new_account_source_binding"
            or training.get("plan_sha256") != digest
            or account.get("plan_sha256") != digest
            or account.get("funded_date_tuple_sha256")
            != plan["calendar_binding"]["funded_date_tuple_sha256"]
            or parity.get("status") != "original_Stage1_0bp_reference_replay_parity_passed"
            or not parity.get("all_action_and_risk_state_rows_matched")
            or parity.get("max_absolute_nav_usd_difference") != 0
            or parity.get("source_plan_sha256") != change.get("earlier_plan_sha256")
            or change.get("current_plan_sha256") != digest
            or not change.get("unchanged_parent_source_account_model_cost_gate_and_calendar")):
        raise ValueError("blocked: plan-bound prefit source and A parity receipts differ")


def _funded_date_digest(frame: pd.DataFrame) -> str:
    dates = frame[["signal_date", "trade_date", "exit_date"]]
    serialized = "\n".join("|".join(str(v.date()) for v in row)
                           for row in dates.itertuples(index=False, name=None)) + "\n"
    return hashlib.sha256(serialized.encode()).hexdigest()


def seal_funded_targets(*, root: Path, card: Path) -> dict:
    """After six sealed predictions, bind both smoothed funded target streams."""
    from qlib.workflow import R
    from src.workflow.minute_pit_allocator import _ensemble
    from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights

    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / "plan.json").read_text())
    _require_execution_approval(root, card, plan)
    seal_path = card / "model_seal.json"
    if not seal_path.is_file():
        raise ValueError("blocked: model seal is missing")
    seal = json.loads(seal_path.read_text())
    if (seal["status"] != "six_models_and_predictions_sealed"
            or seal["plan_sha256"] != _sha(card / "plan.json")
            or seal["completed_fits"] != 6 or seal.get("target_sha256")):
        raise ValueError("blocked: model seal is incomplete or targets already attempted")
    state_path = card / "run_state.json"
    state = json.loads(state_path.read_text())
    if state.get("plan_sha256") != seal["plan_sha256"] or state.get("targets_attempted"):
        raise ValueError("blocked: funded target attempt already consumed")
    tracking = card / "qlib_recorder"
    artifacts = tracking / seal["experiment_id"] / seal["recorder_id"] / "artifacts"
    prepared = prepare_account_sources(root=root, card=card)
    if prepared["receipt"]["plan_sha256"] != seal["plan_sha256"]:
        raise ValueError("blocked: account source binding differs after model seal")
    predictions = {}
    with R.uri_context(str(tracking)):
        recorder = R.get_recorder(recorder_id=seal["recorder_id"],
                                  experiment_id=seal["experiment_id"])
        for family in ["B_flags_only", "C_realization"]:
            for seed in plan["model"]["seeds"]:
                key = family + "_" + str(seed)
                for kind in ("models", "predictions"):
                    if _sha(artifacts / kind / key) != seal["artifact_sha256"][kind][key]:
                        raise ValueError("blocked: sealed fitted object differs")
                predictions[(family, seed)] = recorder.load_object("predictions/" + key)
        base = recorder.load_object("models/B_flags_only_" + str(plan["model"]["seeds"][0]))
    dates = pd.DatetimeIndex(prepared["market"].signal_date)
    returns = prepared["market"].set_index("signal_date")[["soxl_return", "soxx_return"]]
    targets = {}
    for family in ["B_flags_only", "C_realization"]:
        values = _ensemble(predictions, family, plan["model"]["seeds"])
        if len(values) != 946 or not dates.isin(values.index).all():
            raise ValueError("blocked: candidate prediction calendar differs")
        targets[family] = _smooth_weights(values.reindex(dates), spec=base.spec,
                                           candidate=base.candidate,
                                           asset_returns=returns)
        pd.testing.assert_index_equal(targets[family].index, dates, check_names=False)
    state["targets_attempted"] = True
    atomic_json(state, state_path)
    with R.start(experiment_name=plan["experiment"], uri=str(tracking),
                 recorder_id=seal["recorder_id"], resume=True):
        recorder = R.get_recorder()
        for family in ["B_flags_only", "C_realization"]:
            recorder.save_objects(artifact_path="targets", **{family: targets[family]})
    seal["target_sha256"] = {
        family: hashlib.sha256(pd.util.hash_pandas_object(target, index=True).values.tobytes()).hexdigest()
        for family, target in targets.items()}
    seal["target_artifact_sha256"] = {
        family: _sha(artifacts / "targets" / family) for family in targets}
    seal["status"] = "models_predictions_and_targets_sealed"
    atomic_json(seal, seal_path)
    state["targets_done"] = ["B_flags_only", "C_realization"]
    atomic_json(state, state_path)
    return seal


def execute_0bp_accounts(*, root: Path, card: Path,
                         sealed_targets: dict[str, pd.DataFrame],
                         recorder,
                         prepared: dict | None = None) -> dict:
    """Run only the two frozen B/C accounts after all six fits are sealed."""
    from src.backtest.native_raw_corporate import native_raw_validate
    from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
    from src.workflow.stage1_domain_heads import _provider

    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / "plan.json").read_text())
    _require_execution_approval(root, card, plan)
    seal_path = card / "model_seal.json"
    if not seal_path.is_file():
        raise ValueError("blocked: all six fitted models/predictions must be sealed first")
    seal = json.loads(seal_path.read_text())
    if (seal.get("status") != "models_predictions_and_targets_sealed"
            or seal.get("plan_sha256") != _sha(card / "plan.json")
            or seal.get("completed_fits") != 6
            or seal.get("recorder_id") != recorder.id):
        raise ValueError("blocked: model barrier is incomplete")
    expected_arms = {"B_flags_only", "C_realization"}
    if set(sealed_targets) != expected_arms:
        raise ValueError("blocked: both frozen B/C targets are required")
    if not isinstance(seal.get("target_sha256"), dict) or set(seal["target_sha256"]) != expected_arms:
        raise ValueError("blocked: target identity manifest differs")
    tracking = card / "qlib_recorder"
    artifacts = tracking / seal["experiment_id"] / seal["recorder_id"] / "artifacts"
    prepared = prepare_account_sources(root=root, card=card) if prepared is None else prepared
    if prepared["receipt"]["plan_sha256"] != _sha(card / "plan.json"):
        raise ValueError("blocked: prepared account source plan differs")
    frame = prepared["frame"]
    signals = pd.DatetimeIndex(frame.signal_date)
    columns = ["soxl_weight", "soxx_weight", "cash_weight",
               "tail_probability", "alpha_probability"]
    for name, target in sealed_targets.items():
        pd.testing.assert_index_equal(target.index, signals, check_names=False)
        if list(target.columns) != columns or not np.isfinite(target.to_numpy(float)).all():
            raise ValueError("blocked: sealed " + name + " target schema or finite values differ")
        if (target[columns[:3]].lt(0).any().any()
                or not np.allclose(target[columns[:3]].sum(axis=1), 1., atol=1e-10)):
            raise ValueError("blocked: sealed " + name + " target leaves funded simplex")
        if hashlib.sha256(pd.util.hash_pandas_object(target, index=True).values.tobytes()).hexdigest() != seal["target_sha256"][name]:
            raise ValueError("blocked: sealed " + name + " target bytes differ")
        if _sha(artifacts / "targets" / name) != seal["target_artifact_sha256"][name]:
            raise ValueError("blocked: sealed " + name + " Recorder artifact differs")
        pd.testing.assert_frame_equal(recorder.load_object("targets/" + name), target)
    provider_inputs = {"points": prepared["points"], "guard": prepared["guard"],
                       "plan": {"provider_output": plan["economics_first_stage"]["provider_output"]}}
    provider, provider_manifest = _provider(root, provider_inputs)
    recorder.save_objects(artifact_path="inputs", native_provider_manifest=provider_manifest)
    recovery = IntradayRecoveryDirectAllocationSpec(root=root, **prepared["recovery_spec"])
    action = Stage1OpenActionValueSpec(root=root, **prepared["action_spec"])
    execution = {"initial_cash": plan["economics_first_stage"]["initial_cash_usd"],
                 "tariff": prepared["parent"]["tariff"]}
    state_path = card / "run_state.json"
    if not state_path.is_file():
        raise ValueError("blocked: durable fit/account run state is missing")
    paths, books = {}, {}
    for name in ("B_flags_only", "C_realization"):
        state = json.loads(state_path.read_text())
        if state.get("plan_sha256") != _sha(card / "plan.json"):
            raise ValueError("blocked: run-state plan differs")
        key = name + "_0"
        if key in state.get("paths_done", []):
            path = recorder.load_object("paths/" + key)
        else:
            if key in state.get("path_attempts", []) or len(state.get("path_attempts", [])) >= 2:
                raise ValueError("blocked: candidate path attempt already consumed")
            state.setdefault("path_attempts", []).append(key)
            atomic_json(state, state_path)
            candidate_frame = frame.copy(deep=True)
            for column in columns:
                candidate_frame[column] = sealed_targets[name][column].to_numpy(copy=True)
            path = replay_stage1_raw(
                candidate_frame, point_prices=prepared["points"], schedule=prepared["schedule"],
                plan=execution, controller=prepared["controller"], recovery_spec=recovery,
                action_spec=action, selection=prepared["selection"], slippage_bps=0)
            recorder.save_objects(artifact_path="paths", **{key: path})
            state.setdefault("paths_done", []).append(key)
            atomic_json(state, state_path)
        if _funded_date_digest(path["replay"]) != plan["calendar_binding"]["funded_date_tuple_sha256"]:
            raise ValueError("blocked: candidate funded dates differ")
        if len(path["points"]) != 3778:
            raise ValueError("blocked: candidate point calendar differs")
        if key in state.get("native_done", []):
            book = recorder.load_object("native/" + key)
        else:
            if key in state.get("native_attempts", []) or len(state.get("native_attempts", [])) >= 2:
                raise ValueError("blocked: native account attempt already consumed")
            state.setdefault("native_attempts", []).append(key)
            atomic_json(state, state_path)
            book = native_raw_validate(path, provider_uri=provider,
                point_prices=prepared["points"], schedule=prepared["schedule"],
                tariff=prepared["parent"]["tariff"], cost=0,
                initial_cash=plan["economics_first_stage"]["initial_cash_usd"])
            recorder.save_objects(artifact_path="native", **{key: book})
            state.setdefault("native_done", []).append(key)
            atomic_json(state, state_path)
        if book["audit"]["status"] != "passed":
            raise ValueError("blocked: candidate native validation failed")
        pd.testing.assert_frame_equal(
            book["authority"][["signal_date", "trade_date", "exit_date"]].reset_index(drop=True),
            frame[["signal_date", "trade_date", "exit_date"]].reset_index(drop=True))
        paths[name], books[name] = path, book
    return {"paths": paths, "books": books, "provider": provider,
            "provider_manifest": provider_manifest,
            "reference_books": prepared["references"]}


def analyze_0bp_accounts(*, plan: dict, completed: dict) -> dict:
    """Apply the frozen SOXX gates to both independently funded paths."""
    from src.analysis.integrated_allocation import (
        paired_metrics, stationary_bootstrap_excess,
    )

    candidates = completed["books"]
    references = completed["reference_books"]
    soxx = references["SOXX"]["authority"]
    arm_names = ("B_flags_only", "C_realization")
    if set(candidates) != set(arm_names):
        raise ValueError("blocked: both native arms are required")
    alpha = .025 / 17
    metric_rows, annual_rows, quarter_rows, rollup_rows = [], [], [], []
    all_rolling = {}
    bounds = []
    for name in arm_names:
        account = candidates[name]["authority"]
        path = completed["paths"][name]["replay"]
        pd.testing.assert_frame_equal(account, path, check_exact=False, rtol=1e-10, atol=1e-7)
        result = paired_metrics(account, soxx)
        sampled_dd = -100 * float(path.monitored_drawdown.max())
        metric_rows.append({"family": name, "slippage_bps": 0,
                            "sampled_max_drawdown_pct": sampled_dd,
                            **result})
        for period_kind in ("year", "quarter"):
            labels = pd.to_datetime(account.trade_date).dt.to_period(
                "Y" if period_kind == "year" else "Q").astype(str)
            for period in labels.unique():
                take = labels.eq(period)
                item = {"family": name, "slippage_bps": 0,
                        "kind": period_kind, "period": period,
                        "complete_period": (period not in (labels.iloc[0], labels.iloc[-1])),
                        **paired_metrics(account.loc[take], soxx.loc[take])}
                (annual_rows if period_kind == "year" else quarter_rows).append(item)
        daily = np.log1p(account.strategy_return.to_numpy(float)) - np.log1p(soxx.strategy_return.to_numpy(float))
        rolling_log = pd.Series(daily).rolling(126).sum().iloc[125:].reset_index(drop=True)
        windows = pd.DataFrame({
            "first_trade": account.trade_date.iloc[:len(rolling_log)].to_numpy(),
            "last_exit": account.exit_date.iloc[125:].to_numpy(),
            "relative_log": rolling_log,
            "relative_return_pct": 100 * np.expm1(rolling_log),
        })
        if len(windows) != 819 or not np.isfinite(windows.relative_log).all():
            raise ValueError("blocked: complete 126-session windows differ")
        all_rolling[name] = windows
        rollup_rows.append({"family": name, "slippage_bps": 0,
                            "window_sessions": 126, "windows": len(windows),
                            "positive_fraction": float(windows.relative_log.gt(0).mean()),
                            "minimum_relative_return_pct": float(windows.relative_return_pct.min()),
                            "median_relative_return_pct": float(windows.relative_return_pct.median())})
        bounds.append({"family": name, "comparator": "SOXX", "slippage_bps": 0,
                       **stationary_bootstrap_excess(account, soxx, mean_block=63,
                           resamples=10000, alpha=alpha, seed=20261014)})
    b, c = (candidates[name]["authority"] for name in arm_names)
    bounds.append({"family": "C_realization", "comparator": "B_flags_only", "slippage_bps": 0,
                   **stationary_bootstrap_excess(c, b, mean_block=63,
                       resamples=10000, alpha=alpha, seed=20261014)})
    metric = pd.DataFrame(metric_rows)
    annual = pd.DataFrame(annual_rows)
    quarter = pd.DataFrame(quarter_rows)
    rollup = pd.DataFrame(rollup_rows)
    uncertainty = pd.DataFrame(bounds)
    if set(annual.loc[annual.complete_period, "period"]) != {"2024", "2025"}:
        raise ValueError("blocked: complete Stage1 annual coverage differs")
    soxx_bounds = uncertainty.loc[uncertainty.comparator.eq("SOXX")].set_index("family")
    gates = {}
    for name in arm_names:
        row = metric.set_index("family").loc[name]
        years = annual.loc[annual.family.eq(name) & annual.complete_period]
        rolling = rollup.set_index("family").loc[name]
        gates[name] = {
            "net_soxx_wealth": bool(row.relative_wealth > 1 + 1e-8),
            "available_drawdown_30pct": bool(row.max_drawdown_pct >= -30
                and row.sampled_max_drawdown_pct >= -30),
            "cvar_no_worse_soxx": bool(row.cvar5_pct >= row.benchmark_cvar5_pct),
            "complete_annual_70pct": bool(len(years) >= 2
                and years.relative_wealth.gt(1 + 1e-8).mean() >= .7),
            "rolling_126_70pct": bool(rolling.positive_fraction >= .7),
            "SOXX_bootstrap_lower_positive": bool(soxx_bounds.loc[name, "daily_lower_bound"] > 0),
            "full_minute_risk_qualified": False,
            "independent_forward_qualified": False,
        }
    matching = paired_metrics(c, b)
    matching_lower = float(uncertainty.loc[
        uncertainty.comparator.eq("B_flags_only"), "daily_lower_bound"].iloc[0])
    numeric_increment = matching["relative_wealth"] > 1 + 1e-8 and matching_lower > 0
    coarse = {name: all(value for key, value in gate.items()
                        if key not in {"full_minute_risk_qualified", "independent_forward_qualified"})
              for name, gate in gates.items()}
    if coarse["B_flags_only"]:
        selected = "C_realization" if coarse["C_realization"] and numeric_increment else "B_flags_only"
    elif coarse["C_realization"]:
        selected = "C_realization"
    else:
        selected = None
    return {"metrics": metric, "annual_metrics": annual,
            "quarterly_metrics": quarter, "rolling_summary": rollup,
            "rolling": all_rolling, "uncertainty": uncertainty,
            "gates": gates, "module_increment": {"C_over_B_relative_wealth": matching["relative_wealth"],
                "C_minus_B_daily_lower_bound": matching_lower,
                "numeric_increment_supported": bool(numeric_increment)},
            "historical_0bp_candidate": selected,
            "coarse_historical_0bp_gates": coarse}


def run(*, root: Path, card: Path) -> dict:
    """Execute the single frozen B/C research version after independent GO."""
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder

    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / "plan.json").read_text())
    _require_execution_approval(root, card, plan)
    state_path = card / "run_state.json"
    if state_path.is_file() and json.loads(state_path.read_text()).get("analysis_done"):
        return resume_nas_archive(root=root, card=card)
    if state_path.is_file() and json.loads(state_path.read_text()).get("analysis_attempted"):
        raise ValueError("blocked: fixed bootstrap analysis attempt consumed")
    old_summary = json.loads((card / "summary.json").read_text())
    if old_summary.get("new_fits", 0) or old_summary.get("new_native_accounts", 0):
        raise ValueError("blocked: this fixed experiment was already evaluated")
    if not (card / "model_seal.json").exists():
        fit_and_seal_models(root=root, card=card)
    seal = json.loads((card / "model_seal.json").read_text())
    if seal["status"] == "six_models_and_predictions_sealed":
        seal = seal_funded_targets(root=root, card=card)
    if seal["status"] != "models_predictions_and_targets_sealed":
        raise ValueError("blocked: candidate target barrier incomplete")
    prepared = prepare_account_sources(root=root, card=card)
    tracking = card / "qlib_recorder"
    with R.uri_context(str(tracking)):
        rec = R.get_recorder(recorder_id=seal["recorder_id"],
                             experiment_id=seal["experiment_id"])
        targets = {name: rec.load_object("targets/" + name)
                   for name in ("B_flags_only", "C_realization")}
    rec.start_run()
    try:
        completed = execute_0bp_accounts(root=root, card=card,
                                         sealed_targets=targets, recorder=rec,
                                         prepared=prepared)
        state = json.loads(state_path.read_text())
        state["analysis_attempted"] = True
        atomic_json(state, state_path)
        result = analyze_0bp_accounts(plan=plan, completed=completed)
        for name in ["metrics", "annual_metrics", "quarterly_metrics",
                     "rolling_summary", "uncertainty"]:
            value = result[name]
            if len(value) > 100:
                raise ValueError("blocked: slim sidecar unexpectedly large")
            value.to_csv(card / (name + ".csv"), index=False)
            rec.save_objects(artifact_path="analysis", **{name: value})
        rec.save_objects(artifact_path="analysis/rolling", **result["rolling"])
        rec.save_objects(artifact_path="analysis", gates=result["gates"],
                         module_increment=result["module_increment"])
        decision = ("historical_0bp_pass_needs_other_cost_risk_and_forward"
                    if result["historical_0bp_candidate"] else
                    "not_promoted_fixed_Stage1_guidance_receiver_0bp")
        summary = {
            "experiment": plan["experiment"], "plan_sha256": _sha(card / "plan.json"),
            "decision": decision, "objective_achieved": False,
            "historical_0bp_candidate": result["historical_0bp_candidate"],
            "gates": result["gates"],
            "module_increment": result["module_increment"],
            "new_fits": 6, "new_native_accounts": 2,
            "reused_market_history": True, "independent_forward_qualified_days": 0,
            "recorder_id": rec.id, "experiment_id": str(rec.experiment_id),
            "recorder_uri": str(tracking),
            "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id),
            "provider_uri": str(completed["provider"]), "region": "us",
            "artifact_policy": {"policy": "slim_card_recorder_first",
                "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*",
                    "predictions/*", "targets/*", "paths/*", "native/*",
                    "analysis/rolling/*"]},
            "limitations": ["historical first-public source timestamp/update coverage unverified",
                "2023-2026 history already used in earlier selection",
                "available marks do not qualify every full-minute drawdown",
                "2/5bp own-state accounts and independent forward evidence absent"],
        }
        rec.save_objects(artifact_path="analysis", summary=summary)
        rec.log_metrics(completed_fits=6, completed_native_accounts=2,
                        historical_0bp_pass=bool(result["historical_0bp_candidate"]))
        rec.end_run(status=Recorder.STATUS_FI)
    except BaseException:
        rec.end_run(status=Recorder.STATUS_FA)
        raise
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts"
    atomic_json({"recorder_id": rec.id,
                 "entries": [{"path": str(path.relative_to(card)),
                              "sha256": _sha(path), "bytes": path.stat().st_size}
                             for path in sorted(artifact_root.rglob("*")) if path.is_file()]},
                card / "verification_manifest.json")
    source_binding = json.loads((root / Path(plan["parent"]["summary"]).parent
                                 / "nas_binding.json").read_text())
    source_binding.pop("evidence", None)
    atomic_json(source_binding, card / "nas_binding.json")
    atomic_json(summary, card / "summary.json")
    rows = result["metrics"].set_index("family")
    rolling = result["rolling_summary"].set_index("family")
    lines = ["# 原 Stage1 的 NVDA 指引实现值接收层：固定0bp结果", "",
             "完成匹配标记B和数值C各三seed，共六次冻结拟合、两本独立0bp原价Qlib账户；不重训现役动作/恢复模型。0bp仍扣IBKR Tiered费用。", "",
             "|账户|净收益%|相对SOXX财富|可用点回撤%|126日正超额%|",
             "|---|---:|---:|---:|---:|"]
    for family in ("B_flags_only", "C_realization"):
        item = rows.loc[family]
        lines.append(f"|{family}|{item.return_pct:.3f}|{item.relative_wealth:.6f}|"
                     f"{item.sampled_max_drawdown_pct:.3f}|{100*rolling.loc[family,'positive_fraction']:.2f}|")
    lines += ["", "决定：" + decision + "。",
              "完整2024/2025年度、全部季度及819个重叠126日窗口完整保留；匹配B/C及SOXX的预登记区块下界见uncertainty.csv。",
              "历史曾选择过这条来源，故本轮只属开发反馈；原始首次公开时点/更新完整性、全分钟风险及独立前向仍不合格。无生产或真实券商变更。",
              "下一步：" + ("另行冻结2/5bp独立账户和前向来源/经济观察。" if result["historical_0bp_candidate"]
                           else "关闭这个固定接收层版本，不调来源延迟、符号、窗口、seed、损失、风险阈值或成本挽救；继续另一个实质机制。"),
              "", "Qlib Recorder：`" + rec.id + "`；NAS归档、来源与恢复证据见summary.json、verification_manifest.json和nas_binding.json。"]
    (card / "conclusion.md").write_text("\n".join(lines) + "\n")
    state = json.loads(state_path.read_text())
    state["analysis_done"] = True
    atomic_json(state, state_path)
    return resume_nas_archive(root=root, card=card)


def resume_nas_archive(*, root: Path, card: Path) -> dict:
    """Archive completed evidence without another fit, account, or bootstrap."""
    from src.data.nas_market_data import archive_nas_recorder

    root, card = Path(root).resolve(), Path(card).resolve()
    state_path = card / "run_state.json"
    state = json.loads(state_path.read_text())
    summary = json.loads((card / "summary.json").read_text())
    if (not state.get("analysis_done") or summary.get("new_fits") != 6
            or summary.get("new_native_accounts") != 2
            or summary.get("plan_sha256") != _sha(card / "plan.json")
            or state.get("plan_sha256") != summary["plan_sha256"]
            or state.get("recorder_id") != summary.get("recorder_id")):
        raise ValueError("blocked: completed analysis checkpoint differs")
    manifest = json.loads((card / "verification_manifest.json").read_text())
    if manifest["recorder_id"] != summary["recorder_id"]:
        raise ValueError("blocked: completed Recorder manifest differs")
    for entry in manifest["entries"]:
        path = card / entry["path"]
        if _sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("blocked: completed Recorder bytes differ")
    if "nas_evidence" not in summary:
        summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json",
            workspace_root=root, card=card, recorder_id=summary["recorder_id"],
            experiment_id=summary["experiment_id"])
        atomic_json(summary, card / "summary.json")
    state["completed"] = True
    atomic_json(state, state_path)
    return summary
