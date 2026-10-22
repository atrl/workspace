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
            "receipt": receipt}


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
