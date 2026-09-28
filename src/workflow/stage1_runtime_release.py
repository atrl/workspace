"""Load an explicitly activated, immutable SIM release without refitting models."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.pit_allocation_retrain import validate_frozen_bundle
from src.workflow.stage1_open_action_value import (
    Stage1OpenActionValueSpec, prepare_on_policy_open, prepare_on_policy_recovery,
)
from src.workflow.stage1_futu_sim import RuntimeState, _with_intraday, infer_frozen_allocation


CONTROLLER_FIELDS = ("drawdown", "current_beta_cap", "raw_beta", "risk_state", "risk_clock", "history")
ON_POLICY_STATE = "own_on_policy_equity_peak_history_and_risk_clock"
PAPER_TRIAL_OVERRIDABLE_GATES = {
    "development_return_above_baseline", "stress_25bps_return_above_baseline",
    "max_drawdown_not_worse", "worst_day_not_worse", "cvar5_not_worse", "worst_5d_not_worse",
    "open_trade_days_reduced_8pct", "turnover_not_increased", "annual_wealth_floor_pass",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contained_path(parent: Path, value: str) -> Path:
    path = (parent / value).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("blocked: release artifact path escapes its declared root")
    return path


def _verify_hash(path: Path, expected: str) -> None:
    if not expected or not path.is_file() or _hash(path) != expected:
        raise ValueError(f"blocked: release artifact hash mismatch: {path.name}")


def _validate_training_cutoff(value: Any, cutoff: str) -> None:
    try:
        end = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("blocked: frozen model has an invalid training cutoff") from error
    if pd.isna(end) or end.tzinfo is not None or end > pd.Timestamp(cutoff):
        raise ValueError("blocked: model training exceeds or omits the frozen cutoff")


def _paper_trial_receipt(
    root: Path, release: dict[str, Any], summary: dict[str, Any], bundle: dict[str, Any],
) -> dict[str, Any]:
    """Authorize only the exact disclosed failed development result in SIM."""
    reference = release.get("paper_trial_authorization")
    if not isinstance(reference, dict):
        raise ValueError("blocked: failed development gates require a specific user paper-trial receipt")
    path = _contained_path(root, str(reference.get("receipt_path", "")))
    _verify_hash(path, str(reference.get("receipt_sha256", "")))
    receipt = _read(path)
    expected_fields = {
        "version": 1,
        "kind": "explicit_user_unpromoted_sim_trial",
        "execution_environment": "SIMULATE",
        "release_id": release["release_id"],
        "action_summary_sha256": release["action"]["summary_sha256"],
        "action_recorder_id": summary["recorder_id"],
        "allocation_recorder_id": bundle["recorder_id"],
        "allocation_manifest_sha256": release["allocation"]["manifest_sha256"],
        "decision": "run_unpromoted_candidate_in_simulate_only",
    }
    if any(receipt.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("blocked: paper-trial receipt does not authorize this exact SIM release")
    if not isinstance(receipt.get("user_instruction"), str) or not receipt["user_instruction"].strip():
        raise ValueError("blocked: paper-trial receipt is missing the user's explicit instruction")
    try:
        authorized = pd.Timestamp(receipt["authorized_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("blocked: invalid paper-trial authorization timestamp") from error
    if pd.isna(authorized) or authorized.tzinfo is None or authorized > pd.Timestamp.now(tz="UTC"):
        raise ValueError("blocked: paper-trial authorization must be dated, timezone-aware, and not in the future")
    gates = summary.get("development_gates")
    if not isinstance(gates, dict) or not gates or any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("blocked: development failures must be explicitly identified before paper-trial authorization")
    failures = {name for name, passed in gates.items() if not passed}
    if not failures.issubset(PAPER_TRIAL_OVERRIDABLE_GATES):
        raise ValueError("blocked: paper-trial authorization cannot bypass scientific identity or timing gates")
    allowed = receipt.get("allowed_failed_development_gates")
    if (
        not failures or not isinstance(allowed, list) or any(not isinstance(name, str) for name in allowed)
        or len(set(allowed)) != len(allowed) or set(allowed) != failures
    ):
        raise ValueError("blocked: paper-trial receipt must name exactly the disclosed failed development gates")
    return {
        "receipt_path": str(path.relative_to(root)), "receipt_sha256": reference["receipt_sha256"],
        "authorized_at": authorized.isoformat(), "allowed_failed_development_gates": sorted(failures),
    }


def load_runtime_release(root: Path, release_path: Path) -> dict[str, Any]:
    release = _read(release_path)
    if release.get("version") != 1 or release.get("execution_environment") != "SIMULATE" or not release.get("release_id"):
        raise ValueError("blocked: invalid Stage1 SIM release contract")
    allocation = release["allocation"]
    allocation_card = _contained_path(root, allocation["card_path"])
    manifest_path = allocation_card / "frozen_bundle_manifest.json"
    _verify_hash(manifest_path, allocation.get("manifest_sha256", ""))
    bundle = validate_frozen_bundle(allocation_card)
    if bundle.get("recorder_id") != allocation.get("recorder_id"):
        raise ValueError("blocked: allocation release recorder identity mismatch")
    training_window = bundle.get("training_window")
    if not isinstance(training_window, list) or len(training_window) != 2:
        raise ValueError("blocked: allocation model training exceeds or omits the frozen cutoff")
    _validate_training_cutoff(training_window[1], "2022-12-30")
    allocation_summary = _read(allocation_card / "summary.json")
    if allocation_summary.get("recorder_id") != bundle["recorder_id"] or allocation_summary.get("experiment") != bundle["experiment"]:
        raise ValueError("blocked: allocation card and frozen manifest identity differ")
    selected_entries = [entry for entry in bundle["artifacts"] if entry["path"].endswith("/selected_model.json")]
    if len(selected_entries) != 1:
        raise ValueError("blocked: allocation bundle has no unique frozen selection")
    selected_path = _contained_path(allocation_card, selected_entries[0]["path"])
    selected = _read(selected_path)
    allocation_artifact_dir = selected_path.parent
    model_name = f"models/{selected['candidate']}.pkl"
    if not any(_contained_path(allocation_card, entry["path"]) == allocation_artifact_dir / model_name for entry in bundle["artifacts"]):
        raise ValueError("blocked: allocation model is not sealed by its frozen manifest")

    action = release["action"]
    action_card = _contained_path(root, action["card_path"])
    summary_path = action_card / "summary.json"
    _verify_hash(summary_path, action.get("summary_sha256", ""))
    summary = _read(summary_path)
    if summary.get("recorder_id") != action.get("recorder_id"):
        raise ValueError("blocked: action release recorder identity mismatch")
    if summary.get("selection_uses_outer_test") is not False:
        raise ValueError("blocked: action model used outer data for selection")
    paper_trial_authorization = None
    if summary.get("development_gate_pass") is not True:
        if summary.get("development_gate_pass") is not False:
            raise ValueError("blocked: action development gate result is missing")
        paper_trial_authorization = _paper_trial_receipt(root, release, summary, bundle)
    source = summary.get("source_stage1", {})
    if source.get("validation_controller_state") != ON_POLICY_STATE:
        raise ValueError("blocked: action model was not validated with its own controller state")
    if source.get("allocation_source_recorder_id") != bundle["recorder_id"] or source.get("allocation_source_experiment") != bundle["experiment"]:
        raise ValueError("blocked: action and allocation source identities differ")
    recorder_dir = _contained_path(action_card, str(summary["recorder_local_dir"]))
    if recorder_dir.name != action["recorder_id"]:
        raise ValueError("blocked: action artifact directory and recorder identity differ")
    models = {}
    for name in ("frozen_open_action_value_model", "frozen_recovery_model"):
        metadata = summary["runtime_models"][name]
        if metadata.get("available") is not True:
            raise ValueError("blocked: runtime action model is absent or trained beyond the frozen window")
        _validate_training_cutoff(metadata.get("training_signal_end"), "2022-12-31")
        path = _contained_path(recorder_dir, metadata["artifact"])
        _verify_hash(path, metadata.get("sha256", ""))
        # Only after the immutable summary and this exact artifact digest pass.
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("blocked: released action artifact has no prediction interface")
        models[name] = model
    return {
        "release": release, "allocation_bundle": bundle,
        "allocation_artifact_dir": allocation_artifact_dir,
        "action_summary": summary, "models": models,
        "paper_trial_authorization": paper_trial_authorization,
    }


def _observed_mark(value: str | pd.Timestamp | None, execution_date: str | None) -> pd.Timestamp:
    if value is None or execution_date is None:
        raise ValueError("blocked: executable release requires dated current execution prices")
    observed = pd.Timestamp(value)
    if observed.tzinfo is None:
        raise ValueError("blocked: execution price timestamp must include a timezone")
    observed = observed.tz_convert("America/New_York")
    if observed.date().isoformat() != execution_date or not 570 <= observed.hour * 60 + observed.minute < 960:
        raise ValueError("blocked: release execution price mark is stale or outside the regular session")
    return observed


def _controller_context(opened: dict[str, Any]) -> dict[str, Any]:
    return {
        "drawdown": float(opened["drawdown"]), "current_beta_cap": float(opened["current_beta_cap"]),
        "raw_beta": float(opened["raw_beta"]), "risk_state": int(opened["risk_state"]),
        "risk_clock": int(opened["risk_clock"]), "history": [float(value) for value in opened["history"]],
    }


def _weights(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < -1e-10) or not np.isclose(values.sum(), 1.0):
        raise ValueError("blocked: runtime release produced invalid target weights")
    return dict(zip(("SOXL", "SOXX", "cash"), values.tolist()))


def build_release_target(
    *, root: Path, release_path: Path, current_prices: dict[str, float] | None,
    include_current_intraday: bool, execution_date: str | None,
    live_controller_state: RuntimeState | None,
    current_prices_as_of: str | pd.Timestamp | None = None,
    open_decision_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = load_runtime_release(root, release_path)
    release, summary = loaded["release"], loaded["action_summary"]
    bundle = loaded["allocation_bundle"]
    frame, audit = infer_frozen_allocation(
        root=root, artifact_dir=loaded["allocation_artifact_dir"],
        source_identity={"source_experiment": bundle["experiment"], "source_recorder_id": bundle["recorder_id"],
                         "stage1_recorder_id": summary["recorder_id"], "release_id": release["release_id"]},
        current_prices=current_prices, execution_date=execution_date, current_prices_as_of=current_prices_as_of,
    )
    signal_date = frame.index.max().date().isoformat()
    base = {
        "signal_date": signal_date, "source_audit": audit, "release_id": release["release_id"],
        "controller_state_source": "paper_ledger_cycle", "real_broker_order": False,
        "paper_trial_only": True, "research_promotion_status": "not_promoted",
        "development_gate_pass": summary["development_gate_pass"],
        "paper_trial_authorization": loaded["paper_trial_authorization"],
    }
    if audit["smoothing"]["status"] != "ready":
        return base | {"status": "needs_execution_mark", "reason": "frozen_allocation_requires_current_execution_mark"}
    if live_controller_state is None:
        return base | {"status": "blocked", "reason": "live_paper_ledger_required"}
    state = live_controller_state
    if (
        not np.isfinite([state.equity, state.peak]).all() or min(state.equity, state.peak) <= 0
        or state.previous is None or not np.isfinite(state.history or []).all()
    ):
        raise ValueError("blocked: invalid paper ledger controller state")
    _weights(state.previous)
    if current_prices is None or set(current_prices) != {"SOXL", "SOXX"} or not all(
        np.isfinite(float(value)) and float(value) > 0 for value in current_prices.values()
    ):
        return base | {"status": "blocked", "reason": "current_execution_prices_required"}
    observed = _observed_mark(current_prices_as_of, execution_date)
    if signal_date >= str(execution_date):
        raise ValueError("blocked: release signal is not a prior settled session")
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=root, source_experiment=bundle["experiment"])
    selected = summary["selected_model"]
    action_spec = Stage1OpenActionValueSpec(
        root=root, experiment=summary["experiment"], logistic_c=float(selected["logistic_c"]),
        execute_probability_threshold=float(selected["execute_probability_threshold"]),
        maximum_skippable_turnover=float(selected["maximum_skippable_turnover"]),
        mandatory_derisk_beta_delta=float(selected["mandatory_derisk_beta_delta"]),
    )
    recovery_action = False
    recovery_probability = None
    if include_current_intraday:
        context = open_decision_context
        if not isinstance(context, dict) or any(context.get(key) != expected for key, expected in (
            ("release_id", release["release_id"]), ("signal_date", signal_date), ("execution_date", execution_date),
        )):
            return base | {"status": "blocked", "reason": "missing_matching_open_decision_context"}
        open_mark = _observed_mark(context.get("quote_as_of"), execution_date)
        if open_mark != observed:
            raise ValueError("blocked: recovery must reuse the observed open settlement mark")
        controller = _controller_context(context["controller"])
        if not np.isfinite([controller[key] for key in ("drawdown", "current_beta_cap", "raw_beta")]).all() or not np.isfinite(controller["history"]).all():
            raise ValueError("blocked: saved open controller contains invalid values")
        frame = frame.copy()
        frame.loc[frame.index[-1], "trade_date"] = pd.Timestamp(execution_date)
        frame = _with_intraday(frame, recovery_spec=recovery_spec)
        row = frame.iloc[-1].copy()
        if not np.isfinite(float(row.get("early_ret_soxx", np.nan))):
            return base | {"status": "blocked", "reason": "current_first_hour_data_required"}
        recovered = prepare_on_policy_recovery(
            row, open_state=controller, recovery_spec=recovery_spec,
            model=loaded["models"]["frozen_recovery_model"], selected_model=summary["source_stage1"]["selected_model"],
        )
        recovery_action = bool(recovered["recovery_action"])
        probability = recovered["recovery_probability"]
        recovery_probability = float(probability) if np.isfinite(probability) else None
        weights = recovered["target_weights"] if recovery_action else state.previous
        decision = "rebalance" if recovery_action else "hold"
        decision_context = context
    else:
        row = frame.iloc[-1].copy()
        opened = prepare_on_policy_open(
            row, pretrade_weights=state.previous, equity=state.equity, peak=state.peak,
            risk_state=state.risk_state, risk_clock=state.risk_clock, history=list(state.history or []),
            recovery_spec=recovery_spec, model=loaded["models"]["frozen_open_action_value_model"],
            action_spec=action_spec,
        )
        controller = _controller_context(opened)
        weights = opened["open_weights"]
        decision = "rebalance" if opened["execute_open_target"] else "hold"
        decision_context = {
            "controller": controller, "quote_as_of": observed.isoformat(), "release_id": release["release_id"],
            "signal_date": signal_date, "execution_date": execution_date,
        }
    weights = _weights(weights)
    return base | {
        "status": "ready", "slot": "recovery" if include_current_intraday else "open",
        "decision": decision, "weights": weights, "open_weights": weights,
        "recovery_weights": weights, "recovery_action": recovery_action,
        "recovery_probability": recovery_probability, "decision_context": decision_context,
        "portfolio_drawdown": controller["drawdown"], "current_beta_cap": controller["current_beta_cap"],
        "raw_effective_beta": controller["raw_beta"],
        "runtime_replay_parity_max_abs_error": None,
    }
