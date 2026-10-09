"""Audit the Stage1 lineage and forward-evidence eligibility, fail-closed.

Two separate questions are answered without conflating them:

1. **Legacy lineage integrity** -- whether the historical research chain kept its
   holdout clean and left auditable recorders.
2. **Active release readiness** -- whether the currently bound paper/SIM release
   has accumulated *eligible* paired forward evidence against the contract in
   ``production/stage1_regime_paper_20261001/forward_plan.json``.

Every input is fail-closed. A missing card, a missing field, an unknown
freshness value, an empty promotion set, an empty recorder directory, or a bare
integer count never grants a fresh walk-forward. ``completed_slots`` dates are
*raw observed* dates, not eligible forward sessions: without complete-session,
chronology, pairing and identity validation the eligible count is ``unknown``
(or ``0``) and the status is ``needs_data``.

This module is read-only. It does not resolve, start, or modify any recorder.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Legacy historical lineage, oldest research card first.
LINEAGE = (
    "intraday_recovery_direct_allocation_v1",
    "intraday_on_policy_staged_validation_v1",
    "stage1_open_action_value_v1",
    "stage1_confidence_veto_recovery_v1",
    "stage1_economic_action_20260929",
    "stage1_regime_deployment_20261001",
)
PRODUCTION_CARD = "stage1_futu_sim_v1"
RECORDER_DIR = "qlib_recorder"
# Outer windows that the project has already inspected and reused.
NON_FRESH_OUTER = {"reused_diagnostic_not_fresh_unseen_test"}
# Outer freshness is only asserted for values on this explicit whitelist. A null,
# "unknown", or any unrecognised string is treated as unknown, never as fresh.
FRESH_OUTER_WHITELIST = {"fresh_unseen_holdout", "fresh_unseen_test"}
STATUS_VALUES = ("ok", "missing_card", "missing_field", "unknown_freshness", "empty_recorder")
# Raw slot statuses that *might* correspond to a completed session.
RAW_COMPLETED_STATUSES = ("filled_confirmed", "hold_completed")


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _declared_recorders(summary: dict) -> dict:
    return {
        "recorder_id": summary.get("recorder_id"),
        "source_recorder_id": summary.get("source_recorder_id"),
    }


def _recorder_state(card_dir: Path, recorder_id: str) -> str:
    """Classify a declared recorder: ``present``, ``empty_dir`` or ``absent``.

    An empty recorder directory is *not* presence; it must not grant eligibility.
    """
    root = card_dir / RECORDER_DIR
    if not root.is_dir():
        return "absent"
    if any(root.glob(f"*/{recorder_id}/*")):
        return "present"
    if any(root.glob(f"*/{recorder_id}")):
        return "empty_dir"
    return "absent"


def audit_card(root: Path, name: str, *, card_root: str) -> dict:
    """Audit one card, fail-closed on any missing or unknown input."""
    card_dir = root / card_root / name
    summary_path = card_dir / "summary.json"
    row = {"card": name, "card_root": card_root, "status": "missing_card", "decision": None,
           "selection_uses_outer_test": None, "outer_status": None, "outer_is_fresh": None,
           "recorder_id": None, "recorder_state": "absent", "recorder_present": False,
           "source_recorder_id": None, "recorder_present": False}
    if not summary_path.is_file():
        return row
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    declared = _declared_recorders(summary)
    recorder_id = declared["recorder_id"]
    # Missing decision/freshness fields are unknown, never assumed clean.
    if summary.get("decision") is None or "outer_status" not in summary:
        row.update(status="missing_field", decision=summary.get("decision"),
                   outer_status=summary.get("outer_status"), outer_is_fresh=None,
                   recorder_id=recorder_id, source_recorder_id=declared["source_recorder_id"])
        if recorder_id:
            row["recorder_state"] = _recorder_state(card_dir, recorder_id)
            row["recorder_present"] = row["recorder_state"] == "present"
        return row
    outer_status = summary.get("outer_status")
    if outer_status in FRESH_OUTER_WHITELIST:
        outer_is_fresh = True
    elif outer_status in NON_FRESH_OUTER:
        outer_is_fresh = False
    else:
        # Null, "unknown", or any unrecognised value is not a freshness claim.
        outer_is_fresh = None
    state = _recorder_state(card_dir, recorder_id) if recorder_id else "absent"
    status = "ok"
    if outer_is_fresh is None:
        status = "unknown_freshness"
    if recorder_id and state == "empty_dir":
        status = "empty_recorder"
    row.update({
        "card": name, "card_root": card_root, "status": status,
        "decision": summary.get("decision"),
        "selection_uses_outer_test": summary.get("selection_uses_outer_test"),
        "outer_status": outer_status, "outer_is_fresh": outer_is_fresh,
        "recorder_id": recorder_id, "recorder_state": state,
        "recorder_present": state == "present",
        "source_recorder_id": declared["source_recorder_id"],
    })
    return row


def audit_lineage(root: Path, *, lineage=LINEAGE, production: str = PRODUCTION_CARD) -> pd.DataFrame:
    rows = [audit_card(root, name, card_root=("production" if name == production else "research"))
            for name in (*lineage, production)]
    return pd.DataFrame(rows)


def lineage_findings(table: pd.DataFrame) -> dict:
    """Aggregate the legacy lineage, fail-closed on any unknown state."""
    _require(not table.empty, "lineage table is empty")
    decisions = table["decision"].astype("object").fillna("").astype(str)
    promotion_cards = table[decisions.str.startswith("promoted", na=False)]
    selection_used_outer = table["selection_uses_outer_test"].astype("boolean")
    any_selection_used_outer = bool(selection_used_outer.fillna(True).any())  # unknown -> treat as used
    outer_fresh = table["outer_is_fresh"].astype("boolean")
    return {
        "cards": int(len(table)),
        "ok_cards": int(table["status"].eq("ok").sum()),
        "missing_card_cards": table.loc[table["status"].eq("missing_card"), "card"].tolist(),
        "missing_field_cards": table.loc[table["status"].eq("missing_field"), "card"].tolist(),
        "unknown_freshness_cards": table.loc[table["status"].eq("unknown_freshness"), "card"].tolist(),
        "empty_recorder_cards": table.loc[table["status"].eq("empty_recorder"), "card"].tolist(),
        "promoted_cards": int(len(promotion_cards)),
        "empty_promotion_set": bool(promotion_cards.empty),
        "recorders_declared": int(table["recorder_id"].notna().sum()),
        "recorders_present": int(table["recorder_present"].sum()),
        "recorders_missing": int((table["recorder_id"].notna() & ~table["recorder_present"]).sum()),
        "missing_recorder_cards": table.loc[
            table["recorder_id"].notna() & ~table["recorder_present"], "card"].tolist(),
        "outer_not_fresh_cards": table.loc[outer_fresh.eq(False), "card"].tolist(),
        "outer_unknown_cards": table.loc[outer_fresh.isna(), "card"].tolist(),
        "any_selection_used_outer": any_selection_used_outer,
        "all_promotions_have_recorder": bool(
            not promotion_cards.empty and promotion_cards["recorder_present"].all()),
        "all_cards_ok": bool(table["status"].eq("ok").all()),
    }


def walk_forward_verdict(findings: dict, *, forward_evidence: dict) -> dict:
    """Report the walk-forward status. This is diagnostic and has NO success branch.

    There is no configuration of ``findings`` or ``forward_evidence`` that yields
    ``passed_fresh_walk_forward = True``. Promotion is only ever granted by the
    existing sealed validation contract, never by a caller-supplied count. This
    function always reports ``qualification = needs_data`` and ``passed = False``
    and lists every reason the evidence is not yet sufficient.
    """
    _require(isinstance(forward_evidence, dict), "forward evidence must be a diagnostic mapping")
    reasons = []
    if not findings.get("all_cards_ok", False):
        reasons.append("one or more lineage cards are missing, incomplete or unknown")
    if findings.get("empty_promotion_set", True):
        reasons.append("no promoted card exists to validate")
    if findings.get("any_selection_used_outer", True):
        reasons.append("at least one card selected using the outer window (or freshness unknown)")
    if not findings.get("all_promotions_have_recorder", False):
        reasons.append("a promoted card has no present, non-empty recorder")
    reasons.extend(forward_evidence.get("reasons", []) or [])
    reasons.append("formal promotion requires the existing sealed validation contract, not this diagnostic")
    return {
        "passed_fresh_walk_forward": False,
        "qualification": "needs_data",
        "eligible_forward_sessions": forward_evidence.get("eligible_sessions"),
        "observed_forward_sessions": forward_evidence.get("raw_observed_count"),
        "settled_forward_sessions": forward_evidence.get("settled_count"),
        "required_forward_sessions": forward_evidence.get("required_sessions"),
        "evidence_status": forward_evidence.get("status", "needs_data"),
        "failed_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Forward evidence: a diagnostic report with no caller-supplied success branch
# ---------------------------------------------------------------------------

def forward_evidence_report(*, root: Path, forward_plan: dict,
                            contamination: dict | None = None) -> dict:
    """Report observed and settled forward counts as a pure diagnostic.

    This computes counts from repository state only. Caller-supplied dictionaries
    are rejected so no caller-provided number can trigger a success branch. The
    status is always ``needs_data``: without a matured, complete, identity-paired,
    hash-verified validation contract the eligibility is genuinely unknown, and it
    is never inferred from raw slot dates.
    """
    _require(isinstance(forward_plan, dict), "forward plan must be a mapping")
    required = forward_plan.get("minimum_complete_paired_sessions")
    if not isinstance(required, int) or required <= 0:
        raise ValueError("blocked: forward plan has no integer minimum_complete_paired_sessions")
    if contamination is None or not isinstance(contamination, dict):
        raise ValueError("blocked: contamination declaration must be provided")
    raw_dates = _raw_observed_dates(root)
    settled = _settled_dates(root)
    reasons = []
    if contamination.get("windows"):
        reasons.append("a contaminated window is declared and must be excluded before any count")
    if not settled:
        reasons.append("no fully settled forward sessions found")
    reasons.append(
        "no matured identity-paired, hash-verified validation contract; raw and settled dates are not "
        "eligible forward sessions")
    return {
        "status": "needs_data",
        "qualification": "needs_data",
        "eligible_sessions": None,
        "raw_observed_dates": raw_dates,
        "raw_observed_count": len(raw_dates),
        "settled_dates": settled,
        "settled_count": len(settled),
        "required_sessions": required,
        "required_per_regime": forward_plan.get("minimum_sessions_per_regime"),
        "required_transitions": forward_plan.get("minimum_regime_transitions"),
        "contamination": contamination,
        "reasons": reasons,
    }


def _settled_dates(root: Path) -> list[str]:
    """Dates that carry a settled qlib mark in the active sleeve ledger."""
    ledger = root / "runtime/soxl/stage1_futu_sim/paper_ledger_v2.json"
    if not ledger.is_file():
        return []
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    marks = payload.get("settled_marks")
    if not isinstance(marks, list):
        return []
    dates = {mark.get("date") for mark in marks if isinstance(mark, dict)
             and isinstance(mark.get("date"), str) and mark.get("source") == "qlib.data.D.features"}
    return sorted(d for d in dates if d)


def _raw_observed_dates(root: Path) -> list[str]:
    state = root / "runtime/soxl/stage1_futu_sim/execution_state_v2.json"
    if not state.is_file():
        return []
    payload = json.loads(state.read_text(encoding="utf-8"))
    dates = {
        value.get("execution_session_date")
        for value in payload.get("completed_slots", {}).values()
        if isinstance(value, dict) and value.get("status") in RAW_COMPLETED_STATUSES
    }
    return sorted(d for d in dates if isinstance(d, str) and d)


def load_forward_plan(root: Path) -> dict:
    """Read the declared forward plan from the release manifest (not verified).

    This reads the plan text only. It does **not** verify the manifest seal, so
    the result is ``verified = False`` and must be treated as a declared plan, not
    an authorization. A seal check is deliberately left to the existing sealed
    loader rather than duplicated here.
    """
    path = root / "production/stage1_regime_paper_20261001/release_manifest.json"
    _require(path.is_file(), "sealed release manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    plan = manifest.get("forward_plan")
    _require(isinstance(plan, dict), "release manifest has no forward_plan")
    return plan


def active_release(root: Path) -> dict:
    """Read the bound active release identity (indices its allocation and action cards)."""
    path = root / "production/stage1_futu_sim_v1/runtime_release.json"
    _require(path.is_file(), "active runtime release is missing")
    release = json.loads(path.read_text(encoding="utf-8"))
    return {
        "release_id": release.get("release_id"),
        "execution_environment": release.get("execution_environment"),
        "research_promotion_status": release.get("research_promotion_status"),
        "allocation_recorder_id": (release.get("allocation") or {}).get("recorder_id"),
        "action_recorder_id": (release.get("action") or {}).get("recorder_id"),
        "allocation_card": (release.get("allocation") or {}).get("card_path"),
        "action_card": (release.get("action") or {}).get("card_path"),
    }


def active_release_development_risk(root: Path) -> dict:
    """Report the active release's development gates from the verified loader.

    The release identity is resolved and hash-verified through
    ``load_runtime_release`` so a different release cannot be silently read. The
    development-gate result is read from the verified ``action_summary`` rather
    than from a hard-coded card path.
    """
    release_path = root / "production/stage1_futu_sim_v1/runtime_release.json"
    if not release_path.is_file():
        return {"status": "needs_data", "reason": "active runtime release is missing"}
    try:
        from src.workflow.stage1_runtime_release import load_runtime_release
        loaded = load_runtime_release(root, release_path)
    except Exception as error:  # noqa: BLE001 - any verification failure is needs_data
        return {"status": "needs_data", "reason": f"release verification failed: {error}"}
    summary = loaded["action_summary"]
    gates = summary.get("development_gates")
    if not isinstance(gates, dict):
        return {"status": "needs_data", "reason": "action summary has no development_gates"}
    failed = sorted(key for key, value in gates.items() if value is False)
    return {
        "status": "ok",
        "release_id": str(loaded["release"].get("release_id")),
        "allocation_recorder_id": str(loaded["allocation_bundle"].get("recorder_id")),
        "action_recorder_id": str(summary.get("recorder_id")),
        "development_gates": gates,
        "failed_development_gates": failed,
        "development_gate_pass": bool(summary.get("development_gate_pass") is True),
        "paper_trial_authorization": bool(loaded.get("paper_trial_authorization")),
    }
