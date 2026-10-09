"""Audit the Stage1 research lineage for validation integrity.

The Stage1 production candidate descends from a chain of research cards. Two
questions decide whether it is truly validated:

1. Did every card keep its holdout clean (``selection_uses_outer_test == False``
   and a declared non-fresh outer status), and did each promoted card actually
   leave an auditable recorder behind?
2. Has the strategy ever passed a *fresh* walk-forward, or only reused a
   diagnostic window that the whole project has already inspected?

This module reads card summaries read-only, resolves each declared recorder
against the working tree, and reports the lineage, its provenance gaps, and the
forward-evidence status. It does not resolve, start, or modify any recorder.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# The Stage1 production lineage, oldest research card first.
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
# A card whose outer window has already been inspected project-wide.
NON_FRESH_OUTER = {"reused_diagnostic_not_fresh_unseen_test"}


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _declared_recorder(summary: dict) -> str | None:
    for key in ("recorder_id", "source_recorder_id"):
        value = summary.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _recorder_present(card_dir: Path, recorder_id: str) -> bool:
    root = card_dir / RECORDER_DIR
    return root.is_dir() and any(root.glob(f"*/{recorder_id}"))


def audit_lineage(root: Path, *, lineage=LINEAGE, production: str = PRODUCTION_CARD) -> pd.DataFrame:
    rows = []
    for name in (*lineage, production):
        card_root = "production" if name == production else "research"
        card_dir = root / card_root / name
        summary_path = card_dir / "summary.json"
        if not summary_path.is_file():
            rows.append({"card": name, "card_root": card_root, "status": "missing_card",
                         "decision": None, "selection_uses_outer_test": None, "outer_status": None,
                         "outer_is_fresh": True, "recorder_id": None, "recorder_present": False,
                         "source_recorder_id": None})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        recorder_id = _declared_recorder(summary)
        rows.append({
            "card": name,
            "card_root": card_root,
            "decision": summary.get("decision"),
            "selection_uses_outer_test": summary.get("selection_uses_outer_test"),
            "outer_status": summary.get("outer_status"),
            "outer_is_fresh": summary.get("outer_status") not in NON_FRESH_OUTER,
            "recorder_id": recorder_id,
            "recorder_present": bool(recorder_id) and _recorder_present(card_dir, recorder_id),
            "source_recorder_id": summary.get("source_recorder_id"),
        })
    return pd.DataFrame(rows)


def lineage_findings(table: pd.DataFrame) -> dict:
    _require(not table.empty, "lineage table is empty")
    decisions = table["decision"].astype("object").fillna("").astype(str)
    promotion_cards = table[decisions.str.startswith("promoted", na=False)]
    return {
        "cards": int(len(table)),
        "promoted_cards": int(len(promotion_cards)),
        "recorders_declared": int(table["recorder_id"].notna().sum()),
        "recorders_present": int(table["recorder_present"].sum()),
        "recorders_missing": int((table["recorder_id"].notna() & ~table["recorder_present"]).sum()),
        "missing_recorder_cards": table.loc[
            table["recorder_id"].notna() & ~table["recorder_present"], "card"].tolist(),
        "outer_not_fresh_cards": table.loc[table["outer_is_fresh"].eq(False), "card"].tolist(),
        "any_selection_used_outer": bool(
            table["selection_uses_outer_test"].astype("boolean").fillna(False).any()),
        "all_promotions_have_recorder": bool(
            promotion_cards.empty or promotion_cards["recorder_present"].all()),
    }


def walk_forward_verdict(findings: dict, *, forward_sessions: int, required_forward_sessions: int) -> dict:
    """Decide whether the lineage has ever passed a fresh walk-forward.

    A fresh pass requires: no promotion relied on the outer test, every promotion
    left an auditable recorder, and enough genuinely forward sessions were
    observed.
    """
    fresh_forward_enough = forward_sessions >= required_forward_sessions
    passed = bool(
        not findings["any_selection_used_outer"]
        and findings["all_promotions_have_recorder"]
        and fresh_forward_enough
    )
    reasons = []
    if findings["any_selection_used_outer"]:
        reasons.append("at least one card selected using the outer window")
    if not findings["all_promotions_have_recorder"]:
        reasons.append("a promoted card has no committed recorder")
    if not fresh_forward_enough:
        reasons.append(
            f"only {forward_sessions} managed forward sessions < required {required_forward_sessions}")
    return {
        "passed_fresh_walk_forward": passed,
        "forward_sessions": int(forward_sessions),
        "required_forward_sessions": int(required_forward_sessions),
        "failed_reasons": reasons,
    }
