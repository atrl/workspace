"""Source-qualification workflow for the TSMC monthly-revenue PIT information plane.

Read-only over the approved raw docs; zero models. The ``source_gate`` only becomes
``pass`` after all preflight checks hold AND the qlib ``D.features`` readback matches
the reference asof-state within tolerance. Any drift yields ``needs_data``/``blocked``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data import tsmc_monthly_revenue_store as T

EXPERIMENT = "tsmc_monthly_revenue_pit_qualification_20261010"
SCHEMA_VERSION = "tsmc_monthly_revenue_pit_v1"
QLIB_DOCS = ["https://qlib.readthedocs.io/en/latest/component/data.html",
             "https://qlib.readthedocs.io/en/latest/component/recorder.html"]
FIELD_TOL = {f: 1e-4 for f in T.FIELDS}


def _resolve_root(root) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _verify_docs_manifest(root: Path, docs_manifest: list[dict]) -> dict:
    """Re-read every approved doc and confirm bytes SHA matches the recorded manifest."""
    docs = root / T.RAW_DIR / "docs"
    mismatches, missing = [], []
    for item in docs_manifest:
        path = docs / f"{item['accn']}_{Path(item['url']).name}"
        if not path.is_file():
            missing.append(item["accn"]); continue
        if T._sha(path) != item["sha256"]:
            mismatches.append(item["accn"])
    return {"checked": len(docs_manifest), "missing": missing, "sha_mismatches": mismatches,
            "ok": not missing and not mismatches}


def _all_source_versions(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        parsed = r["parsed"]
        rows.append({"period": r["period"], "accn": r["accessionNumber"], "accepted_utc": r["acceptanceDateTime"],
                     "filingDate": r["filingDate"], "url": r["url"], "raw_sha256": r["raw_sha256"],
                     "format": parsed["format"], "unit": parsed.get("unit"),
                     "reported_mom": parsed.get("reported_mom"), "reported_yoy": parsed.get("reported_yoy"),
                     "current_ntd": parsed.get("current_ntd", parsed.get("current_approx"))})
    if not rows:
        return pd.DataFrame(columns=["period", "accn", "accepted_utc", "is_first_version"])
    frame = pd.DataFrame(rows).sort_values(["period", "accepted_utc"]).reset_index(drop=True)
    frame["is_first_version"] = ~frame.duplicated("period", keep="first")
    return frame


def run_tsmc_monthly_revenue_qualification(*, root=None, card_dir=None) -> dict[str, Any]:
    spec_root = _resolve_root(root)
    card = Path(card_dir) if card_dir is not None else (spec_root / "research" / EXPERIMENT)
    card.mkdir(parents=True, exist_ok=True)
    plan_path = spec_root / "research" / EXPERIMENT / "preregistration_plan.json"
    calendar, frozen = T.us_session_calendar(spec_root)

    # Build events from the ACTUAL approved docs (source of truth), then preflight.
    frame, docs_manifest, records = T.build_events_from_raw(spec_root, calendar)
    checks: dict[str, Any] = {}

    # (0) source-of-truth checks BEFORE any build/qualification.
    try:
        T.assert_source_asof(frame)
        checks["source_asof_ok"] = True
    except ValueError as exc:
        checks["source_asof_ok"] = str(exc)
    cov = T.coverage(frame)
    checks["coverage_complete"] = bool(cov["complete"])
    checks["expected_present_130"] = bool(cov["expected"] == cov["present"] == 130)
    checks["numeric_gate_failures_empty"] = not cov["numeric_gate_failures"]
    checks["extra_not_admitted"] = cov["not_admitted_to_scope"]
    checks["docs_manifest"] = _verify_docs_manifest(spec_root, docs_manifest)
    checks["parser_oracle"] = T.verify_parser_oracle(frame)

    blocking = [k for k in ("source_asof_ok", "coverage_complete", "expected_present_130",
                            "numeric_gate_failures_empty") if checks.get(k) is not True]
    if not checks["docs_manifest"]["ok"]:
        blocking.append("docs_manifest")
    if not checks["parser_oracle"]["all_pass"]:
        blocking.append("parser_oracle")
    if blocking:
        summary = _base_summary(spec_root, plan_path, checks, cov, frozen, frame, records)
        summary.update({"decision": "needs_data", "source_gate": "blocked", "blocking_checks": blocking})
        summary.update(_record(spec_root, card, frame, records, docs_manifest, None, None, frozen, plan_path))
        _write_card(summary, card)
        return summary

    # (1) build store + readback + reference comparison.
    manifest = T.build_qlib_store(spec_root, frame, calendar)
    readback = T.readback(spec_root)
    expected = T.reference_asof_state(frame, calendar)
    rb = readback.rename(columns={f"${f}": f for f in T.FIELDS}).sort_values("datetime").reset_index(drop=True)
    exp = expected.sort_values("datetime").reset_index(drop=True)
    calendar_match = bool(len(rb) == len(exp) and np.array_equal(pd.to_datetime(rb["datetime"]).to_numpy(),
                                                                 pd.to_datetime(exp["datetime"]).to_numpy()))
    field_errors = {}
    for field in T.FIELDS:
        a = pd.to_numeric(rb[field], errors="coerce").to_numpy(float)
        b = pd.to_numeric(exp[field], errors="coerce").to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        nan_match = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        max_err = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        field_errors[field] = {"nan_pattern_match": nan_match, "max_abs_err": max_err,
                               "ok": nan_match and max_err <= FIELD_TOL[field]}
    checks["readback_calendar_match"] = calendar_match
    checks["readback_fields"] = field_errors
    checks["events_available"] = int((rb["event_available"] == 1).sum())
    checks["factor_clock_events"] = int(np.isfinite(pd.to_numeric(rb["acceleration_pct_points"], errors="coerce")).sum())
    gate_pass = calendar_match and all(v["ok"] for v in field_errors.values()) and checks["events_available"] == 130

    summary = _base_summary(spec_root, plan_path, checks, cov, frozen, frame, records)
    summary.update({"decision": "source_plane_qualified_information_only" if gate_pass else "needs_data",
                    "source_gate": "pass" if gate_pass else "blocked",
                    "provider": {"uri": T.PROVIDER_URI, "instrument": T.INSTRUMENT, "manifest": manifest,
                                 "readback_rows": int(len(rb)), "events_available": checks["events_available"]}})
    summary.update(_record(spec_root, card, frame, records, docs_manifest, readback, expected, frozen, plan_path))
    _write_card(summary, card)
    return summary


def _base_summary(spec_root, plan_path, checks, cov, frozen, frame, records) -> dict:
    return {
        "experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION, "card_status": "active_research",
        "models_trained": 0, "returns_computed": False, "ic_computed": False, "alpha": False,
        "predictive_not_tested": True, "promotion_eligible": False,
        "plan_sha256": T._sha(plan_path),
        "source_dimensions": {
            "coverage": cov,
            "scope_extra_not_admitted": cov["not_admitted_to_scope"],
            "precision": {"main_table": "reported_rounded_ntd_million", "old_paragraph": "approx_ntd_billion"},
            "clock": {"accepted_utc_exact": True, "eligible_session": "first US session strictly after acceptance ET date",
                      "calendar_frozen": frozen},
            "model_gate": "not_applicable_zero_models", "production_collector": "needs_data"},
        "checks": checks,
        "qualified_source_events": int((frame["admitted_to_scope"]).sum()) if "admitted_to_scope" in frame else int(cov["present"]),
        "raw_metadata_scope_extra": cov["not_admitted_to_scope"],
        "readback_vs_reference": "serialization_fidelity_not_independent_algorithm",
        "causal_contracts": ["parser_oracle_12", "late_prior_availability", "holiday_next_us_session",
                             "future_revision_rejected", "all_versions_source_asof"],
        "source_as_of_note": "source_as_of=2026-10-10 is treated as the first midnight UTC boundary; no fabricated date-end",
        "data_lineage": {"raw_dir": T.RAW_DIR, "submissions_sha256": T._sha(spec_root / T.RAW_DIR / "submissions.json"),
                         "docs_manifest_sha256": T._sha(spec_root / T.RAW_DIR / "docs_manifest.json"),
                         "calendar_uri": frozen["uri"], "calendar_sha256": frozen["sha256"]},
        "qlib_docs_reference": QLIB_DOCS,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": ["events", "all_source_versions", "docs_manifest",
                                                        "qlib_feature_readback", "reference_asof_state", "calendar"],
                            "slim_sidecars": ["summary.json", "workflow.yaml", "conclusion.md"]},
    }


def _record(spec_root, card, frame, records, docs_manifest, readback, expected, frozen, plan_path):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import logging
    import warnings
    for name in ("qlib", "mlflow", "qlib.data", "qlib.workflow"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    import qlib
    from qlib.workflow import R

    uri = (Path(card) / "qlib_recorder").resolve()
    uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec_root / "data/qlib/soxl").resolve()), region="us")
    saved = {"events": frame, "all_source_versions": _all_source_versions(records),
             "docs_manifest": pd.DataFrame(docs_manifest),
             "calendar": pd.DataFrame([frozen])}
    if readback is not None:
        saved["qlib_feature_readback"] = readback
    if expected is not None:
        saved["reference_asof_state"] = expected
    with R.start(experiment_name=EXPERIMENT, recorder_name="tsmc_source_qual", uri=str(uri)):
        rec = R.get_recorder()
        rec.log_params(schema_version=SCHEMA_VERSION, instrument=T.INSTRUMENT, models_trained=0,
                       source_gate="pending")
        rec.save_objects(**saved)
        rec.log_artifact(str(plan_path), artifact_path="plan")
        rid, eid = rec.id, str(rec.experiment_id)
    return {"recorder_id": rid, "experiment_id": eid, "recorder_uri": str(uri),
            "recorder_local_dir": str(uri / eid / rid), "saved_objects": sorted(saved)}


def _write_card(summary, card: Path) -> None:
    (card / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    workflow = {"schema_version": SCHEMA_VERSION, "experiment": summary["experiment"], "models_trained": 0,
                "decision": summary["decision"], "source_gate": summary.get("source_gate"),
                "source_dimensions": summary["source_dimensions"], "checks": summary.get("checks"),
                "provider_uri": T.PROVIDER_URI, "instrument": T.INSTRUMENT, "qlib_docs_reference": QLIB_DOCS}
    (card / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary) -> str:
    d = summary["source_dimensions"]
    cov = d["coverage"]
    gate = summary.get("source_gate")
    lines = [
        "# TSMC 月营收 PIT 信息数据面（源资格，model=0）\n",
        f"- source_gate=`{gate}`，decision=`{summary['decision']}`；0 model / 0 returns / 0 IC；"
        "predictive_not_tested=True（信息预测效果本轮未测，不claim alpha）。\n",
        f"- 因子：`reported_yoy_acceleration_pct_points`；clock=max(current,prior) 原文可用性→严格下一 US session；"
        "revision 契约=`first_as_filed_only`。\n",
        f"- coverage：expected {cov['expected']}，present {cov['present']}，missing {cov['missing']}，"
        f"not_admitted {cov['not_admitted_to_scope']}，numeric_gate_failures {cov['numeric_gate_failures']}，complete={cov['complete']}。\n",
        f"- 维度：scope_extra_not_admitted / precision / clock / model_gate / production_collector(needs_data)。\n",
        "- 边界：非交易价格信息 instrument；发布日期不假定每月10日；不使用 consensus/beat 口径；不扩窗口、不为过门缩范围。\n",
        "- 下一动作：先独立评估该信息因子的前向预测/经济价值（predictive_not_tested），不得据此直接进入 production。\n",
    ]
    if summary.get("recorder_id"):
        lines.append(f"- 子 Recorder：{summary['recorder_id']}。\n")
    return "".join(lines)
