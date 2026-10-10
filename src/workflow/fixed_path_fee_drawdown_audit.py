"""Fixed-path fee vs drawdown audit workflow (Step A only; controller not re-run).

Implements: (1) 4 source-inherited native cases verified against the parent recorder
(stage status + exact full-calendar dates/returns/three ending weights/object SHA) + 8
real counterfactual native reconciliations with expected stage presence/counts; (2) real
source exit_date inheritance; (3) resettled strategy_return + resettled ending weights;
(4) the single frozen actual emergency mask shared by all settlements and an explicit
fixed0-vs-source comparison; (5) explicit 0.22 threshold from the plan; (6) daily
dd/nav/peak/peak-date/episode saved to the recorder; (7) provider + source-object hash
verification, a card-frozen source_qa.json input, and per-case qlib risk_analysis.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis import fixed_path_fee_drawdown as A
from src.workflow import coherent_price_replay_experiment as C

EXPERIMENT = "stage1_fixed_path_fee_drawdown_audit_20261010"
SCHEMA_VERSION = "stage1_fixed_path_fee_drawdown_audit_v2"
PARENT_CARD = "research/stage1_uniform_layer_attribution_20261010"
PARENT_RECORDER = "6c6fac4f2d974545a4ca8ed8b149a17c"
PARENT_SUMMARY_SHA = "90ae79b7e16a6f3ba68aa75f8b044b8a92e5578a86246f6947a7974c72aab20f"
PARENT_MANIFEST_SHA = "806e41ee64eadce742f00c1e2002867667142e58bfa25cb002ca9941e8061f35"
LAYER = "open_risk_projection_nohold_no_recovery"
WINDOWS = ("development_walk_forward", "outer_reused_diagnostic")
EXPECTED_SESSIONS = {"development_walk_forward": 1756, "outer_reused_diagnostic": 934}
BASE_COSTS = (10.0, 25.0)
SETTLEMENT_COSTS = (0.0, 10.0, 25.0)
FROZEN_REPLAY_SHA = {
    "development_walk_forward::10": "865d39cdddaad69d6d94e7907c006d1a92820940d5ec1aa76c75d9649409c1c7",
    "development_walk_forward::25": "3b5f2bbe1815625e190e47059a8da380b9e64056c9e07541e1c7f754321a56ee",
    "outer_reused_diagnostic::10": "35804aba75799d913b9b453381b83051fe537221fbf64ee4360f2d804b3306be",
    "outer_reused_diagnostic::25": "51fae60de1dd86a615da739e2bed4b513099eb7789c2b40a15f66373765c277d",
}
PROVIDER_KEY = {"intraday_provider": "data/qlib/stage1_uniform_intraday_20261009",
                "execution_mark_provider": "data/qlib/stage1_uniform_execution_marks_20261009",
                "coherent_provider": "data/qlib/stage1_uniform_daily_marks_20261009"}
QLIB_DOCS = C.QLIB_DOCS + ["https://qlib.readthedocs.io/en/latest/reference/api.html#qlib.contrib.evaluate.risk_analysis"]
WEIGHT_KEYS = ("ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight")
DATE_KEYS = ("signal_date", "trade_date", "exit_date")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_parent(root: Path, plan: dict) -> tuple[dict, dict]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # repo default; does not override an explicit user value
    from qlib.workflow import R
    import qlib
    if _sha(root / PARENT_CARD / "summary.json") != PARENT_SUMMARY_SHA:
        raise ValueError("blocked: parent summary identity changed")
    if _sha(root / PARENT_CARD / "artifact_manifest.json") != PARENT_MANIFEST_SHA:
        raise ValueError("blocked: parent artifact_manifest identity changed")
    manifest = json.loads((root / PARENT_CARD / "artifact_manifest.json").read_text())
    path_index = {Path(a["path"]).name: a["path"] for a in manifest["artifacts"]}
    sha_index = {Path(a["path"]).name: a["sha256"] for a in manifest["artifacts"]}
    parent_summary = json.loads((root / PARENT_CARD / "summary.json").read_text())
    frozen_providers = plan.get("source_binding", {}).get("provider_manifest_sha256", {})
    if frozen_providers:
        actual = {name: _sha(root / rel / "qlib_store_manifest.json") for name, rel in PROVIDER_KEY.items()}
        if actual != frozen_providers:
            raise ValueError("blocked: provider identity changed vs plan")
    qlib.init(provider_uri=str((root / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    with R.uri_context(str((root / PARENT_CARD / "qlib_recorder").resolve())):
        rec = R.get_recorder(recorder_id=PARENT_RECORDER, experiment_name=PARENT_CARD.split("/")[-1])
        local = Path(rec.get_local_dir()) / "artifacts"
        replays, natives, checked = {}, {}, {}
        for key, expected in FROZEN_REPLAY_SHA.items():
            actual = _sha(local / f"replay::{LAYER}::{key}")
            if actual != expected:
                raise ValueError(f"blocked: frozen replay {key} identity changed")
            replays[key] = rec.load_object(f"replay::{LAYER}::{key}")
            checked[key] = actual
        for key in FROZEN_REPLAY_SHA:
            for stage in ("prefix_reconciliation", "suffix_reconciliation"):
                name = f"native::{LAYER}::{key}::{stage}"
                if (local / name).exists():
                    natives[f"{key}::{stage}"] = rec.load_object(name)
    return replays, {"checked": checked, "natives": natives, "providers": frozen_providers,
                     "path_index": path_index, "sha_index": sha_index, "summary": parent_summary}


def _object_sha_ok(root: Path, parent: dict, name: str) -> bool:
    rel = parent["path_index"].get(name)
    return rel is not None and _sha(root / PARENT_CARD / rel) == parent["sha_index"].get(name)


def _stage_frames(parent: dict, key: str):
    """Only the stages that actually exist for this case (outer has no prefix)."""
    return {stage: parent["natives"].get(f"{key}::{stage}_reconciliation") for stage in ("prefix", "suffix")}


def _align(nat: pd.DataFrame, src: pd.DataFrame):
    """Exact calendar alignment by trade_date; returns None if missing/duplicated."""
    src_idx = src.set_index("trade_date")
    if not src_idx.index.is_unique or not nat["trade_date"].is_unique:
        return None
    try:
        return src_idx.loc[nat["trade_date"].to_numpy()].reset_index(drop=True)
    except KeyError:
        return None


def _native_base_verify(root: Path, base: pd.DataFrame, parent: dict, *, window: str, cost: float) -> dict:
    """Verify inherited native stages against the source replay (no re-run).

    Requires each stage's status == passed, the expected session count, an exact
    full-window calendar (unique/ordered, no dropna/dedup/inner-join hiding), the three
    dates, native_return vs source strategy_return, and native soxl/soxx/cash ending
    weights vs source ending weights, plus the object SHA in the parent manifest.
    """
    key = f"{window}::{int(cost)}"
    case = parent["summary"]["native"]["results"][f"{LAYER}::{key}"]
    src = base.copy()
    for c in DATE_KEYS:
        src[c] = pd.to_datetime(src[c])
    total_expected = EXPECTED_SESSIONS.get(window, len(src))
    src = src.sort_values("trade_date").reset_index(drop=True)
    out = {"origin": "source_inherited", "parent_all_stages_passed": bool(case.get("all_stages_passed"))}
    frames = _stage_frames(parent, key)
    covered, stage_meta = [], {}
    for stage, frame in frames.items():
        stage_case = case.get(stage)
        if stage_case is None:  # outer has no prefix stage
            out[stage] = {"status": "not_required", "passed": True, "sessions": 0, "sessions_expected": 0}
            continue
        expected = int(stage_case["sessions"])
        meta = {"expected": expected, "status_read": stage_case.get("status"),
                "sha_ok": _object_sha_ok(root, parent, f"native::{LAYER}::{key}::{stage}_reconciliation")}
        stage_meta[stage] = meta
        if frame is None:
            out[stage] = {"status": "missing", "passed": False, "sessions": 0, "sessions_expected": expected}
            continue
        nat = frame.copy()
        for c in DATE_KEYS:
            nat[c] = pd.to_datetime(nat[c])
        nat = nat.sort_values("trade_date").reset_index(drop=True)
        meta["nat"] = nat
        covered.append(nat["trade_date"])
    # full-window exact calendar (union of the actual stages, no dedup/dropna)
    if covered:
        union = pd.concat(covered, ignore_index=True)
        union_ok = bool(len(union) == len(union.unique()) == total_expected
                        and union.is_monotonic_increasing
                        and np.array_equal(union.sort_values().to_numpy(), src["trade_date"].to_numpy()))
    else:
        union_ok = False
    all_ok = bool(case.get("all_stages_passed")) and union_ok
    for stage, meta in stage_meta.items():
        nat = meta.get("nat")
        if nat is None:
            all_ok = False
            continue
        tol = float(case[stage].get("tolerance", 1e-7))
        expected = meta["expected"]
        aligned = _align(nat, src)
        dates_ok = bool(aligned is not None and len(aligned) == expected
                        and (aligned["signal_date"] == nat["signal_date"]).all()
                        and (aligned["exit_date"] == nat["exit_date"]).all())
        ret_err = float((nat["native_return"] - aligned["strategy_return"]).abs().max()) if aligned is not None else np.inf
        cash = 1.0 - nat["native_soxl_ending_weight"] - nat["native_soxx_ending_weight"]
        w_err = float(max((nat["native_soxl_ending_weight"] - aligned["ending_soxl_weight"]).abs().max(),
                          (nat["native_soxx_ending_weight"] - aligned["ending_soxx_weight"]).abs().max(),
                          (cash - aligned["ending_cash_weight"]).abs().max())) if aligned is not None else np.inf
        status_ok = meta["status_read"] == "passed"
        passed = bool(status_ok and dates_ok and ret_err <= tol and w_err <= tol and meta["sha_ok"]
                      and len(nat) == expected)
        out[stage] = {"status": "passed" if passed else "failed", "passed": passed, "sessions": int(len(nat)),
                      "sessions_expected": expected, "stage_status_read": meta["status_read"],
                      "dates_exact": dates_ok, "object_sha_ok": meta["sha_ok"],
                      "maximum_return_error": ret_err, "maximum_ending_weight_error": w_err, "tolerance": tol}
        all_ok = all_ok and passed
    out["full_calendar_exact"] = union_ok
    out["total_sessions_expected"] = total_expected
    out["all_stages_passed"] = bool(all_ok)
    return out


def _source_native_readback(card: Path, plan: dict) -> dict:
    rel = plan["source_binding"].get("source_qa_file", "source_qa.json")
    path = card / rel
    if not path.exists():
        return {"status": "needs_data", "reason": f"missing frozen input {rel}"}
    actual = _sha(path)
    expected = plan["source_binding"].get("source_qa_sha256")
    if expected and actual != expected:
        raise ValueError("blocked: source_qa sha mismatch vs plan")
    a = json.loads(path.read_text())
    checks = a["checks"]
    if a.get("recorder_id") != PARENT_RECORDER:
        raise ValueError("blocked: source_qa recorder identity mismatch")
    return {"status": "recorded", "file": rel, "sha256": actual, "recorder_id": a["recorder_id"],
            "checks_count": len(checks), "all_passed": bool(all(c["passed"] for c in checks)),
            "max_ending_weight_error": float(max(c["maximum_ending_weight_error"] for c in checks)),
            "max_return_error": float(max(c["maximum_return_error"] for c in checks)),
            "cash_weight_method": a.get("cash_weight_method"),
            "not_relying_on_default_zero": a.get("not_relying_on_default_zero_missing_schedule_weight_error"),
            "checks": checks}


def _risk_report(native_returns: np.ndarray, research_returns: np.ndarray) -> dict:
    from qlib.contrib.evaluate import risk_analysis
    return {label: risk_analysis(pd.Series(np.asarray(arr, float)), N=252, freq=None, mode="product")
            for label, arr in (("native", native_returns), ("research", research_returns))}


def _counterfactual_schedule(base: pd.DataFrame, resettled: pd.DataFrame) -> pd.DataFrame:
    """Resettled targets/flags with resettled returns and resettled ending weights."""
    return pd.DataFrame({
        "signal_date": base["signal_date"], "trade_date": base["trade_date"], "exit_date": base["exit_date"],
        "strategy_return": resettled["strategy_return"],
        "execute_open_target": base["execute"].astype(bool), "recovery_action": base["recovery_action"].astype(bool),
        "open_soxl_weight": base["desired_soxl"].fillna(0.0), "open_soxx_weight": base["desired_soxx"].fillna(0.0),
        "open_cash_weight": base["desired_cash"].fillna(1.0),
        "soxl_weight": base["recovery_soxl"].where(base["recovery_action"], base["desired_soxl"]).fillna(0.0),
        "soxx_weight": base["recovery_soxx"].where(base["recovery_action"], base["desired_soxx"]).fillna(0.0),
        "cash_weight": base["recovery_cash"].where(base["recovery_action"], base["desired_cash"]).fillna(1.0),
        "ending_soxl_weight": resettled["ending_soxl_weight"], "ending_soxx_weight": resettled["ending_soxx_weight"],
        "ending_cash_weight": resettled["ending_cash_weight"]})


def _counterfactual_stage_verdict(entry: dict, *, status: str, expected_prefix: int, expected_suffix: int) -> dict:
    """Expected-stage presence + session count + status; never PASS on an empty dict."""
    stages = {}
    if expected_prefix > 0:
        stages["prefix"] = {"expected": expected_prefix, "present": "prefix" in entry,
                            "status": entry.get("prefix", {}).get("status"),
                            "sessions": entry.get("prefix", {}).get("sessions")}
    stages["suffix"] = {"expected": expected_suffix, "present": "suffix" in entry,
                        "status": entry.get("suffix", {}).get("status"),
                        "sessions": entry.get("suffix", {}).get("sessions")}
    passed = bool(stages) and status == "passed" and all(
        s["present"] and s["status"] == "passed" and s["sessions"] == s["expected"] for s in stages.values())
    return {"all_stages_passed": bool(passed),
            "present_all_stages": bool(stages) and all(s["present"] for s in stages.values()), "stages": stages}


def _native_counterfactual(root: Path, base: pd.DataFrame, resettled: pd.DataFrame, *, window: str,
                           base_cost: float, settlement_cost: float, panel: pd.DataFrame) -> dict:
    """Real Qlib native for a non-base settlement: resettled returns + resettled ending weights."""
    coherent = (root / PROVIDER_KEY["coherent_provider"]).resolve()
    execution = (root / PROVIDER_KEY["execution_mark_provider"]).resolve()
    schedule = _counterfactual_schedule(base, resettled)
    boundary = pd.Timestamp(C._read_manifest(coherent)["intraday_price_source_from"])
    expected_prefix = int((pd.to_datetime(schedule["trade_date"]) < boundary).sum())
    expected_suffix = int(len(schedule) - expected_prefix)
    key = f"settle{int(settlement_cost)}_b{int(base_cost)}::{window}::{int(settlement_cost)}"
    spec = type("S", (), {"root": root})()
    try:
        res, objs = C._native_reconcile(spec, panel, {key: schedule}, coherent, execution,
                                       target_conventions={key: "pre_fee_account"})
        entry = res["results"][key]
        verdict = _counterfactual_stage_verdict(entry, status=res.get("status"), expected_prefix=expected_prefix,
                                                expected_suffix=expected_suffix)
        recon = pd.concat([objs[f"{key}::{stage}_reconciliation"] for stage in ("prefix", "suffix")
                           if f"{key}::{stage}_reconciliation" in objs], ignore_index=True)
        return {"origin": "recomputed_settlement", "boundary": res.get("boundary"),
                "expected_prefix_sessions": expected_prefix, "expected_suffix_sessions": expected_suffix,
                "reconciliation": recon, "objects": objs, **verdict}
    except ValueError as exc:
        return {"origin": "recomputed_settlement", "all_stages_passed": False, "status": "needs_data", "reason": str(exc)}


def _source_actual_mask(source: pd.DataFrame, resettled: pd.DataFrame, *, threshold: float) -> np.ndarray:
    return source["emergency_active"].astype(bool).to_numpy()


def _intersection_episodes(mask: np.ndarray, dates: pd.DatetimeIndex) -> list:
    stamps = pd.DatetimeIndex(dates).to_numpy()
    episodes, start = [], None
    for i, flag in enumerate(mask.astype(int)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            episodes.append({"start": str(pd.Timestamp(stamps[start]).date()),
                             "end": str(pd.Timestamp(stamps[i - 1]).date()), "days": i - start})
            start = None
    if start is not None:
        episodes.append({"start": str(pd.Timestamp(stamps[start]).date()),
                         "end": str(pd.Timestamp(stamps[-1]).date()), "days": len(mask) - start})
    return episodes


def _actual_vs_hypothetical(source: pd.DataFrame, resettled: pd.DataFrame, *, threshold: float, base_settlement: bool) -> dict:
    actual = source["emergency_active"].astype(bool).to_numpy()
    src_dd = source["drawdown_before"].to_numpy(float)
    res_dd = resettled["drawdown_before"].to_numpy(float)
    hyp = res_dd >= threshold
    base_dd_ok = bool(np.allclose(src_dd, res_dd, rtol=0, atol=1e-12))
    if base_settlement and not base_dd_ok:  # real assert: base re-settlement must match source exactly
        raise ValueError("blocked: base settlement drawdown_before does not match source")
    inter = actual & hyp
    dates = pd.to_datetime(source["trade_date"]).reset_index(drop=True)
    return {"actual_emergency_days": int(actual.sum()), "hypothetical_emergency_days": int(hyp.sum()),
            "actual_and_hypothetical": int(inter.sum()),
            "actual_and_hypothetical_share": float(inter.sum() / actual.sum()) if actual.sum() else None,
            "actual_and_hypothetical_episodes": _intersection_episodes(inter, dates),
            "actual_dd_before_matches_source": base_dd_ok,
            "note": "actual mask is the frozen source emergency_active; hypothetical is per settlement cost"}


def _fixed0_vs_source(source: pd.DataFrame, fixed0: pd.DataFrame, *, threshold: float) -> dict:
    """The fixed-0 re-settlement against the frozen source actual emergency mask."""
    actual = source["emergency_active"].astype(bool).to_numpy()
    hyp = fixed0["drawdown_before"].to_numpy(float) >= threshold
    inter = actual & hyp
    dates = pd.to_datetime(source["trade_date"]).reset_index(drop=True)
    return {"actual_emergency_days": int(actual.sum()), "fixed0_emergency_days": int(hyp.sum()),
            "fixed0_and_actual_days": int(inter.sum()),
            "fixed0_share_of_actual": float(inter.sum() / actual.sum()) if actual.sum() else None,
            "fixed0_only_days": int((hyp & ~actual).sum()), "actual_only_days": int((actual & ~hyp).sum()),
            "fixed0_and_actual_episodes": _intersection_episodes(inter, dates)}


def run_fixed_path_fee_drawdown_audit(*, root=None, card_dir=None, source_card_dir=None) -> dict:
    """``source_card_dir`` holds the frozen plan/source_qa inputs; ``card_dir`` is the output.

    They default to the same canonical research card, but tests may pass a separate output
    dir so a tmp output card never masquerades as the frozen source.
    """
    C._quiet()
    spec_root = Path(root).resolve() if root else Path(".").resolve()
    card = Path(card_dir) if card_dir else (spec_root / "research" / EXPERIMENT)
    source_card = Path(source_card_dir) if source_card_dir else card
    card.mkdir(parents=True, exist_ok=True)
    plan_path = source_card / "preregister_plan.json"
    plan = json.loads(plan_path.read_text())
    threshold = float(plan.get("emergency_threshold", 0.22))
    source_qa = _source_native_readback(source_card, plan)
    replays, parent = _load_parent(spec_root, plan)
    panel = C.rebuild_market_inputs(C.load_frozen(spec_root)["inputs/prepared_inputs.pkl"]["frame"],
                                    coherent_provider=spec_root / PROVIDER_KEY["coherent_provider"],
                                    intraday_provider=spec_root / PROVIDER_KEY["intraday_provider"])
    from src.analysis.fixed_path_fee_drawdown import _validate_base
    cases, daily_objects, native_objects, risk_objects = {}, {}, {}, {}
    for window in WINDOWS:
        for base_cost in BASE_COSTS:
            base = replays[f"{window}::{int(base_cost)}"].reset_index(drop=True)
            _validate_base(base, convention="pre_fee_account")
            base_native = _native_base_verify(spec_root, base, parent, window=window, cost=base_cost)
            if not base_native["all_stages_passed"]:
                raise ValueError(f"blocked: inherited native verification failed {window}@{base_cost}")
            bkey = f"{window}::{int(base_cost)}"
            frames = _stage_frames(parent, bkey)
            both = pd.concat([f for f in frames.values() if f is not None], ignore_index=True).sort_values("trade_date")
            risk_objects[f"risk_analysis::{bkey}::{int(base_cost)}"] = _risk_report(
                both["native_return"].to_numpy(float), both["research_return"].to_numpy(float))
            f0_res = A.resettle_preset_path(base, settlement_cost=0.0, convention="pre_fee_account")
            f0_stats = A.drawdown_stats(f0_res, emergency_threshold=threshold)
            f0_res = f0_res.copy()
            f0_res["drawdown_before"] = f0_stats["drawdown_before"]
            f0_mask = np.asarray(f0_stats["drawdown_before"]) >= threshold
            fixed0_cmp = _fixed0_vs_source(base, f0_res, threshold=threshold)
            for settlement in SETTLEMENT_COSTS:
                resettled = A.resettle_preset_path(base, settlement_cost=settlement, convention="pre_fee_account")
                st = A.drawdown_stats(resettled, emergency_threshold=threshold)
                enriched = resettled.copy()
                enriched["drawdown_before"] = st["drawdown_before"]
                equity = np.concatenate([[1.0], np.cumprod(1.0 + resettled["strategy_return"].to_numpy(float))])
                enriched["equity"] = equity[:-1]
                enriched["peak"] = np.maximum.accumulate(equity)[:-1]
                enriched["peak_capital_date"] = st["peak_capital_date"]
                enriched["emergency_hypothetical"] = np.asarray(st["drawdown_before"]) >= threshold
                enriched["emergency_fixed0"] = f0_mask
                enriched["settlement_cost_bps"] = settlement
                key = f"{window}::{int(base_cost)}::{int(settlement)}"
                origin = "source_inherited" if settlement == base_cost else "recomputed_settlement"
                if origin == "source_inherited":
                    parity = A.base_parity(base, resettled)
                    if parity["status"] != "passed":
                        raise ValueError(f"blocked: base parity failed {key}: {parity}")
                    native = base_native
                else:
                    parity = {"status": "counterfactual"}
                    native = _native_counterfactual(spec_root, base, resettled, window=window, base_cost=base_cost,
                                                    settlement_cost=settlement, panel=panel)
                    if native.get("all_stages_passed") and native.get("reconciliation") is not None:
                        native["risk_status"] = "confirmed"
                        risk_objects[f"risk_analysis::{key}"] = _risk_report(
                            native["reconciliation"]["native_return"].to_numpy(float),
                            native["reconciliation"]["research_return"].to_numpy(float))
                    else:
                        native["risk_status"] = "needs_data"  # never take a risk result without a native reconciliation
                    for name, value in native.pop("objects", {}).items():
                        native_objects[f"native::{key}::{name}"] = value
                    native.pop("reconciliation", None)
                daily_objects[key] = enriched
                cases[key] = {"window": window, "base_controller_cost_bps": base_cost, "settlement_cost_bps": settlement,
                              "origin_stage": origin, "parity": parity, "native": native,
                              "true_max_drawdown": st["true_max_drawdown"], "emergency_days": st["emergency_days"],
                              "emergency_share": st["emergency_share"], "emergency_longest_days": st["emergency_longest_days"],
                              "peak_capital_first_date": st["peak_capital_date"][0] if st["peak_capital_date"] else None,
                              "actual_vs_hypothetical": _actual_vs_hypothetical(base, enriched, threshold=threshold,
                                                                                base_settlement=(settlement == base_cost)),
                              "fixed0_vs_source": fixed0_cmp}
    summary = {
        "experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION, "status": "diagnostic",
        "models_trained": 0, "new_strategy": 0, "orders": 0, "promotion_eligible": False, "scientific_promotion": False,
        "step_b_authorized": False, "plan_sha256": _sha(plan_path), "plan_full_sha256": _sha(plan_path),
        "emergency_threshold": threshold,
        "source_binding": {"parent_card": PARENT_CARD, "parent_commit": "c0672b77", "recorder_id": PARENT_RECORDER,
                           "summary_sha256": PARENT_SUMMARY_SHA, "artifact_manifest_sha256": PARENT_MANIFEST_SHA,
                           "frozen_replay_sha256": parent["checked"], "provider_manifest_sha256": parent["providers"],
                           "source_qa_file": plan["source_binding"].get("source_qa_file", "source_qa.json"),
                           "source_qa_sha256": plan["source_binding"].get("source_qa_sha256"),
                           "funding_convention": "pre_fee_account"},
        "source_native_qa": source_qa,
        "region": "us", "as_of": parent["summary"]["as_of"],
        "source_snapshot_as_of": parent["summary"]["source_snapshot_as_of"],
        "price_vintage_as_of": parent["summary"]["price_vintage_as_of"],
        "provider_lineage": parent["summary"]["provider_lineage"],
        "native_coverage": {"by_origin": {k: v["native"].get("all_stages_passed") for k, v in cases.items()},
                            "passed_count": int(sum(1 for v in cases.values() if v["native"].get("all_stages_passed"))),
                            "case_count": len(cases),
                            "inherited_expected": 4, "recomputed_expected": 8,
                            "inherited_passed": int(sum(1 for v in cases.values()
                                                        if v["origin_stage"] == "source_inherited" and v["native"].get("all_stages_passed"))),
                            "recomputed_passed": int(sum(1 for v in cases.values()
                                                         if v["origin_stage"] == "recomputed_settlement" and v["native"].get("all_stages_passed")))},
        "fixed0_vs_source": {f"{w}::{int(c)}": cases[f"{w}::{int(c)}::0"]["fixed0_vs_source"]
                             for w in WINDOWS for c in BASE_COSTS},
        "cases": cases, "cases_count": len(cases),
        "risk_case_coverage": {k: v["native"].get("risk_status", "confirmed" if k else "needs_data") for k, v in cases.items()},
        "caveats": ["actual mask is the frozen source emergency_active; hypothetical masks are per settlement cost",
                    "no unique market-vs-fee causal attribution", "reused history; not fresh out-of-sample",
                    "10bp to 25bp is a 15bp step; fees are deducted once and the rest is state feedback",
                    "no alpha; no promotion; Step B fee-neutral guard not executed"],
        "qlib_docs_reference": QLIB_DOCS,
    }
    recorded = _record(spec_root, card, daily_objects, cases, native_objects, risk_objects, plan_path)
    risk_table = recorded.pop("risk_table")
    summary.update(recorded)
    _write_card(summary, card, risk_table)
    return summary


def _risk_small_table(risk_objects: dict, cases: dict) -> pd.DataFrame:
    """One block per case key (must equal the 12-case set); needs_data cases carry no fabricated risk value."""
    rows = []
    for case in sorted(cases):
        report = risk_objects.get(f"risk_analysis::{case}")
        status = "confirmed" if report is not None else cases[case].get("native", {}).get("risk_status", "needs_data")
        if report is None:
            rows.append({"case": case, "series": "native", "metric": "status", "value": np.nan, "status": status})
            continue
        for label, frame in report.items():
            for metric, value in frame.iloc[:, 0].items():
                rows.append({"case": case, "series": label, "metric": metric, "value": float(value), "status": status})
    return pd.DataFrame(rows)


def _record(spec_root, card, daily_objects, cases, native_objects, risk_objects, plan_path):
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
    qlib.init(provider_uri=str((spec_root / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    saved = {f"daily::{k}": v for k, v in daily_objects.items()}
    saved.update(native_objects)
    saved.update(risk_objects)
    saved["risk_analysis_summary"] = _risk_small_table(risk_objects, cases)
    saved["case_summary"] = pd.DataFrame([{"case": k, "origin_stage": v["origin_stage"],
                                            "native_all_stages_passed": v["native"].get("all_stages_passed"),
                                            "true_max_drawdown": v["true_max_drawdown"],
                                            "emergency_days": v["emergency_days"],
                                            **v["actual_vs_hypothetical"]} for k, v in cases.items()])
    with R.start(experiment_name=EXPERIMENT, recorder_name="fixed_path_fee_dd", uri=str(uri)):
        rec = R.get_recorder()
        rec.log_params(schema_version=SCHEMA_VERSION, models_trained=0, cases=len(cases),
                       native_passed=sum(1 for v in cases.values() if v["native"].get("all_stages_passed")))
        rec.save_objects(**saved)
        rec.log_artifact(str(plan_path), artifact_path="plan")
        rid, eid = rec.id, str(rec.experiment_id)
    return {"recorder_id": rid, "experiment_id": eid, "recorder_uri": str(uri), "saved_objects": sorted(saved),
            "risk_table": saved["risk_analysis_summary"]}


def _write_card(summary, card, risk_table):
    risk_names = sorted({k for k in summary["saved_objects"] if k.startswith("risk_analysis::")})
    summary["artifact_policy"] = {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                  "recorder_only_artifacts": sorted([f"daily::{k}" for k in summary["cases"]] + risk_names + ["case_summary", "risk_analysis_summary", "native::*"]),
                                  "slim_sidecars": ["summary.json", "workflow.yaml", "conclusion.md", "fixed_path_cases.csv", "risk_analysis_summary.csv"]}
    (card / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame([{"case": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, dict)},
                   **v["actual_vs_hypothetical"]} for k, v in summary["cases"].items()]).to_csv(
        card / "fixed_path_cases.csv", index=False)
    risk_table.to_csv(card / "risk_analysis_summary.csv", index=False)
    workflow = {"schema_version": SCHEMA_VERSION, "experiment": summary["experiment"], "status": "diagnostic",
                "models_trained": 0, "emergency_threshold": summary["emergency_threshold"],
                "native_coverage": summary["native_coverage"], "source_binding": summary["source_binding"],
                "source_native_qa_status": summary["source_native_qa"].get("status"),
                "provider_lineage": summary["provider_lineage"],
                "source_snapshot_as_of": summary["source_snapshot_as_of"],
                "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS}
    (card / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    selected = {"selection": "none_prescribed_target_path_audit_only", "models_trained": 0,
                "source_card": PARENT_CARD, "source_recorder_id": PARENT_RECORDER,
                "frozen_paths": list(summary["source_binding"]["frozen_replay_sha256"]),
                "controller_rerun": False, "new_strategy": 0, "promotion_eligible": False}
    (card / "selected_model.json").write_text(json.dumps(selected, indent=2) + "\n")




def _conclusion(summary) -> str:
    qa = summary["source_native_qa"]
    nc = summary["native_coverage"]
    lines = ["# 固定路径费用-回撤审计（Step A，诊断；0 model/0 strategy/0 orders）\n",
             f"- 相同实际目标/flag路径，controller 不重跑；funding=`pre_fee_account`；阈值 {summary['emergency_threshold']}（旧 controller 假设）。\n",
             f"- native：passed {nc['passed_count']}/{nc['case_count']}（inherited {nc['inherited_passed']}/4，recomputed {nc['recomputed_passed']}/8）；"
             f"plan sha `{summary['plan_full_sha256'][:12]}`。\n",
             f"- 源 native 复核（冻结输入 source_qa.json，sha `{qa.get('sha256','')[:12]}`）：父 recorder `{qa.get('recorder_id')}`，"
             f"{qa.get('checks_count')} checks，all_passed={qa.get('all_passed')}，max ending-weight error {qa.get('max_ending_weight_error')}。\n",
             "### fixed0 对照源 actual（回答 stop_rule）\n"]
    for key, v in summary["fixed0_vs_source"].items():
        lines.append(f"- {key}：源 actual {v['actual_emergency_days']} 日，fixed0 {v['fixed0_emergency_days']} 日，"
                     f"交集 {v['fixed0_and_actual_days']} 日（占 actual {v['fixed0_share_of_actual']}），"
                     f"仅 fixed0 {v['fixed0_only_days']}，仅 actual {v['actual_only_days']}。\n")
    lines.append("### 各 case（actual mask = 源 emergency_active；hypothetical 按 settlement cost）\n")
    for key, v in summary["cases"].items():
        c = v["actual_vs_hypothetical"]
        lines.append(f"- {key}（{v['origin_stage']}，native={v['native'].get('all_stages_passed')}）：trueMaxDD {v['true_max_drawdown']:.4f}；"
                     f"actual {c['actual_emergency_days']}，hypothetical {c['hypothetical_emergency_days']}，"
                     f"交集 {c['actual_and_hypothetical']}（占比 {c['actual_and_hypothetical_share']}）；base DD_before 与源吻合={c['actual_dd_before_matches_source']}。\n")
    lines += ["- 机制：费用只扣一次；10bp→25bp 是 15bp 步长，其余为净 NAV→drawdown_before→risk 的状态反馈。\n",
              "- actual 与 hypothetical 不混用；不作可加 DD 因果分解，不唯一归因市场与费差。\n",
              "- 限制：外窗为复用历史；不晋升（notpromoted）；Step B 未授权；下一机制须另预登记与独立验证。\n"]
    return "".join(lines)
