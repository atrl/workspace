"""Sealed, read-only economic-action observations; never a broker execution path."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_futu_sim import infer_frozen_allocation, runtime_state_from_paper_ledger


POLICY_SOURCE = "src/workflow/stage1_economic_action.py"
SHADOW_SOURCE = "src/workflow/stage1_economic_shadow.py"
SYMBOLS = ("SOXL", "SOXX")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(parent: Path, value: str | Path) -> Path:
    path = (parent / value).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("blocked: shadow artifact path escapes its declared root")
    return path


def _verify(path: Path, expected: str) -> None:
    if not expected or not path.is_file() or _sha(path) != expected:
        raise ValueError(f"blocked: shadow artifact hash mismatch: {path.name}")


def _bundle(card: Path, expected: str) -> tuple[dict, set[Path]]:
    manifest_path = card / "frozen_bundle_manifest.json"
    _verify(manifest_path, expected)
    manifest = _read(manifest_path)
    paths = set()
    for entry in [*manifest.get("artifacts", []), *manifest.get("recorder_metadata", [])]:
        path = _inside(card, entry["path"])
        _verify(path, entry["sha256"])
        paths.add(path)
    if not paths:
        raise ValueError("blocked: shadow bundle contains no sealed artifacts")
    return manifest, paths


def load_shadow_candidate(*, root: Path, card_path: str | Path, candidate: str,
                          expected_manifest_sha256: str) -> dict[str, Any]:
    """Failed research candidates may be observed, but cannot acquire execution authority."""
    root = Path(root).resolve()
    card = _inside(root, card_path)
    manifest, sealed = _bundle(card, expected_manifest_sha256)
    summary_path = card / "summary.json"
    if summary_path not in sealed:
        raise ValueError("blocked: research summary must be sealed by the shadow manifest")
    summary = _read(summary_path)
    if manifest.get("recorder_id") != summary.get("recorder_id") or manifest.get("experiment") != summary.get("experiment"):
        raise ValueError("blocked: shadow summary and bundle identity differ")
    code_hashes = manifest.get("code_sha256", {})
    if not {POLICY_SOURCE, SHADOW_SOURCE}.issubset(code_hashes):
        raise ValueError("blocked: shadow bundle must seal policy and shadow implementation")
    for name, digest in code_hashes.items():
        _verify(_inside(root, name), digest)
    _verify(Path(__file__).resolve(), code_hashes[SHADOW_SOURCE])
    metadata = summary.get("shadow_models", {}).get(candidate)
    if not isinstance(metadata, dict) or metadata.get("available") is not True:
        raise ValueError("blocked: requested shadow candidate is not available")
    policy = metadata.get("policy")
    if policy not in {"rules_only", "net_value"}:
        raise ValueError("blocked: unsupported economic shadow policy")
    relative = summary.get("recorder_relative_dir")
    if relative:
        recorder = _inside(card, relative)
    else:
        matches = list((card / "qlib_recorder").glob(f"*/{summary['recorder_id']}"))
        if len(matches) != 1:
            raise ValueError("blocked: shadow Recorder cannot be resolved uniquely")
        recorder = matches[0].resolve()
    if recorder.name != summary["recorder_id"]:
        raise ValueError("blocked: shadow Recorder path has a different identity")
    model_path = _inside(recorder, metadata["artifact"])
    if model_path not in sealed:
        raise ValueError("blocked: shadow model is not sealed by the manifest")
    _verify(model_path, metadata.get("sha256", ""))
    allocation = summary["allocation_source"]
    allocation_card = _inside(root, allocation["card_path"])
    allocation_bundle, allocation_files = _bundle(allocation_card, allocation["manifest_sha256"])
    if allocation_bundle.get("recorder_id") != allocation["recorder_id"]:
        raise ValueError("blocked: shadow allocation identity differs")
    selections = [path for path in allocation_files if path.name == "selected_model.json" and path.parent != allocation_card]
    if len(selections) != 1:
        raise ValueError("blocked: shadow allocation has no unique retained selection")
    selection = _read(selections[0])
    allocation_model = selections[0].parent / "models" / f"{selection['candidate']}.pkl"
    if allocation_model not in allocation_files:
        raise ValueError("blocked: frozen allocator is not sealed")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    method = "predict_proba" if policy == "rules_only" else "predict"
    if not callable(getattr(model, method, None)):
        raise ValueError("blocked: shadow estimator has the wrong prediction interface")
    return {"summary": summary, "model": model, "metadata": metadata, "policy": policy,
            "allocation_bundle": allocation_bundle, "allocation_artifact_dir": selections[0].parent,
            "policy_sha256": code_hashes[POLICY_SOURCE],
            "identity": {"experiment": summary["experiment"], "recorder_id": summary["recorder_id"],
                         "candidate": candidate, "manifest_sha256": expected_manifest_sha256,
                         "summary_sha256": _sha(summary_path), "model_sha256": metadata["sha256"]}}


def _prepare_open(row, *, expected_policy_sha256: str, **kwargs):
    from src.workflow import stage1_economic_action
    _verify(Path(stage1_economic_action.__file__).resolve(), expected_policy_sha256)
    return stage1_economic_action.prepare_economic_open(row, **kwargs)


def _quote_time(value: str, execution_date: str, *, mode: str, now: datetime | None) -> pd.Timestamp:
    observed = pd.Timestamp(value)
    if observed.tzinfo is None:
        raise ValueError("blocked: shadow quote requires a timezone")
    local = observed.tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    if local.date().isoformat() != execution_date or not 570 <= minute < 960:
        raise ValueError("blocked: shadow needs a same-session regular-market quote")
    if mode == "forward":
        current = pd.Timestamp(now or datetime.now(timezone.utc))
        if current.tzinfo is None:
            raise ValueError("blocked: shadow clock requires a timezone")
        age = (current - observed).total_seconds()
        current = current.tz_convert("America/New_York")
        if age < -5 or age > 120 or current.date() != local.date():
            raise ValueError("blocked: stale or future shadow quote")
        if not 571 <= current.hour * 60 + current.minute <= 576:
            raise ValueError("blocked: forward open shadow runs only during 09:31-09:36 ET")
    return observed


def _integer_changes(*, quantities: dict, prices: dict, cash: float, equity: float,
                     weights: np.ndarray, execute: bool, hard_cap: float,
                     maximum_soxl_weight: float, minimum_cash_weight: float, cost_bps: float) -> dict:
    target = ({symbol: int(equity * float(weights[i]) / prices[symbol]) for i, symbol in enumerate(SYMBOLS)}
              if execute else dict(quantities))
    deltas = {symbol: target[symbol] - quantities[symbol] for symbol in SYMBOLS}
    projected_cash = cash - sum(deltas[symbol] * prices[symbol] for symbol in SYMBOLS)
    changes = [{"symbol": symbol, "signed_quantity": deltas[symbol], "side": "BUY" if deltas[symbol] > 0 else "SELL"}
               for symbol in SYMBOLS if deltas[symbol]]
    changes.sort(key=lambda row: row["side"] != "SELL")
    traded = sum(abs(deltas[symbol]) * prices[symbol] for symbol in SYMBOLS)
    scenarios = {}
    for assumed_bps in sorted({10.0, 25.0, float(cost_bps)}):
        fee = traded * assumed_bps / 10_000
        net_equity, net_cash = equity - fee, projected_cash - fee
        risky = {symbol: target[symbol] * prices[symbol] / net_equity for symbol in SYMBOLS} if net_equity > 0 else dict.fromkeys(SYMBOLS, math.inf)
        cash_weight = net_cash / net_equity if net_equity > 0 else -math.inf
        beta = 3 * risky["SOXL"] + risky["SOXX"]
        checks = {"cash_nonnegative": net_cash >= -1e-9, "positive_nav": net_equity > 0,
                  "minimum_cash": cash_weight >= minimum_cash_weight - 1e-12,
                  "maximum_soxl": risky["SOXL"] <= maximum_soxl_weight + 1e-12,
                  "hard_beta_cap": beta <= hard_cap + 1e-12}
        scenarios[f"{assumed_bps:g}bps"] = {"assumed_fee": fee, "net_equity": net_equity,
            "net_cash": net_cash, "net_weights": risky | {"cash": cash_weight}, "effective_beta": beta,
            "checks": checks, "hard_limits_pass": all(checks.values())}
    return {"current_quantities": quantities, "target_quantities": target, "quantity_deltas": deltas,
            "nonzero_changes": changes, "would_change_shares": bool(changes),
            "projected_cash_before_fees": projected_cash, "risky_traded_notional": traded,
            "planning_cost_bps": cost_bps, "same_integer_plan_fee_scenarios": scenarios,
            "planning_cost_hard_limits_pass": scenarios[f"{cost_bps:g}bps"]["hard_limits_pass"],
            "basis": "Integer floor of fee-affordable execution target. Scenarios reprice the same share plan; fees are assumptions, not verified broker fees."}


def _calendar(start: str, end: str) -> pd.DatetimeIndex:
    from qlib.data import D
    return pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq="day"))


def build_economic_open_shadow(
    *, root: Path | str, card_path: str | Path, candidate: str, expected_manifest_sha256: str,
    ledger_path: Path | str, prices: dict[str, float], quote_as_of: str, execution_date: str,
    mode: str = "forward", execution_state_path: Path | str | None = None, now: datetime | None = None,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Observe one decision from the last reconciled account; no continuous candidate NAV is implied."""
    if mode not in {"forward", "snapshot"}:
        raise ValueError("blocked: unsupported shadow observation mode")
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("blocked: invalid shadow cost assumption")
    observed = _quote_time(quote_as_of, execution_date, mode=mode, now=now)
    decision_at = pd.Timestamp(now or datetime.now(timezone.utc)) if mode == "forward" else observed
    prices = {key: float(value) for key, value in prices.items()}
    if set(prices) != set(SYMBOLS) or any(not math.isfinite(v) or v <= 0 for v in prices.values()):
        raise ValueError("blocked: shadow requires positive SOXL/SOXX prices")
    root, ledger_path = Path(root).resolve(), Path(ledger_path).resolve()
    ledger_hash = _sha(ledger_path)
    ledger = _read(ledger_path)
    if ledger.get("execution_environment") != "SIMULATE" or ledger.get("reconciliation", {}).get("status") != "matched":
        raise ValueError("blocked: shadow requires a reconciled SIM ledger")
    ledger_at = pd.Timestamp(ledger["reconciliation"]["as_of"])
    ledger_cutoff = decision_at + pd.Timedelta(seconds=5) if mode == "forward" else observed + pd.Timedelta(seconds=120)
    if ledger_at.tzinfo is None or ledger_at > ledger_cutoff:
        raise ValueError("blocked: ledger state is later than the shadow observation")
    if any(str(mark["date"]) >= execution_date for mark in ledger.get("settled_marks", [])):
        raise ValueError("blocked: same-session future settled marks cannot enter open shadow")
    state_path = Path(execution_state_path).resolve() if execution_state_path else None
    state_hash = _sha(state_path) if state_path else None
    if state_path:
        execution_state = _read(state_path)
        if execution_state.get("pending_slots") or execution_state.get("failed_slots"):
            raise ValueError("blocked: unresolved account execution state cannot seed shadow")
    quantities = {}
    for symbol in SYMBOLS:
        value = float(ledger["positions"].get(symbol, 0))
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError("blocked: shadow requires nonnegative integer holdings")
        quantities[symbol] = int(value)
    cash = float(ledger["cash"])
    equity = cash + sum(quantities[s] * prices[s] for s in SYMBOLS)
    if not math.isfinite(cash) or cash < 0 or equity <= 0:
        raise ValueError("blocked: invalid shadow account equity")
    loaded = load_shadow_candidate(root=root, card_path=card_path, candidate=candidate,
                                   expected_manifest_sha256=expected_manifest_sha256)
    metadata = loaded["metadata"]
    trained_end = pd.Timestamp(metadata["training_signal_end"])
    label_available = pd.Timestamp(metadata["label_available_at_max"])
    if (pd.isna(trained_end) or trained_end >= pd.Timestamp(execution_date)
            or pd.isna(label_available) or label_available.tzinfo is None or label_available >= observed):
        raise ValueError("blocked: shadow model labels are not mature at the observation")
    bundle = loaded["allocation_bundle"]
    frame, audit = infer_frozen_allocation(
        root=root, artifact_dir=loaded["allocation_artifact_dir"],
        source_identity={"source_experiment": bundle["experiment"], "source_recorder_id": bundle["recorder_id"],
                         "shadow_recorder_id": loaded["summary"]["recorder_id"]},
        current_prices=prices, execution_date=execution_date, current_prices_as_of=observed,
    )
    if audit.get("smoothing", {}).get("status") != "ready" or frame.empty:
        raise ValueError("blocked: shadow frozen allocation needs an observed execution mark")
    signal_date = frame.index.max().date().isoformat()
    if signal_date >= execution_date:
        raise ValueError("blocked: shadow signal is not a prior settled session")
    from src.data.market_update import _max_settled_us_daily_date
    expected_signal = _max_settled_us_daily_date(now=observed.to_pydatetime()).isoformat()
    if signal_date != expected_signal:
        raise ValueError("blocked: shadow signal is stale against the exchange calendar")
    calendar = _calendar(signal_date, execution_date)
    prior_sessions = calendar[calendar < pd.Timestamp(execution_date)]
    if len(prior_sessions) == 0 or prior_sessions[-1].date().isoformat() != signal_date:
        raise ValueError("blocked: shadow allocation is stale against its Qlib calendar")
    if not ledger.get("settled_marks") or max(str(m["date"]) for m in ledger["settled_marks"]) < signal_date:
        raise ValueError("blocked: shadow account history has not reached the signal session")
    spec = IntradayRecoveryDirectAllocationSpec(root=root, source_experiment=bundle["experiment"])
    marked_ledger = dict(ledger, prices=prices, equity=equity, peak_equity=max(float(ledger["peak_equity"]), equity))
    state = runtime_state_from_paper_ledger(marked_ledger, recovery_spec=spec)
    fill_dates = [str(fill["create_time"])[:10] for fill in ledger.get("fills", [])]
    if any(day > execution_date for day in fill_dates):
        raise ValueError("blocked: future fill cannot seed a shadow decision")
    for fill in ledger.get("fills", []):
        filled_at = pd.Timestamp(fill["create_time"])
        if filled_at.tzinfo is None:
            filled_at = filled_at.tz_localize("America/New_York")
        if filled_at > decision_at:
            raise ValueError("blocked: a fill after the decision cannot seed shadow holdings")
    last_trade = max(fill_dates) if fill_dates else None
    days_since_trade = 0
    if last_trade and last_trade < execution_date:
        between = _calendar(last_trade, execution_date)
        days_since_trade = int(((between > pd.Timestamp(last_trade)) & (between < pd.Timestamp(execution_date))).sum())
    opened = _prepare_open(
        frame.iloc[-1].copy(), expected_policy_sha256=loaded["policy_sha256"],
        pretrade_weights=state.previous.copy(), equity=state.equity, peak=state.peak,
        risk_state=state.risk_state, risk_clock=state.risk_clock, history=list(state.history),
        recovery_spec=spec, model=loaded["model"], policy=loaded["policy"], days_since_trade=days_since_trade,
        cost_bps=cost_bps,
    )
    weights = np.asarray(opened["open_weights"], dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or weights.min() < 0 or not np.isclose(weights.sum(), 1):
        raise ValueError("blocked: economic shadow returned invalid weights")
    execution_weights = np.asarray(opened["execution_target_weights"], dtype=float)
    if execution_weights.shape != (3,) or not np.isfinite(execution_weights).all() or execution_weights.min() < 0 or not np.isclose(execution_weights.sum(), 1):
        raise ValueError("blocked: economic shadow returned invalid pre-fee execution weights")
    plan = _integer_changes(quantities=quantities, prices=prices, cash=cash, equity=equity, weights=execution_weights,
                            execute=bool(opened["execute_open_target"]), hard_cap=float(opened["hard_beta_cap"]),
                            maximum_soxl_weight=spec.maximum_soxl_weight, minimum_cash_weight=spec.minimum_cash_weight,
                            cost_bps=cost_bps)
    if not plan["planning_cost_hard_limits_pass"]:
        raise ValueError("blocked: shadow integer plan violates a hard limit under its declared fee assumption")
    if _sha(ledger_path) != ledger_hash or (state_path and _sha(state_path) != state_hash):
        raise ValueError("blocked: account state changed during the shadow observation; retry from fresh inputs")
    return _ready({
        "status": "ready", "observation_mode": mode, "execution_mode": "READ_ONLY_SHADOW",
        "research_promotion_status": "not_promoted", "orders_submitted": 0, "submit_supported": False,
        "own_state_simulation": False, "legacy_paper_trial_receipt_used": False,
        "candidate_identity": loaded["identity"], "policy": loaded["policy"],
        "signal_date": signal_date, "slot": "open", "execution_date": execution_date, "quote_as_of": observed.isoformat(),
        "decision_as_of": decision_at.isoformat(),
        "ledger_observed_at": ledger_at.isoformat(), "ledger_sha256": ledger_hash,
        "execution_state_sha256": state_hash, "state_source": "last_reconciled_actual_SIM_ledger_marked_at_supplied_prices",
        "account_cash": cash, "account_equity_at_supplied_prices": equity, "actual_pretrade_weights": state.previous,
        "decision": "rebalance" if opened["execute_open_target"] else "hold",
        "decision_reason": opened["decision_reason"], "predicted_net_advantage_bps": opened.get("predicted_net_advantage_bps"),
        "target_weights": dict(zip((*SYMBOLS, "cash"), weights)), "desired_open_weights": opened["desired_open_weights"],
        "execution_target_weights": dict(zip((*SYMBOLS, "cash"), execution_weights)),
        "hard_beta_cap": opened["hard_beta_cap"], "soft_beta_cap": opened["soft_beta_cap"],
        "decision_features": opened["features"], "days_since_trade": days_since_trade,
        "integer_plan": plan, "source_audit": audit,
        "limits": ["One-step actual-account shadow, not a continuously executed candidate portfolio.",
                   "Ledger was read locally; this function does not refresh broker positions or quotes.",
                   "This observer evaluates open decisions only; it does not run the recovery slot.",
                   "Snapshot mode is retrospective acceptance evidence, never a forward observation."]})


def _ready(value):
    if isinstance(value, dict):
        return {str(k): _ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_ready(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def run_once(**inputs) -> dict:
    """Write only the separate shadow namespace; deduplicate ready forward observations."""
    root = Path(inputs["root"]).resolve()
    candidate, session = inputs["candidate"], inputs["execution_date"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", candidate) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", session):
        raise ValueError("blocked: invalid shadow candidate/session identity")
    mode = inputs.get("mode", "forward")
    if mode not in {"forward", "snapshot"}:
        raise ValueError("blocked: invalid shadow mode")
    directory = root / "runtime/soxl/stage1_economic_shadow" / mode / session
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if mode == "forward" else "_" + pd.Timestamp(inputs["quote_as_of"]).strftime("%H%M%S")
    output = directory / f"{candidate}{suffix}.json"
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.exists():
            prior = _read(output)
            if prior["candidate_identity"]["manifest_sha256"] != inputs["expected_manifest_sha256"]:
                raise ValueError("blocked: shadow candidate revision changed within the same session")
            return {"status": "already_recorded", "observation_path": str(output), "orders_submitted": 0,
                    "research_promotion_status": "not_promoted", "observation_mode": mode}
        result = build_economic_open_shadow(**inputs)
        result["recorded_at"] = datetime.now(timezone.utc).isoformat()
        result["observation_path"] = str(output)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record one economic-action shadow from supplied quotes; no broker connection.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args(argv)
    inputs = _read(Path(args.input_json))
    result = build_economic_open_shadow(**inputs) if args.no_record else run_once(**inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
