"""Independent continuous paper holdings; no broker or active-account inputs.

Each decision is an observed event. The book uses fractional quantities, quoted
prices and proportional assumed cash fees. It is price-return evidence: cash
distributions and corporate actions are never silently inferred. Missing slots
block advancement instead of inventing retrospective decisions.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np

from src.backtest.position_accounting import settle_target_session


NY = ZoneInfo("America/New_York")
SYMBOLS = ("SOXL", "SOXX")
PAPER_MODE = "INDEPENDENT_PAPER"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity_sha256(value) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def new_paper_state(*, candidate_identity: dict, initial_cash: float = 100_000,
                    cost_bps: float = 10.0) -> dict:
    if not candidate_identity or not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("blocked: paper requires candidate identity and positive initial cash")
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("blocked: invalid paper cost assumption")
    return {
        "version": 1, "execution_mode": PAPER_MODE, "research_promotion_status": "not_promoted",
        "candidate_identity": deepcopy(candidate_identity),
        "candidate_sha256": identity_sha256(candidate_identity),
        "initial_cash": float(initial_cash), "cost_bps": float(cost_bps),
        "cash": float(initial_cash), "positions": dict.fromkeys(SYMBOLS, 0.0),
        "equity": float(initial_cash), "peak": float(initial_cash),
        "risk_state": 0, "risk_clock": 0, "history": [], "days_since_trade": 0,
        "pending_session": None, "last_event_at": None, "events": {},
        "orders_submitted": 0, "own_state_simulation": True,
        "accounting": "fractional_quantities_price_returns_assumed_proportional_cash_fees",
        "limitations": ["No broker fills or integer-share execution are implied.",
                        "Distributions and corporate actions require separate verified adjustments; none are inferred."],
    }


def _aware(value: str, label: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"blocked: invalid {label}") from error
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f"blocked: {label} requires timezone")
    return stamp.astimezone(NY)


def _check_event(state: dict, event: dict) -> tuple[str, str, datetime]:
    if (state.get("version") != 1 or state.get("execution_mode") != PAPER_MODE
            or state.get("own_state_simulation") is not True
            or state.get("orders_submitted") != 0):
        raise ValueError("blocked: unsupported independent paper state")
    if identity_sha256(state["candidate_identity"]) != state["candidate_sha256"]:
        raise ValueError("blocked: paper candidate identity changed")
    if (not math.isfinite(float(state["cost_bps"])) or not 0 <= float(state["cost_bps"]) < 10_000
            or not math.isfinite(float(state["peak"])) or float(state["peak"]) <= 0
            or not np.isfinite(state["history"]).all()):
        raise ValueError("blocked: invalid paper cost or controller history")
    if event.get("candidate_sha256") != state["candidate_sha256"]:
        raise ValueError("blocked: event belongs to another candidate")
    session, slot = event.get("session"), event.get("slot")
    if not isinstance(session, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", session) or slot not in {"open", "recovery"}:
        raise ValueError("blocked: invalid paper session or slot")
    if not re.fullmatch(r"[0-9a-f]{64}", str(event.get("input_sha256", ""))):
        raise ValueError("blocked: paper event needs a sealed input hash")
    event_id = f"{session}:{slot}"
    if event.get("event_id", event_id) != event_id:
        raise ValueError("blocked: paper event id differs from session/slot")
    observed = _aware(event["observed_at"], "event observation")
    quote = _aware(event["quote_as_of"], "quote observation")
    if "feature_available_at" in event and _aware(event["feature_available_at"], "feature availability") > observed:
        raise ValueError("blocked: paper features are not yet available")
    minute = observed.hour * 60 + observed.minute
    lower = 571 if slot == "open" else 631
    if (observed.date().isoformat() != session or quote.date() != observed.date()
            or observed.weekday() >= 5 or not lower <= minute < lower + 6
            or not 570 <= quote.hour * 60 + quote.minute < 960
            or not -5 <= (observed - quote).total_seconds() <= 120):
        raise ValueError("blocked: stale, future, or wrong-slot paper observation")
    prices = event.get("prices", {})
    if set(prices) != set(SYMBOLS) or any(not math.isfinite(float(v)) or float(v) <= 0 for v in prices.values()):
        raise ValueError("blocked: paper requires positive finite SOXL/SOXX prices")
    return event_id, identity_sha256(event), observed


def _mark(state: dict, prices: dict) -> tuple[float, np.ndarray]:
    values = np.array([state["positions"][symbol] * float(prices[symbol]) for symbol in SYMBOLS] + [state["cash"]], dtype=float)
    if not np.isfinite(values).all() or values.min() < -1e-9 or values.sum() <= 0:
        raise ValueError("blocked: invalid independent paper holdings")
    equity = float(values.sum())
    return equity, values / equity


def _trade(state: dict, *, prices: dict, equity: float, weights: np.ndarray, target) -> dict:
    if target is None:
        # A HOLD preserves quantities and cash exactly; normalization roundoff
        # must not manufacture changes in the candidate's owned position path.
        state["equity"] = equity
        return {"assumed_fee": 0.0, "traded_notional": 0.0,
                "posttrade_equity": equity, "posttrade_weights": weights.copy()}
    if target is not None:
        candidate = np.asarray(target, dtype=float)
        if candidate.shape != (3,) or not np.isfinite(candidate).all() or candidate.min() < 0:
            raise ValueError("blocked: invalid paper target")
    settled = settle_target_session(weights, open_target_weights=target,
                                    asset_returns=[0.0, 0.0, 0.0], cost_bps=state["cost_bps"])
    values = equity * settled["ending_values"]
    state["positions"] = {symbol: float(values[i] / float(prices[symbol])) for i, symbol in enumerate(SYMBOLS)}
    state["cash"] = float(values[2])
    state["equity"] = float(values.sum())
    return {"assumed_fee": equity * settled["fees"], "traded_notional": equity * settled["turnover"],
            "posttrade_equity": state["equity"], "posttrade_weights": settled["ending_weights"]}


def process_paper_event(state: dict, event: dict, *, open_decider: Callable,
                        recovery_decider: Callable) -> tuple[dict, dict]:
    """Pure event transition; callbacks use the candidate book and supplied features.

    Required event fields: session, slot, observed_at, quote_as_of, prices,
    input_sha256, candidate_sha256, features. Open also requires signal_date
    and previous_session, both supplied from the authoritative Qlib calendar.
    An open callback returns the existing prepare-open contract; recovery
    returns the prepare-recovery contract. The caller seals callback identities.
    """
    event_id, event_hash, observed = _check_event(state, event)
    prior = state["events"].get(event_id)
    if prior:
        if prior["event_sha256"] != event_hash:
            raise ValueError("blocked: paper slot already recorded with different inputs")
        return deepcopy(state), dict(prior, status="already_recorded")
    if state["last_event_at"] and observed <= _aware(state["last_event_at"], "prior event"):
        raise ValueError("blocked: paper event time is not increasing")
    result = deepcopy(state)
    session, slot, prices = event["session"], event["slot"], event["prices"]
    pending = result["pending_session"]
    equity, weights = _mark(result, prices)
    result["equity"] = equity
    completed_return = None
    if slot == "open":
        previous_session = event.get("previous_session")
        if (not isinstance(previous_session, str) or previous_session >= session
                or event.get("signal_date") != previous_session):
            raise ValueError("blocked: open signal must match prior Qlib trading session")
        if pending:
            if not pending["recovery_observed"]:
                raise ValueError("blocked: previous recovery observation is missing; no hindsight replay")
            if pending["session"] != previous_session:
                raise ValueError("blocked: missing intervening open session; no hindsight replay")
            completed_return = equity / pending["start_equity"] - 1.0
            result["history"].append(completed_return)
            result["days_since_trade"] = 0 if pending["traded"] else result["days_since_trade"] + 1
        result["peak"] = max(float(result["peak"]), equity)
        decision = open_decider(row=deepcopy(event["features"]), state=deepcopy(result), pretrade_weights=weights.copy())
        risk_state, risk_clock = decision["risk_state"], decision["risk_clock"]
        if risk_state not in (0, 1, 2, 3) or not isinstance(risk_clock, (int, np.integer)) or risk_clock < 0:
            raise ValueError("blocked: invalid paper controller transition")
        result["risk_state"], result["risk_clock"] = int(risk_state), int(risk_clock)
        target = decision["execution_target_weights"] if decision["execute_open_target"] else None
        trade = _trade(result, prices=prices, equity=equity, weights=weights, target=target)
        context = deepcopy(decision)
        context["open_weights"] = trade["posttrade_weights"]
        context["branch_open_net_equity"] = trade["posttrade_equity"] / equity
        result["pending_session"] = {
            "session": session, "signal_date": event["signal_date"], "start_equity": equity,
            "open_prices": deepcopy(prices), "open_observed_at": event["observed_at"],
            "open_state": _json_ready(context), "recovery_observed": False,
            "traded": trade["traded_notional"] > 1e-8,
        }
    else:
        if not pending or pending["session"] != session or pending["recovery_observed"]:
            raise ValueError("blocked: recovery requires an unconsumed same-session open")
        context = deepcopy(pending["open_state"])
        context["pretrade_weights"] = weights.copy()
        decision = recovery_decider(row=deepcopy(event["features"]), state=deepcopy(result),
                                    pretrade_weights=weights.copy(), open_state=context)
        target = decision["target_weights"] if decision["recovery_action"] else None
        trade = _trade(result, prices=prices, equity=equity, weights=weights, target=target)
        pending["recovery_observed"] = True
        pending["traded"] = pending["traded"] or trade["traded_notional"] > 1e-8
    result["last_event_at"] = event["observed_at"]
    receipt = _json_ready({
        "status": "recorded", "execution_mode": PAPER_MODE, "orders_submitted": 0,
        "own_state_simulation": True, "research_promotion_status": "not_promoted",
        "event_id": event_id, "event_sha256": event_hash, "candidate_sha256": result["candidate_sha256"],
        "input_sha256": event["input_sha256"], "observed_at": event["observed_at"],
        "quote_as_of": event["quote_as_of"], "prices": deepcopy(prices),
        "pretrade_equity": equity, "pretrade_weights": weights, "completed_session_return": completed_return,
        "decision": decision, "trade": trade, "positions": result["positions"], "cash": result["cash"],
        "risk_state": result["risk_state"], "risk_clock": result["risk_clock"],
    })
    result["events"][event_id] = receipt
    return _json_ready(result), receipt


def commit_paper_event(*, state_path: Path | str, event: dict, initial_state: dict,
                       open_decider: Callable, recovery_decider: Callable,
                       evidence_writer: Callable, now: datetime | None = None) -> dict:
    """Lock, record Qlib evidence, then atomically replace the independent book.

    evidence_writer(event, receipt, state) must persist through a Qlib recorder.
    On recorder failure the local state is unchanged. A crash after evidence
    and before replace can repeat the same event artifact, never a broker trade.
    """
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("blocked: paper commit clock requires timezone")
    observed = _aware(event["observed_at"], "event observation")
    if not -5 <= (current - observed).total_seconds() <= 120:
        raise ValueError("blocked: paper commit cannot consume a stale or future event")
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else deepcopy(initial_state)
        if state["candidate_sha256"] != initial_state["candidate_sha256"]:
            raise ValueError("blocked: existing paper book belongs to another candidate")
        if any(state[key] != initial_state[key] for key in ("initial_cash", "cost_bps")):
            raise ValueError("blocked: existing paper book accounting assumptions changed")
        updated, receipt = process_paper_event(state, event, open_decider=open_decider,
                                               recovery_decider=recovery_decider)
        if receipt["status"] == "already_recorded":
            return receipt
        evidence_identity = evidence_writer(deepcopy(event), deepcopy(receipt), deepcopy(updated))
        if evidence_identity is not None:
            receipt["qlib_evidence"] = _json_ready(evidence_identity)
            updated["events"][receipt["event_id"]] = deepcopy(receipt)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(_encoded(updated) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return receipt


def load_sealed_recorder_objects(*, root: Path | str, manifest_path: Path | str,
                                 expected_manifest_sha256: str) -> tuple[dict, dict]:
    """Verify a release and load its objects through the native Qlib recorder.

    Manifest keys: execution_mode, recorder_uri, experiment_name, recorder_id,
    code_sha256 ({repo_relative_path: hash}), objects ({name: {path, sha256}}).
    The caller initializes Qlib with its audited provider first. recorder_uri
    is local and relative to root; object paths are recorder
    artifact paths. This does not create a recorder or mutate research evidence.
    """
    root = Path(root).resolve()
    path = Path(manifest_path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected_manifest_sha256:
        raise ValueError("blocked: paper release manifest identity mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("execution_mode") != PAPER_MODE:
        raise ValueError("blocked: release is not independent paper")
    code = manifest.get("code_sha256", {})
    own_source = "src/workflow/stage1_regime_paper.py"
    if own_source not in code:
        raise ValueError("blocked: paper book implementation is not sealed")
    for name, digest in code.items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("blocked: paper release code hash mismatch")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != code[own_source]:
        raise ValueError("blocked: imported paper implementation differs from release")
    uri = (root / manifest["recorder_uri"]).resolve()
    if not uri.is_relative_to(root) or not uri.is_dir():
        raise ValueError("blocked: paper source recorder is unavailable")
    from qlib.workflow import R
    with R.uri_context(str(uri)):
        recorder = R.get_recorder(recorder_id=manifest["recorder_id"], experiment_name=manifest["experiment_name"])
        if recorder.id != manifest["recorder_id"]:
            raise ValueError("blocked: paper source recorder identity differs")
        artifact_root = (Path(recorder.get_local_dir()) / "artifacts").resolve()
        if not artifact_root.is_relative_to(uri):
            raise ValueError("blocked: paper recorder artifact location escapes the sealed store")
        objects = {}
        for name, entry in manifest["objects"].items():
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("blocked: invalid paper object artifact path")
            artifact = (artifact_root / relative).resolve()
            if not artifact.is_relative_to(artifact_root) or hashlib.sha256(artifact.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("blocked: paper object hash mismatch")
            objects[name] = recorder.load_object(str(relative))
    return manifest, objects
