"""Observe paired candidate/baseline paper books; never connect to broker trading.

One captured quote and feature group feeds independent candidate and baseline
books at each declared fee level. The frozen release stays not_promoted. Every
accepted book event is written to a new Qlib recorder before local state commit.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.schedule.us_open_quotes import read_session_quotes
from src.workflow.stage1_regime_paper import (
    _json_ready, commit_paper_event, identity_sha256, load_sealed_recorder_objects, new_paper_state,
)


NY = ZoneInfo("America/New_York")
RUNTIME_ROOT = "runtime/soxl/stage1_regime_paper"
OPEN_FIELDS = ("soxl_weight", "soxx_weight", "cash_weight", "tail_probability", "alpha_probability")
BAR_FIELDS = ("$open", "$high", "$low", "$close", "$volume", "$bar_count")
FIRST_HOUR_CLOSES = ("09:45", "10:00", "10:15", "10:30")


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("blocked: paper release path escapes repository")
    return path


def _write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _slot(now: datetime) -> str | None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("blocked: paper scheduler clock requires timezone")
    current = now.astimezone(NY)
    if current.weekday() >= 5:
        return None
    minute = current.hour * 60 + current.minute
    return "open" if 571 <= minute < 577 else ("recovery" if 631 <= minute < 637 else None)


def _initialize_provider(path: Path):
    import qlib
    if not path.is_dir() or not (path / "qlib_store_manifest.json").is_file():
        raise ValueError("needs_data: audited Qlib provider is unavailable")
    qlib.init(provider_uri=str(path), region="us", kernels=1)


def _load_release(root: Path, config: dict):
    path = _inside(root, config["release_manifest"])
    if _sha(path) != config["expected_manifest_sha256"]:
        raise ValueError("blocked: paper release manifest identity mismatch")
    preview = json.loads(path.read_text())
    _initialize_provider(_inside(root, preview["daily_provider_uri"]))
    manifest, objects = load_sealed_recorder_objects(root=root, manifest_path=path,
        expected_manifest_sha256=config["expected_manifest_sha256"])
    required_code = {"src/schedule/stage1_regime_paper.py", "src/schedule/us_open_quotes.py",
                     "src/workflow/stage1_regime_action.py", "src/workflow/stage1_economic_action.py",
                     "src/workflow/stage1_futu_sim.py", "src/workflow/stage1_open_action_value.py",
                     "src/backtest/position_accounting.py"}
    if not required_code.issubset(manifest.get("code_sha256", {})):
        raise ValueError("blocked: paired paper release does not seal all runtime policies")
    if (manifest.get("research_promotion_status") != "not_promoted"
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", manifest["release_id"])
            or sorted(manifest["cost_bps"]) != [10, 25]):
        raise ValueError("blocked: unsupported paired paper release contract")
    if set(objects) != {"action_model", "recovery_model", "baseline_action_model"}:
        raise ValueError("blocked: paired paper release model set is incomplete")
    for name in ("action_model", "recovery_model", "baseline_action_model"):
        method = "predict" if name == "action_model" else "predict_proba"
        if not callable(getattr(objects[name], method, None)):
            raise ValueError(f"blocked: paper {name} lacks its prediction interface")
    return manifest, objects


def _allocation_artifacts(root: Path, manifest: dict):
    source = manifest["allocation_source"]
    card = _inside(root, source["card_path"])
    bundle_path = card / "frozen_bundle_manifest.json"
    if _sha(bundle_path) != source["manifest_sha256"]:
        raise ValueError("blocked: frozen allocation bundle identity changed")
    bundle = json.loads(bundle_path.read_text())
    sealed = set()
    for entry in [*bundle.get("artifacts", []), *bundle.get("recorder_metadata", [])]:
        path = _inside(card, entry["path"])
        if _sha(path) != entry["sha256"]:
            raise ValueError("blocked: frozen allocation artifact identity changed")
        sealed.add(path)
    if bundle["recorder_id"] != source["recorder_id"]:
        raise ValueError("blocked: frozen allocation source identity differs")
    selections = [path for path in sealed if path.name == "selected_model.json" and path.parent != card]
    if len(selections) != 1:
        raise ValueError("blocked: frozen allocation selection is ambiguous")
    selected = json.loads(selections[0].read_text())
    if selections[0].parent / "models" / f"{selected['candidate']}.pkl" not in sealed:
        raise ValueError("blocked: allocator weights are not sealed")
    return selections[0].parent, source


def build_open_inputs(*, root: Path, manifest: dict, session: str, prices: dict,
                      quote_as_of: str, observed_at: datetime) -> dict:
    """Infer only the prior settled signal and its trailing SOXX regime fields."""
    from qlib.data import D
    from src.data.market_update import _max_settled_us_daily_date
    from src.workflow.stage1_futu_sim import infer_frozen_allocation
    from src.workflow.stage1_regime_action import REGIME_FEATURES, load_regime_features

    daily = _inside(root, manifest["daily_provider_uri"])
    _initialize_provider(daily)
    expected = _max_settled_us_daily_date(now=observed_at).isoformat()
    calendar = pd.DatetimeIndex(D.calendar(end_time=expected, freq="day"))
    if calendar.empty or calendar[-1].date().isoformat() != expected:
        raise ValueError("needs_data: daily Qlib calendar has not reached the previous settled session")
    artifact_dir, source = _allocation_artifacts(root, manifest)
    frame, audit = infer_frozen_allocation(root=root, artifact_dir=artifact_dir,
        source_identity={"source_recorder_id": source["recorder_id"], "source_card": source["card_path"]},
        current_prices=prices, execution_date=session, current_prices_as_of=quote_as_of)
    if frame.empty or audit.get("smoothing", {}).get("status") != "ready":
        raise ValueError("needs_data: frozen allocation needs a current execution mark")
    signal_date = pd.Timestamp(frame.index.max()).date().isoformat()
    if signal_date != expected:
        raise ValueError("needs_data: allocation signal differs from the prior settled exchange session")
    row = frame.loc[pd.Timestamp(signal_date)]
    features = {name: float(row[name]) for name in OPEN_FIELDS}
    regime = load_regime_features(_inside(root, manifest["regime_provider_uri"]), signal_date, signal_date)
    features.update({name: float(regime.loc[pd.Timestamp(signal_date), name]) for name in REGIME_FEATURES})
    if not np.isfinite(list(features.values())).all():
        raise ValueError("needs_data: incomplete causal paper opening features")
    # Whitelisting prevents forward returns, next-open prices and later
    # first-hour columns from escaping the full inference frame.
    return {"features": features, "signal_date": signal_date, "previous_session": expected,
            "source": {"allocation": source, "allocation_audit": audit,
                       "daily_manifest_sha256": _sha(daily / "qlib_store_manifest.json"),
                       "regime": regime.attrs["regime_source"]},
            "feature_available_at": pd.Timestamp(f"{signal_date} 16:00:00", tz=NY).isoformat()}


def read_first_hour_features(*, provider_uri: Path, session: str, observed_at: datetime) -> dict:
    """Read exactly the four close-stamped bars known before 10:31 ET."""
    from qlib.data import D

    current = observed_at.astimezone(NY)
    available = pd.Timestamp(f"{session} 10:30:00", tz="America/New_York")
    if current <= available.to_pydatetime():
        raise ValueError("needs_data: first hour is not complete")
    _initialize_provider(provider_uri)
    manifest_path = provider_uri / "qlib_store_manifest.json"
    manifest_hash = _sha(manifest_path)
    raw = D.features(["SOXL", "SOXX"], list(BAR_FIELDS), start_time=f"{session} 09:45:00",
                     end_time=f"{session} 10:30:00", freq="15min").reset_index()
    if raw.empty or _sha(manifest_path) != manifest_hash:
        raise ValueError("needs_data: first-hour provider empty or changed during read")
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    if raw.datetime.dt.tz is not None:
        raw["datetime"] = raw.datetime.dt.tz_convert(NY).dt.tz_localize(None)
    # Use the identical groupby aggregation and vector arithmetic as the
    # historical first-hour audit. Series.sum/scalar arithmetic can round
    # high-volume float32 inputs differently from the trained feature path.
    aggregated = raw.sort_values(["instrument", "datetime"]).groupby("instrument").agg(
        early_open=("$open", "first"), early_close=("$close", "last"),
        early_high=("$high", "max"), early_low=("$low", "min"),
        early_volume=("$volume", "sum"), early_bar_count=("$bar_count", "sum"),
    )
    aggregated["early_ret"] = aggregated.early_close / aggregated.early_open - 1.0
    aggregated["early_range"] = aggregated.early_low / aggregated.early_high - 1.0
    aggregated["early_from_low"] = aggregated.early_close / aggregated.early_low - 1.0
    features, rows = {}, []
    for symbol in ("SOXL", "SOXX"):
        bars = raw[raw.instrument == symbol].sort_values("datetime")
        if (tuple(bars.datetime.dt.strftime("%H:%M")) != FIRST_HOUR_CLOSES
                or not bars.datetime.dt.strftime("%Y-%m-%d").eq(session).all()
                or not np.isfinite(bars.loc[:, list(BAR_FIELDS)].to_numpy(dtype=float)).all()
                or (bars.loc[:, ["$open", "$high", "$low", "$close"]].to_numpy() <= 0).any()
                or (bars["$volume"] < 0).any()):
            raise ValueError(f"needs_data: {symbol} exact first-hour bars are unavailable")
        if ((bars["$high"] < bars[["$open", "$close", "$low"]].max(axis=1)).any()
                or (bars["$low"] > bars[["$open", "$close", "$high"]].min(axis=1)).any()):
            raise ValueError(f"needs_data: {symbol} first-hour OHLC values are inconsistent")
        suffix = symbol.lower()
        for name in ("early_ret", "early_range", "early_from_low", "early_volume", "early_bar_count"):
            features[f"{name}_{suffix}"] = float(aggregated.loc[symbol, name])
        rows.extend([{**{name: float(row[name]) for name in BAR_FIELDS},
                      "instrument": symbol, "datetime": row.datetime.isoformat()} for _, row in bars.iterrows()])
    return {"features": features, "feature_available_at": available.isoformat(),
            "source": {"provider_uri": str(provider_uri), "provider_manifest_sha256": manifest_hash,
                       "source_api": "qlib.data.D.features", "bars": rows,
                       "bar_count_semantics": "full-session bar count, never minute coverage"}}


def _deciders(root: Path, manifest: dict, objects: dict, *, role: str, cost_bps: float):
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    from src.workflow.stage1_economic_action import prepare_economic_recovery
    from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec, prepare_on_policy_open, prepare_on_policy_recovery
    from src.workflow.stage1_regime_action import RegimeActionSpec, prepare_regime_open

    source_spec = replace(IntradayRecoveryDirectAllocationSpec(root=root, **manifest["recovery_spec"]), transaction_cost_bps=cost_bps)
    action_spec = replace(Stage1OpenActionValueSpec(root=root, **manifest["baseline_action_spec"]), transaction_cost_bps=cost_bps)
    regime_spec = RegimeActionSpec(**manifest["regime_action_spec"])

    def opened(*, row, state, pretrade_weights):
        common = dict(pretrade_weights=pretrade_weights, equity=state["equity"], peak=state["peak"],
            risk_state=state["risk_state"], risk_clock=state["risk_clock"], history=state["history"], recovery_spec=source_spec)
        if role == "candidate":
            return prepare_regime_open(pd.Series(row), **common, model=objects["action_model"], spec=regime_spec,
                                       days_since_trade=state["days_since_trade"], cost_bps=cost_bps)
        result = prepare_on_policy_open(pd.Series(row), **common, model=objects["baseline_action_model"], action_spec=action_spec)
        # Preserve the original baseline's raw target and cash-fee convention.
        result["execution_target_weights"] = result["desired_open_weights"]
        return result

    def recovered(*, row, state, pretrade_weights, open_state):
        common = dict(open_state=open_state, recovery_spec=source_spec,
                      model=objects["recovery_model"], selected_model=manifest["selected_recovery_model"])
        if role == "candidate":
            return prepare_economic_recovery(pd.Series(row), **common, pretrade_weights=pretrade_weights, cost_bps=cost_bps)
        return prepare_on_policy_recovery(pd.Series(row), **common)

    return opened, recovered


def _evidence_writer(runtime: Path, manifest: dict, book: str):
    def write(event, receipt, state):
        from qlib.workflow import R
        uri = runtime / "qlib_recorder"
        experiment = f"{manifest['release_id']}_forward_events"
        with R.start(experiment_name=experiment, uri=str(uri), recorder_name=f"{book}_{event['session']}_{event['slot']}"):
            recorder = R.get_recorder()
            recorder.log_params(book=book, candidate_sha256=state["candidate_sha256"], event_sha256=receipt["event_sha256"],
                                shared_input_sha256=event["input_sha256"], own_state_simulation=True, orders_submitted=0)
            recorder.save_objects(**{"event.pkl": event, "receipt.pkl": receipt, "state.pkl": state})
            return {"experiment_name": experiment, "recorder_id": recorder.id,
                    "recorder_uri": str(uri), "state_sha256": identity_sha256(state)}
    return write


def run(*, root, config_path, now=None) -> dict:
    """Run one observed slot, or return waiting without a market connection."""
    root = Path(root).resolve()
    live_clock = now is None
    observed = now or datetime.now(NY)
    slot = _slot(observed)
    base = {"orders_submitted": 0, "execution_mode": "INDEPENDENT_PAPER",
            "own_state_simulation": True, "research_promotion_status": "not_promoted"}
    if slot is None:
        return base | {"status": "waiting", "reason": "outside_open_or_recovery_observation_window", "observed_at": observed.isoformat()}
    runtime = None
    try:
        config = json.loads(Path(config_path).read_text())
        manifest, objects = _load_release(root, config)
        for name in ("not_before", "label_available_at_max"):
            cutoff = pd.Timestamp(manifest[name])
            if cutoff.tzinfo is None or cutoff > pd.Timestamp(observed):
                raise ValueError(f"blocked: paper release {name} is not available at observation")
        runtime = root / RUNTIME_ROOT / manifest["release_id"]
        runtime.mkdir(parents=True, exist_ok=True)
        with (runtime / "scheduler.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return base | {"status": "already_running"}
            session = observed.astimezone(NY).date().isoformat()
            group_path = runtime / "groups" / f"{session}_{slot}.json"
            identities, states, pending_books = {}, {}, []
            for cost in manifest["cost_bps"]:
                for role in ("candidate", "baseline"):
                    book = f"{role}_{cost:g}bps"
                    identity = {"release_manifest_sha256": config["expected_manifest_sha256"], "role": role, "cost_bps": cost}
                    identities[book] = new_paper_state(candidate_identity=identity, initial_cash=manifest["initial_cash"], cost_bps=cost)
                    state_path = runtime / "books" / f"{book}.json"
                    states[book] = json.loads(state_path.read_text()) if state_path.exists() else identities[book]
                    if states[book]["candidate_sha256"] != identities[book]["candidate_sha256"]:
                        raise ValueError("blocked: paired book release identity changed")
                    if f"{session}:{slot}" not in states[book]["events"]:
                        pending_books.append((book, role, cost))
            if not pending_books:
                result = base | {"status": "already_recorded", "session": session, "slot": slot, "books": list(states)}
                _write(runtime / "latest_status.json", result)
                return result
            if group_path.exists():
                group = json.loads(group_path.read_text())
                digest = group.pop("shared_input_sha256")
                if identity_sha256(group) != digest or group["release_manifest_sha256"] != config["expected_manifest_sha256"]:
                    raise ValueError("blocked: paired observation group identity changed")
                group["shared_input_sha256"] = digest
            else:
                prices, quote_at = read_session_quotes(host=config.get("host", "127.0.0.1"), port=int(config.get("port", 11111)), slot=slot, now=observed)
                if slot == "open":
                    inputs = build_open_inputs(root=root, manifest=manifest, session=session, prices=prices,
                                               quote_as_of=quote_at, observed_at=observed)
                else:
                    open_path = runtime / "groups" / f"{session}_open.json"
                    if not open_path.is_file():
                        raise ValueError("blocked: recovery has no observed opening input group")
                    open_group = json.loads(open_path.read_text())
                    original_hash = open_group.pop("shared_input_sha256")
                    if identity_sha256(open_group) != original_hash:
                        raise ValueError("blocked: saved opening input group changed")
                    if any(not state.get("pending_session") or state["pending_session"]["session"] != session for state in states.values()):
                        raise ValueError("blocked: recovery requires paired same-session candidate-owned openings")
                    early = read_first_hour_features(provider_uri=_inside(root, manifest["intraday_provider_uri"]), session=session, observed_at=observed)
                    inputs = {"features": open_group["inputs"]["features"] | early["features"],
                              "signal_date": open_group["inputs"]["signal_date"],
                              "previous_session": open_group["inputs"]["previous_session"],
                              "feature_available_at": early["feature_available_at"],
                              "source": {"open_shared_input_sha256": original_hash, "first_hour": early["source"]}}
                decided_at = datetime.now(NY) if live_clock else observed
                if pd.Timestamp(inputs["feature_available_at"]) > pd.Timestamp(decided_at):
                    raise ValueError("blocked: paper features are not yet available")
                group = _json_ready({"session": session, "slot": slot, "observed_at": decided_at.isoformat(), "quote_as_of": quote_at,
                         "prices": prices, "inputs": inputs, "release_manifest_sha256": config["expected_manifest_sha256"]})
                group["shared_input_sha256"] = identity_sha256(group)
                _write(group_path, group)
            results = {}
            for book, role, cost in pending_books:
                inputs = group["inputs"]
                event = {"session": session, "slot": slot, "observed_at": group["observed_at"], "quote_as_of": group["quote_as_of"],
                         "prices": group["prices"], "features": inputs["features"], "input_sha256": group["shared_input_sha256"],
                         "candidate_sha256": identities[book]["candidate_sha256"], "signal_date": inputs["signal_date"],
                         "previous_session": inputs["previous_session"], "feature_available_at": inputs["feature_available_at"],
                         "source": inputs["source"]}
                opened, recovered = _deciders(root, manifest, objects, role=role, cost_bps=cost)
                receipt = commit_paper_event(state_path=runtime / "books" / f"{book}.json", event=event,
                    initial_state=identities[book], open_decider=opened, recovery_decider=recovered,
                    evidence_writer=_evidence_writer(runtime, manifest, book), now=None if live_clock else observed)
                results[book] = {key: receipt[key] for key in ("status", "event_sha256", "pretrade_equity", "trade", "qlib_evidence")}
            result = base | {"status": "recorded", "session": session, "slot": slot,
                             "shared_input_sha256": group["shared_input_sha256"], "books": results,
                             "forward_complete_sessions": min(len(json.loads((runtime / "books" / f"{book}.json").read_text())["history"]) for book in states)}
            _write(runtime / "latest_status.json", result)
            return result
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        result = base | {"status": "needs_data" if str(error).startswith("needs_data:") else "blocked",
                         "slot": slot, "reason": str(error), "observed_at": observed.isoformat()}
        if runtime is not None:
            _write(runtime / "latest_status.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(root=args.root, config_path=args.config), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
