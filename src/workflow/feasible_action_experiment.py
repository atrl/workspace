"""Feasible-action training domain experiment.

A pre-registered training-domain legality contrast on the uniform price vintage:

- ``control_all_state`` : annual Ridge models fit on every matured own-state row
  (must exactly reproduce the sealed economic control).
- ``candidate_feasible``: annual Ridge models fit only on rows where HOLD was a
  legal decision (``routing_eligible = not hard_risk_violation``).

Baselines are the SAME identities the frozen economic experiment used, replayed
through ``_simulate_on_policy_router``:

- ``no_hold_reference``   : never applies the open HOLD action; always executes the
  legal upstream target (``models={}``).
- ``active_hold_baseline``: the legacy rules-only open policy on ``legacy_models``.
- ``cash_reference``      : an explicitly named all-cash reference (NOT ``no_hold``).

All variants share the frozen coherent input, hyper-parameters and native
prefix/suffix reconciliation. All models, scalers, training rows, legality flags,
maturity, predictions, labels and native artifacts are saved.

The empirical prior that illegal rows dominate the label extremes is NOT supported
(they have lower variance and only ~5% of the top decile); this is a pre-declared
legality contrast, not evidence that filtering removes noise.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.workflow import coherent_price_replay_experiment as C
from src.workflow.coherent_price_replay_experiment import _metrics as simple_metrics
from src.workflow.stage1_open_action_value import (
    Stage1OpenActionValueSpec,
    _metrics as open_metrics,
    _simulate_on_policy_router,
)

EXPERIMENT = "stage1_feasible_action_training_20261009"
SCHEMA_VERSION = "stage1_feasible_action_training_v1"
QLIB_DOCS = C.QLIB_DOCS
SOURCE_CARD = C.SOURCE_CARD
SOURCE_RECORDER = C.SOURCE_RECORDER
SOURCE_OBJECT_SHA256 = C.SOURCE_OBJECT_SHA256
COSTS = (10.0, 25.0)
SEGMENT_LABELS = C.SEGMENT_LABELS
PROVIDERS = {
    "intraday_provider": "data/qlib/stage1_uniform_intraday_20261009",
    "execution_mark_provider": "data/qlib/stage1_uniform_execution_marks_20261009",
    "coherent_provider": "data/qlib/stage1_uniform_daily_marks_20261009",
}
VARIANTS = {"control_all_state": "all", "candidate_feasible": "feasible"}
FROZEN_CONTROL_KEY = C.FROZEN_CONTROL_KEY
MINIMUM_ANNUAL_WEALTH_RATIO = 0.985
OPEN_TRADE_DAYS_REDUCTION = 0.92


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _quiet() -> None:
    C._quiet()


@dataclass(frozen=True)
class Spec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")


def _freeze_providers(root: Path) -> dict:
    frozen = {}
    for name, rel in PROVIDERS.items():
        path = root / rel
        manifest_path = path / "qlib_store_manifest.json"
        _require(manifest_path.is_file(), f"{name} manifest missing at {rel}")
        manifest = json.loads(manifest_path.read_text())
        frozen[name] = {
            "uri": rel, "manifest_sha256": C._sha256(manifest_path),
            "as_of": manifest.get("as_of"), "freq": manifest.get("freq"),
            "purpose": manifest.get("purpose"),
            "complete_session_validation": manifest.get("complete_session_validation"),
            "trading_calendar_sha256": manifest.get("trading_calendar_sha256"),
            "source_manifest_sha256": manifest.get("source_manifest_sha256") or manifest.get("repair_manifest_sha256"),
            "daily_source_manifest_sha256": manifest.get("daily_source_manifest_sha256"),
            "intraday_price_source_from": manifest.get("intraday_price_source_from"),
        }
    return frozen


def _replay_net_value(models, market: pd.DataFrame, *, cost: float, common: dict) -> pd.DataFrame:
    from src.workflow.stage1_economic_training import simulate_economic_policy
    return simulate_economic_policy(market, models=models, policy="net_value", cost_bps=cost, **common)


def _replay_policy_router(market, *, models, cost, common, segment, spec) -> pd.DataFrame:
    return _simulate_on_policy_router(market, models=models, spec=replace(spec, transaction_cost_bps=cost),
                                      segment=segment, **common)


def soxx_buy_hold_benchmark(panel: pd.DataFrame, *, costs_bps=COSTS) -> list[dict]:
    """SOXX buy-and-hold from cash with the SAME fee model (no terminal sell).

    Full SOXX funding on the first open costs ``c`` of the notional, so the
    funded notional is ``1/(1+c)`` and net wealth is ``prod(1+r)/(1+c)``; the
    position is held to the end with no assumed terminal liquidation.
    """
    rows = []
    for segment, span in SEGMENT_LABELS.items():
        band = panel[panel["segment"] == segment]
        r = band["soxx_return"].to_numpy(float)
        _require(np.isfinite(r).all() and (r > -1.0).all(), f"SOXX return non-finite in {span}")
        growth = float(np.prod(1.0 + r))
        wealth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
        entry = {"benchmark": "SOXX_buy_hold_open_to_open", "segment": span, "sessions": int(r.size),
                 "gross_return_pct": 100.0 * (growth - 1.0),
                 "gross_max_drawdown_pct": 100.0 * float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
                 "cost_model": "fund SOXX from cash at the first open, hold to the end, no terminal sell"}
        for cost in costs_bps:
            net = growth / (1.0 + cost / 10_000.0)
            entry[f"net{int(cost)}_return_pct"] = 100.0 * (net - 1.0)
        rows.append(entry)
    return rows


def soxx_buy_hold_path(panel: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """Actual SOXX buy-and-hold path with the SAME fee model as the candidates."""
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    rows = []
    for segment, span in SEGMENT_LABELS.items():
        band = panel[panel["segment"] == segment].sort_values("signal_date")
        previous, wealth = np.array([0.0, 0.0, 1.0]), 1.0
        for position, (_, row) in enumerate(band.iterrows()):
            target = (fee_affordable_target(np.array([0.0, 1.0, 0.0]), pretrade_weights=previous, cost_bps=cost_bps)
                      if position == 0 else None)
            settled = settle_target_session(previous, open_target_weights=target,
                                            asset_returns=[row["soxl_return"], row["soxx_return"], 0.0], cost_bps=cost_bps)
            previous = settled["ending_weights"]
            wealth *= settled["net_equity"]
            rows.append({"segment": span, "cost_bps": cost_bps, "signal_date": row["signal_date"],
                         "trade_date": row["trade_date"], "exit_date": row["exit_date"],
                         "soxl_weight": previous[0], "soxx_weight": previous[1], "cash_weight": previous[2],
                         "turnover": settled["turnover"], "fees": settled["fees"], "net_equity": wealth})
    return pd.DataFrame(rows)


def soxx_50_50_benchmark(panel: pd.DataFrame, *, costs_bps=COSTS) -> list[dict]:
    """Legal 50/50 SOXL/SOXX daily-rebalanced control with real rebalance fees."""
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    rows = []
    for segment, span in SEGMENT_LABELS.items():
        band = panel[panel["segment"] == segment].sort_values("signal_date")
        for cost in costs_bps:
            wealth, peak, previous = 1.0, 1.0, np.array([0.0, 0.0, 1.0])
            max_drawdown = 0.0
            for _, row in band.iterrows():
                target = fee_affordable_target(np.array([0.5, 0.5, 0.0]), pretrade_weights=previous, cost_bps=cost)
                settled = settle_target_session(previous, open_target_weights=target,
                                                asset_returns=[row["soxl_return"], row["soxx_return"], 0.0], cost_bps=cost)
                previous = settled["ending_weights"]
                wealth *= settled["net_equity"]
                peak = max(peak, wealth)
                max_drawdown = min(max_drawdown, wealth / peak - 1.0)
            rows.append({"benchmark": "SOXX_SOXL_50_50_daily_rebalanced", "segment": span, "cost_bps": cost,
                         "net_return_pct": 100.0 * (wealth - 1.0), "net_max_drawdown_pct": 100.0 * max_drawdown})
    return rows


def _annual_rows(records: list[dict], replays: dict, *, cost: float = 10.0) -> list[dict]:
    rows = []
    for key, replay in replays.items():
        variant, span, key_cost = key.split("::")
        if float(key_cost) != cost:
            continue
        if span != SEGMENT_LABELS["development_walk_forward"]:
            continue
        frame = replay.copy()
        frame["year"] = pd.to_datetime(frame["signal_date"]).dt.year
        for year, section in frame.groupby("year"):
            rows.append({"year": int(year), "variant": variant, "segment": span,
                         **open_metrics(section, variant=variant, segment=span, cost_bps=cost)})
    return rows


def _gates(records: list[dict], replays: dict, soxx: list[dict]) -> dict:
    """Full frozen gate set vs no_hold/active; missing reference -> needs_data."""
    lookup = {(r["variant"], r["segment"], r["cost_bps"]): r for r in records}
    annual = pd.DataFrame(_annual_rows(records, replays))
    if annual.empty or "variant" not in annual.columns:
        rows = []
        for reference in ("no_hold_reference", "active_hold_baseline"):
            rows.append({"candidate": "candidate_feasible", "reference": reference,
                         "status": "needs_data", "missing": ["annual_terminal_wealth"], "all_pass": False})
        return {"thresholds": {"annual_wealth_floor": MINIMUM_ANNUAL_WEALTH_RATIO,
                               "open_trade_days_reduction": OPEN_TRADE_DAYS_REDUCTION},
                "rows": rows, "note": "full frozen gate set; missing annual rows -> needs_data"}
    rows = []
    segment = SEGMENT_LABELS["development_walk_forward"]
    for candidate_name in ("candidate_feasible",):
        for reference in ("no_hold_reference", "active_hold_baseline"):
            missing = [k for k in ((candidate_name, segment, 10.0), (candidate_name, segment, 25.0),
                                   (reference, segment, 10.0), (reference, segment, 25.0)) if k not in lookup]
            if missing:
                rows.append({"candidate": candidate_name, "reference": reference, "status": "needs_data",
                             "missing": [str(m) for m in missing], "all_pass": False})
                continue
            candidate, baseline = lookup[(candidate_name, segment, 10.0)], lookup[(reference, segment, 10.0)]
            stress_candidate, stress_baseline = lookup[(candidate_name, segment, 25.0)], lookup[(reference, segment, 25.0)]
            a = annual[(annual.variant == candidate_name)].set_index("year")
            b = annual[(annual.variant == reference)].set_index("year")
            common_years = sorted(set(a.index) & set(b.index))
            if not common_years:
                rows.append({"candidate": candidate_name, "reference": reference, "status": "needs_data",
                             "missing": ["annual_terminal_wealth"], "all_pass": False})
                continue
            annual_ratio = min(float(a.loc[y, "terminal_wealth"] / b.loc[y, "terminal_wealth"])
                               for y in common_years if y >= 2017 and b.loc[y, "terminal_wealth"] != 0)
            checks = {
                "development_return_above_baseline": bool(candidate["return_pct"] > baseline["return_pct"]),
                "stress_25bps_return_above_baseline": bool(stress_candidate["return_pct"] > stress_baseline["return_pct"]),
                "max_drawdown_not_worse": bool(candidate["max_drawdown_pct"] >= baseline["max_drawdown_pct"] - 1e-10),
                "worst_day_not_worse": bool(candidate["worst_day_pct"] >= baseline["worst_day_pct"] - 1e-10),
                "cvar5_not_worse": bool(candidate["cvar5_pct"] >= baseline["cvar5_pct"] - 1e-10),
                "worst_5d_not_worse": bool(candidate["worst_rolling_5d_pct"] >= baseline["worst_rolling_5d_pct"] - 1e-10),
                "open_trade_days_reduced_8pct": bool(candidate["open_trade_days"] <= OPEN_TRADE_DAYS_REDUCTION * baseline["open_trade_days"]),
                "turnover_not_increased": bool(candidate["total_turnover"] <= baseline["total_turnover"]),
                "annual_wealth_floor_pass": bool(annual_ratio >= MINIMUM_ANNUAL_WEALTH_RATIO),
            }
            rows.append({"candidate": candidate_name, "reference": reference, "status": "ok",
                         "minimum_annual_wealth_ratio": annual_ratio, **checks,
                         "all_pass": bool(all(checks.values()))})
    gross_soxx = {row["segment"]: row["gross_return_pct"] for row in soxx}
    for row in rows:
        row.setdefault("gross_soxx_return_pct", gross_soxx.get(segment))
    return {"thresholds": {"annual_wealth_floor": MINIMUM_ANNUAL_WEALTH_RATIO,
                           "open_trade_days_reduction": OPEN_TRADE_DAYS_REDUCTION},
            "rows": rows,
            "note": "full frozen gate set; a missing reference is needs_data, never skipped"}


def run_feasible_action_training(*, root: Path | str | None = None, card_dir: Path | str | None = None) -> dict[str, Any]:
    from src.workflow.feasible_action_training import train_feasible_action_book
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec

    _quiet()
    spec_root = C.resolve_root(root)
    spec = Spec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    target_card.mkdir(parents=True, exist_ok=True)
    plan_path = spec.root / "research" / spec.experiment / "plan.json"
    _require(plan_path.is_file(), "plan.json is missing")

    providers = _freeze_providers(spec.root)
    coherent = (spec.root / PROVIDERS["coherent_provider"]).resolve()
    intraday = (spec.root / PROVIDERS["intraday_provider"]).resolve()
    execution = (spec.root / PROVIDERS["execution_mark_provider"]).resolve()
    for ok, reason, label in (
        (*C._coherent_provider_ready(coherent), "coherent"),
        (*C._intraday_provider_ready(intraday), "intraday"),
        (*C._execution_provider_ready(execution), "execution"),
    ):
        _require(ok, f"{label} provider not ready: {reason}")

    frozen = C.load_frozen(spec.root)
    prepared = frozen["inputs/prepared_inputs.pkl"]
    frame = prepared["frame"].copy()
    common = dict(recovery_models=prepared["recovery_models"],
                  source_spec=IntradayRecoveryDirectAllocationSpec(root=spec.root),
                  selected_model=prepared["selected_model"])
    panel = C.rebuild_market_inputs(frame, coherent_provider=coherent, intraday_provider=intraday).reset_index(drop=True)

    books = {name: train_feasible_action_book(panel, domain=domain, update_through_year=FROZEN_CONTROL_KEY,
                                              cost_bps=10.0, **common)
             for name, domain in VARIANTS.items()}
    control_reproduction = bool(np.allclose(books["control_all_state"]["replay"]["strategy_return"],
                                            books["control_all_state"]["base_replay"]["strategy_return"],
                                            rtol=0, atol=1e-12))
    legacy_identity = _legacy_identity_parity(frame, frozen["analysis/portfolio_replays.pkl"], prepared,
                                              Stage1OpenActionValueSpec(root=spec.root), common)

    old_spec = Stage1OpenActionValueSpec(root=spec.root)
    replays, records = {}, []

    def add(variant, span, cost, replay):
        key = f"{variant}::{span}::{int(cost)}"
        replays[key] = replay
        records.append(open_metrics(replay, variant=variant, segment=span, cost_bps=cost))

    for name, book in books.items():
        for segment, span in SEGMENT_LABELS.items():
            market = panel[panel["segment"] == segment]
            for cost in COSTS:
                add(name, span, cost, _replay_net_value(book["models"], market, cost=cost, common=common))
    for segment, span in SEGMENT_LABELS.items():
        market = panel[panel["segment"] == segment]
        for cost in COSTS:
            add("active_hold_baseline", span, cost,
                _replay_policy_router(market, models=prepared["legacy_models"], cost=cost, common=common,
                                      segment=span, spec=old_spec))
            add("no_hold_reference", span, cost,
                _replay_policy_router(market, models={}, cost=cost, common=common, segment=span, spec=old_spec))
            cash = market[["signal_date", "trade_date", "exit_date"]].copy()
            cash["strategy_return"] = 0.0
            cash["turnover"] = 0.0
            cash["execute_open_trade"] = False
            cash["execute_open_target"] = False
            cash["recovery_action"] = False
            cash["open_soxl_weight"] = 0.0
            cash["open_soxx_weight"] = 0.0
            cash["open_cash_weight"] = 1.0
            cash["soxl_weight"] = 0.0
            cash["soxx_weight"] = 0.0
            cash["cash_weight"] = 1.0
            cash["ending_soxl_weight"] = 0.0
            cash["ending_soxx_weight"] = 0.0
            cash["ending_cash_weight"] = 1.0
            add("cash_reference", span, cost, cash)

    native, native_objects = _native_reconcile_per_key(spec, panel, replays, coherent, execution)
    soxx = soxx_buy_hold_benchmark(panel, costs_bps=COSTS)
    soxx_5050 = soxx_50_50_benchmark(panel, costs_bps=COSTS)
    benchmark_paths = {cost: soxx_buy_hold_path(panel, cost_bps=cost) for cost in COSTS}
    gates = _gates(records, replays, soxx)
    predictions = {name: book["replay"][["signal_date", "segment", "predicted_net_advantage_bps",
                                         "execute_open_target", "decision_reason"]].copy()
                   for name, book in books.items()}

    inventory = {}
    for name, book in books.items():
        models = book["models"]
        inventory[name] = {"domain": VARIANTS[name], "annual_models": int(sum(m is not None for m in models.values())),
                           "years": sorted(int(y) for y in models),
                           "fallback_years": [d["model_year"] for d in book["diagnostics"] if d["fallback"]],
                           "feasible_rows": book["feasible_rows"], "infeasible_rows": book["infeasible_rows"],
                           "diagnostics": book["diagnostics"]}
    inventory["control_reproduction"] = {"variant": "control_all_state", "reproduces_sealed_control": control_reproduction,
                                         "source": "analysis/training_results.pkl", "key": FROZEN_CONTROL_KEY}

    summary = {
        "experiment": spec.experiment, "schema_version": SCHEMA_VERSION, "card_status": "active_research",
        "decision": "feasible_action_domain_contrast_reported",
        "promotion_eligible": False, "objective_achieved": False,
        "plan_sha256": C._sha256(plan_path),
        "plan_registered_at_utc": json.loads(plan_path.read_text()).get("registered_at_utc"),
        "source_card": SOURCE_CARD, "source_recorder_id": SOURCE_RECORDER,
        "source_object_sha256": SOURCE_OBJECT_SHA256,
        "baseline_identities": {
            "no_hold_reference": "open HOLD layer disabled; always executes the legal upstream target (models={})",
            "active_hold_baseline": "legacy rules-only open policy on legacy_models",
            "cash_reference": "explicit all-cash reference, NOT no_hold"},
        "provider_freeze": providers,
        "early_field_repair": panel.attrs.get("early_field_repair"),
        "control_reproduction": control_reproduction,
        "legacy_identity_parity": legacy_identity,
        "model_inventory": inventory,
        "empirical_prior": {
            "feasible_share": books["candidate_feasible"]["feasible_rows"] /
                              (books["candidate_feasible"]["feasible_rows"] + books["candidate_feasible"]["infeasible_rows"]),
            "note": "illegal rows have lower label variance and ~5% of the top decile; the strong "
                    "'illegal rows dominate the extremes' premise is NOT supported",
            "framing": "pre-declared decision-time legality contrast, not proof that filtering removes noise",
        },
        "results": records, "gates": gates, "soxx_benchmark": soxx, "soxx_50_50_benchmark": soxx_5050,
        "soxx_benchmark_actual_net_return_pct": {int(c): (100.0 * (benchmark_paths[c]["net_equity"].iloc[-1] - 1.0)) for c in COSTS},
        "native_reconciliation": native,
        "reporting_caveat": "no stable-excess claim unless the candidate is above gross SOXX",
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": []},
    }
    summary.update(_record(spec=spec, card_dir=target_card, summary=summary, books=books, replays=replays,
                           native_objects=native_objects, predictions=predictions, benchmark_paths=benchmark_paths,
                           plan_path=plan_path))
    _write_card(summary, card_dir=target_card)
    return summary


def _native_reconcile_per_key(spec, panel, replays, coherent, execution) -> tuple[dict, dict]:
    """Reconcile every replay key with its DECLARED target convention.

    Economic-action variants (control/candidate) record fee-affordable
    (``post_fee_desired``) targets and reconcile with the default helper. The
    on-policy reference router records raw projected (``pre_fee_account``)
    targets; the root is extending ``native_intraday_reconciliation`` with a
    ``target_convention`` flag for them, so until that lands those keys are marked
    ``blocked`` with the exact reason and convention, never called unverifiable.
    """
    reference_prefixes = ("active_hold_baseline", "no_hold_reference")
    results, objects = {}, {}
    for key, replay in replays.items():
        convention = "pre_fee_account" if key.startswith(reference_prefixes) else "post_fee_desired"
        try:
            outcome, saved = C._native_reconcile(spec, panel, {key: replay}, coherent, execution,
                                                 target_conventions={key: convention})
            results[key] = {**outcome["results"][key], "target_convention": convention,
                            "commission_per_1": _commission(key)}
            objects.update(saved)
        except ValueError as exc:
            is_reference = key.startswith(reference_prefixes)
            entry = {"status": "blocked" if is_reference else "failed", "reason": str(exc),
                     "target_convention": convention, "commission_per_1": _commission(key)}
            if is_reference:
                entry["pending_interface"] = ("root native_intraday_reconciliation(target_convention='pre_fee_account', "
                                               "freq='day'|'15min') not yet available")
            results[key] = entry
    non_reference_failed = [k for k, v in results.items()
                            if v.get("status") == "failed" and not k.startswith(reference_prefixes)]
    all_passed = all((v.get("suffix", {}).get("status") == "passed") or v.get("status") == "passed" for v in results.values())
    if all_passed:
        status = "passed"
    elif not non_reference_failed:
        status = "passed_with_blocked_references"
    else:
        status = "failed"
    manifest = _read_manifest(coherent)
    return {"status": status, "boundary": manifest.get("intraday_price_source_from"),
            "target_conventions": {"control_all_state": "post_fee_desired", "candidate_feasible": "post_fee_desired",
                                   "active_hold_baseline": "pre_fee_account", "no_hold_reference": "pre_fee_account",
                                   "cash_reference": "trivial_never_trades"},
            "commission_per_1": {int(c): c / 10_000.0 for c in COSTS},
            "results": results,
            "note": "per-key declared target convention; blocked references carry the exact reason, not skipped"}, objects


def _commission(key: str) -> float:
    return float(key.rsplit("::", 1)[1]) / 10_000.0


def _read_manifest(path: Path) -> dict:
    manifest = Path(path) / "qlib_store_manifest.json"
    return json.loads(manifest.read_text()) if manifest.is_file() else {}


def _legacy_identity_parity(frame, source_replays, prepared, spec, common) -> dict:
    """Replay active/no_hold on the ORIGINAL frame and require exact source-17 parity.

    If this does not match per day, the source identity (route/spec/inputs) is
    wrong and the run must block, rather than blaming the uniform vintage.
    """
    checks = []
    for variant, models in (("active_hold_baseline", prepared["legacy_models"]),
                            ("no_hold_reference", {})):
        for segment, span in SEGMENT_LABELS.items():
            market = frame[frame["segment"] == segment]
            for cost in COSTS:
                replay = _replay_policy_router(market, models=models, cost=cost, common=common,
                                               segment=span, spec=spec).reset_index(drop=True)
                source = source_replays[(source_replays["variant"] == variant)
                                        & (source_replays["segment"] == span)
                                        & (source_replays["cost_bps"] == cost)].reset_index(drop=True)
                length_ok = len(replay) == len(source)
                returns_ok = bool(length_ok and np.allclose(replay["strategy_return"], source["strategy_return"],
                                                             rtol=0, atol=1e-12))
                weights_ok = bool(length_ok and all(np.allclose(replay[c], source[c], rtol=0, atol=1e-12)
                                                    for c in ("ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight")))
                checks.append({"variant": variant, "segment": span, "cost_bps": cost, "sessions": int(len(replay)),
                               "returns_ok": returns_ok, "ending_weights_ok": weights_ok})
    passed = all(c["returns_ok"] and c["ending_weights_ok"] for c in checks)
    _require(passed, f"legacy identity parity failed against source 17: {checks}")
    return {"status": "passed", "checks": checks,
            "note": "original frame + same legacy models/spec reproduce source 17 per day; uniform-vintage differences are then input-vintage sensitivity"}


def _record(*, spec, card_dir, summary, books, replays, native_objects, predictions, benchmark_paths, plan_path) -> dict[str, Any]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    _quiet()
    import qlib
    from qlib.workflow import R

    recorder_uri = (Path(card_dir) / "qlib_recorder").resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    saved = {
        "training_rows_control": books["control_all_state"]["training_rows"],
        "training_rows_candidate": books["candidate_feasible"]["training_rows"],
        "diagnostics_control": books["control_all_state"]["diagnostics"],
        "diagnostics_candidate": books["candidate_feasible"]["diagnostics"],
        "replays": replays,
    }
    # Split native artifacts per case so the recorder does not grow one huge object.
    for name, value in native_objects.items():
        saved[f"native::{name}"] = value
    for name, book in books.items():
        for year, model in book["models"].items():
            saved[f"model_{name}_{year}"] = model
    for name, table in predictions.items():
        saved[f"predictions_{name}"] = table
    for cost, path in benchmark_paths.items():
        saved[f"benchmark_soxx_buy_hold_{int(cost)}"] = path
    with R.start(experiment_name=spec.experiment, recorder_name="feasible_action", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(schema_version=SCHEMA_VERSION, source_recorder_id=SOURCE_RECORDER,
                            plan_sha256=summary["plan_sha256"], control_reproduction=summary["control_reproduction"])
        recorder.save_objects(**saved)
        recorder.log_artifact(str(plan_path), artifact_path="plan")
        snapshot = Path(card_dir) / "provider_binding.json"
        snapshot.write_text(json.dumps({"providers": summary["provider_freeze"],
                                        "source_object_sha256": SOURCE_OBJECT_SHA256,
                                        "plan_sha256": summary["plan_sha256"]}, indent=2, default=str), encoding="utf-8")
        recorder.log_artifact(str(snapshot), artifact_path="inputs")
        recorder_id, experiment_id = recorder.id, str(recorder.experiment_id)
        local_dir = recorder_uri / experiment_id / recorder_id
    summary["artifact_policy"]["recorder_only_artifacts"] = sorted(saved)
    return {"recorder_id": recorder_id, "experiment_id": experiment_id, "recorder_uri": str(recorder_uri),
            "recorder_local_dir": str(local_dir), "saved_objects": sorted(saved), "qlib_docs_reference": QLIB_DOCS}


def _write_card(summary: dict, *, card_dir: Path) -> None:
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame(summary["results"]).to_csv(card_dir / "feasible_action_comparison.csv", index=False)
    pd.DataFrame(summary["gates"]["rows"]).to_csv(card_dir / "gate_results.csv", index=False)
    pd.DataFrame(summary["soxx_benchmark"]).to_csv(card_dir / "soxx_benchmark.csv", index=False)
    pd.DataFrame(summary["soxx_50_50_benchmark"]).to_csv(card_dir / "soxx_50_50_benchmark.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "plan_sha256": summary["plan_sha256"], "source_recorder_id": SOURCE_RECORDER,
        "providers": summary["provider_freeze"], "variants": VARIANTS, "costs_bps": list(COSTS),
        "baseline_identities": summary["baseline_identities"],
        "control_reproduction": summary["control_reproduction"],
        "native_reconciliation_status": (summary.get("native_reconciliation") or {}).get("status"),
        "native_target_conventions": (summary.get("native_reconciliation") or {}).get("target_conventions"),
        "native_commission_per_1": (summary.get("native_reconciliation") or {}).get("commission_per_1"),
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    lines = [
        "# Stage1 feasible-action 训练域对照（中文结论）\n",
        "- 做了什么：同一统一 vintage 冻结 coherent 输入上，1 个预声明候选（feasible 域）与 all-state 对照；"
        "超参、label fee、minrows、硬修复规则全不变；不调参、不搜阈值。\n",
        f"- plan sha `{summary['plan_sha256'][:12]}`；控制复现（filter=all == 封存 control）：{summary['control_reproduction']}。\n",
        f"- legacy identity parity（original frame + same legacy models/spec vs 源 17，逐日）：{summary.get('legacy_identity_parity', {}).get('status')}。"
        "若一致，则 uniform 与源 17 的 active/no_hold 差异来自**输入 vintage 敏感性**（legacy state feedback），"
        "不能笼统说“价格影响近零”：economic control 在同价路径变化小只是其特性，不等于所有策略路径不敏感。\n",
        "- baseline 身份：`no_hold_reference`=禁用开盘 HOLD 动作层、始终执行合法上游目标；"
        "`active_hold_baseline`=legacy rules-only；`cash_reference`=显式全现金（**不是** no_hold）。\n",
        f"- 前提校正：实证 feasible 比例 {summary['empirical_prior']['feasible_share']:.4f}，非法行 label 方差更低、"
        "极端 top-decile 仅约 5% —— “非法样本主导极值”的强假设**不成立**；本卡是预声明合法性对照，"
        "**不能**把过滤说成已知解决噪声。\n",
        "### 结果（按 segment × cost 分组）\n",
    ]
    for r in summary["results"]:
        lines.append(f"- {r['variant']} {r['segment']} @{r['cost_bps']:g}bp：{r['return_pct']:.2f}% / "
                     f"MaxDD {r['max_drawdown_pct']:.2f}% / CVaR {r['cvar5_pct']:.2f}%\n")
    lines.append("### SOXX buy-hold（真实成本，无终值假设卖出）\n")
    for row in summary["soxx_benchmark"]:
        lines.append(f"- {row['segment']}：gross {row['gross_return_pct']:.2f}% / "
                     f"net10 {row['net10_return_pct']:.2f}% / net25 {row['net25_return_pct']:.2f}%\n")
    lines.append("### 50:50 实际再平衡费用对照\n")
    for row in summary["soxx_50_50_benchmark"]:
        lines.append(f"- {row['segment']} @{row['cost_bps']:g}bp：net {row['net_return_pct']:.2f}%\n")
    native = summary.get("native_reconciliation") or {}
    lines.append(f"- native：status=`{native.get('status')}`，boundary=`{native.get('boundary')}`；"
                 f"target convention：{native.get('target_conventions')}；commission/1：{native.get('commission_per_1')}。"
                 "control/candidate 默认 postFee 完整对账；active/no_hold 为 pre_fee_account，"
                 "待根接口 `target_convention` 就绪后完整复跑，现标 blocked 并保留精确原因（非不可验证）。\n")
    lines.append("- 报告纪律：未超过 gross SOXX 不宣称稳定超额；失败项如实保留。\n")
    if summary.get("recorder_id"):
        lines.append(f"- 子 Recorder：{summary['recorder_id']}；保存全部 annual models/scaler/legality flag/maturity/"
                     "predictions/labels/native reports/positions。\n")
    return "".join(lines)
