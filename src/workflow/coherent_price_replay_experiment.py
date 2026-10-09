"""Increment 1: coherent-price replay, separating price effect from training effect.

Two comparisons, never merged:

- **Price effect**: keep the frozen base forecast/selection and the frozen control
  action model, and replay them on the NEW coherent price panel (real 15-minute
  marks once that source exists, daily prefix before it). Only prices change.
- **Training effect**: keep the coherent panel fixed and train the pre-registered
  cost-matched action book on it.

The 2016 prefix is reconciled with the open-only native helper; the intraday
suffix (with recovery) is reconciled with the intraday native helper, continuing
the prefix's settled cash and quantities. The card fails closed with
``needs_data`` when the coherent/execution providers are absent; a missing
complete native reconciliation is recorded, never faked.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT = "stage1_coherent_price_replay_20261009"
SCHEMA_VERSION = "stage1_coherent_price_replay_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
SOURCE_CARD = "research/stage1_economic_action_20260929"
SOURCE_RECORDER = "17f81fbf27ee46859816ec1ac52151f6"
SOURCE_OBJECT_SHA256 = {
    "inputs/prepared_inputs.pkl": "8baa4f74449c1da41eb189f51c75074a61dacb8d7c8540710f4f67560d58d4aa",
    "analysis/portfolio_replays.pkl": "ac7281e80aa78d2cc69c9c97dadd51356ac883e85ee87d53ca46eaa976b4b306",
    "analysis/training_results.pkl": "a0ab9916a62cc2a00af89876b423d5cb419eec64fa7418abcefbc9b11d728e4b",
    "analysis/annual.pkl": "b8f7423783600612c6b696decb73c15abea604ce9a494242c94a84ff8be733cc",
}
FROZEN_CONTROL_KEY = 2022
COSTS = (10.0, 25.0)
SEGMENT_LABELS = {"development_walk_forward": "development_walk_forward",
                  "outer_diagnostic": "outer_reused_diagnostic"}
PARITY_TOLERANCE = 1e-7


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class CoherentSpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")
    intraday_provider: str = "data/qlib/stage1_repaired_intraday_20261009"
    coherent_provider: str = "data/qlib/stage1_coherent_daily_marks_20261009"
    execution_mark_provider: str = "data/qlib/stage1_execution_marks_20261009"


def resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quiet() -> None:
    import logging
    import warnings
    for name in ("qlib", "mlflow", "qlib.data", "qlib.workflow"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*future calendar.*")
    warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")


def provider_ready(path: Path, *, freq: str = "day") -> bool:
    return (path / "calendars" / f"{freq}.txt").is_file() and (path / "features").is_dir()


def _read_manifest(path: Path) -> dict:
    manifest = path / "qlib_store_manifest.json"
    if not manifest.is_file():
        return {}
    try:
        return json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return {}


def _coherent_provider_ready(path: Path) -> tuple[bool, str]:
    """A coherent provider is only usable when the root builder stamped it complete.

    Existence of a calendar and some bars is deliberately NOT enough: the builder
    fails closed on missing sessions / missing first bars and records the native
    aggregation provenance. Without that stamp this returns False so the run
    cannot silently aggregate only the first observed bars into a fake full day.
    """
    if not provider_ready(path, freq="day"):
        return False, "coherent_provider_absent"
    manifest = _read_manifest(path)
    if "intraday_price_source_from" not in manifest or "daily_source_manifest_sha256" not in manifest:
        return False, "coherent_provider_manifest_missing_builder_stamp"
    if "no price scaling" not in manifest.get("aggregation", ""):
        return False, "coherent_provider_aggregation_rule_not_verified"
    return True, "ready"


def _intraday_provider_ready(path: Path) -> tuple[bool, str]:
    if not provider_ready(path, freq="15min"):
        return False, "intraday_provider_absent"
    manifest = _read_manifest(path)
    if manifest.get("complete_session_validation") is not True:
        return False, "intraday_provider_complete_session_validation_missing"
    if not manifest.get("trading_calendar_sha256"):
        return False, "intraday_provider_missing_trading_calendar_sha256"
    if not (manifest.get("repair_manifest_sha256") or manifest.get("source_manifest_sha256")):
        return False, "intraday_provider_missing_source_manifest_sha256"
    return True, "ready"


def _execution_provider_ready(path: Path) -> tuple[bool, str]:
    if not provider_ready(path, freq="15min"):
        return False, "execution_provider_absent"
    manifest = _read_manifest(path)
    if not manifest.get("source_ohlcv_unchanged") or "execution_price_rule" not in manifest:
        return False, "execution_provider_manifest_missing_builder_stamp"
    return True, "ready"


def load_frozen(root: Path) -> dict:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    _quiet()
    import qlib
    from qlib.workflow import R

    base = (root / SOURCE_CARD / "qlib_recorder").resolve()
    _require(base.is_dir(), "frozen economic recorder is unavailable")
    qlib.init(provider_uri=str((root / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    objects = {}
    with R.uri_context(str(base)):
        recorder = R.get_recorder(recorder_id=SOURCE_RECORDER, experiment_name=Path(SOURCE_CARD).name)
        local = Path(recorder.get_local_dir()) / "artifacts"
        for name, expected in SOURCE_OBJECT_SHA256.items():
            actual = _sha256(local / name)
            _require(actual == expected, f"frozen object hash mismatch for {name}")
            objects[name] = recorder.load_object(name)
    return objects


def _metrics(replay: pd.DataFrame) -> dict:
    r = replay["strategy_return"].to_numpy(float)
    growth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(growth)
    tail = np.sort(r)[:max(1, int(np.ceil(0.05 * r.size)))]
    rolling5 = pd.Series(r).rolling(5).apply(lambda x: float(np.prod(1.0 + x) - 1.0), raw=True)
    std = r.std(ddof=1)
    return {
        "sessions": int(r.size), "terminal_wealth": float(growth[-1]),
        "return_pct": 100.0 * float(growth[-1] - 1.0),
        "max_drawdown_pct": 100.0 * float((growth / peak - 1.0).min()),
        "cvar5_pct": 100.0 * float(tail.mean()),
        "worst_day_pct": 100.0 * float(r.min()),
        "worst_rolling_5d_pct": 100.0 * float(rolling5.min()),
        "sharpe": float(r.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "total_turnover": float(replay["turnover"].sum()),
        "open_trade_days": int(replay["execute_open_trade"].sum()),
    }


def rebuild_daily_returns(frame: pd.DataFrame, *, provider_uri: Path) -> pd.DataFrame:
    """Replace daily return columns from the coherent daily provider's day-freq opens.

    Reading the exit open from the same coherent daily source keeps the last
    prefix session's ``exit_open`` identical to the first suffix session's open,
    so the prefix-to-suffix share value cannot jump.
    """
    import qlib
    from qlib.data import D
    from src.analysis.stage1_regime_validation import SYMBOLS

    sessions = frame[["trade_date", "exit_date"]]
    trade_dates, exit_dates = pd.to_datetime(sessions.trade_date), pd.to_datetime(sessions.exit_date)
    _require(trade_dates.notna().all() and exit_dates.notna().all(), "coherent rebuild requires finite session dates")
    qlib.init(provider_uri=str(Path(provider_uri).resolve()), region="us", kernels=1)
    prices = D.features(list(SYMBOLS), ["$open"], start_time=str(trade_dates.min().date()),
                        end_time=str(exit_dates.max().date()), freq="day")
    opens = prices["$open"].unstack("instrument").astype(float)
    out = frame.copy()
    for symbol in SYMBOLS:
        entry = opens[symbol].reindex(trade_dates).to_numpy(float)
        exit_ = opens[symbol].reindex(exit_dates).to_numpy(float)
        _require(np.isfinite(entry).all() and np.isfinite(exit_).all() and (entry > 0).all() and (exit_ > 0).all(),
                 f"coherent daily provider omitted {symbol} opening marks")
        out[f"{symbol.lower()}_return"] = exit_ / entry - 1.0
    return out


def rebuild_early_fields(frame: pd.DataFrame, *, provider_uri: Path) -> pd.DataFrame:
    """Rebuild EVERY first-hour feature from one provider as a single source.

    This writes all of ``early_ret``, ``early_range``, ``early_from_low``,
    ``early_volume`` and ``early_bar_count`` for both symbols from the repaired
    intraday provider; it never leaves stale ``range``/``volume``/``high`` values
    from the old source mixed in. The signal columns and model identity are not
    touched. Fail-closed if any session lacks the exact first-hour bar stamps.
    """
    from src.analysis.stage1_regime_validation import (_read_first_hour_bars, BAR_FIELDS,
                                                       EARLY_FIELDS, FIRST_HOUR_CLOSES, SYMBOLS)

    dates = pd.to_datetime(frame["trade_date"])
    _require(dates.notna().all() and not dates.duplicated().any(),
             "early rebuild requires unique finite trade dates")
    bars = _read_first_hour_bars(provider_uri, start=str(dates.min().date()), end=str(dates.max().date()))
    bars["datetime"] = pd.to_datetime(bars.datetime)
    if bars.datetime.dt.tz is not None:
        bars["datetime"] = bars.datetime.dt.tz_convert("America/New_York").dt.tz_localize(None)
    bars["date"] = bars.datetime.dt.normalize()
    bars = bars.sort_values(["instrument", "date", "datetime"])
    first_four = bars.groupby(["instrument", "date"], group_keys=False).head(4)
    # Preserve float32 aggregation semantics exactly, as the retained features do.
    recomputed = first_four.groupby(["instrument", "date"]).agg(
        early_open=("$open", "first"), early_close=("$close", "last"),
        early_high=("$high", "max"), early_low=("$low", "min"),
        early_volume=("$volume", "sum"), early_bar_count=("$bar_count", "sum"),
    )
    recomputed["early_ret"] = recomputed.early_close / recomputed.early_open - 1.0
    recomputed["early_range"] = recomputed.early_low / recomputed.early_high - 1.0
    recomputed["early_from_low"] = recomputed.early_close / recomputed.early_low - 1.0

    days = pd.DatetimeIndex(dates.dt.normalize().unique()).sort_values()
    covered = pd.DatetimeIndex(sorted(pd.DatetimeIndex(bars["date"]).unique()))
    uncovered = days.difference(covered)
    # The repaired provider may precede the frame as a prefix (no repair there,
    # and no recovery), but it must never leave an internal gap.
    if len(uncovered):
        _require(len(covered) and uncovered.max() < covered.min(),
                 "repaired intraday provider has an internal coverage gap")
    index = pd.MultiIndex.from_product([list(SYMBOLS), covered], names=["instrument", "date"])
    stamps = first_four.groupby(["instrument", "date"]).datetime.apply(
        lambda s: tuple(s.dt.strftime("%H:%M"))).reindex(index)
    _require(stamps.notna().all(), "repaired intraday provider omitted a covered session")
    _require(all(value == FIRST_HOUR_CLOSES for value in stamps), "repaired first-hour stamps are not the exact close stamps")
    recomputed = recomputed.reindex(index)
    _require(recomputed.notna().all().all(), "recomputed first-hour features are incomplete")
    prices = first_four[["$open", "$high", "$low", "$close"]].to_numpy(float)
    _require(np.isfinite(prices).all() and (prices > 0).all() and (first_four["$volume"] >= 0).all(),
             "invalid repaired first-hour prices or volume")

    out = frame.copy()
    normalized = dates.dt.normalize()
    mask = normalized.isin(covered).to_numpy()
    for symbol in SYMBOLS:
        suffix = symbol.lower()
        block = recomputed.loc[symbol]
        for name in EARLY_FIELDS:
            column = out[f"{name}_{suffix}"].to_numpy(float)
            column[mask] = normalized.map(block[name]).to_numpy(float)[mask]
            out[f"{name}_{suffix}"] = column
    out.attrs["early_field_repair"] = {
        "source": str(Path(provider_uri).resolve()), "repaired_sessions": int(len(covered)),
        "repaired_from": str(covered.min().date()), "prior_vintage_prefix_sessions": int(len(uncovered)),
        "fields": list(EARLY_FIELDS), "symbols": list(SYMBOLS),
        "note": "all first-hour fields are rebuilt together from one repaired source in the covered region; the unmatched prefix carries no recovery",
    }
    return out


def rebuild_market_inputs(frame: pd.DataFrame, *, coherent_provider: Path, intraday_provider: Path) -> pd.DataFrame:
    """Unified repair: daily returns (coherent provider) + ALL first-hour fields (intraday provider)."""
    out = rebuild_daily_returns(frame, provider_uri=coherent_provider)
    return rebuild_early_fields(out, provider_uri=intraday_provider)


def run_coherent_price_replay(*, root: Path | str | None = None, card_dir: Path | str | None = None,
                              coherent_provider: str | None = None,
                              execution_mark_provider: str | None = None,
                              intraday_provider: str | None = None) -> dict[str, Any]:
    _quiet()
    spec_root = resolve_root(root)
    spec = CoherentSpec(root=spec_root,
                        coherent_provider=coherent_provider or CoherentSpec.coherent_provider,
                        execution_mark_provider=execution_mark_provider or CoherentSpec.execution_mark_provider,
                        intraday_provider=intraday_provider or CoherentSpec.intraday_provider)
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    target_card.mkdir(parents=True, exist_ok=True)
    coherent = (spec.root / spec.coherent_provider).resolve()
    execution = (spec.root / spec.execution_mark_provider).resolve()
    intraday = (spec.root / spec.intraday_provider).resolve()
    plan_path = spec.root / "research" / spec.experiment / "plan.json"

    coherent_ok, coherent_reason = _coherent_provider_ready(coherent)
    execution_ok, execution_reason = _execution_provider_ready(execution)
    intraday_ok, intraday_reason = _intraday_provider_ready(intraday)
    if not (coherent_ok and execution_ok and intraday_ok):
        summary = _needs_data(spec, coherent, execution, intraday, plan_path,
                              coherent_reason=coherent_reason, execution_reason=execution_reason,
                              intraday_reason=intraday_reason)
        _write_card(summary, card_dir=target_card)
        return summary

    from src.workflow.economic_cost_matched_training import train_cost_matched_book

    frozen = load_frozen(spec.root)
    prepared = frozen["inputs/prepared_inputs.pkl"]
    frame = prepared["frame"].copy()
    common = dict(recovery_models=prepared["recovery_models"],
                  source_spec=_source_spec(spec.root), selected_model=prepared["selected_model"])

    panel = rebuild_market_inputs(frame, coherent_provider=coherent, intraday_provider=intraday)

    # Price-effect control: the SAVED annual models from the parent recorder are read
    # back directly (hash-bound, no refit). ``training_results[2022]["models"]`` holds
    # the frozen 2016..2023 annual models the source variant already used, so this is
    # the real frozen identity, not a fresh fit passed off as frozen.
    training_results = frozen["analysis/training_results.pkl"]
    _require(FROZEN_CONTROL_KEY in training_results, "frozen control {0} missing".format(FROZEN_CONTROL_KEY))
    control_models = training_results[FROZEN_CONTROL_KEY]["models"]
    _require(isinstance(control_models, dict) and len(control_models) >= 6,
             "frozen control annual models are incomplete")
    # The hash-bound frozen control must reproduce the parent replay on the OLD
    # frame before it is allowed to stand as the price-effect control.
    control_check = _control_parity(frame, frozen["analysis/portfolio_replays.pkl"], control_models, common)
    # Training-effect candidate: cost-matched books trained on the coherent panel.
    books = {cost: train_cost_matched_book(panel, label_cost_bps=cost, evaluation_cost_bps=cost,
                                           update_through_year=FROZEN_CONTROL_KEY, **common) for cost in COSTS}

    records, replays = [], {}
    for segment, span in SEGMENT_LABELS.items():
        market = panel[panel["segment"] == segment]
        for cost in COSTS:
            control = _replay(control_models, market, cost=cost, common=common)
            candidate = _replay(books[cost]["models"], market, cost=cost, common=common)
            replays[f"control_price::{span}::{int(cost)}"] = control
            replays[f"candidate_training::{span}::{int(cost)}"] = candidate
            records.append({"family": "price_effect_control", "segment": span, "cost_bps": cost, **_metrics(control)})
            records.append({"family": "training_effect_candidate", "segment": span, "cost_bps": cost, **_metrics(candidate)})

    native, native_objects = _native_reconcile(spec, panel, replays, coherent, execution)
    price_delta = _price_effect_delta(records, replays, frozen["analysis/portfolio_replays.pkl"])
    soxx = _soxx_benchmark(panel)
    gates = _gates(records)
    summary = {
        "experiment": spec.experiment, "schema_version": SCHEMA_VERSION, "card_status": "active_research",
        "increment": "increment_1_price_consistency",
        "decision": "coherent_price_replay_reported",
        "promotion_eligible": False, "objective_achieved": False,
        "plan_sha256": _sha256(plan_path) if plan_path.is_file() else None,
        "source_card": SOURCE_CARD, "source_recorder_id": SOURCE_RECORDER,
        "source_object_sha256": SOURCE_OBJECT_SHA256,
        "frozen_control": {"object": "analysis/training_results.pkl", "key": FROZEN_CONTROL_KEY,
                           "control_parity": control_check},
        "coherent_provider": str(coherent.relative_to(spec.root)),
        "execution_mark_provider": str(execution.relative_to(spec.root)),
        "intraday_provider": str(intraday.relative_to(spec.root)),
        "separation_rule": "price effect and training effect are reported as separate comparisons, never merged",
        "results": records, "native_reconciliation": native,
        "price_effect_delta_vs_old_daily": price_delta,
        "soxx_benchmark": soxx, "gates": gates, "early_field_repair": panel.attrs.get("early_field_repair"),
        "limitations": [
            "Historical mark convention only; not real 09:31/10:31 broker fills.",
            "Diagnostic only; produces no promotion.",
            "Reused history: outer is not a clean holdout.",
        ],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": []},
    }
    summary.update(_record(spec=spec, card_dir=target_card, summary=summary, replays=replays,
                           native_objects=native_objects, plan_path=plan_path))
    _write_card(summary, card_dir=target_card)
    return summary


def _control_parity(frame, source_replays, control_models, common) -> dict:
    """The saved frozen annual models must reproduce the parent control on old data."""
    from src.workflow.stage1_economic_training import simulate_economic_policy

    checks = []
    for segment, span in SEGMENT_LABELS.items():
        market = frame[frame["segment"] == segment]
        for cost in COSTS:
            replay = simulate_economic_policy(market, models=control_models, policy="net_value",
                                              cost_bps=cost, **common).reset_index(drop=True)
            source = source_replays[(source_replays["variant"] == "own_state_net_value_frozen_2022")
                                    & (source_replays["segment"] == span)
                                    & (source_replays["cost_bps"] == cost)].reset_index(drop=True)
            _require(len(replay) == len(source), f"control parity length mismatch {span} @{cost:g}bp")
            returns_ok = bool(np.allclose(replay["strategy_return"], source["strategy_return"], rtol=0, atol=1e-12))
            weights_ok = bool(all(np.allclose(replay[c], source[c], rtol=0, atol=1e-12)
                                  for c in ("ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight")))
            checks.append({"segment": span, "cost_bps": cost, "sessions": int(len(replay)),
                           "returns_ok": returns_ok, "ending_weights_ok": weights_ok})
    _require(all(c["returns_ok"] and c["ending_weights_ok"] for c in checks),
             f"frozen control parity failed: {checks}")
    return {"status": "passed", "checks": checks}


def _price_effect_delta(records: list[dict], replays: dict, source_replays: pd.DataFrame) -> list[dict]:
    """Coherent price-effect control versus the EXACT old daily frozen control.

    Old metrics are computed from the parent ``analysis/portfolio_replays.pkl``
    object for the same segment x cost keys; a missing key blocks instead of being
    skipped. This isolates the effect of the price panel on the same frozen model.
    """
    metrics = {(r["segment"], r["cost_bps"]): r for r in records if r["family"] == "price_effect_control"}
    rows = []
    for (segment, cost), coherent in sorted(metrics.items()):
        old = source_replays[(source_replays["variant"] == "own_state_net_value_frozen_2022")
                             & (source_replays["segment"] == segment) & (source_replays["cost_bps"] == cost)]
        _require(not old.empty, f"missing exact old control for {segment} @{cost:g}bp")
        legacy = _metrics(old)
        rows.append({"segment": segment, "cost_bps": cost,
                     "coherent_return_pct": coherent["return_pct"], "old_daily_return_pct": legacy["return_pct"],
                     "return_delta_pct": coherent["return_pct"] - legacy["return_pct"],
                     "coherent_max_drawdown_pct": coherent["max_drawdown_pct"],
                     "old_daily_max_drawdown_pct": legacy["max_drawdown_pct"],
                     "max_drawdown_delta_pct": coherent["max_drawdown_pct"] - legacy["max_drawdown_pct"],
                     "coherent_cvar5_pct": coherent["cvar5_pct"], "old_daily_cvar5_pct": legacy["cvar5_pct"],
                     "old_control_variant": "own_state_net_value_frozen_2022"})
    return rows


def soxx_buy_hold_benchmark(panel: pd.DataFrame, *, costs_bps=COSTS) -> list[dict]:
    """SOXX buy-and-hold from cash with REAL costs (no fictional fee-free holding).

    Buys at the segment's first open and liquidates at the last exit open, so the
    entry and exit costs are charged: net wealth = (1-c)^2 * prod(1+r).
    """
    rows = []
    for segment, span in SEGMENT_LABELS.items():
        band = panel[panel["segment"] == segment]
        r = band["soxx_return"].to_numpy(float)
        _require(np.isfinite(r).all() and (r > -1.0).all(), f"SOXX return non-finite in {span}")
        growth = float(np.prod(1.0 + r))
        peak = np.maximum.accumulate(np.concatenate([[1.0], np.cumprod(1.0 + r)]))
        entry = {"benchmark": "SOXX_buy_hold_open_to_open", "segment": span, "sessions": int(r.size),
                 "gross_return_pct": 100.0 * (growth - 1.0),
                 "gross_max_drawdown_pct": 100.0 * float((np.concatenate([[1.0], np.cumprod(1.0 + r)]) / peak - 1.0).min()),
                 "cost_model": "buy at first open and sell at last exit open; no fee-free holding"}
        for cost in costs_bps:
            net = (1.0 - cost / 10_000.0) ** 2 * growth
            entry[f"net{int(cost)}_return_pct"] = 100.0 * (net - 1.0)
        rows.append(entry)
    return rows


def _soxx_benchmark(panel: pd.DataFrame) -> list[dict]:
    return soxx_buy_hold_benchmark(panel, costs_bps=COSTS)


def _gates(records: list[dict]) -> dict:
    """Candidate versus the coherent price-effect control, per segment x cost."""
    control = {(r["segment"], r["cost_bps"]): r for r in records if r["family"] == "price_effect_control"}
    rows = []
    for r in records:
        if r["family"] != "training_effect_candidate":
            continue
        base = control[(r["segment"], r["cost_bps"])]
        checks = {
            "return_above_control": bool(r["return_pct"] > base["return_pct"]),
            "max_drawdown_not_worse": bool(r["max_drawdown_pct"] >= base["max_drawdown_pct"] - 1e-10),
            "worst_day_not_worse": bool(r["worst_day_pct"] >= base["worst_day_pct"] - 1e-10),
            "cvar5_not_worse": bool(r["cvar5_pct"] >= base["cvar5_pct"] - 1e-10),
            "worst_5d_not_worse": bool(r["worst_rolling_5d_pct"] >= base["worst_rolling_5d_pct"] - 1e-10),
        }
        rows.append({"candidate": r["family"], "segment": r["segment"], "cost_bps": r["cost_bps"],
                     "reference": "price_effect_control", **checks, "all_pass": bool(all(checks.values()))})
    return {"rows": rows, "note": "candidate must beat the same-price frozen control before any promotion"}


def _source_spec(root: Path):
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    return IntradayRecoveryDirectAllocationSpec(root=root)


def _replay(models, market: pd.DataFrame, *, cost: float, common: dict) -> pd.DataFrame:
    from src.workflow.stage1_economic_training import simulate_economic_policy
    return simulate_economic_policy(market, models=models, policy="net_value", cost_bps=cost, **common)


def _native_reconcile(spec, panel, replays, coherent, execution, *, target_conventions=None):
    """Prefix (open-only) then suffix (intraday) native reconciliation, chained.

    The prefix settles from cash at the coherent daily opens; its final native
    cash and quantities are valued at the boundary day's native opening marks and
    handed to the suffix as inherited ``initial_weights``/``account``. The prefix
    selection is carried, not independently re-validated, and the audit marks it.
    """
    import qlib
    from qlib.data import D
    from src.analysis.native_intraday_schedule import native_intraday_reconciliation
    from src.analysis.stage1_regime_validation import SYMBOLS

    manifest = _read_manifest(coherent)
    if not manifest.get("intraday_price_source_from"):
        return {"status": "blocked_native_reconciliation_missing",
                "reason": "coherent provider does not declare the intraday boundary"}, {}
    boundary = pd.Timestamp(manifest["intraday_price_source_from"])
    qlib.init(provider_uri=str(Path(coherent).resolve()), region="us", kernels=1)
    first_exit = pd.to_datetime(panel["exit_date"]).max()
    opens = D.features(list(SYMBOLS), ["$open"], start_time=str(pd.to_datetime(panel["trade_date"]).min().date()),
                       end_time=str(first_exit.date()), freq="day")["$open"].unstack("instrument").astype(float)
    results, native_objects = {}, {}
    for key, replay in replays.items():
        family, span, cost = key.split("::")
        cost_bps = float(cost)
        convention = (target_conventions or {}).get(key, "post_fee_desired")
        segment = "development_walk_forward" if span == SEGMENT_LABELS["development_walk_forward"] else "outer_diagnostic"
        market = panel[panel["segment"] == segment].reset_index(drop=True)
        schedule = replay.reset_index(drop=True)
        _require(len(market) == len(schedule), f"{key}: native frame/schedule length mismatch")
        prefix_mask = pd.to_datetime(schedule["trade_date"]) < boundary
        entry = {"family": family, "segment": span, "cost_bps": cost_bps, "target_convention": convention}
        prefix_nav, initial = 100_000.0, None
        if prefix_mask.any():
            pf = market[prefix_mask].reset_index(drop=True)
            pr = schedule[prefix_mask].reset_index(drop=True)
            prefix = native_intraday_reconciliation(pf, pr, provider_uri=coherent, cost_bps=cost_bps,
                                                    account=100_000.0, freq="day", target_convention=convention)
            entry["prefix"] = prefix["audit"]
            last, exit_day = pr["trade_date"].iloc[-1], pr["exit_date"].iloc[-1]
            position = prefix["positions"][last]
            cash = float(position.get_cash())
            quantities = position.get_stock_amount_dict()
            values = np.array([quantities.get(s, 0.0) * float(opens.loc[exit_day, s]) for s in SYMBOLS] + [cash])
            prefix_nav = float(values.sum())
            _require(prefix_nav > 0 and np.isfinite(prefix_nav), f"{key}: invalid prefix terminal wealth")
            initial = values / prefix_nav
            native_objects[f"{key}::prefix_report"] = prefix["report"]
            native_objects[f"{key}::prefix_positions"] = prefix["positions"]
            native_objects[f"{key}::prefix_reconciliation"] = prefix["reconciliation"]
        sf = market[~prefix_mask].reset_index(drop=True)
        sr = schedule[~prefix_mask].reset_index(drop=True)
        suffix = native_intraday_reconciliation(sf, sr, provider_uri=execution, cost_bps=cost_bps,
                                                account=prefix_nav, initial_weights=initial,
                                                freq="15min", target_convention=convention)
        entry["suffix"] = suffix["audit"]
        entry["prefix_inherited"] = initial is not None
        native_objects[f"{key}::suffix_report"] = suffix["report"]
        native_objects[f"{key}::suffix_positions"] = suffix["positions"]
        native_objects[f"{key}::suffix_reconciliation"] = suffix["reconciliation"]
        native_objects[f"{key}::suffix_events"] = suffix["events"]
        results[key] = entry
    return {"status": "passed", "boundary": str(boundary.date()), "results": results,
            "note": "prefix day + suffix 15min with per-case target convention; cross-frequency cash/quantity handoff at boundary opens"}, native_objects


def _needs_data(spec, coherent, execution, intraday, plan_path, *, coherent_reason, execution_reason, intraday_reason) -> dict:
    return {
        "experiment": spec.experiment, "schema_version": SCHEMA_VERSION, "card_status": "active_research",
        "increment": "increment_1_price_consistency",
        "decision": "needs_data_coherent_provider_missing",
        "promotion_eligible": False, "objective_achieved": False,
        "plan_sha256": _sha256(plan_path) if plan_path.is_file() else None,
        "coherent_provider": str(coherent.relative_to(spec.root)),
        "execution_mark_provider": str(execution.relative_to(spec.root)),
        "intraday_provider": str(intraday.relative_to(spec.root)),
        "provider_ready": {"coherent": coherent_reason, "execution": execution_reason, "intraday": intraday_reason},
        "reason": ("coherent/execution/intraday mark providers are not all fully built/verified yet; run fails closed "
                   "and does not aggregate partial bars into a full day"),
        "results": [],
        "limitations": ["Needs the native coherent/execution providers; no fallback to a different price source."],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": []},
    }


def _record(*, spec, card_dir, summary, replays, native_objects, plan_path) -> dict[str, Any]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    _quiet()
    import qlib
    from qlib.workflow import R

    recorder_uri = (Path(card_dir) / "qlib_recorder").resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    saved = {"replays": replays, "native": native_objects}
    with R.start(experiment_name=spec.experiment, recorder_name="coherent", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(schema_version=SCHEMA_VERSION, source_recorder_id=SOURCE_RECORDER,
                            coherent_provider=summary["coherent_provider"],
                            execution_mark_provider=summary["execution_mark_provider"])
        recorder.save_objects(**saved)
        if plan_path.is_file():
            recorder.log_artifact(str(plan_path), artifact_path="plan")
        recorder_id, experiment_id = recorder.id, str(recorder.experiment_id)
        local_dir = recorder_uri / experiment_id / recorder_id
    summary["artifact_policy"]["recorder_only_artifacts"] = sorted(saved)
    return {"recorder_id": recorder_id, "experiment_id": experiment_id, "recorder_uri": str(recorder_uri),
            "recorder_local_dir": str(local_dir), "saved_objects": sorted(saved), "qlib_docs_reference": QLIB_DOCS}


def _write_card(summary: dict, *, card_dir: Path) -> None:
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if summary["results"]:
        pd.DataFrame(summary["results"]).to_csv(card_dir / "coherent_replay_comparison.csv", index=False)
    for key, name in (("soxx_benchmark", "soxx_benchmark.csv"), ("price_effect_delta_vs_old_daily", "price_effect_delta.csv")):
        if summary.get(key):
            pd.DataFrame(summary[key]).to_csv(card_dir / name, index=False)
    if summary.get("gates", {}).get("rows"):
        pd.DataFrame(summary["gates"]["rows"]).to_csv(card_dir / "gate_results.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "increment": summary["increment"], "coherent_provider": summary["coherent_provider"],
        "execution_mark_provider": summary["execution_mark_provider"],
        "intraday_provider": summary.get("intraday_provider"),
        "separation_rule": summary.get("separation_rule"),
        "early_field_repair": summary.get("early_field_repair"),
        "native_reconciliation_status": (summary.get("native_reconciliation") or {}).get("status"),
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    lines = [
        "# Stage1 coherent 价格重放（增量 1，中文结论）\n",
        "- 做了什么：冻结 base forecast/selection 与 control 动作模型（身份不变），仅在新 coherent 价格 panel 上重放（价格效应）；"
        "再在同一 panel 上训练费用一致候选（训练效应）；两者分开报告，不合并。\n",
        f"- coherent provider：`{summary['coherent_provider']}`；intraday provider：`{summary.get('intraday_provider')}`；"
        f"execution mark provider：`{summary['execution_mark_provider']}`。\n",
        f"- 首小时输入修复：全部 EARLY_FIELDS（ret/range/from-low/volume/bar_count × SOXL/SOXX）统一从 repaired 源重建；"
        f"{summary.get('early_field_repair')}。\n",
        f"- price control 来源：`analysis/training_results.pkl` key={summary.get('frozen_control', {}).get('key')} 的年度 models，"
        f"直接读回不 refit；control parity=`{summary.get('frozen_control', {}).get('control_parity', {}).get('status')}`。\n",
        f"- 决定：`{summary['decision']}`。\n",
        "### 价格效应（同冻结模型、仅换价格；旧值为父 Recorder 精确对象指标）\n",
    ]
    for row in summary.get("price_effect_delta_vs_old_daily", []):
        lines.append(f"- {row['segment']} @{row['cost_bps']:g}bp：coherent {row['coherent_return_pct']:.4f}% vs "
                     f"old daily {row['old_daily_return_pct']:.4f}%（Δ {row['return_delta_pct']:+.4f}pp），"
                     f"MaxDD {row['coherent_max_drawdown_pct']:.4f}% vs {row['old_daily_max_drawdown_pct']:.4f}%"
                     f"（Δ {row['max_drawdown_delta_pct']:+.4f}pp）\n")
    lines.append("（以上仅为这些已测路径上的影响幅度，不是普适结论。）\n")
    lines.append("- 注意：本卡价格效应只测 economic control 路径；legacy/rules-only 的 state feedback 路径可能对 source vintage 明显更敏感，需单独核验。\n")
    lines.append("### 训练效应（同 coherent panel 新候选项）\n")
    for row in summary["results"]:
        if row["family"] == "training_effect_candidate":
            lines.append(f"- {row['segment']} @{row['cost_bps']:g}bp：{row['return_pct']:.2f}% / MaxDD {row['max_drawdown_pct']:.2f}%\n")
    native = summary.get("native_reconciliation") or {}
    lines += [
        f"- native 对账：status=`{native.get('status')}`，boundary=`{native.get('boundary')}`；"
        "2016 prefix open-only、2018-12-31+ intraday 承接 prefix 末现金/股数；prefix 选择被承接但不重新验证。\n",
        f"- 门：候选须超过同价冻结 control；{summary.get('gates', {}).get('rows')}\n",
        "- 仅历史 opening/10:30 close 代理，不证明真实 09:31/10:31 成交；候选 not_promoted 直到独立 forward。\n",
    ]
    if summary.get("recorder_id"):
        lines.append(f"- 子 Recorder：{summary['recorder_id']}；保存真实 reports/positions/reconciliation/replays。\n")
    return "".join(lines)
