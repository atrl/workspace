"""Fixed 5-session action-label ablation (one candidate, no parameter search).

Trains the annual Ridge action models on the FROZEN parent-state features joined
to the fixed 5-session label from the diagnostic card, then replays the candidate
own-state account on the same uniform panel at 10bp and 25bp. The original 1-day
models/replays are reused as the control; they are never re-trained. Baselines keep
their source identities. Outer is a reused diagnostic, never a fresh out-of-sample.

Off-policy limitation: training uses the frozen parent trajectory, so the candidate
is NOT trained on its own on-policy multi-day rollout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis import multisession_label as M
from src.workflow import coherent_price_replay_experiment as C
from src.workflow import feasible_action_experiment as X
from src.workflow.stage1_economic_training import (
    ECONOMIC_FEATURES,
    MINIMUM_TRAINING_ROWS,
    RIDGE_ALPHA,
    _event_at,
    _model_at,
    build_economic_dataset,
    mature_economic_rows,
    simulate_economic_policy,
)
from src.workflow.stage1_open_action_value import (
    Stage1OpenActionValueSpec,
    _metrics as open_metrics,
    _simulate_on_policy_router,
)

from src.workflow.multisession_label_audit import _lineage as _lineage_impl

EXPERIMENT = "stage1_multisession_action_ablation_20261010"
SCHEMA_VERSION = "stage1_multisession_action_ablation_v1"
DIAG_CARD = "research/stage1_multisession_label_audit_20261010"
DIAG_RECORDER = "5adde25b815042a78396492ea71169d1"
LABELS_10_SHA256 = "6f1582fd490e466a461a5854b25c5ebd467116a551b871eba3b179429a360858"
COSTS_BPS = (10.0, 25.0)
SEGMENT_LABELS = M.SEGMENT_LABELS
FIT_YEARS = (2017, 2018, 2019, 2020, 2021, 2022, 2023)
FROZEN_CONTROL_KEY = 2022
ANNUAL_WEALTH_RATIO = 0.985
OPEN_TRADE_DAYS_REDUCTION = 0.92
QLIB_DOCS = C.QLIB_DOCS


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _lineage(root: Path) -> dict:
    return _lineage_impl(root)


def _resolve_root(root) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def load_diag_labels(root: Path) -> pd.DataFrame:
    from qlib.workflow import R

    base = root / DIAG_CARD / "qlib_recorder"
    with R.uri_context(str(base)):
        recorder = R.get_recorder(recorder_id=DIAG_RECORDER, experiment_name=DIAG_CARD.split("/")[-1])
        artifacts = Path(recorder.get_local_dir()) / "artifacts"
        _require(M._file_or_tree_sha(artifacts / "labels_10") == LABELS_10_SHA256, "diag labels_10 hash mismatch")
        labels = recorder.load_object("labels_10")
    return labels


def build_candidate_rows(control: dict, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    training = control["training_rows"].sort_values("signal_date").reset_index(drop=True)
    _require(len(labels) == len(training), "diag labels do not fully cover the parent rows")
    _require(not labels["signal_date"].duplicated().any(), "duplicate diag label signal_date")
    join = labels[["signal_date", "window_status", "label_5d_bps", "horizon_available_at"]].sort_values("signal_date")
    merged = training.merge(join, on="signal_date", how="left", validate="one_to_one")
    _require(len(merged) == len(training), "candidate join lost parent rows")
    merged = merged.copy()
    # Preserve the ORIGINAL source target identity under prefixed names before
    # aliasing in the candidate 5-session label/clock (so source identity is not lost).
    merged["source_execute_advantage_bps"] = merged["execute_advantage_bps"]
    merged["source_label_available_at"] = merged["label_available_at"]
    merged["source_label_status"] = merged["label_status"]
    merged["candidate_label_5d_bps"] = pd.to_numeric(merged["label_5d_bps"], errors="coerce")
    available = merged["window_status"].eq("available")
    _require(bool(available.sum() == int(merged["candidate_label_5d_bps"].notna().sum())),
             "non-available window carries a numeric 5-day label")
    _require(bool(merged.loc[available, "horizon_available_at"].notna().all()),
             "available window missing its maturity clock")
    _require(bool(merged.loc[~available, "candidate_label_5d_bps"].isna().all()),
             "unknown-row candidate label was filled (must stay NaN)")
    _require(bool(merged.loc[~available, "horizon_available_at"].isna().all()),
             "unknown-row candidate maturity was filled (must stay NaT)")
    # Reuse the sealed DatasetH plumbing by aliasing the candidate label/clock.
    merged["execute_advantage_bps"] = merged["candidate_label_5d_bps"]
    merged["label_available_at"] = merged["horizon_available_at"]
    _require(bool(merged.loc[~available, "execute_advantage_bps"].isna().all())
             and bool(merged.loc[~available, "label_available_at"].isna().all()),
             "candidate alias filled an unknown row")
    audit = {
        "parent_rows": int(len(training)),
        "candidate_rows_with_finite_5d_label": int(merged["candidate_label_5d_bps"].notna().sum()),
        "excluded_unknown_or_crossyear": int(merged["candidate_label_5d_bps"].isna().sum()),
        "null_label_filled_with_zero": 0,
        "unknown_rows_retained": int((~available).sum()),
        "missing_filled_count": 0,
        "window_status_counts": {k: int(v) for k, v in merged["window_status"].fillna("missing").value_counts().items()},
        "illegal_hold_counterfactual_rows": int((~merged["routing_eligible"].astype(bool)).sum()),
        "note": "source identity features are not filtered by routing_eligible; illegal-HOLD rows are counted as a limitation",
    }
    return merged, audit


def fit_candidate_models(rows: pd.DataFrame, *, frame: pd.DataFrame) -> tuple[dict, list[dict], list[pd.DataFrame]]:
    dataset = build_economic_dataset(rows)
    first_decision = {}
    for year, section in frame.assign(_y=pd.to_datetime(frame["trade_date"]).dt.year).groupby("_y"):
        first_decision[int(year)] = _event_at(section["trade_date"].iloc[0], minute=31)
    years = sorted(int(y) for y in pd.to_datetime(frame["signal_date"]).dt.year.unique())
    models: dict[int, Any] = {}
    diagnostics = []
    predictions = []
    last = None
    for year in years:
        if year in FIT_YEARS:
            prepared, cutoff = mature_economic_rows(dataset, model_year=year, first_decision_at=first_decision[year])
            model = None
            if len(prepared) >= MINIMUM_TRAINING_ROWS:
                model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
                model.fit(prepared.loc[:, list(ECONOMIC_FEATURES)], prepared["execute_advantage_bps"])
            last = model
            maturity = pd.to_datetime(prepared["label_available_at"], utc=True)
            matured_dates = pd.DatetimeIndex(prepared.index.get_level_values("datetime"))
            eligibility = rows.set_index("signal_date")["routing_eligible"].reindex(matured_dates)
            if model is not None:
                predictions.append(pd.DataFrame({
                    "model_year": year, "signal_date": matured_dates,
                    "prediction": model.predict(prepared.loc[:, list(ECONOMIC_FEATURES)]),
                    "train_label": prepared["execute_advantage_bps"].to_numpy(float)}))
            maturity_ok = bool(len(prepared) == 0 or maturity.max() < pd.Timestamp(cutoff))
            diagnostics.append({
                "model_year": year, "fit": True, "training_rows": int(len(prepared)),
                "model_available": model is not None,
                "information_cutoff_exclusive": cutoff.isoformat(),
                "max_label_available_at": maturity.max().isoformat() if len(prepared) else None,
                "maturity_strictly_before_cutoff": maturity_ok,
                "illegal_hold_training_rows": int((~eligibility.astype(bool)).sum()),
                "label_cost_bps": 25.0, "ridge_alpha": RIDGE_ALPHA, "minimum_training_rows": MINIMUM_TRAINING_ROWS,
                "training_state": "fixed_parent_state_label_ablation",
            })
        models[year] = last
    return models, diagnostics, predictions


def _replay_candidate(panel, models, *, cost, common):
    return simulate_economic_policy(panel, models=models, policy="net_value", cost_bps=cost,
                                    **{k: common[k] for k in ("recovery_models", "source_spec", "selected_model")})


def _soxx_metrics(panel: pd.DataFrame, *, costs) -> dict:
    """Per-segment SOXX buy-hold from the same ledger (one entry fee, no final fake sell)."""
    path = {cost: X.soxx_buy_hold_path(panel, cost_bps=cost) for cost in costs}
    out = {}
    for cost in costs:
        for segment, span in SEGMENT_LABELS.items():
            band = panel[panel["segment"] == segment]
            seg_path = path[cost][path[cost]["segment"] == span].reset_index(drop=True)
            _require(len(seg_path) == len(band), f"blocked: SOXX path length mismatch for {span}@{cost:g}bp")
            curve = np.concatenate([[1.0], seg_path["net_equity"].to_numpy(float)])
            daily = curve[1:] / curve[:-1] - 1.0
            return_pct = 100.0 * float(curve[-1] - 1.0)
            out[(segment, cost)] = {"stock": "SOXX_buy_hold", "segment": span, "cost_bps": cost,
                                    "sessions": int(len(daily)), "return_pct": return_pct,
                                    "daily_returns": daily,
                                    "max_drawdown_pct": 100.0 * float((curve / np.maximum.accumulate(curve) - 1.0).min()),
                                    "sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 else None,
                                    "cumulative_return_pct": return_pct,
                                    "signal_date": seg_path["signal_date"].to_numpy()}
    return out


def _paired_block_ci(cand: np.ndarray, ref: np.ndarray, *, block: int = 20, bootstrap: int = 2000,
                     seed: int = 20261010) -> dict:
    n = len(cand)
    if n < block:
        return {"mean_log_relative_bps": None, "ci_low_bps": None, "ci_high_bps": None, "blocks": 0, "status": "needs_data"}
    lr = np.log1p(cand) - np.log1p(ref)
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block + 1)
    draws = []
    for _ in range(bootstrap):
        chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in chosen])[:n]
        draws.append(float(lr[idx].mean()))
    return {"mean_log_relative_bps": 10_000.0 * float(lr.mean()),
            "ci_low_bps": 10_000.0 * float(np.quantile(draws, 0.025)),
            "ci_high_bps": 10_000.0 * float(np.quantile(draws, 0.975)),
            "terminal_wealth_ratio": float(np.prod(1.0 + cand) / np.prod(1.0 + ref)),
            "blocks": int(len(starts)), "block": block, "bootstrap": bootstrap, "seed": seed, "status": "ok"}


def _paired_uncertainty(replays: dict, soxx: dict, *, block: int = 20) -> list[dict]:
    rows = []
    for segment, span in SEGMENT_LABELS.items():
        for cost in COSTS_BPS:
            cand = replays[f"candidate_label_5d::{segment}::{int(cost)}"].reset_index(drop=True)
            control = replays[f"control_all_state::{segment}::{int(cost)}"].reset_index(drop=True)
            _require(bool(np.array_equal(pd.to_datetime(cand["trade_date"]).to_numpy(),
                                         pd.to_datetime(control["trade_date"]).to_numpy())),
                     f"candidate/control calendar mismatch {span}@{cost:g}bp")
            for name, ref_daily in (("control_all_state", control["strategy_return"].to_numpy(float)),
                                    ("soxx_buy_hold", soxx[(segment, cost)]["daily_returns"])):
                ci = _paired_block_ci(cand["strategy_return"].to_numpy(float), ref_daily, block=block)
                rows.append({"segment": span, "cost_bps": cost, "reference": name, **ci})
    return rows


def _gate_rows(records: list[dict], annual: pd.DataFrame, soxx: dict) -> list[dict]:
    lookup = {(r["variant"], r["segment"], r["cost_bps"]): r for r in records}
    dev_annual = annual[(annual["segment"] == "development_walk_forward") & (annual["year"].between(2017, 2022))
                        & (annual["cost_bps"] == 10)]
    rows = []
    segment = "development_walk_forward"
    for reference in ("no_hold_reference", "active_hold_baseline"):
        candidate = lookup.get(("candidate_label_5d", segment, 10.0))
        baseline = lookup.get((reference, segment, 10.0))
        stress_c = lookup.get(("candidate_label_5d", segment, 25.0))
        stress_b = lookup.get((reference, segment, 25.0))
        if None in (candidate, baseline, stress_c, stress_b):
            rows.append({"reference": reference, "status": "needs_data", "all_pass": False})
            continue
        a = dev_annual[dev_annual["variant"] == "candidate_label_5d"].set_index("year")
        b = dev_annual[dev_annual["variant"] == reference].set_index("year")
        years = [y for y in sorted(set(a.index) & set(b.index)) if b.loc[y, "terminal_wealth"] != 0]
        annual_ratio = min(float(a.loc[y, "terminal_wealth"] / b.loc[y, "terminal_wealth"]) for y in years) if years else None
        checks = {
            "development_return_above_baseline": bool(candidate["return_pct"] > baseline["return_pct"]),
            "stress_25bps_return_above_baseline": bool(stress_c["return_pct"] > stress_b["return_pct"]),
            "max_drawdown_not_worse": bool(candidate["max_drawdown_pct"] >= baseline["max_drawdown_pct"] - 1e-10),
            "worst_day_not_worse": bool(candidate["worst_day_pct"] >= baseline["worst_day_pct"] - 1e-10),
            "cvar5_not_worse": bool(candidate["cvar5_pct"] >= baseline["cvar5_pct"] - 1e-10),
            "worst_5d_not_worse": bool(candidate["worst_rolling_5d_pct"] >= baseline["worst_rolling_5d_pct"] - 1e-10),
            "open_trade_days_reduced_8pct": bool(candidate["open_trade_days"] <= OPEN_TRADE_DAYS_REDUCTION * baseline["open_trade_days"]),
            "turnover_not_increased": bool(candidate["total_turnover"] <= baseline["total_turnover"]),
            "annual_wealth_floor_pass": bool(annual_ratio is not None and annual_ratio >= ANNUAL_WEALTH_RATIO),
        }
        rows.append({"candidate": "candidate_label_5d", "reference": reference, "status": "ok",
                     "gate_scope": "development_2017_2022", "minimum_annual_wealth_ratio": annual_ratio,
                     **checks, "all_pass": bool(all(checks.values()))})
    return rows


def _soxx_comparison(records: list[dict], annual: pd.DataFrame, soxx: dict, replays: dict) -> list[dict]:
    """Same-cost, same-calendar SOXX comparison; relative wealth = ratio, annual win rate."""
    lookup = {(r["variant"], r["segment"], r["cost_bps"]): r for r in records}
    rows = []
    for segment, span in SEGMENT_LABELS.items():
        for cost in COSTS_BPS:
            candidate = lookup.get(("candidate_label_5d", span, cost))
            benchmark = soxx[(segment, cost)]
            if candidate is None:
                continue
            years = pd.to_datetime(benchmark["signal_date"]).year
            soxx_annual = pd.Series(benchmark["daily_returns"]).groupby(years).apply(lambda r: float(np.prod(1.0 + r) - 1.0))
            cand_annual = annual[(annual["variant"] == "candidate_label_5d") & (annual["segment"] == span)
                                 & (annual["cost_bps"] == int(cost))].set_index("year")["terminal_wealth"] - 1.0
            common_years = sorted(set(soxx_annual.index) & set(cand_annual.index))
            wins = [1 for y in common_years if cand_annual.loc[y] >= soxx_annual.loc[y]]
            daily = replays[f"candidate_label_5d::{segment}::{int(cost)}"]["strategy_return"].to_numpy(float)
            rows.append({
                "segment": span, "cost_bps": cost,
                "candidate_return_pct": candidate["return_pct"], "soxx_return_pct": benchmark["return_pct"],
                "return_delta_pct_points": candidate["return_pct"] - benchmark["return_pct"],
                "relative_wealth_pct": 100.0 * ((1.0 + candidate["return_pct"] / 100.0)
                                                / (1.0 + benchmark["return_pct"] / 100.0) - 1.0),
                "candidate_sharpe": (float(np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252))
                                     if daily is not None and np.std(daily, ddof=1) > 0 else None),
                "soxx_sharpe": benchmark["sharpe"],
                "candidate_max_drawdown_pct": candidate["max_drawdown_pct"],
                "soxx_max_drawdown_pct": benchmark["max_drawdown_pct"],
                "annual_win_rate": (len(wins) / len(common_years)) if common_years else None,
                "common_years": [int(y) for y in common_years],
            })
    return rows


def _oos_action_predictions(replays: dict) -> pd.DataFrame:
    rows = []
    for key, replay in replays.items():
        if not key.startswith("candidate_label_5d"):
            continue
        _, raw_segment, cost = key.split("::")
        for _, row in replay.iterrows():
            rows.append({"signal_date": row["signal_date"], "model_year": row.get("model_year"),
                         "cost_bps": float(cost), "segment": SEGMENT_LABELS[raw_segment],
                         "predicted_net_advantage_bps": row.get("predicted_net_advantage_bps"),
                         "execute_open_target": bool(row.get("execute_open_target")),
                         "recovery_action": bool(row.get("recovery_action")),
                         "mandatory_position_repair": bool(row.get("mandatory_position_repair"))})
    return pd.DataFrame(rows)


def _risk_table(replays: dict, soxx: dict) -> list[dict]:
    from qlib.contrib.evaluate import risk_analysis

    def _ra(series):
        out = risk_analysis(series, freq="day", mode="product")
        return {str(index): float(value) for index, value in out.iloc[:, 0].items()}

    rows = []
    for key, replay in replays.items():
        if not key.startswith("candidate_label_5d"):
            continue
        _, raw_segment, cost = key.split("::")
        series = pd.Series(replay["strategy_return"].to_numpy(float), index=pd.to_datetime(replay["trade_date"]))
        rows.append({"book": "candidate_label_5d", "segment": SEGMENT_LABELS[raw_segment], "cost_bps": float(cost),
                     **_ra(series)})
    for (segment, cost), benchmark in soxx.items():
        series = pd.Series(benchmark["daily_returns"], index=pd.to_datetime(benchmark["signal_date"]))
        rows.append({"book": "soxx_buy_hold", "segment": benchmark["segment"], "cost_bps": cost, **_ra(series)})
    return rows


def run_multisession_action_ablation(*, root=None, card_dir=None) -> dict[str, Any]:
    from dataclasses import replace

    spec_root = _resolve_root(root)
    target_card = Path(card_dir) if card_dir is not None else (spec_root / "research" / EXPERIMENT)
    target_card.mkdir(parents=True, exist_ok=True)
    plan_path = spec_root / "research" / EXPERIMENT / "plan.json"
    _require(plan_path.is_file(), "plan.json is missing")

    control = M.load_frozen_control(spec_root)
    panel, common = control["panel"], control["common"]
    legacy_models = C.load_frozen(spec_root)["inputs/prepared_inputs.pkl"]["legacy_models"]
    labels = load_diag_labels(spec_root)
    rows, rows_audit = build_candidate_rows(control, labels)
    models, diagnostics, predictions = fit_candidate_models(rows, frame=panel)

    old_spec = Stage1OpenActionValueSpec(root=spec_root)
    replays, records = {}, []

    def add(variant, segment, cost, replay):
        replays[f"{variant}::{segment}::{int(cost)}"] = replay
        record = open_metrics(replay, variant=variant, segment=SEGMENT_LABELS[segment], cost_bps=cost)
        daily = replay["strategy_return"].to_numpy(float)
        record["sharpe"] = (float(np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252))
                            if np.std(daily, ddof=1) > 0 else None)
        records.append(record)

    for segment in SEGMENT_LABELS:
        market = panel[panel["segment"] == segment]
        for cost in COSTS_BPS:
            add("candidate_label_5d", segment, cost, _replay_candidate(market, models, cost=cost, common=common))
            saved = control["replays"][f"control_all_state::{SEGMENT_LABELS[segment]}::{int(cost)}"].reset_index(drop=True)
            add("control_all_state", segment, cost, saved)
            add("active_hold_baseline", segment, cost,
                _simulate_on_policy_router(market, models=legacy_models,
                                           spec=replace(old_spec, transaction_cost_bps=cost), segment=SEGMENT_LABELS[segment],
                                           recovery_models=common["recovery_models"], source_spec=common["source_spec"],
                                           selected_model=common["selected_model"]))
            add("no_hold_reference", segment, cost,
                _simulate_on_policy_router(market, models={}, spec=replace(old_spec, transaction_cost_bps=cost),
                                           segment=SEGMENT_LABELS[segment], recovery_models=common["recovery_models"],
                                           source_spec=common["source_spec"], selected_model=common["selected_model"]))
            cash = market[["signal_date", "trade_date", "exit_date"]].copy()
            for column, value in (("strategy_return", 0.0), ("turnover", 0.0), ("execute_open_trade", False),
                                  ("execute_open_target", False), ("recovery_action", False), ("cash_weight", 1.0),
                                  ("soxl_weight", 0.0), ("soxx_weight", 0.0), ("open_soxl_weight", 0.0),
                                  ("open_soxx_weight", 0.0), ("open_cash_weight", 1.0), ("ending_soxl_weight", 0.0),
                                  ("ending_soxx_weight", 0.0), ("ending_cash_weight", 1.0)):
                cash[column] = value
            add("cash_reference", segment, cost, cash)

    annual_rows = []
    for key, replay in replays.items():
        variant, raw_segment, cost = key.split("::")
        span = SEGMENT_LABELS[raw_segment]
        frame = replay.copy()
        frame["year"] = pd.to_datetime(frame["signal_date"]).dt.year
        for year, section in frame.groupby("year"):
            annual_rows.append({"year": int(year), "variant": variant, "segment": span, "cost_bps": int(cost),
                                "terminal_wealth": float(np.prod(1.0 + section["strategy_return"].to_numpy(float)))})
    annual = pd.DataFrame(annual_rows)

    soxx = _soxx_metrics(panel, costs=COSTS_BPS)
    gates = _gate_rows(records, annual, soxx)
    paired = _paired_uncertainty(replays, soxx)
    soxx_report = _soxx_comparison(records, annual, soxx, replays)
    oos_predictions = _oos_action_predictions(replays)
    risk_table = _risk_table(replays, soxx)

    # Native: candidate economic variants, post_fee_desired, uniform providers.
    coherent = (spec_root / X.PROVIDERS["coherent_provider"]).resolve()
    execution = (spec_root / X.PROVIDERS["execution_mark_provider"]).resolve()
    candidate_replays = {k: v for k, v in replays.items() if k.startswith("candidate_label_5d")}
    try:
        native, native_objects = C._native_reconcile(_Spec(spec_root), panel, candidate_replays, coherent, execution,
                                                     target_conventions={k: "post_fee_desired" for k in candidate_replays})
    except ValueError as exc:
        native, native_objects = {"status": "blocked", "reason": str(exc)}, {}

    nine_ok = bool(gates) and all(g.get("status") == "ok" and g.get("all_pass") for g in gates)
    native_ok = native.get("status") == "passed"
    dev_control = [p for p in paired if p["reference"] == "control_all_state" and p["segment"] == "development_walk_forward"]
    dev_soxx = [p for p in paired if p["reference"] == "soxx_buy_hold" and p["segment"] == "development_walk_forward"]

    def _positive(rows):
        return bool(rows) and all(r["status"] == "ok" and r["ci_low_bps"] is not None and r["ci_low_bps"] > 0 for r in rows)

    # Qualification uses DEVELOPMENT ONLY; outer is reported separately as a diagnostic.
    reasons = []
    if not nine_ok:
        reasons.append("nine_gates_failed")
    if not _positive(dev_control):
        reasons.append("dev_paired_ci_vs_control_not_positive")
    if not _positive(dev_soxx):
        reasons.append("dev_paired_ci_vs_soxx_not_positive")
    if not native_ok:
        reasons.append("candidate_native_not_passed")
    if nine_ok and native_ok and _positive(dev_control) and _positive(dev_soxx):
        decision, stop_route = "historical_pass_needs_independent_forward", None
    elif not native_ok:
        decision, stop_route = "needs_data", "end_fixed_label_route_no_parameter_search"
    else:
        decision, stop_route = "not_promoted", "end_fixed_label_route_no_parameter_search"
    summary = {
        "experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION, "card_status": "active_research",
        "decision": decision, "decision_reasons": reasons, "promotion_eligible": False, "objective_achieved": False,
        "qualification_scope": "development_only; outer=candidate_diagnostic_not_qualification",
        "stop_route": stop_route,
        "plan_sha256": M._file_or_tree_sha(plan_path),
        "data_lineage": _lineage(spec_root),
        "source_bindings": {"diag_recorder": DIAG_RECORDER, "labels_10_sha256": LABELS_10_SHA256,
                            "control_recorder": M.CONTROL_RECORDER, "prepared_sha256": control["prepared_sha256"],
                            "candidate_model_object": "recorder-only candidate_model_<year>/candidate_scaler_<year>",
                            "source_target_identity": "preserved as source_* columns; candidate 5-day alias is separate"},
        "candidate_audit": rows_audit, "fit_diagnostics": diagnostics,
        "model_years": sorted(int(y) for y in models), "annual_fits": list(FIT_YEARS),
        "results": records, "gates": gates, "paired_uncertainty": paired, "soxx_comparison": soxx_report,
        "risk_analysis": risk_table,
        "native": native,
        "off_policy_limitation": "trained on the frozen parent trajectory, not the candidate's own on-policy multi-day rollout",
        "outer": "reused_diagnostic",
        "qlib_docs_reference": QLIB_DOCS,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": ["candidate_models", "candidate_scalers", "training_rows",
                                                        "fit_diagnostics", "predictions", "replays", "native"]},
    }
    summary.update(_record(spec_root, target_card, summary, models, rows, diagnostics, predictions, oos_predictions,
                           risk_table, replays, native_objects, plan_path))
    _write_card(summary, target_card)
    return summary


class _Spec:
    def __init__(self, root):
        self.root = Path(root)
        self.experiment = EXPERIMENT


def _record(spec_root, card_dir, summary, models, rows, diagnostics, predictions, oos_predictions, risk_table,
            replays, native_objects, plan_path):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    M.C._quiet() if hasattr(M, "C") else None
    import logging
    import warnings
    for name in ("qlib", "mlflow", "qlib.data", "qlib.workflow"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    import qlib
    from qlib.workflow import R

    recorder_uri = (Path(card_dir) / "qlib_recorder").resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((Path(spec_root) / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    saved = {"training_rows": rows, "fit_diagnostics": pd.DataFrame(diagnostics), "replays": replays,
             "training_predictions": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
             "oos_action_predictions": oos_predictions, "risk_analysis": pd.DataFrame(risk_table)}
    for year, model in models.items():
        saved[f"candidate_model_{year}"] = model
        if model is not None:
            saved[f"candidate_scaler_{year}"] = model.named_steps["standardscaler"]
    for name, value in native_objects.items():
        saved[f"native::{name}"] = value
    with R.start(experiment_name=EXPERIMENT, recorder_name="multisession_ablation", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(schema_version=SCHEMA_VERSION, plan_sha256=summary["plan_sha256"],
                            decision=summary["decision"], annual_fits=len(FIT_YEARS))
        recorder.save_objects(**saved)
        recorder.log_artifact(str(plan_path), artifact_path="plan")
        recorder_id, experiment_id = recorder.id, str(recorder.experiment_id)
    summary["artifact_policy"]["recorder_only_artifacts"] = sorted(saved)
    summary["recorder_local_dir"] = str((recorder_uri / experiment_id / recorder_id))
    summary["saved_objects"] = sorted(saved)
    return {"recorder_id": recorder_id, "experiment_id": experiment_id,
            "recorder_uri": str(recorder_uri), "saved_objects": sorted(saved)}


def _write_card(summary, card_dir: Path) -> None:
    (card_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
                                           encoding="utf-8")
    pd.DataFrame(summary["results"]).to_csv(card_dir / "ablation_comparison.csv", index=False)
    pd.DataFrame(summary["gates"]).to_csv(card_dir / "gate_results.csv", index=False)
    pd.DataFrame(summary["paired_uncertainty"]).to_csv(card_dir / "paired_uncertainty.csv", index=False)
    pd.DataFrame(summary["soxx_comparison"]).to_csv(card_dir / "soxx_comparison.csv", index=False)
    pd.DataFrame(summary["risk_analysis"]).to_csv(card_dir / "risk_analysis.csv", index=False)
    workflow = {"schema_version": SCHEMA_VERSION, "experiment": summary["experiment"], "plan_sha256": summary["plan_sha256"],
                "decision": summary["decision"], "annual_fits": list(FIT_YEARS),
                "source_bindings": summary["source_bindings"], "off_policy_limitation": summary["off_policy_limitation"],
                "data_lineage": summary["data_lineage"], "qualification_scope": summary["qualification_scope"],
                "native_status": summary["native"].get("status"),
                "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS}
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary) -> str:
    lines = [
        "# 固定 5-session label 动作消融（候选，中文结论）\n",
        "- 仅把目标换成 fixed 5-session label；1 候选 + 7 年度 fit（2017-2023），2016 fallback，2024-2026 carry 2023；"
        "0 调参/选符号/挑窗口。\n",
        f"- training_state=`fixed_parent_state_label_ablation`；continuation=original frozen net_value router；"
        "训练用冻结父状态，**不是**候选自身 on-policy 多日标签（off-policy 偏移为限制）。\n",
        f"- 源绑定：diag recorder `{DIAG_RECORDER}` labels_10 sha `{LABELS_10_SHA256[:12]}`；control recorder `{M.CONTROL_RECORDER}`。\n",
        f"- candidate rows：{summary['candidate_audit']}。\n",
        f"- 决定：`{summary['decision']}`；reasons={summary['decision_reasons']}；stop_route={summary['stop_route']}。\n",
        "### 原九项门（仅 development 2017-2022，outer 不入 annual 门）\n",
    ]
    for g in summary["gates"]:
        lines.append(f"- ref {g.get('reference')}：status={g.get('status')} all_pass={g.get('all_pass')} "
                     f"{ {k: v for k, v in g.items() if isinstance(v, bool)} }\n")
    lines.append("### SOXX 同费用比较（relative wealth = candidate 相对 SOXX 的比值；非相对 control）\n")
    for r in summary["soxx_comparison"]:
        lines.append(f"- {r['segment']} @{r['cost_bps']:g}bp：cand {r['candidate_return_pct']:.2f}% vs SOXX {r['soxx_return_pct']:.2f}%；"
                     f"relative_wealth(vs SOXX) {r['relative_wealth_pct']:.2f}%；return_delta {r['return_delta_pct_points']:.2f}pp；"
                     f"annual win rate {r['annual_win_rate']}。\n")
    lines.append("### paired 20-session block uncertainty（cost×segment 独立）\n")
    for r in summary["paired_uncertainty"]:
        lines.append(f"- {r['segment']} @{r['cost_bps']:g}bp vs {r['reference']}：mean log rel {r['mean_log_relative_bps']} "
                     f"CI [{r['ci_low_bps']},{r['ci_high_bps']}] blocks={r['blocks']} status={r['status']}。\n")
    lines.append(f"- native：{summary['native'].get('status')}。\n")
    lines.append("- 边界：outer=reused_diagnostic（非 fresh）；历史看似通过仍需独立前向，不替换现役；"
                 "诊断 CI 不明确只是探索，不预称 predictive gate 通过。\n")
    if summary.get("recorder_id"):
        lines.append(f"- 子 Recorder：{summary['recorder_id']}。\n")
    return "".join(lines)
