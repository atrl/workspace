"""Prior-only selection of saved training budgets; never fits a new model.

The caller verifies the parent Recorder manifest before passing its objects.
This module verifies tensor/input identities, reconstructs each original
training incumbent, and keeps the original all-seed training acceptance rule.
Historical years have already been examined: these are diagnostic replays.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd

from src.data.benchmark_anchor_utility import transform_features
from src.data.integrated_allocation import frame_sha256
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy, policy_from_dict, policy_to_dict
from src.model.constrained_net_excess import evaluate_policy, evaluate_ensemble
from src.workflow.benchmark_anchor_utility import replay_models
from src.workflow.stable_excess_integration import benchmark_path

BUDGETS = (0, 15, 30, 60, 120)
VARIANTS = ("price", "price_vol")
SEEDS = (17, 41, 71)
YEARS = tuple(range(2016, 2023))
OUTER_YEARS = tuple(range(2018, 2023))
COSTS = (10.0, 25.0)


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _finite_scalar(value):
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) and np.isfinite(value))


def reconstruct_incumbent(audit, fold, *, budget, spec, variant, year, seed):
    """Reconstruct original first-strict-improvement selection through a cap.

Stored training scores are immutable Recorder evidence, not validation scores.
Every checkpoint tensor is hash-checked; the chosen policy is independently
evaluated on the exact original training inputs before it can be replayed.
"""
    _require(budget in BUDGETS and variant in VARIANTS and year in YEARS and seed in SEEDS,
             "unregistered checkpoint identity or budget")
    _require(audit.get("variant") == variant and audit.get("year") == year and audit.get("seed") == seed,
             "saved fit identity differs")
    _require(audit.get("failure") is None and audit.get("qualification_eligible") is True,
             "failed parent fit cannot supply budget candidates")
    _require(audit.get("spec") == asdict(spec) and spec.updates == 120 and tuple(spec.cost_bps) == COSTS,
             "saved training spec differs from frozen parent")
    _require(not fold["invalid_training"] and fold["scaler"]["eligible"],
             "invalid original training inputs")
    _require(audit.get("training_rows") == len(fold["rows"]), "saved training row count differs")
    for key, value in (("training_feature_sha256", fold["features"]),
                       ("training_return_sha256", fold["returns"])):
        _require(audit.get(key) == _array_hash(value), "saved training matrix identity differs")
    _require(frame_sha256(fold["rows"]) == fold["audit"]["training_rows_sha256"],
             "saved training row identity differs")
    _require(fold["rows"].index.strftime("%Y-%m-%d").tolist() == fold["audit"]["eligible_dates"],
             "saved training dates differ")
    transformed = transform_features(fold["rows"], fold["scaler"], variant)
    _require(not transformed["invalid_forecast_mask"].any()
             and np.array_equal(transformed["features"], fold["features"]),
             "original scaler no longer reproduces training inputs")
    _require(np.array_equal(fold["rows"][["soxl_return", "soxx_return"]].to_numpy(float), fold["returns"]),
             "original training return labels differ")
    expected_cutoff = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    _require(fold["scaler"]["model_year"] == year and fold["scaler"]["variant"] == variant
             and pd.Timestamp(fold["scaler"]["cutoff"]) == expected_cutoff,
             "original scaler year or cutoff differs")
    for column in ("feature_available_at", "return_available_at"):
        dates = _aware(fold["rows"][column], column, allow_missing=False)
        _require(dates.lt(expected_cutoff).all(), "saved training input is not strictly mature")
    history = audit.get("epochs", [])
    _require([entry.get("epoch") for entry in history] == list(range(121)),
             "parent checkpoint inventory is incomplete or unordered")
    incumbent = policy_to_dict(BenchmarkAnchorPolicy(seed=seed, anchor=True))
    selected_epoch, selected_F = "anchor", 0.0
    for entry in history:
        _require(_finite_scalar(entry.get("F")) and isinstance(entry.get("feasible"), (bool, np.bool_)),
                 "checkpoint score or feasibility is invalid")
        policy = policy_from_dict(entry["checkpoint"])
        _require(policy.seed == seed, "checkpoint seed differs")
        if budget > 0 and entry["epoch"] <= budget and entry["feasible"] and entry["F"] > selected_F:
            incumbent, selected_epoch, selected_F = entry["checkpoint"], entry["epoch"], float(entry["F"])
    evaluation = evaluate_policy(policy_from_dict(incumbent), fold["features"], fold["returns"], spec=spec)
    _require(evaluation["feasible"] and _finite_scalar(evaluation["F"])
             and np.isclose(evaluation["F"], selected_F, rtol=0, atol=1e-12),
             "reconstructed training incumbent fails independent evaluation")
    if budget == 120:
        _require(audit["selected_epoch"] == selected_epoch
                 and np.isclose(audit["F"], selected_F, rtol=0, atol=1e-12),
                 "120-update reconstruction differs from saved incumbent")
    return deepcopy(incumbent), {
        "variant": variant, "year": year, "seed": seed, "budget": budget,
        "selected_epoch": selected_epoch, "training_F": selected_F,
        "checkpoint_sha256": incumbent["tensor_sha256"],
        "training_feature_sha256": audit["training_feature_sha256"],
        "training_return_sha256": audit["training_return_sha256"],
        "scaler_sha256": fold["scaler"]["scaler_sha256"],
        "training_feasibility_passed": True, "new_fit": False,
    }


def reconstruct_budget_bundles(folds, training_audits, *, spec):
    """Return ``bundles[(variant, budget)][year]`` and a small audit table."""
    wanted = {(v, y, s) for v in VARIANTS for y in YEARS for s in SEEDS}
    _require(wanted <= set(training_audits), "missing registered parent training audit")
    _require({(v, y) for v in VARIANTS for y in YEARS} <= set(folds), "missing registered parent fold")
    bundles, indices = {}, []
    for variant in VARIANTS:
        for budget in BUDGETS:
            annual = {}
            for year in YEARS:
                fold = folds[(variant, year)]
                seed_models = {}
                for seed in SEEDS:
                    payload, index = reconstruct_incumbent(training_audits[(variant, year, seed)], fold,
                        budget=budget, spec=spec, variant=variant, year=year, seed=seed)
                    seed_models[seed] = payload
                    indices.append(index)
                accepted, audit = evaluate_ensemble([policy_from_dict(seed_models[s]) for s in SEEDS],
                    fold["features"], fold["returns"], spec=spec)
                _require(audit["failure"] is None and audit["qualification_eligible"]
                         and audit["training_feasibility_passed"], "reconstructed ensemble is invalid")
                annual[year] = {
                    "variant": variant, "model_year": year, "training_budget": budget,
                    "scaler": deepcopy(fold["scaler"]), "scaler_sha256": fold["scaler"]["scaler_sha256"],
                    "seed_models": seed_models,
                    "ensemble_models": [policy_to_dict(model) for model in accepted],
                    "ensemble_audit": {key: value for key, value in audit.items()
                        if key not in ("training_paths", "rejected_ensemble_training_paths")},
                    "invalid_training": False, "invalid_training_dates": [], "numerical_failure": False,
                }
            bundles[(variant, budget)] = annual
    return bundles, pd.DataFrame(indices)


def _aware(series, name, *, allow_missing):
    _require(allow_missing or not series.isna().any(), f"missing {name}")
    for value in series.dropna():
        _require(pd.Timestamp(value).tzinfo is not None, f"{name} must include timezone")
    return pd.to_datetime(series, utc=True)


def purged_validation_frame(frame, *, outer_year):
    """Keep the complete matured prefix of the preceding two signal years.

    A missing/interior-unmatured label is an integrity error, never an excuse to
    compress the holdings path. Only the trailing immature suffix is purged.
    No value outside the two validation years is used for scoring or filtering.
    """
    _require(outer_year in OUTER_YEARS, "unregistered outer validation year")
    _require(isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is None
             and frame.index.is_monotonic_increasing and not frame.index.has_duplicates,
             "validation source needs unique ordered session dates")
    inner_years = (outer_year - 2, outer_year - 1)
    part = frame.loc[frame.index.year.isin(inner_years)].copy()
    _require(not part.empty and set(part.index.year) == set(inner_years), "missing preceding validation year")
    _require(pd.DatetimeIndex(pd.to_datetime(part.signal_date)).equals(part.index),
             "validation signal date identity differs")
    _require(pd.DatetimeIndex(pd.to_datetime(part.trade_date.iloc[:-1])).equals(part.index[1:])
             and pd.DatetimeIndex(pd.to_datetime(part.exit_date.iloc[:-2])).equals(part.index[2:])
             and np.array_equal(pd.to_datetime(part.exit_date.iloc[:-1]).to_numpy(),
                                pd.to_datetime(part.trade_date.iloc[1:]).to_numpy()),
             "validation session path has an interior calendar gap")
    cutoff = pd.Timestamp(f"{outer_year}-01-01", tz="America/New_York")
    available = _aware(part.return_available_at, "return_available_at", allow_missing=True)
    _require(not (available.isna() & part.exit_date.notna()).any(),
             "observed validation exit lacks outcome availability")
    feature_time = _aware(part.feature_available_at, "feature_available_at", allow_missing=False)
    decision = _aware(part.decision_at, "decision_at", allow_missing=False)
    _require(feature_time.lt(decision).all() and feature_time.lt(cutoff).all(),
             "validation features were unavailable at decision")
    known_outcome = available.notna()
    _require(decision.loc[known_outcome].lt(available.loc[known_outcome]).all(),
             "validation outcome was not after decision")
    mature = available.notna() & available.lt(cutoff)
    count = int(mature.sum())
    _require(count > 0 and mature.iloc[:count].all() and not mature.iloc[count:].any(),
             "validation maturity has an interior gap")
    rows = part.iloc[:count].copy()
    _require(set(rows.index.year) == set(inner_years), "no mature outcome in a validation year")
    returns = rows[["soxl_return", "soxx_return"]].to_numpy(float)
    _require(np.isfinite(returns).all() and (returns > -1).all(), "invalid mature validation returns")
    return rows, {
        "outer_year": outer_year, "inner_years": list(inner_years), "selection_cutoff": cutoff.isoformat(),
        "rows_before_purge": len(part), "validation_rows": len(rows), "purged_tail_rows": len(part) - len(rows),
        "first_signal_date": str(rows.index[0].date()), "last_signal_date": str(rows.index[-1].date()),
        "label_available_at_max": available.iloc[:count].max().isoformat(),
        "validation_frame_sha256": frame_sha256(rows),
        "validation_frame_hash_scope": "full retained source rows, including unused auxiliary labels and metadata; not a consumed-feature hash",
        "selection_uses_outer": False, "path_start": "CASH at first validation trade",
        "path_continuity": "held state carried across both validation years",
    }


def select_budget(frame, bundles, *, variant, outer_year):
    """Choose the worst-cost mean-log-excess winner; exact ties favor less fit.

    Returns (budget, candidate_scores, audit). No validation risk gate, trading
    overlay, feature selection or seed selection is introduced.
    """
    _require(variant in VARIANTS, "unregistered validation variant")
    rows, audit = purged_validation_frame(frame, outer_year=outer_year)
    references = {cost: benchmark_path(rows, cost_bps=cost) for cost in COSTS}
    scores = []
    for budget in BUDGETS:
        annual = bundles[(variant, budget)]
        exact_null = all(not np.count_nonzero(payload["state_dict"]["theta"])
            for year in audit["inner_years"] for payload in annual[year]["ensemble_models"])
        _require(budget != 0 or exact_null, "explicit null contains a non-null model")
        cost_scores = {}
        for cost in COSTS:
            path = replay_models(rows, annual, variant=variant, cost_bps=cost)
            reference = references[cost]
            keys = ["signal_date", "trade_date", "exit_date"]
            _require(path[keys].equals(reference[keys]) and len(path) == len(rows),
                     "validation policy and benchmark dates differ")
            for flag in ("observed_forecast_fallback", "invalid_training_fallback", "numerical_failure"):
                _require(flag in path and path[flag].dtype == bool and not path[flag].any(),
                         "validation contains source or numerical failure")
            excess = np.log1p(path.strategy_return.to_numpy(float)) - np.log1p(reference.strategy_return.to_numpy(float))
            _require(np.isfinite(excess).all(), "nonfinite validation net log excess")
            if exact_null:
                _require(np.max(np.abs(excess)) <= 1e-12, "explicit null differs from matching SOXX")
                cost_scores[cost] = 0.0
            else:
                cost_scores[cost] = float(excess.mean())
        scores.append({"variant": variant, "outer_year": outer_year, "budget": budget,
            "mean_log_excess_10bp": cost_scores[10.0], "mean_log_excess_25bp": cost_scores[25.0],
            "selection_score": min(cost_scores.values()), "validation_rows": len(rows),
            "selection_uses_outer": False})
    score_frame = pd.DataFrame(scores)
    chosen = min(scores, key=lambda row: (-row["selection_score"], row["budget"]))
    selected = int(chosen["budget"])
    score_frame["selected"] = score_frame.budget.eq(selected)
    return selected, score_frame, {**audit, "variant": variant, "selected_budget": selected,
        "selection_metric": "minimum across two costs of mean daily net log excess vs matching SOXX",
        "tie_break": "smaller budget on exact score ties; explicit null has budget zero", "all_three_seeds_used": True}


def annual_bundles(bundles, *, variant, selected_by_year):
    """Attach chosen yearly policies, keeping 2016/17 as fixed-120 warmup."""
    _require(variant in VARIANTS and set(selected_by_year) == set(OUTER_YEARS),
             "adaptive annual selection coverage differs")
    _require(all(value in BUDGETS for value in selected_by_year.values()), "unregistered selected budget")
    return {year: deepcopy(bundles[(variant, 120 if year < 2018 else selected_by_year[year])][year])
            for year in YEARS}
