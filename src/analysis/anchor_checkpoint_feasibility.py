"""Training-only feasibility of an immutable, complete checkpoint trajectory.

This is an analysis worker, not a model selector. It never fits, updates,
exports a replacement model, or consumes diagnostic-year allocations. Qlib
owns the source objects and native risk analysis; the orchestrator owns all
Recorder access, content-hash authorization and parallel job assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import DATES, _dates, paired_metrics
from src.model.benchmark_anchor_utility import (
    AnchorUtilitySpec, BenchmarkAnchorPolicy, evaluate_policy, policy_from_dict,
    policy_to_dict, risk_groups, rollout_alphas,
)
from src.model.integrated_allocation import _cpu_threads


class CheckpointIntegrityError(ValueError):
    """Fail closed with optional full offending paths for the new Recorder."""

    def __init__(self, message, evidence=None):
        self.evidence = {} if evidence is None else evidence
        super().__init__(message)


@dataclass(frozen=True)
class CheckpointFeasibilitySpec:
    objective_match_tolerance: float = 1e-10
    benchmark_match_tolerance: float = 1e-10
    risk_tolerance_percentage_points: float = 1e-10
    positive_utility_margin: float = 1e-8
    expected_epochs: tuple[int, ...] = tuple(range(121))
    costs_bps: tuple[float, ...] = (10., 25.)
    minimum_rows: int = 252
    torch_threads: int = 1

    def __post_init__(self):
        values = [self.objective_match_tolerance, self.benchmark_match_tolerance,
                  self.risk_tolerance_percentage_points, self.positive_utility_margin]
        if not np.isfinite(values).all() or min(values) < 0 or self.minimum_rows < 1 or self.torch_threads < 1:
            raise ValueError("blocked: invalid feasibility audit tolerances or size")
        if not self.expected_epochs or self.expected_epochs != tuple(range(len(self.expected_epochs))):
            raise ValueError("blocked: a complete ordered epoch0 trajectory is required")
        if tuple(self.costs_bps) != (10., 25.):
            raise ValueError("blocked: this diagnostic requires the two original cost scenarios")


def _require(value, message, evidence=None):
    if not bool(value):
        raise CheckpointIntegrityError("blocked: " + message, evidence)


def _sha_array(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _cost_values(values, costs, label):
    try:
        converted = {float(key): float(value) for key, value in values.items()}
    except (TypeError, ValueError, AttributeError) as exc:
        raise CheckpointIntegrityError(f"blocked: invalid {label}") from exc
    _require(len(converted) == len(values) == len(costs) and set(converted) == set(costs), label + " coverage")
    _require(np.isfinite(list(converted.values())).all(), label + " nonfinite values")
    return converted


def _frame(dates, path, cost):
    result = dates.copy()
    result["strategy_return"] = np.asarray(path["net_equity"]) - 1
    result["turnover"] = path["turnover"]
    result["fees"] = path["fees"]
    result["cost_bps"] = float(cost)
    return result


def _independent_benchmark(dates, returns, cost):
    """Fully funded SOXX entry, then no further trades, without policy code."""
    net = 1 + np.asarray(returns[:, 1], dtype=float).copy()
    rate = cost / 10000
    net[0] /= 1 + rate
    fees, turnover = np.zeros(len(net)), np.zeros(len(net))
    turnover[0] = 1 / (1 + rate)
    fees[0] = rate / (1 + rate)
    return _frame(dates, {"net_equity": net, "fees": fees, "turnover": turnover}, cost)


def classify_checkpoint(J, metrics_by_cost, *, spec: CheckpointFeasibilitySpec):
    """Tail gates and raw utility signs stay separate from numerical parity."""
    _require(np.isfinite(J), "nonfinite recomputed utility")
    _require(set(metrics_by_cost) == set(spec.costs_bps), "risk cost coverage")
    result = {}
    for cost in spec.costs_bps:
        metric = metrics_by_cost[cost]
        dd_gap = metric["max_drawdown_pct"] - metric["benchmark_max_drawdown_pct"]
        cvar_gap = metric["cvar5_pct"] - metric["benchmark_cvar5_pct"]
        _require(np.isfinite([dd_gap, cvar_gap, metric["log_excess_sum"]]).all(), "nonfinite risk/excess metrics")
        suffix = str(int(cost))
        result[f"drawdown_gap_{suffix}_pp"] = float(dd_gap)
        result[f"cvar_gap_{suffix}_pp"] = float(cvar_gap)
        result[f"drawdown_{suffix}_pass"] = dd_gap >= -spec.risk_tolerance_percentage_points
        result[f"cvar_{suffix}_pass"] = cvar_gap >= -spec.risk_tolerance_percentage_points
        result[f"net_log_excess_{suffix}"] = float(metric["log_excess_sum"])
    feasible = all(result[f"{risk}_{int(cost)}_pass"] for cost in spec.costs_bps for risk in ("drawdown", "cvar"))
    result.update({"risk_feasible": feasible, "risk_and_nonnegative_J": feasible and J >= 0,
                   "risk_and_strict_positive_J": feasible and J > 0,
                   "risk_and_positive_J_margin": feasible and J > spec.positive_utility_margin,
                   "both_costs_raw_positive_log_excess": all(result[f"net_log_excess_{int(c)}"] > 0 for c in spec.costs_bps)})
    return result


def summarize_checkpoint_statuses(statuses: pd.DataFrame):
    """First/best labels describe evidence only; no model is returned."""
    _require(not statuses.empty and statuses["explicit_null"].sum() == 1, "one explicit null required")
    learner = statuses.loc[~statuses.explicit_null].copy()
    null = statuses.loc[statuses.explicit_null].iloc[0]
    _require(not learner.empty and not learner.epoch.duplicated().any(), "unique learner epochs required")
    learner = learner.sort_values("epoch", kind="stable")
    result = {"learner_checkpoint_count": len(learner), "explicit_null_count": 1,
              "null_risk_feasible": bool(null.risk_feasible), "null_J": float(null.J),
              "checkpoint_identifiers_are_diagnostic_only": True}
    categories = ("risk_feasible", "risk_and_nonnegative_J", "risk_and_strict_positive_J", "risk_and_positive_J_margin")
    for name in categories:
        subset = learner.loc[learner[name]]
        result[name + "_count"] = len(subset)
        result["first_" + name + "_epoch"] = int(subset.epoch.iloc[0]) if len(subset) else None
        if len(subset):
            best = subset.sort_values(["J", "epoch"], ascending=[False, True], kind="stable").iloc[0]
            result["best_" + name + "_epoch"] = int(best.epoch)
            result["best_" + name + "_J"] = float(best.J)
            result["best_" + name + "_details"] = {key: best[key] for key in best.index
                if key.startswith(("drawdown_gap_", "cvar_gap_", "net_log_excess_", "recomputed_minus_stored_"))}
        else:
            result["best_" + name + "_epoch"] = None
            result["best_" + name + "_J"] = None
            result["best_" + name + "_details"] = None
    for cost in (10, 25):
        for risk in ("drawdown", "cvar"):
            result[f"{risk}_{cost}_pass_count"] = int(learner[f"{risk}_{cost}_pass"].sum())
    best_epoch, best_J = "exact_SOXX_null", float(null.J)
    for row in learner.loc[learner.risk_feasible].itertuples():
        if row.J > best_J:  # Null wins exact ties; chronological order breaks learner ties.
            best_epoch, best_J = int(row.epoch), float(row.J)
    result["null_inclusive_best_risk_feasible_epoch"] = best_epoch
    result["null_inclusive_best_risk_feasible_J"] = best_J
    result["raw_J_sign_disagreement_count"] = int(learner.J_sign_disagreement.sum())
    return result


def evaluate_checkpoint_fit(features, returns, dates, training_audit, *, variant: str,
                            model_year: int, seed: int,
                            spec: CheckpointFeasibilitySpec = CheckpointFeasibilitySpec()):
    """Reduce one immutable seed/fold's full checkpoint inventory sequentially.

    The caller must authorize the source artifact bytes and original registered
    spec before loading this worker. All returned best/first epoch labels are
    diagnostics; no deployable checkpoint, ensemble or alternative test path
    is emitted. At most one checkpoint's two full paths are live at a time.
    """
    _require(variant in ("constant", "price", "price_vol") and model_year in range(2016, 2023)
             and seed in (17, 41, 71), "outside the registered development fit inventory")
    _require(training_audit.get("variant") == variant and training_audit.get("year") == model_year
             and training_audit.get("seed") == seed, "run identity mismatch")
    _require(training_audit.get("failure") is None and training_audit.get("qualification_eligible") is True,
             "parent fit is not numerically eligible")
    x, ret = np.asarray(features, dtype=np.float64), np.asarray(returns, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[1] == 27 and ret.shape == (len(x), 2), "feature/return schema mismatch")
    _require(np.isfinite(x).all() and np.isfinite(ret).all() and (ret > -1).all(), "invalid fixed features or returns")
    _require(len(x) >= spec.minimum_rows and training_audit.get("training_rows") == len(x), "training-row coverage mismatch")
    _require(_sha_array(x) == training_audit.get("training_feature_sha256"), "training feature hash mismatch")
    _require(_sha_array(ret) == training_audit.get("training_return_sha256"), "training return hash mismatch")
    date_frame = _dates(pd.DataFrame(dates).loc[:, list(DATES)])
    _require(len(date_frame) == len(x), "date-row coverage mismatch")
    _require(date_frame.signal_date.min() >= pd.Timestamp("2011-01-03")
             and date_frame.exit_date.max() < pd.Timestamp(f"{model_year}-01-01"), "non-prior or out-of-scope training dates")
    history = training_audit.get("epochs", [])
    _require(tuple(h.get("epoch") for h in history) == spec.expected_epochs, "missing, duplicate, reordered or extra cached epochs")
    original_spec = AnchorUtilitySpec(**training_audit["spec"])
    _require(tuple(original_spec.cost_bps) == spec.costs_bps and original_spec.updates + 1 == len(history), "stored training specification mismatch")
    _require(len(x) >= original_spec.minimum_rows, "parent minimum-row contract changed")
    expected_groups = risk_groups(len(x), original_spec.risk_group_length)
    _require([tuple(g) for g in training_audit["risk_groups"]] == expected_groups, "original risk groups changed")
    identity = {"variant": variant, "model_year": int(model_year), "seed": int(seed)}
    metrics_rows, joint_rows, reproduction_rows = [], [], []
    benchmark_reports = {}
    max_J_error, max_component_error, max_benchmark_error = 0., 0., 0.
    parent_selected = training_audit["selected_epoch"]
    with _cpu_threads(spec.torch_threads):
        # Exact original utility convention, independently checked against a
        # direct SOXX entry-and-HOLD accounting construction.
        anchors = {cost: rollout_alphas(np.zeros((len(x), 3)), ret, cost_bps=cost) for cost in spec.costs_bps}
        benchmarks = {cost: _independent_benchmark(date_frame, ret, cost) for cost in spec.costs_bps}
        from qlib.contrib.evaluate import risk_analysis
        for cost in spec.costs_bps:
            error = float(np.max(np.abs(anchors[cost]["net_equity"] - (1 + benchmarks[cost].strategy_return.to_numpy()))))
            max_benchmark_error = max(max_benchmark_error, error)
            _require(error <= spec.benchmark_match_tolerance, "independent SOXX reference mismatch", {"cost_bps": cost, "anchor_path": anchors[cost]})
            benchmark_reports[cost] = risk_analysis(pd.Series(benchmarks[cost].strategy_return.to_numpy(), index=date_frame.trade_date), N=252, freq=None, mode="product")
        null_model = BenchmarkAnchorPolicy(seed=seed, anchor=True)
        null_payload = policy_to_dict(null_model)
        checkpoints = [("exact_SOXX_null", null_payload, {c: 0. for c in spec.costs_bps}, 0., True)]
        for h in history:
            stored_components = _cost_values(h["centered_utility"], spec.costs_bps, "stored per-cost utility")
            _require(np.isfinite(h["J"]) and float(h["J"]) == min(stored_components.values()), "stored worst-cost scalar inconsistent")
            _require(h["checkpoint"].get("seed") == seed, "checkpoint seed mismatch")
            checkpoints.append((h["epoch"], h["checkpoint"], stored_components, float(h["J"]), False))
        for epoch, payload, stored_components, stored_J, explicit_null in checkpoints:
            try:
                model = policy_from_dict(payload)
            except (ValueError, RuntimeError, KeyError) as exc:
                raise CheckpointIntegrityError("blocked: invalid cached tensor snapshot", {**identity, "epoch": epoch, "reason": str(exc)}) from exc
            evaluation = evaluate_policy(model, x, ret, spec=original_spec, anchors=anchors)
            J = evaluation["J"]
            differences = {c: evaluation["centered_utility"][c] - stored_components[c] for c in spec.costs_bps}
            J_difference = J - stored_J
            max_J_error = max(max_J_error, abs(J_difference))
            max_component_error = max(max_component_error, *(abs(v) for v in differences.values()))
            _require(abs(J_difference) <= spec.objective_match_tolerance and all(abs(v) <= spec.objective_match_tolerance for v in differences.values()),
                     "recomputed cached utility differs from its original scalar", {**identity, "epoch": epoch,
                     "stored_J": stored_J, "recomputed_J": J, "component_differences": differences,
                     "checkpoint": payload, "dates": date_frame, "paths": evaluation["paths"]})
            if explicit_null:
                _require(J == 0. and all(v == 0. for v in evaluation["centered_utility"].values()), "exact null centered utility is not zero")
            by_cost = {}
            selected_marker = parent_selected == ("anchor" if explicit_null else epoch)
            theta_zero = bool(np.count_nonzero(payload["state_dict"]["theta"]) == 0)
            for cost in spec.costs_bps:
                metric = paired_metrics(_frame(date_frame, evaluation["paths"][cost], cost), benchmarks[cost])
                by_cost[cost] = metric
                metrics_rows.append({**identity, "epoch": epoch, "explicit_null": explicit_null,
                    "theta_is_zero": theta_zero, "parent_selected_epoch_marker": selected_marker,
                    "checkpoint_tensor_sha256": payload["tensor_sha256"], "cost_bps": cost, **metric,
                    "centered_utility": evaluation["centered_utility"][cost], "stored_centered_utility": stored_components[cost],
                    "recomputed_minus_stored_U": differences[cost], "U_sign_disagreement": bool(np.sign(evaluation["centered_utility"][cost]) != np.sign(stored_components[cost])),
                    "J": J, "stored_J": stored_J})
            flags = classify_checkpoint(J, by_cost, spec=spec)
            if explicit_null:
                _require(flags["risk_feasible"], "explicit null violates its own risk gates")
            joint = {**identity, "epoch": epoch, "explicit_null": explicit_null, "theta_is_zero": theta_zero,
                "checkpoint_tensor_sha256": payload["tensor_sha256"], "parent_selected_epoch_marker": selected_marker,
                "J": J, "stored_J": stored_J, "recomputed_minus_stored_J": J_difference,
                "J_sign_disagreement": bool(np.sign(J) != np.sign(stored_J)), **flags,
                **{f"recomputed_minus_stored_U_{int(c)}": differences[c] for c in spec.costs_bps}}
            joint_rows.append(joint)
            reproduction_rows.append({**identity, "epoch": epoch, "explicit_null": explicit_null,
                "J": J, "stored_J": stored_J, "recomputed_minus_stored_J": J_difference,
                "J_sign_disagreement": joint["J_sign_disagreement"],
                **{f"recomputed_minus_stored_U_{int(c)}": differences[c] for c in spec.costs_bps}})
            # Do not accumulate full daily paths for every historical state.
            del evaluation
    metrics, statuses = pd.DataFrame(metrics_rows), pd.DataFrame(joint_rows)
    summary = {**identity, **summarize_checkpoint_statuses(statuses), "parent_selected_epoch": parent_selected,
        "maximum_utility_parity_error": max_J_error, "maximum_component_parity_error": max_component_error,
        "maximum_independent_benchmark_multiplier_error": max_benchmark_error}
    audit = {**identity, "all_cached_epochs_evaluated": True, "learner_checkpoint_count": len(history),
        "checkpoint_cost_records": len(history) * len(spec.costs_bps), "null_cost_records": len(spec.costs_bps),
        "training_rows": len(x), "first_signal": str(date_frame.signal_date.iloc[0].date()),
        "last_exit": str(date_frame.exit_date.iloc[-1].date()), "training_cutoff": f"{model_year}-01-01T00:00:00-05:00",
        "training_feature_sha256": _sha_array(x), "training_return_sha256": _sha_array(ret),
        "dates_sha256": hashlib.sha256(date_frame.to_csv(index=False,date_format="%Y-%m-%d").encode()).hexdigest(),
        "maximum_utility_parity_error": max_J_error, "maximum_component_parity_error": max_component_error,
        "maximum_independent_benchmark_multiplier_error": max_benchmark_error,
        "risk_groups": expected_groups, "global_mean_observations": len(x),
        "risk_tolerance_percentage_points": spec.risk_tolerance_percentage_points,
        "objective_match_tolerance": spec.objective_match_tolerance,
        "raw_negative_J_rescued_by_tolerance": False, "positive_utility_margin": spec.positive_utility_margin,
        "native_risk_api": "qlib.contrib.evaluate.risk_analysis(N=252,freq=None,mode='product')",
        "supplemental_risk": "Original initial-NAV-anchored compounded MaxDD and exact fractional lower5percent CVaR",
        "model_fitting_performed": False, "out_of_sample_rows_used": False,
        "new_model_selection_performed": False, "deployable_model_exported": False}
    return {"checkpoint_metrics": metrics, "checkpoint_status": statuses, "fit_summary": summary,
            "utility_reproduction": pd.DataFrame(reproduction_rows), "benchmark_risk_reports": benchmark_reports,
            "audit": audit}


__all__ = ["CheckpointFeasibilitySpec", "CheckpointIntegrityError", "evaluate_checkpoint_fit",
           "classify_checkpoint", "summarize_checkpoint_statuses"]
