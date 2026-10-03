"""Full-path net-excess proposals with benchmark tail-feasible incumbents.

The architecture, funding kernel and truncated-gradient implementation are the
immutable benchmark-anchor parent. Four signed dual variables change optimizer
proposals; only the fixed objective and original raw risk gates can admit an
incumbent. Feasibility is a training-path property, not a forward guarantee.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import math
from typing import Any

import numpy as np
import torch

from src.model.benchmark_anchor_utility import (
    AnchorUtilitySpec, BenchmarkAnchorPolicy, NumericalFitFailure, _audit_anchor_paths,
    _matrix, _validate_parameters, evaluate_policy as parent_evaluate_policy,
    policy_from_dict, policy_to_dict, project_theta, risk_groups, rollout_alphas,
    truncated_utility_gradient, utility_and_daily_weights,
)
from src.model.integrated_allocation import _cpu_threads
from src.objective.benchmark_tail_constraints import (
    TailConstraintError, combined_daily_adjoints, evaluate_constraints, evaluate_risk, reference_risk,
)


@dataclass(frozen=True)
class ConstrainedNetExcessSpec(AnchorUtilitySpec):
    downside_coefficient: float = 0.
    dual_learning_rate: float = .003
    risk_tolerance_percentage_points: float = 1e-10

    def __post_init__(self):
        super().__post_init__()
        if self.downside_coefficient != 0 or tuple(self.cost_bps) != (10., 25.):
            raise ValueError("blocked: constrained objective requires zero RMS penalty and original two costs")
        if (not np.isfinite([self.dual_learning_rate, self.risk_tolerance_percentage_points]).all()
                or self.dual_learning_rate <= 0 or self.risk_tolerance_percentage_points < 0):
            raise ValueError("blocked: invalid dual step or raw risk tolerance")


def _parent_spec(spec, *, downside):
    values = {f.name: getattr(spec, f.name) for f in fields(AnchorUtilitySpec)}
    values["downside_coefficient"] = float(downside)
    return AnchorUtilitySpec(**values)


def _hash_array(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def update_multipliers(multipliers, post_primal_g, *, learning_rate):
    """Signed post-primal ascent with only the declared nonnegative projection."""
    dual, g = np.asarray(multipliers, dtype=float), np.asarray(post_primal_g, dtype=float)
    if (dual.shape != (4,) or g.shape != (4,) or not np.isfinite(dual).all()
            or not np.isfinite(g).all() or (dual < 0).any()
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise NumericalFitFailure("invalid four-constraint dual update")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.maximum(0., dual + learning_rate * g)
    if not np.isfinite(result).all():
        raise NumericalFitFailure("nonfinite dual proposal; no multiplier cap is permitted")
    return result


def _risk_record(constraint):
    """Small scalar audit only: daily risk adjoints/tail weights stay ephemeral."""
    risk, reference = constraint["risk"], constraint["normalizers"]
    return {"max_drawdown_pct": risk["max_drawdown_pct"], "cvar5_pct": risk["cvar5_pct"],
            "benchmark_max_drawdown_pct": -100 * float(reference[0]),
            "benchmark_cvar5_pct": -100 * float(reference[1]),
            "raw_loss_gaps": constraint["raw_loss_gaps"].tolist(),
            "raw_percentage_point_gaps": constraint["raw_percentage_point_gaps"].tolist(),
            "normalizers": reference.tolist(), "risk_pass": constraint["risk_pass"].tolist(),
            "normalizers_valid_for_optimization": bool(np.isfinite(reference).all() and (reference > 0).all()),
            "feasible": constraint["feasible"], "peak_index": risk["peak_index"],
            "trough_index": risk["trough_index"], "tail_boundary_return": risk["tail_boundary_return"],
            "tail_boundary_count": risk["tail_boundary_count"]}


def _evaluate(models, x, returns, *, spec, anchors, references):
    objective = parent_evaluate_policy(models, x, returns, spec=_parent_spec(spec, downside=0), anchors=anchors)
    constraints, legacy = {}, {}
    legacy_spec = _parent_spec(spec, downside=1)
    for cost in spec.cost_bps:
        logs = np.log(objective["paths"][cost]["net_equity"])
        constraints[cost] = evaluate_constraints(logs, references[cost],
            risk_tolerance_percentage_points=spec.risk_tolerance_percentage_points)
        relative = logs - np.log(anchors[cost]["net_equity"])
        old = utility_and_daily_weights(relative, spec=legacy_spec)
        legacy[cost] = float(old["utility"] + math.sqrt(spec.rms_epsilon))
    g = np.concatenate([constraints[c]["normalized_g"] for c in spec.cost_bps])
    return {"F": objective["J"], "A_by_cost": objective["centered_utility"],
            "cost_weights": objective["cost_weights"], "paths": objective["paths"],
            "daily_mean_log_excess_by_cost": {c: objective["components"][c]["total_log_excess"] / len(x) for c in spec.cost_bps},
            "aggregate_log_excess_by_cost": {c: objective["components"][c]["total_log_excess"] for c in spec.cost_bps},
            "legacy_J": min(legacy.values()), "legacy_centered_utility_by_cost": legacy,
            "constraints_by_cost": constraints, "normalized_g": g,
            "feasible": all(constraints[c]["feasible"] for c in spec.cost_bps),
            "objective_evaluation": objective}


def evaluate_policy(models, features, returns, *, spec: ConstrainedNetExcessSpec,
                    anchors=None, references=None):
    x, r = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(r) or not len(x) or (r <= -1).any():
        raise ValueError("blocked: invalid constrained chronological inputs")
    anchors = anchors or {c: rollout_alphas(np.zeros((len(r), 3)), r, cost_bps=c) for c in spec.cost_bps}
    references = references or {c: reference_risk(np.log(anchors[c]["net_equity"])) for c in spec.cost_bps}
    return _evaluate(models, x, r, spec=spec, anchors=anchors, references=references)


def feasible_improvement(evaluation, incumbent_F):
    """Legacy RMS utility and a changing Lagrangian never enter acceptance."""
    return bool(evaluation["feasible"] and evaluation["F"] > incumbent_F)


def gradient_proxy(evaluation, multipliers, *, spec: ConstrainedNetExcessSpec):
    """Global maximization weights for the immutable parent TBPTT kernel.

    Both cost branches are active in this *proxy*; their objective terms already
    contain the true worst-cost branch weights. The original F evaluation is
    not modified, and constraint terms are never gated by its inactive branch.
    """
    dual = np.asarray(multipliers, dtype=float)
    if dual.shape != (4,) or not np.isfinite(dual).all() or (dual < 0).any():
        raise NumericalFitFailure("invalid gradient risk multipliers")
    original = evaluation["objective_evaluation"]
    components, hashes = {}, {}
    risk_hash = hashlib.sha256()
    for i, cost in enumerate(spec.cost_bps):
        F_weights = original["components"][cost]["daily_weights"] * original["cost_weights"][cost]
        constraint = evaluation["constraints_by_cost"][cost]
        combined = combined_daily_adjoints(F_weights, constraint, dual[2*i:2*i+2])
        components[cost] = {"daily_weights": combined}
        hashes[cost] = _hash_array(combined)
        risk_hash.update(np.asarray(constraint["normalized_adjoints"], dtype=float).tobytes())
    return {"paths": original["paths"], "components": components,
            "cost_weights": {c: 1. for c in spec.cost_bps}}, {
            "combined_daily_weights_sha256_by_cost": hashes,
            "normalized_risk_adjoints_sha256": risk_hash.hexdigest()}


def _record(epoch, model, current, selected_epoch, dual_used, dual_next, *, gradient=None,
            primal_step_norm=0., pre_primal_lagrangian=None):
    theta = model.theta.detach().numpy()
    post_lagrangian = float(-current["F"] + dual_used @ current["normalized_g"])
    if not np.isfinite(post_lagrangian):
        raise NumericalFitFailure("nonfinite diagnostic Lagrangian")
    return {"epoch": epoch, "F": current["F"], "A_by_cost": current["A_by_cost"],
            "centered_worst_utility": current["F"], "centered_utility_by_cost": current["A_by_cost"],
            "daily_mean_log_excess_by_cost": current["daily_mean_log_excess_by_cost"],
            "aggregate_log_excess_by_cost": current["aggregate_log_excess_by_cost"],
            "legacy_J": current["legacy_J"], "legacy_centered_utility_by_cost": current["legacy_centered_utility_by_cost"],
            "cost_weights": current["cost_weights"], "feasible": current["feasible"],
            "risks_by_cost": {c: _risk_record(v) for c, v in current["constraints_by_cost"].items()},
            "lambda_used": dual_used.tolist(), "postg": current["normalized_g"].tolist(),
            "lambda_next": dual_next.tolist(), "dual_update_applied": epoch > 0,
            "pre_primal_lagrangian": pre_primal_lagrangian,
            "post_primal_lagrangian_at_lambda_used": post_lagrangian,
            "theta_mass": float(theta.sum()), "theta_active_coordinates": int((theta > 0).sum()),
            "theta_minimum": float(theta.min()), "primal_step_norm": float(primal_step_norm),
            "selected_epoch_so_far": selected_epoch, "gradient": gradient,
            "checkpoint": policy_to_dict(model)}


def _fallback_evaluation(anchors, costs):
    # Same-path raw feasibility remains exact even if normalization cannot be
    # used. A numerical failure still disqualifies this accounting-only null.
    constraints = {}
    for c in costs:
        risk = evaluate_risk(np.log(anchors[c]["net_equity"]))
        constraints[c] = {"risk": risk, "normalizers": np.array([risk["drawdown_loss"], risk["cvar5_loss"]]),
                          "raw_loss_gaps": np.zeros(2), "raw_percentage_point_gaps": np.zeros(2),
                          "risk_pass": np.ones(2, dtype=bool), "feasible": True}
    return {"F": 0., "A_by_cost": {c: 0. for c in costs}, "paths": anchors,
            "legacy_J": 0., "legacy_centered_utility_by_cost": {c: 0. for c in costs},
            "daily_mean_log_excess_by_cost": {c: 0. for c in costs},
            "aggregate_log_excess_by_cost": {c: 0. for c in costs},
            "constraints_by_cost": constraints, "normalized_g": None, "feasible": True}


def fit_policy(features, returns, *, seed: int, spec: ConstrainedNetExcessSpec):
    x, ret = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(ret) or len(x) < spec.minimum_rows or (ret <= -1).any():
        raise ValueError("blocked: insufficient or invalid mature chronological rows")
    history, failure, epoch, boundaries = [], None, "reference", None
    optimizer = None
    optimizer_steps = 0
    with _cpu_threads(spec.torch_threads):
        learner = BenchmarkAnchorPolicy(seed=seed)
        anchor = BenchmarkAnchorPolicy(seed=seed, anchor=True)
        initial_payload = policy_to_dict(learner)
        anchors = {c: rollout_alphas(np.zeros((len(ret), 3)), ret, cost_bps=c) for c in spec.cost_bps}
        anchor_audit = _audit_anchor_paths(anchors, ret)
        incumbent, incumbent_eval, selected_epoch = anchor, _fallback_evaluation(anchors, spec.cost_bps), "anchor"
        references = {}
        dual = np.zeros(4)
        try:
            references = {c: reference_risk(np.log(anchors[c]["net_equity"])) for c in spec.cost_bps}
            incumbent_eval = _evaluate(anchor, x, ret, spec=spec, anchors=anchors, references=references)
            if incumbent_eval["F"] != 0 or not incumbent_eval["feasible"] or np.any(incumbent_eval["normalized_g"] != 0):
                raise NumericalFitFailure("exact SOXX anchor is not F0 and raw risk equality")
            epoch = 0
            current = _evaluate(learner, x, ret, spec=spec, anchors=anchors, references=references)
            if feasible_improvement(current, incumbent_eval["F"]):
                incumbent, incumbent_eval, selected_epoch = deepcopy(learner), current, 0
            history.append(_record(0, learner, current, selected_epoch, dual, dual))
            optimizer = torch.optim.Adam(learner.parameters(), lr=spec.learning_rate, weight_decay=0.)
            for epoch in range(1, spec.updates + 1):
                used = dual.copy()
                pre_lagrangian = float(-current["F"] + used @ current["normalized_g"])
                if not np.isfinite(pre_lagrangian):
                    raise NumericalFitFailure("nonfinite current Lagrangian")
                proxy, adjoint_audit = gradient_proxy(current, used, spec=spec)
                gradient = truncated_utility_gradient(learner, x, ret, proxy, spec=_parent_spec(spec, downside=0))
                if boundaries is None:
                    boundaries = gradient["chunk_boundaries"]
                elif gradient["chunk_boundaries"] != boundaries:
                    raise NumericalFitFailure("gradient chunk coverage changed during fit")
                gradient.pop("chunk_boundaries")  # Stored once per fit, not 120 duplicates.
                gradient["gradient_proxy_value"] = gradient.pop("surrogate_loss")
                gradient.update(adjoint_audit)
                gradient["parameter_epoch"] = epoch - 1
                gradient["unclipped_global_norm"] = float(torch.nn.utils.clip_grad_norm_(learner.parameters(), spec.gradient_clip))
                if not np.isfinite(gradient["unclipped_global_norm"]):
                    raise NumericalFitFailure("nonfinite global primal gradient norm")
                before = [p.detach().clone() for p in learner.parameters()]
                optimizer.step()
                optimizer_steps += 1
                with torch.no_grad():
                    learner.theta.copy_(project_theta(learner.theta))
                _validate_parameters(learner)
                step_norm = float(torch.sqrt(sum((p.detach()-b).square().sum() for p,b in zip(learner.parameters(),before))))
                if not np.isfinite(step_norm):
                    raise NumericalFitFailure("nonfinite projected primal step")
                current = _evaluate(learner, x, ret, spec=spec, anchors=anchors, references=references)
                dual = update_multipliers(used, current["normalized_g"], learning_rate=spec.dual_learning_rate)
                if feasible_improvement(current, incumbent_eval["F"]):
                    incumbent, incumbent_eval, selected_epoch = deepcopy(learner), current, epoch
                history.append(_record(epoch, learner, current, selected_epoch, used, dual,
                    gradient=gradient, primal_step_norm=step_norm, pre_primal_lagrangian=pre_lagrangian))
        except (NumericalFitFailure, TailConstraintError) as exc:
            failure = {"epoch": epoch, "reason": str(exc), "qualification_eligible": False,
                       "accounting_anchor_only": True}
            incumbent, incumbent_eval, selected_epoch = anchor, _fallback_evaluation(anchors, spec.cost_bps), "anchor"
        incumbent.eval()
    groups = risk_groups(len(x), spec.risk_group_length)
    metadata = {"spec": asdict(spec), "seed": int(seed), "training_rows": len(x),
        "objective_name": "worst_cost_mean_net_log_excess_scaled63", "selected_epoch": selected_epoch,
        "failure": failure, "qualification_eligible": failure is None,
        "null_selected": bool(torch.count_nonzero(incumbent.theta) == 0),
        "training_feasible": bool(incumbent_eval["feasible"]),
        "training_feasibility_passed": bool(incumbent_eval["feasible"]),
        "F": incumbent_eval["F"], "selected_F": incumbent_eval["F"],
        "centered_worst_utility": incumbent_eval["F"], "centered_utility_by_cost": incumbent_eval["A_by_cost"],
        "daily_mean_log_excess_by_cost": incumbent_eval["daily_mean_log_excess_by_cost"],
        "aggregate_log_excess_by_cost": incumbent_eval["aggregate_log_excess_by_cost"],
        "legacy_J": incumbent_eval["legacy_J"], "legacy_centered_utility_by_cost": incumbent_eval["legacy_centered_utility_by_cost"],
        "selected_risks_by_cost": {c: _risk_record(v) for c,v in incumbent_eval["constraints_by_cost"].items()},
        "selected_normalized_g": incumbent_eval["normalized_g"],
        "epochs": history, "history": history, "training_paths": incumbent_eval["paths"],
        "benchmark_net_equity": {c: anchors[c]["net_equity"] for c in spec.cost_bps},
        "benchmark_risk_normalizers": {c: v["normalizers"].tolist() for c,v in references.items()},
        "anchor_audit": anchor_audit, "initial_tensor_sha256": initial_payload["tensor_sha256"],
        "training_feature_sha256": _hash_array(x), "training_return_sha256": _hash_array(ret),
        "risk_groups": groups, "risk_group_overlap_observations": sum(b-a for a,b in groups)-len(x),
        "gradient_chunk_boundaries": boundaries,
        "final_optimizer_state": deepcopy(optimizer.state_dict()) if optimizer is not None else None,
        "optimizer_initial_settings": deepcopy(optimizer.defaults) if optimizer is not None else None,
        "optimizer_state_scope": "Final optimizer trajectory, not necessarily the retained incumbent; parameter epoch counts actual Adam steps",
        "optimizer_state_parameter_epoch": optimizer_steps if optimizer is not None else None,
        "optimizer_library_version": torch.__version__,
        "dual_order": ["10_drawdown", "10_cvar5", "25_drawdown", "25_cvar5"],
        "optimizer_diagnostics": {"updates_completed": optimizer_steps,
            "finite_post_update_audits_completed": max(0,len(history)-1), "initial_theta_mass": .5,
            "weight_decay": 0., "lambda_initial": [0.,0.,0.,0.], "lambda_final": dual.tolist(),
            "maximum_forward_parity_error": max((h["gradient"]["forward_parity_max_error"] for h in history if h["gradient"]),default=0.),
            "incumbent_separate_from_optimizer_trajectory": True, "per_epoch_daily_adjoints_saved": False,
            "normalized_constraints_change_raw_feasibility": False,
            "lagrangian_used_for_incumbent_ranking": False, "legacy_J_used_for_acceptance_or_gradient": False},
        "checkpoint_selection": "First strict F improvement among complete-path raw-tail-feasible states; exact SOXX wins ties",
        "gradient_scope": "Both-cost risk adjoints plus active worst-cost F adjoints; true carried-state63/21 TBPTT; exact full-path incumbent audit",
        "gradient_limits": "Nonsmooth fees/risk ties and truncated state effects; no global optimum, KKT or forward-risk guarantee",
        "no_reinforcement_learning": True}
    return incumbent, metadata


def evaluate_ensemble(models, features, returns, *, spec: ConstrainedNetExcessSpec):
    policies = list(models)
    if len(policies) != 3:
        raise ValueError("blocked: exactly all three registered seed models are required")
    x, ret = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(ret) or not len(x) or (ret <= -1).any():
        raise ValueError("blocked: invalid ensemble chronological inputs")
    anchors = {c: rollout_alphas(np.zeros((len(ret),3)),ret,cost_bps=c) for c in spec.cost_bps}
    anchor_audit = _audit_anchor_paths(anchors,ret)
    failure = None
    try:
        references = {c: reference_risk(np.log(anchors[c]["net_equity"])) for c in spec.cost_bps}
        result = _evaluate(policies,x,ret,spec=spec,anchors=anchors,references=references)
        rejected = result["F"] < 0 or not result["feasible"]
        selected = [BenchmarkAnchorPolicy(seed=policies[0].seed,anchor=True)] if rejected else policies
        selected_eval = _evaluate(selected,x,ret,spec=spec,anchors=anchors,references=references) if rejected else result
    except (NumericalFitFailure,TailConstraintError) as exc:
        failure = {"reason":str(exc),"qualification_eligible":False,"accounting_anchor_only":True}
        rejected = True
        selected = [BenchmarkAnchorPolicy(seed=policies[0].seed,anchor=True)]
        result = selected_eval = _fallback_evaluation(anchors,spec.cost_bps)
    audit = {"component_count":3,"all_three_seed_models_used":True,"failure":failure,
        "training_feasible":bool(selected_eval["feasible"]),
        "training_feasibility_passed":bool(selected_eval["feasible"]),
        "qualification_eligible":failure is None,
        "ensemble_F":result["F"],"ensemble_centered_utility":result["A_by_cost"],
        "ensemble_feasible":result["feasible"],"ensemble_legacy_J":result["legacy_J"],
        "ensemble_risks_by_cost":{c:_risk_record(v) for c,v in result["constraints_by_cost"].items()},
        "ensemble_rejected_for_training_utility_or_risk":rejected,
        "anchor_selected":all(bool(torch.count_nonzero(m.theta)==0) for m in selected),
        "null_selected":all(bool(torch.count_nonzero(m.theta)==0) for m in selected),
        "F":selected_eval["F"],"selected_F":selected_eval["F"],
        "centered_worst_utility":selected_eval["F"],"centered_utility_by_cost":selected_eval["A_by_cost"],
        "legacy_J":selected_eval["legacy_J"],"legacy_centered_utility_by_cost":selected_eval["legacy_centered_utility_by_cost"],
        "selected_risks_by_cost":{c:_risk_record(v) for c,v in selected_eval["constraints_by_cost"].items()},
        "selected_normalized_g":selected_eval["normalized_g"],"training_paths":selected_eval["paths"],
        "rejected_ensemble_training_paths":result["paths"] if rejected and failure is None else None,
        "anchor_audit":anchor_audit,"legacy_J_is_report_only":True}
    return selected,audit


__all__ = ["ConstrainedNetExcessSpec","fit_policy","evaluate_policy","evaluate_ensemble",
           "gradient_proxy","update_multipliers","feasible_improvement",
           "BenchmarkAnchorPolicy","policy_to_dict","policy_from_dict"]
