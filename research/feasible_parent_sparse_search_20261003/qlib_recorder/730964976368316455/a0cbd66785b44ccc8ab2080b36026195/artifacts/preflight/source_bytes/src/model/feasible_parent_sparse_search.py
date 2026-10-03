"""Finite constrained search around an immutable feasible parent policy.

The search changes33 gauge-fixed head coordinates, not the frozen encoder.
Every proposal uses the exact two-cost owned portfolio. Solver termination is
an optimization diagnostic; raw full-path risk gates alone admit incumbents.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
import traceback

import numpy as np
import scipy
from scipy.optimize import minimize
import torch

from src.model.benchmark_anchor_utility import (
    BenchmarkAnchorPolicy, NumericalFitFailure, _audit_anchor_paths, _matrix,
    numpy_rollout as legacy_rollout, policy_from_dict as parent_from_dict,
    policy_to_dict as parent_to_dict, utility_and_daily_weights,
)
from src.model.constrained_net_excess import (
    ConstrainedNetExcessSpec, _risk_record, _parent_spec,
)
from src.model.integrated_allocation import _cpu_threads
from src.model.sparse_holdings_net_excess import (
    SparseHoldingsPolicy, _path_diagnostics, embed_parent_policy,
    is_exact_anchor, numpy_rollout,
    policy_from_dict, policy_target, policy_to_dict, replay_policy, zero_anchor,
)
from src.objective.benchmark_tail_constraints import TailConstraintError, reference_risk, evaluate_constraints
from src.model.paired_sparse_paths import rollout_costs


@dataclass(frozen=True)
class FeasibleParentSearchSpec:
    max_evaluations: int = 2000
    scipy_maxiter: int = 2000
    rhobeg: float = 1.
    tol: float = 1e-6
    catol: float = 0.
    cost_bps: tuple[float, float] = (10., 25.)
    minimum_rows: int = 252
    torch_threads: int = 1
    risk_tolerance_percentage_points: float = 1e-10
    risk_group_length: int = 63
    rms_epsilon: float = 1e-12

    def __post_init__(self):
        if (not isinstance(self.max_evaluations, int) or self.max_evaluations < 1
                or not isinstance(self.scipy_maxiter, int) or self.scipy_maxiter < 1
                or self.minimum_rows < 63 or self.torch_threads < 1):
            raise ValueError("blocked: invalid search evaluation/row/thread budget")
        if (tuple(self.cost_bps) != (10., 25.) or self.risk_group_length != 63
                or self.rms_epsilon != 1e-12 or self.risk_tolerance_percentage_points != 1e-10):
            raise ValueError("blocked: original objective and raw risk gates are fixed")
        if not np.isfinite([self.rhobeg, self.tol, self.catol]).all() or self.rhobeg <= 0 or self.tol <= 0 or self.catol != 0:
            raise ValueError("blocked: invalid COBYLA numerical settings")

    def evaluation_spec(self):
        # Only value evaluation uses this sealed contract. No legacy optimizer,
        # update, gradient or dual variable is invoked by the new search.
        return ConstrainedNetExcessSpec(minimum_rows=self.minimum_rows,
            torch_threads=self.torch_threads, cost_bps=tuple(self.cost_bps),
            risk_tolerance_percentage_points=self.risk_tolerance_percentage_points)


SearchSpec = FeasibleParentSearchSpec
FREE_ROWS = (0, 2, 3)


class ParentContractError(ValueError):
    """A wrong or infeasible parent cannot silently become a new start."""


class EvaluationBudgetExceeded(RuntimeError):
    """The fixed full-path budget ended; not a numerical failure."""


def _hash_array(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def encoder_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.hidden.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.asarray(value.detach().cpu().numpy(), dtype='<f8').tobytes())
    return digest.hexdigest()


def _coordinates(value):
    result = np.asarray(value, dtype='<f8')
    if result.shape != (33,) or not np.isfinite(result).all():
        raise NumericalFitFailure("invalid33-coordinate search proposal")
    return np.ascontiguousarray(result)


def gauge_coordinates(model):
    """Remove common-score and bias/holdings-simplex null directions."""
    policy_to_dict(model)
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    A = state['A'] - state['A'][1]
    a = state['a'] - state['a'][1]
    K = state['K'] - state['K'][1]
    head = np.column_stack((A[list(FREE_ROWS)],
        (a + K[:, 1])[list(FREE_ROWS)],
        (K[:, 0] - K[:, 1])[list(FREE_ROWS)],
        (K[:, 2] - K[:, 1])[list(FREE_ROWS)]))
    return _coordinates(head.reshape(-1)).copy()


def model_from_coordinates(template, coordinates):
    """Decode only the head. Canonical null remains a separate incumbent."""
    policy_to_dict(template)
    head = _coordinates(coordinates).reshape(3, 11)
    model = deepcopy(template)
    with torch.no_grad():
        model.A.zero_(); model.a.zero_(); model.K.zero_()
        model.A[list(FREE_ROWS)] = torch.from_numpy(head[:, :8].copy())
        model.a[list(FREE_ROWS)] = torch.from_numpy(head[:, 8].copy())
        model.K[list(FREE_ROWS), 0] = torch.from_numpy(head[:, 9].copy())
        model.K[list(FREE_ROWS), 2] = torch.from_numpy(head[:, 10].copy())
    model.requires_grad_(False)
    return model.eval()


def prepare_parent(parent_model, *, seed=None):
    """Return a validated exact old parent and its equivalent gauge template."""
    parent = parent_from_dict(parent_model) if isinstance(parent_model, dict) else deepcopy(parent_model)
    if not isinstance(parent, BenchmarkAnchorPolicy):
        raise ParentContractError("blocked: original BenchmarkAnchorPolicy parent required")
    parent_to_dict(parent)
    if seed is not None and parent.seed != int(seed):
        raise ParentContractError("blocked: parent seed identity differs")
    embedded = embed_parent_policy(parent)
    model = model_from_coordinates(embedded, gauge_coordinates(embedded))
    if encoder_hash(parent) != encoder_hash(model):
        raise ParentContractError("blocked: embedding changed frozen parent encoder")
    return parent, model


def evaluate_policy(models, features, returns, *, spec=None, anchors=None, references=None):
    spec = spec or FeasibleParentSearchSpec()
    x, r = _panel(features, returns, 1)
    if anchors is None:
        anchors = {c: numpy_rollout(zero_anchor(), x, r, cost_bps=c) for c in spec.cost_bps}
    if references is None:
        references = {c: reference_risk(np.log(anchors[c]['net_equity'])) for c in spec.cost_bps}
    paths = rollout_costs(models, x, r, costs=spec.cost_bps)
    return evaluate_paths(paths, anchors, references, spec=spec)


def evaluate_paths(paths, anchors, references, *, spec):
    """SealedF/risk arithmetic with a verified shared-state path producer."""
    components, constraints, legacy = {}, {}, {}
    mean_spec = _parent_spec(spec.evaluation_spec(), downside=0)
    old_spec = _parent_spec(spec.evaluation_spec(), downside=1)
    for cost in spec.cost_bps:
        logs = np.log(paths[cost]['net_equity'])
        relative = logs - np.log(anchors[cost]['net_equity'])
        components[cost] = utility_and_daily_weights(relative, spec=mean_spec)
        constraints[cost] = evaluate_constraints(logs, references[cost],
            risk_tolerance_percentage_points=spec.risk_tolerance_percentage_points)
        legacy[cost] = utility_and_daily_weights(relative, spec=old_spec)['utility'] + math.sqrt(spec.rms_epsilon)
    A = {c: components[c]['utility'] for c in spec.cost_bps}
    F = min(A.values())
    active = [c for c,v in A.items() if v == F]
    weights = {c: 1/len(active) if c in active else 0. for c in spec.cost_bps}
    return {'F':F, 'A_by_cost':A, 'cost_weights':weights, 'paths':paths,
        'daily_mean_log_excess_by_cost':{c:components[c]['total_log_excess']/len(paths[c]['net_equity']) for c in spec.cost_bps},
        'aggregate_log_excess_by_cost':{c:components[c]['total_log_excess'] for c in spec.cost_bps},
        'legacy_J':min(legacy.values()), 'legacy_centered_utility_by_cost':legacy,
        'constraints_by_cost':constraints,
        'normalized_g':np.concatenate([constraints[c]['normalized_g'] for c in spec.cost_bps]),
        'feasible':all(constraints[c]['feasible'] for c in spec.cost_bps),
        'objective_evaluation':{'J':F, 'centered_utility':A, 'cost_weights':weights,
                              'components':components, 'paths':paths, 'anchor_utility':0.}}


def _summary(evaluation):
    return {"F": float(evaluation['F']), "A_by_cost": evaluation['A_by_cost'],
        "centered_worst_utility": float(evaluation['F']),
        "centered_utility_by_cost": evaluation['A_by_cost'],
        "daily_mean_log_excess_by_cost": evaluation['daily_mean_log_excess_by_cost'],
        "aggregate_log_excess_by_cost": evaluation['aggregate_log_excess_by_cost'],
        "legacy_J": evaluation['legacy_J'],
        "legacy_centered_utility_by_cost": evaluation['legacy_centered_utility_by_cost'],
        "feasible": bool(evaluation['feasible']),
        "risks_by_cost": {c: _risk_record(v) for c, v in evaluation['constraints_by_cost'].items()},
        "normalized_g": (np.asarray(evaluation['normalized_g']).copy()
                         if evaluation['normalized_g'] is not None else None),
        "action_diagnostics": {c: _path_diagnostics(p) for c,p in evaluation['paths'].items()}}


def _panel(features, returns, minimum_rows):
    x, r = _matrix(features, 27, 'features'), _matrix(returns, 2, 'returns')
    if len(x) != len(r) or len(x) < minimum_rows or (r <= -1).any():
        raise ValueError("blocked: invalid or insufficient mature chronological panel")
    return x, r


def _reference(x, r, spec, seed):
    anchor = zero_anchor(seed)
    anchors = {c: numpy_rollout(anchor, x, r, cost_bps=c) for c in spec.cost_bps}
    audit = _audit_anchor_paths(anchors, r)
    references = {c: reference_risk(np.log(p['net_equity'])) for c,p in anchors.items()}
    return anchor, anchors, references, audit


def revalidate_parent(parent_model, features, returns, *, spec=None, seed=None,
                      anchors=None, references=None):
    """Compare original fixed forward policy with current gauge/kernel values."""
    spec = spec or FeasibleParentSearchSpec()
    x, r = _panel(features, returns, spec.minimum_rows)
    parent, embedded = prepare_parent(parent_model, seed=seed)
    if anchors is None or references is None:
        _, anchors, references, _ = _reference(x, r, spec, parent.seed)
    current = evaluate_policy(embedded, x, r, spec=spec, anchors=anchors, references=references)
    error_by_field, nav_error, F_error = {}, 0., 0.
    for cost in spec.cost_bps:
        old = legacy_rollout(parent, x, r, cost_bps=cost)
        new = current['paths'][cost]
        for field in ('target_weights','pretrade_weights','ending_weights','fees','net_equity'):
            error_by_field[field] = max(error_by_field.get(field, 0.),
                float(np.max(np.abs(old[field] - new[field]))))
        nav_error = max(nav_error, float(np.max(np.abs(new['equity']/old['equity'] - 1))))
        old_A = 63. * np.mean(np.log(old['net_equity']) - np.log(anchors[cost]['net_equity']))
        F_error = max(F_error, abs(float(old_A) - current['A_by_cost'][cost]))
    if max(error_by_field.values()) > 1e-12 or nav_error > 1e-10 or F_error > 1e-12:
        raise ParentContractError("blocked: original parent/gauge/current-kernel path parity failed")
    if not current['feasible'] or current['F'] < 0:
        raise ParentContractError("blocked: embedded parent failed original raw risk or F gate")
    return embedded, current, {"passed": True, "maximum_daily_error_by_field": error_by_field,
        "maximum_relative_NAV_error": nav_error, "maximum_A_F_error": F_error,
        "parent_tensor_sha256": parent_to_dict(parent)['tensor_sha256'],
        "embedded_tensor_sha256": policy_to_dict(embedded)['tensor_sha256'],
        "frozen_encoder_sha256": encoder_hash(embedded),
        "source_parent_seed": parent.seed}


def _anchor_like(template):
    model = zero_anchor(template.seed)
    model.hidden.load_state_dict(template.hidden.state_dict())
    model.requires_grad_(False)
    return model.eval()


class SearchEvaluator:
    """One exact-byte cache shared by SciPy objective and constraint callbacks.

    The initial parent evaluation is injected from its mandatory revalidation
    and counts as unique evaluation1. Cache entries contain no daily arrays.
    """
    def __init__(self, template, x, returns, *, spec, anchors, references, parent_evaluation):
        self.template, self.x, self.returns = template, x, returns
        self.spec, self.anchors, self.references = spec, anchors, references
        self.encoder_sha256 = encoder_hash(template)
        self.cache, self.trials = {}, []
        self.requested_callbacks = {"objective": 0, "constraints": 0, "initial_parent": 1}
        self.cache_hits = 0
        self.attempted_full_path_evaluations = 1
        self.last_requested_coordinates = None
        self.incumbent = _anchor_like(template)
        self.incumbent_evaluation = evaluate_paths(anchors, anchors, references, spec=spec)
        self.selected_origin, self.selected_trial = 'null', 'anchor'
        if parent_evaluation['feasible'] and parent_evaluation['F'] > 0:
            self.incumbent, self.incumbent_evaluation = deepcopy(template), parent_evaluation
            self.selected_origin, self.selected_trial = 'parent', 'parent'
        self.current_evaluation = parent_evaluation
        self._register(gauge_coordinates(template), template, parent_evaluation, initial=True)

    def _register(self, coordinates, model, evaluation, *, initial=False):
        values = [evaluation['F'], *evaluation['A_by_cost'].values(), *evaluation['normalized_g']]
        if not np.isfinite(values).all():
            raise NumericalFitFailure("nonfinite whole-path search objective/constraints")
        if encoder_hash(model) != self.encoder_sha256:
            raise NumericalFitFailure("search changed frozen encoder")
        improves = not initial and evaluation['feasible'] and evaluation['F'] > self.incumbent_evaluation['F']
        selected_model = deepcopy(model) if improves else None
        next_origin = 'search' if improves else self.selected_origin
        next_trial = len(self.trials) if improves else self.selected_trial
        record = _summary(evaluation)
        record.update({"trial_id": len(self.trials), "coordinates": coordinates.copy(),
            "coordinate_sha256": _hash_array(coordinates), "initial_parent": bool(initial),
            "tensor_sha256": policy_to_dict(model)['tensor_sha256'],
            "selected_origin_so_far": next_origin,
            "selected_trial_so_far": next_trial})
        self.cache[coordinates.tobytes()] = record
        self.trials.append(record)
        if improves:
            self.incumbent, self.incumbent_evaluation = selected_model, evaluation
            self.selected_origin, self.selected_trial = next_origin, next_trial
        self.current_evaluation = evaluation
        return record

    def evaluate(self, coordinates, *, caller):
        self.requested_callbacks[caller] = self.requested_callbacks.get(caller, 0) + 1
        self.last_requested_coordinates = np.asarray(coordinates).copy()
        coordinates = _coordinates(coordinates)
        key = coordinates.tobytes()
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        if self.attempted_full_path_evaluations >= self.spec.max_evaluations:
            raise EvaluationBudgetExceeded("registered unique whole-two-cost-path budget exhausted")
        self.attempted_full_path_evaluations += 1
        model = model_from_coordinates(self.template, coordinates)
        evaluation = evaluate_policy(model, self.x, self.returns, spec=self.spec,
            anchors=self.anchors, references=self.references)
        return self._register(coordinates, model, evaluation)

    def objective(self, coordinates):
        return -self.evaluate(coordinates, caller='objective')['F']

    def constraints(self, coordinates):
        return -self.evaluate(coordinates, caller='constraints')['normalized_g'].copy()


def _solver_summary(result):
    if result is None:
        return None
    report = {k: getattr(result, k, None) for k in ['success','status','message','nfev','fun','maxcv']}
    for key in ['success']:
        if report[key] is not None: report[key] = bool(report[key])
    for key in ['status','nfev']:
        if report[key] is not None: report[key] = int(report[key])
    for key in ['fun','maxcv']:
        if report[key] is not None: report[key] = float(report[key])
    report['message'] = str(report['message'])
    if getattr(result, 'x', None) is not None:
        report['returned_coordinates'] = _coordinates(result.x).copy()
        report['returned_coordinate_sha256'] = _hash_array(result.x)
    if any(report[k] is not None and not math.isfinite(report[k]) for k in ('fun','maxcv')):
        raise NumericalFitFailure("nonfinite COBYLA result diagnostic")
    return report


def fit_policy(features, returns, *, parent_model, seed, spec=None, parent_identity=None):
    """Search once from the verified parent; retain raw-feasible bestF only."""
    spec = spec or FeasibleParentSearchSpec()
    x, r = _panel(features, returns, spec.minimum_rows)
    with _cpu_threads(spec.torch_threads):
        _, anchors, references, anchor_audit = _reference(x, r, spec, seed)
        template, parent_eval, parent_audit = revalidate_parent(parent_model, x, r,
            spec=spec, seed=seed, anchors=anchors, references=references)
        evaluator = SearchEvaluator(template, x, r, spec=spec, anchors=anchors,
            references=references, parent_evaluation=parent_eval)
        solver_result, failure = None, None
        termination = 'not_started'
        try:
            result = minimize(evaluator.objective, gauge_coordinates(template), method='COBYLA',
                constraints=[{'type': 'ineq', 'fun': evaluator.constraints}],
                options={'maxiter': spec.scipy_maxiter, 'rhobeg': spec.rhobeg,
                         'tol': spec.tol, 'catol': spec.catol, 'disp': 0})
            solver_result = _solver_summary(result)
            termination = 'solver_returned'
        except EvaluationBudgetExceeded as exc:
            termination = 'unique_evaluation_budget'
            solver_result = {'success': False, 'message': str(exc), 'nfev': None, 'status': None}
        except (NumericalFitFailure, TailConstraintError, FloatingPointError, OverflowError) as exc:
            termination = 'numerical_failure'
            failure = {'reason': str(exc), 'exception_type': type(exc).__name__,
                'kind': 'numerical_search_failure', 'accounting_only': True,
                'completed_unique_evaluations': len(evaluator.trials),
                'requested_coordinates': evaluator.last_requested_coordinates,
                'qualification_eligible': False,
                'retained_accounting_incumbent_origin': evaluator.selected_origin}
        except Exception as exc:
            # Preserve evidence for the runner to save before aborting the
            # campaign. Unknown code/solver errors are never healthy no-edge.
            # Parent/source/native validation occurs outside this boundary;
            # KeyboardInterrupt and SystemExit deliberately are not caught.
            termination = 'unexpected_solver_exception'
            failure = {'reason':str(exc), 'exception_type':type(exc).__name__,
                'kind':'unexpected_solver_exception', 'accounting_only':True,
                'qualification_eligible':False, 'campaign_abort_required':True,
                'traceback':traceback.format_exc(),
                'completed_unique_evaluations':len(evaluator.trials),
                'requested_coordinates':evaluator.last_requested_coordinates,
                'retained_accounting_incumbent_origin':evaluator.selected_origin}
        model, chosen = evaluator.incumbent, evaluator.incumbent_evaluation
        before, after = parent_audit['frozen_encoder_sha256'], encoder_hash(model)
        if before != after:
            raise NumericalFitFailure("retained model encoder differs from parent")
        chosen_summary = _summary(chosen)
        metadata = {
            'schema_version': 'feasible_parent_search_fit_v1', 'spec': asdict(spec),
            'seed': int(seed), 'training_rows': len(x), 'trainable_search_coordinates': 33,
            'frozen_encoder_sha256_before': before, 'frozen_encoder_sha256_after': after,
            'parent_identity': deepcopy(parent_identity), 'parent_revalidation': parent_audit,
            'parent_model': policy_to_dict(template), 'parent_evaluation': _summary(parent_eval),
            'parent_training_paths': parent_eval['paths'],
            'parent_F': parent_eval['F'], 'F_delta_from_parent': chosen['F'] - parent_eval['F'],
            'F_delta': chosen['F'] - parent_eval['F'],
            'selected_origin': evaluator.selected_origin, 'selected_trial': evaluator.selected_trial,
            'selected_F': chosen['F'], 'F': chosen['F'],
            'centered_worst_utility': chosen['F'], 'centered_utility_by_cost': chosen['A_by_cost'],
            'daily_mean_log_excess_by_cost': chosen['daily_mean_log_excess_by_cost'],
            'aggregate_log_excess_by_cost': chosen['aggregate_log_excess_by_cost'],
            'legacy_J': chosen['legacy_J'], 'legacy_centered_utility_by_cost': chosen['legacy_centered_utility_by_cost'],
            'selected_risks_by_cost': chosen_summary['risks_by_cost'],
            'selected_normalized_g': chosen_summary['normalized_g'],
            'selected_action_diagnostics': chosen_summary['action_diagnostics'],
            'training_paths': chosen['paths'], 'benchmark_net_equity': {c:p['net_equity'] for c,p in anchors.items()},
            'training_feasible': bool(chosen['feasible']), 'training_feasibility_passed': bool(chosen['feasible']),
            'null_selected': is_exact_anchor(model), 'failure': failure, 'qualification_eligible': failure is None,
            'accounting_only': failure is not None,
            'trials': evaluator.trials,
            'trial_count': len(evaluator.trials), 'unique_evaluations': len(evaluator.trials),
            'attempted_full_path_evaluations': evaluator.attempted_full_path_evaluations,
            'requested_callbacks': evaluator.requested_callbacks, 'cache_hits': evaluator.cache_hits,
            'solver_nfev': solver_result.get('nfev') if solver_result else None,
            'termination': termination,
            'optimizer_diagnostics': {
                'method': 'COBYLA', 'library': 'scipy', 'library_version': scipy.__version__,
                'options': {'maxiter': spec.scipy_maxiter, 'rhobeg': spec.rhobeg, 'tol': spec.tol, 'catol': spec.catol, 'disp':0},
                'unique_full_path_evaluations': len(evaluator.trials),
                'attempted_full_path_evaluations': evaluator.attempted_full_path_evaluations,
                'unique_full_path_cap': spec.max_evaluations,
                'requested_callbacks': evaluator.requested_callbacks,
                'cache_hits': evaluator.cache_hits, 'termination': termination, 'solver_result': solver_result,
                'solver_success_is_feasibility_certificate': False,
                'model_gradient_calls': 0, 'optimizer_restarts': 0,
                'final_verification_search_candidates': 0,
                'initial_parent_counts_as_unique_evaluation': True,
                'source_parity_reference_paths_outside_search': 2,
                'full_daily_arrays_saved_per_trial': False,
            },
            'training_feature_sha256': _hash_array(x), 'training_return_sha256': _hash_array(r),
            'anchor_audit': anchor_audit,
            'checkpoint_selection': 'Raw feasible strictF improvement, ties canonicalSOXX thenparent thenearliestsearch',
            'legacy_J_is_report_only': True, 'no_reinforcement_learning': True,
            'limitations': 'Finite derivative-free local search; no optimality or forward-risk guarantee. Frozenencoder/warmstart/capacity/optimizer/budget changed jointly.',
        }
    return model, metadata


def evaluate_ensemble(models, features, returns, *, parent_models, spec=None):
    """Compare only three complete owned books, never component subsets."""
    spec = spec or FeasibleParentSearchSpec()
    candidates = list(models); supplied_parents = list(parent_models)
    if len(candidates) != 3 or len(supplied_parents) != 3:
        raise ValueError("blocked: all three candidate and parent seed models required")
    if [m.seed for m in candidates] != [17,41,71]:
        raise ValueError("blocked: declared seed order17,41,71 required")
    x, r = _panel(features, returns, spec.minimum_rows)
    parents = []
    for candidate, value in zip(candidates, supplied_parents):
        if isinstance(value, SparseHoldingsPolicy):
            parent = deepcopy(value)
            if parent.seed != candidate.seed: raise ParentContractError("blocked: ensemble parent seed differs")
        else:
            _, parent = prepare_parent(value, seed=candidate.seed)
        if encoder_hash(candidate) != encoder_hash(parent):
            raise ParentContractError("blocked: ensemble candidate/parent encoder identity differs")
        parents.append(parent)
    anchor = _anchor_like(parents[0])
    _, anchors, references, anchor_audit = _reference(x, r, spec, anchor.seed)
    chosen = evaluate_paths(anchors, anchors, references, spec=spec)
    selected, origin, failure = [anchor], 'null', None
    parent_eval = candidate_eval = None
    try:
        parent_eval = evaluate_policy(parents, x, r, spec=spec, anchors=anchors, references=references)
        if parent_eval['feasible'] and parent_eval['F'] > chosen['F']:
            selected, chosen, origin = parents, parent_eval, 'parent'
        candidate_eval = evaluate_policy(candidates, x, r, spec=spec, anchors=anchors, references=references)
        if candidate_eval['feasible'] and candidate_eval['F'] > chosen['F']:
            selected, chosen, origin = candidates, candidate_eval, 'search'
    except (NumericalFitFailure, TailConstraintError, FloatingPointError, OverflowError) as exc:
        failure = {'reason':str(exc), 'exception_type':type(exc).__name__,
                   'kind':'numerical_ensemble_failure', 'accounting_only':True,
                   'qualification_eligible':False, 'retained_accounting_incumbent_origin':origin}
    selected_summary = _summary(chosen)
    parent_summary = _summary(parent_eval) if parent_eval is not None else None
    candidate_summary = _summary(candidate_eval) if candidate_eval is not None else None
    return selected, {
        'schema_version': 'feasible_parent_search_ensemble_v1', 'component_count':3,
        'all_three_seed_models_used':True, 'selected_origin':origin,
        'selected_component_tensor_sha256': [policy_to_dict(m)['tensor_sha256'] for m in selected],
        'candidate_component_tensor_sha256': [policy_to_dict(m)['tensor_sha256'] for m in candidates],
        'parent_component_tensor_sha256': [policy_to_dict(m)['tensor_sha256'] for m in parents],
        'candidate_ensemble':candidate_summary, 'parent_ensemble':parent_summary,
        'candidate_F':candidate_eval['F'] if candidate_eval is not None else None,
        'candidate_feasible':candidate_eval['feasible'] if candidate_eval is not None else False,
        'candidate_failure':failure if candidate_eval is None else None,
        'parent_ensemble_feasible':parent_eval['feasible'] if parent_eval is not None else False,
        'parent_ensemble_failure':failure if parent_eval is None else None,
        'candidate_ensemble_F':candidate_eval['F'] if candidate_eval is not None else None,
        'parent_ensemble_F':parent_eval['F'] if parent_eval is not None else None,
        'candidate_minus_parent_F':candidate_eval['F']-parent_eval['F'] if candidate_eval is not None and parent_eval is not None else None,
        'candidate_training_paths':candidate_eval['paths'] if candidate_eval is not None else None,
        'parent_training_paths':parent_eval['paths'] if parent_eval is not None else None,
        'training_paths':chosen['paths'], 'F':chosen['F'], 'selected_F':chosen['F'],
        'centered_worst_utility':chosen['F'], 'centered_utility_by_cost':chosen['A_by_cost'],
        'daily_mean_log_excess_by_cost':chosen['daily_mean_log_excess_by_cost'],
        'aggregate_log_excess_by_cost':chosen['aggregate_log_excess_by_cost'],
        'legacy_J':chosen['legacy_J'], 'legacy_centered_utility_by_cost':chosen['legacy_centered_utility_by_cost'],
        'selected_risks_by_cost':selected_summary['risks_by_cost'],
        'selected_normalized_g':selected_summary['normalized_g'],
        'selected_action_diagnostics':selected_summary['action_diagnostics'],
        'training_feasible':bool(chosen['feasible']), 'training_feasibility_passed':bool(chosen['feasible']),
        'anchor_selected':origin=='null', 'null_selected':origin=='null',
        'failure':failure, 'qualification_eligible':failure is None,
        'accounting_only':failure is not None,
        'anchor_audit':anchor_audit, 'legacy_J_is_report_only':True,
        'incumbent_comparison': 'Three full books only; ties null,parent,candidate. No hybrid or subset enumeration.',
    }


__all__ = ['FeasibleParentSearchSpec','SearchSpec','SparseHoldingsPolicy','prepare_parent',
    'revalidate_parent','gauge_coordinates','model_from_coordinates','encoder_hash',
    'policy_to_dict','policy_from_dict','policy_target','numpy_rollout','replay_policy',
    'zero_anchor','is_exact_anchor','evaluate_policy','evaluate_paths','fit_policy','evaluate_ensemble',
    'SearchEvaluator','ParentContractError','EvaluationBudgetExceeded']
