"""Read-only summary of the already-retained failed synthetic experiment.

No rollout, fitting, optimizer step, market input, or candidate selection occurs.
"""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


BASE = Path('/tmp')
PAYLOAD = BASE / 'sparse-holdings-known-information-full.pkl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_row(e):
    return {
        'epoch': e['epoch'], 'F': e['F'], 'feasible': e['feasible'],
        'raw_gap_pp_in_cost10_D_C_cost25_D_C_order': [
            v for cost in [10.0, 25.0]
            for v in e['risks_by_cost'][cost]['raw_percentage_point_gaps']
        ],
        'lambda_next': e['lambda_next'],
        'gradient_parameter_epoch': (e['gradient'] or {}).get('parameter_epoch'),
        'encoder_gradient_norm': (e['gradient'] or {}).get('encoder_gradient_norm'),
        'head_gradient_norm': (e['gradient'] or {}).get('head_gradient_norm'),
        'K_gradient_norm': (e['gradient'] or {}).get('K_gradient_norm'),
        'primal_step_norm': e['primal_step_norm'],
    }


def main():
    p = pickle.loads(PAYLOAD.read_bytes())
    assert sha(PAYLOAD) == '3d589805851fc3c4a959fd00e53f9f5fd8409adef0d9c8da111c095b9c3f36d5'
    independent = json.loads((BASE / 'sparse-holdings-validation/known-information-independent-audit.json').read_text())
    report = {
        'schema_version': 'sparse_holdings_failed_optimizer_diagnosis_v1',
        'scope': 'Read only aggregation of retained synthetic states and latest paths; no third fit, new rollout, market data or policy selection.',
        'market_fits_permitted': False,
        'registered_training_gate_passed': False,
        'fixture': {'rows': 504, 'index_type': 'synthetic ordinal 0..503, not dates',
                    'features': 'x0 alternates +1/-1; other26 columns0',
                    'returns': 'SOXX=.002+.008*x0; SOXL=3*SOXX'},
        'retention_provenance': json.loads((BASE / 'sparse-holdings-failed-source/identity_receipt.json').read_text()),
        'initial_original_per_epoch_trajectories_retained': False,
        'original_vs_retention_all_state_identity_claimed': False,
        'registered_witness': p['prior_probe']['witness'],
        'independent_validation': {
            'path': '/tmp/sparse-holdings-validation/known-information-independent-audit.json',
            'sha256': sha('/tmp/sparse-holdings-validation/known-information-independent-audit.json'),
            'checkpoint_count': independent['checkpoint_count'],
            'checkpoint_cost_paths': independent['checkpoint_cost_paths'],
            'maximum_errors': independent['maximum_reconstruction_errors'],
            'failures': independent['failures'],
        },
        'seeds': [],
    }
    for fit in p['fits']:
        epochs = fit['metadata']['epochs']
        assert len(epochs) == 121 and [e['epoch'] for e in epochs] == list(range(121))
        all_s = [e['epoch'] for e in epochs if all(
            d['sole_expert_fraction']['soxx'] == 1.0 for d in e['action_diagnostics'].values())]
        zeros = [e['epoch'] for e in epochs if (e['gradient'] or {}).get('all_parameter_gradients_exact_zero')]
        maximum = max(epochs, key=lambda e: e['F'])
        item = {
            'seed': fit['seed'], 'numerical_failure': fit['metadata']['failure'],
            'selected_epoch': fit['metadata']['selected_epoch'], 'selected_F': fit['metadata']['F'],
            'states': len(epochs), 'positive_F_states': sum(e['F'] > 0 for e in epochs),
            'feasible_states': sum(e['feasible'] for e in epochs),
            'positive_F_feasible_states': sum(e['feasible'] and e['F'] > 0 for e in epochs),
            'negative_F_feasible_states': sum(e['feasible'] and e['F'] < 0 for e in epochs),
            'first_all_SOXX_singleton_epoch': all_s[0] if all_s else None,
            'all_SOXX_singleton_persists_to_end': all_s == list(range(all_s[0], 121)) if all_s else False,
            'first_zero_gradient_update': zeros[0] if zeros else None,
            'zero_gradient_update_count': len(zeros),
            'zero_gradient_persists_to_end': zeros == list(range(zeros[0], 121)) if zeros else False,
            'maximum_F_state': state_row(maximum),
            'final_state': state_row(epochs[-1]),
            'dual_minimum': min(min(e['lambda_next']) for e in epochs),
            'dual_maximum': max(max(e['lambda_next']) for e in epochs),
            'latest_path_details_by_cost': {},
        }
        for cost, path in fit['latest_evaluation']['paths'].items():
            q = path['expert_weights']; h = path['pretrade_weights']; z = path['target_weights']
            target_s = np.array([0., 1., 0.])
            item['latest_path_details_by_cost'][str(cost)] = {
                'q_first_six': q[:6].tolist(), 'q_final_two': q[-2:].tolist(),
                'target_first_six': z[:6].tolist(), 'target_final_two': z[-2:].tolist(),
                'positive_signal_mean_q': q[p['features'][:, 0] > 0].mean(0).tolist(),
                'negative_signal_mean_q': q[p['features'][:, 0] < 0].mean(0).tolist(),
                'active_experts': np.flatnonzero((q > 0).any(0)).tolist(),
                'sole_expert_diagnostics': epochs[-1]['action_diagnostics'][cost],
                'maximum_SOXL_weight': float(z[:, 0].max()),
                'maximum_cash_weight': float(z[:, 2].max()),
                'final_pretrade_distance_to_SOXX_L1': float(np.abs(h[-1] - target_s).sum()),
                'first20_net_log_excess': float(np.log(path['net_equity'][:20] /
                    fit['metadata']['benchmark_net_equity'][cost][:20]).sum()),
                'remaining484_net_log_excess': float(np.log(path['net_equity'][20:] /
                    fit['metadata']['benchmark_net_equity'][cost][20:]).sum()),
                'total_net_log_excess': float(np.log(path['net_equity'] /
                    fit['metadata']['benchmark_net_equity'][cost]).sum()),
            }
        report['seeds'].append(item)
    report['interpretation'] = [
        'Every positive-F learned state fails at least one unchanged raw DD/CVaR gate; the retained exact benchmark is the correct rejection.',
        'Seeds17/41 enter an all-SOXX singleton. The sparsemax score Jacobian is exactly zero there; later zero parameter gradients do not supply an escape direction. Adam momentum moves parameters without restoring active alternatives in this recorded budget.',
        'Seed71 retains support {SOXX,HOLD} and nonzero parameter gradients. Starting in cash causes partial entry, then owned holdings approach SOXX. The two active experts become economically coincident: d target / d q_H along q_S+q_H=1 equals h-e_SOXX and vanishes as h approaches SOXX. Inactive SOXL/cash score derivatives are zero inside this support cell.',
        'At maximumF, the independent reconstruction finds broadly similar SOXL exposure on positive and negative signals and no active cash expert. Positive gross edge and rising tail violations precede reversion toward the benchmark; this does not establish useful conditional allocation learning.',
        'A representable fixed witness has positive feasibleF, while accounting/gradient probes and363-state reconstruction pass. This supports a finite-budget constrained-search failure with support saturation, not class infeasibility, absence of information or a new numerical error.',
        'The strict-support Jacobian is an exact local derivative away from support boundaries. Smooth derivative validation does not establish optimizer convergence or every nonsmooth composition derivative.',
    ]
    report['prohibited_inference'] = [
        'No deployment or market-fit qualification follows from this failed toy.',
        'The original26ed run did not retain all trajectories; only summary agreement is asserted against retention3c690.',
        'No conclusion about unseen market performance follows from a synthetic witness or optimizer repair proposal.',
    ]
    report['evidence'] = [
        {'path': str(path), 'sha256': sha(path)} for path in [
            PAYLOAD, BASE/'sparse-holdings-known-information-probe.json',
            BASE/'sparse-holdings-known-information-trajectories.csv',
            BASE/'sparse-holdings-failed-source/initial_26ed/sparse_holdings_net_excess.py',
            BASE/'sparse-holdings-failed-source/retention_3c690/sparse_holdings_net_excess.py',
            Path(__file__),
        ]]
    output = BASE/'sparse-holdings-failed-optimization-diagnosis.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'sha256': sha(output),
                      'seeds': [{k:s[k] for k in ['seed','positive_F_states','feasible_states',
                          'positive_F_feasible_states','first_all_SOXX_singleton_epoch',
                          'first_zero_gradient_update']} for s in report['seeds']]}, indent=2))


if __name__ == '__main__':
    main()
