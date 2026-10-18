"""One scoped, read-only attribution from the sealed S/Z experiment Recorder."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import os

import numpy as np
import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json

B = 'B_day_night_total'
C = 'C_day_night_contrast'
COSTS = (0, 2, 5)
SEEDS = (71, 172, 273)
FIELDS = ('risk_state', 'risk_clock', 'execute_open_target', 'recovery_action', 'ending_beta', 'ending_cash_weight')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _beta(frame, prefix=''):
    return 3 * frame[prefix + 'soxl_weight'] + frame[prefix + 'soxx_weight']


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _periods(objects):
    rows = []
    for kind in ('annual', 'quarterly', 'monthly'):
        data = objects['analysis/' + kind + '_metrics']
        for cost in COSTS:
            candidate = data.loc[data.family.eq(C) & data.slippage_bps.eq(cost)].set_index('period')
            for comparator in (B, 'SOXX'):
                control = data.loc[data.family.eq(comparator) & data.slippage_bps.eq(cost)].set_index('period')
                pd.testing.assert_index_equal(candidate.index, control.index)
                pd.testing.assert_series_equal(candidate.complete_period, control.complete_period)
                for period, value in candidate.iterrows():
                    other = control.loc[period]
                    ratio = float(value.terminal_wealth / other.terminal_wealth)
                    rows.append({'kind': kind, 'period': str(period), 'complete_period': bool(value.complete_period),
                        'slippage_bps': cost, 'comparator': comparator, 'C_return_pct': float(value.return_pct),
                        'comparator_return_pct': float(other.return_pct),
                        'return_percentage_point_difference': float(value.return_pct - other.return_pct),
                        'relative_wealth': ratio, 'C_wins': ratio > 1 + 1e-8})
    table = pd.DataFrame(rows); wins = []
    for (kind, cost, comparator), group in table.groupby(['kind', 'slippage_bps', 'comparator'], sort=False):
        for completeness in ('all', 'complete', 'partial'):
            sample = group if completeness == 'all' else group.loc[group.complete_period.eq(completeness == 'complete')]
            wins.append({'kind': kind, 'slippage_bps': int(cost), 'comparator': comparator, 'coverage': completeness,
                         'periods': len(sample), 'wins': int(sample.C_wins.sum()), 'loss_or_tie': int((~sample.C_wins).sum())})
    rolling = []
    for cost in COSTS:
        left = objects[f'analysis/rolling/{C}_{cost}']
        right = objects[f'analysis/rolling/{B}_{cost}']
        pd.testing.assert_frame_equal(left[['first_trade', 'last_exit']], right[['first_trade', 'last_exit']])
        _require(len(left) == 819, 'complete fixed rolling window count differs')
        for comparator, differences in ((B, left.relative_log - right.relative_log), ('SOXX', left.relative_log)):
            rolling.append({'slippage_bps': cost, 'comparator': comparator, 'windows': len(left),
                'first_window_trade': str(left.first_trade.iloc[0]), 'last_window_exit': str(left.last_exit.iloc[-1]),
                'positive_windows': int(differences.gt(0).sum()), 'positive_fraction': float(differences.gt(0).mean()),
                'minimum_relative_log': float(differences.min()), 'maximum_relative_log': float(differences.max()),
                'mean_relative_log': float(differences.mean()), 'overlapping_not_independent': True,
                'method': 'Subtract the two stored SOXX-relative log windows for C/B; no return replay or bootstrap.'})
    return table, wins, rolling


def _paths_and_proposals(objects):
    authority = {}
    for family in (B, C):
        for cost in COSTS:
            book = objects[f'native/{family}_{cost}']
            _require(book['audit']['status'] == 'passed', 'unvalidated native account')
            frame = book['authority'].reset_index(drop=True).copy()
            _require(len(frame) == 944, 'native account interval count differs')
            frame['ending_beta'] = _beta(frame, 'ending_')
            authority[(family, cost)] = frame
    base = authority[(B, 0)]
    dates = pd.DatetimeIndex(base.signal_date)
    for frame in authority.values():
        pd.testing.assert_frame_equal(frame[['signal_date', 'trade_date', 'exit_date']], base[['signal_date', 'trade_date', 'exit_date']])
    raw, smooth = {}, {}
    for family in (B, C):
        values = objects['analysis/ensembles'][family]
        _require(len(values) == 946 and values.index.is_unique, 'original raw prediction tails missing')
        raw[family] = values.reindex(dates)
        smooth[family] = objects['analysis/smoothed_targets'][family].reindex(dates)
    for values in (*raw.values(), *smooth.values()):
        _require(values.shape == (944, 5) and np.isfinite(values.to_numpy()).all(), 'saved proposal coverage differs')
    proposal_differences = []
    for kind, frames in (('raw', raw), ('smoothed', smooth)):
        values = {family: frame.assign(nominal_beta=_beta(frame)) for family, frame in frames.items()}
        for field in ('soxl_weight', 'soxx_weight', 'cash_weight', 'tail_probability', 'alpha_probability', 'nominal_beta'):
            difference = values[C][field] - values[B][field]
            proposal_differences.append({'kind': kind, 'field': field, 'rows': 944,
                'B_mean': float(values[B][field].mean()), 'C_mean': float(values[C][field].mean()),
                'mean_C_minus_B': float(difference.mean()), 'MAE': float(difference.abs().mean()),
                'RMSE': float(np.sqrt(np.square(difference).mean())), 'maximum_absolute_difference': float(difference.abs().max()),
                'domain': 'All original944 account signals; two non-account raw prediction tails remain in the source Recorder.'})
    metrics = objects['analysis/metrics'].set_index(['family', 'slippage_bps'])
    paths = []
    for (family, cost), frame in authority.items():
        row = {'family': family, 'slippage_bps': cost, 'sessions': 944,
            'net_return_pct': float(metrics.loc[(family, cost), 'return_pct']),
            'raw_mean_nominal_beta': float(_beta(raw[family]).mean()), 'raw_mean_cash_weight': float(raw[family].cash_weight.mean()),
            'smoothed_mean_nominal_beta': float(_beta(smooth[family]).mean()), 'smoothed_mean_cash_weight': float(smooth[family].cash_weight.mean()),
            'actual_mean_ending_nominal_beta': float(frame.ending_beta.mean()),
            'actual_mean_ending_soxl_weight': float(frame.ending_soxl_weight.mean()),
            'actual_mean_ending_soxx_weight': float(frame.ending_soxx_weight.mean()),
            'actual_mean_ending_cash_like_weight': float(frame.ending_cash_weight.mean()),
            'actual_mean_spendable_cash_usd': float(frame.ending_spendable_cash.mean()),
            'zero_SOXL_sessions': int(frame.ending_soxl_quantity.eq(0).sum()),
            'zero_SOXL_fraction': float(frame.ending_soxl_quantity.eq(0).mean()),
            'nonzero_risk_state_sessions': int(frame.risk_state.ne(0).sum()),
            'open_execute_sessions': int(frame.execute_open_target.astype(bool).sum()),
            'firsthour_recovery_sessions': int(frame.recovery_action.astype(bool).sum()),
            'risk_state_counts': json.dumps(_safe(frame.risk_state.value_counts().sort_index().to_dict()), sort_keys=True),
            'risk_clock_counts': json.dumps(_safe(frame.risk_clock.value_counts().sort_index().to_dict()), sort_keys=True)}
        paths.append(row)
    pairs = [((C, cost), (B, cost)) for cost in COSTS]
    pairs += [((family, left), (family, right)) for family in (B, C) for left, right in combinations(COSTS, 2)]
    differences = []
    for left, right in pairs:
        a, b = authority[left], authority[right]
        for field in FIELDS:
            _require(field in a and field in b, 'stored native state field absent: ' + field)
            changed = ~(a[field].eq(b[field]) | (a[field].isna() & b[field].isna()))
            delta = a[field].astype(float) - b[field].astype(float)
            first = a.loc[changed, 'trade_date']
            differences.append({'left_family': left[0], 'left_cost': left[1], 'right_family': right[0], 'right_cost': right[1],
                'field': field, 'different_sessions': int(changed.sum()),
                'first_difference_trade_date': None if first.empty else str(pd.Timestamp(first.iloc[0]).date()),
                'mean_difference': float(delta.mean()), 'MAE': float(delta.abs().mean()),
                'max_absolute_difference': float(delta.abs().max()), 'comparison': 'exact recorded value; no fitted epsilon',
                'role': 'Descriptive endogenous path difference; not a counterfactual or single-controller/fee cause.'})
    return pd.DataFrame(paths), pd.DataFrame(differences), proposal_differences


def _training(objects):
    rows, pairs, frames = [], [], {}
    fields = ('allocation_loss', 'primary_bce', 'auxiliary_bce', 'original_loss', 'supervision_half_mse', 'loss', 'global_preclip_norm')
    for family, target in ((B, 'S'), (C, 'Z')):
        for seed in SEEDS:
            trace = pd.DataFrame(objects[f'training_trace/{family}_{seed}'])
            _require(len(trace) == 300 and np.array_equal(trace.epoch, np.arange(1, 301))
                     and np.array_equal(trace.completed_step_count, np.arange(1, 301)), 'fixed actual300-step trace missing')
            _require(trace.target_name.eq(target).all() and np.isfinite(trace.loc[:, list(fields)].to_numpy()).all(), 'training trace fields differ')
            frames[(family, seed)] = trace
            row = {'family': family, 'seed': seed, 'supervision_target': target, 'completed_steps': 300,
                   'clip_active_steps': int(trace.global_preclip_norm.gt(2).sum()),
                   'interpretation': 'Recorded pre-update training losses. Different S/Z MSE values do not compare alpha or out-of-fit information.'}
            for field in fields:
                row.update({field + '_first': float(trace[field].iloc[0]), field + '_last': float(trace[field].iloc[-1]),
                            field + '_mean': float(trace[field].mean()), field + '_minimum': float(trace[field].min()),
                            field + '_maximum': float(trace[field].max())})
            row['weighted_supervision_first'] = .02 * float(trace.supervision_half_mse.iloc[0])
            row['weighted_supervision_last'] = .02 * float(trace.supervision_half_mse.iloc[-1])
            rows.append(row)
    for seed in SEEDS:
        left, right = frames[(C, seed)], frames[(B, seed)]
        same = left.sampled_indices_sha256.eq(right.sampled_indices_sha256)
        row = {'seed': seed, 'paired_epochs': 300, 'sampling_hash_equal_epochs': int(same.sum()),
            'sampling_hash_all_equal': bool(same.all()),
            'first_sampling_difference_epoch': None if same.all() else int(left.loc[~same, 'epoch'].iloc[0]),
            'sampling_hash_check_only': 'Recorded hashes compared; no RNG resampling or training run.'}
        for field in ('allocation_loss', 'primary_bce', 'auxiliary_bce', 'original_loss', 'loss'):
            diff = left[field] - right[field]
            row[field + '_last_C_minus_B'] = float(diff.iloc[-1])
            row[field + '_all_step_MAE'] = float(diff.abs().mean())
        pairs.append(row)
    return pd.DataFrame(rows), pairs


def run(*, root, card):
    import qlib
    root, card = Path(root).resolve(), Path(card).resolve()
    scope = json.loads((card / 'attribution_scope.json').read_text())
    _require(os.environ.get('DAY_NIGHT_ATTRIBUTION') == '1', 'bounded attribution disabled')
    _require(sha(Path(__file__)) == scope['code_sha256'], 'frozen attribution implementation differs')
    _require(sha(card / 'independent_evaluation.json') == scope['independent_evaluation_sha256'], 'closed-result review differs')
    attempt_path = card / 'attribution_attempt.json'
    _require(not attempt_path.exists() and not (card / 'failure_attribution.json').exists(), 'one attribution attempt already consumed')
    before = {name: sha(card / name) for name in ('verification_manifest.json', 'nas_binding.json')}
    _require(before['verification_manifest.json'] == scope['manifest_sha256'], 'sealed manifest differs')
    summary = json.loads((card / 'summary.json').read_text())
    _require(summary['recorder_id'] == scope['recorder_id']
             and summary['nas_evidence']['sha256'] == scope['archive_sha256'], 'sealed Recorder/archive differs')
    state = {'status': 'started', 'scope_sha256': sha(card / 'attribution_scope.json'), 'recorder_id': scope['recorder_id'],
             'new_fit_forward_replay_native_CI_sourceHTTP_NASdownload': 0, 'completed_tables': {}}
    atomic_json(state, attempt_path)
    try:
        qlib.init(provider_uri=summary['provider_uri'], region='us', kernels=1)
        objects, audit = load_verified_recorder_objects(card, expected_recorder_id=scope['recorder_id'],
            artifacts=scope['artifacts'], expected_manifest_sha256=scope['manifest_sha256'],
            expected_artifact_sha256=scope['artifact_sha256'])
        periods, wins, rolling = _periods(objects)
        paths, states, proposals = _paths_and_proposals(objects)
        training, sampling = _training(objects)
        tables = {'periods': periods, 'path_summary': paths, 'state_differences': states, 'training': training}
        for name, table in tables.items():
            _require(len(table) <= 500, 'bounded slim attribution table exceeded')
            path = card / ('attribution_' + name + '.csv'); table.to_csv(path, index=False)
            state['completed_tables'][name] = {'path': path.name, 'sha256': sha(path), 'rows': len(table)}
            atomic_json(state, attempt_path)
        after = {name: sha(card / name) for name in before}
        _require(before == after, 'sealed manifest/archive binding changed')
        for item in audit['artifacts'].values():
            _require(sha(item['local_path']) == item['sha256'], 'source artifact changed during attribution')
        result = {'scope_sha256': state['scope_sha256'], 'recorder_id': scope['recorder_id'],
            'archive_sha256': scope['archive_sha256'], 'code_sha256': scope['code_sha256'],
            'source_artifacts': audit['artifacts'], 'small_tables': state['completed_tables'],
            'period_win_counts': wins, 'rolling_all_windows': rolling, 'B_C_proposal_differences': proposals,
            'paired_training_differences_and_sampling': sampling,
            'classification_summary': objects['analysis/classification'].to_dict(orient='records'),
            'recorded_economic_gates': objects['analysis/summary']['gates'],
            'new_fit_forward_replay_native_CI_sourceHTTP_NASdownload': 0,
            'sealed_manifest_archive_binding_and_loaded_artifacts_unchanged': True,
            'limits': ['S and Z are different normalized training targets; their MSE is not comparable alpha or evaluation predictability.',
                'Original H5 tail/alpha classification is not a direct test of evaluation S/Z, which was never constructed.',
                'Nominal beta is3*SOXLweight+SOXXweight, not an estimated market beta; ending cash-like includes receivables.',
                'Risk/clock/action/fee differences describe coupled endogenous full paths; no counterfactual/controller ablation was run.',
                'All stored periods and all819 rolling windows retained; overlap is not independent evidence.',
                'Fixed C remains closed. Matched B cannot replace it; no head/lambda/seed/window/sign or threshold rescue.']}
        atomic_json(_safe(result), card / 'failure_attribution.json')
        state.update(status='completed', result_sha256=sha(card / 'failure_attribution.json'),
                     source_artifacts_read=len(scope['artifacts']), sealed_evidence_unchanged=True)
        atomic_json(state, attempt_path)
        return {'path_summary': paths, 'period_win_counts': pd.DataFrame(wins), 'rolling_summary': pd.DataFrame(rolling),
                'training_sampling': pd.DataFrame(sampling), 'sealed_evidence_unchanged': True}
    except BaseException as error:
        state.update(status='failed', exception_type=type(error).__name__, exception=str(error)); atomic_json(state, attempt_path)
        raise
