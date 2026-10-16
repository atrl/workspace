"""Descriptive attribution of saved native Qlib holdings; no policy replay."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

NAME = 'etf_transfer_attribution_20261016'
PLAN_SHA = '6c0c30a1ecc34fc4421bd34e1f96ba431ddafb27c5115146bae48925f63b097f'
ASSETS = ['SOXL', 'SOXX']


def holdings_attribution(book, prices, schedule, raw_targets, *, initial_cash):
    """Reconcile four actual holding segments to each saved native daily NAV."""
    path = book['authority'].reset_index(drop=True)
    points = book['points'].set_index('stamp')
    orders = book['orders']
    targets = book['targets'].set_index('trade_date')
    daily, segments = [], []
    previous_nav = initial_cash
    for row in path.itertuples():
        day, exit_day = pd.Timestamp(row.trade_date), pd.Timestamp(row.exit_date)
        close = prices.index[prices.index.normalize() == day][-1]
        stamps = [day + pd.Timedelta('9h47min'), day + pd.Timedelta('10h32min'), close,
                  exit_day + pd.Timedelta('9h45min'), exit_day + pd.Timedelta('9h47min')]
        fee = float(orders.loc[orders.trade_date.eq(day), 'total'].sum()) if len(orders) else 0.
        contribution = np.zeros(2)
        for slot, (start, end) in enumerate(zip(stamps[:-1], stamps[1:])):
            p0, p1 = prices.loc[start, ASSETS].to_numpy(float), prices.loc[end, ASSETS].to_numpy(float)
            q = points.loc[start, ['q_soxl', 'q_soxx']].to_numpy(float)
            ratio, income = np.ones(2), np.zeros(2)
            if slot == 2:
                for action in schedule.get(exit_day, []):
                    i = ASSETS.index(action['symbol'])
                    if action['kind'] == 'split': ratio[i] *= action['ratio']
                    elif action['kind'] == 'entitle': income[i] += float(action['rate'])
            asset_return = (ratio * p1 + income) / p0 - 1
            exposure = q * p0 / previous_nav
            value = exposure * asset_return
            contribution += value
            for i, asset in enumerate(ASSETS):
                segments.append({'trade_date': day, 'asset': asset, 'slot': slot,
                                 'weight': exposure[i], 'asset_return': asset_return[i],
                                 'contribution': value[i]})
        reconstructed = contribution.sum() - fee / previous_nav
        if not np.isfinite(reconstructed): raise ValueError('nonfinite reconstruction')
        np.testing.assert_allclose(reconstructed, row.strategy_return, rtol=1e-9, atol=1e-9)
        raw = raw_targets.loc[pd.Timestamp(row.signal_date)]
        target = targets.loc[day]
        daily.append({'trade_date': day, 'exit_date': exit_day, 'return': row.strategy_return,
                      'SOXL_contribution': contribution[0], 'SOXX_contribution': contribution[1],
                      'fee_contribution': -fee / previous_nav, 'closure_error': reconstructed-row.strategy_return,
                      'risk_state': row.risk_state, 'opening_soxl_raw': raw.soxl_weight,
                      'opening_soxl_controlled': target.soxl_weight,
                      'opening_soxx_raw': raw.soxx_weight, 'opening_soxx_controlled': target.soxx_weight,
                      'opening_cash_raw': raw.cash_weight, 'opening_cash_controlled': target.cash_weight})
        previous_nav = row.native_expected_nav
    result = pd.DataFrame(daily)
    np.testing.assert_allclose((1 + result['return']).cumprod(), path.equity, rtol=1e-10)
    return result, pd.DataFrame(segments)


def describe(daily, segments, *, metadata, full_calendar):
    selections = [('all', 'all', daily.index, False)]
    for freq, kind in [('Y', 'year'), ('Q', 'quarter')]:
        periods = daily.trade_date.dt.to_period(freq)
        for period in periods.unique():
            observed = pd.DatetimeIndex(daily.loc[periods.eq(period), 'trade_date'])
            expected = full_calendar[full_calendar.to_period(freq) == period]
            complete = (period > full_calendar[0].to_period(freq) and period < full_calendar[-1].to_period(freq) and observed.equals(expected))
            selections.append((kind, str(period), daily.index[periods.eq(period)], not complete))
    selections += [('rolling126', str(daily.trade_date.iloc[i].date()), daily.index[i:i+126], False)
                   for i in range(len(daily)-125)]
    summaries, decomposed = [], []
    for kind, period, index, partial in selections:
        g = daily.loc[index]
        meta = {**metadata, 'kind': kind, 'period': period, 'partial_boundary_period': partial,
                'sessions': len(g), 'first_trade': str(g.trade_date.iloc[0].date()),
                'last_exit': str(g.exit_date.iloc[-1].date())}
        summaries.append({**meta, 'compounded_return': float(np.expm1(np.log1p(g['return']).sum())),
                          **{col: float(g[col].sum()) for col in ['SOXL_contribution', 'SOXX_contribution', 'fee_contribution']},
                          'max_closure_error': float(g.closure_error.abs().max()),
                          **{f'{asset}_raw_minus_controlled_mean': float((g[f'opening_{asset}_raw']-g[f'opening_{asset}_controlled']).mean()) for asset in ['soxl','soxx','cash']}})
        selected = segments[segments.trade_date.isin(g.trade_date)]
        for (asset, slot), x in selected.groupby(['asset', 'slot']):
            mean = len(x) * x.weight.mean() * x.asset_return.mean()
            covariance = float(((x.weight-x.weight.mean())*(x.asset_return-x.asset_return.mean())).sum())
            np.testing.assert_allclose(mean+covariance, x.contribution.sum(), atol=1e-12)
            decomposed.append({**meta, 'asset': asset, 'slot': slot, 'mean_weight': float(x.weight.mean()),
                               'mean_return': float(x.asset_return.mean()), 'mean_exposure_contribution': float(mean),
                               'covariance_contribution': covariance, 'total_contribution': float(x.contribution.sum())})
    return pd.DataFrame(summaries), pd.DataFrame(decomposed)


def run(root):
    import qlib
    import yaml
    from qlib.workflow import R
    from qlib.data import D
    from src.workflow.minute_action_value import _source
    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import archive_nas_recorder
    root = Path(root).resolve(); card = root/'research'/NAME
    if hashlib.sha256((card/'plan.json').read_bytes()).hexdigest() != PLAN_SHA: raise ValueError('frozen plan changed')
    review = json.loads((card/'design_review.json').read_text())
    if review.get('decision') not in ['approved', 'approved_for_execution']: raise ValueError('design not approved')
    if (card/'summary.json').exists(): return json.loads((card/'summary.json').read_text())
    plan = json.loads((card/'plan.json').read_text())
    qlib.init(provider_uri=str(root/'data/qlib/soxl'), region='us', kernels=1)
    full_calendar = pd.DatetimeIndex(D.calendar(freq='day'))
    provider = root/'data/qlib/frozen_policy_raw_cashbook_20261015'
    qlib.init(provider_uri=str(provider), region='us', kernels=1)
    calendar = D.calendar(freq='1min')
    source_objects, audits = {}, {}
    summaries, decomposed, daily_all, segments_all, references, states = [], [], {}, {}, {}, []
    for parent, source in plan['sources'].items():
        if 'tail_scoring' in parent:
            objects, audit = _source(root, source, ['analysis/summary', 'analysis/score_summary'])
            source_objects[parent], audits[parent] = objects, audit
            continue
        names = ['inputs/frame', 'inputs/raw_points', 'inputs/schedule', 'inputs/execution_plan', 'analysis/targets',
                 'analysis/metrics','analysis/paired_comparison','analysis/rolling_summary','analysis/annual_metrics']
        names += [f'native/raw_{arm}_{cost}' for arm in ['control', 'candidate'] for cost in plan['slippage_bps']]
        objects, audit = _source(root, source, names)
        source_objects[parent], audits[parent] = objects, audit
        prices = objects['inputs/raw_points']; frame = objects['inputs/frame']
        if not prices.index.isin(calendar).all(): raise ValueError('native marks outside Qlib calendar')
        if len(frame) != 944: raise ValueError('frozen calendar differs')
        for key in ['metrics','paired_comparison','rolling_summary','annual_metrics']: references[parent+'/'+key] = objects['analysis/'+key]
        for cost in plan['slippage_bps']:
            for arm in ['control', 'candidate']:
                book = objects[f'native/raw_{arm}_{cost}']
                pd.testing.assert_frame_equal(book['authority'][['signal_date','trade_date','exit_date']].reset_index(drop=True), frame[['signal_date','trade_date','exit_date']].reset_index(drop=True))
                daily, segments = holdings_attribution(book, prices, objects['inputs/schedule'], objects['analysis/targets'][arm], initial_cash=objects['inputs/execution_plan']['initial_cash'])
                daily['prior_trend_state'] = np.where(frame.soxx_trend20.to_numpy()>0, 'positive', 'nonpositive')
                key = f'{parent}_{arm}_{cost}'; daily_all[key], segments_all[key] = daily, segments
                meta = {'parent':parent,'arm':arm,'slippage_bps':cost}
                a,b = describe(daily, segments, metadata=meta, full_calendar=full_calendar)
                for state in ['risk_state', 'prior_trend_state']:
                    for value,g in daily.groupby(state):
                        states.append({**meta,'state':state,'value':str(value),'sessions':len(g),'noncontiguous_descriptive_not_causal':True,**{col:float(g[col].sum()) for col in ['SOXL_contribution','SOXX_contribution','fee_contribution']}})
                summaries.append(a); decomposed.append(b)
    totals, components = pd.concat(summaries, ignore_index=True), pd.concat(decomposed, ignore_index=True)
    keys = ['parent','slippage_bps','kind','period','asset','slot']
    paired = components[components.arm.eq('candidate')].merge(components[components.arm.eq('control')],on=keys,suffixes=('_candidate','_control'),validate='one_to_one')
    paired = paired[keys].assign(**{col+'_delta':paired[col+'_candidate']-paired[col+'_control'] for col in ['mean_exposure_contribution','covariance_contribution','total_contribution','mean_weight']})
    small = {'state_summary':pd.DataFrame(states), 'period_summary':totals[totals.kind.ne('rolling126')], 'component_summary':components[components.kind.ne('rolling126')], 'paired_components':paired[paired.kind.ne('rolling126')]}
    tracking = card/'qlib_recorder'
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','analysis/daily','analysis/segments','analysis/rolling*','sources/*']}
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path='inputs',plan=plan,source_audits=audits,source_objects=source_objects)
        rec.save_objects(artifact_path='sources',analysis=Path(__file__).read_text())
        rec.save_objects(artifact_path='analysis',daily=daily_all,segments=segments_all,rolling_summary=totals[totals.kind.eq('rolling126')],rolling_components=components[components.kind.eq('rolling126')],rolling_paired=paired[paired.kind.eq('rolling126')],parent_reference=references,**small)
        summary={'experiment':NAME,'plan_sha256':PLAN_SHA,'decision':'descriptive_diagnosis_only_parent_routes_remain_not_promoted','objective_achieved':False,'new_fits':0,'new_inference':0,'new_policy_accounts':0,'daily_books':len(daily_all),'max_daily_return_closure_error':float(totals.max_closure_error.max()),'calendar_rows_per_book':944,'rolling_windows_per_book':819,'provider_uri':str(provider),'region':'us','as_of':'2026-10-09','qlib_docs':'https://qlib.readthedocs.io/en/latest/component/workflow.html','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'limitations':['Historical development diagnosis, not independent evidence','Mean exposure and covariance are additive arithmetic return contributions, not tradable counterfactual or compounded allocation alpha','Controller target differences include legal projection, smoothing and endogenous risk path; no controller-off account','Inherited raw cash corporate convention and sampled risk coverage remain unqualified for promotion']}
        rec.save_objects(artifact_path='analysis',summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    for name,table in small.items():table.to_csv(card/(name+'.csv'),index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    first=next(iter(plan['sources'].values()));binding=json.loads((root/first['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'plan':'plan.json','module':'src.analysis.etf_transfer_attribution','provider_uri':str(provider),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    atomic_json(summary,card/'summary.json')
    return summary
