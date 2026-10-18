"""One bounded attribution from sealed preopen-TOM Recorder outputs only."""
from itertools import product
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json

C='C_preopen_tom'
B='B_preopen_tom_context'
A='A_active_research_replay'
FAMILIES=(C,B,A,'SOXX')
MASKS=('first3_branch_known','expected_last_branch_known','proxy_known')


def _json_scalars(value):
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.floating):return float(value)
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,dict):return {_json_scalars(k):_json_scalars(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_scalars(v) for v in value]
    return value


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _beta(frame,prefix=''):
    return 3*frame[prefix+'soxl_weight']+frame[prefix+'soxx_weight']


def run(*,root,card):
    import qlib
    root,card=Path(root),Path(card)
    scope=json.loads((card/'failure_attribution_scope.json').read_text())
    if (card/'failure_attribution.json').exists():raise ValueError('fixed attribution already complete; no repeat')
    summary=json.loads((card/'summary.json').read_text());manifest=json.loads((card/'verification_manifest.json').read_text())
    before={name:sha(card/name) for name in ('verification_manifest.json','nas_binding.json')}
    if summary['recorder_id']!=scope['recorder_id'] or summary['nas_evidence']['sha256']!=scope['archive_sha256']:
        raise ValueError('sealed Recorder/archive differs')
    names=['analysis/ensembles','analysis/smoothed_targets','analysis/classification','analysis/training',
        'analysis/annual_metrics','analysis/quarterly_metrics','analysis/metrics','inputs/canonical','evaluation/prepared']
    names += [f'native/{family}_{cost}' for family in (C,B) for cost in (0,2,5)]
    digests={name:next(item['sha256'] for item in manifest['entries'] if item['path'].endswith('/artifacts/'+name)) for name in names}
    qlib.init(provider_uri=summary['provider_uri'],region='us',kernels=1)
    objects,audit=load_verified_recorder_objects(card,expected_recorder_id=scope['recorder_id'],artifacts=names,
        expected_manifest_sha256=scope['manifest_sha256'],expected_artifact_sha256=digests)
    e=objects['evaluation/prepared'];dates=pd.DatetimeIndex(e['frame'].signal_date)
    canonical=objects['inputs/canonical'].reindex(dates)
    if len(dates)!=944 or not canonical.account_eligible.eq(True).all():raise ValueError('account/canonical alignment differs')
    if not canonical[list(MASKS)].isin([False,True]).all().all():raise ValueError('saved canonical masks differ')
    if not canonical.loc[canonical.proxy_known,'preopen_tom_proxy_nullable'].isin([0.,1.]).all():raise ValueError('saved binary values differ')
    if not canonical.loc[~canonical.proxy_known,'preopen_tom_proxy_nullable'].isna().all():raise ValueError('unknown canonical values lost')
    books={(family,cost):(objects[f'native/{family}_{cost}'] if family in (C,B) else e['references'][(family,cost)])
        for family in FAMILIES for cost in (0,2,5)}
    authority={}
    for key,book in books.items():
        value=book['authority'].reset_index(drop=True).copy()
        if not pd.DatetimeIndex(value.signal_date).equals(dates):raise ValueError('native dates differ')
        value['ending_beta']=_beta(value,'ending_')
        authority[key]=value
    raw={family:objects['analysis/ensembles'][family].reindex(dates) for family in (C,B)}
    smooth={family:objects['analysis/smoothed_targets'][family].reindex(dates) for family in (C,B)}
    for values in [*raw.values(),*smooth.values()]:
        if not np.isfinite(values.to_numpy(float)).all():raise ValueError('saved target has missing account row')
    metrics=objects['analysis/metrics'].set_index(['family','slippage_bps'])
    path_rows=[]
    for key,value in authority.items():
        family,cost=key;row={'family':family,'slippage_bps':cost,'sessions':len(value),
            'net_return_pct':float(metrics.loc[key,'return_pct']),
            'close_max_drawdown_pct':float(metrics.loc[key,'close_max_drawdown_pct']),
            'sampled_point_max_drawdown_pct':float(metrics.loc[key,'sampled_point_max_drawdown_pct']),
            'actual_mean_ending_beta':float(value.ending_beta.mean()),
            'actual_mean_ending_soxl_weight':float(value.ending_soxl_weight.mean()),
            'actual_mean_ending_soxx_weight':float(value.ending_soxx_weight.mean()),
            'actual_mean_ending_cash_like_weight':float(value.ending_cash_weight.mean()),
            'actual_mean_spendable_cash_usd':float(value.ending_spendable_cash.mean())}
        for field in ('risk_state','execute_open_target','recovery_action'):
            row[field+'_nonzero_days']=int(value[field].ne(0).sum()) if field in value else None
        if family in (C,B):
            row.update(raw_proposal_mean_beta=float(_beta(raw[family]).mean()),
                raw_proposal_mean_cash=float(raw[family].cash_weight.mean()),
                smoothed_proposal_mean_beta=float(_beta(smooth[family]).mean()),
                smoothed_proposal_mean_cash=float(smooth[family].cash_weight.mean()),
                proposal_basis='saved own equal3seed ensemble and saved smoothing, no new forward')
        else:row['proposal_basis']='no neural raw proposal invented for reference'
        path_rows.append(row)
    output_difference=[]
    for kind,values in [('raw',raw),('smoothed',smooth)]:
        for field in ['soxl_weight','soxx_weight','cash_weight','tail_probability','alpha_probability']:
            diff=values[C][field]-values[B][field]
            output_difference.append({'kind':kind,'field':field,'rows':len(diff),
                'mean_C_minus_B':float(diff.mean()),'mean_absolute_difference':float(diff.abs().mean()),
                'RMSE':float(np.sqrt(np.square(diff).mean())),'max_absolute_difference':float(diff.abs().max())})
    differences=[]
    pairs=[((C,cost),(family,cost)) for cost in (0,2,5) for family in (B,A)]
    pairs += [((C,a),(C,b)) for a,b in ((0,2),(0,5),(2,5))]
    for left,right in pairs:
        l,r=authority[left],authority[right]
        for field in ('risk_state','risk_clock','execute_open_target','recovery_action','ending_beta','ending_cash_weight'):
            if field not in l or field not in r:
                differences.append({'left_family':left[0],'left_cost':left[1],'right_family':right[0],'right_cost':right[1],
                                    'field':field,'available':False});continue
            changed=~(l[field].eq(r[field]) | (l[field].isna()&r[field].isna()))
            first=l.loc[changed,'trade_date'];delta=l[field].astype(float)-r[field].astype(float)
            differences.append({'left_family':left[0],'left_cost':left[1],'right_family':right[0],'right_cost':right[1],
                'field':field,'available':True,'different_sessions':int(changed.sum()),
                'first_difference':None if first.empty else str(first.iloc[0].date()),
                'mean_difference':float(delta.mean()),'max_absolute_difference':float(delta.abs().max()),
                'role':'endogenous full-path descriptive difference, not causal attribution'})
    periods=[]
    for kind in ('annual','quarterly'):
        data=objects[f'analysis/{kind}_metrics']
        for cost in (0,2,5):
            candidate=data.loc[data.family.eq(C)&data.slippage_bps.eq(cost)].set_index('period')
            for family in (B,A,'SOXX'):
                compare=data.loc[data.family.eq(family)&data.slippage_bps.eq(cost)].set_index('period')
                pd.testing.assert_index_equal(candidate.index,compare.index)
                if not candidate.complete_period.equals(compare.complete_period):raise ValueError('period completeness differs')
                for period,row in candidate.iterrows():
                    other=compare.loc[period];ratio=float(row.terminal_wealth/other.terminal_wealth)
                    periods.append({'kind':kind,'period':str(period),'complete_period':bool(row.complete_period),
                        'slippage_bps':cost,'comparator':family,'C_net_return_pct':float(row.return_pct),
                        'comparator_net_return_pct':float(other.return_pct),'relative_wealth':ratio,'C_wins':ratio>1+1e-8})
    period_frame=pd.DataFrame(periods);wins=[]
    for key,group in period_frame.groupby(['kind','slippage_bps','comparator']):
        for status in ('all','complete','partial'):
            part=group if status=='all' else group.loc[group.complete_period.eq(status=='complete')]
            wins.append(dict(zip(['kind','slippage_bps','comparator'],key))|
                {'coverage':status,'periods':len(part),'wins':int(part.C_wins.sum()),'loss_or_tie':int((~part.C_wins).sum())})
    category=pd.Series('unknown',index=dates)
    category.loc[canonical.preopen_tom_proxy_nullable.eq(0)]='0'
    category.loc[canonical.preopen_tom_proxy_nullable.eq(1)]='1'
    grouped=[]
    # Keep the complete predetermined 8mask x3value x4family x3cost table, including empty cells.
    for pattern in product((False,True),repeat=3):
        pattern_mask=np.ones(len(canonical),dtype=bool)
        for name,value in zip(MASKS,pattern):pattern_mask &= canonical[name].to_numpy(dtype=bool)==value
        for calendar_value in ('0','1','unknown'):
            take=pattern_mask & category.eq(calendar_value).to_numpy();count=int(take.sum())
            for family in FAMILIES:
                for cost in (0,2,5):
                    values=authority[(family,cost)].loc[take]
                    logs=np.log1p(values.strategy_return.to_numpy(float))
                    row={**dict(zip(MASKS,pattern)),'calendar_value':calendar_value,'family':family,'slippage_bps':cost,
                        'observations':count,'zero_support':count==0,
                        'sum_recorded_log_net_return':float(logs.sum()) if count else np.nan,
                        'mean_recorded_log_net_return':float(logs.mean()) if count else np.nan,
                        'compounded_recorded_interval_return_pct':float(100*np.expm1(logs.sum())) if count else np.nan,
                        'actual_mean_ending_beta':float(values.ending_beta.mean()) if count else np.nan,
                        'actual_mean_ending_cash_like_weight':float(values.ending_cash_weight.mean()) if count else np.nan,
                        'role':'potentially noncontiguous existing net intervals, not new tradable policy/backtest or single-day causality'}
                    if family in (C,B):
                        row.update(raw_proposal_mean_beta=float(_beta(raw[family]).iloc[take].mean()) if count else np.nan,
                            raw_proposal_mean_cash=float(raw[family].cash_weight.iloc[take].mean()) if count else np.nan,
                            smoothed_proposal_mean_beta=float(_beta(smooth[family]).iloc[take].mean()) if count else np.nan)
                    grouped.append(row)
    tables={'periods':period_frame,'path_summary':pd.DataFrame(path_rows),'state_differences':pd.DataFrame(differences),
            'calendar_groups':pd.DataFrame(grouped)}
    for name,table in tables.items():
        if len(table)>500:raise ValueError('slim attribution limit exceeded')
        table.to_csv(card/('attribution_'+name+'.csv'),index=False)
    if len(tables['calendar_groups'])!=288 or sum(row['observations'] for row in grouped if row['family']==C and row['slippage_bps']==2)!=944:
        raise ValueError('predetermined calendar grouping omitted or duplicated dates')
    result={'scope_sha256':sha(card/'failure_attribution_scope.json'),'recorder_id':scope['recorder_id'],
        'archive_sha256':scope['archive_sha256'],'source_artifacts':audit['artifacts'],
        'small_tables':{name:'attribution_'+name+'.csv' for name in tables},'all_period_win_counts':wins,
        'B_C_output_differences':output_difference,'all_original_classification':objects['analysis/classification'].to_dict(orient='records'),
        'all_six_training_summaries':objects['analysis/training'].to_dict(orient='records'),
        'calendar_groups_prescribed':288,'account_intervals_each':944,'new_fit_forward_replay_CI_sourceHTTP_NASreads':0,
        'limits':['Group net returns are conditioned slices of actual endogenous full paths, not counterfactual standalone accounts.',
                  'H5 shared-head classification is not a direct standalone calendar-H1 information test; numeric input enters the allocation head directly.',
                  'Proposal/state/holding differences show transmission but do not identify a sole controller or fee cause.',
                  'The fixed version is closed; no mask,venue,window,offset,seed or threshold selection follows these diagnostics.'],
        'sealed_manifest_NAS_binding_unchanged':before=={name:sha(card/name) for name in before}}
    if not result['sealed_manifest_NAS_binding_unchanged']:raise ValueError('sealed evidence changed')
    atomic_json(_json_scalars(result),card/'failure_attribution.json')
    return {'path_summary':tables['path_summary'],'period_win_counts':pd.DataFrame(wins),
            'calendar_supported_groups':tables['calendar_groups'].loc[tables['calendar_groups'].observations.gt(0)],
            'sealed_evidence_unchanged':True}


def finalize_output(*,root,card):
    """Finish the interrupted JSON output, without repeating native attribution."""
    import qlib
    root,card=Path(root),Path(card)
    attempt=json.loads((card/'failure_attribution_attempt_1.json').read_text())
    repair=json.loads((card/'output_finalize_scope.json').read_text())
    scope=json.loads((card/'failure_attribution_scope.json').read_text())
    if (card/'failure_attribution.json').exists():raise ValueError('JSON output already finalized')
    if sha(card/'failure_attribution_scope.json')!=repair['original_scope_sha256']:
        raise ValueError('original scope changed')
    if sha(card/repair['original_code_snapshot'])!=repair['original_code_sha256']:
        raise ValueError('original code snapshot changed')
    for file,item in attempt['completed_tables'].items():
        if sha(card/file)!=item['sha256']:raise ValueError('original completed table changed: '+file)
    tables={name:pd.read_csv(card/('attribution_'+name+'.csv')) for name in ('periods','path_summary','state_differences','calendar_groups')}
    # These are counts of the already-saved period verdicts, not new account or group returns.
    wins=[]
    for kind in ('annual','quarterly'):
        for cost in (0,2,5):
            for family in (B,A,'SOXX'):
                source=tables['periods'];source=source.loc[source.kind.eq(kind)&source.slippage_bps.eq(cost)&source.comparator.eq(family)]
                for status in ('all','complete','partial'):
                    part=source if status=='all' else source.loc[source.complete_period.eq(status=='complete')]
                    wins.append({'kind':kind,'slippage_bps':int(cost),'comparator':family,'coverage':status,
                        'periods':int(len(part)),'wins':int(part.C_wins.sum()),'loss_or_tie':int((~part.C_wins).sum())})
    summary=json.loads((card/'summary.json').read_text());manifest=json.loads((card/'verification_manifest.json').read_text())
    before={name:sha(card/name) for name in ('verification_manifest.json','nas_binding.json')}
    names=repair['same_recorder_small_artifacts']
    digests={name:next(item['sha256'] for item in manifest['entries'] if item['path'].endswith('/artifacts/'+name)) for name in names}
    qlib.init(provider_uri=summary['provider_uri'],region='us',kernels=1)
    objects,audit=load_verified_recorder_objects(card,expected_recorder_id=scope['recorder_id'],artifacts=names,
        expected_manifest_sha256=scope['manifest_sha256'],expected_artifact_sha256=digests)
    # Exactly the original prescribed944 signal interval selection.
    raw={family:objects['analysis/ensembles'][family].loc['2023-01-03':'2026-10-07'] for family in (C,B)}
    smooth={family:objects['analysis/smoothed_targets'][family] for family in (C,B)}
    output_difference=[]
    for kind,values in [('raw',raw),('smoothed',smooth)]:
        if len(values[C])!=944 or not values[C].index.equals(values[B].index):raise ValueError('fixed target rows differ')
        for field in ['soxl_weight','soxx_weight','cash_weight','tail_probability','alpha_probability']:
            diff=values[C][field]-values[B][field]
            output_difference.append({'kind':kind,'field':field,'rows':len(diff),'mean_C_minus_B':float(diff.mean()),
                'mean_absolute_difference':float(diff.abs().mean()),'RMSE':float(np.sqrt(np.square(diff).mean())),
                'max_absolute_difference':float(diff.abs().max())})
    result={'scope_sha256':repair['original_scope_sha256'],'recorder_id':scope['recorder_id'],
        'archive_sha256':scope['archive_sha256'],'initial_attempt_receipt':'failure_attribution_attempt_1.json',
        'output_finalize_scope':'output_finalize_scope.json','original_tables':attempt['completed_tables'],
        'finalize_small_source_artifacts':audit['artifacts'],
        'small_tables':{name:'attribution_'+name+'.csv' for name in tables},'all_period_win_counts':wins,
        'B_C_output_differences':output_difference,'all_original_classification':objects['analysis/classification'].to_dict(orient='records'),
        'all_six_training_summaries':objects['analysis/training'].to_dict(orient='records'),
        'calendar_groups_prescribed':288,'account_intervals_each':944,'new_fit_forward_replay_CI_sourceHTTP_NASreads':0,
        'native_or_group_return_calculations_repeated':False,
        'limits':['Group returns are noncontiguous slices of recorded full paths, not separate tradable accounts or individual-day causality.',
                  'H5 classification is not a direct calendar-H1 information test; no new model-forward or feature permutation was run.',
                  'Proposal and control-state transmission is descriptive, not a sole controller/fees cause.',
                  'Fixed version remains closed; no threshold,window,offset,mask or seed rescue.'],
        'sealed_manifest_NAS_binding_unchanged':before=={name:sha(card/name) for name in before}}
    if not result['sealed_manifest_NAS_binding_unchanged']:raise ValueError('sealed evidence changed')
    atomic_json(_json_scalars(result),card/'failure_attribution.json')
    for file,item in attempt['completed_tables'].items():
        if sha(card/file)!=item['sha256']:raise ValueError('completed table changed during outputfinalize')
    return {'phase':'output_finalize','original_four_tables_unchanged':True,'path_summary':tables['path_summary'],
            'period_win_counts':pd.DataFrame(wins),'sealed_evidence_unchanged':True}
