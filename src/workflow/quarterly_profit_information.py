"""Paired reported-quarter profit diffusion test; no strategy accounts."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.seed_disagreement_information import full_calendar_bootstrap
from src.core.recorder_sources import load_verified_recorder_objects
from src.data.annual_valuation_information import MARKET as PRICE_MARKET, TARGETS, attach_labels, count_selection, read_market_inputs
from src.data.nas_market_data import archive_nas_recorder
from src.model.monthly_valuation_information import MonthlyValuationRidge

NAME = 'etf_quarterly_profit_information_20261016'
NUISANCE = ['known_profit_age_mean_days','holdings_age_days','pair_coverage_weight','current_profit_coverage_weight']
MARKET = PRICE_MARKET + NUISANCE
VALUATION = ['profit_direction_diffusion','profit_bounded_growth','quarterly_loss_weight']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def read_basket(root, binding):
    import qlib
    from qlib.data import D
    card = root/binding['source_card']
    if digest(card/'summary.json') != binding['summary_sha256']:
        raise ValueError('quarter source summary changed')
    for relative, expected in binding['provider_hashes'].items():
        if digest(root/relative) != expected:
            raise ValueError('quarter source provider changed')
    qlib.init(provider_uri=str(root/binding['provider_uri']), region='us', kernels=1)
    objects, audit = load_verified_recorder_objects(card, expected_recorder_id=binding['recorder_id'],
        expected_manifest_sha256=binding['manifest_sha256'], artifacts=[binding['lineage_artifact']])
    lineage = objects[binding['lineage_artifact']].copy()
    lineage['signal_at'] = pd.to_datetime(lineage.signal_at, utc=True).dt.tz_convert('America/New_York')
    lineage.index = pd.DatetimeIndex(lineage.signal_at).tz_localize(None).normalize()
    if not lineage.index.is_unique:
        raise ValueError('duplicate source month')
    if not lineage.signal_at.dt.strftime('%H:%M').eq('16:15').all():
        raise ValueError('source clock differs from frozen cutoff')
    values = D.features([binding['instrument']], list(binding['field_map'].values()),
                        start_time=str(lineage.index.min().date()), end_time=str(lineage.index.max().date()), freq='day')
    values = values.xs(binding['instrument'], level='instrument').rename(columns={v:k for k,v in binding['field_map'].items()})
    result = lineage[binding['metadata_columns']].join(values, how='left').rename(columns={'eligible':'source_qualified'})
    if not result.source_qualified.isin([True,False,0,1]).all():
        raise ValueError('unknown quarterly qualification')
    result['source_qualified'] = result.source_qualified.astype(bool)
    result['profit_direction_diffusion'] = result.profit_improve_weight-result.profit_deteriorate_weight
    if not np.isfinite(result.loc[result.source_qualified, NUISANCE+VALUATION].to_numpy()).all():
        raise ValueError('qualified quarterly source has unknown fields')
    return result, audit


def make_dataset(frame, train_mask, test_mask):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    selected = frame.loc[train_mask | test_mask]
    data = pd.concat({'feature': selected[MARKET+VALUATION], 'label': selected[TARGETS]}, axis=1)
    data.index = pd.MultiIndex.from_arrays([selected.index, ['SOXX_BASKET']*len(selected)], names=['datetime', 'instrument'])
    train_dates, test_dates = frame.index[train_mask], frame.index[test_mask]
    return DatasetH(handler=DataHandlerLP.from_df(data), segments={
        'train': (train_dates.min(), train_dates.max()), 'test': (test_dates.min(), test_dates.max())})


def prepare(root):
    """Source/calendar count gate, explicitly before any return label creation."""
    from qlib.workflow import R
    root = Path(root).resolve(); card = root/'research'/NAME
    plan = json.loads((card/'plan.json').read_text())
    review = json.loads((card/'design_review.json').read_text())
    if review['plan_sha256'] != digest(card/'plan.json') or review['decision'] != 'approved_for_execution_after_source_count_binding':
        raise ValueError('information design approval missing')
    binding = json.loads((card/'source_binding.json').read_text())
    source_identity = json.loads((card/'execution_source_identity.json').read_text())
    for file, expected in source_identity.items():
        if digest(root/file) != expected:
            raise ValueError('information implementation changed')
    if (card/'prepared_state.json').exists():
        state = json.loads((card/'prepared_state.json').read_text())
        if state['plan_sha256'] != digest(card/'plan.json') or state['source_binding_sha256'] != digest(card/'source_binding.json'):
            raise ValueError('prepared state identity changed')
        return state
    basket, audit = read_basket(root, binding)
    frame, assets, prices = read_market_inputs(root, plan)
    columns = ['source_qualified', *NUISANCE, *VALUATION]
    frame = frame.join(basket[columns], how='left')
    frame['source_qualified'] = frame.source_qualified.fillna(False).astype(bool)
    frame['market_complete'] = np.isfinite(frame[MARKET]).all(axis=1)
    frame.loc[~frame.source_qualified, VALUATION] = np.nan
    count_receipt = count_selection(frame, plan)
    year = count_receipt['selected_first_execution_year']
    if year is not None and not frame.loc[frame.execution_year.ge(year), 'market_complete'].all():
        raise ValueError('incomplete market features in fixed evaluation calendar')
    tracking = card/'qlib_recorder'
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path='inputs', plan=plan, source_binding=binding, source_audit=audit,
                         prelabel_frame=frame, assets=assets, market_prices=prices, basket=basket,
                         count_receipt=count_receipt, source_identity=source_identity)
        rec.save_objects(artifact_path='sources', **{p.replace('/', '__'):(root/p).read_text() for p in source_identity})
        rec.log_metrics(prelabel_months=len(frame), completed_market_fits=0)
    count_receipt.update(plan_sha256=digest(card/'plan.json'), source_binding_sha256=digest(card/'source_binding.json'))
    write_json(card/'prelabel_count_receipt.json', count_receipt)
    state = {'recorder_id':rec.id, 'experiment_id':str(rec.experiment_id), 'recorder_uri':str(tracking),
             'plan_sha256':digest(card/'plan.json'), 'source_binding_sha256':digest(card/'source_binding.json'),
             'code_identity_sha256':digest(card/'execution_source_identity.json'), 'count_receipt_sha256':digest(card/'prelabel_count_receipt.json'),
             'selected_first_execution_year':year, 'labels_computed':False, 'fits':0}
    write_json(card/'prepared_state.json', state)
    return state


def run(root):
    import yaml
    from qlib.workflow import R
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
    root = Path(root).resolve(); card = root/'research'/NAME
    state = prepare(root)
    if (card/'summary.json').exists():
        return json.loads((card/'summary.json').read_text())
    approval = json.loads((card/'model_execution_approval.json').read_text())
    for key, file in [('plan_sha256','plan.json'), ('source_binding_sha256','source_binding.json'),
                      ('count_receipt_sha256','prelabel_count_receipt.json'), ('code_identity_sha256','execution_source_identity.json')]:
        if approval[key] != digest(card/file):
            raise ValueError('model execution identity mismatch: '+key)
    if approval['decision'] != 'approved_bounded_information_fits' or state['selected_first_execution_year'] is None:
        raise ValueError('model fit not authorized by source/count review')
    plan = json.loads((card/'plan.json').read_text())
    import qlib
    qlib.init(provider_uri=str(root/plan['provider_uri']),region='us',kernels=1)
    R.set_uri(state['recorder_uri'])
    rec = R.get_recorder(recorder_id=state['recorder_id'], experiment_id=state['experiment_id'])
    prelabel = rec.load_object('inputs/prelabel_frame')
    frame = attach_labels(prelabel, rec.load_object('inputs/assets'))
    first_year = state['selected_first_execution_year']
    progress_path = card/'fit_attempts.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'attempted':[], 'completed':[]}
    predictions, audits = [], []
    with R.start(experiment_name=NAME, uri=state['recorder_uri'], recorder_id=state['recorder_id'], resume=True):
        rec = R.get_recorder()
        rec.save_objects(artifact_path='inputs', labelled_frame=frame, model_execution_approval=approval)
        for year in range(first_year, 2027):
            test = frame.execution_year.eq(year)
            cutoff = frame.loc[test, 'signal_at'].iloc[0]
            train = frame.source_qualified & frame.market_complete & frame.exit_at.lt(cutoff)
            if int(train.sum()) < plan['minimum_train_rows']:
                raise ValueError('insufficient paired training rows')
            dataset = make_dataset(frame, train, test)
            models, result = {}, frame.loc[test, ['signal_at','entry_at','exit_at','source_qualified',*TARGETS]].copy()
            for arm, features in [('control', MARKET), ('candidate', MARKET+VALUATION)]:
                key = f'{arm}_{year}'
                if key in progress['completed']:
                    model = rec.load_object('models/'+key)
                else:
                    if key in progress['attempted'] or len(progress['attempted']) >= plan['model']['maximum_fits']:
                        raise ValueError('fixed fit attempt budget exhausted')
                    progress['attempted'].append(key); write_json(progress_path, progress)
                    model = MonthlyValuationRidge(features, TARGETS, alpha=plan['model']['alpha'], minimum_label_std=plan['model']['minimum_label_std'])
                    model.fit(dataset)
                    rec.save_objects(artifact_path='models', **{key:model})
                    progress['completed'].append(key); write_json(progress_path, progress)
                prediction = model.predict(dataset).xs('SOXX_BASKET', level='instrument')
                if not prediction.index.equals(result.index):
                    raise ValueError('monthly prediction calendar differs')
                for target in TARGETS:
                    result[arm+'_'+target] = prediction[target]
                models[arm] = model
            for attribute in ['label_means','label_scales']:
                pd.testing.assert_series_equal(getattr(models['control'], attribute), getattr(models['candidate'], attribute))
            for attribute in ['medians','means','scales']:
                pd.testing.assert_series_equal(getattr(models['control'], attribute), getattr(models['candidate'], attribute).loc[MARKET])
            for target in TARGETS:
                result.loc[~result.source_qualified, 'candidate_'+target] = result.loc[~result.source_qualified, 'control_'+target]
                result['training_std_'+target] = models['control'].label_scales[target]
            for arm in models:
                result[arm+'_normalized_mse'] = sum(((result[arm+'_'+target]-result[target])/models['control'].label_scales[target])**2 for target in TARGETS)/2
            result['paired_improvement'] = result.control_normalized_mse-result.candidate_normalized_mse
            result['execution_year'] = year
            predictions.append(result)
            audits.append({'execution_year':year, 'cutoff':str(cutoff), 'training_months':int(train.sum()),
                           'last_train_maturity':str(frame.loc[train, 'exit_at'].max()), 'test_months':len(result),
                           'control_fallback_months':int((~result.source_qualified).sum()),
                           'candidate_inactive_features':models['candidate'].audit['inactive_features']})
            rec.log_metrics(completed_market_fits=len(progress['completed']))
        daily = pd.concat(predictions).sort_index()
        if not np.isfinite(daily[['control_normalized_mse','candidate_normalized_mse','paired_improvement']].to_numpy()).all():
            raise ValueError('nonfinite paired monthly scores; no silent month omission')
        periods = pd.DatetimeIndex(daily.entry_at).tz_localize(None).to_period('M')
        if not periods.equals(pd.period_range(periods[0], periods[-1], freq='M')):
            raise ValueError('monthly evaluation has gaps')
        annual = daily.groupby('execution_year').agg(months=('paired_improvement','size'), control_mse=('control_normalized_mse','mean'),
            candidate_mse=('candidate_normalized_mse','mean'), paired_improvement=('paired_improvement','mean'), qualified_months=('source_qualified','sum')).reset_index()
        annual['complete_year'] = annual.months.eq(12)
        complete = annual.loc[annual.complete_year]
        ci, indices, replicates = full_calendar_bootstrap(daily.paired_improvement.to_numpy(), **{k:plan['bootstrap'][k] for k in ['mean_block','resamples','seed','alpha']})
        gates = {'pooled_improvement_positive':ci['paired_mean_loss_improvement']>0,
                 'bootstrap_lower_positive':ci['lower_bound']>0,
                 'complete_year_70pct':bool(len(complete)>=2 and complete.paired_improvement.gt(0).mean()>=.7)}
        passed = all(gates.values())
        policy = {'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml',
                  'recorder_only_artifacts':['inputs/*','sources/*','models/*','analysis/monthly_scores','analysis/bootstrap_indices','analysis/bootstrap_replicates']}
        summary = {**state, 'experiment':NAME, 'decision':'information_increment_requires_separate_portfolio_design' if passed else 'not_promoted_fixed_quarterly_profit_information',
                   'labels_computed':True, 'fits':len(progress['completed']), 'market_fits':len(progress['completed']), 'new_strategy_accounts':0,
                   'evaluation_months':len(daily), 'qualified_evaluation_months':int(daily.source_qualified.sum()),
                   'first_entry':str(daily.entry_at.iloc[0]), 'last_exit':str(daily.exit_at.iloc[-1]),
                   'gates':gates, 'paired_normalized_MSE':ci, 'objective_achieved':False, 'qualified_independent_forward':False,
                   'source_proxy':'reported-quarter same-channel profit diffusion; current and paired coverage/age controlled in both arms; adjusted-price labels, not cash-account returns',
                   'provider_uri':str(root/plan['provider_uri']), 'as_of':plan['price_as_of'], 'region':'us',
                   'recorder_local_dir':str(Path(state['recorder_uri'])/state['experiment_id']/state['recorder_id']), 'artifact_policy':policy}
        rec.save_objects(artifact_path='analysis', monthly_scores=daily, annual_scores=annual, fit_audits=audits,
                         bootstrap_indices=indices, bootstrap_replicates=replicates, summary=summary)
        artifacts = Path(summary['recorder_local_dir'])/'artifacts'
        rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
    annual.to_csv(card/'annual_scores.csv', index=False)
    write_json(card/'fit_audit.json', audits)
    write_json(card/'verification_manifest.json', {'recorder_id':rec.id, 'entries':[
        {'path':str(p.relative_to(card)), 'sha256':digest(p), 'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]})
    source = root/json.loads((card/'source_binding.json').read_text())['source_card']
    binding = json.loads((source/'nas_binding.json').read_text()); binding.pop('evidence',None)
    write_json(card/'nas_binding.json', binding)
    summary['nas_evidence'] = archive_nas_recorder(card/'nas_binding.json', workspace_root=root, card=card, recorder_id=rec.id, experiment_id=state['experiment_id'])
    write_json(card/'summary.json', summary)
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'plan':'plan.json','module':'src.workflow.quarterly_profit_information','artifact_policy':policy},sort_keys=False))
    conclusion = f'''# 季度盈利扩散月度信息检验

完成{summary['market_fits']}次固定Ridge拟合，{len(daily)}个连续月度评价位置，{summary['qualified_evaluation_months']}个月有合格季度盈利扩散代理；其余候选严格使用匹配对照预测。全部年度见annual_scores.csv，最后不足12个月的年度单列。

控制减候选的双标签等权、训练方差归一MSE改善为{ci['paired_mean_loss_improvement']:.8f}，固定月度区块区间[{ci['lower_bound']:.8f}, {ci['upper_bound']:.8f}]。完整年度通过门为{gates['complete_year_70pct']}。未增加窗口、种子、正则或特征搜索。

决定：{summary['decision']}。没有组合账户、净收益/回撤证据或晋升；稳定超额目标未达成。季度利润口径和调整价格标签均为明确研究代理；缺失与披露时点仍可能内生，不能解释成干净的因果效应，所有评价历史已复用，不是独立前向。

下一步：{'预测信息门通过，但不能据此认定配置有效。' if passed else '关闭这套固定季度盈利扩散均值预测组合；只复用保存预测作有限失败归因，不调参救援本版本。'} 在本结果揭晓前已独立登记的季度利润直接配置实验使用直接组合效用，不使用本模型预测，其执行不以本次MSE结果有利为条件；仍须单独通过来源、扣费超额、风险与稳定性检验。
'''
    (card/'conclusion.md').write_text(conclusion)
    return summary
