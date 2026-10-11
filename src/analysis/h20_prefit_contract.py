"""Plan-driven native input qualification, without fitting or investment replay."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.data.generic_horizon import (actual_calendar_prices, attach_horizon_scope,
    build_horizon_labels_from_qlib, mature_mask, require_horizon_scope, validate_price_calendar)
from src.data.provider_content_hash import provider_content_hash


def _require(condition, message):
    if not condition:
        raise ValueError('blocked: ' + message)


def run_prefit_contract(*, root, card_dir) -> dict:
    """Read frozen providers and a genuine 25-feature/two-label DatasetH. No fit."""
    root, card = Path(root).resolve(), Path(card_dir).resolve()
    plan_path = card / 'preregister_plan.json'
    plan = json.loads(plan_path.read_text())
    binding, h = plan['source_binding'], plan['horizon']['H20_candidate']
    _require(h == 20 and plan['horizon']['H5_reference'] == 5, 'registered horizons differ')
    _require(plan['authorization']['training_authorized'] is False
             and plan['authorization']['replay_authorized'] is False, 'prefit requires an unexecuted plan')
    stores = {key: root / binding[key] for key in ('price_features_provider', 'execution_provider')}

    def checked_hashes():
        values = {}
        for key, store in stores.items():
            digest, inventory = provider_content_hash(store)
            _require(digest == binding['provider_content_sha256'][store.name], 'frozen provider differs: ' + key)
            _require(inventory, 'empty provider inventory')
            values[key] = digest
        return values

    before = checked_hashes()
    start, end = binding['data_range']
    observed = {}
    for key, store in stores.items():
        calendar, prices = actual_calendar_prices(store, start_time=start, end_time=end, symbols=('SOXL', 'SOXX'))
        validate_price_calendar(calendar, prices)
        observed[key] = (calendar, prices)
    pcal, pprices = observed['price_features_provider']
    ecal, eprices = observed['execution_provider']
    _require(pcal.equals(ecal), 'frozen provider calendars differ')
    pd.testing.assert_frame_equal(pprices, eprices, check_exact=True, check_freq=False)
    label_frames, metrics = {}, {}
    for horizon in (plan['horizon']['H5_reference'], h):
        labels, audit = build_horizon_labels_from_qlib(pd.DataFrame(index=ecal),
            provider_uri=stores['execution_provider'], start_time=start, end_time=end, horizon_sessions=horizon)
        unknown = labels[[f'soxl_return_h{horizon}', f'soxx_return_h{horizon}']].isna().any(axis=1)
        availability = labels[f'label_h{horizon}_available_at']
        _require(int(unknown.sum()) == horizon + 1 and not unknown.iloc[:-(horizon + 1)].any()
                 and unknown.iloc[-(horizon + 1):].all(), 'unknown labels must form the terminal suffix')
        _require(availability.isna().equals(unknown), 'outcome and availability missingness differ')
        mask = mature_mask(availability, plan['oof_and_terminal']['evaluation_cutoff'])
        metrics[str(horizon)] = dict(rows=len(labels), terminal_unknown=int(unknown.sum()),
            mature_rows=int(mask.sum()), last_mature_signal=str(labels.index[mask][-1].date()),
            label_provider=binding['execution_provider'], mean_unit=audit['mean_unit'],
            availability_clock=audit['availability_clock'])
        label_frames[horizon] = labels

    from src.data.convex_allocation import FEATURE_COLUMNS
    from src.data.integrated_allocation import build_native_dataset, frame_sha256
    from src.data.oos_horizon_inputs import prepare_horizon_inputs
    parent_name = plan['h5_reference']['card']
    parent_card = root / 'research' / parent_name
    expected = binding['reference_parents'][parent_name]
    _require(hashlib.sha256((parent_card / 'summary.json').read_bytes()).hexdigest() == expected['summary_sha256'],
             'H5 reference summary identity differs')
    parent_plan = json.loads((parent_card / 'plan.json').read_text())
    _require(parent_plan['source']['provider_uri'] == binding['price_features_provider']
             and parent_plan['source']['execution_provider_uri'] == binding['execution_provider']
             and parent_plan['source']['as_of'] == binding['as_of'], 'parent source differs')
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
    bundle = prepare_horizon_inputs(root, parent_plan)
    _require(tuple(plan['features']['list']) == FEATURE_COLUMNS, 'registered features differ')
    training_frame, _ = build_horizon_labels_from_qlib(bundle['frame'],
        provider_uri=stores['execution_provider'], start_time=str(bundle['calendar'][0].date()),
        end_time=binding['as_of'], horizon_sessions=h)
    label_columns = [f'soxl_return_h{h}', f'soxx_return_h{h}']
    metadata = ['signal_date', 'feature_available_at', 'decision_at', 'trade_date', 'exit_date',
                'return_available_at', f'exit_date_h{h}', f'label_h{h}_available_at']
    dataset = build_native_dataset(training_frame, feature_columns=FEATURE_COLUMNS,
        label_columns=label_columns, metadata_columns=metadata, as_of=binding['as_of'])
    attach_horizon_scope(dataset, h)
    dataset.calendar = bundle['calendar']
    require_horizon_scope(dataset, h)
    prepared = dataset.prepare('all', col_set=['feature', 'label', 'metadata'])
    _require(tuple(prepared['feature'].columns) == FEATURE_COLUMNS
             and list(prepared['label'].columns) == label_columns, 'native schema differs')
    _require(np.isfinite(prepared['feature'].to_numpy(float)).all(), 'nonfinite native features')
    _require(int(prepared['label'].isna().any(axis=1).sum()) == h + 1, 'DatasetH lost unknown outcomes')
    wrong_blocked = False
    try:
        require_horizon_scope(dataset, plan['horizon']['H5_reference'])
    except ValueError:
        wrong_blocked = True
    _require(wrong_blocked, 'wrong DatasetH horizon accepted')
    after = checked_hashes()
    _require(before == after, 'provider changed during readback')
    final = dict(schema_version='h20_prefit_contract_v3', experiment=plan['experiment'],
        stage='prefit_input_contract', status='prefit_input_contract_pass_model_integration_pending',
        fit_performed=False, models_trained=0, investment_outcomes=False,
        promotion_eligible=False, promotion_decision='not_promoted', real_model_integration_pending=True,
        source=dict(provider_uri=binding['price_features_provider'], execution_provider_uri=binding['execution_provider'],
                    region=binding['region'], as_of=binding['as_of'], data_range=[start, end]),
        provider_content_sha256_before=before, provider_content_sha256_after=after,
        provider_unchanged_by_readback=True, calendar_rows=len(ecal),
        native_open_feature_long_rows=pprices.size, wide_open_price_shape=list(pprices.shape), horizons=metrics,
        native_training_dataset=dict(class_name=type(dataset).__name__, rows=len(prepared), features=len(FEATURE_COLUMNS),
            labels=label_columns, first_signal=str(training_frame.index[0].date()), terminal_unknown=h + 1,
            wrong_horizon_blocked=True, calendar_bound=True, prepared_sha256=frame_sha256(prepared)),
        parent_h5_financial_target_impact='unknown', parent_h5_fully_comparable=False, qlib_docs=plan['qlib_docs'],
        artifact_policy=dict(policy='slim_card_recorder_first', workflow_config='workflow.yaml',
            recorder_only_artifacts=['prefit/labels_h5.pkl', 'prefit/labels_h20.pkl',
                                    'prefit/training_dataset.pkl', 'prefit/horizon_metrics.pkl']))
    import qlib
    from qlib.workflow import R
    uri = (card / 'qlib_recorder').resolve()
    qlib.init(provider_uri=str(stores['price_features_provider']), region=binding['region'], kernels=1)
    with R.start(experiment_name=plan['experiment'], recorder_name='prefit_input_contract', uri=str(uri)):
        rec = R.get_recorder()
        rec.log_params(stage=final['stage'], horizon_sessions=h, models_trained=0,
                       plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
        rec.save_objects(artifact_path='prefit', **{f'labels_h{k}.pkl': v for k, v in label_frames.items()},
                         **{'training_dataset.pkl': prepared, 'horizon_metrics.pkl': pd.DataFrame(metrics).T})
        final.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id), recorder_uri=str(uri),
                     recorder_local_dir=rec.get_local_dir())
        rec.save_objects(artifact_path='prefit', **{'summary.pkl': final})
    _require(rec.status == 'FINISHED', 'Recorder did not finish')
    _require(set(final['artifact_policy']['recorder_only_artifacts']).issubset(rec.list_artifacts('prefit')),
             'declared Recorder artifacts missing')
    final['recorder_status'] = rec.status
    final['source_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        'src/analysis/h20_prefit_contract.py', 'src/data/generic_horizon.py',
        'src/data/provider_content_hash.py')}
    entries = []
    for name in final['artifact_policy']['recorder_only_artifacts']:
        path = Path(final['recorder_local_dir']) / 'artifacts' / name
        entries.append(dict(path=str(path.relative_to(card)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            bytes=path.stat().st_size))
    (card / 'verification_manifest.json').write_text(json.dumps(
        dict(recorder_id=rec.id, entries=entries), indent=2) + '\n')
    # Save final metadata to this newly completed run without resuming or changing its status.
    rec.save_objects(artifact_path='prefit', **{'summary.pkl': final})
    (card / 'summary.json').write_text(json.dumps(final, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (card / 'workflow.yaml').write_text(yaml.safe_dump(dict(experiment=plan['experiment'], stage=final['stage'],
        source=final['source'], recorder_id=rec.id, recorder_uri=str(uri), artifact_policy=final['artifact_policy']), sort_keys=False))
    (card / 'selected_model.json').write_text(json.dumps(dict(selected_model=None, reason='input qualification only; no fit',
        candidate_horizon=h, reference_horizon=plan['horizon']['H5_reference'], promotion_decision='not_promoted'),
        indent=2, ensure_ascii=False) + '\n')
    (card / 'conclusion.md').write_text('# H20 输入资格检查\n\n'
        f'实际 Qlib 双来源读回 {len(ecal)} 个评估交易日，哈希与冻结计划一致且未变化。'
        f'H20 成熟标签 {metrics[str(h)]["mature_rows"]} 条，末尾 {h + 1} 条保持未知。\n\n'
        f'实际 DatasetH 有 {len(prepared)} 条记录、{len(FEATURE_COLUMNS)} 个特征、两个 H20 标签，'
        f'从 {training_frame.index[0].date()} 开始，错误 horizon 被阻断。'
        '这仅验证输入，未训练、未回测，无收益或 regime/window 投资结论。\n\n'
        '决定：not_promoted。H5 经济影响尚未核验，下一步完成原生模型和选择接口，再执行固定 H20 实验。'
        f'Recorder `{rec.id}`，状态 `{rec.status}`。\n')
    return final
