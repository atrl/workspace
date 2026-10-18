"""Frozen paired S/Z supervision with a train-only, six-model access barrier."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.day_night_labels import CLOCKS, TARGETS, original_target_digest
from src.data.minute_intake import atomic_json
from src.workflow.preopen_tom_allocation import (
    _native_dataset, _prepare_all_groups, _original_tensors, audit_training_residency,
)

NAME = 'stage1_day_night_supervision_20261018'
ARMS = {'B_day_night_total': 'S', 'C_day_night_contrast': 'Z'}
SEEDS = [71, 172, 273]
CANDIDATE = 'C_day_night_contrast'
REQUIRED = ('SOXX', 'B_day_night_total', 'B_shared', 'A_active_research_replay', 'strong9097')
FAMILIES = (*REQUIRED, CANDIDATE, 'additional27baa')
DATES = ['signal_date', 'trade_date', 'exit_date']
OUTPUTS = ['soxl_weight', 'soxx_weight', 'cash_weight', 'tail_probability', 'alpha_probability']
EXECUTION_PROVIDER = 'data/qlib/stage1_day_night_supervision_20261018_execution'
CODE_FILES = ['src/workflow/day_night_supervision.py', 'src/model/day_night_supervision_allocator.py',
    'tests/test_day_night_supervision_allocator.py', 'src/model/minute_pit_allocator.py',
    'src/workflow/preopen_tom_allocation.py', 'src/workflow/minute_pit_allocator.py',
    'src/workflow/pit_constituent_auxiliary_training_allocation.py', 'src/workflow/aux_projection_preflight.py',
    'src/workflow/stage1_domain_heads.py', 'src/backtest/stage1_domain_heads_raw.py',
    'src/backtest/native_raw_corporate.py', 'src/backtest/raw_corporate_policy.py',
    'src/backtest/qlib_continuous_position.py', 'src/backtest/ibkr_tiered.py',
    'src/workflow/stage1_open_action_value.py', 'src/workflow/intraday_recovery_direct_allocation.py',
    'src/analysis/stage1_domain_heads.py', 'src/analysis/integrated_allocation.py',
    'src/core/recorder_sources.py', 'src/data/minute_intake.py', 'src/data/qlib_native.py',
    'src/data/nas_market_data.py', 'src/data/day_night_labels.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root):
    return {name: sha(Path(root) / name) for name in CODE_FILES}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _verify_bytes(root, source):
    card = root / source['card']; summary = json.loads((card / 'summary.json').read_text())
    _require(summary['recorder_id'] == source['recorder_id'], 'source Recorder differs')
    _require(sha(card / 'verification_manifest.json') == source['inventory_sha256'], 'source manifest differs')
    _require(json.loads((card / 'nas_binding.json').read_text())['evidence']['sha256'] == source['archive'], 'source archive differs')
    for name, digest in source['artifact_sha256'].items():
        _require(sha(Path(summary['recorder_local_dir']) / 'artifacts' / name) == digest, 'source bytes changed: ' + name)


def _read(root, source):
    _verify_bytes(root, source)
    return load_verified_recorder_objects(root / source['card'], expected_recorder_id=source['recorder_id'],
        artifacts=source['artifacts'], expected_manifest_sha256=source['inventory_sha256'],
        expected_artifact_sha256=source['artifact_sha256'])


def _source_c(root, plan):
    source = plan['source_C']
    for name in ('inventory', 'summary', 'source_plan', 'result_review'):
        spec = source[name]
        _require(sha(root / spec['path']) == spec['sha256'], 'source C ' + name + ' differs')
    review = json.loads((root / source['result_review']['path']).read_text())
    _require(review.get('verdict') == 'ACCEPT_CONDITIONAL_TRAINING_LABEL_SOURCE'
             and review.get('source_label_qualified') is True, 'source C not qualified')
    summary = json.loads((root / source['summary']['path']).read_text())
    _require(summary['recorder_id'] == source['recorder_id'] and summary['fixed_label_rows'] == 3014
             and summary['original_targets_unchanged'] is True and summary['algebra_all_passed'] is True,
             'source C qualified cohort differs')
    _require(summary['new_fits'] == summary['new_accounts'] == 0
             and summary['historical_PIT_qualified'] is False and summary['independent_forward_qualified'] is False,
             'source-only qualification boundary differs')
    binding = {'card': source['card'], 'recorder_id': source['recorder_id'], 'archive': source['archive'],
        'inventory_sha256': source['inventory']['sha256'], 'artifacts': list(source['artifact_sha256']),
        'artifact_sha256': source['artifact_sha256']}
    return binding, summary


def _dates(frame, instrument, rows):
    _require(isinstance(frame.index, pd.MultiIndex) and list(frame.index.names) == ['datetime', 'instrument'],
             'native training index schema differs')
    _require(set(frame.index.get_level_values('instrument')) == {instrument}, 'unexpected source instrument')
    dates = pd.DatetimeIndex(frame.index.get_level_values('datetime'))
    _require(len(frame) == rows and dates.is_unique and dates.is_monotonic_increasing and dates.tz is None
             and dates.equals(dates.normalize()), 'original full-cohort date identity differs')
    return dates


def append_supervision(primary, source_labels, reference, mapping, *, expected_rows=3014,
                       expected_original_columns=85, expected_digest=None):
    """Strict date proof then positional assignment; never join/fill/rename PG index."""
    dates = _dates(primary, 'SOXL_SOXX_CASH', expected_rows)
    source_dates = _dates(source_labels, 'SOXL_SOXX', expected_rows)
    pd.testing.assert_index_equal(dates, source_dates, check_names=False)
    source_columns = [('original', name) for name in (*CLOCKS, *TARGETS)] + [
        ('fixed_day_night_label', name) for name in ('D_SOXL', 'N_SOXL', 'endpoint_SOXL', 'D_SOXX', 'N_SOXX',
            'endpoint_SOXX', 'D_relative', 'N_relative', 'S', 'Z', 'S_endpoint')]
    _require(list(source_labels.columns) == source_columns, 'accepted source-C 19-column grouped schema differs')
    _require(len(primary.columns) == expected_original_columns and 'supervision' not in primary.columns.get_level_values(0),
             'original columns changed or supervision already appended')
    original = pd.concat([primary['meta'].loc[:, list(CLOCKS)], primary['label'].loc[:, list(TARGETS)]], axis=1).copy()
    original.index = pd.DatetimeIndex(dates, name='signal_date')
    source_original = source_labels['original'].copy()
    source_original.index = pd.DatetimeIndex(source_dates, name='signal_date')
    pd.testing.assert_frame_equal(original, source_original, check_exact=True)
    pd.testing.assert_frame_equal(original, reference, check_exact=True)
    pd.testing.assert_index_equal(mapping.index, original.index, check_names=False)
    _require(len(mapping) == expected_rows and list(mapping.columns) == ['d' + str(i) for i in range(6)], 'H5 mapping changed')
    for field, day in zip(CLOCKS, ('d0', 'd1', 'd5')):
        actual = pd.DatetimeIndex(original[field]).tz_convert('America/New_York').tz_localize(None).normalize()
        pd.testing.assert_index_equal(actual, pd.DatetimeIndex(mapping[day]), check_names=False)
    digest = original_target_digest(original)
    if expected_digest is not None:
        _require(digest == expected_digest, 'original H1/H5/clock digest differs from accepted source C')
    raw = source_labels['fixed_day_night_label'].loc[:, ['S', 'Z']]
    _require(all(dtype == np.dtype('float64') for dtype in raw.dtypes) and np.isfinite(raw.to_numpy()).all(),
             'fixed S/Z must be complete finite float64 values')
    augmented = primary.copy()
    for name in ('S', 'Z'):
        augmented[('supervision', name)] = raw[name].to_numpy(copy=True)
    pd.testing.assert_frame_equal(augmented.iloc[:, :expected_original_columns], primary, check_exact=True)
    pd.testing.assert_index_equal(augmented.index, primary.index, exact=True)
    _require(augmented.shape == (expected_rows, expected_original_columns + 2), 'two-column append changed cohort')
    raw_targets = augmented['supervision'].copy()
    return augmented, raw_targets, {'rows': expected_rows, 'original_columns': expected_original_columns,
        'augmented_columns': expected_original_columns + 2, 'source_instrument': 'SOXL_SOXX',
        'primary_instrument': 'SOXL_SOXX_CASH', 'original_target_sha256': digest,
        'original_index_and_columns_exact': True, 'datetime_one_to_one_exact_order': True,
        'join_drop_fill_or_index_rename': False}


def _new_model(old, target, scalers, root):
    from src.model.day_night_supervision_allocator import DayNightSupervisionAllocator
    model = DayNightSupervisionAllocator(target_name=target, target_scaler=scalers['stats'][target],
        base_features=old.base_features, extra_features=[], feature_masks=dict(old.feature_masks), spec={**asdict(old.spec), 'root': str(root)},
        seed=old.seed, cutoff=str(old.cutoff), train_end=str(old.train_end.date()), minimum_observations=old.minimum_observations)
    _require(asdict(model.candidate) == asdict(old.candidate), 'complete original objective candidate changed')
    _require({k: v for k, v in asdict(model.spec).items() if k != 'root'} ==
             {k: v for k, v in asdict(old.spec).items() if k != 'root'}, 'complete original spec changed')
    for name in ('base_features', 'extra_features', 'feature_masks', 'seed', 'cutoff', 'train_end', 'minimum_observations'):
        _require(getattr(model, name) == getattr(old, name), 'constructor differs: ' + name)
    _require(len(model.base_features) == 39 and not model.extra_features, 'only original39/16 input model allowed')
    return model


def _stage_approval(root, card, plan, stage):
    from src.workflow.aux_projection_preflight import _software
    _require(plan.get('frozen') is True, 'plan not frozen')
    torch.set_num_threads(1)
    _require(_software() == plan['software_reference'], 'frozen numerical backend differs')
    for spec in plan['unchanged_code_authorities']:
        _require(sha(root / spec['path']) == spec['sha256'], 'unchanged code authority drifted')
    for spec in [plan['active_release'], *plan['inheritance'].values(), *plan['prospective_adoption'].values()]:
        if isinstance(spec, dict) and 'path' in spec and 'sha256' in spec:
            _require(sha(root / spec['path']) == spec['sha256'], 'frozen contract/active source changed')
    identity_file = card / 'implementation_identity.json'
    identity = json.loads(identity_file.read_text()); digest = sha(card / 'plan.json'); code = code_identity(root)
    _require(identity['plan_sha256'] == digest and identity['code_sha256'] == code, 'implementation identity differs')
    prefix = 'preflight_' if stage == 'preflight' else ''
    env = 'DAY_NIGHT_SUPERVISION_PREFLIGHT' if stage == 'preflight' else 'DAY_NIGHT_SUPERVISION_EXECUTE'
    _require(os.environ.get(env) == '1', stage + ' not enabled')
    other_env = 'DAY_NIGHT_SUPERVISION_EXECUTE' if stage == 'preflight' else 'DAY_NIGHT_SUPERVISION_PREFLIGHT'
    _require(os.environ.get(other_env) != '1', 'phase environments must not overlap')
    review_file = card / (prefix + 'design_review.json'); review = json.loads(review_file.read_text())
    verdict = 'APPROVE_NO_FIT_PREFLIGHT' if stage == 'preflight' else 'APPROVE_DESIGN'
    _require(review.get('verdict') == verdict and review.get('plan_sha256') == digest and review.get('code_sha256') == code,
             stage + ' review is missing or stale')
    owner_file = card / (prefix + 'execution_approval.json'); owner = json.loads(owner_file.read_text())
    _require(owner.get('approved') is True and owner.get('plan_sha256') == digest and owner.get('code_sha256') == code
             and owner.get('review_sha256') == sha(review_file) and owner.get('implementation_identity_sha256') == sha(identity_file),
             stage + ' owner GO absent/stale')
    expected = {'training_only_no_fit': True, 'fits': 0, 'policy_paths': 0, 'native_accounts': 0, 'new_market_HTTP': 0} if stage == 'preflight' else {
        'fits': 6, 'policy_paths': 6, 'native_accounts': 6, 'reused_native_accounts': 15, 'paired_intervals': 15, 'new_market_HTTP': 0}
    _require(owner.get('scope') == expected, 'stage execution scope differs')
    if stage == 'fit':
        for key, path in [('preflight_manifest_sha256', card / 'preflight/verification_manifest.json'),
                          ('preflight_summary_sha256', card / 'preflight/summary.json')]:
            _require(owner.get(key) == review.get(key) == sha(path), 'both final review and GO must bind actual preflight')
    _source_c(root, plan)  # metadata only; no objects here
    return {'plan_sha256': digest, 'code_sha256': code, 'review_sha256': sha(review_file),
            'owner_approval_sha256': sha(owner_file), 'implementation_identity_sha256': sha(identity_file)}


def _seal(rec, root, out, summary, transport_card):
    from src.data.nas_market_data import archive_nas_recorder
    artifacts = Path(summary['recorder_local_dir']) / 'artifacts'
    atomic_json({'recorder_id': rec.id, 'entries': [{'path': str(p.relative_to(out)), 'sha256': sha(p),
        'bytes': p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]}, out / 'verification_manifest.json')
    transport = json.loads((root / transport_card / 'nas_binding.json').read_text())
    atomic_json({'schema_version': 'recorder_evidence_binding_v1', 'authority': 'yanxuan_nas', 'ssh': transport['ssh'],
        'source_id': NAME + '-' + summary['plan_sha256'], 'as_of': summary['as_of']}, out / 'nas_binding.json')
    summary['nas_evidence'] = archive_nas_recorder(out / 'nas_binding.json', workspace_root=root, card=out,
        recorder_id=rec.id, experiment_id=str(rec.experiment_id))
    atomic_json(summary, out / 'summary.json')
    return summary


def prepare(*, root, card):
    """Only separately approved no-fit stage: original train-only and source-C objects."""
    import qlib
    from qlib.workflow import R
    from src.model.day_night_supervision_allocator import fit_supervision_scalers, initialization_contract
    root, card = Path(root).resolve(), Path(card).resolve(); plan = json.loads((card / 'plan.json').read_text())
    approval = _stage_approval(root, card, plan, 'preflight')
    out = card / 'preflight'; out.mkdir(exist_ok=True); state_path = out / 'run_state.json'
    _require(not state_path.exists(), 'preflight attempt exists; no automatic repeat')
    state = {'status': 'started', 'plan_sha256': approval['plan_sha256'], 'new_fits': 0, 'new_accounts': 0,
             'evaluation_objects_deserialized': 0, 'new_price_queries': 0}
    atomic_json(state, state_path)
    try:
        qlib.init(provider_uri=str(root / plan['provider_and_native_workflow']['init_provider_uri']), region='us', kernels=1)
        sources = plan['training_sources']
        primary, primary_audit = _read(root, sources['training_only'])
        originals, model_audit = _read(root, sources['original_models'])
        tensors, tensor_audit = _read(root, sources['mechanical_tensors'])
        source_binding, source_summary = _source_c(root, plan)
        labels, label_audit = _read(root, source_binding)
        dataset, auxiliary = primary['inputs/training_dataset'], primary['inputs/auxiliary_dataset']
        residency = [audit_training_residency(dataset, expected_rows=3014, expected_columns=85, maximum_signal_date='2022-12-21', role='primary'),
            audit_training_residency(auxiliary, expected_rows=20741, expected_columns=20, maximum_signal_date='2022-12-22', role='auxiliary')]
        train = _prepare_all_groups(dataset, 'train'); aux = _prepare_all_groups(auxiliary, 'train')
        source_labels = _prepare_all_groups(labels['labels/fixed_H5_components_S_Z'], 'all')
        # Source Dataset is also admitted as whole train-only resident data; no setup_data.
        _require(set(labels['labels/fixed_H5_components_S_Z'].segments) == {'all'}, 'unexpected label-source segments')
        for name in ('_data', '_infer', '_learn'):
            frame = getattr(labels['labels/fixed_H5_components_S_Z'].handler, name, None)
            _require(isinstance(frame, pd.DataFrame), 'label Dataset lacks self-contained native cache')
            pd.testing.assert_frame_equal(frame, source_labels, check_exact=True)
        maturity = train[('meta', 'label_available_at')]
        _require((maturity.notna() & maturity.lt(pd.Timestamp(plan['training']['cutoff']))).all()
                 and np.isfinite(train['label'][list(TARGETS)].to_numpy()).all(), 'original mature/finite cohort changed')
        _require(pd.to_datetime(aux[('meta', 'observations_end')]).max() <= pd.Timestamp(plan['training']['train_end']), 'auxiliary maturity differs')
        augmented, raw_targets, alignment = append_supervision(train, source_labels,
            labels['reference/original_train_index_clocks_and_targets'], labels['reference/h5_mapping'],
            expected_digest=source_summary['original_target_sha256'])
        scales = fit_supervision_scalers(raw_targets)  # exactly once, same all3014 rows for S and Z
        paired_dataset = _native_dataset(augmented, {'train': ('2011-01-03', '2022-12-30')})
        auxiliary.config(dump_all=True, recursive=True)
        residency.append(audit_training_residency(paired_dataset, expected_rows=3014, expected_columns=87,
            maximum_signal_date='2022-12-21', role='paired_supervision_primary'))
        original_tensors, ps, cs = _original_tensors(train, aux, originals['models/base_outer_71'])
        for name, value in original_tensors.items():
            _require(torch.equal(value, tensors['inputs/tensors'][name]), 'original training tensor differs: ' + name)
            _require(hashlib.sha256(value.numpy().tobytes()).hexdigest() == primary['inputs/training_audit']['tensor_sha256'][name],
                     'original training tensor hash differs: ' + name)
        initial = []
        for seed in SEEDS:
            old = originals[f'models/base_outer_{seed}']
            _require(old.seed == seed and old.cutoff == pd.Timestamp(plan['training']['cutoff'])
                     and old.train_end == pd.Timestamp(plan['training']['train_end']) and old.spec.epochs == 300,
                     'original training metadata differs')
            _require(old.primary_scaler == ps and old.common_scaler == cs, 'original scaler differs')
            left, right = _new_model(old, 'S', scales, root), _new_model(old, 'Z', scales, root)
            initial.append({'seed': seed, **initialization_contract(old, left, right)})
        # Verification of eval artifact bytes only. No eval object is deserialized.
        for binding in plan['deferred_evaluation_sources'].values():
            _verify_bytes(root, binding)
        tracking = out / 'qlib_recorder'
        with R.start(experiment_name=NAME + '_preflight', uri=str(tracking)):
            rec = R.get_recorder(); state.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id)); atomic_json(state, state_path)
            rec.save_objects(artifact_path='inputs', plan=plan, approval=approval, primary_dataset=paired_dataset,
                auxiliary_dataset=auxiliary, target_scaling=scales, original_models=originals,
                training_audit=primary['inputs/training_audit'])
            rec.save_objects(artifact_path='analysis', alignment=alignment, initialization=initial, residency=residency,
                source_audits={'primary': primary_audit, 'models': model_audit, 'tensors': tensor_audit, 'labels': label_audit})
            for name, expected in [('primary_dataset', augmented), ('auxiliary_dataset', aux)]:
                pd.testing.assert_frame_equal(_prepare_all_groups(rec.load_object('inputs/' + name), 'train'), expected, check_exact=True)
            summary = {'experiment': NAME + '_preflight', 'plan_sha256': approval['plan_sha256'], 'status': 'completed',
                'decision': 'eligible_for_formal_design_review', 'fit_eligible': True, 'primary_rows': 3014, 'auxiliary_rows': 20741,
                'original_columns': 85, 'augmented_columns': 87, 'original_target_sha256': alignment['original_target_sha256'],
                'all_original_training_tensors_scalers_exact': True, 'all_resident_frames_train_only': True,
                'target_scaling_fitted_once': True, 'target_scaling_finite': True, 'no_predictive_score_screen': True,
                'initialization': initial, 'evaluation_objects_deserialized': 0, 'new_price_queries': 0,
                'new_fits': 0, 'new_accounts': 0, 'new_source_HTTP': 0, 'recorder_id': rec.id,
                'experiment_id': str(rec.experiment_id), 'recorder_uri': str(tracking),
                'recorder_local_dir': str(tracking / str(rec.experiment_id) / rec.id),
                'provider_uri': str(root / plan['provider_and_native_workflow']['init_provider_uri']), 'as_of': '2022-12-30',
                'historical_PIT_qualified': False, 'independent_forward_qualified': False,
                'artifact_policy': {**plan['artifact_policy'], 'workflow_config': '../workflow.yaml'}}
            rec.save_objects(artifact_path='analysis', summary=summary)
            rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in (Path(summary['recorder_local_dir']) / 'artifacts').rglob('*')))
        summary = _seal(rec, root, out, summary, sources['training_only']['card'])
        atomic_json(summary, card / 'preflight.json'); state['status'] = 'completed'; atomic_json(state, state_path)
        (card / 'conclusion.md').write_text('# S/Z 监督：无拟合预检完成\n\n只完成获批的训练域来源/严格日期与instrument映射、85→87列保真、原tensor/scaler与初始化核验。原3014/20741和全部目标/时钟未变；两套S/Z缩放常数仅拟合一次，未评判可预测性。\n\n0真实optimizer step、0fit/account、0评价对象或新价格查询。正式6fits及6成本独立账户仍需最终设计评审与单独owner GO；不因标签或软件检查通过而晋升。\n')
        return summary
    except BaseException as error:
        state.update(status='failed', exception_type=type(error).__name__, exception=str(error)); atomic_json(state, state_path)
        raise


def _own_preflight_source(card):
    pre = card / 'preflight'; summary = json.loads((pre / 'summary.json').read_text())
    _require(summary.get('fit_eligible') is True and summary.get('status') == 'completed', 'preflight not eligible')
    manifest = json.loads((pre / 'verification_manifest.json').read_text())
    names = ['inputs/primary_dataset', 'inputs/auxiliary_dataset', 'inputs/target_scaling', 'inputs/original_models', 'inputs/training_audit']
    digests = {}
    for name in names:
        rows = [row for row in manifest['entries'] if row['path'].endswith('/artifacts/' + name)]
        _require(len(rows) == 1, 'preflight artifact ambiguous/missing')
        digests[name] = rows[0]['sha256']
    return {'card': str(pre), 'recorder_id': summary['recorder_id'],
        'archive': json.loads((pre / 'nas_binding.json').read_text())['evidence']['sha256'],
        'inventory_sha256': sha(pre / 'verification_manifest.json'), 'artifacts': names, 'artifact_sha256': digests}


def _claim(state, path, phase, key, maximum):
    _require(key not in state[phase + '_attempts'] and len(state[phase + '_attempts']) < maximum,
             'consumed ' + phase + ' attempt; no automatic retry')
    state[phase + '_attempts'].append(key); atomic_json(state, path)


def _model_barrier(rec, artifact_root, state, plan):
    expected = [f'{family}_{seed}' for family, seed in plan['training']['fit_order']]
    _require(len(expected) == 6 and state['fits_done'] == expected and set(state['model_sha256']) == set(expected), 'six-model barrier incomplete')
    for key in expected:
        _require(sha(artifact_root / 'models' / key) == state['model_sha256'][key], 'sealed final model changed')
    return {key: rec.load_object('models/' + key) for key in expected}


def _evaluate(root, plan, models, prepared):
    """No S/Z labels are created or required here; only after all six models seal."""
    from src.workflow.minute_pit_allocator import _prediction_dates, _ensemble
    from src.workflow.pit_constituent_auxiliary_training_allocation import _smooth_weights
    from src.analysis.stage1_domain_heads import information_metrics
    _require(len(models) == 6, 'incomplete model set')
    sources, audits = {}, {}
    for role, source in plan['deferred_evaluation_sources'].items():
        sources[role], audits[role] = _read(root, source)
    dataset = sources['prediction_dataset']['evaluation/dataset']
    original = _prepare_all_groups(dataset, 'predict')
    _require(original.shape == (946, 85) and 'supervision' not in original.columns.get_level_values(0), 'original prediction-only schema differs')
    dataset.config(dump_all=True, recursive=True)
    prediction_dates = pd.DatetimeIndex(original.index.get_level_values('datetime'))
    _require(prediction_dates.is_unique and prediction_dates.is_monotonic_increasing, 'prediction dates differ')
    predictions = {(family, seed): _prediction_dates(models[f'{family}_{seed}'].predict(dataset)) for family in ARMS for seed in SEEDS}
    for values in predictions.values():
        pd.testing.assert_index_equal(pd.DatetimeIndex(values.index), prediction_dates, check_names=False)
    ensembles = {family: _ensemble(predictions, family, SEEDS) for family in ARMS}
    a = sources['account_inputs']; frame = a['inputs/frame'].reset_index(drop=True).copy(); market = a['inputs/market'].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[DATES], market[DATES])
    points, schedule = a['inputs/raw_points'], a['inputs/schedule']; strong = sources['strong_native']
    pd.testing.assert_frame_equal(points, strong['inputs/raw_points'])
    _require(plan['deferred_evaluation_sources']['account_inputs']['artifact_sha256']['inputs/schedule'] ==
             plan['deferred_evaluation_sources']['strong_native']['artifact_sha256']['inputs/schedule'], 'corporate schedule identity differs')
    _require(len(frame) == 944 and len(points) == 3778, 'raw account calendar/grid differs')
    dates = pd.DatetimeIndex(frame.signal_date)
    timing = original['meta'].copy(); timing.index = prediction_dates
    decision = timing.reindex(dates).decision_at
    _require(decision.notna().all(), 'non-executable prediction tail entered account')
    pd.testing.assert_index_equal(pd.DatetimeIndex(decision.dt.tz_localize(None).dt.normalize()),
                                  pd.DatetimeIndex(frame.trade_date), check_names=False)
    _require(len(prediction_dates.difference(dates)) == 2, 'two original non-account prediction tails must remain')
    for source_plan in (a['inputs/plan'], strong['inputs/plan']):
        account = source_plan.get('accounts', source_plan)
        _require(source_plan['tariff'] == plan['tariff'] and account['initial_cash'] == 100000
                 and account['slippage_bps'] == [0, 2, 5], 'reference cost/capital contract differs')
    references = {}
    for item in plan['reference_books']:
        book = sources[item['source_role']][item['artifact']]
        pd.testing.assert_frame_equal(book['authority'][DATES].reset_index(drop=True), frame[DATES])
        _require(book['audit']['status'] == 'passed' and pd.DatetimeIndex(book['points'].stamp).equals(points.index), 'reference native/grid differs')
        np.testing.assert_allclose(book['points'].iloc[0].nav, 100000, rtol=0, atol=1e-8)
        references[(item['family'], item['cost'])] = book
    _require(len(references) == 15, 'all fifteen sealed references required')
    returns = market.set_index('signal_date')[['soxl_return', 'soxx_return']]
    base = prepared['inputs/original_models']['models/base_outer_71']
    targets = {family: _smooth_weights(values.reindex(dates), spec=base.spec, candidate=base.candidate, asset_returns=returns)
               for family, values in ensembles.items()}
    info = information_metrics(predictions, ensembles, dataset, as_of=plan['accounts']['terminal'])
    # Exact frozen prediction date axis, including leading signal and terminal
    # tail dates. This is no reconstructed weekday/TOM/source-price calendar.
    calendar = pd.DataFrame({'trade_date': prediction_dates})
    expected = pd.DatetimeIndex(calendar.trade_date)
    account_days = pd.DatetimeIndex(frame.trade_date)
    pd.testing.assert_index_equal(expected[(expected >= account_days[0]) & (expected <= account_days[-1])], account_days, check_names=False)
    e = {'plan': plan, 'frame': frame, 'market': market, 'points': points, 'schedule': schedule, 'guard': a['inputs/guard'],
        'controller': a['inputs/controller'], 'selection': a['inputs/selection'], 'recovery_spec': a['inputs/recovery_spec'],
        'action_spec': a['inputs/action_spec'], 'references': references, 'daily_calendar_frame': calendar}
    return e, dataset, predictions, ensembles, targets, info, audits


def _monthly_metrics(books, calendar):
    """All fixed native account months, Recorder-only and without new replay/CI."""
    from src.analysis.stage1_domain_heads import _period_coverage
    from src.analysis.integrated_allocation import paired_metrics
    rows = []
    for (family, cost), book in books.items():
        path = book['authority']; benchmark = books[('SOXX', cost)]['authority']
        labels, coverage = _period_coverage(pd.DatetimeIndex(path.trade_date),
            pd.DatetimeIndex(calendar.trade_date), 'M')
        for period in labels.unique():
            mask = labels == period
            rows.append({'family': family, 'slippage_bps': cost, 'kind': 'month', 'period': str(period),
                         **coverage[period], **paired_metrics(path.loc[mask], benchmark.loc[mask])})
    return pd.DataFrame(rows)


def run(*, root, card):
    import qlib
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    root, card = Path(root).resolve(), Path(card).resolve(); plan = json.loads((card / 'plan.json').read_text())
    approval = _stage_approval(root, card, plan, 'fit'); state_path = card / 'run_state.json'
    _require(not state_path.exists(), 'formal attempt exists; no automatic repeat')
    state = {'plan_sha256': approval['plan_sha256'], 'fit_attempts': [], 'fits_done': [], 'path_attempts': [], 'paths_done': [],
             'native_attempts': [], 'native_done': [], 'model_sha256': {}, 'phase': 'loading_prepared_training'}
    atomic_json(state, state_path)
    try:
        qlib.init(provider_uri=str(root / plan['provider_and_native_workflow']['init_provider_uri']), region='us', kernels=1)
        prepared, prefit_audit = _read(root, _own_preflight_source(card))
        audit_training_residency(prepared['inputs/primary_dataset'], expected_rows=3014, expected_columns=87, maximum_signal_date='2022-12-21', role='paired_primary')
        audit_training_residency(prepared['inputs/auxiliary_dataset'], expected_rows=20741, expected_columns=20, maximum_signal_date='2022-12-22', role='auxiliary')
    except BaseException as error:
        state.update(failed=True, failed_phase=state['phase'], exception_type=type(error).__name__, exception=str(error))
        atomic_json(state, state_path)
        raise
    tracking = card / 'qlib_recorder'; training = []
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder(); state.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id), phase='training'); atomic_json(state, state_path)
        artifacts = tracking / str(rec.experiment_id) / rec.id / 'artifacts'
        rec.save_objects(artifact_path='inputs', plan=plan, approval=approval, preflight_audit=prefit_audit,
            primary_dataset=prepared['inputs/primary_dataset'], auxiliary_dataset=prepared['inputs/auxiliary_dataset'],
            target_scaling=prepared['inputs/target_scaling'])
        rec.save_objects(artifact_path='implementation', code={name: (root / name).read_text() for name in CODE_FILES})
        for family, seed in plan['training']['fit_order']:
            key = f'{family}_{seed}'; _claim(state, state_path, 'fit', key, 6)
            old = prepared['inputs/original_models'][f'models/base_outer_{seed}']
            model = _new_model(old, ARMS[family], prepared['inputs/target_scaling'], root)
            try:
                model.fit(prepared['inputs/primary_dataset'], auxiliary_dataset=prepared['inputs/auxiliary_dataset'])
                for name in ('training_rows', 'auxiliary_rows', 'last_mature_label_at', 'epochs', 'checkpoint_policy'):
                    _require(model.audit[name] == old.audit[name], 'original training contract differs: ' + name)
                _require(model.primary_scaler == old.primary_scaler and model.common_scaler == old.common_scaler, 'original scalers drifted')
                rec.save_objects(artifact_path='models', **{key: model})
                rec.save_objects(artifact_path='training_trace', **{key: model.training_trace})
                rec.save_objects(artifact_path='optimizer_state', **{key: model.optimizer_state_dict})
                state['model_sha256'][key] = sha(artifacts / 'models' / key); state['fits_done'].append(key); atomic_json(state, state_path)
                training.append({'family': family, 'seed': seed, **model.audit})
                rec.log_metrics(completed_fits=len(state['fits_done'])); print('completed fit', key, flush=True)
            except BaseException as error:
                failed = {'exception_type': type(error).__name__, 'exception': str(error),
                    'audit': getattr(model, 'audit', None), 'training_trace': getattr(model, 'training_trace', []),
                    'completed_step_count': getattr(model, 'completed_step_count', None),
                    'optimizer_state': getattr(model, 'optimizer_state_dict', None), 'optimizer_state_reconstructed': False}
                if getattr(model, 'network', None) is not None:
                    failed['network_state'] = {k: v.detach().clone() for k, v in model.network.state_dict().items()}
                rec.save_objects(artifact_path='failed_fits', **{key: failed})
                state.update(failed=True, failed_phase='training', exception_type=type(error).__name__, exception=str(error)); atomic_json(state, state_path)
                raise
        models = _model_barrier(rec, artifacts, state, plan)
        state.update(phase='models_fixed_before_evaluation', all_six_models_fixed=True, evaluation_unlocked=True); atomic_json(state, state_path)
        rec.save_objects(artifact_path='inputs', evaluation_unlock={'model_sha256': state['model_sha256'],
            'evaluation_objects_read_before_barrier': 0, 'fit_order': plan['training']['fit_order']})
    # qlib.init/source reads/native validator stay outside global R.start;
    # the same explicit Recorder is resumed and closed around all later work.
    rec.start_run()
    try:
        state['phase'] = 'evaluation'; atomic_json(state, state_path)
        e, dataset, predictions, ensembles, targets, info, audits = _evaluate(root, plan, models, prepared)
        rec.save_objects(artifact_path='evaluation', dataset=dataset, prepared=e, source_audits=audits)
        rec.save_objects(artifact_path='predictions', **{f'{family}_{seed}': value for (family, seed), value in predictions.items()})
        rec.save_objects(artifact_path='analysis', ensembles=ensembles, smoothed_targets=targets, classification=info, training=pd.DataFrame(training))
        from src.workflow.stage1_domain_heads import _provider
        from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
        from src.backtest.native_raw_corporate import native_raw_validate
        from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
        from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
        from src.analysis.stage1_domain_heads import analyze_paired_accounts
        provider_config = {**e, 'plan': {**plan, 'provider_output': EXECUTION_PROVIDER}}
        provider, manifest = _provider(root, provider_config)
        rec.save_objects(artifact_path='inputs', native_provider_manifest=manifest,
            execution_provider_config={'provider_output': EXECUTION_PROVIDER, 'plan_modified': False,
                'source': 'post-barrier frozen raw_points and guard only'})
        recovery = IntradayRecoveryDirectAllocationSpec(root=root, **e['recovery_spec'])
        action = Stage1OpenActionValueSpec(root=root, **e['action_spec'])
        execution = {'initial_cash': 100000, 'tariff': plan['tariff']}; books = dict(e['references'])
        for family in ARMS:
            frame = e['frame'].copy()
            for name in OUTPUTS:
                frame[name] = targets[family][name].to_numpy(copy=True)
            for cost in (0, 2, 5):
                key = f'{family}_{cost}'; state['phase'] = 'path'; _claim(state, state_path, 'path', key, 6)
                path = replay_stage1_raw(frame, point_prices=e['points'], schedule=e['schedule'], plan=execution,
                    controller=e['controller'], recovery_spec=recovery, action_spec=action, selection=e['selection'], slippage_bps=cost)
                rec.save_objects(artifact_path='paths', **{key: path}); state['paths_done'].append(key); atomic_json(state, state_path)
                state['phase'] = 'native'; _claim(state, state_path, 'native', key, 6)
                book = native_raw_validate(path, provider_uri=provider, point_prices=e['points'], schedule=e['schedule'], tariff=plan['tariff'], cost=cost, initial_cash=100000)
                rec.save_objects(artifact_path='native', **{key: book}); state['native_done'].append(key); atomic_json(state, state_path)
                books[(family, cost)] = book; rec.log_metrics(completed_native_accounts=len(state['native_done']))
                print('completed native', key, flush=True)
        state['phase'] = 'analysis'; state['analysis_attempted'] = True; atomic_json(state, state_path)
        results = analyze_paired_accounts(books, e['daily_calendar_frame'], plan, candidate_family=CANDIDATE,
            required_comparators=REQUIRED, expected_families=FAMILIES)
        monthly = _monthly_metrics(books, e['daily_calendar_frame'])
        rec.save_objects(artifact_path='analysis', monthly_metrics=monthly)
        for name, value in results.items():
            if name == 'rolling':
                rec.save_objects(artifact_path='analysis/rolling', **value)
            else:
                rec.save_objects(artifact_path='analysis', **{name: value})
            if name in ('metrics', 'paired_comparison', 'annual_metrics', 'quarterly_metrics', 'rolling_summary', 'uncertainty'):
                _require(len(value) <= 500, 'slim report unexpectedly large')
                value.to_csv(card / (name + '.csv'), index=False)
        info.to_csv(card / 'information_metrics.csv', index=False); pd.DataFrame(training).to_csv(card / 'training_summary.csv', index=False)
        economic = all(value for key, value in results['gates'].items() if key not in ('qualified_independent_forward', 'fresh_fullminute_risk_qualified'))
        summary = {'experiment': NAME, 'plan_sha256': approval['plan_sha256'],
            'decision': 'historical_economic_pass_needs_risk_and_independent_forward' if economic else 'not_promoted_fixed_day_night_supervision',
            'objective_achieved': False, 'promotion_approved': False, 'gates': results['gates'], 'coarse_economic_gates_pass': economic,
            'new_fits': 6, 'new_policy_paths': 6, 'ensemble_families': 2, 'new_native_accounts': 6, 'reused_native_accounts': 15,
            'total_comparison_accounts': 21, 'required_paired_intervals': 15, 'model_sha256': state['model_sha256'],
            'sessions_each': 944, 'native_points_each': 3778, 'recorder_id': rec.id, 'experiment_id': str(rec.experiment_id),
            'recorder_uri': str(tracking), 'recorder_local_dir': str(tracking / str(rec.experiment_id) / rec.id),
            'provider_uri': str(provider), 'as_of': '2026-10-09', 'historical_PIT_qualified': False, 'independent_forward_qualified': False,
            'new_source_HTTP': 0, 'new_market_source_price_queries': 0, 'evaluation_S_Z_labels': 0, 'production_changed': False,
            'account_price_access': 'After six-model barrier: pinned raw_points/market plus derived Qlib execution-provider/native-backtest reads; no new market source or S/Z evaluation prices.',
            'history_status': 'reused_history_not_independent_validation', 'independent_evaluation': 'pending',
            'artifact_policy': {**plan['artifact_policy'], 'workflow_config': 'workflow.yaml'}}
        rec.save_objects(artifact_path='analysis', summary=summary); rec.save_objects(artifact_path='execution', receipt=state)
        rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')))
        rec.end_run(status=Recorder.STATUS_FI)
        summary = _seal(rec, root, card, summary, plan['training_sources']['training_only']['card'])
        _conclusion(card, summary, results['metrics']); state.update(completed=True, phase='completed'); atomic_json(state, state_path)
        return summary
    except BaseException as error:
        state.update(failed=True, failed_phase=state['phase'], exception_type=type(error).__name__, exception=str(error)); atomic_json(state, state_path)
        rec.save_objects(artifact_path='execution', failure=state); rec.end_run(status=Recorder.STATUS_FA)
        raise


def _conclusion(card, summary, metrics):
    lines = ['# 固定 H5 日夜监督：配对实验', '',
        '完成B(S)/C(Z)各3seed固定300epoch，全部6模型封存后才读取评价对象。原3014/20741、39/16输入及H1/H5目标保留；新监督仅追加87列Dataset的独立S/Z组，推理不需新标签。原973参数后加同9参数零head，保留单次全局clip及AdamW，整体监督/优化路径为处理。', '',
        '两固定等权ensemble各自0/2/5费用与内生状态，6独立政策路径/6新native及15旧references；不按训练MSE、分类分数或seed表现选择账户。', '',
        '| 费用bp | C累计收益% | 匹配B累计收益% | SOXX累计收益% | C收盘最大回撤% |',
        '|---:|---:|---:|---:|---:|']
    for cost in (0, 2, 5):
        rows = metrics.loc[metrics.slippage_bps.eq(cost)].set_index('family')
        # Metric labels follow the existing native paired analyzer's schema.
        lines.append(f"| {cost} | {rows.loc[CANDIDATE, 'return_pct']:.3f} | {rows.loc['B_day_night_total', 'return_pct']:.3f} | {rows.loc['SOXX', 'return_pct']:.3f} | {rows.loc[CANDIDATE, 'close_max_drawdown_pct']:.3f} |")
    lines += ['', '结论：`' + summary['decision'] + '`。全部年度/季度及126期滚动和15CI保留；训练MSE仅是不同目标的训练诊断，不是alpha或晋升证据。', '',
        '历史已复用，PIT、独立前向及新fullminute风险资格仍未满足；不自动晋升、不救援head/权重/seed/阈值。下一步为独立结果验收，任何失败归因须另定有界范围。']
    (card / 'conclusion.md').write_text('\n'.join(lines) + '\n')
