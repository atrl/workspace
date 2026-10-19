"""P1 only: frozen train-only sources and static model/mark qualification."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json

NAME = 'path_area_action_bridge_20261018'
CODE_FILES = ['src/workflow/path_area_source_preflight.py', 'src/data/path_area_source_preflight.py',
    'tests/test_path_area_source_preflight.py', 'src/core/recorder_sources.py',
    'src/data/minute_intake.py', 'src/data/nas_market_data.py', 'src/data/qlib_native.py',
    'src/data/minute_allocation_features.py',
    'src/workflow/preopen_tom_allocation.py', 'src/workflow/pit_constituent_auxiliary_training_allocation.py']


def require(value, message):
    if not value:
        raise ValueError('needs_data: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_identity(root, plan):
    files = list(dict.fromkeys([*CODE_FILES, *(x['path'] for x in plan['static_code_bindings']),
                              *plan['source_clock_identity']]))
    result = {p: sha(Path(root) / p) for p in files}
    import qlib
    package = Path(qlib.__file__).parent
    for name in ['__init__.py', 'data/data.py', 'data/storage/file_storage.py', 'data/dataset/handler.py', 'utils/serial.py']:
        result['installed_qlib/' + name] = sha(package / name)
    return result


def _bytes(path, root, expected=None, role=None):
    path = Path(path)
    row = {'path': str(path.relative_to(root)), 'role': role, 'exists': path.is_file(), 'expected_sha256': expected}
    if row['exists']:
        row.update(sha256=sha(path), bytes=path.stat().st_size)
    row['matches_inherited_identity'] = None if expected is None else row.get('sha256') == expected
    return row


def capture_byte_identity(*, root, card):
    """Authorized opaque-byte metadata capture; never unpickle or decode bins."""
    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / 'plan.json').read_text())
    entries, sources = [], {}
    for name in ['PG_train_only', 'original_OOF_candidates', 'corporate_events_metadata']:
        source = plan['sources'][name]; source_card = root / source['card']
        summary = json.loads((source_card / 'summary.json').read_text())
        manifest = source_card / 'verification_manifest.json'
        entries.extend([_bytes(source_card / 'summary.json', root, source.get('summary_sha256'), name + ':summary'),
                        _bytes(manifest, root, source['inventory_sha256'], name + ':manifest')])
        require(summary['recorder_id'] == source['recorder_id'], 'source Recorder locator differs')
        artifact_map = source.get('artifact_sha256', source['artifacts'])
        require(isinstance(artifact_map, dict), 'exact artifact map missing')
        for artifact, expected in artifact_map.items():
            entries.append(_bytes(Path(summary['recorder_local_dir']) / 'artifacts' / artifact, root, expected, name + ':' + artifact))
        sources[name] = {'card': source['card'], 'recorder_id': source['recorder_id'],
            'recorder_uri': summary['recorder_uri'], 'recorder_local_dir': summary['recorder_local_dir'],
            'manifest_name': 'verification_manifest.json', 'artifacts': artifact_map,
            'retrieval_mode': 'existing_local_verified_Recorder_only', 'object_count': len(artifact_map)}
        if source.get('archive'):
            binding = json.loads((source_card / 'nas_binding.json').read_text())
            require(binding['evidence']['sha256'] == source['archive'], 'declared source archive differs')
            sources[name]['archive_sha256'] = source['archive']
    providers = {}
    for name, freq in [('raw_minute', '1min'), ('daily_calendar', 'day'), ('auxiliary_PIT_provider', 'day')]:
        source = plan['sources'][name]; provider = root / source['provider_uri']
        for filename, expected in [('qlib_store_manifest.json', source.get('manifest_sha256')),
                                   (f'calendars/{freq}.txt', source.get('calendar_sha256')),
                                   ('instruments/all.txt', None), ('qlib.yaml', None)]:
            entries.append(_bytes(provider / filename, root, expected, name + ':' + filename))
        manifest = json.loads((provider / 'qlib_store_manifest.json').read_text())
        if name == 'raw_minute':
            instruments = plan['P1_train_only_source_no_fit']['price_bounds']['symbols']
            fields = plan['P1_train_only_source_no_fit']['price_bounds']['core_fields']
        elif name == 'auxiliary_PIT_provider':
            instruments = source['fixed_instruments']; fields = source['P1_field_allowlist']
            require(manifest['instruments'] == instruments, 'PIT fixed universe differs')
        else:
            instruments, fields = [], []
        bin_paths = []
        for symbol in instruments:
            for field in fields:
                path = provider / 'features' / symbol.lower() / (field.removeprefix('$') + '.' + freq + '.bin')
                entries.append(_bytes(path, root, None, name + ':opaque_bin'))
                bin_paths.append(str(path.relative_to(root)))
        providers[name] = {'provider_uri': source['provider_uri'], 'freq': freq, 'bin_paths': bin_paths,
            'inherited_per_bin_sha256_available': False, 'current_bytes_are_snapshot_origin_proof': False,
            'manifest_source_declaration': manifest.get('source')}
    for item in plan['static_code_bindings']:
        entries.append(_bytes(root / item['path'], root, item['sha256'], 'frozen_code'))
    for path, expected in plan['source_clock_identity'].items():
        entries.append(_bytes(root / path, root, expected, 'source_clock_code'))
    entries.append(_bytes(root / 'src/data/qlib_native.py', root, None, 'native_float32_store_writer'))
    for metadata in [plan['sources']['raw_minute']['source_inventory'], plan['registered_evidence']['original_minute_plan']]:
        entries.append(_bytes(root / metadata['path'], root, metadata['sha256'], 'source_declaration_metadata'))
    authority = plan['exchange_session_authority']
    version = importlib.metadata.version(authority['package'])
    require(version == authority['version'], 'exchange calendar package version differs')
    for module in authority['module_identities'].values():
        entries.append(_bytes(Path(module['path']), root, module['sha256'], 'exchange_schedule_code'))
    result = {'schema_version': 1, 'plan_sha256': sha(card / 'plan.json'), 'mode': 'opaque_hash_size_and_locator_only',
        'sources': sources, 'providers': providers, 'entries': entries,
        'source_object_count': sum(s['object_count'] for s in sources.values()),
        'core_bins': 12, 'PIT_flag_bins': 138, 'exchange_calendars_version': version,
        'missing_paths': [r['path'] for r in entries if not r['exists']],
        'mismatched_inherited_paths': [r['path'] for r in entries if r['matches_inherited_identity'] is False],
        'origin_lineage_gaps': ['Core and PIT manifests do not supply inherited per-bin SHA256; current byte identities do not independently authenticate inherited source contents. P2 admission remains separate.'],
        'objects_unpickled': 0, 'numeric_bin_values_decoded': 0, 'D_calendar_calls': 0, 'D_features_calls': 0,
        'model_forward_area_labels_fits_accounts': 0, 'new_source_HTTP': 0}
    require(result['source_object_count'] == 17, 'source whitelist differs from17')
    atomic_json(result, card / 'source_byte_identity.json')
    return result


def verify_byte_identity(root, frozen):
    for row in frozen['entries']:
        path = Path(root) / row['path']
        if row['exists']:
            require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'frozen bytes changed: ' + row['path'])
        else:
            require(not path.exists(), 'previously missing source appeared without new admission')


def _init(provider):
    import qlib
    qlib.init(provider_uri=str(provider), region='us', kernels=1, expression_cache=None,
              dataset_cache=None, default_disk_cache=0, clear_mem_cache=True)


@contextmanager
def bounded_raw_slices(provider, *, freq, fields, instruments, start, end, calendar=None):
    """Constrain raw FileFeatureStorage slices; observe legitimate internal calendar reads."""
    from qlib.data.storage.file_storage import FileCalendarStorage, FileFeatureStorage
    provider = Path(provider).resolve()
    allowed = {str((provider / 'features' / symbol.lower() / (field.removeprefix('$') + '.' + freq + '.bin')).resolve())
               for symbol in instruments for field in fields}
    original_get = FileFeatureStorage.__getitem__; original_cal = FileCalendarStorage.data
    observed = {'calendar': None if calendar is None else pd.DatetimeIndex(calendar), 'reads': [], 'internal_calendar_reads': []}
    def observe_calendar(storage):
        require(storage.uri.resolve() == (provider / 'calendars' / (freq + '.txt')).resolve(), 'unexpected internal calendar/provider')
        values = original_cal.fget(storage)
        observed['calendar'] = pd.DatetimeIndex(values)
        observed['internal_calendar_reads'].append({'path': str(storage.uri), 'rows': len(values), 'role': 'D.features internal indexing metadata'})
        return values
    def bounded(storage, item):
        path = str(storage.uri.resolve()); cal = observed['calendar']
        require(path in allowed and cal is not None, 'unapproved field or calendar unavailable before numeric decode')
        require(cal.is_unique and cal.is_monotonic_increasing and cal.tz is None, 'invalid native indexing calendar')
        require(isinstance(item, slice) and item.step is None and item.start is not None and item.stop is not None,
                'unbounded numeric slice')
        first = int(cal.searchsorted(pd.Timestamp(start), side='left'))
        last = int(cal.searchsorted(pd.Timestamp(end), side='right')) - 1
        require(first <= item.start <= item.stop - 1 <= last, 'numeric slice exceeds declared train bounds')
        require(path not in {r['path'] for r in observed['reads']}, 'duplicate numeric field slice')
        observed['reads'].append({'path': path, 'start_index': int(item.start), 'end_index': int(item.stop - 1),
            'start': str(cal[item.start]), 'end': str(cal[item.stop - 1])})
        return original_get(storage, item)
    FileCalendarStorage.data = property(observe_calendar)
    FileFeatureStorage.__getitem__ = bounded
    try:
        yield observed
    finally:
        FileFeatureStorage.__getitem__ = original_get
        FileCalendarStorage.data = original_cal


def _native_features(provider, *, freq, instruments, fields, start, end, calendar, state, state_path):
    from qlib.data import D
    require(state['D_features_calls'] < 3, 'native feature budget exhausted')
    state['D_features_calls'] += 1; atomic_json(state, state_path)
    guard = None
    try:
        with bounded_raw_slices(provider, freq=freq, fields=fields, instruments=instruments,
                                start=start, end=end, calendar=calendar) as guard:
            frame = D.features(instruments, fields, start_time=start, end_time=end, freq=freq, disk_cache=0)
    except BaseException:
        state.setdefault('failed_native_read_attempts', []).append({'provider': str(provider), 'freq': freq,
            'start': start, 'end': end, 'raw_field_slices': [] if guard is None else guard['reads'],
            'internal_calendar_reads': [] if guard is None else guard['internal_calendar_reads']})
        atomic_json(state, state_path)
        raise
    require(set(frame.index.names) == {'datetime', 'instrument'}, 'native field index schema differs')
    frame = frame.reorder_levels(['datetime', 'instrument']).sort_index()
    dates = frame.index.get_level_values('datetime')
    require(len(frame) and dates.min() >= pd.Timestamp(start) and dates.max() <= pd.Timestamp(end), 'native value range differs')
    require(all(dtype == np.dtype('float32') for dtype in frame.dtypes), 'raw native dtype differs')
    require(len(guard['reads']) == len(instruments) * len(fields), 'not every admitted raw field was actually read')
    return frame, guard


def _dataset(frame, segment='all'):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.handler import DataHandler
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments={segment: (None, None)})
    dataset.config(dump_all=True, recursive=True)
    pd.testing.assert_frame_equal(pickle.loads(pickle.dumps(dataset)).prepare(segment, col_set=DataHandler.CS_RAW), frame)
    return dataset


def _approval(root, card, plan):
    require(os.environ.get('PATH_AREA_P1_EXECUTE') == '1', 'P1 source execution disabled')
    identity_path = card / 'implementation_identity.json'
    identity = json.loads(identity_path.read_text()); current = code_identity(root, plan)
    require(plan['frozen'] and identity['plan_sha256'] == sha(card / 'plan.json') and identity['code_sha256'] == current,
            'P1 plan/code differs')
    byte_sha = sha(card / 'source_byte_identity.json')
    require(identity['source_byte_identity_sha256'] == byte_sha, 'source byte inventory changed')
    review_path = card / 'implementation_review.json'; review = json.loads(review_path.read_text())
    require(review.get('verdict') == 'APPROVE_P1_IMPLEMENTATION' and review.get('plan_sha256') == identity['plan_sha256']
            and review.get('code_sha256') == current and review.get('source_byte_identity_sha256') == byte_sha,
            'P1 implementation review absent/stale')
    approval_path = card / 'P1_execution_approval.json'; owner = json.loads(approval_path.read_text())
    maximum = {'source_objects': 17, 'D_calendar_calls': 2, 'D_features_calls': 3,
               'new_source_HTTP': 0, 'forward_gradient_area_newlabels_fits_accounts': 0}
    require(owner.get('approved') is True and owner.get('phase') == 'P1'
            and owner.get('plan_sha256') == identity['plan_sha256']
            and owner.get('implementation_identity_sha256') == sha(identity_path)
            and owner.get('implementation_review_sha256') == sha(review_path)
            and owner.get('source_byte_identity_sha256') == byte_sha and owner.get('maximum_counts') == maximum,
            'P1 owner GO missing/stale')
    return {'plan_sha256': identity['plan_sha256'], 'implementation_identity_sha256': sha(identity_path),
        'implementation_review_sha256': sha(review_path), 'owner_approval_sha256': sha(approval_path),
        'source_byte_identity_sha256': byte_sha, 'code_sha256': current}


def _source(root, spec, locator):
    return load_verified_recorder_objects(Path(root) / spec['card'], expected_recorder_id=spec['recorder_id'],
        artifacts=list(locator['artifacts']), expected_artifact_sha256=locator['artifacts'],
        expected_manifest_sha256=spec['inventory_sha256'], manifest_name=locator['manifest_name'])


def _expected_model_spec(root, parent):
    from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationSpec
    keys = ['hidden_units', 'tail_hidden_units', 'epochs', 'learning_rate', 'weight_decay',
        'maximum_soxl_weight', 'minimum_cash_weight', 'primary_tail_loss_weight', 'turnover_loss_weight',
        'training_cost_bps', 'auxiliary_batch_size', 'purge_rows', 'smoothing_span', 'rebalance_threshold']
    spec = PitAuxiliaryAllocationSpec(root=Path(root), experiment=parent['experiment'], train_start=parent['train_start'],
        development_years=tuple(parent['development_years']), outer=tuple(parent['outer_window']),
        **{k: parent[k] for k in keys})
    masks = {**{f: 'minute_core_ready' for f in parent['feature_groups']['core']},
             **{f: 'minute_peer_ready' for f in parent['feature_groups']['peer']}}
    return asdict(spec), masks


def _parent_auxiliary_audit(auxiliary, audit):
    from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES
    require(isinstance(auxiliary, dict) and set(auxiliary) == {'2020', '2021', '2022', 'outer'}, 'unexpected admitted multifold auxiliary keys')
    results = {}
    required = [*COMMON_FEATURES, 'synthetic_2x_crash_label', 'synthetic_2x_upside_label', 'training_weight']
    for key, frame in auxiliary.items():
        require(isinstance(frame, pd.DataFrame), 'auxiliary fold must be DataFrame')
        end = pd.Timestamp(audit['train_ends'][key]); dates = pd.to_datetime(frame['datetime'])
        require(end <= pd.Timestamp('2022-12-30') and (dates.empty or dates.max() <= end),
                'multifold auxiliary resident data exceeds declared bound')
        source = audit['auxiliary_audits'][key]
        source_causal = (pd.Timestamp(source['raw_observation_end']) <= end
            and source.get('union_backfill_used') is False and source.get('post_cutoff_actual_2x_calibration_used') is False
            and source.get('label_construction') == 'synthetic 2x forward labels built after truncating raw observations at training cutoff')
        selected = frame.loc[frame['$pit_available'].gt(.5) & frame['training_eligible']].dropna(subset=required)
        results[key] = {'resident_rows': len(frame), 'last_resident_date': str(dates.max()) if len(frame) else None,
            'fold_train_end': str(end.date()), 'source_raw_observation_end': source['raw_observation_end'],
            'qualified_rows': len(selected), 'original_prelabel_crop_source_audit_pass': bool(source_causal),
            'observations_end_field_is_independent_maturity_proof': False}
    return {'folds': results, 'later_to_early_fold_labels_resident_through_2022': True,
            'not_an_early_fold_only_load': True, 'new_labels_constructed': 0}


def _instrument_spans(path, universe):
    spans = {}
    for line in Path(path).read_text().splitlines():
        symbol, first, last = line.split('\t')
        require(symbol not in spans, 'duplicate native instrument metadata')
        spans[symbol] = (pd.Timestamp(first), pd.Timestamp(last))
    require(set(spans) == set(universe), 'native instrument span universe differs')
    return spans


def run_p1(*, root, card):
    from qlib.data import D
    from qlib.data.dataset.handler import DataHandler
    from qlib.workflow import R
    from src.data.nas_market_data import archive_nas_recorder
    from src.data.path_area_source_preflight import (validate_training_frames, build_fold_views, frame_digest,
        qualify_saved_model, qualify_pit_pool, qualify_minute_coverage, corporate_metadata_qualification)
    from src.workflow.preopen_tom_allocation import audit_training_residency
    root, card = Path(root).resolve(), Path(card).resolve()
    plan = json.loads((card / 'plan.json').read_text()); approval = _approval(root, card, plan)
    frozen = json.loads((card / 'source_byte_identity.json').read_text()); verify_byte_identity(root, frozen)
    state_path = card / 'P1_run_state.json'
    state = {'phase': 'P1', 'status': 'started', 'plan_sha256': approval['plan_sha256'], 'source_object_load_attempted': 0,
        'source_objects_loaded': 0, 'D_calendar_calls': 0, 'D_features_calls': 0, 'new_source_HTTP': 0,
        'model_forward': 0, 'gradients': 0, 'area': 0, 'new_labels': 0, 'fits': 0, 'accounts': 0}
    with state_path.open('x') as stream:
        json.dump(state, stream, indent=2)
    objects = {}; audits = {}; gaps = []; rec = None
    primary = auxiliary = fold_package = coverage = None
    native_reads = {}; pit_frames = {}; pit_results = {}; model_results = []; input_datasets = {}
    corporate = {'P2_qualified': False, 'unknowns': ['corporate source not yet qualified']}
    try:
        _init(root / plan['sources']['daily_calendar']['provider_uri'])
        for name in ['PG_train_only', 'original_OOF_candidates', 'corporate_events_metadata']:
            spec, locator = plan['sources'][name], frozen['sources'][name]
            state['source_object_load_attempted'] += len(locator['artifacts']); atomic_json(state, state_path)
            try:
                objects[name], audits[name] = _source(root, spec, locator)
                state['source_objects_loaded'] += len(locator['artifacts']); atomic_json(state, state_path)
            except (ValueError, OSError) as error:
                gaps.append({'stage': name, 'reason': str(error), 'possible_partial_object_residency': list(locator['artifacts'])})
        state['D_calendar_calls'] += 1; atomic_json(state, state_path)
        daily = pd.DatetimeIndex(D.calendar(freq='day'))
        require(daily.is_unique and daily.is_monotonic_increasing and daily.tz is None, 'invalid native daily calendar')
        original = objects.get('original_OOF_candidates'); pg = objects.get('PG_train_only')
        if pg is not None:
            audits['PG_residency'] = {
                'primary': audit_training_residency(pg['inputs/training_dataset'], expected_rows=3014, expected_columns=85,
                    maximum_signal_date='2022-12-21', role='original_primary'),
                'auxiliary': audit_training_residency(pg['inputs/auxiliary_dataset'], expected_rows=20741, expected_columns=20,
                    maximum_signal_date='2022-12-30', role='outer_auxiliary')}
            primary = pg['inputs/training_dataset'].prepare('train', col_set=DataHandler.CS_RAW)
            auxiliary = pg['inputs/auxiliary_dataset'].prepare('train', col_set=DataHandler.CS_RAW)
            if original is not None:
                parent = original['inputs/plan']; parent_audit = original['inputs/audit']
                require(parent == json.loads((root / plan['registered_evidence']['original_minute_plan']['path']).read_text()), 'parent plan object differs from frozen plan JSON')
                base = parent_audit['original_features']
                audits['training'] = validate_training_frames(primary, auxiliary, base_features=base)
                fold_package = build_fold_views(primary, daily, base_features=base)
                audits['parent_auxiliary'] = _parent_auxiliary_audit(original['inputs/auxiliary'], parent_audit)
                spec, masks = _expected_model_spec(root, parent)
                for year in plan['fold_contract']['reusable_years']:
                    fold = fold_package['folds'][year]; aux_audit = audits['parent_auxiliary']['folds'][str(year)]
                    require(pd.Timestamp(parent_audit['train_ends'][str(year)]) == fold['train_end'], 'parent purged fold end differs')
                    for seed in plan['preserved_contract']['seeds']:
                        key = f'models/base_{year}_{seed}'; model = original[key]
                        result = qualify_saved_model(model, fold, base_seed=seed, base_features=base,
                            expected_spec=spec, feature_masks=masks,
                            auxiliary_rows=aux_audit['qualified_rows'] if aux_audit['original_prelabel_crop_source_audit_pass'] else None)
                        if model.minimum_observations != parent['minimum_observed_minute_training_sessions']:
                            result['static_qualified'] = False; result['failures'].append('minimum_observations differs')
                        model_results.append({'artifact': key, 'year': year, 'base_seed': seed, **result})
                # Views contain original input/timing only; original labels remain
                # in their admitted training authority, never copied to a prediction view.
                view = primary.loc[:, primary.columns.get_level_values(0).isin(['feature', 'common', 'meta'])].copy()
                input_datasets['original_input_timing'] = _dataset(view, 'train')
                for year, fold in fold_package['folds'].items():
                    input_datasets['fold_' + str(year)] = _dataset(fold['train'], 'train')
        if fold_package is not None:
            pit = plan['sources']['auxiliary_PIT_provider']; pit_provider = root / pit['provider_uri']
            spans = _instrument_spans(pit_provider / 'instruments/all.txt', pit['fixed_instruments'])
            for year in plan['fold_contract']['fixed_missing_years']:
                end = str(fold_package['folds'][year]['train_end'].date())
                _init(pit_provider)
                try:
                    flags, read = _native_features(pit_provider, freq='day', instruments=pit['fixed_instruments'],
                        fields=pit['P1_field_allowlist'], start=pit['fixed_start'], end=end, calendar=None,
                        state=state, state_path=state_path)
                    pit_frames[year] = flags; native_reads['PIT_' + str(year)] = read
                    pit_results[year] = qualify_pit_pool(flags, calendar=read['calendar'], instruments=pit['fixed_instruments'],
                        start=pit['fixed_start'], end=end, source_qualified=True, instrument_spans=spans)
                    if pit_results[year]['status'] == 'proven_empty':
                        input_datasets['empty_aux_' + str(year)] = _dataset(auxiliary.iloc[:0].copy(), 'train')
                except (ValueError, OSError) as error:
                    pit_results[year] = {'status': 'unknown', 'reason': str(error), 'empty_auxiliary_authorized': False}
            bounds = plan['P1_train_only_source_no_fit']['price_bounds']; provider = root / plan['sources']['raw_minute']['provider_uri']
            _init(provider)
            state['D_calendar_calls'] += 1; atomic_json(state, state_path)
            core_calendar = pd.DatetimeIndex(D.calendar(freq='1min'))
            try:
                bars, read = _native_features(provider, freq='1min', instruments=bounds['symbols'], fields=bounds['core_fields'],
                    start=bounds['start'], end=bounds['end'], calendar=core_calendar, state=state, state_path=state_path)
                native_reads['core'] = read
                import exchange_calendars as xc
                authority = plan['exchange_session_authority']
                schedule = xc.get_calendar(authority['calendar'], start=authority['schedule_start'], end=authority['schedule_end']).schedule
                schedule_days = pd.DatetimeIndex(schedule.index).tz_localize(None)
                native_days = daily[(daily >= pd.Timestamp(authority['schedule_start'])) & (daily <= pd.Timestamp(authority['schedule_end']))]
                require(schedule_days.equals(native_days), 'frozen exchange schedule and native sessions conflict')
                coverage = qualify_minute_coverage(bars, fold_package['timing'], schedule, source_clock_qualified=True)
                input_datasets['bounded_core_minutes'] = _dataset(pd.concat({'feature': bars}, axis=1))
                audits['exchange_schedule'] = {'package': authority['package'], 'version': authority['version'],
                    'calendar': authority['calendar'], 'native_session_equality': True, 'schedule': schedule}
                del bars
            except (ValueError, OSError) as error:
                gaps.append({'stage': 'core_mark_coverage', 'reason': str(error)})
            events = objects.get('corporate_events_metadata')
            if events is not None:
                corporate = corporate_metadata_qualification(events['data/cash_events.pkl'], events['data/split_events.pkl'],
                    start=bounds['start'][:10], end=bounds['end'][:10], source_qualified=True, sessions=daily)
        if 'training' in audits:
            require(frame_digest(primary) == audits['training']['primary_sha256']
                    and frame_digest(auxiliary) == audits['training']['auxiliary_sha256'], 'original PG training values mutated')
        verify_byte_identity(root, frozen)
        # All source reads/provider switches completed before the new writable run.
        tracking = card / 'P1' / 'qlib_recorder'; result_card = card / 'P1'; result_card.mkdir(exist_ok=True)
        with R.start(experiment_name=NAME + '_P1', uri=str(tracking)):
            rec = R.get_recorder(); state.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id)); atomic_json(state, state_path)
            rec.save_objects(artifact_path='inputs', plan=plan, approval=approval, source_byte_identity=frozen,
                admitted_source_objects=objects, source_read_audits=audits, daily_calendar=daily,
                bounded_PIT_flags=pit_frames, datasets=input_datasets)
            rec.save_objects(artifact_path='analysis', model_eligibility=model_results, fold_package=fold_package,
                membership_qualification=pit_results, minute_coverage=coverage, corporate_qualification=corporate,
                native_read_audits=native_reads, gaps=gaps, execution_state=dict(state))
            rec.save_objects(artifact_path='implementation', code={p: (root / p).read_text() for p in CODE_FILES if p.endswith('.py')})
            summary = {'experiment': NAME + '_P1', 'phase': 'P1', 'status': 'completed_pending_independent_review',
                'decision': 'source_static_qualification_only_not_P2_authorization',
                'recorder_id': rec.id, 'experiment_id': str(rec.experiment_id), 'recorder_uri': str(tracking),
                'recorder_local_dir': str(tracking / str(rec.experiment_id) / rec.id), 'plan_sha256': approval['plan_sha256'],
                'source_byte_identity_sha256': approval['source_byte_identity_sha256'],
                'provider_uri': {k: plan['sources'][k]['provider_uri'] for k in ['daily_calendar', 'raw_minute', 'auxiliary_PIT_provider']},
                'region': 'us', 'training_observation_end': '2022-12-30', 'as_of': '2026-10-09',
                'counts': {k: v for k, v in state.items() if k not in ['status', 'recorder_id', 'experiment_id', 'plan_sha256']},
                'primary_rows': None if primary is None else len(primary), 'auxiliary_rows': None if auxiliary is None else len(auxiliary),
                'saved_models_checked': len(model_results), 'saved_models_static_qualified': sum(bool(r['static_qualified']) for r in model_results),
                'membership_status': {str(y): r['status'] for y, r in pit_results.items()},
                'coverage_rows': None if coverage is None else len(coverage),
                'prefix_valid_rows': None if coverage is None else int(coverage['prefix_mask'].sum()),
                'source_gaps': gaps, 'origin_lineage_gaps': frozen['origin_lineage_gaps'],
                'multifold_auxiliary_residency': 'All admitted2020/2021/2022/outer frames may be resident; not earlier-fold-only.',
                'corporate_residency': 'Whole two event-metadata objects may include later dates; only bounded event view is qualified.',
                'intraday_residency': 'Bounded2018-10-12..2022-12-30 includes afternoons; prefix mask uses only09:31..09:45.',
                'P2_authorized': False, 'P3_authorized': False, 'historical_PIT_qualified': False,
                'artifact_policy': {'policy': 'slim_card_recorder_first', 'workflow_config': '../workflow.yaml',
                    'recorder_only_artifacts': ['inputs/admitted_source_objects', 'inputs/datasets', 'inputs/bounded_PIT_flags',
                        'analysis/fold_package', 'analysis/minute_coverage', 'analysis/native_read_audits']}}
            rec.save_objects(artifact_path='analysis', summary=summary)
            artifacts = Path(summary['recorder_local_dir']) / 'artifacts'
            rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),
                            saved_models_static_qualified=summary['saved_models_static_qualified'])
        atomic_json({'recorder_id': rec.id, 'entries': [{'path': str(p.relative_to(result_card)), 'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(artifacts.rglob('*')) if p.is_file()]}, result_card / 'verification_manifest.json')
        transport = json.loads((root / plan['sources']['PG_train_only']['card'] / 'nas_binding.json').read_text())
        atomic_json({'schema_version': 'recorder_evidence_binding_v1', 'authority': 'yanxuan_nas', 'ssh': transport['ssh'],
            'source_id': NAME + '_P1_' + approval['plan_sha256'], 'as_of': summary['as_of']}, result_card / 'nas_binding.json')
        summary['nas_evidence'] = archive_nas_recorder(result_card / 'nas_binding.json', workspace_root=root, card=result_card,
            recorder_id=rec.id, experiment_id=str(rec.experiment_id))
        atomic_json(summary, result_card / 'summary.json')
        atomic_json({**summary, 'artifact_policy': {**summary['artifact_policy'], 'workflow_config': 'workflow.yaml'},
            'qualification_authority': 'P1/summary.json'}, card / 'summary.json')
        state['status'] = 'completed'; atomic_json(state, state_path)
        return summary
    except BaseException as error:
        state.update(status='failed', exception_type=type(error).__name__, exception=str(error)); atomic_json(state, state_path)
        if rec is not None:
            rec.save_objects(artifact_path='execution', failure=state)
            rec.end_run(status='FAILED')
        raise
