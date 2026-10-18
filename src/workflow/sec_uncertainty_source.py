"""B0 only: admitted offline NPORT metadata and one native calendar query."""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.data.sec_uncertainty_metadata import select_b0_metadata

NAME = 'sec_uncertainty_source_20261018'
CODE_FILES = ['src/data/sec_uncertainty_metadata.py', 'src/workflow/sec_uncertainty_source.py',
              'tests/test_sec_uncertainty_metadata.py', 'src/core/recorder_sources.py',
              'src/data/minute_intake.py', 'src/data/nas_market_data.py']
SCOPE = {'source_metadata_objects': 3, 'D_calendar_calls': 1, 'D_features_calls': 0,
         'source_HTTP': 0, 'dictionary_or_body_reads': 0, 'word_counts': 0,
         'features': 0, 'fits': 0, 'accounts': 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root):
    return {name: sha(Path(root) / name) for name in CODE_FILES}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata_dataset(selection, anchor):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    if selection.empty:
        return None, None
    frame = selection.copy()
    frame['conditional_decision_at'] = anchor['conditional_decision_at']
    frame['historical_PIT_qualified'] = False
    frame.index = pd.MultiIndex.from_arrays([[pd.Timestamp(anchor['available_session'])] * len(frame),
                                            frame.candidate_cik], names=['datetime', 'instrument'])
    dataset = DatasetH(handler=DataHandlerLP.from_df(pd.concat({'meta': frame}, axis=1)), segments={'all': (None, None)})
    dataset.config(dump_all=True, recursive=True)
    pd.testing.assert_frame_equal(dataset.prepare('all', col_set='meta'), frame)
    return dataset, frame


def _approval(root, card, plan):
    _require(plan.get('frozen') is True and os.environ.get('SEC_UNCERTAINTY_B0_EXECUTE') == '1', 'B0 execution not enabled')
    digest = sha(card / 'plan.json'); current = code_identity(root)
    identity_path = card / 'implementation_identity.json'; identity = json.loads(identity_path.read_text())
    _require(identity['plan_sha256'] == digest and identity['code_sha256'] == current, 'B0 implementation identity differs')
    review_path = card / 'B0_implementation_review.json'; review = json.loads(review_path.read_text())
    _require(review.get('verdict') == 'APPROVE_B0_IMPLEMENTATION' and review.get('plan_sha256') == digest
             and review.get('code_sha256') == current, 'B0 independent implementation review absent/stale')
    path = card / 'B0_execution_approval.json'; owner = json.loads(path.read_text())
    _require(owner.get('approved') is True and owner.get('phase') == 'B0' and owner.get('plan_sha256') == digest
             and owner.get('implementation_identity_sha256') == sha(identity_path)
             and owner.get('implementation_review_sha256') == sha(review_path) and owner.get('scope') == SCOPE,
             'B0 owner approval absent/stale or scope differs')
    _require(sha(root / plan['frozen_from']['review_path']) == plan['frozen_from']['review_sha256'], 'source design review differs')
    source = plan['reused_metadata_sources']
    _require(sha(root / source['known_basket_contract']['path']) == source['known_basket_contract']['sha256'], 'prior basket contract differs')
    return {'plan_sha256': digest, 'code_sha256': current, 'implementation_identity_sha256': sha(identity_path),
            'implementation_review_sha256': sha(review_path), 'owner_approval_sha256': sha(path)}


def _read(root, source):
    source_card = root / source['card']
    _require(json.loads((source_card / 'nas_binding.json').read_text())['evidence']['sha256'] == source['archive'], 'source archive identity differs')
    return load_verified_recorder_objects(source_card, expected_recorder_id=source['recorder_id'],
        artifacts=list(source['artifacts']), expected_manifest_sha256=source['inventory_sha256'],
        expected_artifact_sha256=source['artifacts'])


def run_b0(*, root, card):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.nas_market_data import archive_nas_recorder
    root, card = Path(root).resolve(), Path(card).resolve(); plan = json.loads((card / 'plan.json').read_text())
    approval = _approval(root, card, plan)
    out = card / 'B0'; out.mkdir(exist_ok=True); attempt = out / 'run_state.json'
    _require(not attempt.exists(), 'B0 attempt already exists; no automatic repeat or reselection')
    state = {'status': 'started', 'plan_sha256': approval['plan_sha256'], **{name: 0 for name in SCOPE}}
    atomic_json(state, attempt)
    try:
        source = plan['reused_metadata_sources']; native = source['native_calendar_metadata']
        calendar_path, manifest_path = root / native['path'], root / native['provider_manifest_path']
        _require(sha(calendar_path) == native['sha256'] and sha(manifest_path) == native['provider_manifest_sha256'], 'native calendar metadata identity differs')
        qlib.init(provider_uri=str(root / native['provider_uri']), region='us', kernels=1,
                  expression_cache=None, dataset_cache=None)
        # Native verified source readers must precede any writable global R.start.
        state['source_metadata_objects'] = 2; atomic_json(state, attempt)
        nport, nport_audit = _read(root, source['nport'])
        state['source_metadata_objects'] = 3; atomic_json(state, attempt)
        mapping, mapping_audit = _read(root, source['issuer_mapping'])
        bounds = plan['phase_B']['B0']['native_calendar']['bounds']
        state['D_calendar_calls'] = 1; atomic_json(state, attempt)
        calendar = pd.DatetimeIndex(D.calendar(start_time=bounds[0], end_time=bounds[1], freq='day'))
        _require(len(calendar) > 0 and calendar.min() >= pd.Timestamp(bounds[0]) and calendar.max() <= pd.Timestamp(bounds[1]), 'calendar query bounds differ')
        result = select_b0_metadata(nport['inputs/holdings'], nport['inputs/holdings_manifest'],
            mapping['analysis/issuer_mapping'], calendar, contract=source['nport'])
        _require(sha(calendar_path) == native['sha256'] and sha(manifest_path) == native['provider_manifest_sha256'], 'calendar bytes changed during access')
        for audit in (nport_audit, mapping_audit):
            for item in audit['artifacts'].values():
                _require(sha(item['local_path']) == item['sha256'], 'source metadata bytes changed during B0')
        dataset, expected = metadata_dataset(result['selection'], result['anchor'])
        tracking = out / 'qlib_recorder'
        with R.start(experiment_name=NAME + '_B0', uri=str(tracking)):
            rec = R.get_recorder(); state.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id)); atomic_json(state, attempt)
            rec.save_objects(artifact_path='inputs', plan=plan, approval=approval, holdings=nport['inputs/holdings'],
                holdings_manifest=nport['inputs/holdings_manifest'], issuer_mapping=mapping['analysis/issuer_mapping'],
                native_calendar=calendar, source_audits={'nport': nport_audit, 'mapping': mapping_audit})
            rec.save_objects(artifact_path='analysis', anchor=result['anchor'], security_ledger=result['security_ledger'],
                candidate_ledger=result['candidate_ledger'], selection=result['selection'], mapping_gaps=result['mapping_gaps'])
            if dataset is not None:
                rec.save_objects(artifact_path='metadata', selection_dataset=dataset)
                pd.testing.assert_frame_equal(rec.load_object('metadata/selection_dataset').prepare('all', col_set='meta'), expected)
            rec.save_objects(artifact_path='implementation', code={name: (root / name).read_text() for name in CODE_FILES})
            summary = {'experiment': NAME + '_B0', 'phase': 'B0', 'status': 'completed_pending_independent_review',
                'decision': result['summary']['selection_status'], **result['summary'],
                'plan_sha256': approval['plan_sha256'], 'recorder_id': rec.id, 'experiment_id': str(rec.experiment_id),
                'recorder_uri': str(tracking), 'recorder_local_dir': str(tracking / str(rec.experiment_id) / rec.id),
                'provider_uri': str(root / native['provider_uri']), 'region': 'us', 'as_of': bounds[1],
                'as_of_meaning': 'Bounded calendar eligibility horizon only; not a source-metadata vintage or first-receipt claim.',
                'calendar_query_bounds': bounds, 'source_object_scope': 'All three admitted metadata objects resident, including later holdings metadata; only frozen anchor enters selection.',
                'anchor': {key: str(value) if isinstance(value, pd.Timestamp) else value for key, value in result['anchor'].items()},
                'selected_provisional_ciks': result['selection'].candidate_cik.tolist(),
                'operation_counts': {name: state[name] for name in SCOPE}, 'source_objects_loaded': 3,
                'filing_accessions_selected': 0, 'historical_PIT_qualified': False, 'first_receipt_qualified': False,
                'source_pilot_ready': False, 'model_eligibility': False, 'B1_status': 'needs_separate_B1_GO_and_declared_identity_binding',
                'A_status': 'needs_dictionary_identity_or_usage_separate_branch', 'C_status': 'not_implemented_not_authorized',
                'native_metadata_direct_prepare_roundtrip': 'passed' if dataset is not None else 'not_created_empty_selection',
                'new_fits': 0, 'new_accounts': 0, 'new_source_HTTP': 0,
                'qlib_docs': plan['qlib_artifact_contract']['docs'],
                'artifact_policy': {'policy': 'slim_card_recorder_first', 'workflow_config': '../workflow.yaml',
                    'recorder_only_artifacts': ['inputs/holdings', 'inputs/holdings_manifest', 'inputs/issuer_mapping',
                        'inputs/native_calendar', 'analysis/security_ledger', 'analysis/candidate_ledger', 'metadata/selection_dataset']}}
            rec.save_objects(artifact_path='analysis', summary=summary)
            artifacts = Path(summary['recorder_local_dir']) / 'artifacts'
            rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),
                            selected_provisional_issuers=len(result['selection']), new_source_HTTP=0)
        atomic_json({'recorder_id': rec.id, 'entries': [{'path': str(p.relative_to(out)), 'sha256': sha(p),
            'bytes': p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]}, out / 'verification_manifest.json')
        transport = json.loads((root / source['nport']['card'] / 'nas_binding.json').read_text())
        atomic_json({'schema_version': 'recorder_evidence_binding_v1', 'authority': 'yanxuan_nas', 'ssh': transport['ssh'],
            'source_id': NAME + '-B0-' + approval['plan_sha256'], 'as_of': bounds[1], 'source_plan': '../plan.json'}, out / 'nas_binding.json')
        summary['nas_evidence'] = archive_nas_recorder(out / 'nas_binding.json', workspace_root=root, card=out,
            recorder_id=rec.id, experiment_id=str(rec.experiment_id))
        atomic_json(summary, out / 'summary.json')
        result['selection'].to_csv(card / 'B0_selection.csv', index=False)
        result['mapping_gaps'].to_csv(card / 'B0_mapping_gaps.csv', index=False)
        entry = {**summary, 'card_experiment': NAME, 'qualification_authority': 'B0/summary.json and B0/verification_manifest.json',
                 'B0_summary': 'B0/summary.json', 'B0_manifest': 'B0/verification_manifest.json',
                 'artifact_policy': {**summary['artifact_policy'], 'workflow_config': 'workflow.yaml'}}
        atomic_json(entry, card / 'summary.json')
        state['status'] = 'completed'; atomic_json(state, attempt)
        (card / 'conclusion.md').write_text('# SEC uncertainty 来源：B0 离线阶段\n\n'
            + '完成三个白名单元数据对象与一次2011–2022原生日历读取；固定首篮子条件资格日2019-11-26 09:29NY。保留每条security及身份缺口，按规范CIK去重后锁定前两个provisional候选。\n\n'
            + '结果：`' + summary['decision'] + '`；候选CIK：' + ', '.join(summary['selected_provisional_ciks']) + '。这不是历史issuer/原10-K资格，更不是全成分、PIT或模型资格。缺失/歧义保留，不能用后期篮子或第三家替换。\n\n'
            + '本轮0 HTTP、0词表/正文、0词频/特征、0价格/标签Dataset、0fit/account。整个旧holdings元数据对象含后期记录和披露市值/权重，已如实保留；只anchor标识进入选择，未用权重/市值选样。\n\n'
            + '待独立B0验收；B1仍需单独GO与诚实声明identity绑定，实际SEC接受性未知。A词表许可/身份独立未决，C未实现。Recorder：`' + rec.id + '`。\n')
        return summary
    except BaseException as error:
        state.update(status='failed', exception_type=type(error).__name__, exception=str(error)); atomic_json(state, attempt)
        raise
