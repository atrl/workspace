"""Independent B1 metadata/header pilot; bounded HTTP, never filing bodies."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener
import re

import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.data.sec_uncertainty_accessions import parse_submissions, annual_pair_progress, check_header

CODE_FILES = ['src/data/sec_uncertainty_accessions.py', 'src/workflow/sec_uncertainty_accession_source.py',
              'tests/test_sec_uncertainty_accessions.py', 'src/core/recorder_sources.py',
              'src/data/minute_intake.py', 'src/data/nas_market_data.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _now():
    return datetime.now(timezone.utc).isoformat()


def _require(value, message):
    if not value:
        raise ValueError(message)


def code_identity(root):
    return {name: sha(Path(root) / name) for name in CODE_FILES}


def _is_challenge(body):
    return bool(re.search(br'undeclared automated tool|request rate threshold exceeded|automated access|cf-chl-|challenge-platform|<title>\s*(?:Just a moment|Attention Required)', body[:200000], re.I))


def _record_failure(rec, state, client, request_receipts):
    """Keep the actual failed run and every acquired response, without retry."""
    if rec is None:
        return
    rec.save_objects(artifact_path='execution', failure=dict(state))
    if client is not None:
        rec.save_objects(artifact_path='source', HTTP_state=client.state, request_receipts=request_receipts)
        for number, body in client.raw_responses.items():
            rec.save_objects(artifact_path='source/raw_http', **{f'request_{number:02d}': body})
    rec.end_run(status='FAILED')


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class BoundedSEC:
    """Single process/lock, durable pre-request reservation and no retries."""
    def __init__(self, work, *, ciks, user_agent, opener=None, clock=time.monotonic, sleeper=time.sleep):
        self.work = Path(work); self.work.mkdir(parents=True, exist_ok=True)
        self.path = self.work / 'http_state.json'
        _require(not self.path.exists(), 'HTTP attempt already exists; no implicit retry/resume')
        self.ciks, self.ua = tuple(ciks), user_agent
        self.opener = opener if opener is not None else build_opener(_NoRedirect()).open
        self.clock, self.sleep = clock, sleeper
        self.lock = threading.Lock(); self.last_launch = None
        self.histories = {cik: set() for cik in ciks}; self.headers = {cik: set() for cik in ciks}
        self.raw_responses = {}
        self.state = {'attempts': [], 'host_stops': {}, 'maximum': 10, 'retries': 0, 'redirects_followed': 0}
        atomic_json(self.state, self.path)

    def allow_history(self, cik, catalog):
        self.histories[cik] = {item['name'] for item in catalog}

    def allow_header(self, cik, accession):
        _require(bool(re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession)), 'invalid selected accession')
        self.headers[cik].add(accession)

    def get(self, kind, cik, key=None):
        _require(cik in self.ciks, 'unselected issuer')
        if kind == 'current':
            url = f'https://data.sec.gov/submissions/CIK{cik}.json'
        elif kind == 'history':
            _require(key in self.histories[cik] and '/' not in key and '..' not in key, 'unlisted history URL')
            url = 'https://data.sec.gov/submissions/' + key
        elif kind == 'header':
            _require(key in self.headers[cik], 'unselected header')
            url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{key.replace("-", "")}/{key}.hdr.sgml'
        else:
            raise ValueError('B1 only permits current/history/header requests')
        host = urlparse(url).hostname
        with self.lock:
            if host in self.state['host_stops']:
                return None, {'url': url, 'kind': kind, 'cik': cik, 'no_request': True, 'reason': 'host_stopped'}
            if any(x['url'] == url for x in self.state['attempts']):
                return None, {'url': url, 'kind': kind, 'cik': cik, 'no_request': True, 'reason': 'duplicate_URL_no_retry'}
            counts = self.state['attempts']
            limits = {'current': 2, 'history': 4, 'header': 4}
            _require(len(counts) < 10 and sum(x['kind'] == kind for x in counts) < limits[kind], 'HTTP hard budget exhausted')
            _require(kind != 'history' or sum(x['kind'] == 'history' and x['cik'] == cik for x in counts) < 2,
                     'two-history-per-issuer budget exhausted')
            if self.last_launch is not None:
                self.sleep(max(0., 1.0 - (self.clock() - self.last_launch)))
            self.last_launch = self.clock()
            receipt = {'number': len(counts) + 1, 'kind': kind, 'cik': cik, 'url': url,
                       'started_at': _now(), 'state': 'reserved_before_request', 'user_agent': self.ua}
            counts.append(receipt); atomic_json(self.state, self.path)
            body = b''; status = None; headers = {}; error = None
            try:
                request = Request(url, headers={'User-Agent': self.ua, 'Accept-Encoding': 'identity',
                                               'Accept': 'application/json,text/plain,*/*;q=0.1'})
                try:
                    response = self.opener(request, timeout=30)
                except HTTPError as failure:
                    response = failure
                with response:
                    status = int(response.status if hasattr(response, 'status') else response.code)
                    headers = {name: response.headers.get(name) for name in
                               ('Date', 'Content-Type', 'Content-Encoding', 'Location', 'Retry-After', 'Last-Modified', 'ETag')}
                    body = response.read(16 * 1024 * 1024 + 1)
                    if len(body) > 16 * 1024 * 1024:
                        error = 'response_exceeds_fixed_16MiB_metadata_limit_partial_bytes_retained'
            except Exception as failure:
                error = type(failure).__name__ + ': ' + str(failure)
            self.raw_responses[receipt['number']] = body
            raw_path = self.work / f'{receipt["number"]:02d}_{kind}_{cik}.bin'
            raw_path.write_bytes(body)
            challenge = _is_challenge(body)
            receipt.update(state='response_recorded', status=status, headers=headers, error=error, completed_at=_now(),
                           raw_sha256=sha(raw_path), raw_bytes=len(body), raw_path=str(raw_path), challenge=challenge,
                           redirect_not_followed=bool(status is not None and 300 <= status < 400))
            if status in (403, 429) or challenge:
                self.state['host_stops'][host] = {'status': status, 'challenge': challenge, 'at': _now()}
            atomic_json(self.state, self.path)
            if status != 200 or error or challenge:
                return None, dict(receipt)
            try:
                encoding = headers.get('Content-Encoding')
                if encoding and encoding.lower() not in ('identity', 'gzip'):
                    raise ValueError('unsupported content encoding')
                decoded = gzip.decompress(body) if encoding and encoding.lower() == 'gzip' else body
                _require(len(decoded) <= 16 * 1024 * 1024, 'decoded metadata limit exceeded')
                if _is_challenge(decoded):
                    receipt.update(challenge=True, challenge_detection='decoded_response')
                    self.state['host_stops'][host] = {'status': status, 'challenge': True, 'at': _now()}
                    atomic_json(self.state, self.path)
                    return None, dict(receipt)
                return decoded, dict(receipt)
            except Exception as failure:
                receipt['decode_error'] = str(failure); atomic_json(self.state, self.path)
                return None, dict(receipt)


def _approval(root, parent, card, plan):
    _require(os.environ.get('SEC_UNCERTAINTY_B1_EXECUTE') == '1', 'B1 source execution disabled')
    digest = sha(parent / 'B1_plan.json'); identity = json.loads((card / 'implementation_identity.json').read_text())
    current = code_identity(root)
    _require(plan.get('frozen') and identity['plan_sha256'] == digest and identity['code_sha256'] == current, 'B1 code/plan identity differs')
    review_path = card / 'implementation_review.json'; review = json.loads(review_path.read_text())
    _require(review.get('verdict') == 'APPROVE_B1_IMPLEMENTATION' and review.get('plan_sha256') == digest
             and review.get('code_sha256') == current, 'B1 implementation review absent/stale')
    approval_path = card / 'execution_approval.json'; owner = json.loads(approval_path.read_text())
    _require(owner.get('approved') is True and owner.get('phase') == 'B1' and owner.get('plan_sha256') == digest
             and owner.get('implementation_identity_sha256') == sha(card / 'implementation_identity.json')
             and owner.get('implementation_review_sha256') == sha(review_path)
             and owner.get('budget') == plan['budget'], 'B1 separate owner GO absent/stale')
    for name, field in [('plan.json', 'parent_plan_sha256'), ('source_design_review.json', 'parent_source_design_review_sha256'),
                        ('B0_result_review.json', 'B0_result_review_sha256'), ('B0_readback_verification.json', 'B0_readback_sha256'),
                        ('transport_identity_review.json', 'transport_identity_review_sha256')]:
        _require(sha(parent / name) == plan[field], 'preceding qualification identity differs: ' + name)
    return {'plan_sha256': digest, 'code_sha256': current, 'owner_approval_sha256': sha(approval_path),
            'implementation_review_sha256': sha(review_path), 'implementation_identity_sha256': sha(card / 'implementation_identity.json')}


def run_b1(*, root, card):
    import qlib
    from qlib.workflow import R
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from src.data.nas_market_data import archive_nas_recorder
    root, card = Path(root).resolve(), Path(card).resolve(); parent = card.parent
    plan = json.loads((parent / 'B1_plan.json').read_text()); approval = _approval(root, parent, card, plan)
    state_path = card / 'run_state.json'
    state = {'status': 'started', 'phase': 'B1', 'plan_sha256': approval['plan_sha256'], 'source_objects_loaded': 0,
             'source_HTTP': 0, 'D_calendar_calls': 0, 'D_features_calls': 0, 'filing_body_requests': 0,
             'word_counts': 0, 'features': 0, 'fits': 0, 'accounts': 0}
    # Exclusive creation prevents a second process silently consuming the same attempt.
    with state_path.open('x') as stream:
        json.dump(state, stream, indent=2)
    rec = None; client = None; request_receipts = []
    try:
        prior = json.loads((parent / 'plan.json').read_text())
        provider = root / prior['reused_metadata_sources']['native_calendar_metadata']['provider_uri']
        qlib.init(provider_uri=str(provider), region='us', kernels=1, expression_cache=None, dataset_cache=None)
        source = plan['source']; source_card = root / source['card']
        _require(sha(source_card / 'summary.json') == source['summary_sha256']
                 and json.loads((source_card / 'nas_binding.json').read_text())['evidence']['sha256'] == source['archive_sha256'], 'sealed B0 source differs')
        objects, source_audit = load_verified_recorder_objects(source_card, expected_recorder_id=source['recorder_id'],
            artifacts=list(source['artifacts']), expected_manifest_sha256=source['manifest_sha256'],
            expected_artifact_sha256=source['artifacts'])
        state['source_objects_loaded'] = 4; atomic_json(state, state_path)
        selection, anchor, security, calendar = [objects[name] for name in
            ('analysis/selection', 'analysis/anchor', 'analysis/security_ledger', 'inputs/native_calendar')]
        _require(selection.candidate_cik.tolist() == plan['selected_provisional_ciks'] and anchor['available_session'] == '2019-11-26', 'locked sample/anchor differs')
        work = Path(tempfile.mkdtemp(prefix='sec_uncertainty_B1_')); state['temporary_raw_root'] = str(work); atomic_json(state, state_path)
        client = BoundedSEC(work, ciks=plan['selected_provisional_ciks'], user_agent=plan['user_agent'])
        tracking = card / 'qlib_recorder'
        with R.start(experiment_name=plan['experiment'], uri=str(tracking)):
            rec = R.get_recorder(); state.update(recorder_id=rec.id, experiment_id=str(rec.experiment_id)); atomic_json(state, state_path)
            rec.save_objects(artifact_path='inputs', plan=plan, approval=approval, B0_objects=objects, B0_source_audit=source_audit)
            rec.save_objects(artifact_path='implementation', code={name: (root / name).read_text() for name in CODE_FILES})
            parsed = {}; progress = {}
            def fetch(kind, cik, key=None):
                body, receipt = client.get(kind, cik, key)
                request_receipts.append(receipt)
                state['source_HTTP'] = len(client.state['attempts']); atomic_json(state, state_path)
                if not receipt.get('no_request'):
                    rec.save_objects(artifact_path='source/raw_http', **{f'request_{receipt["number"]:02d}': Path(receipt['raw_path']).read_bytes()})
                rec.save_objects(artifact_path='source', request_receipts=request_receipts, HTTP_state=client.state)
                return body, receipt
            # Freeze both CIKs before any request. Failure never changes this list.
            for cik in plan['selected_provisional_ciks']:
                body, receipt = fetch('current', cik)
                if body is None:
                    progress[cik] = {'status': 'gap', 'reason': 'current_submissions_unavailable', 'receipt': receipt}; continue
                try:
                    parsed[cik] = parse_submissions(body, cik=cik, role='current')
                    client.allow_history(cik, parsed[cik]['catalog'])
                except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
                    progress[cik] = {'status': 'gap', 'reason': str(error)}
            for cik, current in parsed.items():
                histories = {}
                for _ in range(3):
                    check = annual_pair_progress(current, histories)
                    if check['status'] != 'need_history':
                        break
                    item = check['next']
                    body, receipt = fetch('history', cik, item['name'])
                    if body is None:
                        check = {'status': 'gap', 'reason': 'declared_history_unavailable', 'receipt': receipt}; break
                    try:
                        histories[item['name']] = parse_submissions(body, cik=cik, role='history', declared=item)['page']
                    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
                        check = {'status': 'gap', 'reason': str(error)}; break
                if check['status'] == 'need_history':
                    check = {'status': 'gap', 'reason': 'page_limit_cannot_certify_pair'}
                progress[cik] = check
                rec.save_objects(artifact_path='source/parsed_metadata', **{cik: {'current': current, 'histories': histories, 'selection_progress': check}})
            ledger = []
            for cik in plan['selected_provisional_ciks']:
                check = progress[cik]
                names = security.loc[security.candidate_cik.eq(cik), 'name'].tolist()
                for role in ('previous', 'subsequent'):
                    if check['status'] != 'metadata_pair_selected_pending_headers':
                        ledger.append({'cik': cik, 'role': role, 'accession': None, 'header_qualified': False,
                            'metadata_version_status': 'unknown_no_pair', 'body_first_vintage_status': 'unknown_body_not_requested',
                            'gap_reasons': [check.get('reason', 'metadata_pair_not_qualified')]})
                        continue
                    filing = check['pair'][role]
                    client.allow_header(cik, filing['accessionNumber'])
                    body, receipt = fetch('header', cik, filing['accessionNumber'])
                    if body is None:
                        result = {'cik': cik, 'role': role, 'accession': filing['accessionNumber'], 'header_qualified': False,
                            'metadata_version_status': 'unknown_header_unavailable', 'body_first_vintage_status': 'unknown_body_not_requested',
                            'gap_reasons': ['selected_header_unavailable']}
                    else:
                        result = check_header(body, cik=cik, filing=filing, anchor_names=names, calendar=calendar, role=role)
                    result['header_receipt'] = receipt; ledger.append(result)
            # Header-disproved ordering/cutoff closes these slots, never selects
            # another annual accession or requests another page/header.
            for cik in plan['selected_provisional_ciks']:
                left, right = [x for x in ledger if x['cik'] == cik]
                if left.get('header_qualified') and right.get('header_qualified'):
                    old, new = left['accepted_header_literal_EST'], right['accepted_header_literal_EST']
                    if not (old < new and pd.Timestamp('2019-11-26', tz='America/New_York') <= new < pd.Timestamp('2023-01-01', tz='America/New_York')):
                        for item in (left, right):
                            item['header_qualified'] = False; item['gap_reasons'].append('header_disproves_provisional_pair_order_or_cutoff_no_reselection')
            qualified = len(ledger) == 4 and all(x['header_qualified'] for x in ledger)
            allowed_bodies = []
            if qualified:
                for row in ledger:
                    acc = row['accession']; allowed_bodies.append({'cik': row['cik'], 'accession': acc,
                        'url': f'https://www.sec.gov/Archives/edgar/data/{int(row["cik"])}/{acc.replace("-", "")}/{acc}.txt',
                        'expected_primary_document': row['primary_document_metadata'], 'header_sha256': row['header_receipt']['raw_sha256']})
            table = pd.DataFrame([{key: json.dumps(_safe(value), ensure_ascii=False) if isinstance(value, (dict, list)) else value
                                   for key, value in row.items()} for row in ledger])
            table.index = pd.MultiIndex.from_arrays([[pd.Timestamp(anchor['available_session'])] * 4,
                [row['cik'] + '_' + row['role'] for row in ledger]], names=['datetime', 'instrument'])
            dataset = DatasetH(handler=DataHandlerLP.from_df(pd.concat({'meta': table}, axis=1)), segments={'all': (None, None)})
            dataset.config(dump_all=True, recursive=True)
            rec.save_objects(artifact_path='analysis', ledger=ledger, selection_progress=progress, qualified_C_URL_manifest=allowed_bodies,
                             metadata_dataset=dataset, request_receipts=request_receipts)
            pd.testing.assert_frame_equal(rec.load_object('analysis/metadata_dataset').prepare('all', col_set='meta'), table)
            summary = {'experiment': plan['experiment'], 'phase': 'B1', 'status': 'completed_pending_independent_review',
                'decision': 'conditional_four_header_metadata_ready_for_C_review' if qualified else 'needs_source_metadata_clock_or_identity_qualification',
                'four_original_headers_qualified': qualified, 'selected_provisional_ciks': plan['selected_provisional_ciks'],
                'fixed_ledger_slots': 4, 'selected_accessions': [x.get('accession') for x in ledger],
                'gap_rows': sum(not x['header_qualified'] for x in ledger), 'source_HTTP_requests': len(client.state['attempts']),
                'HTTP_by_kind': {kind: sum(x['kind'] == kind for x in client.state['attempts']) for kind in ('current', 'history', 'header')},
                'stopped_hosts': client.state['host_stops'], 'retries': 0, 'redirects_followed': 0,
                'source_objects_loaded': 4, 'new_D_calendar_calls': 0, 'D_features_calls': 0, 'filing_body_requests': 0,
                'dictionary_reads': 0, 'word_counts': 0, 'features': 0, 'new_fits': 0, 'new_accounts': 0,
                'body_first_vintage_qualified': False, 'first_receipt_qualified': False, 'historical_PIT_qualified': False,
                'C_execution_authorized': False, 'current_metadata_is_first_publication_evidence': False,
                'native_metadata_direct_prepare_roundtrip': 'passed', 'plan_sha256': approval['plan_sha256'],
                'native_metadata_index_semantics': 'B0 anchor is a pilot grouping index only, never filing availability; qualification uses separate conditional_available_at and clock columns.',
                'recorder_id': rec.id, 'experiment_id': str(rec.experiment_id), 'recorder_uri': str(tracking),
                'recorder_local_dir': str(tracking / str(rec.experiment_id) / rec.id), 'provider_uri': str(provider),
                'region': 'us', 'as_of': _now()[:10], 'historical_pair_cutoff': '2023-01-01',
                'artifact_policy': {'policy': 'slim_card_recorder_first', 'workflow_config': 'workflow.yaml',
                    'recorder_only_artifacts': ['source/raw_http', 'source/parsed_metadata', 'inputs/B0_objects', 'analysis/metadata_dataset']}}
            rec.save_objects(artifact_path='analysis', summary=summary)
            artifacts = Path(summary['recorder_local_dir']) / 'artifacts'
            rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),
                            source_HTTP_requests=len(client.state['attempts']), qualified_headers=sum(x['header_qualified'] for x in ledger))
        atomic_json({'recorder_id': rec.id, 'entries': [{'path': str(p.relative_to(card)), 'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(artifacts.rglob('*')) if p.is_file()]}, card / 'verification_manifest.json')
        transport = json.loads((source_card / 'nas_binding.json').read_text())
        atomic_json({'schema_version': 'recorder_evidence_binding_v1', 'authority': 'yanxuan_nas', 'ssh': transport['ssh'],
            'source_id': plan['experiment'] + '-' + approval['plan_sha256'], 'as_of': summary['as_of']}, card / 'nas_binding.json')
        summary['nas_evidence'] = archive_nas_recorder(card / 'nas_binding.json', workspace_root=root, card=card,
            recorder_id=rec.id, experiment_id=str(rec.experiment_id))
        atomic_json(_safe(summary), card / 'summary.json')
        atomic_json(_safe(ledger), card / 'selection_ledger.json')
        atomic_json(_safe(request_receipts), card / 'request_receipts.json')
        state.update(status='completed', source_HTTP=len(client.state['attempts'])); atomic_json(state, state_path)
        (card / 'conclusion.md').write_text('# SEC uncertainty B1 元数据/header结果\n\n'
            + summary['decision'] + '。两固定CIK/四ledger slots保留，不换issuer或年度。source HTTP次数：'
            + str(summary['source_HTTP_requests']) + '（NAS操作性归档另计），失败/停host亦保留证据。\n\n'
            + '0正文/词表/词频/特征/价格标签/fit/account。公开钟/历史identity/PAC元数据风险保留；PAC不被解释为正文修改，body first vintage仍unknown。C仍需独立验收及单独GO，整体来源或模型资格没有自动取得。\n')
        return summary
    except BaseException as error:
        state.update(status='failed', exception_type=type(error).__name__, exception=str(error),
                     source_HTTP=len(client.state['attempts']) if client is not None else 0)
        atomic_json(state, state_path)
        try:
            _record_failure(rec, state, client, request_receipts)
        except BaseException as persistence_error:
            state['failure_evidence_persistence_error'] = type(persistence_error).__name__ + ': ' + str(persistence_error)
            atomic_json(state, state_path)
        raise
