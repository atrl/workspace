"""One focused synthetic contract; no SEC data or network request is made."""
import io
import json
import gzip

import pandas as pd
import pytest

from src.data.sec_uncertainty_accessions import parse_submissions, annual_pair_progress, check_header
from src.workflow.sec_uncertainty_accession_source import BoundedSEC, _NoRedirect, _record_failure


def test_b1_complete_catalog_original_pair_headers_and_transport_budget(tmp_path):
    cik = '0000000001'
    def row(number, filed, report='', form='8-K'):
        return {'accessionNumber': f'0001193125-{filed[2:4]}-{number:06d}', 'filingDate': filed,
                'reportDate': report, 'form': form, 'acceptanceDateTime': filed + 'T21:00:00Z',
                'primaryDocument': 'primary.htm'}
    def arrays(rows):
        return {key: [r[key] for r in rows] for key in rows[0]}
    def file(name, rows):
        return {'name': name, 'filingFrom': min(r['filingDate'] for r in rows),
                'filingTo': max(r['filingDate'] for r in rows), 'filingCount': len(rows)}
    def current(rows, files):
        return parse_submissions(json.dumps({'cik': 1, 'name': 'ALPHA INC', 'formerNames': [],
            'filings': {'recent': arrays(rows), 'files': files}}).encode(), cik=cik, role='current')
    older = [row(1, '2018-01-02'), row(2, '2018-11-20', '2018-09-30', '10-K'),
             row(3, '2019-12-02', '2019-09-30', '10-K'), row(4, '2019-12-10', '2019-09-30', '10-K/A'),
             row(5, '2020-12-31')]
    recent = [row(6, '2021-01-04'), row(7, '2022-02-01', '2021-12-31', '10-K')]
    name = f'CIK{cik}-submissions-001.json'; cur = current(recent, [file(name, older)])
    step = annual_pair_progress(cur, {})
    assert step['status'] == 'need_history' and step['next']['name'] == name
    page = parse_submissions(json.dumps(arrays(older)).encode(), cik=cik, role='history', declared=step['next'])['page']
    ready = annual_pair_progress(cur, {name: page})
    assert ready['status'] == 'metadata_pair_selected_pending_headers'
    assert ready['pair']['subsequent']['accessionNumber'] == older[2]['accessionNumber']
    assert ready['pair']['previous']['accessionNumber'] == older[1]['accessionNumber']
    assert all(r['form'] == '10-K' for r in ready['pair'].values())
    with pytest.raises(ValueError, match='range/count'):
        parse_submissions(json.dumps(arrays(older[:-1])).encode(), cik=cik, role='history', declared=step['next'])
    # Natural date gaps between adjacent catalog partitions do not certify a
    # firstfound recent annual: successor then immediate predecessor are needed.
    prior_rows = older[:2] + [row(8, '2019-11-20')]
    following_rows = older[2:]
    first, second = f'CIK{cik}-submissions-010.json', f'CIK{cik}-submissions-011.json'
    gap_cur = current(recent, [file(first, prior_rows), file(second, following_rows)])
    next_step = annual_pair_progress(gap_cur, {})
    assert next_step['rule'] == 'strict_catalog_gap_successor_first' and next_step['next']['name'] == second
    successor = parse_submissions(json.dumps(arrays(following_rows)).encode(), cik=cik, role='history', declared=next_step['next'])['page']
    next_step = annual_pair_progress(gap_cur, {second: successor})
    assert next_step['next']['name'] == first
    predecessor = parse_submissions(json.dumps(arrays(prior_rows)).encode(), cik=cik, role='history', declared=next_step['next'])['page']
    assert annual_pair_progress(gap_cur, {second: successor, first: predecessor})['status'] == 'metadata_pair_selected_pending_headers'
    empty_successor = [row(9, '2019-12-02'), row(10, '2019-12-20')]
    later_rows = [row(11, '2020-02-03', '2019-09-30', '10-K'), row(12, '2020-12-31')]
    third = f'CIK{cik}-submissions-012.json'
    right_gap = current(recent, [file(first, prior_rows), file(second, empty_successor), file(third, later_rows)])
    action = annual_pair_progress(right_gap, {})
    empty_page = parse_submissions(json.dumps(arrays(empty_successor)).encode(), cik=cik, role='history', declared=action['next'])['page']
    assert annual_pair_progress(right_gap, {second: empty_page})['status'] == 'gap'
    # A fiscal 52-week interval can have report ends in the same civil year.
    same_year = current([row(20, '2020-02-07', '2020-01-03', '10-K'), row(21, '2021-02-05', '2020-12-31', '10-K')], [])
    assert annual_pair_progress(same_year, {}, anchor_date='2020-03-01')['status'] == 'metadata_pair_selected_pending_headers'
    selected = ready['pair']['subsequent']
    header = f'''<SEC-HEADER>
<ACCEPTANCE-DATETIME>20191202160000
ACCESSION NUMBER: {selected['accessionNumber']}
CONFORMED SUBMISSION TYPE: 10-K
CONFORMED PERIOD OF REPORT: 20190930
FILED AS OF DATE: 20191202
DATE AS OF CHANGE: 20191202
FILER:
 COMPANY DATA:
  COMPANY CONFORMED NAME: ALPHA INC
  CENTRAL INDEX KEY: 0000000001
  FISCAL YEAR END: 0930
 FILING VALUES:
  FORM TYPE: 10-K
</SEC-HEADER>
'''.encode()
    calendar = pd.bdate_range('2018-01-01', '2022-12-30')
    def inspect(body=header, filing=selected):
        return check_header(body, cik=cik, filing=filing, anchor_names=['Alpha Inc.'], calendar=calendar, role='subsequent')
    okay = inspect()
    assert okay['header_qualified'] and okay['conditional_available_at'] == pd.Timestamp('2019-12-03 09:29', tz='America/New_York')
    assert okay['metadata_version_status'] == 'no_later_header_change_reported'
    assert okay['body_first_vintage_status'] == 'unknown_body_not_requested'
    assert not inspect(header.replace(b'CENTRAL INDEX KEY: 0000000001', b'CENTRAL INDEX KEY: 0000000002'))['header_qualified']
    assert not inspect(header.replace(b'FILED AS OF DATE: 20191202', b'FILED AS OF DATE: 20191203'))['header_qualified']
    bad_clock = dict(selected, acceptanceDateTime='2019-12-02T16:00:00Z', accepted_provisional=pd.Timestamp('2019-12-02T16:00:00Z'))
    assert not inspect(filing=bad_clock)['header_qualified']
    revised = inspect(header.replace(b'DATE AS OF CHANGE: 20191202', b'DATE AS OF CHANGE: 20191220'))
    assert not revised['header_qualified'] and 'later_header' in revised['metadata_version_status']
    assert revised['body_first_vintage_status'] == 'unknown_body_not_requested'
    # A fake opener performs zero network IO. Reservation/rate/redirect/stop
    # behavior is exercised with synthetic response bytes only.
    class Response(io.BytesIO):
        def __init__(self, status=200, body=b'{}', encoding=None):
            super().__init__(body); self.status = status; self.headers = {'Content-Type': 'application/json', 'Content-Encoding': encoding}
    clock = [0.]; starts = []
    def now(): return clock[0]
    def sleep(seconds): clock[0] += seconds
    def opener(request, timeout):
        starts.append(clock[0]); return Response()
    other = '0000000002'
    client = BoundedSEC(tmp_path / 'budget', ciks=[cik, other], user_agent='synthetic-project/1.0', opener=opener, clock=now, sleeper=sleep)
    for who in [cik, other]:
        client.get('current', who)
        names = [f'CIK{who}-submissions-{i:03d}.json' for i in [1, 2]]
        client.allow_history(who, [{'name': n} for n in names])
        for item in names: client.get('history', who, item)
        for j in [1, 2]:
            acc = f'0001193125-20-{int(who)*10+j:06d}'; client.allow_header(who, acc); client.get('header', who, acc)
    assert len(client.state['attempts']) == 10 and all(b-a >= 1 for a, b in zip(starts, starts[1:]))
    client.allow_header(cik, '0001193125-20-999999')
    with pytest.raises(ValueError, match='budget'):
        client.get('header', cik, '0001193125-20-999999')
    blocked = BoundedSEC(tmp_path / 'blocked', ciks=[cik, other], user_agent='synthetic-project/1.0',
        opener=lambda *a, **k: Response(403), clock=now, sleeper=sleep)
    assert blocked.get('current', cik)[0] is None
    assert blocked.get('current', other)[1]['no_request']
    assert len(blocked.state['attempts']) == 1
    compressed = gzip.compress(b'<html><title>Just a moment</title>challenge-platform</html>')
    challenge = BoundedSEC(tmp_path / 'gzip_challenge', ciks=[cik, other], user_agent='synthetic-project/1.0',
        opener=lambda *a, **k: Response(200, compressed, 'gzip'), clock=now, sleeper=sleep)
    decoded, receipt = challenge.get('current', cik)
    assert decoded is None and receipt['challenge'] and receipt['challenge_detection'] == 'decoded_response'
    assert challenge.get('current', other)[1]['no_request'] and len(challenge.state['attempts']) == 1
    assert challenge.raw_responses[1] == compressed
    class Recorder:
        def __init__(self): self.saved = {}; self.status = None
        def save_objects(self, artifact_path, **objects): self.saved[artifact_path] = objects
        def end_run(self, status): self.status = status
    fake_rec = Recorder()
    challenge.state['attempts'].append({'number': 2, 'state': 'reserved_before_request'})
    _record_failure(fake_rec, {'status': 'failed', 'exception': 'synthetic'}, challenge, [receipt])
    assert fake_rec.status == 'FAILED' and fake_rec.saved['source']['HTTP_state']['attempts'][-1]['state'] == 'reserved_before_request'
    assert fake_rec.saved['source/raw_http']['request_01'] == compressed
    redirect = BoundedSEC(tmp_path / 'redirect', ciks=[cik], user_agent='synthetic-project/1.0',
        opener=lambda *a, **k: Response(302), clock=now, sleeper=sleep)
    assert redirect.get('current', cik)[1]['redirect_not_followed']
    assert _NoRedirect().redirect_request(None, None, 302, None, None, 'https://example.invalid') is None
