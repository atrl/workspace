from copy import deepcopy
from decimal import Decimal
import base64
import hashlib
import json

import pandas as pd
import pytest

from src.data.etf_corporate_actions import (
    ISHARES_FIELDS, DIREXION_FIELDS, parse_ishares_distribution_workbook,
    parse_direxion_distribution_table, qualify_etf_corporate_actions,
    compare_futu_cash_events, verify_historical_2014_source,
)


def workbook(rows, *, header=ISHARES_FIELDS, outside='href="a&b"'):
    def row(values):
        return '<ss:Row>' + ''.join(f'<ss:Cell ss:Index="{i}"><ss:Data ss:Type="String">{v}</ss:Data></ss:Cell>'
                                  for i, v in enumerate(values, 1)) + '</ss:Row>'
    return (f'<Workbook><bad {outside}/><ss:Worksheet ss:Name="Distributions"><ss:Table>'
            + row(header) + ''.join(row(r) for r in rows)
            + '</ss:Table></ss:Worksheet></Workbook>').encode()


def test_isolated_sheet_does_not_repair_unrelated_bad_xml_or_round_amounts():
    raw = workbook([['Mar 04, 2024', 'Mar 04, 2024', 'Mar 07, 2024',
                     '0.166176', '0.166176', '0', '0', '0']])
    before = raw[:]
    frame = parse_ishares_distribution_workbook(raw)
    assert raw == before
    assert frame.iloc[0]['Ex-Date'] == '2024-03-04'
    assert frame.iloc[0]['Income'] == '0.166176'
    assert frame.attrs['source_tokens'][0]['Ex-Date'] == 'Mar 04, 2024'


@pytest.mark.parametrize('mutation', [
    lambda b: b.replace(b'ss:Index="2"', b'ss:Index="1"', 1),
    lambda b: b.replace(b'ss:Index="2"', b'ss:Index="9"', 1),
    lambda b: b.replace(b'<ss:Cell ss:Index="1">', b'<ss:Cell ss:Index="1" ss:Formula="x">', 1),
    lambda b: b.replace(b'Income', b'Bad Header', 1),
    lambda b: b.replace(b'0.01', b'a&b', 1),
    lambda b: b + b,
])
def test_workbook_rejects_ambiguous_or_invalid_target_sheet(mutation):
    raw = workbook([['2024-03-04'] * 3 + ['0.01', '0.01', '0', '0', '0']])
    with pytest.raises(ValueError):
        parse_ishares_distribution_workbook(mutation(raw))


def test_dom_table_retains_dash_tokens_and_rejects_missing_interior_rows():
    text = ','.join(DIREXION_FIELDS) + '\n03/04/2024,03/04/2024,03/07/2024,0.01000,—,—\n'
    frame = parse_direxion_distribution_table(text.encode())
    assert frame.iloc[0]['Ex Date'] == '2024-03-04'
    assert frame.iloc[0]['Income Dividend'] == '0.01000'
    assert frame.iloc[0]['Short-Term Capital Gain'] == '—'
    with pytest.raises(ValueError, match='cell/row'):
        parse_direxion_distribution_table((text + '\n' + text.split('\n')[1]).encode())


@pytest.fixture
def synthetic():
    splits = [
        dict(instrument='SOXL', new_shares_per_old_share=4, record_date='2015-05-18',
             payable_after_close_date='2015-05-19', effective_trading_date='2015-05-20', evidence='split15'),
        dict(instrument='SOXL', new_shares_per_old_share=15, record_date='2021-02-26',
             payable_after_close_date='2021-03-01', effective_trading_date='2021-03-02', evidence='split21'),
        dict(instrument='SOXX', new_shares_per_old_share=3, record_date='2024-03-04',
             payable_after_close_date='2024-03-06', effective_trading_date='2024-03-07', evidence='split24'),
    ]
    soxx = pd.DataFrame([
        ['2014-12-23', '2014-12-23', '2014-12-26', '0.000000', '0.000000', '0', '0', '0'],
        ['2024-03-04', '2024-03-04', '2024-03-07', '0.166176', '0.166176', '0', '0', '0'],
        ['2024-03-07', '2024-03-07', '2024-03-11', '0.100000', '0.100000', '0', '0', '0'],
    ], columns=ISHARES_FIELDS)
    soxl = pd.DataFrame([
        ['2014-12-26', '2014-12-23', '2014-12-31', '0.00532', '—', '—'],
        ['2016-12-22', '2016-12-20', '2016-12-28', '0.00000', '2.76442', '—'],
        ['2026-09-22', '2026-09-22', '2026-09-29', '0.09290', '—', '—'],
    ], columns=DIREXION_FIELDS)
    sources = {}
    for symbol, frame, exfield, positive in [('SOXX', soxx, 'Ex-Date', 2), ('SOXL', soxl, 'Ex Date', 3)]:
        sources[symbol] = dict(source_fields=list(frame.columns), expected_distribution_rows=len(frame),
            expected_ex_date_min=frame[exfield].min(), expected_ex_date_max=frame[exfield].max(),
            expected_positive_rows=positive, expected_zero_rows=len(frame)-positive,
            cash_basis='current_split_shares' if symbol == 'SOXX' else 'event_original_shares',
            basis_evidence_grade='synthetic_source', component_total_abs_tolerance_decimal='0.000001')
    live = dict(capture_kind='http_response_body', body_sha256='0'*64)
    plan = dict(market_as_of='2026-09-25', sources=sources, split_events=splits,
        source_archive={'files': [dict(archive_relative_path=k, capture_kind='http_response_body', sha256='1'*64)
                                  for k in ('split15', 'split21', 'split24')]},
        source_versions={'soxx_ishares_workbook_20261009': live,
            'soxl_direxion_dom_table_20261009': {**live, 'capture_kind':'browser_dom_table_transcription'},
            'soxl_direxion_distribution_archive_20141224': dict(archive_timestamp_utc='2014-12-24T23:42:56Z',
                archive_cdx_sha1_base32='synthetic', source_evidence_known_by_utc='2014-12-24T23:42:56Z')},
        declared_discrepancies=[dict(instrument='SOXL', ex_date='2014-12-23',
            issuer_original_amount_decimal='0.00532', futu_amount_decimal='0.00133', evidence_integrity='synthetic')],
        source_comparison_precision_contract={'soxx_undeclared_difference_max_abs_decimal':'0.00001',
                                             'soxl_undeclared_difference_max_abs_decimal':'0.00001'})
    return plan, {'SOXX':soxx, 'SOXL':soxl}, pd.bdate_range('2014-12-24', '2026-09-25')


def qualify(synthetic):
    p, rows, calendar = synthetic
    return qualify_etf_corporate_actions(rows, p['split_events'], native_calendar=calendar,
                                        market_as_of=p['market_as_of'], source_contract=p)


def test_exact_share_basis_zero_rows_and_unpaid_cash_are_retained(synthetic):
    cash, splits, audit = qualify(synthetic)
    amounts = cash.set_index('event_id').original_total_decimal
    assert Decimal(amounts['SOXX:cash:2024-03-04']) == Decimal('0.498528')
    assert Decimal(amounts['SOXX:cash:2024-03-07']) == Decimal('0.100000')
    assert Decimal(amounts['SOXL:cash:2014-12-23']) == Decimal('0.00532')
    assert Decimal(amounts['SOXL:cash:2016-12-20']) == Decimal('2.76442')
    assert len(cash) == 6 and len(splits) == 3
    assert audit['by_instrument']['SOXX']['zero_rows'] == 1
    assert audit['by_instrument']['SOXX']['outside_reference_range'] == 1
    pending = cash.loc[cash.ex_date.eq('2026-09-22')].iloc[0]
    assert pending.payable_status_at_market_as_of == 'unpaid_receivable_not_spendable'
    assert pending.pay_reference_calendar_status == 'outside_reference_range'
    assert cash.announcement_at.isna().all()
    assert cash.source_evidence_known_by_utc.notna().sum() == 1
    historical = cash[cash.instrument.eq('SOXL') & cash.ex_date.eq('2014-12-23')].iloc[0]
    assert historical.source_capture_kind == 'browser_dom_table_transcription'
    assert historical.basis_source_version == 'soxl_direxion_distribution_archive_20141224'
    assert pd.Timestamp(historical.source_evidence_known_by_utc).date().isoformat() > historical.ex_date


def test_interior_calendar_missing_rejected_without_dropping_source_row(synthetic):
    p, rows, calendar = synthetic
    with pytest.raises(ValueError, match='absent from native calendar'):
        qualify_etf_corporate_actions(rows, p['split_events'], native_calendar=calendar.drop(pd.Timestamp('2016-12-20')),
                                     market_as_of=p['market_as_of'], source_contract=p)


@pytest.mark.parametrize('bad', ['NaN', 'Infinity', '-0.01', ''])
def test_nonfinite_negative_missing_amount_rejected(synthetic, bad):
    p, rows, calendar = synthetic
    rows['SOXL'].loc[1, 'Short-Term Capital Gain'] = bad
    with pytest.raises(ValueError):
        qualify(synthetic)


def test_future_split_and_changed_receipt_kind_rejected(synthetic):
    p, rows, calendar = synthetic
    future = dict(p['split_events'][-1], effective_trading_date='2026-11-05')
    with pytest.raises(ValueError, match='frozen'):
        qualify_etf_corporate_actions(rows, [*p['split_events'], future], native_calendar=calendar,
                                     market_as_of=p['market_as_of'], source_contract=p)
    p['source_archive']['files'][0]['capture_kind'] = 'failed_http_response_body'
    with pytest.raises(ValueError, match='valid archived HTTP'):
        qualify(synthetic)


def test_exact_decimal_component_tolerance_and_duplicate_event(synthetic):
    p, rows, calendar = synthetic
    rows['SOXX'].loc[1, 'Income'] = '0.166175'
    qualify(synthetic)  # exact one displayed six-decimal unit
    rows['SOXX'].loc[1, 'Income'] = '0.166174'
    with pytest.raises(ValueError, match='component precision'):
        qualify(synthetic)
    rows['SOXX'].loc[1, 'Income'] = '0.166176'
    rows['SOXL'].iloc[2] = rows['SOXL'].iloc[1]
    with pytest.raises(ValueError, match='duplicate'):
        qualify(synthetic)


def test_vendor_comparison_retains_all_differences_and_issuer_amounts(synthetic):
    cash, _, _ = qualify(synthetic)
    before = cash.copy(deep=True)
    rehab = {'SOXX':pd.DataFrame({'ex_div_date':['2024-03-04','2024-03-07'],
                                 'per_cash_div':['0.49852','0.10000']}),
             'SOXL':pd.DataFrame({'ex_div_date':['2014-12-23','2016-12-20','2026-09-22'],
                                 'per_cash_div':['0.00133','2.76442','0.09290']})}
    diff, audit = compare_futu_cash_events(cash, rehab, source_contract=synthetic[0])
    assert diff.loc[diff.ex_date.eq('2024-03-04'), 'difference_decimal'].iloc[0] == '0.000008'
    assert diff.loc[diff.ex_date.eq('2014-12-23'), 'difference_decimal'].iloc[0] == '0.00399'
    pd.testing.assert_frame_equal(cash, before, check_exact=True)
    assert audit['all_issuer_amounts_preserved']
    rehab['SOXX'].loc[0, 'per_cash_div'] = '0.49'
    with pytest.raises(ValueError, match='undeclared material'):
        compare_futu_cash_events(cash, rehab, source_contract=synthetic[0])


def test_vendor_missing_event_requires_declared_exact_exception(synthetic):
    cash, _, _ = qualify(synthetic)
    p = synthetic[0]
    rehab = {'SOXX':pd.DataFrame({'ex_div_date':['2024-03-07'], 'per_cash_div':['0.10000']}),
             'SOXL':pd.DataFrame({'ex_div_date':['2014-12-23','2016-12-20','2026-09-22'],
                                 'per_cash_div':['0.00133','2.76442','0.09290']})}
    with pytest.raises(ValueError, match='undeclared missing'):
        compare_futu_cash_events(cash, rehab, source_contract=p)
    p['declared_discrepancies'].append(dict(instrument='SOXX',ex_date='2024-03-04',
        futu_cash_event_present=False, expected_original_share_amount_decimal='0.498528'))
    frame, _ = compare_futu_cash_events(cash, rehab, source_contract=p)
    assert frame.loc[frame.ex_date.eq('2024-03-04'), 'status'].iloc[0] == 'declared_vendor_missing_event'


def test_historical_archive_integrity_and_known_by_not_announcement():
    body = (b'<ul><li>Ex Date - 12/23/2014</li><li>Record Date - 12/26/2014</li>'
        b'<li>Pay Date - 12/31/2014</li></ul><table><tr><td>Ticker</td><td>Fund</td><td>CUSIP</td>'
        b'<td>Distribution Amount</td><td>Distribution Per Share Rate</td></tr>'
        b'<tr><td>SOXL</td><td>Example</td><td>25459W458</td><td>$7,448.00</td><td>0.00532</td></tr></table>')
    digest = base64.b32encode(hashlib.sha1(body).digest()).decode()
    contract = dict(body_sha256=hashlib.sha256(body).hexdigest(), cdx_sha1_base32=digest,
        capture_utc='2014-12-24T23:42:56Z', original_url='http://issuer.invalid/distributions',
        exact_amount_usd_decimal='7448.00', exact_original_share_rate_decimal='0.00532')
    cdx = json.dumps([['original','timestamp','digest','statuscode'],
                     [contract['original_url'],'20141224234256',digest,'200']]).encode()
    result = verify_historical_2014_source(body, cdx, contract)
    assert result['announcement_at'] is None and result['known_by_is_after_ex_date']
    assert result['economic_ex_date'] == '2014-12-23'
    assert not result['ex_ante_availability_certified']
    with pytest.raises(ValueError, match='hash'):
        verify_historical_2014_source(body + b' ', cdx, contract)
    wrong = json.loads(cdx); wrong[1][3] = '404'
    with pytest.raises(ValueError, match='CDX'):
        verify_historical_2014_source(body, json.dumps(wrong).encode(), contract)
