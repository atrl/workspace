"""Deterministic B0 basket/issuer metadata selection; no SEC transport or text."""
import json
import re

import numpy as np
import pandas as pd

HOLDING_FIELDS = ('report_date', 'filing_date', 'accession', 'series_id', 'ticker', 'cusip',
                  'isin', 'name', 'country', 'weight_pct', 'market_value_usd')
MAPPING_FIELDS = ('ticker', 'cik', 'identity_status', 'source', 'cusips', 'entity_name')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_cik(value):
    """No fractional, boolean, zero, malformed or overlength identifier repair."""
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            return None
        value = int(value)
    text = str(value).strip()
    if not re.fullmatch(r'[0-9]{1,10}', text) or int(text) == 0:
        return None
    return f'{int(text):010d}'


def _text(value):
    return '' if pd.isna(value) else str(value).strip()


def select_b0_metadata(holdings, holdings_manifest, mapping, calendar, *, contract):
    """Keep full anchor security rows; freeze two distinct provisional CIKs only."""
    _require(isinstance(holdings, pd.DataFrame) and set(HOLDING_FIELDS) <= set(holdings), 'NPORT metadata schema differs')
    _require(isinstance(mapping, pd.DataFrame) and set(MAPPING_FIELDS) <= set(mapping), 'issuer mapping schema differs')
    calendar = pd.DatetimeIndex(calendar)
    _require(calendar.tz is None and calendar.is_unique and calendar.is_monotonic_increasing
             and calendar.equals(calendar.normalize()), 'native calendar dates invalid')
    _require(canonical_cik(holdings_manifest['cik']) == contract['fund_cik']
             and holdings_manifest['series_id'] == contract['series_id']
             and holdings_manifest['contract_id'] == contract['contract_id'], 'fund/series/contract identity differs')
    _require(holdings_manifest['sha256'] == contract['declared_raw_holdings_sha256'], 'declared raw NPORT identity differs')
    _require(len(holdings) == holdings_manifest['row_count'], 'NPORT metadata row count differs')
    declared = contract['metadata_declared_scope']
    _require(len(holdings) == declared['holdings_rows'] and len(holdings_manifest['filings']) == declared['filings'],
             'frozen corpus size differs')
    h = holdings.loc[:, list(HOLDING_FIELDS)].copy()
    h['source_row'] = np.arange(len(h), dtype=int)
    for name in ('filing_date', 'report_date'):
        h[name] = pd.to_datetime(h[name], errors='raise')
        _require(h[name].notna().all() and h[name].dt.tz is None
                 and h[name].eq(h[name].dt.normalize()).all(), 'NPORT date schema differs')
    _require(h.series_id.eq(contract['series_id']).all(), 'holdings series differs')
    _require(h.report_date.le(h.filing_date).all(), 'report period occurs after filing')
    filings = pd.DataFrame(holdings_manifest['filings'])
    keys = ['report_date', 'filing_date', 'accession']
    for name in keys[:2]:
        filings[name] = pd.to_datetime(filings[name], errors='raise')
    _require(not filings.duplicated(keys).any() and filings.form.isin(['NPORT-P', 'NPORT-P/A']).all(), 'filing inventory ambiguous')
    actual_keys = set(h[keys].itertuples(index=False, name=None))
    declared_keys = set(filings[keys].itertuples(index=False, name=None))
    _require(actual_keys == declared_keys, 'holdings and filing inventory keys differ')
    positions = calendar.searchsorted(filings.filing_date, side='right')
    filings['available_session'] = [calendar[i] if i < len(calendar) else pd.NaT for i in positions]
    expected = contract['earliest_snapshot_expected']
    key = (pd.Timestamp(expected['report_date']), pd.Timestamp(expected['filing_date']), expected['accession'])
    anchor_meta = filings.loc[filings.report_date.eq(key[0]) & filings.filing_date.eq(key[1]) & filings.accession.eq(key[2])]
    _require(len(anchor_meta) == 1 and anchor_meta.form.iloc[0] == 'NPORT-P', 'frozen original anchor not present')
    session = anchor_meta.available_session.iloc[0]
    _require(pd.notna(session) and session == pd.Timestamp('2019-11-26'), 'strict-next native anchor session differs')
    _require(filings.available_session.min() == session, 'fixed corpus earliest public basket differs')
    known = filings.loc[filings.available_session.le(session)]
    known = known.loc[known.report_date.eq(known.report_date.max())]
    known = known.loc[known.filing_date.eq(known.filing_date.max())]
    _require(len(known) == 1 and tuple(known[keys].iloc[0]) == key, 'late older-report amendment displaced anchor')
    basket = h.loc[h.report_date.eq(key[0]) & h.filing_date.eq(key[1]) & h.accession.eq(key[2])].copy()
    _require(len(basket) == int(anchor_meta.equity_count.iloc[0]), 'anchor security count differs')
    _require(not basket.duplicated(['ticker', 'cusip', 'isin']).any(), 'duplicate source security row')
    rows = []; gaps = []
    for row in basket.itertuples(index=False):
        ticker, cusip, isin = _text(row.ticker), _text(row.cusip).upper(), _text(row.isin).upper()
        matches = mapping.loc[mapping.ticker.astype(str).str.strip().eq(ticker)]
        candidates = sorted({c for c in matches.cik.map(canonical_cik) if c is not None})
        reasons = []
        if matches.empty:
            reasons.append('missing_candidate_mapping')
        elif len(candidates) != 1:
            reasons.append('missing_or_multiple_candidate_CIKs')
        candidate = candidates[0] if len(candidates) == 1 else None
        if candidate == contract['fund_cik']:
            candidate = None; reasons.append('fund_CIK_is_not_an_issuer_CIK')
        if not cusip and not isin:
            reasons.append('source_security_identifier_missing')
        mapping_cusips = sorted(set(re.findall(r'(?<![A-Z0-9])[A-Z0-9]{9}(?![A-Z0-9])',
            ' '.join(matches.cusips.map(_text)).upper())))
        if mapping_cusips and cusip and cusip not in mapping_cusips:
            reasons.append('CUSIP_metadata_conflict_historical_identity_unknown')
        if ticker == 'MRVL' and row.report_date < pd.Timestamp('2021-04-20') and candidate == '0001835632':
            reasons.append('MRVL_successor_CIK_not_historical_identity_proof')
        statuses = sorted(set(matches.identity_status.map(_text)))
        # A unique candidate remains selected even when its historical identity
        # is unknown. No foreign/form or text outcome may replace it later.
        reasons.append('historical_identity_and_original_10K_eligibility_not_admitted_in_B0')
        security_id = f'CUSIP:{cusip}|ISIN:{isin}' if cusip or isin else f'UNRESOLVED_SOURCE_ROW:{row.source_row}'
        item = {'source_row': int(row.source_row), 'ticker': ticker, 'cusip': cusip, 'isin': isin,
            'security_id': security_id, 'name': _text(row.name), 'country': _text(row.country),
            'report_date': row.report_date, 'filing_date': row.filing_date, 'basket_accession': row.accession,
            'candidate_cik': candidate, 'candidate_mapping_rows': len(matches),
            'all_candidate_ciks': json.dumps(candidates), 'prior_identity_status': json.dumps(statuses),
            'mapping_sources': json.dumps(sorted(set(matches.source.map(_text)))),
            'historical_identity_qualified': False, 'original_10K_eligible': 'unknown',
            'gap_reasons': ';'.join(reasons)}
        rows.append(item)
        gaps.append({key: item[key] for key in ('source_row', 'ticker', 'security_id', 'candidate_cik', 'gap_reasons', 'prior_identity_status')})
    security = pd.DataFrame(rows)
    eligible = security.loc[security.candidate_cik.notna()].sort_values(['candidate_cik', 'security_id', 'source_row'])
    candidates = []
    for cik, group in eligible.groupby('candidate_cik', sort=True):
        candidates.append({'candidate_cik': cik, 'security_ids': json.dumps(group.security_id.tolist()),
            'source_rows': json.dumps([int(x) for x in group.source_row]), 'tickers': json.dumps(group.ticker.tolist()),
            'underlying_security_rows': len(group), 'historical_identity_qualified': False,
            'original_10K_eligible': 'unknown', 'next_network_stage': 'needs_separate_B1_GO_and_declared_identity_binding',
            'first_original_accession': None, 'previous_original_accession': None, 'body_URLs_qualified': False})
    candidate_frame = pd.DataFrame(candidates, columns=['candidate_cik', 'security_ids', 'source_rows', 'tickers',
        'underlying_security_rows', 'historical_identity_qualified', 'original_10K_eligible', 'next_network_stage',
        'first_original_accession', 'previous_original_accession', 'body_URLs_qualified'])
    selection = candidate_frame.iloc[:2].copy(); complete = len(selection) == 2 and selection.candidate_cik.nunique() == 2
    selected = set(selection.candidate_cik)
    security['selected_provisional_issuer'] = security.candidate_cik.isin(selected)
    anchor = {**expected, 'fund_cik': contract['fund_cik'], 'series_id': contract['series_id'], 'contract_id': contract['contract_id'],
        'available_session': str(session.date()), 'conditional_decision_at': (session + pd.Timedelta(hours=9, minutes=29)).tz_localize('America/New_York'),
        'filing_clock_precision': 'day', 'first_receipt_qualified': False,
        'scope': 'Earliest in fixed admitted corpus only; late old-report amendments cannot replace this anchor.'}
    summary = {'selection_complete': complete, 'distinct_provisional_CIKs_selected': len(selection),
        'selection_status': 'two_provisional_issuers_frozen_needs_B1' if complete else 'needs_two_distinct_candidate_CIKs_no_replacement',
        'anchor_security_rows': len(security), 'all_anchor_security_rows_preserved': True,
        'unmapped_or_ambiguous_CIK_rows': int(security.candidate_cik.isna().sum()),
        'distinct_candidate_CIKs_in_anchor': len(candidate_frame), 'historical_identity_qualified': False,
        'original_10K_eligibility_qualified': False, 'word_counts_or_features_created': False,
        'full_holdings_object_resident': True, 'full_object_may_include_post2022_metadata': True,
        'disclosed_weight_market_value_fields_resident_but_unused_for_selection': True,
        'early_history_unknown_until': str(anchor['conditional_decision_at'])}
    return {'anchor': anchor, 'security_ledger': security, 'candidate_ledger': candidate_frame,
            'selection': selection, 'mapping_gaps': pd.DataFrame(gaps), 'summary': summary}
