"""Bounded B1 JSON/header metadata parsing; no filing-body or word analysis."""
import json
import re
from datetime import timedelta, timezone

import pandas as pd

NY = 'America/New_York'
FIELDS = ('accessionNumber', 'filingDate', 'reportDate', 'form', 'acceptanceDateTime', 'primaryDocument')


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('missing/invalid date metadata')
    return pd.Timestamp(value)


def _instant(value):
    # This only interprets the explicit ISO offset as a provisional metadata
    # instant. Header semantics are checked independently; Z alone is no proof.
    if not isinstance(value, str) or 'T' not in value or not re.search(r'(Z|[+-]\d{2}:\d{2})$', value):
        return None
    try:
        stamp = pd.Timestamp(value)
        return stamp.tz_convert('UTC') if stamp.tz is not None else None
    except (ValueError, TypeError):
        return None


def _filename(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value)) and '..' not in value


def parse_submissions(body, *, cik, role, declared=None):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    document = json.loads(body.decode('utf-8-sig'), object_pairs_hook=unique)
    if role == 'current':
        if str(document.get('cik', '')).lstrip('0') != str(int(cik)):
            raise ValueError('submissions CIK mismatch')
        source = document['filings']['recent']
        catalog = []
        for item in document['filings'].get('files', []):
            name = item['name']
            if not (_filename(name) and name.startswith('CIK' + cik + '-submissions-') and name.endswith('.json')):
                raise ValueError('unsafe or foreign history filename')
            begin, end = _date(item['filingFrom']), _date(item['filingTo'])
            if begin > end or type(item.get('filingCount')) is not int or item['filingCount'] < 1:
                raise ValueError('invalid history range/count')
            catalog.append({'name': name, 'start': begin, 'end': end, 'count': item['filingCount']})
        issuer = {key: document.get(key) for key in ('cik', 'name', 'formerNames', 'tickers', 'fiscalYearEnd', 'entityType')}
    else:
        if declared is None:
            raise ValueError('history must be explicitly listed in own current response')
        source = document; catalog = []; issuer = None
    if not isinstance(source, dict) or any(not isinstance(source.get(name), list) for name in FIELDS):
        raise ValueError('unsupported submissions array schema')
    length = len(source['accessionNumber'])
    if length == 0 or any(len(source[name]) != length for name in FIELDS):
        raise ValueError('empty or unequal submissions arrays')
    rows = []
    for i in range(length):
        row = {name: source[name][i] for name in FIELDS}
        if not isinstance(row['accessionNumber'], str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', row['accessionNumber']):
            raise ValueError('invalid accession metadata')
        row['filed'] = _date(row['filingDate'])
        row['report'] = _date(row['reportDate']) if row['reportDate'] else None
        row['accepted_provisional'] = _instant(row['acceptanceDateTime'])
        row['source_page'] = 'recent' if role == 'current' else declared['name']
        rows.append(row)
    begin, end = min(x['filed'] for x in rows), max(x['filed'] for x in rows)
    if role == 'history' and (begin != declared['start'] or end != declared['end'] or length != declared['count']):
        raise ValueError('history payload range/count differs from listed declaration')
    page = {'name': 'recent' if role == 'current' else declared['name'], 'start': begin, 'end': end, 'count': length, 'rows': rows}
    return {'page': page, 'catalog': catalog, 'issuer': issuer}


def _catalog(current):
    catalog = [{k: current['page'][k] for k in ('name', 'start', 'end', 'count')}, *current['catalog']]
    catalog.sort(key=lambda row: (row['start'], row['end'], row['name']))
    if len({row['name'] for row in catalog}) != len(catalog):
        raise ValueError('duplicate catalog name')
    if any(left['end'] >= right['start'] for left, right in zip(catalog, catalog[1:])):
        raise ValueError('overlapping/conflicting catalog ranges')
    return catalog


def _rows(pages):
    unique = {}
    for page in pages.values():
        for row in page['rows']:
            acc = row['accessionNumber']
            if acc in unique and any(unique[acc][name] != row[name] for name in FIELDS):
                raise ValueError('conflicting accession across pages')
            unique.setdefault(acc, row)
    return list(unique.values())


def annual_pair_progress(current, histories, *, anchor_date='2019-11-26', cutoff='2023-01-01'):
    """One containing/successor page and at most one adjacent history request."""
    try:
        catalog = _catalog(current); anchor = pd.Timestamp(anchor_date)
        pages = {'recent': current['page'], **histories}
        if len(histories) > 2 or not set(histories) <= {x['name'] for x in current['catalog']}:
            raise ValueError('unapproved history-page inventory')
        containing = [i for i, row in enumerate(catalog) if row['start'] <= anchor <= row['end']]
        if len(containing) == 1:
            base = containing[0]; rule = 'unique_containing_partition'
        else:
            after = next((i for i, row in enumerate(catalog) if row['start'] > anchor), None)
            if after is None or after == 0 or not catalog[after - 1]['end'] < anchor < catalog[after]['start']:
                raise ValueError('anchor not covered or bracketed by adjacent catalog partitions')
            base = after; rule = 'strict_catalog_gap_successor_first'
        base_entry = catalog[base]
        if base_entry['name'] not in pages:
            return {'status': 'need_history', 'next': base_entry, 'rule': rule, 'catalog': catalog}
        rows = _rows(pages)
        originals = [x for x in rows if x['form'] == '10-K']
        if any(x['accepted_provisional'] is None for x in originals):
            raise ValueError('original10K acceptance schema unknown; no filed-midnight fallback')
        begin = pd.Timestamp(anchor_date, tz=NY); stop = pd.Timestamp(cutoff, tz=NY)
        originals.sort(key=lambda x: (x['accepted_provisional'], x['accessionNumber']))
        subsequent = [x for x in originals if begin <= x['accepted_provisional'] < stop]
        candidate = subsequent[0] if subsequent else None
        prior = None
        if candidate:
            older = [x for x in originals if x['accepted_provisional'] < candidate['accepted_provisional']]
            prior = older[-1] if older else None
        own_post = [x for x in pages[base_entry['name']]['rows']
                    if x['form'] == '10-K' and x['accepted_provisional'] is not None
                    and begin <= x['accepted_provisional'] < stop]
        needed_start = min(anchor, prior['filed']) if prior else anchor
        needed_end = max(anchor, candidate['filed']) if candidate else pd.Timestamp(cutoff) - pd.Timedelta(days=1)
        unread = [x for x in catalog if x['name'] not in pages and x['start'] <= needed_end and x['end'] >= needed_start]
        if candidate is None or prior is None or unread:
            # Never treat a candidate in the already-loaded recent block as
            # earliest while an intervening catalog partition is unread.
            direction = 1 if not own_post and rule != 'strict_catalog_gap_successor_first' else -1
            neighbour = base + direction
            if unread:
                intervening = [i for i, x in enumerate(catalog) if x in unread]
                if rule == 'strict_catalog_gap_successor_first' and any(i > base for i in intervening):
                    raise ValueError('strict catalog gap needs a further successor; no extra direction allowed')
                if any(i > base for i in intervening) and not own_post:
                    neighbour = base + 1
            if not 0 <= neighbour < len(catalog):
                raise ValueError('missing adjacent original annual coverage')
            request = catalog[neighbour]
            if request['name'] in pages or len(histories) >= 2:
                raise ValueError('coverage or pair requires unapproved extra page; no rescue')
            if any(name != base_entry['name'] for name in histories) and base_entry['name'] != 'recent':
                raise ValueError('second adjacent page already consumed')
            return {'status': 'need_history', 'next': request, 'rule': rule, 'catalog': catalog}
        if candidate is None or prior is None:
            raise ValueError('two original annuals not available')
        if prior['report'] is None or candidate['report'] is None:
            raise ValueError('missing original report period')
        interval = (candidate['report'] - prior['report']).days
        if not 350 <= interval <= 380:
            raise ValueError('annual-end interval inconsistent or transitional')
        for selected in (prior, candidate):
            if selected['report'] > selected['filed']:
                raise ValueError('report period after filing')
            if sum(x['report'] == selected['report'] for x in originals) != 1:
                raise ValueError('duplicate/ambiguous original annual report period')
            if not _filename(selected['primaryDocument']):
                raise ValueError('unsafe/missing primaryDocument filename')
        between = [x for x in rows if x['accepted_provisional'] is not None
                   and prior['accepted_provisional'] <= x['accepted_provisional'] <= candidate['accepted_provisional']]
        if any(x['form'] in ('10-KT', '10-KT/A', '10-K/T', '10-K/T/A') for x in between):
            raise ValueError('transitional annual inside selected ordering')
        return {'status': 'metadata_pair_selected_pending_headers', 'pair': {'previous': prior, 'subsequent': candidate},
                'rule': rule, 'catalog': catalog, 'required_filing_date_span': [str(needed_start.date()), str(needed_end.date())],
                'unread_intervening_ranges': [], 'annual_end_interval_days': interval,
                'coverage_meaning': 'Complete listed partition ordering; natural calendar-day gaps are not treated as missing filings.'}
    except (ValueError, KeyError, TypeError) as error:
        return {'status': 'gap', 'reason': str(error), 'pair': None}


def _header_field(text, key):
    matches = re.findall(r'^\s*' + re.escape(key) + r'\s*:\s*(.*?)\s*$', text, re.M | re.I)
    return matches[0] if len(matches) == 1 else None


def _name(value):
    return re.sub(r'[^A-Z0-9]', '', str(value).upper())


def check_header(body, *, cik, filing, anchor_names, calendar, role):
    """Keep uncertain clock/version/identity fields as gaps; never repair a pair."""
    result = {'cik': cik, 'role': role, 'accession': filing['accessionNumber'], 'form': filing['form'],
        'report_date': filing['reportDate'], 'filed_date': filing['filingDate'],
        'primary_document_metadata': filing['primaryDocument'], 'primary_document_body_verified': False,
        'json_acceptance_raw': filing['acceptanceDateTime'], 'header_qualified': False,
        'metadata_version_status': 'unknown_header_not_checked',
        'body_first_vintage_status': 'unknown_body_not_requested', 'historical_PIT_qualified': False,
        'conditional_available_at': None, 'gap_reasons': []}
    gaps = result['gap_reasons']
    try:
        text = body.decode('utf-8-sig')
        if re.search(r'<DOCUMENT>', text, re.I):
            raise ValueError('unexpected body container at header-only endpoint; never parse its text')
        accession = _header_field(text, 'ACCESSION NUMBER')
        form = _header_field(text, 'CONFORMED SUBMISSION TYPE')
        period = _header_field(text, 'CONFORMED PERIOD OF REPORT')
        filed = _header_field(text, 'FILED AS OF DATE')
        changed = _header_field(text, 'DATE AS OF CHANGE')
        filer_matches = list(re.finditer(r'^\s*FILER\s*:\s*$', text, re.M | re.I))
        if len(filer_matches) != 1:
            raise ValueError('missing/multiple FILER sections')
        filer = text[filer_matches[0].end():]
        filer = re.split(r'^\s*(?:REPORTING-OWNER|SUBJECT COMPANY|ISSUER)\s*:', filer, maxsplit=1, flags=re.M | re.I)[0]
        header_cik = _header_field(filer, 'CENTRAL INDEX KEY')
        legal_name = _header_field(filer, 'COMPANY CONFORMED NAME')
        form_in_filer = _header_field(filer, 'FORM TYPE')
        result.update(header_filer_cik=header_cik, header_legal_name=legal_name,
                      header_fiscal_year_end=_header_field(filer, 'FISCAL YEAR END'),
                      header_state_of_incorporation=_header_field(filer, 'STATE OF INCORPORATION'),
                      header_date_as_of_change=changed)
        if accession != filing['accessionNumber'] or form != '10-K' or form_in_filer != '10-K':
            gaps.append('header_accession_or_original_form_conflict')
        if header_cik is None or not header_cik.isdigit() or int(header_cik) != int(cik):
            gaps.append('header_filer_CIK_conflict')
        if not legal_name or _name(legal_name) not in {_name(x) for x in anchor_names if x}:
            gaps.append('historical_name_link_unknown_no_fuzzy_repair')
        for value, expected, field in ((period, filing['reportDate'], 'report_date'), (filed, filing['filingDate'], 'filed_date')):
            if value is None or not re.fullmatch(r'\d{8}', value) or pd.Timestamp(value) != pd.Timestamp(expected):
                gaps.append('header_' + field + '_conflict')
        accepts = re.findall(r'<ACCEPTANCE-DATETIME>\s*(\d{14})', text, re.I)
        raw_acceptance = accepts[0] if len(accepts) == 1 else None
        result['header_acceptance_raw'] = raw_acceptance
        result['header_clock_authority'] = 'SEC PDS/FAQ labels Acceptance Time EST; DST semantics not specified.'
        result['json_clock_status'] = 'explicit_ISO_offset_provisional_until_header_crosscheck'
        # The literal documented fixed EST interpretation is only admitted when
        # it agrees with New York standard time. Summer/ambiguous clocks remain
        # unknown; never try both offsets and choose the one that matches JSON.
        accepted = None
        if raw_acceptance is None:
            gaps.append('header_acceptance_missing_or_ambiguous')
        else:
            naive = pd.to_datetime(raw_acceptance, format='%Y%m%d%H%M%S', errors='raise')
            local = naive.tz_localize(NY, ambiguous='raise', nonexistent='raise')
            if local.utcoffset() != timedelta(hours=-5):
                gaps.append('header_EST_DST_schema_unknown')
            else:
                accepted = naive.tz_localize(timezone(timedelta(hours=-5)))
                candidate = filing['accepted_provisional']
                if candidate is None or accepted.tz_convert('UTC') != candidate:
                    gaps.append('header_JSON_acceptance_clock_mismatch')
                result['accepted_header_literal_EST'] = accepted
        result['exact_first_publication_time_qualified'] = False
        public_date = max(pd.Timestamp(filing['filingDate']), accepted.tz_localize(None).normalize()) if accepted is not None else None
        if changed is None or not re.fullmatch(r'\d{8}', changed):
            result['metadata_version_status'] = 'unknown_DATE_AS_OF_CHANGE'
            gaps.append('DATE_AS_OF_CHANGE_unknown')
        elif public_date is not None and pd.Timestamp(changed) > public_date:
            result['metadata_version_status'] = 'later_header_metadata_change_provenance_unresolved'
            gaps.append('later_header_metadata_PAC_change_unresolved_not_body_mutation_proof')
        elif public_date is not None:
            result['metadata_version_status'] = 'no_later_header_change_reported'
        else:
            result['metadata_version_status'] = 'unknown_clock_prevents_change_date_comparison'
        result['explicit_PAC_marker_present'] = bool(re.search(r'POST[ -]ACCEPTANCE[ -]CORRECTION|<PAC>', text, re.I))
        if result['explicit_PAC_marker_present']:
            result['metadata_version_status'] = 'explicit_PAC_metadata_provenance_unresolved'
            gaps.append('explicit_PAC_metadata_change_not_body_mutation_proof')
        result['PAC_semantics'] = 'PDS says PAC changes header/type/deletion metadata, not document text; no original-body certification follows.'
        if public_date is not None:
            dates = pd.DatetimeIndex(calendar); location = dates.searchsorted(public_date, side='right')
            if location >= len(dates):
                gaps.append('conditional_availability_outside_admitted_calendar')
            else:
                proposed = (dates[location] + pd.Timedelta(hours=9, minutes=29)).tz_localize(NY)
                result['conditional_date_rule_candidate'] = proposed
                if not gaps:
                    result['conditional_available_at'] = proposed
        result['header_qualified'] = not gaps
    except (ValueError, TypeError, UnicodeDecodeError, NotImplementedError) as error:
        gaps.append('header_schema_gap: ' + str(error))
    return result
