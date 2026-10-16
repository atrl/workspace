"""Bounded ETF source qualification; raw evidence belongs to the Qlib Recorder.

No historical ratio is inferred from today's fund page. This module audits
source payloads and disclosed holdings, never computes a market outcome.
"""
from pathlib import Path
import hashlib
import json
import re
import html
import pandas as pd


def inspect_download(payload: bytes) -> dict:
    """Treat HTTP success separately from a valid dated spreadsheet."""
    text = payload.decode('utf-8', errors='replace')
    result = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
              'format': 'unknown', 'worksheets': [], 'valuation_labels': []}
    if '<html' in text[:1000].lower() or '<!doctype html' in text[:1000].lower():
        result['format'] = 'html_not_historical_workbook'
    elif 'urn:schemas-microsoft-com:office:spreadsheet' in text:
        # Issuer SpreadsheetML has unescaped ampersands in hyperlink attributes.
        # Read only Data text nodes and worksheet labels; no XML entity execution.
        result['format'] = 'spreadsheet_xml'
        result['worksheets'] = re.findall(r'<(?:ss:)?Worksheet[^>]*ss:Name="([^"]+)"', text)
        values = [html.unescape(re.sub(r'<[^>]+>', '', x)).strip()
                  for x in re.findall(r'<ss:Data[^>]*>(.*?)</ss:Data>', text, re.S)]
        result['valuation_labels'] = [x for x in values if re.search(r'P/E|P/B|Price.to.Earnings|Price.to.Book', x, re.I)]
        result['date_evidence'] = [x for x in values[:50] if len(x) < 60]
        result['reported_asof'] = [values[i+1] for i,x in enumerate(values[:-1]) if x == 'Fund Holdings as of']
    elif text.lstrip().startswith('{'):
        result['format'] = 'json'
    return result


def holdings_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Inventory authentic disclosed baskets, keeping report and filing dates."""
    keys = ['accession', 'report_date', 'filing_date']
    grouped = frame.groupby(keys, sort=True)
    result = grouped.agg(equity_weight_pct=('weight_pct', 'sum'), symbols=('ticker', 'nunique')).reset_index()
    result['disclosure_lag_days'] = (pd.to_datetime(result.filing_date) - pd.to_datetime(result.report_date)).dt.days
    result['availability_rule'] = 'next exchange session after filing date; no same-day eligibility'
    return result


def companyfacts_inventory(payload: bytes, ticker: str) -> tuple[dict, pd.DataFrame]:
    """Count accession-level income facts; counts are not qualification of EPS."""
    doc = json.loads(payload)
    rows = []
    for tag in ['NetIncomeLoss', 'ProfitLoss', 'EarningsPerShareDiluted', 'StockholdersEquity']:
        for unit, facts in doc.get('facts', {}).get('us-gaap', {}).get(tag, {}).get('units', {}).items():
            for fact in facts:
                rows.append({'ticker': ticker, 'tag': tag, 'unit': unit, **fact})
    data = pd.DataFrame(rows)
    if data.empty:
        return {'ticker': ticker, 'fact_rows': 0}, data
    summary = {'ticker': ticker, 'fact_rows': len(data), 'accessions': data.accn.nunique(),
               'filed_start': data.filed.min(), 'filed_end': data.filed.max(),
               'status': 'candidate_accession_facts_need_original_filing_period_share_basis_validation'}
    return summary, data


def run(root, staging='/tmp/etf_source_probe'):
    """Register the frozen source acquisition, or replay its Recorder audit."""
    import os
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    root, staging = Path(root).resolve(), Path(staging)
    card = root / 'research/etf_valuation_source_20261016'
    provider = root / 'data/qlib/soxx_pit_leveraged_training_v1'
    qlib.init(provider_uri=str(provider), region='us', kernels=1)
    calendar = D.calendar(start_time='2019-01-01', end_time='2026-10-16')
    if (card / 'summary.json').exists():
        summary = json.loads((card / 'summary.json').read_text())
        R.set_uri(summary['recorder_uri'])
        rec = R.get_recorder(recorder_id=summary['recorder_id'], experiment_id=summary['experiment_id'])
        evidence = rec.load_object('inputs/raw_payloads')
        for row in rec.load_object('inputs/acquisition_manifest'):
            if 'sha256' in row:
                assert hashlib.sha256(evidence[row['name']]).hexdigest() == row['sha256']
        return summary
    if hashlib.sha256((card/'plan.json').read_bytes()).hexdigest() != '448fb4c2831b04e48559975e5afeb00897a2fcc27cb8b30b6834478d557109ad':
        raise ValueError('frozen source plan changed')
    plan = json.loads((card/'plan.json').read_text())
    entries = json.loads((staging/'entries.json').read_text())
    if len(entries) > plan['request_budget']:
        raise ValueError('source request budget exceeded')
    if not len(calendar):
        raise ValueError('Qlib calendar unavailable')
    payloads = {x['name']: (staging/x['name']).read_bytes() for x in entries if 'sha256' in x}
    for entry in entries:
        if 'sha256' in entry and hashlib.sha256(payloads[entry['name']]).hexdigest() != entry['sha256']:
            raise ValueError('acquisition payload hash mismatch')
    holdings_path = root/'data/qlib/soxx_pit_leveraged_training_v1/sec_nport_holdings_history.csv'
    holdings_manifest = json.loads(holdings_path.with_name('sec_nport_holdings_manifest.json').read_text())
    if hashlib.sha256(holdings_path.read_bytes()).hexdigest() != holdings_manifest['sha256']:
        raise ValueError('holdings source hash mismatch')
    holdings = pd.read_csv(holdings_path)
    coverage = holdings_coverage(holdings)
    audits, facts, fundamentals = [], {}, []
    for entry in entries:
        audit = dict(entry)
        if entry['name'] in payloads:
            audit.update(inspect_download(payloads[entry['name']]))
        audits.append(audit)
        if entry['name'] in ['nvda.json','intc.json'] and audit.get('format') == 'json':
            summary, data = companyfacts_inventory(payloads[entry['name']], entry['name'].split('.')[0].upper())
            candidates = original_period_candidates(data)
            summary['original_period_candidates'] = len(candidates)
            summary['first_filing_conflicts'] = int(candidates.first_filing_conflict.sum()) if len(candidates) else 0
            fundamentals.append(summary); facts[summary['ticker']] = {'all_facts':data, 'first_filing_candidates':candidates}
    nvda = holdings.loc[holdings.ticker.eq('NVDA'), 'weight_pct']
    rows = [
        {'source':'issuer historical workbook probes','status':'needs_historical_ratio_publication_vintages','qualified_months':0,'detail':'HTTP/body, date and ratio availability in download_audit.json; request date alone never qualifies'},
        {'source':'SEC NPORT local canonical history','status':'candidate_delayed_disclosed_basket','qualified_months':0,'detail':f'{len(holdings)} holdings; {holdings.ticker.nunique()} symbols; {len(coverage)} filings; no exact acceptance clock; next-session conservative eligibility required'},
        {'source':'existing NVDA earnings corpus','status':'partial_basket_only','qualified_months':0,'detail':f'NVDA weight {nvda.min():.4f} to {nvda.max():.4f} percent; cannot represent other constituents; original accession split/clock qualifications remain'},
        {'source':'SEC companyfacts representative probe','status':'candidate_reconstruction_intake','qualified_months':0,'detail':'See fundamentals_inventory.csv; original fiscal-period, accession and share/split alignment remain before whole-basket EP'},
        {'source':'SOXX versus SOX methodology','status':'context_only_needs_dated_weight_files','qualified_months':0,'detail':'2021 index change and 2026 method separately retained; no announced historical weights or continuous index-version history qualified'},
    ]
    tables = {'source_qualification':pd.DataFrame(rows), 'holdings_coverage':coverage, 'fundamentals_inventory':pd.DataFrame(fundamentals)}
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/companyfacts']}
    tracking = card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'], uri=str(tracking)):
        rec=R.get_recorder()
        rec.save_objects(artifact_path='inputs',plan=plan,acquisition_manifest=entries,raw_payloads=payloads,holdings=holdings,holdings_manifest=json.loads(holdings_path.with_name('sec_nport_holdings_manifest.json').read_text()),nas_inventory=(staging/'nas_inventory.txt').read_text(),methodology_sources=json.loads((card/'methodology_sources.json').read_text()))
        rec.save_objects(artifact_path='analysis',download_audit=audits,companyfacts=facts,**tables)
        rec.save_objects(artifact_path='sources',module=Path(__file__).read_text())
        result={'experiment':plan['experiment'],'decision':'needs_qualified_historical_valuation_inputs','objective_achieved':False,'new_fits':0,'new_backtests':0,'qualified_monthly_rows':0,'qualified_input_path':None,'provider_uri':str(provider),'market':'SOXX','region':'us','as_of':'2026-10-16','qlib_calendar_rows':len(calendar),'calendar_first':str(calendar[0]),'calendar_last':str(calendar[-1]),'holdings_rows':len(holdings),'historic_symbols':holdings.ticker.nunique(),'holdings_filings':len(coverage),'max_disclosure_lag_days':int(coverage.disclosure_lag_days.max()),'fundamentals_inventory':fundamentals,'bounded_probe_requests':len(entries),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'plan_sha256':hashlib.sha256((card/'plan.json').read_bytes()).hexdigest(),'qlib_docs':['https://qlib.readthedocs.io/en/latest/component/workflow.html'],'next_action':'Extend accession-level historical earnings intake across 47 observed symbols; resolve ADR/splits and old-report amendments; keep current-ratio series and reconstructed EP separate. Source qualification is not scientific rejection or overall blocker.'}
        rec.save_objects(artifact_path='analysis',summary=result)
        artifacts=Path(result['recorder_local_dir'])/'artifacts'
        rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),qualified_monthly_rows=0,holdings_rows=len(holdings))
    for key, value in tables.items():value.to_csv(card/(key+'.csv'),index=False)
    (card/'download_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    from src.data.nas_market_data import archive_nas_recorder
    binding=json.loads((root/'research/nvda_eps_corpus_20261015/nas_binding.json').read_text())
    binding.pop('evidence',None)
    (card/'nas_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    (card/'verification_manifest.json').write_text(json.dumps({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},indent=2)+'\n')
    result['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    (card/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def original_period_candidates(facts: pd.DataFrame) -> pd.DataFrame:
    """Earliest reported value per period/unit, retaining ambiguous first filings.

    These are intake candidates, not a certified TTM series: original filing text,
    fiscal quarter construction, split/ADR basis and public clock still matter.
    Later comparative restatements never overwrite an earlier candidate.
    """
    if facts.empty:
        return facts.copy()
    keys = ['ticker', 'tag', 'unit', 'start', 'end']
    data = facts.loc[facts.form.isin(['10-K', '10-Q', '20-F', '40-F'])].copy()
    first = data.groupby(keys, dropna=False).filed.transform('min')
    data = data.loc[data.filed.eq(first)].drop_duplicates(keys + ['accn', 'val'])
    data['first_filing_conflict'] = data.groupby(keys, dropna=False).val.transform('nunique').gt(1)
    return data.sort_values(['ticker', 'tag', 'end', 'filed'])
