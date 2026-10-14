"""Read retained SEC tool text without treating it as original HTML bytes."""
import re


def quarterly_filing_identity(record):
    quarter=record.get('reported_fiscal_period') or record.get('reported_fiscal_quarter')
    if not quarter or not re.fullmatch(r'FY\d{4}Q[1-4]',quarter):
        raise ValueError('explicit fiscal quarter required')
    if record.get('event_role') not in ('initial','quarterly_earnings_release'):
        raise ValueError('not an original quarterly release record')
    return {'quarter':quarter,'accession':record['accession'],
            'filing_date':record.get('filing_date'),'release_date':record.get('release_date'),
            'api_accepted_raw':record.get('api_accepted_at_raw') or record.get('api_accepted_original'),
            'index_accepted_raw':record.get('index_accepted_at_raw') or record.get('index_accepted_original')}


def extract_sec_eps_text(text,*,quarter,accession):
    year,q=map(int,re.fullmatch(r'FY(\d{4})Q([1-4])',quarter).groups())
    folder=accession.replace('-','')
    urls=re.findall(r'https://www\.sec\.gov/Archives/edgar/data/1045810/'+folder+r'/[^\s"{}()]+\.htm',text)
    if not urls:raise ValueError('original accession URL absent')
    lines=[re.sub(r'^L\d+:\s*','',line).strip() for line in text.splitlines()]
    starts=[i for i,line in enumerate(lines) if re.fullmatch(r'GAAP(?: Quarterly Financial Comparison|[ |]*)',line)]
    if not starts:raise ValueError('GAAP table absent')
    start=starts[0]
    block=[]
    for line in lines[start+1:]:
        if line.startswith('Non-GAAP'):break
        block.append(line)
    block='\n'.join(block)
    previous_q=4 if q==1 else q-1;previous_y=year-1 if q==1 else year
    header=rf'Q{q}\s+FY{year%100:02}\s*\|\s*Q{previous_q}\s+FY{previous_y%100:02}\s*\|\s*Q{q}\s+FY{(year-1)%100:02}'
    if not re.search(header,block):raise ValueError('quarter comparison columns differ')
    match=re.search(r'Diluted earnings per share\*?\s*\|\s*\$([0-9.]+)\s*\|\s*\$([0-9.]+)\s*\|\s*\$([0-9.]+)',block)
    if not match:raise ValueError('explicit diluted quarterly EPS row absent')
    return {'quarter':quarter,'accession':accession,'source_url':urls[0],
            'current_eps':match.group(1),'prior_quarter_comparative_eps':match.group(2),
            'prior_year_comparative_eps':match.group(3),'accounting_basis':'GAAP','share_basis':'diluted',
            'representation':'retained_web_tool_text_not_original_HTML'}


def extract_company_nongaap_eps_text(text,*,quarter,accession):
    """Issuer non-GAAP is a distinct fact, never certified as I/B/E/S actual."""
    result=extract_sec_eps_text(text,quarter=quarter,accession=accession)
    lines=[re.sub(r'^L\d+:\s*','',line).strip() for line in text.splitlines()]
    starts=[i for i,line in enumerate(lines) if re.fullmatch(r'Non-GAAP(?: Quarterly Financial Comparison|[ |]*)',line)]
    if not starts:raise ValueError('Non-GAAP table absent')
    block='\n'.join(lines[starts[0]+1:])
    year,q=map(int,re.fullmatch(r'FY(\d{4})Q([1-4])',quarter).groups())
    previous_q=4 if q==1 else q-1;previous_y=year-1 if q==1 else year
    header=rf'Q{q}\s+FY{year%100:02}\s*\|\s*Q{previous_q}\s+FY{previous_y%100:02}\s*\|\s*Q{q}\s+FY{(year-1)%100:02}'
    # The first three-quarter EPS row is the quarterly summary, not annual totals.
    match=re.search(r'Diluted earnings per share\*?\s*\|\s*\$([0-9.]+)\s*\|\s*\$([0-9.]+)\s*\|\s*\$([0-9.]+)',block)
    if not match:raise ValueError('Non-GAAP quarterly EPS row absent')
    if not re.search(header,block[:match.start()]):raise ValueError('Non-GAAP quarter columns differ')
    return {**result,'current_eps':match.group(1),'prior_quarter_comparative_eps':match.group(2),
            'prior_year_comparative_eps':match.group(3),'accounting_basis':'issuer_non_GAAP','consensus_basis_certified':False}
