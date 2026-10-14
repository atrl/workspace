"""One fixed as-reported gross-margin change, under delayed filing assumptions."""
import re
import numpy as np
from src.data.sec_eps_text import extract_sec_eps_text
from src.data.delayed_seasonal_eps import delayed_filing_availability,align_delayed_events,FEATURE as EPS_FEATURE,FLAG as EPS_FLAG

FEATURE='nvda_gross_margin_yoy_pp'
FLAG='nvda_gross_margin_ready'


def extract_gross_margin(text,*,quarter,accession):
    identity=extract_sec_eps_text(text,quarter=quarter,accession=accession)
    lines=[re.sub(r'^L\d+:\s*','',line).strip() for line in text.splitlines()]
    start=next(i for i,line in enumerate(lines) if re.fullmatch(r'GAAP(?: Quarterly Financial Comparison|[ |]*)',line))
    for line in lines[start+1:]:
        if line.startswith('Non-GAAP'):break
        if re.match(r'^Gross margin\s*\|',line):
            values=re.findall(r'(-?\d+(?:\.\d+)?)\s*(?:\|\s*)?%',line)
            if len(values)<3:raise ValueError('three explicit gross margin percentages required')
            return {**identity,'gross_margin_pct':values[0],'prior_quarter_margin_pct':values[1],'prior_year_margin_pct':values[2]}
    raise ValueError('GAAP gross margin row absent')


def delayed_margin_events(reports,calendar):
    events=delayed_filing_availability(reports,calendar)
    ordered=reports.sort_values('quarter');margin=ordered.gross_margin_pct.astype(float).reset_index(drop=True)
    if not np.isfinite(margin).all():raise ValueError('nonfinite gross margin')
    events[FEATURE]=margin-margin.shift(4)
    return events


def align_margin_events(events,signal_index,trade_dates):
    return align_delayed_events(events.rename(columns={FEATURE:EPS_FEATURE}),signal_index,trade_dates).rename(columns={EPS_FEATURE:FEATURE,EPS_FLAG:FLAG})
