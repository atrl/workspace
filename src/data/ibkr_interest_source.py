"""Parse official benchmark downloads without inventing absent rate history."""
from io import StringIO
import pandas as pd
import numpy as np


def parse_benchmark_month(payload,*,month,as_of):
    lines=[line.split() for line in payload.splitlines() if line.strip()]
    if not lines or lines[0][0]!='Date' or 'USD' not in lines[0]:raise ValueError('unexpected IBKR rate header')
    header=lines[0];position=header.index('USD');rows=[]
    for tokens in lines[1:]:
        if len(tokens)!=len(header):raise ValueError('IBKR rate row width differs')
        stamp=pd.to_datetime(tokens[0],format='%Y%m%d',errors='raise')
        if stamp.strftime('%Y%m')!=month:raise ValueError('response month differs')
        if stamp>pd.Timestamp(as_of):raise ValueError('source exceeds as_of')
        raw=tokens[position];prior='*' in raw;value=raw.replace('*','').replace('%','')
        if value=='N/A':rate=np.nan
        else:rate=float(value.replace('(','-').replace(')',''))/100
        rows.append({'datetime':stamp,'benchmark_rate':rate,'prior_day_flag':float(prior)})
    frame=pd.DataFrame(rows,columns=['datetime','benchmark_rate','prior_day_flag'])
    if frame.datetime.duplicated().any():raise ValueError('duplicate benchmark date')
    return frame
