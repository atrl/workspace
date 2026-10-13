"""Parse exact issuer units; exposure columns are not assumed to be net assets."""
import io,re
import numpy as np
import pandas as pd


COLUMNS=['TradeDate','AccountTicker','StockTicker','SecurityDescription','Shares','Price','MarketValue','Cusip','HoldingsPercent']


def parse_holdings(raw,*,symbol,observed_at_utc):
    lines=raw.decode('utf-8-sig').splitlines()
    if len(lines)<7 or lines[1].strip()!=symbol:raise ValueError('issuer fund identity differs')
    matched=re.fullmatch(r'Shares Outstanding:\s*(\d+)',lines[2].strip())
    if not matched or int(matched.group(1))<=0:raise ValueError('missing exact outstanding shares')
    start=next((i for i,line in enumerate(lines) if line.startswith('"TradeDate"')),None)
    if start is None:raise ValueError('holdings header missing')
    frame=pd.read_csv(io.StringIO('\n'.join(lines[start:])),keep_default_na=False,dtype=str)
    if frame.columns.tolist()!=COLUMNS or frame.empty or not frame.AccountTicker.eq(symbol).all():raise ValueError('issuer holdings schema differs')
    dates=pd.to_datetime(frame.TradeDate,format='%m/%d/%Y %I:%M:%S %p',errors='raise')
    if dates.nunique()!=1 or not dates.eq(dates.dt.normalize()).all():raise ValueError('expected one reported date, not publication timestamps')
    for field in ['Shares','Price','MarketValue','HoldingsPercent']:
        frame[field]=pd.to_numeric(frame[field],errors='raise')
        if not np.isfinite(frame[field]).all():raise ValueError('invalid numeric holdings')
    observed=pd.Timestamp(observed_at_utc)
    if observed.tzinfo is None:raise ValueError('receipt timezone missing')
    available=observed.tz_convert('America/New_York').ceil('min')
    metadata={'symbol':symbol,'reported_trade_date':str(dates.iloc[0].date()),'shares_outstanding_exact':int(matched.group(1)),'observed_at_utc':observed.isoformat(),'cache_available_at_ny':available.isoformat(),'holdings_rows':len(frame),'swap_asset_rows':int(frame.SecurityDescription.str.contains('SWAP ASSET LEG',regex=False).sum()),'reported_market_value_sum':float(frame.MarketValue.sum()),'reported_holdings_percent_sum':float(frame.HoldingsPercent.sum()),'market_value_sum_is_net_assets':False,'intraday_net_creation_flow':None,'historical_first_publication_certified':False}
    return frame,metadata
