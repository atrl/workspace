"""Bounded same-vendor RSP/SPY daily source parser; no return-account claims."""
import numpy as np
import pandas as pd

def parse_chart(payload,symbol,calendar):
    chart=payload.get('chart',{})
    if chart.get('error') or len(chart.get('result') or [])!=1:raise ValueError('chart result/error')
    result=chart['result'][0];meta=result.get('meta',{})
    for key,value in [('symbol',symbol),('currency','USD'),('exchangeTimezoneName','America/New_York'),('dataGranularity','1d')]:
        if meta.get(key)!=value:raise ValueError('metadata '+key)
    stamps=pd.to_datetime(result.get('timestamp',[]),unit='s',utc=True);dates=stamps.tz_convert('America/New_York').tz_localize(None).normalize()
    if not dates.is_unique:raise ValueError('duplicate dates')
    pd.testing.assert_index_equal(dates.rename(None),calendar.rename(None))
    quotes=result.get('indicators',{}).get('quote',[]);adj=result.get('indicators',{}).get('adjclose',[])
    if len(quotes)!=1 or len(adj)!=1:raise ValueError('missing quote/adjusted block')
    frame=pd.DataFrame({f'quote_{x}':quotes[0].get(x,[]) for x in ['open','high','low','close']},index=dates)
    frame['volume']=quotes[0].get('volume',[]);frame['vendor_adj_close']=adj[0].get('adjclose',[])
    values=frame.to_numpy(float)
    if not np.isfinite(values).all() or not frame.drop(columns='volume').gt(0).all().all() or not frame.volume.ge(0).all():raise ValueError('nonfinite/invalid quotes')
    if not frame.quote_high.ge(frame[['quote_open','quote_close']].max(axis=1)).all() or not frame.quote_low.le(frame[['quote_open','quote_close']].min(axis=1)).all():raise ValueError('OHLC inequality')
    events=result.get('events')
    if not isinstance(events,dict) or not events.get('dividends'):raise ValueError('missing reported dividends/events')
    actions=[]
    for kind in ['dividends','splits','capitalGains']:
        for key,event in events.get(kind,{}).items():
            stamp=int(event['date']);day=pd.Timestamp(stamp,unit='s',tz='UTC').tz_convert('America/New_York').tz_localize(None).normalize()
            if int(key)!=stamp or day not in calendar:raise ValueError('action date/key/calendar mismatch')
            fields=['numerator','denominator'] if kind=='splits' else ['amount']
            v=np.array([event[f] for f in fields],float)
            if not np.isfinite(v).all() or (v<=0).any() if kind=='splits' else not np.isfinite(v).all() or (v<0).any():raise ValueError('invalid action amount/ratio')
            actions.append({'symbol':symbol,'kind':kind,'date':day,**event})
    frame.index.name='datetime';frame['instrument']=symbol
    return frame.reset_index(),pd.DataFrame(actions),{k:meta.get(k) for k in ['symbol','currency','exchangeTimezoneName','dataGranularity','firstTradeDate']}
