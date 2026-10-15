"""Frozen native holdings revaluation; OHLC envelopes are not realized extrema."""
import numpy as np
import pandas as pd


def holding_events(book,frame,quotes,*,initial_cash,tolerance,fee_tolerance):
    records=[];checks=[];previous=np.array([0.,0.,float(initial_cash)])
    orders=book['orders'];native=book['native_orders']
    if len(orders)!=len(native):raise ValueError('native fee rows differ')
    np.testing.assert_allclose(orders.total,native.total,rtol=0,atol=fee_tolerance)
    for stamp,position in sorted(book['native_positions'].items()):
        stamp=pd.Timestamp(stamp);state=np.array([position.get_stock_amount(s) if position.check_stock(s) else 0. for s in ['SOXL','SOXX']]+[float(position.get_cash())])
        if (state<0).any():raise ValueError('bound requires long-only funded native holdings')
        group=orders.loc[orders.stamp.eq(stamp)]
        expected=previous.copy()
        for order in group.itertuples():
            j={'SOXL':0,'SOXX':1}[order.symbol];amount=(-1 if order.sell else 1)*order.adjusted_amount
            expected[j]+=amount;expected[2]-=amount*order.adjusted_price+order.total
        if len(group):
            price=quotes.loc[stamp+pd.Timedelta(minutes=1),['SOXL_open','SOXX_open']].to_numpy(float)
            nav=float(previous[:2]@price+previous[2]);error=max(abs(state[2]-expected[2]),float(np.max(abs(state[:2]-expected[:2])*price)))/max(1.,nav)
            if error>tolerance:raise ValueError('native holdings do not reconcile fixed orders/fees')
            records.append({'stamp':stamp,'soxl':state[0],'soxx':state[1],'cash':state[2]})
            checks.append({'stamp':stamp,'kind':'fixed_fill','relative_error':error})
        else:
            np.testing.assert_allclose(state,previous,rtol=0,atol=fee_tolerance)
        previous=state
    events=pd.DataFrame(records).set_index('stamp').sort_index()
    path=book['authority'].reset_index(drop=True)
    for i,row in path.iterrows():
        state=events.loc[:row.trade_date+pd.Timedelta(hours=23,minutes=59)].iloc[-1]
        actual=state[['soxl','soxx','cash']].to_numpy(float);expected=row[['ending_soxl_quantity','ending_soxx_quantity','ending_cash']].to_numpy(float)
        price=frame.iloc[i][['soxl_exit_open','soxx_exit_open']].to_numpy(float)
        funding_error=max(abs(actual[2]-expected[2]),float(np.max(abs(actual[:2]-expected[:2])*price)))/max(1.,row.equity*initial_cash)
        if funding_error>tolerance:raise ValueError('native daily quantities/cash mismatch')
        nav=float(actual[:2]@price+actual[2]);error=abs(nav-row.equity*initial_cash)/max(1.,nav)
        if error>tolerance:raise ValueError('native daily terminal funding mismatch')
        checks.append({'stamp':row.exit_date+pd.Timedelta(hours=9,minutes=47),'kind':'daily_exit','relative_error':error})
    return events,pd.DataFrame(checks)


def value_path(quotes,events,known,*,start,end,initial_cash):
    start,end=pd.Timestamp(start).tz_localize(None),pd.Timestamp(end).tz_localize(None)
    times=pd.DatetimeIndex(quotes.index);starts=times-pd.Timedelta(minutes=1)
    stamps=events.index.to_numpy(dtype='datetime64[ns]');states=np.vstack([[0.,0.,initial_cash],events[['soxl','soxx','cash']].to_numpy(float)])
    def held(at,side):return states[np.searchsorted(stamps,pd.DatetimeIndex(at).to_numpy(dtype='datetime64[ns]'),side=side)]
    live=(starts>=start)&(times<=end);bar=quotes.loc[live];bar_start=starts[live];state=held(bar_start,'right')
    def nav(data,field,state):return (data[[f'SOXL_{field}',f'SOXX_{field}']].to_numpy(float)*state[:,:2]).sum(axis=1)+state[:,2]
    nav_close=nav(bar,'close',state);low=nav(bar,'low',state);high=nav(bar,'high',state)
    rows=[pd.DataFrame({'stamp':bar.index,'priority':0,'kind':'minute_bar','interval_start':bar_start,'interval_end':bar.index,'nav':nav_close,'lower':low,'upper':high,'soxl':state[:,0],'soxx':state[:,1],'cash':state[:,2]})]
    special=(starts>=start)&(starts<=end)&((starts.time==pd.Timestamp('09:30').time())|starts.isin(events.index)|starts.isin([start,end]))
    opening=quotes.loc[special];at=starts[special]
    for side,priority,kind in [('left',1,'open_before_fill'),('right',2,'open_after_fill')]:
        state=held(at,side);values=nav(opening,'open',state)
        rows.append(pd.DataFrame({'stamp':at,'priority':priority,'kind':kind,'interval_start':at,'interval_end':at,'nav':values,'lower':values,'upper':values,'soxl':state[:,0],'soxx':state[:,1],'cash':state[:,2]}))
    # Explicit original09:45 marks make observed sampling nested in parent evidence.
    at=pd.DatetimeIndex(known.index);mask=(at>=start)&(at<=end);at=at[mask];state=held(at,'left')
    values=(known.loc[at,['SOXL','SOXX']].to_numpy(float)*state[:,:2]).sum(axis=1)+state[:,2]
    rows.append(pd.DataFrame({'stamp':at,'priority':1,'kind':'parent_known','interval_start':at,'interval_end':at,'nav':values,'lower':values,'upper':values,'soxl':state[:,0],'soxx':state[:,1],'cash':state[:,2]}))
    rows.append(pd.DataFrame([{'stamp':start,'priority':-1,'kind':'initial_cash','interval_start':start,'interval_end':start,'nav':initial_cash,'lower':initial_cash,'upper':initial_cash,'soxl':0.,'soxx':0.,'cash':initial_cash}]))
    result=pd.concat(rows,ignore_index=True).sort_values(['stamp','priority'],kind='stable').reset_index(drop=True)
    if not np.isfinite(result[['nav','lower','upper']]).all().all() or (result.lower<=0).any():raise ValueError('nonpositive/nonfinite valuation')
    result['observed_peak']=result.nav.cummax();result['upper_peak']=result.upper.cummax()
    result['observed_drawdown']=1-result.nav/result.observed_peak;result['drawdown_upper_bound']=1-result.lower/result.upper_peak
    if (result.observed_drawdown>result.drawdown_upper_bound+1e-12).any():raise ValueError('OHLC envelope does not enclose observed marks')
    summary={};episodes=[]
    for name,column,value_column in [('observed','observed_drawdown','nav'),('upper_bound','drawdown_upper_bound','lower')]:
        trough=int(result[column].to_numpy().argmax());peakcol='nav' if name=='observed' else 'upper';peak=int(result.loc[:trough,peakcol].to_numpy().argmax())
        summary[name+'_drawdown']=float(result[column].iloc[trough])
        episodes.append({'method':name,'drawdown':float(result[column].iloc[trough]),'peak_value':float(result[peakcol].iloc[peak]),'trough_value':float(result[value_column].iloc[trough]),
            'peak_mark_kind':result.kind.iloc[peak],'peak_interval_start':result.interval_start.iloc[peak],'peak_interval_end':result.interval_end.iloc[peak],
            'trough_mark_kind':result.kind.iloc[trough],'trough_interval_start':result.interval_start.iloc[trough],'trough_interval_end':result.interval_end.iloc[trough],
            'extrema_interpretation':'enclosing OHLC intervals; not simultaneous realized extrema' if name=='upper_bound' else 'observed source marks'})
    summary.update(mark_count=len(result),minute_bar_count=len(bar),terminal_nav=float(result.nav.iloc[-1]))
    summary['qualification']='observed_source_mark_breach' if summary['observed_drawdown']>.30 else ('bounded_RTH_source_risk_within30_under_assumptions' if summary['upper_bound_drawdown']<=.30 else 'unresolved_intraminute_ordering')
    return result,summary,pd.DataFrame(episodes)
