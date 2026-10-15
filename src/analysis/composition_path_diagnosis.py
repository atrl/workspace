"""Sampled-price drawdown episodes and chronology-aware cost-path divergence."""
import numpy as np
import pandas as pd


def sampled_drawdown(frame, book, *, initial_cash, close_times, tolerance):
    path=book['authority'].reset_index(drop=True);q=np.zeros(2);cash=float(initial_cash)
    peak=float(initial_cash);peak_at=frame.trade_date.iloc[0]+pd.Timedelta(hours=9,minutes=45)
    marks=[];episodes=[];active=None
    for i,row in frame.iterrows():
        known=float(q@row[['known_soxl','known_soxx']].to_numpy(float)+cash)
        q=path.iloc[i][['ending_soxl_quantity','ending_soxx_quantity']].to_numpy(float);cash=float(path.ending_cash.iloc[i])
        close=float(q@row[['close_soxl','close_soxx']].to_numpy(float)+cash)
        daily=[]
        for slot,stamp,value in [('known',row.trade_date+pd.Timedelta(hours=9,minutes=45),known),('close',close_times.loc[row.trade_date],close)]:
            if value>=peak:
                if active is not None:
                    episodes.append({**active,'recovery_at':stamp,'right_censored':False});active=None
                peak=value;peak_at=stamp
            dd=max(0.,1-value/peak);daily.append(dd)
            if dd>0 and active is None:
                active={'peak_at':peak_at,'peak_nav':peak,'first_crossing_at':pd.NaT,'trough_at':stamp,'trough_nav':value,'maximum_drawdown':dd}
            if dd>.30 and pd.isna(active['first_crossing_at']):active['first_crossing_at']=stamp
            if active is not None and dd>active['maximum_drawdown']:
                active.update(trough_at=stamp,trough_nav=value,maximum_drawdown=dd)
            marks.append({'trade_date':row.trade_date,'stamp':stamp,'slot':slot,'nav':value,'peak':peak,'peak_at':peak_at,'drawdown':dd,'breach30':dd>.30})
        if abs(max(daily)-path.monitored_drawdown.iloc[i])>tolerance:
            raise ValueError('sampled marks do not reproduce native monitored drawdown')
    if active is not None:episodes.append({**active,'recovery_at':marks[-1]['stamp'],'right_censored':True})
    table=pd.DataFrame(episodes)
    maximum=float(path.monitored_drawdown.max())
    if abs(table.maximum_drawdown.max()-maximum)>tolerance:raise ValueError('episode maximum differs from source drawdown')
    table['is_cost_maximum']=table.maximum_drawdown.eq(table.maximum_drawdown.max())
    table['breached_30pct']=table.first_crossing_at.notna()
    return pd.DataFrame(marks),table.loc[table.is_cost_maximum|table.breached_30pct].reset_index(drop=True)


def cost_divergence(books, frame, *, tolerances, recovery_spec):
    comparisons={};events=[]
    def known_nav(book):return book['targets'].known_equity.reset_index(drop=True)
    for cost in [2,5]:
        left=books[0]['authority'].reset_index(drop=True);right=books[cost]['authority'].reset_index(drop=True)
        pd.testing.assert_frame_equal(left[['signal_date','trade_date','exit_date']],right[['signal_date','trade_date','exit_date']])
        row=pd.DataFrame({'trade_date':left.trade_date,'known_at':left.trade_date+pd.Timedelta(hours=9,minutes=45),'decision_at':left.trade_date+pd.Timedelta(hours=9,minutes=46)})
        for name,book,path in [('zero',books[0],left),('other',books[cost],right)]:
            target=book['targets'].reset_index(drop=True)
            for col in ['risk_state','risk_clock','current_beta_cap','monitored_drawdown','ending_soxl_quantity','ending_soxx_quantity','ending_cash','broker_and_external_fees_usd','slippage_reserve_usd','strategy_return','equity']:
                row[name+'_'+col]=path[col]
            row[name+'_known_nav']=known_nav(book);row[name+'_known_drawdown']=target.known_drawdown
            days=recovery_spec['trailing_loss_days']
            row[name+'_trailing_known_return']=known_nav(book).pct_change().add(1).rolling(days).apply(np.prod,raw=True)-1
            row[name+'_trailing_loss_triggered']=row[name+'_trailing_known_return'].le(-recovery_spec['trailing_loss_trigger'])
            row[name+'_emergency_triggered']=target.known_drawdown.ge(recovery_spec['emergency_drawdown'])
            row[name+'_requested_nominal_beta']=3*target.soxl_weight+target.soxx_weight
            fees=path.broker_and_external_fees_usd+path.slippage_reserve_usd
            row[name+'_same_day_fees_usd']=fees
            row[name+'_prior_cumulative_fees_usd']=fees.cumsum().shift(1,fill_value=0.)
        tests={'fee':(row.zero_same_day_fees_usd-row.other_same_day_fees_usd).abs()>tolerances['fees_usd_absolute'],
            'known_nav':(row.zero_known_nav-row.other_known_nav).abs()/row.zero_known_nav>tolerances['nav_relative'],
            'state_clock':left.risk_state.ne(right.risk_state)|left.risk_clock.ne(right.risk_clock),
            'cap':(left.current_beta_cap-right.current_beta_cap).abs()>tolerances['nominal_beta_cap_absolute']}
        for kind,mask in tests.items():
            first=np.flatnonzero(mask.to_numpy())
            if len(first):events.append({'comparison':f'0_vs_{cost}','event':kind,'row_number':int(first[0]),**row.iloc[first[0]].to_dict()})
            else:events.append({'comparison':f'0_vs_{cost}','event':kind,'row_number':None})
        row['relative_log_daily']=np.log1p(left.strategy_return)-np.log1p(right.strategy_return)
        comparisons[f'0_vs_{cost}']=row
    return pd.DataFrame(events),comparisons
