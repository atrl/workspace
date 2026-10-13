"""Continuous funded event-component ledger with exact raw-share Tiered fees."""
import numpy as np
import pandas as pd
from src.backtest.ibkr_tiered import TieredFees,settle_tiered_event


def replay_events(frame,windows,*,weights,plan,slippage_bps,benchmark=False):
    entries={pd.Timestamp(r.entry_date):r.event_id for r in windows.itertuples()};exits={pd.Timestamp(r.exit_date):r.event_id for r in windows.itertuples()}
    q=np.zeros(2);cash=peak=float(plan['initial_cash']);fees=TieredFees(plan['tariff'],slippage_bps=slippage_bps);rows=[];orders=[];events=[];starts={}
    for i,(_,row) in enumerate(frame.iterrows()):
        day=row.trade_date;known=row[['known_soxl','known_soxx']].to_numpy(float);opening=row[['soxl_open','soxx_open']].to_numpy(float);closing=row[['close_soxl','close_soxx']].to_numpy(float);next_price=row[['soxl_exit_open','soxx_exit_open']].to_numpy(float)
        known_nav=float(q@known+cash);peak=max(peak,known_nav);drawdown=1-known_nav/peak;nav=float(q@opening+cash)
        desired=np.array([0.,1.,0.]) if benchmark and i==0 else (weights if not benchmark and day in entries else None)
        if not benchmark and day in entries:starts[entries[day]]=nav
        trade=settle_tiered_event(q,cash,desired=desired,adjusted_prices=opening,raw_prices=row[['raw_soxl_open','raw_soxx_open']],stamp=day+pd.Timedelta(hours=9,minutes=47),fees=fees)
        q,cash=trade['quantities'],trade['cash'];session=[{**o,'slot':'open','trade_date':day} for o in trade['orders']]
        if not benchmark and day in exits:
            mid=row[['soxl_event_exit','soxx_event_exit']].to_numpy(float)
            trade=settle_tiered_event(q,cash,desired=np.array(plan['outside_event_target']),adjusted_prices=mid,raw_prices=row[['raw_soxl_event_exit','raw_soxx_event_exit']],stamp=day+pd.Timedelta(hours=13,minutes=59),fees=fees)
            q,cash=trade['quantities'],trade['cash'];session += [{**o,'slot':'event_exit','trade_date':day} for o in trade['orders']]
            mid_nav=float(q@mid+cash);peak=max(peak,mid_nav);drawdown=max(drawdown,1-mid_nav/peak)
            event_id=exits[day];events.append({'event_id':event_id,'entry_nav_before_cost':starts[event_id],'exit_nav_after_cost':mid_nav,'event_net_log_return':np.log(mid_nav/starts[event_id])})
        close_nav=float(q@closing+cash);peak=max(peak,close_nav);monitored=max(drawdown,1-close_nav/peak);ending=float(q@next_price+cash);w=np.r_[q*next_price,cash]/ending
        total=sum(o['total'] for o in session);slipped=sum(o['slippage'] for o in session);orders.extend(session)
        rows.append({'signal_date':row.signal_date,'trade_date':day,'exit_date':row.exit_date,'strategy_return':ending/nav-1,'equity':ending/plan['initial_cash'],'native_expected_nav':ending,'turnover':sum(o['notional'] for o in session)/nav,'fees':total/nav,'broker_and_external_fees_usd':total-slipped,'slippage_reserve_usd':slipped,'portfolio_drawdown':drawdown,'monitored_drawdown':monitored,'risk_state':0,'risk_clock':0,'execute_open_target':desired is not None,'recovery_action':False,'ending_soxl_quantity':q[0],'ending_soxx_quantity':q[1],'ending_cash':cash,'ending_soxl_weight':w[0],'ending_soxx_weight':w[1],'ending_cash_weight':w[2]})
    if not benchmark and len(events)!=len(windows):raise ValueError('event missing from full account')
    return {'replay':pd.DataFrame(rows),'orders':pd.DataFrame(orders),'events':pd.DataFrame(events),'monthly_fee_shares':fees.monthly_shares}
