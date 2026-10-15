"""Frozen policy decisions on physical shares, spendable cash and dividend receivables."""
from copy import deepcopy
from decimal import Decimal
import numpy as np
import pandas as pd
from src.backtest.ibkr_tiered import TieredFees
from src.workflow.stage1_open_action_value import prepare_on_policy_open,prepare_on_policy_recovery
from src.workflow.intraday_recovery_direct_allocation import project_legal_weights
from src.model.nominal_beta_composition import cancel_leverage_cash
SYMBOLS=('SOXL','SOXX')


def make_schedule(cash_events,split_events,sessions,start,end):
    actions={}
    def add(day,item):actions.setdefault(pd.Timestamp(day).normalize(),[]).append(item)
    for r in split_events.loc[split_events.instrument.isin(SYMBOLS)].to_dict('records'):
        date=pd.Timestamp(r['effective_trading_date'])
        if pd.Timestamp(start)<=date<=pd.Timestamp(end):add(date,{'kind':'split','symbol':r['instrument'],'id':r['event_id'],'ratio':int(r['new_shares_per_old_share'])})
    for r in cash_events.loc[cash_events.instrument.isin(SYMBOLS)].to_dict('records'):
        date=pd.Timestamp(r['ex_date']);pay=pd.Timestamp(r['pay_date'])
        if not pd.Timestamp(start)<=date<=pd.Timestamp(end):continue
        following=sessions[sessions>pay];available=following[0] if len(following) else None
        item={'kind':'entitle','symbol':r['instrument'],'id':r['event_id'],'rate':str(r['original_total_decimal']),'pay_date':str(pay.date()),'available_date':str(available.date()) if available is not None else None}
        add(date,item)
        if available is not None and available<=pd.Timestamp(end):add(available,{'kind':'pay','symbol':r['instrument'],'id':r['event_id']})
    for day,items in actions.items():
        if day not in sessions:raise ValueError('corporate action outside native sessions')
        for symbol in SYMBOLS:
            kinds={x['kind'] for x in items if x['symbol']==symbol}
            if 'split' in kinds and 'entitle' in kinds:raise ValueError('ambiguous same-day split/dividend share ordering')
        items.sort(key=lambda x:({'split':0,'entitle':1,'pay':2}[x['kind']],x['symbol'],x['id']))
    return actions


class IncomeLedger:
    def __init__(self):self.receivables={};self.applied=set();self.receipts=[]
    @property
    def value(self):return float(sum((x['amount'] for x in self.receivables.values()),Decimal(0)))
    def apply(self,day,q,cash,schedule):
        q=np.asarray(q,float).copy();day=pd.Timestamp(day).normalize();new=[]
        for item in schedule.get(day,[]):
            key=(item['kind'],item['id'])
            if key in self.applied:continue
            i=SYMBOLS.index(item['symbol']);before=cash+self.value;income=Decimal(0)
            if item['kind']=='split':
                amount=q[i]*item['ratio']
                if abs(amount-round(amount))>1e-7:raise ValueError('fractional split cash-in-lieu unsupported')
                q[i]=round(amount)
            elif item['kind']=='entitle':
                if item['id'] in self.receivables:raise ValueError('duplicate entitlement')
                if abs(q[i]-round(q[i]))>1e-7:raise ValueError('noninteger physical entitlement')
                income=Decimal(int(round(q[i])))*Decimal(item['rate'])
                self.receivables[item['id']]={'amount':income,'available_date':item['available_date']}
            else:
                record=self.receivables.get(item['id'])
                if record is None or record['available_date']!=str(day.date()):raise ValueError('payment differs from stored entitlement schedule')
                cash+=float(self.receivables.pop(item['id'])['amount'])
            if not np.isclose(cash+self.value,before+float(income),rtol=1e-12,atol=1e-7):raise ValueError('income conservation')
            self.applied.add(key);receipt={'date':day,**item,'eligible_or_after_shares':float(q[i]),'recognized_income':str(income),'cash_after':cash,'receivable_after':self.value}
            self.receipts.append(receipt);new.append(receipt)
        return q,float(cash),new


def settle_raw(q,cash,receivable,*,desired,prices,stamp,fees):
    held=np.asarray(q,float).copy();prices=np.asarray(prices,float);nav=float(held@prices+cash+receivable)
    if desired is None:return held,float(cash),[]
    target=np.asarray(desired,float)
    if target.shape!=(3,) or target.min()<0 or not np.isclose(target.sum(),1):raise ValueError('invalid target')
    delta=target[:2]*nav-held*prices;changes=np.sign(delta)*np.floor(abs(delta)/prices+1e-10);orders=[]
    for sell in [True,False]:
        for i,symbol in enumerate(SYMBOLS):
            change=int(changes[i])
            if not change or (change<0)!=sell:continue
            amount=abs(change);quote=fees.quote(shares=amount,raw_price=prices[i],sell=sell,stamp=stamp)
            if not sell:
                while amount and quote['notional']+quote['total']>cash+1e-8:
                    amount=max(0,amount-max(1,int(np.ceil((quote['notional']+quote['total']-cash)/prices[i]))))
                    quote=fees.quote(shares=amount,raw_price=prices[i],sell=False,stamp=stamp)
                if not amount:continue
            direction=-1 if sell else 1;held[i]+=direction*amount;cash-=direction*quote['notional']+quote['total']
            fees.commit(amount,stamp=stamp,cap_applied=quote['cap_applied'])
            orders.append({'symbol':symbol,'sell':sell,'raw_shares':amount,'adjusted_amount':amount,'raw_price':prices[i],'adjusted_price':prices[i],'stamp':pd.Timestamp(stamp),**quote})
    if held.min()<-1e-8 or cash<-1e-7:raise ValueError('unfunded physical trade')
    return held,float(cash),orders


def replay_raw_package(frame,*,point_prices,schedule,plan,controller,recovery_spec,action_spec,selection,benchmark=False):
    costs=plan['slippage_bps'];ledgers={c:{'q':np.zeros(2),'cash':float(plan['initial_cash']),'income':IncomeLedger(),'fees':TieredFees(plan['tariff'],slippage_bps=c),'peak':float(plan['initial_cash']),'previous_known':None,'history':[],'state':0,'clock':0,'records':[],'orders':[],'points':[],'targets':[],'pending':None,'endpoint':{}} for c in costs};trace=[]
    rows={pd.Timestamp(r.trade_date):r for _,r in frame.iterrows()};terminal=pd.Timestamp(frame.exit_date.iloc[-1]);days=[*rows,terminal]
    def price(day,clock):return point_prices.loc[day+pd.Timedelta(clock),list(SYMBOLS)].to_numpy(float)
    def point(ledger,stamp,prices):
        r=ledger['income'].value;v=float(ledger['q']@prices+ledger['cash']+r)
        ledger['points'].append({'stamp':stamp,'q_soxl':ledger['q'][0],'q_soxx':ledger['q'][1],'cash':ledger['cash'],'receivable':r,'nav':v});return v
    for day in days:
        known=price(day,'9h45min');opening=price(day,'9h47min');snapshots={}
        for cost,l in ledgers.items():
            l['q'],l['cash'],income=l['income'].apply(day,l['q'],l['cash'],schedule)
            nav=point(l,day+pd.Timedelta('9h45min'),known);l['peak']=max(l['peak'],nav)
            pre=float(l['q']@opening+l['cash']+l['income'].value);l['endpoint'][day+pd.Timedelta('9h47min')]=pre
            if l['pending'] is not None:
                pending=l['pending'];ending=np.r_[l['q']*opening,l['cash']+l['income'].value]/pre
                pending.update(strategy_return=pre/pending.pop('start_nav')-1,equity=pre/plan['initial_cash'],native_expected_nav=pre,ending_soxl_quantity=l['q'][0],ending_soxx_quantity=l['q'][1],ending_cash=l['cash']+l['income'].value,ending_spendable_cash=l['cash'],ending_receivable=l['income'].value,**dict(zip(['ending_soxl_weight','ending_soxx_weight','ending_cash_weight'],ending)))
                l['records'].append(pending);l['pending']=None
            if day==terminal:point(l,day+pd.Timedelta('9h47min'),opening);continue
            row=rows[day]
            if l['previous_known'] is not None:l['history'].append(nav/l['previous_known']-1)
            l['previous_known']=nav
            if benchmark:opened={'risk_state':0,'risk_clock':0,'drawdown':max(0.,1-nav/l['peak']),'current_beta_cap':1.}
            else:opened=prepare_on_policy_open(row,pretrade_weights=np.r_[l['q']*known,l['cash']+l['income'].value]/nav,equity=nav,peak=l['peak'],risk_state=l['state'],risk_clock=l['clock'],history=l['history'],recovery_spec=recovery_spec,model=None,action_spec=action_spec)
            l['state'],l['clock']=opened['risk_state'],opened['risk_clock'];snapshots[cost]={'opened':opened,'known_nav':nav,'pre':pre,'local_cap':opened['current_beta_cap']}
        if day==terminal:break
        row=rows[day];common_cap=min(s['local_cap'] for s in snapshots.values())
        before=np.array([0.,1.,0.]) if benchmark else project_legal_weights(float(row.soxl_weight),float(row.soxx_weight),common_cap,minimum_cash_weight=recovery_spec.minimum_cash_weight,maximum_soxl_weight=recovery_spec.maximum_soxl_weight)
        target=(before if not ledgers[costs[0]]['records'] else None) if benchmark else cancel_leverage_cash(before)
        session_orders={}
        for cost,l in ledgers.items():
            s=snapshots[cost];s['recovery_open_state']={**s['opened'],'current_beta_cap':common_cap,'desired_open_weights':before.copy(),'open_weights':target.copy() if target is not None else before.copy()}
            l['targets'].append({'signal_date':row.signal_date,'trade_date':day,'decision_at':day+pd.Timedelta('9h46min'),'known_equity':s['known_nav'],'known_drawdown':s['opened']['drawdown'],**dict(zip(['soxl_weight','soxx_weight','cash_weight'],target if target is not None else before))})
            l['q'],l['cash'],orders=settle_raw(l['q'],l['cash'],l['income'].value,desired=target,prices=opening,stamp=day+pd.Timedelta('9h47min'),fees=l['fees']);session_orders[cost]=[{**o,'slot':'open','trade_date':day} for o in orders];point(l,day+pd.Timedelta('9h47min'),opening)
        votes={c:{'recovery_action':False,'recovery_probability':np.nan} if benchmark else prepare_on_policy_recovery(row,open_state=snapshots[c]['recovery_open_state'],recovery_spec=recovery_spec,model=controller['frozen_recovery_model'],selected_model=selection) for c in costs}
        recover=all(v['recovery_action'] for v in votes.values());recovery_target=None
        if recover:
            before_rec=votes[costs[0]]['target_weights']
            for vote in votes.values():np.testing.assert_allclose(vote['target_weights'],before_rec,rtol=0,atol=1e-12)
            recovery_target=cancel_leverage_cash(before_rec)
        recovery_prices=price(day,'10h32min');day_points=point_prices.loc[point_prices.index.normalize()==day];close_at=day_points.index[-1];closing=day_points.iloc[-1].to_numpy(float)
        for cost,l in ledgers.items():
            s=snapshots[cost]
            if recover:
                l['q'],l['cash'],orders=settle_raw(l['q'],l['cash'],l['income'].value,desired=recovery_target,prices=recovery_prices,stamp=day+pd.Timedelta('10h32min'),fees=l['fees']);session_orders[cost]+=[{**o,'slot':'recovery','trade_date':day} for o in orders]
            point(l,day+pd.Timedelta('10h32min'),recovery_prices);close=point(l,close_at,closing);l['peak']=max(l['peak'],close)
            orders=session_orders[cost];total=sum(o['total'] for o in orders);slippage=sum(o['slippage'] for o in orders);l['orders']+=orders
            l['pending']={'signal_date':row.signal_date,'trade_date':day,'exit_date':row.exit_date,'start_nav':s['pre'],'turnover':sum(o['notional'] for o in orders)/s['pre'],'fees':total/s['pre'],'broker_and_external_fees_usd':total-slippage,'slippage_reserve_usd':slippage,'portfolio_drawdown':s['opened']['drawdown'],'monitored_drawdown':max(s['opened']['drawdown'],max(0.,1-close/l['peak'])),'risk_state':l['state'],'risk_clock':l['clock'],'current_beta_cap':common_cap,'execute_open_target':target is not None,'recovery_action':recover}
            trace.append({'trade_date':day,'slippage_bps':cost,'known_nav':s['known_nav'],'risk_state':l['state'],'risk_clock':l['clock'],'local_cap':s['local_cap'],'common_cap':common_cap,'spendable_cash':l['cash'],'receivable':l['income'].value,'recovery_probability':votes[cost]['recovery_probability'],'recovery_consensus':recover})
    return {c:{'replay':pd.DataFrame(l['records']),'orders':pd.DataFrame(l['orders']),'points':pd.DataFrame(l['points']),'targets':pd.DataFrame(l['targets']),'corporate_receipts':pd.DataFrame(l['income'].receipts),'monthly_fee_shares':l['fees'].monthly_shares,'endpoints':l['endpoint']} for c,l in ledgers.items()},pd.DataFrame(trace)
