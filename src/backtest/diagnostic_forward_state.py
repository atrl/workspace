"""Event-sourced paper books: clone every book, then commit a complete slot."""
from copy import deepcopy
from dataclasses import dataclass,field
import numpy as np
import pandas as pd
from src.backtest.ibkr_tiered import TieredFees
from src.backtest.raw_corporate_policy import IncomeLedger,settle_raw
from src.model.nominal_beta_composition import cancel_leverage_cash
from src.workflow.intraday_recovery_direct_allocation import project_legal_weights
from src.workflow.stage1_open_action_value import prepare_on_policy_open,prepare_on_policy_recovery
from src.data.diagnostic_forward_inputs import digest,utc,check_clock,validate_quote

FAMILIES=('reference27baa','index_candidate','index_control','SOXX_buy_hold')
COSTS=(0,2,5)

@dataclass
class Book:
    family:str
    cost:int
    fees:object
    q:np.ndarray=field(default_factory=lambda:np.zeros(2))
    cash:float=100000.
    income:object=field(default_factory=IncomeLedger)
    peak:float=100000.
    previous_known:float|None=None
    history:list=field(default_factory=list)
    risk_state:int=0
    risk_clock:int=0
    bought_benchmark:bool=False
    open_snapshot:dict|None=None
    def nav(self,prices):
        prices=np.asarray(prices,float)
        if prices.shape!=(2,) or not np.isfinite(prices).all() or (prices<=0).any():raise ValueError('positive finite prices required')
        if not np.isfinite(self.q).all() or (self.q<0).any() or not np.isfinite(self.cash) or self.cash<0 or not np.isfinite(self.income.value) or self.income.value<0:raise ValueError('invalid book funds/quantities')
        value=float(self.q@prices+self.cash+self.income.value)
        if not np.isfinite(value) or value<=0:raise ValueError('nonpositive/nonfinite NAV')
        return value

class ForwardPaperState:
    def __init__(self,tariff,*,initial_cash=100000.,fee_valid_through='2026-12-31'):
        if not np.isfinite(initial_cash) or initial_cash<=0:raise ValueError('initial cash')
        self.books={f'{family}_{cost}':Book(family,cost,TieredFees(tariff,slippage_bps=cost),cash=float(initial_cash),peak=float(initial_cash)) for family in FAMILIES for cost in COSTS}
        self.intents={};self.committed={};self.events=[];self.ema={f:{'average':None,'accepted':np.array([0.,0.,1.])} for f in FAMILIES[:-1]}
        self.trading_disabled=False;self.unqualified_reasons=[];self.fee_valid_through=pd.Timestamp(fee_valid_through).date();self.last_commit=None
    def snapshot(self):
        return {k:{'q':b.q.tolist(),'cash':b.cash,'R':b.income.value,'receivables':{x:{**v,'amount':str(v['amount'])} for x,v in b.income.receivables.items()},'applied_CA':sorted(b.income.applied),'peak':b.peak,'previous_known':b.previous_known,'history':list(b.history),'risk_state':b.risk_state,'risk_clock':b.risk_clock,'monthly_fees':dict(b.fees.monthly_shares),'fee_tariff':deepcopy(b.fees.tariff),'slippage_bps':b.fees.slippage_bps,'benchmark_bought':b.bought_benchmark,'open_snapshot':deepcopy(b.open_snapshot)} for k,b in self.books.items()}
    def state_digest(self):
        return digest({'books':self.snapshot(),'ema':self.ema,'trading_disabled':self.trading_disabled,'reasons':self.unqualified_reasons,'fee_valid_through':str(self.fee_valid_through),'last_commit':self.last_commit})
    def disable(self,reason):
        self.trading_disabled=True;self.unqualified_reasons.append(str(reason))
    def smooth(self,proposals,daily_open_returns):
        # No future09:47 return: caller must validate the received daily-open basis.
        if set(proposals)!=set(self.ema):raise ValueError('all three model proposals required')
        returns=np.asarray(daily_open_returns,float)
        if returns.shape!=(2,) or not np.isfinite(returns).all() or (returns<=-1).any():raise ValueError('invalid observed opening returns')
        clones=deepcopy(self.ema);out={}
        for family,weights in proposals.items():
            weights=np.asarray(weights,float);self._weights(weights)
            state=clones[family];current=state['accepted']*np.r_[1+returns,1.];current/=current.sum()
            average=weights if state['average'] is None else .5*weights+.5*state['average']
            accepted=average if np.abs(average[:2]-current[:2]).sum()>=.02 else current
            state.update(average=average.copy(),accepted=accepted.copy());out[family]=accepted.copy()
        self.ema=clones;return out
    @staticmethod
    def _weights(weights):
        a=np.asarray(weights,float)
        if a.shape!=(3,) or not np.isfinite(a).all() or (a<0).any() or abs(a.sum()-1)>1e-8:raise ValueError('illegal target simplex')
    def apply_actions(self,session,schedule):
        clones=deepcopy(self.books)
        for book in clones.values():book.q,book.cash,_=book.income.apply(session,book.q,book.cash,schedule)
        self.books=clones
    def open_targets(self,rows,known_prices,*,recovery_spec,action_spec):
        if self.trading_disabled:raise ValueError('future trading disabled')
        clones=deepcopy(self.books);local={};targets={}
        for key,book in clones.items():
            nav=book.nav(known_prices);book.peak=max(book.peak,nav)
            if book.previous_known is not None:book.history.append(nav/book.previous_known-1)
            book.previous_known=nav
            if book.family=='SOXX_buy_hold':opened={'risk_state':0,'risk_clock':0,'drawdown':max(0.,1-nav/book.peak),'current_beta_cap':1.}
            else:opened=prepare_on_policy_open(rows[book.family],pretrade_weights=np.r_[book.q*np.asarray(known_prices),book.cash+book.income.value]/nav,equity=nav,peak=book.peak,risk_state=book.risk_state,risk_clock=book.risk_clock,history=book.history,recovery_spec=recovery_spec,model=None,action_spec=action_spec)
            book.risk_state,book.risk_clock=opened['risk_state'],opened['risk_clock'];local[key]=opened
        for family in FAMILIES:
            keys=[f'{family}_{cost}' for cost in COSTS];cap=min(local[k]['current_beta_cap'] for k in keys)
            if family=='SOXX_buy_hold':before=np.array([0.,1.,0.]);target=None if clones[keys[0]].bought_benchmark else before
            else:
                row=rows[family];before=project_legal_weights(float(row.soxl_weight),float(row.soxx_weight),cap,minimum_cash_weight=recovery_spec.minimum_cash_weight,maximum_soxl_weight=recovery_spec.maximum_soxl_weight);target=cancel_leverage_cash(before)
            for key in keys:targets[key]=None if target is None else target.copy();clones[key].open_snapshot={**local[key],'current_beta_cap':cap,'desired_open_weights':before.copy(),'open_weights':before.copy() if target is None else target.copy()}
        self.books=clones;return targets
    def recovery_targets(self,rows,*,recovery_spec,controller,selection):
        if self.trading_disabled:raise ValueError('future trading disabled')
        votes={};targets={};calls=0
        for family in FAMILIES:
            keys=[f'{family}_{cost}' for cost in COSTS]
            for key in keys:
                book=self.books[key]
                if book.open_snapshot is None:raise ValueError('missing original prefill opening snapshot')
                if family=='SOXX_buy_hold':votes[key]={'recovery_action':False}
                else:
                    votes[key]=prepare_on_policy_recovery(rows[family],open_state=deepcopy(book.open_snapshot),recovery_spec=recovery_spec,model=controller['frozen_recovery_model'],selected_model=selection);calls+=1
            accepted=all(votes[k]['recovery_action'] for k in keys);target=None
            if accepted:
                first=np.asarray(votes[keys[0]]['target_weights'])
                for k in keys:np.testing.assert_allclose(votes[k]['target_weights'],first,rtol=0,atol=1e-12)
                target=cancel_leverage_cash(first)
            for key in keys:targets[key]=None if target is None else target.copy()
        return targets,votes,calls
    def seal(self,event_id,targets,clock,*,slot_at,bindings):
        if self.trading_disabled:raise ValueError('future trading disabled')
        stamp,_=check_clock(clock)
        if stamp>=utc(slot_at):raise ValueError('intent must seal before later quote slot')
        if set(targets)!=set(self.books):raise ValueError('all12 book targets required')
        for target in targets.values():
            if target is not None:self._weights(target)
        data={'event_id':event_id,'targets':{k:None if v is None else np.asarray(v,float).tolist() for k,v in targets.items()},'clock':dict(clock),'slot_at':str(slot_at),'bindings':dict(bindings),'state_sha256':self.state_digest()};identity=digest(data)
        if event_id in self.intents:
            if self.intents[event_id]['sha256']!=identity:raise ValueError('conflicting intent duplicate')
            return deepcopy(self.intents[event_id])
        record={**data,'sha256':identity};self.intents[event_id]=record;return deepcopy(record)
    def execute_slot(self,event_id,quote):
        intent=self.intents[event_id];event_hash=digest({'intent':intent['sha256'],'quote':quote})
        if event_id in self.committed:
            if self.committed[event_id]['event_sha256']!=event_hash:raise ValueError('conflicting delivery duplicate')
            return deepcopy(self.committed[event_id])
        if self.trading_disabled:raise ValueError('future trading disabled')
        try:
            prices=validate_quote(quote,seal=intent['clock'],slot_at=intent['slot_at'])
            if utc(intent['slot_at']).tz_convert('America/New_York').date()>self.fee_valid_through:raise ValueError('fee contract expired')
            if self.state_digest()!=intent['state_sha256']:raise ValueError('state changed after intent seal')
            if self.last_commit is not None and utc(quote['received']['utc'])<=utc(self.last_commit):raise ValueError('nonchronological new fill')
            clones=deepcopy(self.books);reports={}
            for key,book in clones.items():
                book.nav(prices) # validate positive finite NAV BEFORE old settlement.
                book.q,book.cash,orders=settle_raw(book.q,book.cash,book.income.value,desired=intent['targets'][key],prices=prices,stamp=utc(intent['slot_at']).tz_convert('America/New_York').tz_localize(None),fees=book.fees)
                nav=book.nav(prices)
                if book.family=='SOXX_buy_hold' and intent['targets'][key] is not None:book.bought_benchmark=True
                reports[key]={'orders':orders,'nav':nav,'q':book.q.tolist(),'cash':book.cash,'R':book.income.value}
            # Atomic publication: no book or fee counter mutates before all12 succeed.
            self.books=clones;self.last_commit=quote['received']['utc'];event={'event_id':event_id,'event_sha256':event_hash,'intent_sha256':intent['sha256'],'quote_sha256':digest(quote),'reports':reports,'state_sha256':self.state_digest()};self.events.append(event);self.committed[event_id]=event;return deepcopy(event)
        except Exception as error:
            self.disable(type(error).__name__+': '+str(error));raise
    def mark(self,prices,*,update_peak=False):
        clones=deepcopy(self.books);values={}
        for key,book in clones.items():
            values[key]=book.nav(prices)
            if update_peak:book.peak=max(book.peak,values[key])
        self.books=clones;return values
