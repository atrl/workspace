"""Pure raw-share paper income transitions; USD is an explicit research assumption."""
from copy import deepcopy
from decimal import Decimal,InvalidOperation
import hashlib,json
import pandas as pd


def dec(value):
    try:value=Decimal(str(value))
    except InvalidOperation as error:raise ValueError('invalid decimal amount') from error
    if not value.is_finite():raise ValueError('nonfinite decimal amount')
    return value

def _stamp(value):
    stamp=pd.Timestamp(value)
    if stamp.tzinfo is None:raise ValueError('event timestamp must be timezone aware')
    return stamp.tz_convert('UTC')

def new_unit_book(symbol,*,shares,initial_at,cash='0'):
    if dec(shares)<0 or dec(shares)%1 or dec(cash)<0:raise ValueError('invalid whole-share initial state')
    return {'symbol':symbol,'raw_shares':str(dec(shares)),'spendable_cash':str(dec(cash)),
        'initial_cash':str(dec(cash)),'receivables':{},'recognized_gross_income':'0','paid_ids':[],
        'applied_split_ids':[],'processed_events':{},'initial_at':str(_stamp(initial_at)),
        'last_event_at':None,'raw_mark':None,'raw_mark_at':None,'currency':'USD',
        'currency_evidence':'explicit_gross_pretax_paper_assumption_not_source_certificate'}

def _receivable(state):return sum((dec(x['amount']) for x in state['receivables'].values()),Decimal(0))

def apply_unit_event(state,event):
    """Return a new state and receipt. Errors never mutate the supplied state."""
    digest=hashlib.sha256(json.dumps(event,sort_keys=True,allow_nan=False).encode()).hexdigest();eid=event['event_id']
    if eid in state['processed_events']:
        if state['processed_events'][eid]!=digest:raise ValueError('conflicting duplicate event payload')
        return deepcopy(state),{'status':'already_recorded','event_id':eid,'payload_sha256':digest}
    stamp=_stamp(event['effective_at'])
    if stamp<_stamp(state['initial_at']) or (state['last_event_at'] and stamp<_stamp(state['last_event_at'])):raise ValueError('out-of-order new event')
    if event['symbol']!=state['symbol']:raise ValueError('wrong instrument')
    result=deepcopy(state);kind=event['kind'];before_money=dec(state['spendable_cash'])+_receivable(state)
    source=event.get('source_event_id')
    if kind!='mark' and (not isinstance(source,str) or not source.strip()):raise ValueError('source_event_id required')
    if kind=='mark':
        price=dec(event['raw_price'])
        if not price.is_finite() or price<=0:raise ValueError('invalid raw price')
        result['raw_mark']=str(price);result['raw_mark_at']=str(stamp)
    elif kind=='split':
        if source in result['applied_split_ids']:raise ValueError('duplicate economic split')
        ratio=dec(event['new_shares_per_old_share']);shares=dec(result['raw_shares'])*ratio
        if ratio<=0 or shares%1:raise ValueError('unsupported fractional split entitlement')
        result['raw_shares']=str(shares);result['applied_split_ids'].append(source)
        result['raw_mark']=None;result['raw_mark_at']=None
    elif kind=='dividend_entitlement':
        if source in result['receivables'] or source in result['paid_ids']:raise ValueError('duplicate economic entitlement')
        if dec(event['eligible_raw_shares'])!=dec(result['raw_shares']):raise ValueError('eligible holdings snapshot differs')
        rate=dec(event['gross_rate_per_original_share']);amount=dec(event['eligible_raw_shares'])*rate
        if rate<0:raise ValueError('negative cash distribution unsupported')
        available=_stamp(event['declared_available_at'])
        if available<stamp:raise ValueError('payment schedule precedes entitlement')
        result['receivables'][source]={'amount':str(amount),'available_at':str(available)}
        result['recognized_gross_income']=str(dec(result['recognized_gross_income'])+amount)
    elif kind=='receivable_to_cash':
        if source in result['paid_ids'] or source not in result['receivables']:raise ValueError('missing or already paid entitlement')
        entitlement=result['receivables'][source]
        if stamp!=_stamp(entitlement['available_at']) or _stamp(event['declared_available_at'])!=_stamp(entitlement['available_at']):raise ValueError('payment violates stored paper availability')
        result['spendable_cash']=str(dec(result['spendable_cash'])+dec(result['receivables'].pop(source)['amount']))
        result['paid_ids'].append(source)
    else:raise ValueError('unsupported unit event')
    cash=dec(result['spendable_cash']);receivable=_receivable(result);shares=dec(result['raw_shares'])
    if min(cash,receivable,shares)<0:raise ValueError('negative unit book state')
    if cash+receivable-dec(result['recognized_gross_income'])!=dec(result['initial_cash']):raise ValueError('cash and receivable conservation failed')
    if kind in ['split','receivable_to_cash','mark'] and cash+receivable!=before_money:raise ValueError('non-income event changed money total')
    nav=shares*dec(result['raw_mark'])+cash+receivable if result['raw_mark_at']==str(stamp) else None
    result['last_event_at']=str(stamp);result['processed_events'][eid]=digest
    receipt={'status':'processed','event_id':eid,'payload_sha256':digest,'kind':kind,'effective_at':str(stamp),
        'raw_shares':str(shares),'spendable_cash':str(cash),'nonspendable_receivable':str(receivable),
        'recognized_gross_income':result['recognized_gross_income'],'NAV_if_same_time_raw_mark':str(nav) if nav is not None else None,
        'cash_plus_receivable_conserved_except_declared_income':True,'currency_assumption':'USD gross pretax paper only'}
    return result,receipt
