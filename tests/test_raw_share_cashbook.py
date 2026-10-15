from copy import deepcopy
from decimal import Decimal
import pytest
from src.backtest.raw_share_cashbook import new_unit_book,apply_unit_event


def test_cash_conservation_duplicate_conflict_and_split_mark_freshness():
    state=new_unit_book('SOXL',shares=100,initial_at='2023-03-20T09:47:00-04:00')
    def event(eid,kind,at,**extra):return {'event_id':eid,'symbol':'SOXL','kind':kind,'effective_at':at,**extra}
    mark=event('mark','mark','2023-03-20T09:47:00-04:00',raw_price='15')
    state,_=apply_unit_event(state,mark)
    ent=event('ent','dividend_entitlement','2023-03-21T09:30:00-04:00',source_event_id='div',eligible_raw_shares='100',gross_rate_per_original_share='0.02501',declared_available_at='2023-03-29T09:30:00-04:00')
    state,receipt=apply_unit_event(state,ent)
    assert Decimal(receipt['spendable_cash'])==0 and Decimal(receipt['nonspendable_receivable'])==Decimal('2.501') and receipt['NAV_if_same_time_raw_mark'] is None
    pay=event('pay','receivable_to_cash','2023-03-29T09:30:00-04:00',source_event_id='div',declared_available_at='2023-03-29T09:30:00-04:00')
    before_early=deepcopy(state)
    with pytest.raises(ValueError,match='stored paper availability'):
        apply_unit_event(state,{**pay,'effective_at':'2023-03-22T09:30:00-04:00','declared_available_at':'2023-03-22T09:30:00-04:00'})
    assert state==before_early
    state,receipt=apply_unit_event(state,pay)
    assert Decimal(receipt['spendable_cash'])==Decimal('2.501') and not state['receivables'] and receipt['NAV_if_same_time_raw_mark'] is None
    duplicate,_=apply_unit_event(state,ent);assert duplicate==state  # old exact delivery before chronology
    before=deepcopy(state)
    with pytest.raises(ValueError,match='conflicting duplicate'):apply_unit_event(state,{**pay,'declared_available_at':'2023-03-30T09:30:00-04:00'})
    with pytest.raises(ValueError,match='already paid'):apply_unit_event(state,{**pay,'event_id':'renamed_pay'})
    assert state==before
    split=event('split','split','2023-03-30T09:30:00-04:00',source_event_id='split_source',new_shares_per_old_share='3')
    state,receipt=apply_unit_event(state,split)
    assert Decimal(state['raw_shares'])==300 and Decimal(state['spendable_cash'])==Decimal('2.501') and receipt['NAV_if_same_time_raw_mark'] is None
