"""Synthetic clock, future horizon, venue isolation and exact native metadata."""
import pickle

import pandas as pd

from src.workflow.calendar_tom_source import build_date_metadata, metadata_dataset


def test_two_clocks_do_not_backdate_or_turn_asof_tail_into_monthend():
    calendar=pd.bdate_range('2025-12-01','2025-12-12')
    index=pd.DatetimeIndex(['2025-12-08','2025-12-10','2025-12-11','2025-12-12'])
    decisions=pd.to_datetime(['2025-12-09 09:29','2025-12-11 09:29','2025-12-12 09:29',None]).tz_localize('America/New_York')
    timing=pd.DataFrame({'decision_at':decisions,'label_available_at':decisions,'return_available_at':decisions},index=index)
    known=pd.Timestamp('2025-12-10 00:00',tz='America/New_York')
    ledger=pd.DataFrame([
        {'venue':'NYSE','assertion_kind':'weekday_baseline','effective_from':pd.Timestamp('2025-01-01'),
         'effective_until':pd.Timestamp('2025-12-31'),'session_date':pd.NaT,'knowledge_at':known,'document_id':'announced'},
        {'venue':'NYSE','assertion_kind':'weekday_baseline','effective_from':pd.Timestamp('2026-01-01'),
         'effective_until':pd.Timestamp('2026-12-31'),'session_date':pd.NaT,'knowledge_at':known,'document_id':'announced'},
        {'venue':'NYSE','assertion_kind':'planned_closed','effective_from':pd.Timestamp('2026-01-01'),
         'effective_until':pd.Timestamp('2026-01-01'),'session_date':pd.Timestamp('2026-01-01'),
         'knowledge_at':known,'document_id':'announced'},
    ])
    result=build_date_metadata(timing,calendar,ledger,as_of='2025-12-12')
    early=result.loc[(pd.Timestamp('2025-12-08'),'NYSE__original_0929')]
    later=result.loc[(pd.Timestamp('2025-12-10'),'NYSE__original_0929')]
    research=result.loc[(pd.Timestamp('2025-12-10'),'NYSE__research_0946')]
    assert not early.ordinary_horizon_complete  # The later announcement cannot backfill Dec9.
    assert later.ordinary_horizon_complete and later.next_planned_open_after_month_end==pd.Timestamp('2026-01-02')
    assert later.announced_coverage_through==pd.Timestamp('2025-12-31')  # Not observed asof Dec12.
    assert not later.current_open_conditional_0946_assumption and research.current_open_conditional_0946_assumption
    assert not result.current_open_first_receipt_verified.any() and result.current_open_first_receipt_at.isna().all()
    assert not result.historical_PIT_qualified.any()
    assert not result.loc[result.venue.eq('NYSE_ARCA'),'ordinary_horizon_complete'].any()
    assert result.loc[pd.Timestamp('2025-12-12'),'mapped_trade_date'].isna().all()
    assert len(result)==24 and not any('tom' in str(c).lower() for c in result.columns)
    exception=ledger.iloc[2].copy()
    exception['assertion_kind']='temporary_closed'
    exception['session_date']=exception['effective_from']=exception['effective_until']=pd.Timestamp('2026-01-02')
    exception['document_id']='exception'
    exception['knowledge_at']=pd.Timestamp('2025-12-10 12:00',tz='America/New_York')
    updated=build_date_metadata(timing,calendar,pd.concat([ledger,pd.DataFrame([exception])],ignore_index=True),as_of='2025-12-12')
    changed=updated.loc[(pd.Timestamp('2025-12-10'),'NYSE__original_0929')]
    assert changed.next_planned_open_after_month_end==pd.Timestamp('2026-01-05')
    assert changed.known_document_ids=='announced|exception'
    assert changed.knowledge_max_at==exception.knowledge_at
    assert not updated.loc[(pd.Timestamp('2025-12-08'),'NYSE__original_0929')].ordinary_horizon_complete
    unknown=build_date_metadata(timing,calendar,ledger,as_of='2025-12-12',amendment_gaps=[
        {'id':'missing_raw_amendment','venues':['NYSE'],'affected_calendar_years':[2025],
         'unknown_from':'2025-12-10T00:00:00-05:00'}])
    assert unknown.loc[(pd.Timestamp('2025-12-10'),'NYSE__original_0929')].qualification_status=='unknown'
    dataset=metadata_dataset(result)
    restored=pickle.loads(pickle.dumps(dataset))
    pd.testing.assert_frame_equal(restored.prepare('all',col_set='meta'),result)
