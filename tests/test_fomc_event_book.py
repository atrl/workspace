import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data.fomc_event_prices import release_clock
from src.backtest.fomc_event_book import replay_events


def test_statement_timezone_and_fixed_event_exit_before_release():
    stamp=release_clock('<h3>Federal Reserve issues FOMC statement</h3>January 29, 2020 For release at 2:00 p.m. EST','2020-01-29')
    assert stamp.hour==14 and stamp.tzname()=='EST'
    with pytest.raises(ValueError,match='timezone'):
        release_clock('Federal Reserve issues FOMC statement January 29, 2020 For release at 2:00 p.m. EDT','2020-01-29')
    tariff=json.loads((Path(__file__).resolve().parents[1]/'research/ibkr_tiered_minute_diagnosis_20261013/fee_schedule.json').read_text())['tariff']
    days=pd.bdate_range('2020-01-28',periods=4);rows=[]
    for i,day in enumerate(days[:3]):
        row={'trade_date':day,'signal_date':day-pd.Timedelta(days=1),'exit_date':days[i+1]}
        for symbol in ['soxl','soxx']:
            for key in ['known_'+symbol,symbol+'_open',symbol+'_exit_open','close_'+symbol,symbol+'_event_exit','raw_'+symbol+'_open','raw_'+symbol+'_event_exit']:row[key]=100.
        rows.append(row)
    windows=pd.DataFrame([{'entry_date':days[0],'exit_date':days[1],'event_id':'meeting'}])
    book=replay_events(pd.DataFrame(rows),windows,weights=np.array([.3,.55,.15]),plan={'initial_cash':100000,'tariff':tariff,'outside_event_target':[0,0,1]},slippage_bps=2)
    assert set(book['orders'].stamp)=={days[0]+pd.Timedelta(hours=9,minutes=47),days[1]+pd.Timedelta(hours=13,minutes=59)}
    assert book['events'].event_net_log_return.iloc[0]<0
    assert book['replay'].iloc[-1].ending_soxl_quantity==0 and book['replay'].iloc[-1].ending_soxx_quantity==0
