import pandas as pd
import pytest
from src.workflow.quarterly_profit_allocation import CONTEXT,isolated_risk_context


def test_frozen_market_context_replaces_guidance_probabilities_and_rejects_calendar_drift():
    raw=pd.DataFrame({'signal_date':pd.to_datetime(['2023-12-29','2024-01-02']),
                      'trade_date':pd.to_datetime(['2024-01-02','2024-01-03']),
                      'exit_date':pd.to_datetime(['2024-01-03','2024-01-04']),
                      'alpha_probability':[.99,.98],'tail_probability':[.91,.92],
                      'guidance_surprise':[123.,456.]})
    market=raw[['signal_date','trade_date','exit_date']].copy()
    for index,column in enumerate(CONTEXT):market[column]=[index/100,(index+1)/100]
    result,audit=isolated_risk_context(raw,market)
    pd.testing.assert_frame_equal(result[CONTEXT],market[CONTEXT])
    assert 'guidance_surprise' not in result and result.alpha_probability.iloc[0]!=raw.alpha_probability.iloc[0]
    assert audit['context_columns']==CONTEXT
    changed=market.copy();changed.loc[0,'signal_date']=pd.Timestamp('2024-01-02')
    with pytest.raises(ValueError,match='calendar differs'):isolated_risk_context(raw,changed)
