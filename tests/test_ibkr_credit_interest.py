import numpy as np
import pandas as pd
import pytest
from src.analysis.ibkr_credit_interest import audit_credit_interest


def fixture():
    dates=pd.date_range('2024-01-30','2024-02-05')
    orders=pd.DataFrame({'stamp':pd.to_datetime(['2024-01-30 09:47','2024-01-31 09:47']),'sell':[False,True],'notional':[20000.,5000.],'total':[2.,1.]})
    book={'audit':{'status':'passed'},'orders':orders,'authority':pd.DataFrame({'trade_date':dates[:2],'ending_cash':[79998.,84997.]})}
    values=pd.DataFrame({'principal':pd.to_datetime(['2024-02-01','2024-02-02']),'fee':pd.to_datetime(['2024-01-30','2024-01-31'])})
    rules=pd.DataFrame({'annual_credit_rate':.036,'free_cash':10000.,'full_rate_nav':100000.,'day_basis':360.,'rounding':'half_up'},index=dates)
    return book,dict(initial_cash=100000.,daily_nav_without_interest=pd.Series(100000.,index=dates),value_dates=values,contracts=rules,payment_dates={'2024-01':'2024-02-05','2024-02':'2024-03-05'})


def test_native_trade_cash_settlement_and_monthly_interest_stay_separate():
    book,kwargs=fixture();result=audit_credit_interest(book,**kwargs)
    assert result.unsettled_cash_flow.tolist()==[-20000.,-15000.,5000.,0.,0.,0.,0.]
    assert result.interest_accrued_today.tolist()==[9.,9.,7.,7.5,7.5,7.5,7.5]
    assert result.interest_paid_today.tolist()==[0.,0.,0.,0.,0.,0.,18.]
    assert result.iloc[-1].interest_pending==37. and result.iloc[-1].interest_earned_total==55.
    np.testing.assert_allclose(result.interest_earned_total,result.interest_posted_total+result.interest_pending)


def test_missing_calendar_rate_or_value_date_is_not_zero_filled():
    book,kwargs=fixture();kwargs['contracts'].loc['2024-02-03','annual_credit_rate']=np.nan
    with pytest.raises(ValueError,match='credit contract'):audit_credit_interest(book,**kwargs)
    book,kwargs=fixture();kwargs['contracts']=kwargs['contracts'].drop(pd.Timestamp('2024-02-03'))
    with pytest.raises(ValueError,match='consecutive calendar'):audit_credit_interest(book,**kwargs)
    book,kwargs=fixture();kwargs['value_dates'].loc[0,'principal']=pd.NaT
    with pytest.raises(ValueError,match='calendar dates'):audit_credit_interest(book,**kwargs)


def test_nav_scaling_uses_supplied_nav_and_credit_allowance():
    book,kwargs=fixture();day=pd.DatetimeIndex(['2024-01-30']);book['orders']=book['orders'].iloc[:0];book['authority']=pd.DataFrame({'trade_date':day,'ending_cash':[50000.]})
    kwargs.update(initial_cash=50000.,daily_nav_without_interest=pd.Series(50000.,index=day),value_dates=kwargs['value_dates'].iloc[:0],contracts=kwargs['contracts'].iloc[:1],payment_dates={'2024-01':'2024-02-05'})
    result=audit_credit_interest(book,**kwargs);assert result.iloc[0].nav_scaling==.5 and result.iloc[0].interest_accrued_today==2.
