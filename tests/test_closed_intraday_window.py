import pandas as pd
from src.data.closed_intraday_window import closed_window_return


def test_future_bar_is_irrelevant_and_missing_closed_prefix_is_unknown():
    frame=pd.DataFrame({'datetime':pd.date_range('2026-07-01 09:35',periods=4,freq='5min'),'open':[100.,101.,102.,103.],'close':[101.,102.,103.,104.],'volume':[10.,10.,10.,10.],'price_valid':[1.,1.,1.,1.]})
    args={'day':'2026-07-01','cutoff':'09:45','frequency_minutes':5}
    before=closed_window_return(frame,**args)
    frame.loc[3,'close']=10000.
    assert closed_window_return(frame,**args)==before
    assert pd.isna(closed_window_return(frame.drop(index=1),**args))
    frame.loc[2,'volume']=0.
    assert pd.isna(closed_window_return(frame,**args))
    frame.loc[2,"volume"]=10.
    frame.loc[0,"volume"]=0.
    assert pd.isna(closed_window_return(frame,**args))
