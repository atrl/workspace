"""Non-trading diagnostics of frozen allocation scores at actual holding clocks."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def cash_labels(frame,*,horizons,tail_threshold):
    data=frame.reset_index(drop=True);result=data[['signal_date','trade_date']].copy()
    if not data.trade_date.is_monotonic_increasing or data.trade_date.duplicated().any():raise ValueError('invalid holding calendar')
    if not pd.DatetimeIndex(data.exit_date.iloc[:-1]).equals(pd.DatetimeIndex(data.trade_date.iloc[1:])):raise ValueError('nonconsecutive holding intervals')
    for h in horizons:
        exit_price=data.soxx_exit_open.shift(-(h-1))
        result[f'cash_advantage_h{h}']=-np.log(exit_price/data.soxx_open)
        result[f'label_available_h{h}']=pd.to_datetime(data.exit_date.shift(-(h-1)))+pd.Timedelta(hours=9,minutes=49)
    h=5
    future=pd.concat([data.close_soxx.shift(-j) for j in range(h)]+[data.soxx_exit_open.shift(-j) for j in range(h)],axis=1)
    complete=future.notna().all(axis=1)
    downside=(1-future.min(axis=1)/data.soxx_open).clip(lower=0).where(complete)
    result['downside_h5']=downside
    result['tail_h5']=(downside>=-tail_threshold).astype(float).where(complete)
    return result


def score_statistics(score,label,*,binary=False):
    valid=np.isfinite(score)&np.isfinite(label);x=np.asarray(score)[valid];y=np.asarray(label)[valid]
    if len(x)<3 or len(np.unique(x))<2 or len(np.unique(y))<2:return len(x),np.nan
    return len(x),float(roc_auc_score(y,x) if binary else spearmanr(x,y).statistic)


def diagnostic_tables(daily,*,score_names,plan):
    labels=['cash_advantage_h1','cash_advantage_h5','downside_h5','tail_h5'];rows=[];intervals=[]
    for name in score_names:
        for label in labels:
            for year in ['all',*sorted(daily.trade_date.dt.year.unique())]:
                part=daily if year=='all' else daily.loc[daily.trade_date.dt.year.eq(year)]
                n,value=score_statistics(part[name].to_numpy(),part[label].to_numpy(),binary=label=='tail_h5')
                rows.append({'score':name,'label':label,'year':year,'n':n,'metric':'AUC' if label=='tail_h5' else 'Spearman','value':value})
    rng=np.random.default_rng(plan['bootstrap_seed']);n=len(daily);b=plan['bootstrap_block'];samples=plan['bootstrap_resamples']
    starts=rng.integers(0,n,size=(samples,int(np.ceil(n/b))))
    indices=((starts[:,:,None]+np.arange(b))%n).reshape(samples,-1)[:,:n]
    # Shared calendar resamples preserve pairing, missing labels and H5 overlap.
    for name in score_names:
        x=daily[name].to_numpy()
        for label in labels:
            y=daily[label].to_numpy();values=[score_statistics(x[idx],y[idx],binary=label=='tail_h5')[1] for idx in indices]
            valid=np.asarray(values);valid=valid[np.isfinite(valid)]
            intervals.append({'score':name,'label':label,'lower':float(np.quantile(valid,.025)),'upper':float(np.quantile(valid,.975)),'valid_resamples':len(valid),'null':.5 if label=='tail_h5' else 0.})
    return pd.DataFrame(rows),pd.DataFrame(intervals)
