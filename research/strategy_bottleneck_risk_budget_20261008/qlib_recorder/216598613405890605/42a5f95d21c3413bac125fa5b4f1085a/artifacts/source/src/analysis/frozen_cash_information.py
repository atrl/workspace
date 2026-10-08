"""Registered diagnostics of frozen cash forecasts, with no trading policy.

Input rows must come from the workflow's hash-verified native Qlib readback.
OLS below is descriptive calibration only; no forecast is fitted or replaced.
"""
from __future__ import annotations

import hashlib
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.objective.relative_wealth import stationary_block_bootstrap_indices


COSTS = (10., 25.)
POOLED = ('development_2016_2022', 'reused_2023_asof')
REQUIRED = ('signal_date','trade_date','exit_date','model_year','cash_advantage_prediction',
            'prior_mean_prediction','cash_advantage_1d','soxx_return','forecast_available',
            'h1_outcome_available','score_available')
# Bound statistic/rank temporaries independently of the registered draw count.
# The sampler still runs once, preserving its RNG sequence and index identity.
_BOOTSTRAP_BATCH_DRAWS = 128
_BOOTSTRAP_BATCH_VALUES = 262144


def _array(value):
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all():
        raise ValueError('blocked: diagnostic vector must be finite and one dimensional')
    return a


def correlation(prediction,target,*,rank=False):
    """A time-series correlation; an exactly constant series is undefined."""
    p,y=_array(prediction),_array(target)
    if len(p)!=len(y):raise ValueError('blocked: correlation axes differ')
    if len(p)<2:return None,'insufficient_observations'
    if np.ptp(p)==0:return None,'constant_prediction'
    if np.ptp(y)==0:return None,'constant_target'
    if rank:p,y=rankdata(p,method='average'),rankdata(y,method='average')
    p,y=p-p.mean(),y-y.mean()
    value=float(np.dot(p,y)/np.sqrt(np.dot(p,p)*np.dot(y,y)))
    if not np.isfinite(value):return None,'undefined_numeric_correlation'
    return value,None


def within_year_demean(values,years):
    v=_array(values);groups=np.asarray(years)
    if groups.shape!=v.shape:raise ValueError('blocked: demeaning axes differ')
    result=np.empty_like(v)
    for year in np.unique(groups):
        mask=groups==year;part=v[mask]
        # Constant baseline subtraction must not create a spurious tiny signal
        # from floating-point summation of a repeated scalar.
        result[mask]=0. if np.ptp(part)==0 else part-part.mean()
    return result


def cash_hurdle(cost_bps):
    c=float(cost_bps)/10000.
    if not np.isfinite(c) or c<0 or c>=1:raise ValueError('blocked: invalid proportional cost')
    return float(np.log1p(c)-np.log1p(-c))


def event_metrics(target,score,*,cost_bps):
    y,p=_array(target),_array(score)
    if len(y)!=len(p):raise ValueError('blocked: event axes differ')
    hurdle=cash_hurdle(cost_bps);mask=p>hurdle;net=y-hurdle;n=len(y);count=int(mask.sum())
    return {'cost_bps':float(cost_bps),'log_hurdle':hurdle,'event_count':count,
        'coverage':float(count/n) if n else None,
        'mean_event_gross_log_bp':float(10000*y[mask].mean()) if count else None,
        'mean_event_net_log_bp':float(10000*net[mask].mean()) if count else None,
        'positive_net_event_fraction':float((net[mask]>0).mean()) if count else None,
        'mean_masked_net_log_bp_per_observation':float(10000*np.where(mask,net,0.).mean()) if n else None,
        'conditional_mean_undefined_reason':None if count else 'no_events',
        'event_scope':'isolated_SOXX_to_CASH_to_SOXX; not a portfolio return'}


def calibration_hac(target,score,*,maxlags=20):
    y,p=_array(target),_array(score)
    if len(y)!=len(p):raise ValueError('blocked: calibration axes differ')
    n=len(y);X=np.c_[np.ones(n),p]
    rank=int(np.linalg.matrix_rank(X)) if n else 0
    result={'rows':n,'design_rank':rank,'hac_lags':int(maxlags),'hac_kernel':'Bartlett',
        'finite_sample_correction':'N/(N-2)','interval':'normal95percent',
        'scope':'descriptive calibration; no calibrated prediction produced',
        'intercept':None,'slope':None,'intercept_se':None,'slope_se':None,
        'intercept_lower95':None,'intercept_upper95':None,'slope_lower95':None,'slope_upper95':None,
        'defined':False,'undefined_reason':None}
    if n<=2:result['undefined_reason']='insufficient_observations';return result
    if rank<2:result['undefined_reason']='rank_deficient_constant_score';return result
    coefficients=np.linalg.lstsq(X,y,rcond=None)[0]
    residual=y-X@coefficients;scores=X*residual[:,None]
    meat=scores.T@scores
    for lag in range(1,min(maxlags,n-1)+1):
        cross=scores[lag:].T@scores[:-lag]
        meat+=(1-lag/(maxlags+1))*(cross+cross.T)
    bread=np.linalg.inv(X.T@X)
    covariance=(n/(n-2))*bread@meat@bread
    if not np.isfinite(covariance).all() or (np.diag(covariance)<0).any():
        result['undefined_reason']='invalid_hac_covariance';return result
    se=np.sqrt(np.diag(covariance));critical=NormalDist().inv_cdf(.975)
    for i,name in enumerate(('intercept','slope')):
        result[name]=float(coefficients[i]);result[name+'_se']=float(se[i])
        result[name+'_lower95']=float(coefficients[i]-critical*se[i])
        result[name+'_upper95']=float(coefficients[i]+critical*se[i])
    result['defined']=True
    return result


def point_metrics(target,prediction,baseline,years):
    y,p,b=_array(target),_array(prediction),_array(baseline)
    if not(len(y)==len(p)==len(b)):raise ValueError('blocked: point metric axes differ')
    n=len(y);result={'rows':n}
    for name,values in [('prediction',p),('prior_baseline',b),('target',y)]:
        result['mean_'+name]=float(values.mean()) if n else None
    result['prediction_bias']=float((p-y).mean()) if n else None
    result['baseline_bias']=float((b-y).mean()) if n else None
    for loss,func in [('MSE',np.square),('MAE',np.abs)]:
        lp,lb=func(p-y),func(b-y)
        result[loss+'_prediction']=float(lp.mean()) if n else None
        result[loss+'_prior_baseline']=float(lb.mean()) if n else None
        result['paired_'+loss+'_improvement_baseline_minus_prediction']=float((lb-lp).mean()) if n else None
    for role,score in [('prediction',p),('prior_baseline',b)]:
        demeaned=within_year_demean(score,years);dy=within_year_demean(y,years)
        for label,rank in [('Pearson',False),('Spearman',True)]:
            for prefix,u,v in [('time_series',score,y),('within_signal_year_demeaned',demeaned,dy)]:
                key=f'{prefix}_{label}_IC'+('' if role=='prediction' else '_prior_baseline')
                result[key],result[key+'_undefined_reason']=correlation(u,v,rank=rank)
    return result


def summarize_draws(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or np.isinf(a).any():raise ValueError('blocked: invalid bootstrap statistic draws')
    finite=a[np.isfinite(a)];undefined=int(len(a)-len(finite))
    quantiles=np.quantile(finite,[.025,.5,.975]) if len(finite) else [None,None,None]
    result={'total_draws':len(a),'defined_draws':len(finite),'undefined_draws':undefined,
        'lower_2_5':None,'median_50':None,'upper_97_5':None,
        'official_interval_defined':bool(len(a)>0 and undefined==0),
        'undefined_reason':'undefined_draws_present' if undefined else ('no_draws' if not len(a) else None),
        'finite_only_lower_2_5':quantiles[0],'finite_only_median_50':quantiles[1],'finite_only_upper_97_5':quantiles[2],
        'finite_only_scope':'conditional_on_defined_draws; for events conditional_on_nonempty'}
    if result['official_interval_defined']:
        result.update(dict(zip(('lower_2_5','median_50','upper_97_5'),map(float,quantiles))))
    return result


def _row_correlations(prediction,target):
    """Scalar correlation semantics over finite, paired bootstrap rows."""
    constant=(np.ptp(prediction,axis=1)==0)|(np.ptp(target,axis=1)==0)
    p=prediction-prediction.mean(axis=1,keepdims=True)
    y=target-target.mean(axis=1,keepdims=True)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore',under='ignore'):
        cross=np.matmul(p[:,None,:],y[:,:,None]).ravel()
        square_p=np.matmul(p[:,None,:],p[:,:,None]).ravel()
        square_y=np.matmul(y[:,None,:],y[:,:,None]).ravel()
        denominator=np.sqrt(square_p*square_y)
        values=cross/denominator
    values[constant|~np.isfinite(values)]=np.nan
    return values


def _sampled_average_ranks(codes,unique_count,indices):
    """Exact average tied ranks from each draw's sampled value frequencies.

    Codes only encode the original values' ordering. Ranks are rebuilt from
    each draw's multiplicities, including the duplicate rows from resampling.
    """
    sampled=codes[indices]
    offsets=np.arange(len(indices))[:,None]*unique_count
    frequencies=np.bincount((sampled+offsets).ravel(),
        minlength=len(indices)*unique_count).reshape(len(indices),unique_count)
    ranks=frequencies.cumsum(axis=1)-(frequencies-1)/2.
    return np.take_along_axis(ranks,sampled,axis=1)


def bootstrap_window(target,prediction,baseline,years,*,mean_block,resamples=2000,seed=20261003):
    y,p,b=_array(target),_array(prediction),_array(baseline);n=len(y)
    if n<2 or not(len(p)==len(b)==n):raise ValueError('blocked: bootstrap requires paired observations')
    indices=stationary_block_bootstrap_indices(observation_count=n,path_length=n,path_count=resamples,
        expected_block_length=mean_block,seed=seed)
    dp,dy=_array(within_year_demean(p,years)),_array(within_year_demean(y,years))
    improvement=np.square(b-y)-np.square(p-y)
    columns=['paired_MSE_improvement','Pearson_IC','Spearman_IC',
        'within_signal_year_demeaned_Pearson_IC','within_signal_year_demeaned_Spearman_IC']
    for c in COSTS:columns.extend([f'conditional_event_net_mean_bp_{int(c)}',f'masked_event_net_mean_bp_per_observation_{int(c)}'])
    values=np.full((resamples,len(columns)),np.nan)
    rank_sources=[]
    for source in (p,y,dp,dy):
        unique,codes=np.unique(source,return_inverse=True)
        rank_sources.append((codes,len(unique)))
    batch_size=max(1,min(_BOOTSTRAP_BATCH_DRAWS,_BOOTSTRAP_BATCH_VALUES//n))
    for start in range(0,resamples,batch_size):
        stop=min(start+batch_size,resamples);idx=indices[start:stop]
        values[start:stop,0]=improvement[idx].mean(axis=1)
        for column,(u,v) in zip((1,3),((p,y),(dp,dy))):
            sampled_u,sampled_v=u[idx],v[idx]
            values[start:stop,column]=_row_correlations(sampled_u,sampled_v)
            ranked_u=_sampled_average_ranks(*rank_sources[column-1],idx)
            ranked_v=_sampled_average_ranks(*rank_sources[column],idx)
            values[start:stop,column+1]=_row_correlations(ranked_u,ranked_v)
        for column,c in zip((5,7),COSTS):
            hurdle=cash_hurdle(c);mask=p[idx]>hurdle
            net=np.where(mask,y[idx]-hurdle,0.)
            counts=mask.sum(axis=1);totals=net.sum(axis=1)
            # Empty conditional means remain NaN; masked means remain zero.
            np.divide(totals,counts,out=values[start:stop,column],where=counts>0)
            values[start:stop,column]*=10000
            values[start:stop,column+1]=10000*(totals/n)
    draws=pd.DataFrame(values,columns=columns);draws.index.name='draw'
    summaries=pd.DataFrame([{'statistic':name,**summarize_draws(draws[name])} for name in columns])
    return {'statistics':draws,'summary':summaries,
        'indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
        'indices_shape':list(indices.shape),'indices_dtype':str(indices.dtype),
        'seed':seed,'mean_block':mean_block,'resamples':resamples,
        'row_indices_reconstruction':'src.objective.relative_wealth.stationary_block_bootstrap_indices',
        'demeaning':'original evaluated-window within-year means, fixed before resampling',
        'all_statistics_share_row_indices':True,'baseline_event_bootstrap':False}


def _checked_rows(rows,plan):
    if (not isinstance(rows,pd.DataFrame) or not isinstance(rows.index,pd.DatetimeIndex)
            or rows.index.tz is not None or not rows.index.is_monotonic_increasing or rows.index.has_duplicates):
        raise ValueError('blocked: ordered unique naive Qlib session index required')
    missing=set(REQUIRED)-set(rows.columns)
    if missing:raise ValueError(f'blocked: diagnostic columns missing {sorted(missing)}')
    if rows.empty or rows.index.max()>pd.Timestamp(plan['source']['as_of']):raise ValueError('blocked: empty source or dates beyondasof')
    for name in ('forecast_available','h1_outcome_available','score_available'):
        if not all(isinstance(v,(bool,np.bool_)) for v in rows[name]):raise ValueError(f'blocked: explicit boolean {name} required')
    if not pd.DatetimeIndex(pd.to_datetime(rows.signal_date)).equals(rows.index):raise ValueError('blocked: signal/index mismatch')
    if not np.array_equal(rows.score_available,rows.forecast_available & rows.h1_outcome_available):
        raise ValueError('blocked: score mask cannot silently remove eligible rows')
    available=rows.forecast_available.to_numpy();known=rows.h1_outcome_available.to_numpy()
    for column in ('cash_advantage_prediction','prior_mean_prediction'):_array(rows.loc[available,column])
    y=_array(rows.loc[known,'cash_advantage_1d']);source_returns=rows.loc[known,'soxx_return']
    r=_array(source_returns)
    # Verify the frozen label's original arithmetic before numeric promotion:
    # native Qlib simple returns may be float32, and log1p(float32) rounds
    # differently from log1p after promotion. Never replace the saved label.
    expected=np.asarray(-np.log1p(source_returns),dtype=np.float64)
    if (r<=-1).any() or not np.allclose(y,expected,atol=1e-14,rtol=1e-12):raise ValueError('blocked: cash target/physical return differs')
    years=set(plan['windows']['development_signal_years'])|set(plan['windows']['reused_diagnostic_signal_years'])
    if (~rows.forecast_available & rows.index.year.isin(years)).any():raise ValueError('blocked: missing registered evaluation forecast head')
    if not np.array_equal(rows.loc[available,'model_year'].to_numpy(),rows.index.year[available]):
        raise ValueError('blocked: cached annual forecast/model year differs')
    return rows.copy()


def analyze_information(rows,plan):
    """Return small tables and raw statistic draws for a Recorder workflow."""
    if (plan['uncertainty']['method'].find('2000')<0 or tuple(plan['event_diagnostic']['cost_bps'])!=(10,25)
            or plan['experiment']!='frozen_cash_information_20261003'):
        raise ValueError('blocked: registered frozen-cash diagnostic plan required')
    rows=_checked_rows(rows,plan);definitions=[]
    dev=plan['windows']['development_signal_years'];outer=plan['windows']['reused_diagnostic_signal_years']
    for year in [*dev,*outer]:definitions.append((f'signal_{year}','annual',year,rows.index.year==year))
    definitions.extend([(POOLED[0],'pooled',None,rows.index.year.isin(dev)),
                        (POOLED[1],'pooled',None,rows.index.year.isin(outer))])
    window=plan['windows']['user_window']
    user=pd.to_datetime(rows.trade_date).ge(window['trade_start']) & pd.to_datetime(rows.exit_date).le(window['exit_end'])
    definitions.append(('user_trade_2026_08_03','user_window',None,user.to_numpy()))
    points=[];calibrations=[];events=[];bootstrap_rows=[];raw_draws={}
    for name,kind,year,mask in definitions:
        selected=rows.loc[np.asarray(mask) & rows.score_available.to_numpy()]
        y=selected.cash_advantage_1d.to_numpy(float);p=selected.cash_advantage_prediction.to_numpy(float)
        b=selected.prior_mean_prediction.to_numpy(float);ys=selected.index.year.to_numpy()
        identity={'window':name,'window_type':kind,'signal_year':year}
        point={**identity,**point_metrics(y,p,b,ys),'source_window_rows':int(np.sum(mask)),
            'unscored_window_rows':int(np.sum(mask))-len(selected),
            'first_signal':str(selected.index.min().date()) if len(selected) else None,
            'last_signal':str(selected.index.max().date()) if len(selected) else None,
            'baseline_pooled_IC_scope':'may reflect between-year prior-mean drift; not within-year signal'}
        points.append(point)
        for role,score in [('prediction',p),('prior_baseline',b)]:
            calibrations.append({**identity,'forecast_role':role,**calibration_hac(y,score,maxlags=20)})
            for c in COSTS:events.append({**identity,'forecast_role':role,'rows':len(y),**event_metrics(y,score,cost_bps=c)})
        if kind=='pooled':
            for block in (20,60):
                if len(y)<2:raise ValueError('blocked: pooled diagnostic window has insufficient observations')
                result=bootstrap_window(y,p,b,ys,mean_block=block,resamples=2000,seed=20261003)
                summary=result.pop('summary');summary.insert(0,'mean_block',block);summary.insert(0,'window',name)
                bootstrap_rows.append(summary);raw_draws[(name,block)]=result
    return {'point_metrics':pd.DataFrame(points),'calibration':pd.DataFrame(calibrations),
        'event_metrics':pd.DataFrame(events),'bootstrap_summary':pd.concat(bootstrap_rows,ignore_index=True),
        'bootstrap_draws':raw_draws,'audit':{'source_rows':len(rows),'score_rows':int(rows.score_available.sum()),
            'point_windows':len(points),'pooled_bootstrap_windows':list(POOLED),'statistics_per_bootstrap':9,
            'bootstrap_resamples':2000,'mean_blocks':[20,60],'seed':20261003,
            'bias_definition':'mean(prediction-target)','loss_improvement_definition':'baseline_loss-prediction_loss',
            'baseline_events_point_only':True,'bins_or_threshold_search':False,
            'new_forecast_model_fits':0,'portfolio_paths_created':0,'promotable':False}}


__all__=['analyze_information','point_metrics','correlation','within_year_demean',
         'cash_hurdle','event_metrics','calibration_hac','bootstrap_window','summarize_draws']
