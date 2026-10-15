"""Time-isolated derived-feature information test, not a trading policy."""
import numpy as np
import pandas as pd


def mature_before(availability, cutoff):
    return pd.to_datetime(availability).lt(pd.Timestamp(cutoff))


def fit_scaler(values):
    data=np.asarray(values,float)
    if np.isinf(data).any():raise ValueError('infinite feature')
    mean=np.nanmean(data,axis=0)
    if not np.isfinite(mean).all():raise ValueError('training feature has no observed values')
    filled=np.where(np.isnan(data),mean,data);scale=filled.std(axis=0,ddof=0);scale[scale==0]=1.
    return mean,scale


def transform(values,mean,scale):
    data=np.asarray(values,float)
    if np.isinf(data).any():raise ValueError('infinite feature')
    return (np.where(np.isnan(data),mean,data)-mean)/scale


def full_calendar_bootstrap(values,*,mean_block,resamples,seed,alpha):
    values=np.asarray(values,float);rng=np.random.default_rng(seed);n=len(values)
    indices=np.empty((resamples,n),dtype=np.int32);means=np.full(resamples,np.nan)
    for r in range(resamples):
        index=int(rng.integers(n));indices[r,0]=index
        for t in range(1,n):
            index=int(rng.integers(n)) if rng.random()<1/mean_block else (index+1)%n
            indices[r,t]=index
        sampled=values[indices[r]]
        if np.isfinite(sampled).any():means[r]=np.nanmean(sampled)
    valid=means[np.isfinite(means)]
    if not len(valid):raise ValueError('no estimable block replicate')
    return {'paired_mean_loss_improvement':float(np.nanmean(values)),
        'lower_bound':float(np.quantile(valid,alpha)),'upper_bound':float(np.quantile(valid,1-alpha)),
        'valid_replicates':len(valid),'invalid_replicates':int(resamples-len(valid)),
        'calendar_rows':n,'scored_rows':int(np.isfinite(values).sum())},indices,means
