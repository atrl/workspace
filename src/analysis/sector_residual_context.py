"""Fixed nonoverlapping beta-adjusted momentum; no fitted outcome choices."""
import numpy as np
import pandas as pd

EXTRA=['soxx_logmom20','spy_logmom20','prior60_soxx_mean','prior60_soxx_variance','prior60_spy_mean','prior60_spy_variance']
FEATURE='sector_residual_logmom20'

def context_features(closes):
    if list(closes.columns)!=['SOXX','SPY'] or not np.isfinite(closes).all().all() or not closes.gt(0).all().all():raise ValueError('invalid fixed close source')
    log=np.log(closes/closes.shift(1));out=pd.DataFrame(index=closes.index)
    for symbol,prefix in [('SOXX','soxx'),('SPY','spy')]:
        out[prefix+'_logmom20']=np.log(closes[symbol]/closes[symbol].shift(20))
        out['prior60_'+prefix+'_mean']=log[symbol].rolling(60,min_periods=60).mean().shift(20)
        out['prior60_'+prefix+'_variance']=log[symbol].rolling(60,min_periods=60).var(ddof=1).shift(20)
    covariance=log.SOXX.rolling(60,min_periods=60).cov(log.SPY,ddof=1).shift(20)
    valid=out.prior60_spy_variance.gt(1e-12)&covariance.notna()
    beta=(covariance/out.prior60_spy_variance).where(valid)
    out[FEATURE]=out.soxx_logmom20-beta*out.spy_logmom20
    trace=pd.DataFrame({'prior60_covariance':covariance,'prior60_beta':beta,'valid_beta':valid},index=out.index)
    return out[EXTRA+[FEATURE]],trace
