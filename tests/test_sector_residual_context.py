import numpy as np
import pandas as pd
from src.analysis.sector_residual_context import context_features,FEATURE

def test_beta_window_nonoverlap_and_degenerate_unknown():
    index=pd.bdate_range('2010-01-01',periods=130);r=.002+.004*np.sin(np.arange(130));prices=pd.DataFrame({'SPY':100*np.exp(np.cumsum(r)),'SOXX':100*np.exp(np.cumsum(.001+2*r))},index=index)[['SOXX','SPY']]
    f,t=context_features(prices);at=110
    np.testing.assert_allclose(t.prior60_beta.iloc[at],2,rtol=1e-10)
    # Recent-window shock changes momentum, not prior beta; intercept retained.
    changed=prices.copy();changed.iloc[at-19:at+1,0]*=1.03
    g,u=context_features(changed);np.testing.assert_allclose(u.prior60_beta.iloc[at],t.prior60_beta.iloc[at],rtol=1e-12)
    np.testing.assert_allclose(f[FEATURE].iloc[at],.020,rtol=1e-10,atol=1e-12)
    constant=prices.copy();constant.SPY=100*np.exp(.001*np.arange(130))
    q,v=context_features(constant);assert q[FEATURE].iloc[80:].isna().all() and not v.valid_beta.iloc[80:].any()
