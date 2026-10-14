"""Fixed descriptive fields for prior-publication baskets; no predictive fitting."""
import numpy as np
import pandas as pd
from src.data.soxx_pit_holdings import build_pit_soxx_weight_panel
from src.analysis.breadth_change_decomposition import decompose_breadth_changes


def reported_basket_descriptors(prices, holdings):
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('ordered unique price calendar required')
    panel=build_pit_soxx_weight_panel(holdings,calendar=prices.index,availability_lag_days=1)
    weights=panel.pivot(index='datetime',columns='instrument',values='weight_pct')
    columns=prices.columns.union(weights.columns)
    prices=prices.reindex(columns=columns)
    weights=weights.fillna(0.).reindex(index=prices.index,columns=columns,fill_value=0.)
    # Calendar rows before any public basket remain unknown, not all-zero baskets.
    known_dates=pd.DatetimeIndex(panel.datetime.unique())
    weights.loc[~weights.index.isin(known_dates)]=np.nan
    low=prices.rolling(20,min_periods=20).min()
    indicator=prices.le(low).astype(float).where(low.notna()&prices.notna())
    decomposition=decompose_breadth_changes(indicator,weights)
    returns=prices.pct_change(fill_method=None);concentration=[]
    for day in prices.index:
        basket=weights.loc[day];members=basket.gt(0)
        if basket.isna().any() or not members.any() or not np.isfinite(returns.loc[day,members]).all():
            concentration.append(np.nan);continue
        losses=basket.loc[members]*(-returns.loc[day,members]).clip(lower=0)
        total=losses.sum();concentration.append(float(losses.nlargest(5).sum()/total) if total>0 else np.nan)
    result=pd.DataFrame({'new_low_level20':decomposition.current,
                        'new_low_price_contraction5':-decomposition.price_component.rolling(5,min_periods=5).sum(),
                        'downside_top5_share':concentration},index=prices.index)
    result['contraction_known']=result.new_low_price_contraction5.notna().astype(float)
    result['concentration_known']=result.downside_top5_share.notna().astype(float)
    return result
