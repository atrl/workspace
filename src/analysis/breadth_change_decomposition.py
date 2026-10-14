"""Exact descriptive split of price-indicator moves and published-basket changes."""
import numpy as np
import pandas as pd


def decompose_breadth_changes(indicator, weights):
    if not indicator.index.equals(weights.index) or not indicator.columns.equals(weights.columns):
        raise ValueError('indicator and weight axes differ')
    if not indicator.index.is_unique or not indicator.index.is_monotonic_increasing:
        raise ValueError('ordered unique dates required')
    def value(values, basket):
        if basket.isna().any() or not np.isfinite(basket).all() or basket.lt(0).any() or basket.sum()<=0:
            return np.nan
        members=basket.gt(0)
        if not np.isfinite(values.loc[members]).all():return np.nan
        return float((values.loc[members]*basket.loc[members]).sum()/basket.sum())
    rows=[]
    for i,day in enumerate(indicator.index):
        current=value(indicator.iloc[i],weights.iloc[i]);row={'date':day,'current':current,'total_change':np.nan,'price_component':np.nan,'basket_component':np.nan,'reconciliation_error':np.nan}
        if i:
            previous=value(indicator.iloc[i-1],weights.iloc[i-1]);old_basket_current=value(indicator.iloc[i],weights.iloc[i-1])
            if np.isfinite([current,previous,old_basket_current]).all():
                row.update(total_change=current-previous,price_component=old_basket_current-previous,basket_component=current-old_basket_current)
                row['reconciliation_error']=row['total_change']-row['price_component']-row['basket_component']
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')
