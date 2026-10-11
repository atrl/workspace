"""Fixed paired forecast diagnostics from Recorder-owned predictions; no selection."""
import numpy as np
import pandas as pd


def paired_forecast_information(candidate, control, *, cutoff, windows, resolution=1e-8):
    """Test the preregistered excess-response MSE; IC is descriptive only."""
    if not candidate.index.equals(control.index) or not candidate.model_used.equals(control.model_used):
        raise ValueError('blocked: forecast arms differ in dates or model-use mask')
    for col in ('actual_soxl','actual_soxx','label_available_at'):
        pd.testing.assert_series_equal(candidate[col],control[col],check_exact=True)
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError('blocked: forecast diagnostic cutoff requires timezone')
    known=candidate.model_used & candidate.label_available_at.notna() & (candidate.label_available_at<cutoff)
    rows=[]
    for window,years in windows.items():
        mask=known & candidate.index.year.isin(years)
        c,b=candidate.loc[mask],control.loc[mask]
        if c.empty:
            raise ValueError('blocked: no paired mature forecasts for '+window)
        for response in ('soxl','soxx','excess'):
            values=lambda f,p: (f[p+'_soxl']-f[p+'_soxx']) if response=='excess' else f[p+'_'+response]
            actual=values(c,'actual');pred=values(c,'predicted');base=values(b,'predicted');prior=values(c,'prior')
            if not np.isfinite(np.column_stack([actual,pred,base,prior])).all():
                raise ValueError('blocked: invalid paired mature forecast values')
            mse=float(np.square(pred-actual).mean())
            base_mse=float(np.square(base-actual).mean());prior_mse=float(np.square(prior-actual).mean())
            if min(base_mse,prior_mse)<=0:
                raise ValueError('blocked: nonpositive reference forecast MSE')
            rows.append(dict(window=window,response=response,rows=len(c),candidate_mse=mse,
                matched_control_mse=base_mse,own_prior_mse=prior_mse,
                mse_ratio_to_control=mse/base_mse,mse_ratio_to_prior=mse/prior_mse,
                candidate_rank_ic=float(pred.corr(actual,method='spearman')) if pred.nunique()>1 else np.nan,
                control_rank_ic=float(base.corr(actual,method='spearman')) if base.nunique()>1 else np.nan,
                incremental_mse_pass=bool(mse/base_mse<1-resolution and mse/prior_mse<1-resolution),
                ic_is_descriptive_not_significance=True))
    table=pd.DataFrame(rows)
    return table,bool(table.query("response=='excess'").incremental_mse_pass.all())
