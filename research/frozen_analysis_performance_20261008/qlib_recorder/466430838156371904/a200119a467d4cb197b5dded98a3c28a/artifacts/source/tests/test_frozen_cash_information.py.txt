"""Synthetic information/fee/statistical contracts; no market metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.frozen_cash_information as a


def test_correlation_constant_is_undefined_and_ranks_use_ties():
    assert a.correlation([1,1],[2,3])==(None,'constant_prediction')
    assert a.correlation([1,2],[3,3])==(None,'constant_target')
    assert a.correlation([1],[2])==(None,'insufficient_observations')
    assert a.correlation([1,1,2,3],[9,9,4,1],rank=True)[0]==pytest.approx(-1.)


def test_fixed_within_year_demeaning_removes_between_year_drift():
    p=np.array([1.,2.,3.,101.,102.,103.]);y=np.array([3.,2.,1.,103.,102.,101.]);years=[2016]*3+[2017]*3
    assert a.correlation(p,y)[0]>.99
    dp,dy=a.within_year_demean(p,years),a.within_year_demean(y,years)
    assert a.correlation(dp,dy)[0]==pytest.approx(-1.)
    np.testing.assert_array_equal(a.within_year_demean([.1]*3+[.2]*3,years),np.zeros(6))


@pytest.mark.parametrize('cost',[10.,25.])
def test_cash_detour_matches_independent_fee_funded_roundtrip(cost):
    c=cost/10000.;r=np.array([-.02,0.,.03]);y=-np.log1p(r)
    expected=np.log(((1-c)/(1+c))/(1+r))
    np.testing.assert_allclose(y-a.cash_hurdle(cost),expected,atol=3e-16,rtol=0)
    threshold=a.cash_hurdle(cost)
    result=a.event_metrics(y,[threshold,np.nextafter(threshold,np.inf),0.],cost_bps=cost)
    assert result['event_count']==1 and result['coverage']==1/3
    assert result['mean_event_net_log_bp']==pytest.approx(-10000*threshold)


def test_empty_event_is_not_zero_conditional_mean():
    e=a.event_metrics([.02,-.01],[0.,0.],cost_bps=25)
    assert e['event_count']==0 and e['mean_event_net_log_bp'] is None
    assert e['positive_net_event_fraction'] is None and e['mean_masked_net_log_bp_per_observation']==0


def test_point_loss_direction_bias_and_cost_shift_identity():
    y=np.array([.01,-.02,.03,-.01]);p=y+.001;b=np.zeros(4)
    out=a.point_metrics(y,p,b,[2016]*4)
    assert out['prediction_bias']==pytest.approx(.001)
    assert out['paired_MSE_improvement_baseline_minus_prediction']>0
    assert out['paired_MAE_improvement_baseline_minus_prediction']>0
    shift=a.cash_hurdle(25)
    shifted=a.point_metrics(y-shift,p-shift,b-shift,[2016]*4)
    assert shifted['paired_MSE_improvement_baseline_minus_prediction']==pytest.approx(out['paired_MSE_improvement_baseline_minus_prediction'])
    assert out['time_series_Pearson_IC_prior_baseline'] is None


def test_calibration_hac_matches_independent_pairwise_bartlett_sandwich():
    rng=np.random.default_rng(882);p=rng.normal(0,.01,100);y=.001+.7*p+rng.normal(0,.002,100)
    result=a.calibration_hac(y,p)
    X=np.c_[np.ones(100),p];coef=np.linalg.solve(X.T@X,X.T@y);u=y-X@coef
    meat=np.zeros((2,2))
    for t in range(100):
        for s in range(100):
            lag=abs(t-s)
            if lag<=20:meat+=(1-lag/21)*u[t]*u[s]*np.outer(X[t],X[s])
    inv=np.linalg.inv(X.T@X);cov=100/98*inv@meat@inv
    assert result['defined'] and result['intercept']==pytest.approx(coef[0])
    assert result['slope_se']==pytest.approx(np.sqrt(cov[1,1]))
    assert result['slope_lower95']<result['slope']<result['slope_upper95']
    bad=a.calibration_hac(y,np.full(100,.002))
    assert not bad['defined'] and bad['slope'] is None and bad['undefined_reason'].startswith('rank_deficient')


def test_any_undefined_bootstrap_draw_nulls_official_interval():
    result=a.summarize_draws([1.,np.nan,3.])
    assert result['undefined_draws']==1 and result['defined_draws']==2
    assert result['lower_2_5'] is None and not result['official_interval_defined']
    assert result['finite_only_median_50']==2.
    assert a.summarize_draws([np.nan,np.nan])['finite_only_median_50'] is None


def test_bootstrap_is_paired_repeatable_and_fixed_demeaned():
    y=np.array([.01,-.01,.02,-.02]*8);p=np.linspace(-.003,.007,32);b=np.full(32,-.001);years=[2016]*16+[2017]*16
    first=a.bootstrap_window(y,p,b,years,mean_block=20,resamples=20,seed=20261003)
    second=a.bootstrap_window(y,p,b,years,mean_block=20,resamples=20,seed=20261003)
    pd.testing.assert_frame_equal(first['statistics'],second['statistics'])
    assert first['indices_sha256']==second['indices_sha256'] and first['statistics'].shape==(20,9)
    assert first['all_statistics_share_row_indices'] and not first['baseline_event_bootstrap']
    assert not any('prior' in c for c in first['statistics'])
    from src.objective.relative_wealth import stationary_block_bootstrap_indices
    idx=stationary_block_bootstrap_indices(observation_count=32,path_length=32,path_count=20,expected_block_length=20,seed=20261003)[0]
    expected=((b-y)**2-(p-y)**2)[idx].mean()
    assert first['statistics'].iloc[0].paired_MSE_improvement==expected
    dp=a.within_year_demean(p,years);dy=a.within_year_demean(y,years)
    # Batched row dot products may sum in a different order from scalar BLAS.
    assert first['statistics'].iloc[0].within_signal_year_demeaned_Pearson_IC==pytest.approx(
        a.correlation(dp[idx],dy[idx])[0],rel=2e-13,abs=2e-15)


def _scalar_bootstrap_reference(y,p,b,years,indices):
    """Independent pre-batching formulas, including compressed event means."""
    def demean(v):
        out=np.empty_like(v)
        for year in np.unique(years):
            mask=years==year;part=v[mask]
            out[mask]=0. if np.ptp(part)==0 else part-part.mean()
        return out

    def corr(u,v,rank):
        if np.ptp(u)==0 or np.ptp(v)==0:return np.nan
        if rank:
            u=pd.Series(u).rank(method='average').to_numpy()
            v=pd.Series(v).rank(method='average').to_numpy()
        u=u-u.mean();v=v-v.mean()
        with np.errstate(divide='ignore',invalid='ignore',over='ignore',under='ignore'):
            value=np.dot(u,v)/np.sqrt(np.dot(u,u)*np.dot(v,v))
        return value if np.isfinite(value) else np.nan

    dp,dy=demean(p),demean(y);out=[]
    for idx in indices:
        row=[((b-y)**2-(p-y)**2)[idx].mean()]
        for u,v,rank in ((p,y,False),(p,y,True),(dp,dy,False),(dp,dy,True)):
            row.append(corr(u[idx],v[idx],rank))
        for cost in (10.,25.):
            fraction=cost/10000.;hurdle=np.log1p(fraction)-np.log1p(-fraction)
            mask=p[idx]>hurdle;net=y[idx]-hurdle
            row.extend((10000*net[mask].mean() if mask.any() else np.nan,
                        10000*np.where(mask,net,0.).mean()))
        out.append(row)
    return np.asarray(out)


@pytest.mark.parametrize('case',[
    'continuous','ties','constant_prediction','constant_target',
    'constant_within_year','threshold_boundary','tiny_variance','underflow',
])
def test_batched_bootstrap_matches_scalar_all_statistics_and_intervals(case):
    from src.objective.relative_wealth import stationary_block_bootstrap_indices
    rng=np.random.default_rng(911);n=31
    y=rng.normal(0,.01,n);p=rng.normal(.004,.006,n);b=rng.normal(-.001,.001,n)
    years=np.r_[np.full(12,2016),np.full(n-12,2017)]
    if case=='ties':
        p=np.resize([-.005,.002,.006,.01],n);y=np.resize([.01,-.02,.01,.03,-.02],n)
    elif case=='constant_prediction':p[:]=.003
    elif case=='constant_target':y[:]=.01
    elif case=='constant_within_year':p=np.where(years==2016,.003,.006)
    elif case=='threshold_boundary':
        edges=[np.log1p(c/10000.)-np.log1p(-c/10000.) for c in (10.,25.)]
        p=np.resize([v for h in edges for v in (np.nextafter(h,-np.inf),h,np.nextafter(h,np.inf))],n)
    elif case=='tiny_variance':
        p=.002+np.arange(n)*np.spacing(.002);y=.004+rng.integers(-10,10,n)*np.spacing(.004)
    elif case=='underflow':
        p=rng.normal(size=n)*1e-100;y=rng.normal(size=n)*1e-100
    # 131 draws exercise a complete default batch and a nonempty tail batch.
    result=a.bootstrap_window(y,p,b,years,mean_block=3,resamples=131,seed=747)
    indices=stationary_block_bootstrap_indices(observation_count=n,path_length=n,path_count=131,
        expected_block_length=3,seed=747)
    expected=_scalar_bootstrap_reference(y,p,b,years,indices)
    assert result['indices_sha256']==hashlib.sha256(indices.tobytes()).hexdigest()
    assert result['indices_shape']==[131,n] and result['indices_dtype']=='int32'
    actual=result['statistics'].to_numpy()
    np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
    tolerances=[2e-18]+[2e-15]*4+[2e-13]*4
    for j,atol in enumerate(tolerances):
        np.testing.assert_allclose(actual[:,j],expected[:,j],rtol=2e-13,atol=atol,equal_nan=True)
    for j,row in enumerate(result['summary'].itertuples()):
        defined=expected[np.isfinite(expected[:,j]),j]
        assert row.defined_draws==len(defined) and row.undefined_draws==131-len(defined)
        assert row.official_interval_defined==(len(defined)==131)
        if len(defined):
            quantiles=np.quantile(defined,[.025,.5,.975])
            np.testing.assert_allclose([row.finite_only_lower_2_5,row.finite_only_median_50,
                row.finite_only_upper_97_5],quantiles,rtol=2e-13,atol=tolerances[j])
            if row.official_interval_defined:
                np.testing.assert_allclose([row.lower_2_5,row.median_50,row.upper_97_5],
                    quantiles,rtol=2e-13,atol=tolerances[j])
        if not row.official_interval_defined:
            assert pd.isna(row.lower_2_5) and pd.isna(row.median_50) and pd.isna(row.upper_97_5)


def test_sampled_average_ranks_are_exact_for_ties_absent_values_and_constant_draws():
    source=np.array([0.,0.,1.,2.])
    indices=np.array([[0,0,3,1],[2,2,2,2],[1,3,1,3]])
    unique,codes=np.unique(source,return_inverse=True)
    actual=a._sampled_average_ranks(codes,len(unique),indices)
    expected=np.array([[2.,2.,4.,2.],[2.5,2.5,2.5,2.5],[1.5,3.5,1.5,3.5]])
    np.testing.assert_array_equal(actual,expected)


def test_undefined_sampled_correlations_and_empty_events_null_official_intervals(monkeypatch):
    indices=np.array([[0,0,0,0],[0,1,2,3],[1,2,3,1]],dtype=np.int32)
    monkeypatch.setattr(a,'stationary_block_bootstrap_indices',lambda **kw:indices)
    y=np.array([.01,-.02,.02,.03]);p=np.array([0.,.004,.006,.01]);b=np.zeros(4)
    result=a.bootstrap_window(y,p,b,[2016]*4,mean_block=3,resamples=3)
    expected=_scalar_bootstrap_reference(y,p,b,np.full(4,2016),indices)
    np.testing.assert_allclose(result['statistics'],expected,atol=2e-13,rtol=2e-13,equal_nan=True)
    summary=result['summary'].set_index('statistic')
    for name in ['Pearson_IC','Spearman_IC','within_signal_year_demeaned_Pearson_IC',
                 'within_signal_year_demeaned_Spearman_IC','conditional_event_net_mean_bp_10',
                 'conditional_event_net_mean_bp_25']:
        row=summary.loc[name]
        assert row.defined_draws==2 and row.undefined_draws==1
        assert not row.official_interval_defined and pd.isna(row.lower_2_5)
    assert summary.loc['masked_event_net_mean_bp_per_observation_10','official_interval_defined']


def test_bootstrap_batch_size_limits_temporaries_without_changing_draws(monkeypatch):
    y=np.linspace(-.02,.02,17);p=np.sin(np.arange(17))*.01;b=np.zeros(17);years=[2016]*17
    expected=a.bootstrap_window(y,p,b,years,mean_block=3,resamples=11)
    shapes=[];original=a._row_correlations
    def tracked(p,y,**kw):
        shapes.append(p.shape)
        return original(p,y,**kw)
    monkeypatch.setattr(a,'_row_correlations',tracked)
    monkeypatch.setattr(a,'_BOOTSTRAP_BATCH_DRAWS',4)
    monkeypatch.setattr(a,'_BOOTSTRAP_BATCH_VALUES',17*3)
    result=a.bootstrap_window(y,p,b,years,mean_block=3,resamples=11)
    assert [shape[0] for shape in shapes]==[3]*12+[2]*4
    assert all(rows*columns<=51 for rows,columns in shapes)
    assert result['indices_sha256']==expected['indices_sha256']
    pd.testing.assert_frame_equal(result['statistics'],expected['statistics'])
    pd.testing.assert_frame_equal(result['summary'],expected['summary'])


@pytest.mark.parametrize('field', ['target','prediction','baseline','years'])
def test_bootstrap_rejects_invalid_axes_or_nonfinite_values(field):
    arguments={'target':np.arange(3,dtype=float),'prediction':np.arange(3,dtype=float),
        'baseline':np.zeros(3),'years':np.full(3,2016)}
    arguments[field]=np.array([0.,np.nan,1.]) if field!='years' else [2016,2017]
    with pytest.raises(ValueError,match='blocked:'):
        a.bootstrap_window(**arguments,mean_block=3,resamples=2)


def test_bootstrap_rejects_nonfinite_demeaned_values_before_rank_encoding():
    largest=np.finfo(float).max
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(ValueError,match='blocked: diagnostic vector must be finite'):
            a.bootstrap_window([0.,.1],[largest,largest/2],[0.,0.],[2016,2016],
                mean_block=3,resamples=2)


def test_bootstrap_zero_event_draws_retained_and_masked_mean_defined():
    result=a.bootstrap_window(np.linspace(-.02,.02,24),np.zeros(24),np.zeros(24),[2016]*24,
        mean_block=60,resamples=12)
    assert result['statistics'].conditional_event_net_mean_bp_10.isna().all()
    assert result['statistics'].masked_event_net_mean_bp_per_observation_10.eq(0).all()
    row=result['summary'].set_index('statistic').loc['conditional_event_net_mean_bp_10']
    assert row.undefined_draws==12 and pd.isna(row.lower_2_5)


def synthetic_rows():
    dates=pd.DatetimeIndex(['2016-01-04','2016-01-05','2023-01-03','2023-01-04',
                            '2026-09-21','2026-09-22','2026-09-23','2026-09-24','2026-09-25'],name='datetime')
    r=np.array([.01,-.01,.015,-.012,.005,-.007,.008,np.nan,np.nan]);y=-np.log1p(r)
    return pd.DataFrame({'signal_date':dates,'trade_date':dates+pd.Timedelta(days=1),
        'exit_date':dates+pd.Timedelta(days=2),'model_year':dates.year,
        'cash_advantage_prediction':np.linspace(-.001,.003,len(dates)),
        'prior_mean_prediction':np.full(len(dates),-.0005),'cash_advantage_1d':y,'soxx_return':r,
        'forecast_available':True,'h1_outcome_available':np.isfinite(y),'score_available':np.isfinite(y),
        # H5 missing on these late but H1-mature rows must not matter.
        'forecast_available_at':pd.NaT},index=dates)


def plan():
    return json.loads(Path('research/frozen_cash_information_20261003/diagnostic_plan.json').read_text())


def test_input_checks_h1_only_no_sibling_vol_filter_or_silent_row_removal():
    rows=synthetic_rows();rows['future_vol_prediction']=np.nan
    checked=a._checked_rows(rows,plan())
    assert checked.score_available.sum()==7 and len(checked)==9
    bad=rows.copy();bad.loc[bad.index[0],'score_available']=False
    with pytest.raises(ValueError,match='silently'):a._checked_rows(bad,plan())
    bad=rows.copy();bad.loc[bad.index[0],'cash_advantage_prediction']=np.nan
    with pytest.raises(ValueError,match='finite'):a._checked_rows(bad,plan())
    bad=rows.copy();bad.loc[bad.index[0],'cash_advantage_1d']=.5
    with pytest.raises(ValueError,match='target'):a._checked_rows(bad,plan())


def test_native_float32_log1p_identity_preserves_saved_label_and_tolerance():
    rows=synthetic_rows()
    rows['soxx_return']=rows.soxx_return.astype(np.float32)
    rows['cash_advantage_1d']=-np.log1p(rows.soxx_return)
    known=rows.h1_outcome_available
    before=rows.cash_advantage_1d.copy()
    promoted=-np.log1p(rows.loc[known,'soxx_return'].to_numpy(dtype=np.float64))
    assert not np.allclose(rows.loc[known,'cash_advantage_1d'],promoted,atol=1e-14,rtol=1e-12)
    checked=a._checked_rows(rows,plan())
    pd.testing.assert_series_equal(checked.cash_advantage_1d,before)
    assert checked.cash_advantage_1d.dtype==np.float32
    bad=rows.copy();bad.loc[bad.index[0],'cash_advantage_1d']+=np.float32(1e-6)
    with pytest.raises(ValueError,match='target'):a._checked_rows(bad,plan())


def test_full_orchestration_has_only_registered_pooled_bootstrap(monkeypatch):
    calls=[]
    original=a.bootstrap_window
    def short(y,p,b,years,**kw):
        calls.append(kw.copy());kw['resamples']=8
        return original(y,p,b,years,**kw)
    monkeypatch.setattr(a,'bootstrap_window',short)
    result=a.analyze_information(synthetic_rows(),plan())
    assert len(calls)==4 and all(c['resamples']==2000 and c['seed']==20261003 for c in calls)
    assert [c['mean_block'] for c in calls]==[20,60,20,60]
    assert len(result['point_metrics'])==14 and len(result['calibration'])==28
    assert len(result['event_metrics'])==56 and len(result['bootstrap_summary'])==36
    assert result['audit']['new_forecast_model_fits']==0 and not result['audit']['promotable']
    assert len(result['bootstrap_draws'])==4
