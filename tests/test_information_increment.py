import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
from src.data.information_increment import load_information_state,build_matched_information_datasets,bind_matched_information_datasets
from src.data.oos_horizon_inputs import prepare_horizon_inputs
from src.model.generic_annual_ridge_selection import qualify_annual_horizon_inputs
from src.data.convex_allocation import LABEL_COLUMNS

ROOT=Path(__file__).resolve().parents[1]


def _bundle():
    import qlib
    qlib.init(provider_uri=str(ROOT/'data/qlib/published_futu_affine_20260925_v1'),region='us',kernels=1)
    price=prepare_horizon_inputs(ROOT,json.loads((ROOT/'research/purged_ridge_regularization_20261009/plan.json').read_text()))
    summary=json.loads((ROOT/'research/nvda_original_guidance_qualification_20261011/summary.json').read_text())
    state=load_information_state(provider_uri=ROOT/summary['source']['provider_uri'],content_sha256=summary['provider_content_sha256'],
        instrument='NVDA_GUIDE_INFO',field='$guidance_qoq',signal_dates=price['frame'].index)
    return price,state


def test_native_real_information_calendars_masks_and_recorder_roundtrip_zero_fit():
    import pickle
    price,state=_bundle()
    out=build_matched_information_datasets(price,state,as_of='2026-09-25')
    assert (out['rows'],out['known_rows'],out['unknown_rows'])==(3956,3924,32)
    assert out['fit_performed'] is False
    assert len(out['control_features'])==25 and len(out['candidate_features'])==26
    frame=out['frame'];known=frame.source_known.astype(bool)
    assert frame.index[known][0]==pd.Timestamp('2011-02-17')
    assert frame.loc['2011-02-17','source_available_at']==pd.Timestamp('2011-02-17 14:31+00:00')
    assert frame.loc[~known,'information_value'].isna().all() and frame.loc[~known,'feature_available_at'].isna().all()
    assert frame.loc['2014-05-07','information_value']!=frame.loc['2014-05-09','information_value']
    assert (frame.loc[known,'feature_available_at']<=frame.loc[known,'decision_at']).all()
    audits=[]
    for arm in ('control','candidate'):
        q=qualify_annual_horizon_inputs(out[arm],frame,outer_year=2016,feature_columns=out[arm+'_features'],
            label_columns=LABEL_COLUMNS,availability_column='h5_available_at',eligibility_column='source_known')
        audits.append(q['audit'])
    for key in ('inner_identity','outer_identity','validation_identity'):
        assert audits[0][key]==audits[1][key]
    restored=bind_matched_information_datasets(pickle.loads(pickle.dumps(out)),as_of='2026-09-25')
    pd.testing.assert_frame_equal(restored['candidate_prepared'],out['candidate_prepared'],check_exact=True)


def test_information_identity_or_known_clock_failure_blocks_before_fit():
    price,state=_bundle();changed=state.copy()
    changed.loc[changed.source_known.eq(1),'source_available_at']=pd.NaT
    with pytest.raises(ValueError,match='both explicit clocks'):
        build_matched_information_datasets(price,changed,as_of='2026-09-25')
    changed=state.copy();changed.loc[changed.source_known.eq(0),'information_value']=0.
    with pytest.raises(ValueError,match='unknown was filled'):
        build_matched_information_datasets(price,changed,as_of='2026-09-25')
    with pytest.raises(ValueError,match='identity'):
        load_information_state(provider_uri=ROOT/'data/qlib/nvda_original_guidance_20261011_v1',content_sha256='0'*64,
            instrument='NVDA_GUIDE_INFO',field='$guidance_qoq',signal_dates=price['frame'].index)
