"""Source-eligibility contract + 25/26 matched-arm DatasetH prepare (0 fit)."""
import json
import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.generic_horizon import predictor_clock
from src.workflow import vxn_vix_information_increment as V

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research" / "vxn_vix_information_increment_20261011" / "plan.json"


def test_source_eligibility_model_exclusions_are_audited():
    signal = pd.bdate_range("2016-01-04", periods=6)
    known = [0, 1, 1, 1, 0, 1]
    # row2 label unmatured (>= cutoff), row3 feature clock missing
    label_at = ["2016-02-01 09:31+00:00", "2016-02-01 09:31+00:00", "2017-06-01 09:31+00:00",
                "2016-02-01 09:31+00:00", "2016-02-01 09:31+00:00", "2016-02-02 09:31+00:00"]
    feature_at = ["2016-01-04 16:00+00:00", "2016-01-05 16:00+00:00", "2016-01-06 16:00+00:00",
                  None, "2016-01-08 16:00+00:00", "2016-01-11 16:00+00:00"]
    out = V.source_eligibility_model(signal_dates=signal, source_known=known, label_available_at=label_at,
                                     cutoff=pd.Timestamp('2017-01-01',tz='America/New_York'), feature_available_at=feature_at,
                                     prediction_at=predictor_clock(signal))
    audit = out["audit"]
    assert audit["rows_total"] == 6 and audit["rows_eligible"] == 2
    assert audit["exclusion_reasons"]["source_unknown"] == 2
    assert audit["exclusion_reasons"]["label_unmatured_or_nat"] == 1   # row2
    assert audit["exclusion_reasons"]["feature_clock_missing_late_or_after_cutoff"] == 1  # row3
    assert audit["eligible_index_sha256"] != audit["excluded_index_sha256"]
    assert audit["no_row_deleted"] is True
    assert list(out["eligible_mask"]) == [False, True, False, False, False, True]


@pytest.mark.parametrize('source', [[None], ['2016-01-04 09:31']])
def test_known_source_requires_an_explicit_aware_clock(source):
    with pytest.raises(ValueError):
        V.combine_feature_availability(['2016-01-04 16:00+00:00'],source,[1])


def test_cutoff_is_strict_new_york_and_not_silent_utc():
    signal=pd.bdate_range('2016-01-04',periods=2)
    args=dict(signal_dates=signal,source_known=[1,1],
              label_available_at=['2017-01-01 04:59+00:00','2017-01-01 05:00+00:00'],
              feature_available_at=predictor_clock(signal),prediction_at=predictor_clock(signal))
    with pytest.raises(ValueError,match='explicit timezone'):
        V.source_eligibility_model(cutoff='2017-01-01',**args)
    out=V.source_eligibility_model(cutoff=pd.Timestamp('2017-01-01',tz='America/New_York'),**args)
    assert list(out['eligible_mask'])==[True,False]


def test_combine_feature_availability_max_and_unknown_nan():
    price = ["2016-01-04 16:00+00:00", "2016-01-05 16:00+00:00"]
    source = ["2016-01-04 09:31+00:00", "2016-01-06 09:31+00:00"]  # source later than price
    combined = V.combine_feature_availability(price, source, [1, 1])
    assert combined.iloc[0] == pd.Timestamp("2016-01-04 16:00+00:00")
    assert combined.iloc[1] == pd.Timestamp("2016-01-06 09:31+00:00")   # max wins
    unknown = V.combine_feature_availability(price, source, [1, 0])
    assert pd.isna(unknown.iloc[1])                                     # unknown row -> NaN


def test_fit_budget_matches_plan():
    plan = json.loads(PLAN.read_text())
    assert V.fit_budget(plan) == 100
    assert plan["schedule"]["first_eligible_outer"] == 2017
    assert plan["schedule"]["fallback_outer_years"] == [2016]
    assert plan["schedule"]["native_cases"]["count"] == 12
    assert plan["authorization"]["historical_fit_authorized"] is False


@pytest.mark.skipif(not (ROOT / "data/qlib/alfred_tech_iv_initial_20261011_v1").exists(),
                    reason="source store not present")
def test_matched_arm_datasets_prepare_without_fit():
    plan = json.loads(PLAN.read_text())
    out = V.build_matched_arm_datasets(root=ROOT, plan=plan)
    assert out["fit_performed"] is False
    assert len(out["control_features"]) == 25 and len(out["candidate_features"]) == 26
    assert out["control_features"] + ["source_ratio"] == out["candidate_features"]
    assert out["rowset_equal"] is True
    assert out["rows"] == len(out["control_prepared"]) == 3956
    assert out["source_ratio_nan_unknown"] == out["unknown_rows"]  # NaN only on unknown rows
    assert (out['known_rows'],out['unknown_rows'])==(3126,830)
    frame=out['frame'];known=frame.source_known.astype(bool)
    assert frame.index[known][0]==pd.Timestamp('2014-04-22')
    # The bad Oct8 observation becomes the latest released state on Oct9.
    assert frame.loc['2018-10-09','source_known']==0
    assert np.isfinite(frame.loc[known,'source_ratio']).all()
    assert frame.loc[known,'source_available_at'].notna().all()
    assert (frame.loc[known,'feature_available_at']<=frame.loc[known,'decision_at']).all()
    assert frame.loc[~known,'feature_available_at'].isna().all()
    assert out['calendar'][0]==pd.Timestamp('2010-03-11')
    assert out['source_calendar'][0]==pd.Timestamp('2011-01-03')
    assert frame.loc['2014-04-22','source_available_at']==pd.Timestamp('2014-04-22 13:31+00:00')
    audit=V.qualify_matched_annual_inputs(out,plan)
    assert audit['fit_performed'] is False and audit['all_arm_identities_equal']
    assert len(audit['annual_audits'])==10
    assert audit['fallback_audits'][2016]['rows_eligible']<252
    assert audit['annual_audits'][0]['inner_training_rows']>=252
    assert audit['annual_audits'][2]['validation_source_excluded_rows']==1
    # Exercise the exact production boundary: default Qlib serialization drops caches.
    import pickle
    frozen=pickle.loads(pickle.dumps(out))
    restored=V.restore_frozen_arm_datasets(frozen)
    assert V.qualify_matched_annual_inputs(restored,plan)==audit
    pd.testing.assert_frame_equal(restored['candidate'].prepare('all',col_set=['feature','label','metadata']),out['candidate_prepared'])
