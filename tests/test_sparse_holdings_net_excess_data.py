"""Native source reuse and explicit new-policy/old-scaler identity contracts."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_constrained_net_excess_data import frozen_parent, synthetic_source
from test_benchmark_anchor_workflow import history
from src.data.benchmark_anchor_utility import build_anchor_dataset
from src.data.sparse_holdings_net_excess import (
    VARIANT_MAPPING, load_sparse_inputs, parent_preparation_plan, prepare_folds,
    transform_for_variant, variant_mapping,
)


@pytest.fixture
def plan():
    return json.loads((Path(__file__).parents[1]/'research/sparse_holdings_net_excess_20261003/selection_plan.json').read_text())


def test_mapping_is_explicit_and_never_mutates_registration_or_scalers(frozen_parent, plan):
    root, original_plan, objects, _ = frozen_parent
    plan['source']=original_plan['source'];plan['parent_input_source']=original_plan['parent_input_source']
    plan['training']['development_signal_years']=[2016]
    prior_plan=copy.deepcopy(plan)
    previous={identity:copy.deepcopy(fold['scaler']) for identity,fold in objects['parent_dev_folds'].items()}
    bundle=load_sparse_inputs(root,plan)
    assert plan==prior_plan
    assert bundle['source_audit']['mapped_development_folds']==3
    assert bundle['source_audit']['forecast_refit_performed'] is False
    assert bundle['source_audit']['holdings_generated_by_owned_policy_only'] is True
    for (new,year),fold in bundle['development_folds'].items():
        old=VARIANT_MAPPING[new];parent=objects['parent_dev_folds'][(old,year)]
        assert fold['variant']==new and fold['parent_market_variant']==old
        assert fold['scaler']==previous[(old,year)] and fold['scaler']['variant']==old
        np.testing.assert_array_equal(fold['features'],parent['features'])
        np.testing.assert_array_equal(fold['returns'],parent['returns'])
        assert fold['features'].shape[1]==27
    assert [v['name'] for v in parent_preparation_plan(plan)['variants']]==list(VARIANT_MAPPING.values())


@pytest.mark.parametrize('fault',['duplicate','wrong_mapping','wrong_mask'])
def test_aliases_cannot_silently_change_information_sets(plan,fault):
    if fault=='duplicate':plan['variants'][1]['name']='holdings_only'
    elif fault=='wrong_mapping':plan['inputs']['variant_mapping']['price_holdings']='price_vol'
    else:plan['variants'][1]['enabled_market_features']=27
    with pytest.raises(ValueError,match='families|masks|channels'):
        variant_mapping(plan)


def test_unused_forecast_errors_preserve_common_native_rows_and_controls(history,plan):
    frame=history.copy();frame.loc['2014-06-02','future_vol_prediction']=np.nan
    dataset=build_anchor_dataset(frame,as_of='2017-01-10')
    plan['training']['minimum_rows']=63
    folds=prepare_folds(dataset,list(VARIANT_MAPPING),[2015],plan)
    assert len({len(fold['rows']) for fold in folds.values()})==1
    for name in ('holdings_only','price_holdings'):
        assert not folds[(name,2015)]['invalid_training']
    bad=folds[('price_vol_holdings',2015)]
    assert bad['invalid_training'] and '2014-06-02' in bad['invalid_training_dates']
    scaler=folds[('price_holdings',2015)]['scaler']
    with pytest.raises(ValueError,match='different parent scaler mask'):
        transform_for_variant(frame,scaler,variant='price_vol_holdings',plan=plan)


def test_development_matrix_and_scaler_reconstruction_remains_exact(frozen_parent,plan):
    _,_,obj,_=frozen_parent
    dataset=build_anchor_dataset(obj['forecasts'],as_of='2016-01-15')
    bad=copy.deepcopy(obj['parent_dev_folds']);bad[('price',2016)]['features'][0,0]+=.1
    with pytest.raises(ValueError,match='rows or matrices'):
        prepare_folds(dataset,['price_holdings'],[2016],plan,parent_dev_folds=bad)
