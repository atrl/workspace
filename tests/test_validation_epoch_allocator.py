import numpy as np
import pandas as pd
import pytest
import torch
from src.model.validation_epoch_allocator import ValidationEpochAllocator
from src.model.minute_pit_allocator import TARGETS
from src.workflow.minute_pit_allocator import _dataset
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES


def test_validation_excludes_immature_labels_and_rejects_training_overlap():
    dates=pd.date_range('2020-01-01',periods=110)
    features=pd.DataFrame({'x':np.arange(110,dtype=float)},index=dates)
    labels=pd.DataFrame(0.,index=dates,columns=TARGETS)
    common=pd.DataFrame(0.,index=dates,columns=COMMON_FEATURES)
    meta=pd.DataFrame({'label_available_at':dates.tz_localize('America/New_York')},index=dates)
    model=ValidationEpochAllocator.__new__(ValidationEpochAllocator)
    model.train_end=pd.Timestamp('2019-12-31');model.validation_cutoff='2020-04-15 00:00:00-04:00'
    model.base_features=['x'];model.extra_features=[]
    model.primary_scaler={k:{'x':1. if k=='scale' else 0.} for k in ['median','mean','scale']}
    model.common_scaler={k:{f:1. if k=='scale' else 0. for f in COMMON_FEATURES} for k in ['median','mean','scale']}
    def dataset():return _dataset(features,labels,common,meta,{'valid':(None,None)})
    model.prepare_epoch_observation(dataset());before=model.validation_returns.clone()
    late=meta.label_available_at.ge(pd.Timestamp(model.validation_cutoff))
    assert model.validation_audit['rows']==int((~late).sum())
    labels.loc[late,:]=1e5;features.loc[late,'x']=1e5
    model.prepare_epoch_observation(dataset());torch.testing.assert_close(before,model.validation_returns)
    model.train_end=dates[0]
    with pytest.raises(ValueError,match='overlaps'):model.prepare_epoch_observation(dataset())
