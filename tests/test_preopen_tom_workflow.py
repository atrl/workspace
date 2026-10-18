"""One synthetic contract for the no-future-label object-residency boundary."""
import numpy as np
import pandas as pd
import pytest

from src.workflow.preopen_tom_allocation import _native_dataset, audit_training_residency


def test_actual_resident_frames_not_only_prepare_segment_define_training_scope():
    dates=pd.DatetimeIndex(['2022-12-20','2022-12-21'])
    index=pd.MultiIndex.from_arrays([dates,['X','X']],names=['datetime','instrument'])
    columns=pd.MultiIndex.from_tuples([('feature',f'f{i}') for i in range(85)])
    frame=pd.DataFrame(np.zeros((2,85)),index=index,columns=columns)
    dataset=_native_dataset(frame,{'train':('2011-01-03','2022-12-30')})
    audit=audit_training_residency(dataset,expected_rows=2,expected_columns=85,maximum_signal_date='2022-12-21',role='synthetic')
    assert {row['path'].split('.')[-1] for row in audit['resident_frames']} >= {'_data','_infer','_learn'}
    assert not audit['setup_data_called']
    hidden=frame.copy();hidden.index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(['2022-12-20','2023-01-04']),['X','X']],names=index.names)
    full=_native_dataset(hidden,{'train':('2011-01-03','2022-12-30')})
    assert len(full.prepare('train',col_set='__all'))==1
    with pytest.raises(ValueError,match='resident training frame contains unexpected/future'):
        audit_training_residency(full,expected_rows=2,expected_columns=85,maximum_signal_date='2022-12-21',role='synthetic')
    # A loader residual is checked even if the visible three handler caches are clean.
    dataset.handler.data_loader._config={'leaked_frame':hidden}
    with pytest.raises(ValueError,match='resident training frame contains unexpected/future'):
        audit_training_residency(dataset,expected_rows=2,expected_columns=85,maximum_signal_date='2022-12-21',role='synthetic')
    dataset.handler.data_loader._config='/not/an/admitted/reload/path'
    with pytest.raises(ValueError,match='external loader reload'):
        audit_training_residency(dataset,expected_rows=2,expected_columns=85,maximum_signal_date='2022-12-21',role='synthetic')
