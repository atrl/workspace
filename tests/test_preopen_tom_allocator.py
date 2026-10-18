"""Synthetic contract for nullable calendar encoding and matched initialization."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.model.minute_pit_allocator import MinutePitAllocator
from src.model.preopen_tom_allocator import (
    PREOPEN_TOM_FEATURES, PreopenTomAllocator, initialization_contract,
)


def test_preopen_tom_keeps_unknown_distinct_and_controls_initial_information():
    kwargs = dict(base_features=[f"original_{i}" for i in range(39)], spec={"root": "."},
                  seed=71, cutoff="2023-01-01T00:00:00-05:00", train_end="2022-12-30",
                  minimum_observations=30)
    original = MinutePitAllocator(**kwargs, extra_features=[], feature_masks={})
    neutral = PreopenTomAllocator(**kwargs, mode="neutral")
    observed = PreopenTomAllocator(**kwargs, mode="observed")
    features = pd.DataFrame([[1., 1., 1., .5], [1., 1., 1., -.5], [0., 0., 0., np.nan]],
                            columns=PREOPEN_TOM_FEATURES)
    before = features.copy(deep=True)
    neutral._fit_extra_scaler(features)
    observed._fit_extra_scaler(features)
    left, right = neutral._extra_tensor(features), observed._extra_tensor(features)
    assert torch.equal(left[:, :3], right[:, :3])
    assert torch.equal(left[:, 3], torch.zeros(3))
    assert torch.equal(right[:, 3], torch.tensor([.5, -.5, 0.]))
    assert neutral.extra_scaler == observed.extra_scaler == {}
    pd.testing.assert_frame_equal(features, before)
    audit = initialization_contract(original, neutral, observed)
    assert audit["paired_initial_outputs_identical"] and audit["original_tensors_identical"]
    assert audit["added_columns_zero"] and audit["paired_initial_rng_state_identical"]
    for row, value, message in [(2, 0., "must remain NaN"), (0, 1., "already be centered")]:
        invalid = features.copy()
        invalid.loc[row, "preopen_tom_centered"] = value
        with pytest.raises(ValueError, match=message):
            neutral._extra_tensor(invalid)
