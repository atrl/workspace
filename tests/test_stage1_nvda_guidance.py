import torch
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.nvda_own_guidance_realization import FEATURE, FLAGS
from src.model.minute_pit_allocator import MinutePitAllocator
from src.model.stage1_nvda_guidance import (
    Stage1NvdaGuidanceAllocator, initialization_contract,
)
from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationSpec


def _models():
    kwargs = dict(base_features=[f"base_{i}" for i in range(39)],
                  spec=asdict(PitAuxiliaryAllocationSpec(root=Path("."))),
                  seed=71, cutoff="2023-01-01T00:00:00-05:00",
                  train_end="2022-12-30", minimum_observations=3)
    return (MinutePitAllocator(**kwargs, extra_features=[], feature_masks={}),
            Stage1NvdaGuidanceAllocator(**kwargs, mode="neutral"),
            Stage1NvdaGuidanceAllocator(**kwargs, mode="observed"))


def test_guidance_receiver_pair_only_exposes_numeric_to_candidate():
    original, neutral, observed = _models()
    assert all(initialization_contract(original, neutral, observed).values())
    frame = pd.DataFrame({FLAGS[0]: [1., 1., 1., 1., 0.],
                          FLAGS[1]: [1., 1., 1., 0., 0.],
                          FLAGS[2]: [0., 0., 1., 1., 0.],
                          FEATURE: [.1, .2, .3, np.nan, np.nan]})
    for model in (neutral, observed):
        model._fit_extra_scaler(frame.iloc[:4])
    assert neutral.extra_scaler == observed.extra_scaler
    assert observed.extra_scaler[FEATURE]["observations"] == 3
    left, right = neutral._extra_tensor(frame), observed._extra_tensor(frame)
    assert torch.equal(left[:, :len(FLAGS)], right[:, :len(FLAGS)])
    assert left[:, -1].eq(0).all()
    np.testing.assert_allclose(right[:, -1], [-1., 0., 1., 0., 0.], atol=1e-7)
    assert frame[FEATURE].isna().sum() == 2
    with pytest.raises(ValueError, match="unknown guidance"):
        bad = frame.copy(); bad.loc[3, FEATURE] = .7
        observed._extra_tensor(bad)
    with pytest.raises(ValueError, match="too few"):
        observed._fit_extra_scaler(frame.iloc[:2])
    with pytest.raises(ValueError, match="not finite"):
        bad = frame.copy(); bad.loc[1, FEATURE] = np.inf
        observed._extra_tensor(bad)
    with pytest.raises(ValueError, match="flags conflict"):
        bad = frame.copy(); bad.loc[1, FLAGS[0]] = 0.
        observed._extra_tensor(bad)
