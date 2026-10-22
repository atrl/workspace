"""Original Stage1 allocation receiver for a dated NVDA guidance state."""
from __future__ import annotations

import torch
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.data.nvda_own_guidance_realization import FEATURE, FLAGS
from src.model.minute_pit_allocator import MinutePitAllocator
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES


class Stage1NvdaGuidanceAllocator(MinutePitAllocator):
    """Paired Stage1 heads with identical source flags and one numeric contrast."""

    def __init__(self, *, mode: str, base_features: list[str], spec: dict,
                 seed: int, cutoff: str, train_end: str,
                 minimum_observations: int):
        if mode not in ("neutral", "observed"):
            raise ValueError("guidance mode must be neutral or observed")
        if len(base_features) != 39 or len(set(base_features)) != 39:
            raise ValueError("blocked: original 39 Stage1 primary features required")
        if set(base_features).intersection([*FLAGS, FEATURE]):
            raise ValueError("blocked: guidance coordinate already in Stage1")
        self.mode = mode
        super().__init__(base_features=base_features,
                         extra_features=[*FLAGS, FEATURE],
                         feature_masks={FEATURE: FLAGS[1]}, spec=spec,
                         seed=seed, cutoff=cutoff, train_end=train_end,
                         minimum_observations=minimum_observations)

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        for flag in FLAGS:
            if not frame[flag].isin([0., 1.]).all():
                raise ValueError("blocked: guidance flags must be binary")
        if frame.loc[frame[FLAGS[1]].eq(0), FEATURE].notna().any():
            raise ValueError("blocked: unknown guidance value carried forward")
        if frame.loc[frame[FLAGS[1]].eq(1), FEATURE].isna().any():
            raise ValueError("blocked: comparable guidance value missing")

    def _fit_extra_scaler(self, features: pd.DataFrame) -> None:
        self._validate(features)
        observed = features.loc[features[FLAGS[1]].eq(1), FEATURE].astype(float)
        if not np.isfinite(observed).all() or len(observed) < self.minimum_observations:
            raise ValueError("needs_data: too few finite guidance training values")
        scale = float(observed.std(ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("needs_data: constant guidance training values")
        self.extra_scaler = {FEATURE: {
            "mean": float(observed.mean()), "scale": scale,
            "observations": len(observed), "active": True,
        }}

    def _extra_tensor(self, features: pd.DataFrame) -> torch.Tensor:
        self._validate(features)
        if FEATURE not in self.extra_scaler:
            raise ValueError("blocked: guidance scaler not fitted")
        values = np.zeros((len(features), len(FLAGS) + 1), dtype=np.float32)
        values[:, :len(FLAGS)] = features.loc[:, FLAGS].to_numpy(np.float32)
        known = features[FLAGS[1]].eq(1)
        if self.mode == "observed":
            scaler = self.extra_scaler[FEATURE]
            values[known.to_numpy(), -1] = (
                (features.loc[known, FEATURE] - scaler["mean"]) / scaler["scale"]
            ).clip(-8., 8.).to_numpy(np.float32)
        return torch.tensor(values)

    def state_metadata(self) -> dict:
        return {**super().state_metadata(), "guidance_mode": self.mode,
                "guidance_receiver": "original_Stage1_primary_allocation_only"}


def initialization_contract(original: MinutePitAllocator,
                            neutral: Stage1NvdaGuidanceAllocator,
                            observed: Stage1NvdaGuidanceAllocator) -> dict:
    """Verify zero-initialized additions and exact paired initialization."""
    if original.extra_features or neutral.mode != "neutral" or observed.mode != "observed":
        raise ValueError("blocked: wrong guidance initialization arms")
    if not (original.base_features == neutral.base_features == observed.base_features):
        raise ValueError("blocked: base feature order differs")
    if not (asdict(original.spec) == asdict(neutral.spec) == asdict(observed.spec)):
        raise ValueError("blocked: Stage1 model specification differs")
    if not (original.seed == neutral.seed == observed.seed):
        raise ValueError("blocked: initialization seed differs")
    base, left, right = original._initialize(), neutral._initialize(), observed._initialize()
    for name, tensor in left.state_dict().items():
        if not torch.equal(tensor, right.state_dict()[name]):
            raise ValueError("blocked: paired initial tensor differs: " + name)
        if name == "allocation_head.0.weight":
            count = len(original.base_features)
            extra = len(FLAGS) + 1
            if torch.count_nonzero(tensor[:, count:count + extra]):
                raise ValueError("blocked: added guidance weights are not zero")
            tensor = torch.cat([tensor[:, :count], tensor[:, count + extra:]], dim=1)
        if not torch.equal(tensor, base.state_dict()[name]):
            raise ValueError("blocked: original Stage1 initialization differs: " + name)
    primary = torch.linspace(-1, 1, 3 * len(original.base_features)).reshape(3, -1)
    common = torch.linspace(-.5, .5, 3 * len(COMMON_FEATURES)).reshape(3, -1)
    extra = torch.linspace(-.5, .5, 3 * (len(FLAGS) + 1)).reshape(3, -1)
    with torch.no_grad():
        base_outputs = base(primary, common)
        neutral_outputs = left(torch.cat([primary, extra], dim=1), common)
        observed_outputs = right(torch.cat([primary, extra], dim=1), common)
    for a, b, c in zip(base_outputs, neutral_outputs, observed_outputs):
        if not torch.equal(a, b) or not torch.equal(b, c):
            raise ValueError("blocked: guidance initial output differs")
    return {"original_tensors_equal_after_projection": True,
            "paired_initial_tensors_equal": True,
            "all_initial_outputs_equal": True,
            "added_weights_zero": True}
