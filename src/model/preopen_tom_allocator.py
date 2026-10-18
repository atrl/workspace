"""Four primary allocation inputs on the unchanged shared-head allocator."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from src.model.minute_pit_allocator import MinutePitAllocator
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES


PREOPEN_TOM_MASKS = ("first3_branch_known", "expected_last_branch_known", "proxy_known")
PREOPEN_TOM_FEATURES = (*PREOPEN_TOM_MASKS, "preopen_tom_centered")


class PreopenTomAllocator(MinutePitAllocator):
    """B and C share masks; only C exposes the already-centered calendar value.

    Canonical missing values remain NaN until this model's tensor encoding.
    Training, losses, auxiliary sampling, clipping and prediction are inherited.
    No calendar input is added to the common encoder or auxiliary examples.
    """

    def __init__(self, *, mode: str, base_features: list[str], spec: dict, seed: int,
                 cutoff: str, train_end: str, minimum_observations: int):
        if mode not in ("neutral", "observed"):
            raise ValueError("preopen TOM mode must be neutral or observed")
        if len(base_features) != 39 or len(set(base_features)) != 39:
            raise ValueError("preopen TOM requires the original39 primary feature names")
        if set(base_features).intersection(PREOPEN_TOM_FEATURES):
            raise ValueError("preopen TOM inputs overlap original primary features")
        self.mode = mode
        super().__init__(base_features=base_features, extra_features=list(PREOPEN_TOM_FEATURES),
                         feature_masks={"preopen_tom_centered": "proxy_known"},
                         spec=spec, seed=seed, cutoff=cutoff, train_end=train_end,
                         minimum_observations=minimum_observations)

    @staticmethod
    def _validated_extra(features: pd.DataFrame) -> np.ndarray:
        missing = set(PREOPEN_TOM_FEATURES).difference(features.columns)
        if missing:
            raise ValueError("missing preopen TOM inputs: " + ",".join(sorted(missing)))
        masks = features.loc[:, list(PREOPEN_TOM_MASKS)]
        if not masks.isin([0., 1.]).all().all():
            raise ValueError("preopen TOM masks must be explicit binary values")
        values = features["preopen_tom_centered"].to_numpy(dtype=np.float64, na_value=np.nan)
        known = masks["proxy_known"].to_numpy(dtype=np.float64) == 1.
        if not np.isin(values[known], [-.5, .5]).all():
            raise ValueError("known preopen TOM values must already be centered at -0.5 or +0.5")
        if not np.isnan(values[~known]).all():
            raise ValueError("unknown canonical preopen TOM values must remain NaN")
        encoded = np.zeros((len(features), len(PREOPEN_TOM_FEATURES)), dtype=np.float32)
        encoded[:, :3] = masks.to_numpy(dtype=np.float32)
        encoded[known, 3] = values[known].astype(np.float32)
        return encoded

    def _fit_extra_scaler(self, features: pd.DataFrame) -> None:
        self._validated_extra(features)
        self.extra_scaler = {}

    def _extra_tensor(self, features: pd.DataFrame) -> torch.Tensor:
        values = self._validated_extra(features)
        if self.mode == "neutral":
            values[:, 3] = 0.
        return torch.tensor(values)

    def state_metadata(self) -> dict:
        return {**super().state_metadata(), "preopen_tom_mode": self.mode,
                "calendar_encoding": "three shared binary masks; known centered +/-0.5; unknown latent0",
                "extra_scaler_fitted": False}


def initialization_contract(original: MinutePitAllocator, neutral: PreopenTomAllocator,
                            observed: PreopenTomAllocator) -> dict:
    """Check synthetic initial inputs without fitting or reading labels."""
    if original.extra_features or neutral.mode != "neutral" or observed.mode != "observed":
        raise ValueError("initialization requires original, neutral B and observed C")
    for model in (neutral, observed):
        if (model.base_features != original.base_features or model.spec != original.spec
                or model.seed != original.seed or model.cutoff != original.cutoff
                or model.train_end != original.train_end):
            raise ValueError("paired initialization configuration differs")
    base = original._initialize()
    left = neutral._initialize()
    left_rng = torch.get_rng_state().clone()
    right = observed._initialize()
    if not torch.equal(left_rng, torch.get_rng_state()):
        raise ValueError("paired initialization consumes different RNG states")
    count, extra = len(original.base_features), len(PREOPEN_TOM_FEATURES)
    for name, tensor in left.state_dict().items():
        if not torch.equal(tensor, right.state_dict()[name]):
            raise ValueError("paired initial tensor differs: " + name)
        projected = tensor
        if name == "allocation_head.0.weight":
            if torch.count_nonzero(tensor[:, count:count + extra]):
                raise ValueError("new calendar input weights are not zero")
            projected = torch.cat([tensor[:, :count], tensor[:, count + extra:]], dim=1)
        if not torch.equal(projected, base.state_dict()[name]):
            raise ValueError("original initial tensor differs: " + name)
    primary = torch.linspace(-1., 1., 3 * count).reshape(3, count)
    common = torch.linspace(-.5, .5, 3 * len(COMMON_FEATURES)).reshape(3, -1)
    fixture = pd.DataFrame([[1., 1., 1., .5], [1., 1., 1., -.5], [0., 0., 0., np.nan]],
                           columns=PREOPEN_TOM_FEATURES)
    with torch.no_grad():
        base_outputs = base(primary, common)
        left_outputs = left(torch.cat([primary, neutral._extra_tensor(fixture)], dim=1), common)
        right_outputs = right(torch.cat([primary, observed._extra_tensor(fixture)], dim=1), common)
    if not all(torch.equal(a, b) for a, b in zip(left_outputs, right_outputs)):
        raise ValueError("paired initial outputs differ")
    return {"original_tensors_identical": True, "paired_initial_tensors_identical": True,
            "paired_initial_outputs_identical": True, "paired_initial_rng_state_identical": True,
            "added_columns_zero": True,
            "original_vs_extended_output_max_absolute_error": max(
                float((a - b).abs().max()) for a, b in zip(base_outputs, left_outputs)),
            "original_parameters": sum(p.numel() for p in base.parameters()),
            "paired_parameters": sum(p.numel() for p in left.parameters()),
            "paired_initial_sha256": hashlib.sha256(b"".join(
                name.encode() + value.numpy().tobytes() for name, value in sorted(left.state_dict().items())
            )).hexdigest()}
