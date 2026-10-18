"""Six primary-only FINRA coordinates on the original shared-head allocator."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from src.data.finra_allocation_features import (
    FINRA_BINARY_FEATURES, FINRA_FEATURE_MASKS, FINRA_FEATURES, FINRA_NUMERIC_FEATURES,
)
from src.model.minute_pit_allocator import MinutePitAllocator
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES


class FinraAllocator(MinutePitAllocator):
    """B and C differ only in whether transformed q is exposed to the model.

    Inherited fit first selects original mature primary training rows and only
    then calls this scaler. No source coverage filter removes a primary row.
    The shared encoder, original classifiers, auxiliary labels and objective
    remain inherited unchanged. FINRA flags do not modify minute ``FLAGS``.
    """

    def __init__(self, *, mode: str, base_features: list[str], spec: dict, seed: int,
                 cutoff: str, train_end: str, minimum_observations: int):
        if mode not in ("neutral", "observed"):
            raise ValueError("FINRA mode must be neutral or observed")
        if len(base_features) != 39 or len(set(base_features)) != 39:
            raise ValueError("blocked: FINRA adapter requires the original 39 primary features")
        if set(base_features).intersection(FINRA_FEATURES):
            raise ValueError("blocked: FINRA columns already exist in the base feature set")
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        self.mode = mode
        super().__init__(base_features=base_features, extra_features=list(FINRA_FEATURES),
                         feature_masks=FINRA_FEATURE_MASKS, spec=spec, seed=seed,
                         cutoff=cutoff, train_end=train_end, minimum_observations=minimum_observations)

    @staticmethod
    def _validate_flags(features: pd.DataFrame) -> None:
        if not features.loc[:, list(FINRA_BINARY_FEATURES)].isin([0., 1.]).all().all():
            raise ValueError("blocked: FINRA flags must be explicit binary values")

    def _fit_extra_scaler(self, features: pd.DataFrame) -> None:
        self._validate_flags(features)
        self.extra_scaler = {}
        for name in FINRA_NUMERIC_FEATURES:
            known = features.finra_both_ready.eq(1) & np.isfinite(features[name])
            observed = features.loc[known, name].astype(float)
            active = len(observed) >= self.minimum_observations
            deviation = observed.std(ddof=1) if active else np.nan
            self.extra_scaler[name] = {
                "mean": float(observed.mean()) if active else 0.,
                "scale": float(deviation) if np.isfinite(deviation) and deviation > 0 else 1.,
                "observations": len(observed), "active": active,
            }

    def _extra_tensor(self, features: pd.DataFrame) -> torch.Tensor:
        self._validate_flags(features)
        values = np.zeros((len(features), len(FINRA_FEATURES)), dtype=np.float32)
        values[:, :len(FINRA_BINARY_FEATURES)] = features.loc[:, list(FINRA_BINARY_FEATURES)].to_numpy(np.float32)
        for number, name in enumerate(FINRA_FEATURES[len(FINRA_BINARY_FEATURES):], start=len(FINRA_BINARY_FEATURES)):
            scaler = self.extra_scaler[name]
            if not scaler["active"]:
                continue
            known = features.finra_both_ready.eq(1) & np.isfinite(features[name])
            values[known.to_numpy(), number] = (
                (features.loc[known, name] - scaler["mean"]) / scaler["scale"]
            ).clip(-8., 8.).to_numpy(np.float32)
        if self.mode == "neutral":
            values[:, FINRA_FEATURES.index("finra_short_mix")] = 0.
        return torch.tensor(values)

    def state_metadata(self) -> dict:
        return {**super().state_metadata(), "finra_mode": self.mode,
                "finra_source_semantics": "conditional historical short-mark composition, not short interest"}


def initialization_contract(original: MinutePitAllocator, neutral: FinraAllocator,
                            observed: FinraAllocator) -> dict:
    """Inspect paired initialization without fitting or reading any labels."""
    if original.extra_features or neutral.mode != "neutral" or observed.mode != "observed":
        raise ValueError("blocked: initialization requires original, neutral B and observed C")
    for model in (neutral, observed):
        if (model.base_features != original.base_features or model.spec != original.spec
                or model.seed != original.seed):
            raise ValueError("blocked: paired model initialization identities differ")
    base = original._initialize()
    left = neutral._initialize()
    left_rng = torch.get_rng_state().clone()
    right = observed._initialize()
    if not torch.equal(left_rng, torch.get_rng_state()):
        raise ValueError("blocked: paired initialization consumes different RNG states")
    count, extra = len(original.base_features), len(FINRA_FEATURES)
    for name, tensor in left.state_dict().items():
        if not torch.equal(tensor, right.state_dict()[name]):
            raise ValueError("blocked: paired initial tensor differs: " + name)
        projected = tensor
        if name == "allocation_head.0.weight":
            if torch.count_nonzero(tensor[:, count:count + extra]):
                raise ValueError("blocked: FINRA input weights did not initialize at zero")
            projected = torch.cat([tensor[:, :count], tensor[:, count + extra:]], dim=1)
        if not torch.equal(projected, base.state_dict()[name]):
            raise ValueError("blocked: original initial tensor differs: " + name)
    primary = torch.linspace(-1, 1, 3 * count).reshape(3, count)
    common = torch.linspace(-.5, .5, 3 * len(COMMON_FEATURES)).reshape(3, -1)
    extra_x = torch.linspace(-1, 1, 3 * extra).reshape(3, extra)
    neutral_x = extra_x.clone()
    neutral_x[:, FINRA_FEATURES.index("finra_short_mix")] = 0.
    with torch.no_grad():
        base_outputs = base(primary, common)
        left_outputs = left(torch.cat([primary, neutral_x], dim=1), common)
        right_outputs = right(torch.cat([primary, extra_x], dim=1), common)
    for a, b in zip(left_outputs, right_outputs):
        if not torch.equal(a, b):
            raise ValueError("blocked: paired initial outputs differ")
    maximum_error = max(float((a - b).abs().max()) for a, b in zip(base_outputs, left_outputs))
    return {"original_tensors_identical": True, "paired_initial_tensors_identical": True,
            "paired_initial_outputs_identical": True, "paired_initial_rng_state_identical": True,
            "added_columns_zero": True, "original_vs_extended_output_max_absolute_error": maximum_error,
            "original_parameters": sum(p.numel() for p in base.parameters()),
            "paired_parameters": sum(p.numel() for p in left.parameters()),
            "paired_initial_sha256": hashlib.sha256(b"".join(
                name.encode() + value.numpy().tobytes() for name, value in sorted(left.state_dict().items())
            )).hexdigest()}
