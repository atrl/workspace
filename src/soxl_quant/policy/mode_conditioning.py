from __future__ import annotations

import numpy as np

MODE_ONE_HOT_ORDER = ("trend_hold", "defensive_hold", "crash_rebuild", "flat_wait")
_MODE_INDEX = {name: idx for idx, name in enumerate(MODE_ONE_HOT_ORDER)}


def mode_one_hot(mode: str) -> np.ndarray:
    vec = np.zeros(len(MODE_ONE_HOT_ORDER), dtype=np.float32)
    vec[_MODE_INDEX.get(str(mode), _MODE_INDEX["flat_wait"])] = 1.0
    return vec


def append_mode_one_hot(obs: np.ndarray, mode: str) -> np.ndarray:
    base = np.asarray(obs, dtype=np.float32)
    return np.concatenate([base, mode_one_hot(mode)], axis=0)


__all__ = ["MODE_ONE_HOT_ORDER", "append_mode_one_hot", "mode_one_hot"]
