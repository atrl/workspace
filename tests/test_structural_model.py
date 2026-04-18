from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from src.soxl_quant.data import FEATURE_COLUMNS
from src.soxl_quant.rl import build_initial_states, make_observation, observation_dim_for_profile, TrancheLedger
from src.soxl_quant.settings import ModelConfig, RewardConfig, SamplingConfig, StrategyConfig


def _synthetic_frame(rows: int = 120) -> pd.DataFrame:
    data = {}
    base = np.linspace(-0.02, 0.02, rows)
    for i, name in enumerate(FEATURE_COLUMNS):
        data[name] = base + i * 0.001
    data["regime_risk"] = np.linspace(0.30, 0.80, rows)
    data["cum_drop_30"] = np.linspace(-0.05, -0.20, rows)
    data["drawdown_streak_63"] = np.linspace(0.10, 0.70, rows)
    data["below_ma63_days"] = np.linspace(0.10, 0.65, rows)
    data["rolling_new_low_ratio_63"] = np.linspace(0.05, 0.40, rows)
    data["negative_gap_frequency_21"] = np.linspace(0.05, 0.35, rows)
    data["return"] = np.sin(np.linspace(0.0, 3.14, rows)) * 0.01
    return pd.DataFrame(data)


def _structural_profile() -> StrategyConfig:
    return StrategyConfig(
        name="balanced_structural_v1",
        max_iters=80,
        episode_per_iter=10,
        episode_len=300,
        hidden_sizes=(128, 64),
        lr=3e-4,
        update_batch_size=64,
        update_repeat=10,
        reward=RewardConfig(
            return_scale=1.15,
            transaction_cost=0.08,
            turnover_penalty=0.05,
            drawdown_penalty=0.28,
            downside_penalty=0.20,
            risk_budget_penalty=0.24,
            high_exposure_penalty=0.08,
            smoothness_penalty=0.03,
            activity_bonus=1.25,
            inactivity_penalty=0.80,
        ),
        model=ModelConfig(
            observation_window=10,
            action_levels=(0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90),
            use_regime_router=True,
            hard_regime_routing=True,
        ),
        sampling=SamplingConfig(
            regime_stratified=True,
            mild_repeat=2,
            severe_repeat=4,
            persistent_stress_days=0.28,
        ),
    )


def test_observation_window_expands_obs_dim():
    profile = _structural_profile()
    obs_dim = observation_dim_for_profile(profile)
    expected = len(FEATURE_COLUMNS) * profile.model.observation_window + 15 + 5
    assert obs_dim == expected


def test_make_observation_respects_window_layout():
    profile = _structural_profile()
    df = _synthetic_frame(100)
    ledger = TrancheLedger()
    price_path = np.linspace(100.0, 110.0, len(df), dtype=np.float32)
    obs = make_observation(
        df,
        idx=80,
        pos=0.2,
        value=102_000.0,
        peak_value=105_000.0,
        ledger=ledger,
        price_path=price_path,
        observation_window=profile.model.observation_window,
    )
    assert obs.shape[0] == observation_dim_for_profile(profile)
    current = df.iloc[80][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    assert np.allclose(obs[: len(FEATURE_COLUMNS)], current)


def test_regime_stratified_sampling_oversamples_persistent_stress():
    profile = _structural_profile()
    df = _synthetic_frame(900)
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "train.pkl"
        df.to_pickle(pkl_path)
        states = build_initial_states([(pkl_path, df, 0)], profile, smoke=False)
        unique_states = {(state.start_idx, state.end_idx) for state in states}
        assert len(states) > len(unique_states)
