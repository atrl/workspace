from __future__ import annotations

import torch
import pytest

from src.soxl_quant.config import FeatureConfig, ModelConfig, RewardConfig, StrategyConfig
from src.soxl_quant.app.artifacts import serialize_model_config
from src.soxl_quant.data import feature_columns_for_packs
from src.soxl_quant.policy.policy_builder import _build_net


def _reward_config() -> RewardConfig:
    return RewardConfig(
        return_scale=1.0,
        transaction_cost=0.0,
        turnover_penalty=0.0,
        drawdown_penalty=0.0,
        downside_penalty=0.0,
        risk_budget_penalty=0.0,
        high_exposure_penalty=0.0,
        smoothness_penalty=0.0,
        activity_bonus=0.0,
        inactivity_penalty=0.0,
    )


def _profile(model: ModelConfig) -> StrategyConfig:
    return StrategyConfig(
        name="temporal_conv_probe",
        max_iters=1,
        episode_per_iter=1,
        episode_len=20,
        hidden_sizes=(32, 16),
        lr=1e-4,
        update_batch_size=8,
        update_repeat=1,
        reward=_reward_config(),
        feature=FeatureConfig(packs=()),
        model=model,
    )


def test_temporal_conv_model_flag_routes_to_tcn_network():
    model = ModelConfig(
        observation_window=5,
        use_temporal_conv=True,
        temporal_conv_channels=12,
        temporal_conv_kernel=3,
    )
    profile = _profile(model)
    feature_dim = len(feature_columns_for_packs(profile.feature.packs, profile.name))
    obs_dim = feature_dim * model.observation_window + 3

    net = _build_net(obs_dim, profile, profile.model)

    assert type(net).__name__ == "TemporalConvNet"
    assert net.input_dim == obs_dim
    assert net.output_dim == profile.hidden_sizes[-1]


@pytest.mark.parametrize(
    ("model", "expected_class"),
    [
        (ModelConfig(observation_window=5), "Net"),
        (ModelConfig(observation_window=5, use_gru=True), "GRUNet"),
        (ModelConfig(observation_window=5, use_attention=True), "WindowSelfAttentionNet"),
        (ModelConfig(observation_window=5, use_tide=True), "TiDENet"),
        (ModelConfig(observation_window=5, use_regime_router=True), "RegimeMoENet"),
        (ModelConfig(observation_window=5, use_temporal_conv=True), "TemporalConvNet"),
    ],
)
def test_model_flags_route_to_registered_networks(model, expected_class):
    profile = _profile(model)
    feature_dim = len(feature_columns_for_packs(profile.feature.packs, profile.name))
    obs_dim = feature_dim * model.observation_window + 3

    net = _build_net(obs_dim, profile, profile.model)

    assert type(net).__name__ == expected_class


def test_temporal_conv_network_preserves_batch_shape_with_static_features():
    model = ModelConfig(
        observation_window=4,
        use_temporal_conv=True,
        temporal_conv_channels=8,
        temporal_conv_kernel=3,
    )
    profile = _profile(model)
    feature_dim = len(feature_columns_for_packs(profile.feature.packs, profile.name))
    obs_dim = feature_dim * model.observation_window + 2
    net = _build_net(obs_dim, profile, profile.model)

    output, state = net(torch.randn(7, obs_dim))

    assert state is None
    assert output.shape == (7, profile.hidden_sizes[-1])


def test_temporal_conv_config_is_serialized_for_training_artifacts():
    model = ModelConfig(
        observation_window=4,
        use_temporal_conv=True,
        temporal_conv_channels=8,
        temporal_conv_kernel=5,
    )
    payload = serialize_model_config(_profile(model))

    assert payload["use_temporal_conv"] is True
    assert payload["temporal_conv_channels"] == 8
    assert payload["temporal_conv_kernel"] == 5
