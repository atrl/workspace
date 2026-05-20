from __future__ import annotations

import pandas as pd

from src.soxl_quant.backtest import (
    apply_selective_release_overlay,
    selective_release_rebalance_threshold,
)


def test_selective_release_requires_cooldown_and_one_release_signal():
    config = {
        "boost": 0.08,
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
    }
    missing_cooldown = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.10,
            "TRANS_VIX_DIVERGENCE": 0.05,
            "CICC_RETURN_VOLUME_SYNC": 0.72,
        }
    )
    ready = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.30,
            "TRANS_VIX_DIVERGENCE": 0.01,
            "CICC_RETURN_VOLUME_SYNC": 0.72,
        }
    )

    assert apply_selective_release_overlay(0.40, missing_cooldown, 0.50, config) == 0.40
    assert apply_selective_release_overlay(0.40, ready, 0.50, config) == 0.48


def test_selective_release_respects_regime_band_and_never_cuts_position():
    config = {
        "boost": 0.08,
        "max_target": 0.60,
        "min_regime_risk": 0.20,
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
    }
    row = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.40,
            "TRANS_VIX_DIVERGENCE": 0.05,
            "CICC_RETURN_VOLUME_SYNC": 0.50,
        }
    )

    assert apply_selective_release_overlay(0.40, row, 0.70, config) == 0.40
    assert apply_selective_release_overlay(0.66, row, 0.50, config) == 0.66


def test_selective_release_accepts_vix_or_volume_sync_confirmation():
    config = {
        "boost": 0.05,
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
    }
    vix_ready = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.30,
            "TRANS_VIX_DIVERGENCE": 0.05,
            "CICC_RETURN_VOLUME_SYNC": 0.20,
        }
    )
    volume_ready = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.30,
            "TRANS_VIX_DIVERGENCE": 0.01,
            "CICC_RETURN_VOLUME_SYNC": 0.72,
        }
    )

    assert apply_selective_release_overlay(0.40, vix_ready, 0.50, config) == 0.45
    assert apply_selective_release_overlay(0.40, volume_ready, 0.50, config) == 0.45


def test_selective_release_target_lift_can_force_change_from_current_position():
    config = {
        "boost": 0.02,
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
        "min_target_delta": 0.08,
    }
    ready = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.30,
            "TRANS_VIX_DIVERGENCE": 0.01,
            "CICC_RETURN_VOLUME_SYNC": 0.72,
        }
    )

    assert (
        apply_selective_release_overlay(
            target_pos=0.42,
            row=ready,
            regime_risk=0.50,
            overlay_config=config,
            current_pos=0.40,
        )
        == 0.48
    )


def test_selective_release_rebalance_bypass_only_applies_when_gate_is_ready():
    config = {
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
        "rebalance_threshold_override": 0.0,
    }
    ready = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.30,
            "TRANS_VIX_DIVERGENCE": 0.01,
            "CICC_RETURN_VOLUME_SYNC": 0.72,
        }
    )
    missing_confirmation = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.30,
            "TRANS_VIX_DIVERGENCE": 0.01,
            "CICC_RETURN_VOLUME_SYNC": 0.50,
        }
    )

    assert selective_release_rebalance_threshold(0.04, ready, 0.50, config) == 0.0
    assert selective_release_rebalance_threshold(0.04, missing_confirmation, 0.50, config) == 0.04
