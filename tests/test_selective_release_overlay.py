from __future__ import annotations

import pandas as pd

from src.soxl_quant.backtest import (
    apply_selective_release_overlay,
    selective_release_execution_guard_ready,
    selective_release_post_release_de_risk_ready,
    selective_release_post_release_rebalance_threshold,
    selective_release_post_release_target_cap,
    selective_release_post_release_trade_lock_days,
    selective_release_rebalance_threshold,
    selective_release_trade_lock_days,
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


def test_selective_release_overlay_guard_blocks_boost_when_drawdown_or_gap_fails():
    config = {
        "boost": 0.08,
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
        "max_current_drawdown": 0.03,
        "min_release_exposure_gap": 0.10,
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
            target_pos=0.50,
            row=ready,
            regime_risk=0.50,
            overlay_config=config,
            current_pos=0.40,
            current_drawdown=0.02,
        )
        == 0.58
    )
    assert (
        apply_selective_release_overlay(
            target_pos=0.50,
            row=ready,
            regime_risk=0.50,
            overlay_config=config,
            current_pos=0.40,
            current_drawdown=0.05,
        )
        == 0.50
    )
    assert (
        apply_selective_release_overlay(
            target_pos=0.50,
            row=ready,
            regime_risk=0.50,
            overlay_config=config,
            current_pos=0.49,
            current_drawdown=0.02,
        )
        == 0.50
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


def test_selective_release_trade_lock_override_only_applies_when_gate_is_ready():
    config = {
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
        "trade_lock_override": 0,
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

    assert selective_release_trade_lock_days(8, ready, 0.50, config) == 0
    assert selective_release_trade_lock_days(8, missing_confirmation, 0.50, config) == 8

    shortened_config = {**config, "trade_lock_override": 1}
    assert selective_release_trade_lock_days(8, ready, 0.50, shortened_config) == 1

    capped_config = {
        **config,
        "trade_lock_cap": 1,
    }
    del capped_config["trade_lock_override"]
    assert selective_release_trade_lock_days(8, ready, 0.50, capped_config) == 1
    assert selective_release_trade_lock_days(0, ready, 0.50, capped_config) == 0


def test_selective_release_execution_guard_requires_drawdown_room_and_exposure_gap():
    config = {
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
        "max_current_drawdown": 0.03,
        "min_release_exposure_gap": 0.10,
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

    assert selective_release_execution_guard_ready(
        ready,
        0.50,
        config,
        current_pos=0.50,
        target_pos=0.65,
        current_drawdown=0.02,
    )
    assert not selective_release_execution_guard_ready(
        ready,
        0.50,
        config,
        current_pos=0.50,
        target_pos=0.65,
        current_drawdown=0.05,
    )
    assert not selective_release_execution_guard_ready(
        ready,
        0.50,
        config,
        current_pos=0.58,
        target_pos=0.65,
        current_drawdown=0.02,
    )
    assert not selective_release_execution_guard_ready(
        ready,
        0.50,
        config,
        current_pos=0.50,
        target_pos=0.65,
    )
    assert not selective_release_execution_guard_ready(
        missing_confirmation,
        0.50,
        config,
        current_pos=0.50,
        target_pos=0.65,
        current_drawdown=0.02,
    )


def test_selective_release_execution_relaxations_respect_guard_context():
    config = {
        "max_regime_risk": 0.62,
        "cooldown_threshold": 0.25,
        "vix_threshold": 0.04,
        "volume_sync_threshold": 0.68,
        "rebalance_threshold_override": 0.0,
        "trade_lock_cap": 1,
        "max_current_drawdown": 0.03,
        "min_release_exposure_gap": 0.10,
    }
    ready = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.30,
            "TRANS_VIX_DIVERGENCE": 0.01,
            "CICC_RETURN_VOLUME_SYNC": 0.72,
        }
    )

    assert (
        selective_release_rebalance_threshold(
            0.04,
            ready,
            0.50,
            config,
            current_pos=0.50,
            target_pos=0.65,
            current_drawdown=0.02,
        )
        == 0.0
    )
    assert (
        selective_release_rebalance_threshold(
            0.04,
            ready,
            0.50,
            config,
            current_pos=0.50,
            target_pos=0.65,
            current_drawdown=0.05,
        )
        == 0.04
    )
    assert (
        selective_release_trade_lock_days(
            8,
            ready,
            0.50,
            config,
            current_pos=0.50,
            target_pos=0.65,
            current_drawdown=0.02,
        )
        == 1
    )
    assert (
        selective_release_trade_lock_days(
            8,
            ready,
            0.50,
            config,
            current_pos=0.58,
            target_pos=0.65,
            current_drawdown=0.02,
        )
        == 8
    )


def test_selective_release_post_release_cap_only_applies_after_drawdown_trigger():
    config = {
        "post_release_state_days": 90,
        "post_release_drawdown_threshold": 0.04,
        "post_release_target_cap": 0.55,
    }

    assert not selective_release_post_release_de_risk_ready(
        config,
        release_state_days=0,
        release_drawdown=0.10,
    )
    assert not selective_release_post_release_de_risk_ready(
        config,
        release_state_days=5,
        release_drawdown=0.03,
    )
    assert selective_release_post_release_de_risk_ready(
        config,
        release_state_days=5,
        release_drawdown=0.05,
    )
    assert (
        selective_release_post_release_target_cap(
            0.80,
            config,
            release_state_days=5,
            release_drawdown=0.05,
        )
        == 0.55
    )
    assert (
        selective_release_post_release_target_cap(
            0.45,
            config,
            release_state_days=5,
            release_drawdown=0.05,
        )
        == 0.45
    )
    assert (
        selective_release_post_release_target_cap(
            0.80,
            config,
            release_state_days=5,
            release_drawdown=0.03,
        )
        == 0.80
    )


def test_selective_release_post_release_de_risk_only_relaxes_sells():
    config = {
        "post_release_state_days": 90,
        "post_release_drawdown_threshold": 0.04,
        "post_release_target_cap": 0.55,
        "post_release_rebalance_threshold_override": 0.0,
        "post_release_trade_lock_override": 0,
    }

    assert (
        selective_release_post_release_rebalance_threshold(
            0.04,
            config,
            release_state_days=5,
            release_drawdown=0.05,
            direction="sell",
        )
        == 0.0
    )
    assert (
        selective_release_post_release_trade_lock_days(
            8,
            config,
            release_state_days=5,
            release_drawdown=0.05,
            direction="sell",
        )
        == 0
    )
    assert (
        selective_release_post_release_rebalance_threshold(
            0.04,
            config,
            release_state_days=5,
            release_drawdown=0.05,
            direction="buy",
        )
        == 0.04
    )
    assert (
        selective_release_post_release_trade_lock_days(
            8,
            config,
            release_state_days=5,
            release_drawdown=0.05,
            direction="buy",
        )
        == 8
    )
