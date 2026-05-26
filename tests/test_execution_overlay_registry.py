from __future__ import annotations

from src.soxl_quant.policy import execution_overlays
from src.soxl_quant.policy.execution_overlays import (
    POST_RELEASE_CAP_VARIANT,
    SOXX_RELATIVE_WEAKNESS_CAP_VARIANT,
    apply_short_volshock_action_cap,
    apply_sequence_risk_action_cap,
    apply_soxx_relative_weakness_cap,
    overlay_config,
    overlay_variants,
    short_volshock_action_cap_ready,
    sequence_risk_action_cap_ready,
    soxx_overlay_config,
    soxx_overlay_variants,
    soxx_relative_weakness_ready,
)


def test_post_release_cap_config_is_governed_named_overlay():
    config = overlay_config(POST_RELEASE_CAP_VARIANT)

    assert config["type"] == "selective_release"
    assert config["boost"] == 0.08
    assert config["trade_lock_cap"] == 1
    assert config["max_current_drawdown"] == 0.03
    assert config["min_release_exposure_gap"] == 0.10
    assert config["post_release_state_days"] == 90
    assert config["post_release_drawdown_threshold"] == 0.04
    assert config["post_release_target_cap"] == 0.55
    assert config["post_release_rebalance_threshold_override"] == 0.0
    assert config["post_release_trade_lock_override"] == 0


def test_overlay_config_returns_copy_and_variants_keep_baseline_first():
    config = overlay_config(POST_RELEASE_CAP_VARIANT)
    config["boost"] = 999.0

    assert overlay_config(POST_RELEASE_CAP_VARIANT)["boost"] == 0.08
    variants = overlay_variants()
    assert variants[0] == ("baseline", None)
    assert POST_RELEASE_CAP_VARIANT in {name for name, _ in variants}


def test_soxx_overlay_config_returns_copy_and_variants_keep_baseline_first():
    config = soxx_overlay_config(SOXX_RELATIVE_WEAKNESS_CAP_VARIANT)
    config["target_cap"] = 999.0

    assert soxx_overlay_config(SOXX_RELATIVE_WEAKNESS_CAP_VARIANT)["target_cap"] == 0.35
    variants = soxx_overlay_variants()
    assert variants[0] == ("baseline", None)
    assert SOXX_RELATIVE_WEAKNESS_CAP_VARIANT in {name for name, _ in variants}


def test_soxx_relative_weakness_cap_clamps_only_when_weakness_has_no_reversion_cushion():
    config = {
        "min_regime_risk": 0.45,
        "max_regime_risk": 0.90,
        "soxx_bias_max": -0.02,
        "price_ratio_mom_max": 0.0,
        "signed_te_reversion_max": 0.50,
        "target_cap": 0.35,
    }
    weak_without_reversion = {
        "SOXX_BIAS_10": -0.04,
        "PRICE_RATIO_MOM": -0.03,
        "SIGNED_TE_REVERSION_10": 0.20,
    }
    weak_with_reversion = {
        "SOXX_BIAS_10": -0.04,
        "PRICE_RATIO_MOM": -0.03,
        "SIGNED_TE_REVERSION_10": 0.80,
    }

    assert soxx_relative_weakness_ready(weak_without_reversion, 0.62, config) is True
    assert apply_soxx_relative_weakness_cap(0.58, weak_without_reversion, 0.62, config) == 0.35
    assert soxx_relative_weakness_ready(weak_with_reversion, 0.62, config) is False
    assert apply_soxx_relative_weakness_cap(0.58, weak_with_reversion, 0.62, config) == 0.58


def test_soxx_relative_weakness_cap_respects_regime_band_and_never_boosts():
    config = {
        "min_regime_risk": 0.45,
        "max_regime_risk": 0.90,
        "soxx_bias_max": -0.02,
        "price_ratio_mom_max": 0.0,
        "signed_te_reversion_max": 0.50,
        "target_cap": 0.35,
    }
    row = {
        "SOXX_BIAS_10": -0.04,
        "PRICE_RATIO_MOM": -0.03,
        "SIGNED_TE_REVERSION_10": 0.20,
    }

    assert apply_soxx_relative_weakness_cap(0.58, row, 0.30, config) == 0.58
    assert apply_soxx_relative_weakness_cap(0.20, row, 0.62, config) == 0.20


def test_sequence_risk_action_cap_requires_feature_consensus_and_never_boosts():
    config = {
        "sequence_risk_anchor_slope_max": -0.01,
        "sequence_risk_anchor_gap_max": -0.12,
        "sequence_risk_trend_gap_min": 0.18,
        "sequence_risk_min_feature_hits": 2,
        "target_cap": 0.10,
    }
    two_hit_row = {
        "anchor_slope_180": -0.02,
        "anchor_gap_180": -0.08,
        "trend_gap_63": 0.20,
    }
    one_hit_row = {
        "anchor_slope_180": -0.02,
        "anchor_gap_180": -0.08,
        "trend_gap_63": 0.10,
    }

    assert sequence_risk_action_cap_ready(two_hit_row, config) is True
    assert apply_sequence_risk_action_cap(0.58, two_hit_row, config) == 0.10
    assert apply_sequence_risk_action_cap(0.05, two_hit_row, config) == 0.05
    assert sequence_risk_action_cap_ready(one_hit_row, config) is False
    assert apply_sequence_risk_action_cap(0.58, one_hit_row, config) == 0.58


def test_short_volshock_action_cap_uses_fold_thresholds_and_never_boosts():
    config = {
        "realized_vol_21_min": 0.80,
        "vol_of_vol_21_min": 0.25,
        "downside_semivol_21_min": 0.60,
        "short_volshock_min_feature_hits": 1,
        "target_cap": 0.05,
    }
    one_hit_row = {
        "realized_vol_21": 0.90,
        "vol_of_vol_21": 0.10,
        "downside_semivol_21": 0.20,
    }
    no_hit_row = {
        "realized_vol_21": 0.70,
        "vol_of_vol_21": 0.10,
        "downside_semivol_21": 0.20,
    }

    assert short_volshock_action_cap_ready(one_hit_row, config) is True
    assert apply_short_volshock_action_cap(0.58, one_hit_row, config) == 0.05
    assert apply_short_volshock_action_cap(0.03, one_hit_row, config) == 0.03
    assert short_volshock_action_cap_ready(no_hit_row, config) is False
    assert apply_short_volshock_action_cap(0.58, no_hit_row, config) == 0.58


def test_overlay_de_risk_ready_only_when_enabled_cap_forces_sell_side_reduction():
    config = {"de_risk_guard_enabled": True}

    assert (
        execution_overlays.overlay_de_risk_ready(
            current_pos=0.40,
            target_before_overlay=0.58,
            target_after_overlay=0.05,
            overlay_config=config,
        )
        is True
    )
    assert (
        execution_overlays.overlay_de_risk_ready(
            current_pos=0.03,
            target_before_overlay=0.58,
            target_after_overlay=0.05,
            overlay_config=config,
        )
        is False
    )
    assert (
        execution_overlays.overlay_de_risk_ready(
            current_pos=0.40,
            target_before_overlay=0.58,
            target_after_overlay=0.58,
            overlay_config=config,
        )
        is False
    )
    assert (
        execution_overlays.overlay_de_risk_ready(
            current_pos=0.40,
            target_before_overlay=0.58,
            target_after_overlay=0.05,
            overlay_config={},
        )
        is False
    )


def test_overlay_de_risk_overrides_execution_guards_only_when_ready():
    config = {
        "de_risk_guard_enabled": True,
        "de_risk_rebalance_threshold_override": 0.0,
        "de_risk_trade_lock_override": 0,
    }

    assert execution_overlays.overlay_de_risk_rebalance_threshold(0.12, config, de_risk_ready=True) == 0.0
    assert execution_overlays.overlay_de_risk_trade_lock_days(3, config, de_risk_ready=True) == 0
    assert execution_overlays.overlay_de_risk_rebalance_threshold(0.12, config, de_risk_ready=False) == 0.12
    assert execution_overlays.overlay_de_risk_trade_lock_days(3, config, de_risk_ready=False) == 3
