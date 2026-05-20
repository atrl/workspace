from __future__ import annotations

from src.soxl_quant.policy.execution_overlays import (
    POST_RELEASE_CAP_VARIANT,
    SOXX_RELATIVE_WEAKNESS_CAP_VARIANT,
    apply_soxx_relative_weakness_cap,
    overlay_config,
    overlay_variants,
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
