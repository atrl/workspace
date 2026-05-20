from __future__ import annotations

from src.soxl_quant.policy.execution_overlays import (
    POST_RELEASE_CAP_VARIANT,
    overlay_config,
    overlay_variants,
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
