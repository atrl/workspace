from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import src.soxl_quant.runtime as runtime
from src.soxl_quant.config import ExecutionConfig, ExecutionRegimeConfig
from src.soxl_quant.config import load_profile
from src.soxl_quant.policy.execution_overlays import POST_RELEASE_CAP_VARIANT
from src.soxl_quant.policy.execution import (
    execution_buy_cooldown_days,
    execution_regime,
    execution_trade_lock_days,
    rebalance_threshold,
)
from src.soxl_quant.policy.environment import TrancheLedger
from src.soxl_quant.runtime import (
    RuntimeExecutionOverlayState,
    SignalIntent,
    PaperBroker,
    apply_runtime_execution_overlay,
    load_runtime_execution_overlay_state,
    next_runtime_execution_overlay_state,
    reconcile_positions,
    risk_gate,
    runtime_feature_packs,
    save_runtime_execution_overlay_state,
)


def _write_actor_with_summary(root: Path, profile: str, *, promoted: bool, score: float) -> Path:
    model_dir = root / profile
    model_dir.mkdir(parents=True)
    actor_path = model_dir / f"soxl_{profile}_actor.pt"
    actor_path.write_bytes(b"")
    summary = {
        "best_score": score,
        "promotion": {
            "eligible": promoted,
            "walk_forward_evaluated": True,
        },
    }
    (model_dir / f"soxl_{profile}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return actor_path


def test_best_model_for_family_refuses_unpromoted_fallback():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_actor_with_summary(root, "balanced", promoted=False, score=99.0)
        previous_model_dir = runtime.MODEL_DIR
        runtime.MODEL_DIR = root
        try:
            try:
                runtime.best_model_for_family("balanced")
            except RuntimeError as exc:
                assert "no promoted model" in str(exc)
            else:
                raise AssertionError("expected unpromoted fallback to be rejected")
        finally:
            runtime.MODEL_DIR = previous_model_dir


def test_best_model_for_family_selects_highest_promoted_score():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        weaker = _write_actor_with_summary(root, "balanced_rule_residual_v18", promoted=True, score=80.0)
        stronger = _write_actor_with_summary(root, "balanced_rule_residual_v19", promoted=True, score=95.0)
        _write_actor_with_summary(root, "balanced_rule_residual_v24", promoted=False, score=110.0)
        previous_model_dir = runtime.MODEL_DIR
        runtime.MODEL_DIR = root
        try:
            assert runtime.best_model_for_family("balanced") == stronger
            assert runtime.best_model_for_family("balanced") != weaker
        finally:
            runtime.MODEL_DIR = previous_model_dir


def test_best_model_for_family_skips_overlay_profile_without_auto_select_approval():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        approved = _write_actor_with_summary(root, "balanced_rule_residual_v19", promoted=True, score=95.0)
        _write_actor_with_summary(root, "balanced_rule_residual_v22", promoted=True, score=120.0)
        previous_model_dir = runtime.MODEL_DIR
        runtime.MODEL_DIR = root
        try:
            assert runtime.best_model_for_family("balanced") == approved
        finally:
            runtime.MODEL_DIR = previous_model_dir


def test_profile_bound_runtime_overlay_loads_release_signals_but_blocks_auto_select():
    profile = load_profile("balanced_rule_residual_v22")

    assert profile.execution_overlay.enabled is True
    assert profile.execution_overlay.name == POST_RELEASE_CAP_VARIANT
    assert profile.execution_overlay.auto_select_allowed is False
    assert runtime_feature_packs(profile) == ("vix_term", "release_signals_v1")


def test_profile_bound_runtime_overlay_relative_state_path_loads_default_state():
    profile = load_profile("balanced_rule_residual_v22")

    state = load_runtime_execution_overlay_state(profile, account_value=95_000.0)

    assert state == RuntimeExecutionOverlayState(state_days=0, peak_account_value=95_000.0)


def test_runtime_overlay_caps_post_release_target_after_state_drawdown():
    profile = load_profile("balanced_rule_residual_v22")
    row = pd.Series(
        {
            "RB_COOLDOWN_PROXY": 0.30,
            "TRANS_VIX_DIVERGENCE": 0.08,
            "CICC_RETURN_VOLUME_SYNC": 0.20,
        }
    )

    target, decision = apply_runtime_execution_overlay(
        profile,
        target_position=0.50,
        row=row,
        regime_risk=0.50,
        current_position=0.40,
        current_drawdown=0.01,
        account_value=95_000.0,
        state=RuntimeExecutionOverlayState(state_days=12, peak_account_value=100_000.0),
    )

    assert target == 0.55
    assert decision.enabled is True
    assert decision.name == POST_RELEASE_CAP_VARIANT
    assert decision.gate_ready is True
    assert decision.execution_ready is True
    assert decision.post_release_de_risk_ready is True
    assert decision.target_before_overlay == 0.50
    assert decision.target_after_overlay == 0.58
    assert decision.target_after_post_release_cap == 0.55


def test_runtime_overlay_state_activates_after_release_order_and_roundtrips():
    with TemporaryDirectory() as tmp:
        profile = load_profile("balanced_rule_residual_v22")
        profile = replace(
            profile,
            execution_overlay=replace(
                profile.execution_overlay,
                state_path=str(Path(tmp) / "overlay_state.json"),
            ),
        )
        signal = SignalIntent(
            date="2024-03-28",
            symbol="SOXL",
            profile=profile.name,
            model_path="runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_actor.pt",
            seed=42,
            action=6,
            target_position=0.58,
            current_position=0.40,
            price=40.0,
            regime_risk=0.50,
            note="test",
            execution_overlay=POST_RELEASE_CAP_VARIANT,
            overlay_execution_ready=True,
        )

        state = next_runtime_execution_overlay_state(
            profile,
            previous_state=RuntimeExecutionOverlayState(),
            signal=signal,
            account_value=100_000.0,
            order_submitted=True,
        )
        save_runtime_execution_overlay_state(profile, state, date=signal.date)
        loaded = load_runtime_execution_overlay_state(profile, account_value=90_000.0)

        assert state.state_days == 90
        assert state.peak_account_value == 100_000.0
        assert loaded == state


def test_position_ledger_tracks_entry_and_age():
    ledger = TrancheLedger()
    ledger.update(0.0, 0.4, 100.0)
    ledger.update(0.4, 0.4, 105.0)
    age, avg_pnl, add_pnl, high_gap, _last_tranche_pnl, _n_tranches = ledger.snapshot(102.0)
    assert age > 0.0
    assert avg_pnl > 0.0
    assert add_pnl > 0.0
    assert high_gap < 0.0


def test_paper_broker_creates_lot_sized_orders():
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    commands = broker.rebalance("2024-01-02", "SOXL", target_position=0.5, price=50.0, reason="test")
    assert len(commands) == 1
    assert commands[0].side == "BUY"
    assert commands[0].quantity % 100 == 0
    assert broker.shares == commands[0].quantity


def test_risk_gate_blocks_large_one_shot_position_jump():
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    signal = SignalIntent(
        date="2024-01-02",
        symbol="SOXL",
        profile="balanced",
        model_path="runtime/model/card/example/example.pt",
        seed=20260411,
        action=10,
        target_position=0.8,
        current_position=0.0,
        price=50.0,
        regime_risk=0.3,
        note="regime=0.30",
    )
    decision = risk_gate(broker, signal, max_position_change=0.35)
    assert decision.approved is False
    assert any("exceeds limit" in reason for reason in decision.blocking_reasons)


def test_risk_gate_clips_large_position_step_when_enabled():
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    signal = SignalIntent(
        date="2024-01-02",
        symbol="SOXL",
        profile="balanced",
        model_path="runtime/model/card/example/example.pt",
        seed=20260411,
        action=10,
        target_position=0.8,
        current_position=0.0,
        price=50.0,
        regime_risk=0.3,
        note="regime=0.30",
    )

    decision = risk_gate(broker, signal, max_position_change=0.35, clip_position_change=True)

    assert decision.approved is True
    assert decision.capped_target_position == 0.35
    assert not decision.blocking_reasons
    assert any("clipped" in check.detail for check in decision.checks if check.name == "position_step_limit")


def test_reconcile_positions_flags_meaningful_gap():
    record = reconcile_positions("2024-01-02", expected_position=0.40, actual_position=0.22, tolerance=0.03)
    assert record.action_required == "manual_review"
    assert record.position_gap < 0.0


def test_risk_gate_blocks_non_promoted_model():
    broker = PaperBroker(initial_cash=100_000.0, fee_rate=0.0, lot_size=100)
    signal = SignalIntent(
        date="2024-01-02",
        symbol="SOXL",
        profile="balanced",
        model_path="runtime/model/card/example/example.pt",
        seed=20260411,
        action=5,
        target_position=0.2,
        current_position=0.1,
        price=50.0,
        regime_risk=0.3,
        note="regime=0.30",
    )
    decision = risk_gate(broker, signal, model_ready=(False, "model failed promotion gate"))
    assert decision.approved is False
    assert any("promotion gate" in reason for reason in decision.blocking_reasons)


def test_execution_regime_detects_release_only_when_enabled():
    disabled = ExecutionConfig()
    enabled = ExecutionConfig(
        custom_regimes=(
            ExecutionRegimeConfig(
                name="release",
                conditions={
                    "cum_drop_30_max": -0.10,
                    "regime_risk_max": 0.56,
                    "realized_vol_max": 0.60,
                    "trend_strength_min": 0.035,
                    "relative_strength_min": 0.18,
                    "vix_zscore_max": 0.35,
                    "downside_ratio_10_max": 0.48,
                },
            ),
        ),
    )
    kwargs = dict(
        regime_risk=0.50,
        realized_vol=0.52,
        trend_strength=0.05,
        cum_drop_30=-0.14,
        relative_strength=0.22,
        vix_zscore=0.10,
        downside_ratio_10=0.42,
    )
    assert execution_regime(config=disabled, **kwargs) == "base"
    assert execution_regime(config=enabled, **kwargs) == "release"


def test_release_overrides_make_buying_easier_and_faster():
    cfg = ExecutionConfig(
        custom_regimes=(
            ExecutionRegimeConfig(
                name="release",
                conditions={
                    "cum_drop_30_max": -0.10,
                    "regime_risk_max": 0.56,
                    "realized_vol_max": 0.60,
                    "trend_strength_min": 0.035,
                    "relative_strength_min": 0.18,
                    "vix_zscore_max": 0.35,
                    "downside_ratio_10_max": 0.48,
                },
                buy_threshold_multiplier=0.72,
                sell_threshold_multiplier=1.10,
                buy_cooldown_days=0,
                trade_lock_days=1,
            ),
        ),
    )
    release_kwargs = dict(
        regime_risk=0.50,
        realized_vol=0.52,
        trend_strength=0.05,
        cum_drop_30=-0.14,
        relative_strength=0.22,
        vix_zscore=0.10,
        downside_ratio_10=0.42,
        config=cfg,
    )
    buy_threshold = rebalance_threshold(direction="buy", **release_kwargs)
    sell_threshold = rebalance_threshold(direction="sell", **release_kwargs)
    assert buy_threshold < cfg.base_threshold
    assert sell_threshold > cfg.base_threshold
    assert execution_buy_cooldown_days(-0.05, cfg, "release") == 0
    assert execution_trade_lock_days(cfg, "release") == 1
