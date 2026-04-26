from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "runtime"
EXPERIMENTS_DIR = ROOT / "experiments"


@dataclass(frozen=True)
class FeatureConfig:
    packs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RewardConfig:
    return_scale: float
    transaction_cost: float
    turnover_penalty: float
    drawdown_penalty: float
    downside_penalty: float
    risk_budget_penalty: float
    high_exposure_penalty: float
    smoothness_penalty: float
    activity_bonus: float
    inactivity_penalty: float
    opportunity_cost_penalty: float = 0.10
    opportunity_threshold_pos: float = 0.35
    opportunity_threshold_anchor: float = 0.03
    opportunity_threshold_trend: float = 0.01
    pbrs_coef: float = 0.0
    gated_activity_bonus: bool = False
    activity_gate_vol_threshold: float = 0.55
    activity_gate_regime_delta_threshold: float = 0.08
    profit_taking_bonus: float = 0.0
    panic_buy_bonus: float = 0.0
    panic_buy_cum_drop_threshold: float = -0.15
    panic_buy_regime_risk_threshold: float = 0.60
    cooldown_penalty: float = 0.0
    vol_trading_bonus: float = 0.0
    underwater_sell_penalty: float = 0.0
    tranche_profit_threshold: float = 0.03
    use_marginal_reward: bool = False
    marginal_return_scale: float = 1.0
    use_log_reward: bool = False
    log_reward_scale: float = 1.0
    use_alpha_reward: bool = False
    alpha_scale: float = 1.0
    use_kelly_alpha: bool = False
    rule_deviation_penalty: float = 0.0
    micro_rebalance_penalty: float = 0.0
    micro_rebalance_threshold: float = 0.05
    flat_trade_penalty: float = 0.0
    short_hold_turnover_penalty: float = 0.0
    short_hold_threshold: float = 0.16
    stability_bonus: float = 0.0
    trend_alignment_bonus: float = 0.0
    trend_alignment_threshold: float = 0.03
    use_expectancy_reward: bool = False
    expectancy_reward_scale: float = 1.0
    expectancy_position_bonus: float = 0.0
    expectancy_position_penalty: float = 0.0
    use_quality_reward: bool = False
    quality_reward_scale: float = 1.0
    use_nday_reward: bool = False
    nday_window: int = 5
    nday_reward_scale: float = 1.0
    use_sparse_reward: bool = False
    sparse_reward_scale: float = 1.0
    episode_drawdown_penalty: float = 0.0
    episode_drawdown_threshold: float = 0.20
    downside_asymmetry_coef: float = 1.0
    use_cvar_terminal: bool = False
    cvar_terminal_coef: float = 0.0
    cvar_terminal_alpha: float = 0.05
    use_simplified_reward: bool = False
    use_minimal_reward: bool = False
    use_moderate_reward: bool = False
    use_kelly_reward: bool = False
    use_regime_budget: bool = False
    regime_budget_threshold: float = 0.60
    regime_budget_cap: float = 0.30
    regime_budget_penalty: float = 0.50
    rule_imitation_bonus: float = 0.0
    rule_decision_bonus: float = 0.0
    bottomless_pit_penalty: float = 0.0
    # Kelly criterion tracking
    kelly_tracking_bonus: float = 0.0
    kelly_vol_attenuation: bool = True
    kelly_min_signal: float = 0.1
    # Mode-conditional reward shaping (for risk_budget_usage architecture)
    mode_conditional_reward_enabled: bool = False
    flat_wait_pos_penalty: float = 0.0
    crash_rebuild_bonus: float = 0.0
    trend_hold_alignment_multiplier: float = 1.0
    defensive_hold_neutral_bonus: float = 0.0


@dataclass(frozen=True)
class PositionCapConfig:
    start_drawdown: float = 0.10
    full_drawdown: float = 0.30
    min_cap: float = 0.40
    curve_power: float = 1.0
    bull_min_cap: float = 0.65
    bull_vol_threshold: float = 0.40
    bull_anchor_threshold: float = 0.01
    bear_min_cap: float = 0.30
    bear_regime_threshold: float = 0.65


@dataclass(frozen=True)
class PositionBudgetConfig:
    enabled: bool = False
    budget: float = 0.30
    regime_threshold: float = 0.60
    thresholds: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.30, 1.00),
        (0.30, 0.60, 0.70),
        (0.60, 0.90, 0.45),
        (0.90, float("inf"), 0.15),
    )


@dataclass(frozen=True)
class PromotionGateConfig:
    max_test_2022_drawdown: float = 50.0  # Raised from 35.0: 3x ETF 2022 bear realistic floor
    max_test_2022_avg_pos: float = 0.35
    min_test_proxy_sharpe: float = 0.50  # Lowered: backtest_profile Sharpe < evaluate_split
    min_wf_2023_2024_return: float = 0.0
    min_wf_2023_2024_sharpe: float = 0.0
    min_wf_relative_outperformance: float = 15.0  # Actor must beat BnH by at least 15pp


@dataclass(frozen=True)
class SelectionScoreConfig:
    return_weight: float = 1.0
    sharpe_weight: float = 20.0
    max_dd_penalty: float = 0.12
    avg_pos_penalty: float = 0.0
    stress_avg_pos_penalty: float = 0.0
    stress_threshold: float = 0.65
    staged_training: bool = True
    stage_iters: int = 20
    use_pareto: bool = False
    pareto_objectives: Tuple[str, ...] = (
        "test_2022_max_dd",
        "test_proxy_sharpe",
        "test_2023_2024_return",
    )


@dataclass(frozen=True)
class ExecutionRegimeConfig:
    name: str
    conditions: Dict[str, float] = field(default_factory=dict)
    buy_threshold_multiplier: float = 1.0
    sell_threshold_multiplier: float = 1.0
    buy_cooldown_days: int = -1
    trade_lock_days: int = -1


@dataclass(frozen=True)
class ExecutionConfig:
    base_threshold: float = 0.04
    quiet_threshold: float = 0.08
    stress_threshold: float = 0.012
    quiet_regime_threshold: float = 0.35
    quiet_vol_threshold: float = 0.45
    quiet_trend_threshold: float = 0.04
    stress_regime_threshold: float = 0.60
    stress_vol_threshold: float = 0.70
    stress_trend_threshold: float = 0.10
    bull_trend_threshold: float = 0.05
    buy_threshold_multiplier: float = 1.0
    sell_threshold_multiplier: float = 1.0
    bull_buy_threshold_multiplier: float = 1.0
    bull_sell_threshold_multiplier: float = 1.0
    stress_buy_threshold_multiplier: float = 1.0
    stress_sell_threshold_multiplier: float = 1.0
    custom_regimes: Tuple[ExecutionRegimeConfig, ...] = ()
    buy_cooldown_days: int = 1
    bull_buy_cooldown_days: int = -1
    stress_buy_cooldown_days: int = -1
    deep_drop_threshold: float = -0.15
    deep_drop_cooldown_min: int = 3
    deep_drop_cooldown_max: int = 20
    deep_drop_cooldown_scale: float = 30.0
    trade_lock_days: int = 3
    bull_trade_lock_days: int = -1
    stress_trade_lock_days: int = -1
    min_trade_turnover: float = 0.02
    max_drawdown_for_buy: float = 1.0


DEFAULT_ACTION_LEVELS = tuple(round(i / 10, 6) for i in range(11))


@dataclass(frozen=True)
class ModelConfig:
    observation_window: int = 1
    action_levels: Tuple[float, ...] = DEFAULT_ACTION_LEVELS
    position_mapping: str = "default"
    use_regime_router: bool = False
    hard_regime_routing: bool = False
    router_experts: int = 3
    router_top_k: int = 1
    router_noise_std: float = 0.1
    hard_floor: float | None = None
    use_tide: bool = False
    tide_hidden_size: int = 64
    tide_dropout: float = 0.1
    use_temporal_conv: bool = False
    temporal_conv_channels: int = 32
    temporal_conv_kernel: int = 3
    use_gru: bool = False
    gru_hidden_size: int = 64
    dropout: float = 0.0
    use_attention: bool = False
    attention_dim: int = 64
    attention_heads: int = 4
    use_hybrid_overlay: bool = False
    delta_levels: Tuple[float, ...] = (-0.15, 0.0, 0.15)
    use_mode_conditioned_input: bool = False


@dataclass(frozen=True)
class ModeRouterConfig:
    enabled: bool = False
    use_learned: bool = False
    learned_model_path: str = ""
    trend_hold_min_dwell_days: int = 12
    defensive_hold_min_dwell_days: int = 6
    crash_rebuild_min_dwell_days: int = 5
    flat_wait_min_dwell_days: int = 4
    trend_hold_floor: float = 0.65
    trend_hold_cap: float = 0.95
    defensive_hold_floor: float = 0.25
    defensive_hold_cap: float = 0.60
    crash_rebuild_floor: float = 0.15
    crash_rebuild_cap: float = 0.55
    flat_wait_floor: float = 0.00
    flat_wait_cap: float = 0.25
    trend_hold_buy_threshold_multiplier: float = 1.35
    trend_hold_sell_threshold_multiplier: float = 1.10
    defensive_hold_buy_threshold_multiplier: float = 1.10
    defensive_hold_sell_threshold_multiplier: float = 0.95
    crash_rebuild_buy_threshold_multiplier: float = 0.75
    crash_rebuild_sell_threshold_multiplier: float = 0.90
    flat_wait_buy_threshold_multiplier: float = 1.60
    flat_wait_sell_threshold_multiplier: float = 0.80
    # Dynamic band scaling based on volatility regime
    use_dynamic_bands: bool = False
    dynamic_band_vol_baseline: float = 0.50
    dynamic_band_vol_sensitivity: float = 0.30
    dynamic_band_min_scale: float = 0.60
    dynamic_band_max_scale: float = 1.20


@dataclass(frozen=True)
class WarmStartConfig:
    """Rule-based behavioral-cloning pretraining before RL.

    When `enabled=True`, the actor is trained with cross-entropy on
    (obs -> rule-based action) pairs for `steps` gradient updates. This
    shifts the PPO initial policy from random to "mimic rule", which
    reduces seed variance and speeds convergence.

    `sample_size` controls how many transitions are drawn from the
    training initial_states to build the supervised dataset.
    """
    enabled: bool = False
    steps: int = 500
    lr: float = 5e-4
    batch_size: int = 128
    sample_size: int = 2000


@dataclass(frozen=True)
class SamplingConfig:
    regime_stratified: bool = False
    phase_balanced: bool = False
    mild_risk_threshold: float = 0.42
    severe_risk_threshold: float = 0.58
    mild_drawdown_threshold: float = 0.10
    severe_drawdown_threshold: float = 0.18
    mild_repeat: int = 2
    severe_repeat: int = 3
    persistent_stress_days: float = 0.30
    trend_repeat: int = 2
    rebound_repeat: int = 2
    episode_stride: int = 0
    bull_anchor_threshold: float = 0.04
    bull_trend_threshold: float = 0.02
    bull_vol_threshold: float = 0.45
    bull_risk_threshold: float = 0.45
    rebound_anchor_ceiling: float = -0.02
    rebound_trend_floor: float = 0.01
    rebound_risk_ceiling: float = 0.60
    rebound_risk_release: float = 0.06


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    max_iters: int
    episode_per_iter: int
    episode_len: int
    hidden_sizes: Tuple[int, ...]
    lr: float
    update_batch_size: int
    update_repeat: int
    reward: RewardConfig
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    position_cap: PositionCapConfig = field(default_factory=PositionCapConfig)
    position_budget: PositionBudgetConfig = field(default_factory=PositionBudgetConfig)
    promotion_gate: PromotionGateConfig = field(default_factory=PromotionGateConfig)
    selection_score: SelectionScoreConfig = field(default_factory=SelectionScoreConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mode_router: ModeRouterConfig = field(default_factory=ModeRouterConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    checkpoint_path: str = ""
    decision_frequency: int = 1
    algorithm: str = "ppo"
    separate_actor_critic: bool = False
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    weight_decay: float = 0.0


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def experiment_profile_path(name: str) -> Path | None:
    matches = sorted(EXPERIMENTS_DIR.glob(f"*/{name}/profile.yaml"))
    if matches:
        return matches[0]
    return None


def legacy_profile_path(name: str) -> Path | None:
    for path in PROFILE_DIR.rglob(f"cards/{name}/profile.yaml"):
        return path
    return None


def resolve_profile_path(name: str) -> Path:
    experiment_path = experiment_profile_path(name)
    if experiment_path is not None:
        return experiment_path
    legacy_path = legacy_profile_path(name)
    if legacy_path is not None and legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(f"unknown profile '{name}'")


def experiment_dir_for_profile(name: str) -> Path | None:
    path = experiment_profile_path(name)
    return path.parent if path is not None else None


def list_profile_names() -> list[str]:
    legacy = {path.parent.name for path in PROFILE_DIR.rglob("cards/*/profile.yaml")}
    experiments = {path.parent.name for path in EXPERIMENTS_DIR.glob("*/*/profile.yaml")}
    return sorted(legacy | experiments)


def load_profile(name: str) -> StrategyConfig:
    payload = _load_yaml(resolve_profile_path(name))
    reward = RewardConfig(**payload["reward"])
    feature_payload = dict(payload.get("feature", {}))
    if "packs" in feature_payload:
        feature_payload["packs"] = tuple(str(v) for v in feature_payload["packs"])
    feature = FeatureConfig(**feature_payload)
    position_cap = PositionCapConfig(**payload.get("position_cap", {}))
    budget_payload = dict(payload.get("position_budget", {}))
    if "thresholds" in budget_payload:
        budget_payload["thresholds"] = tuple(
            tuple(float(v) for v in row) for row in budget_payload["thresholds"]
        )
    position_budget = PositionBudgetConfig(**budget_payload)
    promotion_gate = PromotionGateConfig(**payload.get("promotion_gate", {}))
    selection_payload = dict(payload.get("selection_score", {}))
    if "pareto_objectives" in selection_payload:
        selection_payload["pareto_objectives"] = tuple(str(v) for v in selection_payload["pareto_objectives"])
    selection_score = SelectionScoreConfig(**selection_payload)
    execution_payload = dict(payload.get("execution", {}))
    custom_regimes = tuple(
        ExecutionRegimeConfig(
            name=str(item["name"]),
            conditions={str(k): float(v) for k, v in dict(item.get("conditions", {})).items()},
            buy_threshold_multiplier=float(item.get("buy_threshold_multiplier", 1.0)),
            sell_threshold_multiplier=float(item.get("sell_threshold_multiplier", 1.0)),
            buy_cooldown_days=int(item.get("buy_cooldown_days", -1)),
            trade_lock_days=int(item.get("trade_lock_days", -1)),
        )
        for item in execution_payload.pop("custom_regimes", [])
    )
    execution = ExecutionConfig(custom_regimes=custom_regimes, **execution_payload)
    model_payload = dict(payload.get("model", {}))
    if "action_levels" in model_payload:
        model_payload["action_levels"] = tuple(float(v) for v in model_payload["action_levels"])
    if "delta_levels" in model_payload:
        model_payload["delta_levels"] = tuple(float(v) for v in model_payload["delta_levels"])
    model = ModelConfig(**model_payload)
    mode_router = ModeRouterConfig(**payload.get("mode_router", {}))
    sampling = SamplingConfig(**payload.get("sampling", {}))
    warm_start = WarmStartConfig(**payload.get("warm_start", {}))
    checkpoint_path = str(payload.get("checkpoint_path", ""))
    return StrategyConfig(
        name=payload["name"],
        max_iters=int(payload["max_iters"]),
        episode_per_iter=int(payload["episode_per_iter"]),
        episode_len=int(payload.get("episode_len", 300)),
        hidden_sizes=tuple(int(v) for v in payload["hidden_sizes"]),
        lr=float(payload.get("lr", 3e-4)),
        update_batch_size=int(payload.get("update_batch_size", 64)),
        update_repeat=int(payload.get("update_repeat", 10)),
        reward=reward,
        feature=feature,
        position_cap=position_cap,
        position_budget=position_budget,
        promotion_gate=promotion_gate,
        selection_score=selection_score,
        execution=execution,
        model=model,
        mode_router=mode_router,
        sampling=sampling,
        warm_start=warm_start,
        checkpoint_path=checkpoint_path,
        decision_frequency=int(payload.get("decision_frequency", 1)),
        algorithm=str(payload.get("algorithm", "ppo")),
        separate_actor_critic=bool(payload.get("separate_actor_critic", False)),
        discount_factor=float(payload.get("discount_factor", 0.99)),
        gae_lambda=float(payload.get("gae_lambda", 0.95)),
        ent_coef=float(payload.get("ent_coef", 0.01)),
        weight_decay=float(payload.get("weight_decay", 0.0)),
    )
