from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/soxl_soxx_vix_full.csv"
QLIB_DATA_PATH = ROOT / "qlib_soxl_binary"
RESULT_PNG = ROOT / "soxl_qlib_strategy_v2.png"
RESULT_TXT = ROOT / "soxl_qlib_strategy_v2_results.txt"
PROFILE_DIR = ROOT / "configs/soxl"
MODEL_DIR = ROOT / "artifacts/models"
RUNTIME_DIR = ROOT / "runtime/soxl"
ACCOUNT = 1_000_000
LOOKBACK = 60
N_ACTIONS = 11
EXTRA_OBS_DIM = 15
N_ASSETS = 5
TRAIN_SEG = ("2010-03-11", "2018-12-31")
VALID_SEG = ("2019-01-01", "2019-12-31")
TEST_SEG = ("2020-01-01", "2026-03-31")

ETF_UNIVERSE = {
    "SOXL": {"underlying": "SOXX", "leverage": 3.0},
    "TQQQ": {"underlying": "QQQ", "leverage": 3.0},
    "UPRO": {"underlying": "SPY", "leverage": 3.0},
    "SPXL": {"underlying": "SPY", "leverage": 3.0},
    "TECL": {"underlying": "XLK", "leverage": 3.0},
}

DEFAULT_ACTION_LEVELS = tuple(round(i / (N_ACTIONS - 1), 6) for i in range(N_ACTIONS))


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
    # 踏空惩罚：在强趋势中仓位过低时扣分（新增，默认值保持向后兼容）
    opportunity_cost_penalty: float = 0.10
    opportunity_threshold_pos: float = 0.35
    opportunity_threshold_anchor: float = 0.03
    opportunity_threshold_trend: float = 0.01
    # PBRS 与动态 activity bonus（新增，默认值保持向后兼容）
    pbrs_coef: float = 0.0
    gated_activity_bonus: bool = False
    activity_gate_vol_threshold: float = 0.55
    activity_gate_regime_delta_threshold: float = 0.08
    # soxl_backtest-inspired optimizations（新增，默认值保持向后兼容）
    profit_taking_bonus: float = 0.0
    panic_buy_bonus: float = 0.0
    panic_buy_cum_drop_threshold: float = -0.15
    panic_buy_regime_risk_threshold: float = 0.60
    cooldown_penalty: float = 0.0
    vol_trading_bonus: float = 0.0
    # tranche-aware profit-taking / underwater-sell guardrails
    underwater_sell_penalty: float = 0.0
    tranche_profit_threshold: float = 0.03
    # marginal expectancy reward: reward = marginal_pnl instead of aggregate pnl
    use_marginal_reward: bool = False
    marginal_return_scale: float = 1.0
    # Kelly-inspired log reward: reward = ln(1 + pnl_ret) to maximize long-term growth rate
    use_log_reward: bool = False
    log_reward_scale: float = 1.0
    # Alpha reward: reward = actual_return - rule_based_return (RL only rewarded for beating rule-based)
    use_alpha_reward: bool = False
    alpha_scale: float = 1.0
    use_kelly_alpha: bool = False
    # Stability & trend-alignment bonuses
    stability_bonus: float = 0.0
    trend_alignment_bonus: float = 0.0
    trend_alignment_threshold: float = 0.03
    # Three alternative reward modes
    use_quality_reward: bool = False
    quality_reward_scale: float = 1.0
    use_nday_reward: bool = False
    nday_window: int = 5
    nday_reward_scale: float = 1.0
    use_sparse_reward: bool = False
    sparse_reward_scale: float = 1.0


@dataclass(frozen=True)
class PositionCapConfig:
    start_drawdown: float = 0.10
    full_drawdown: float = 0.30
    min_cap: float = 0.40
    curve_power: float = 1.0
    # 动态 cap：牛市放宽 / 熊市收紧（新增，默认值保持向后兼容）
    bull_min_cap: float = 0.65
    bull_vol_threshold: float = 0.40
    bull_anchor_threshold: float = 0.01
    bear_min_cap: float = 0.30
    bear_regime_threshold: float = 0.65


@dataclass(frozen=True)
class PromotionGateConfig:
    max_test_2022_drawdown: float = 35.0
    max_test_2022_avg_pos: float = 0.35
    min_test_proxy_sharpe: float = 0.75
    min_wf_2023_2024_return: float = 0.0
    min_wf_2023_2024_sharpe: float = 0.0


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


@dataclass(frozen=True)
class ModelConfig:
    observation_window: int = 1
    action_levels: Tuple[float, ...] = DEFAULT_ACTION_LEVELS
    use_regime_router: bool = False
    hard_regime_routing: bool = False
    router_experts: int = 3
    router_top_k: int = 1
    router_noise_std: float = 0.1
    hard_floor: Optional[float] = None
    use_tide: bool = False
    tide_hidden_size: int = 64
    tide_dropout: float = 0.1
    use_hybrid_overlay: bool = False
    delta_levels: Tuple[float, ...] = (-0.15, 0.0, 0.15)


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
    position_cap: PositionCapConfig = field(default_factory=PositionCapConfig)
    promotion_gate: PromotionGateConfig = field(default_factory=PromotionGateConfig)
    selection_score: SelectionScoreConfig = field(default_factory=SelectionScoreConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    decision_frequency: int = 1
    # PPO / SAC algorithm selection and hyperparameters
    algorithm: str = "ppo"
    separate_actor_critic: bool = False
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01


def artifact_stem(profile_name: str, artifact_tag: str = "") -> str:
    """Build a stable artifact prefix for canonical and namespaced runs."""

    stem = f"soxl_{profile_name}"
    cleaned_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", artifact_tag.strip())
    if cleaned_tag:
        stem = f"{stem}_{cleaned_tag}"
    return stem


def model_artifact_path(profile_name: str, artifact_tag: str = "") -> Path:
    return MODEL_DIR / f"{artifact_stem(profile_name, artifact_tag)}_actor.pt"


def summary_artifact_path(profile_name: str, artifact_tag: str = "") -> Path:
    return MODEL_DIR / f"{artifact_stem(profile_name, artifact_tag)}_summary.json"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_profile(name: str) -> StrategyConfig:
    payload = _load_yaml(PROFILE_DIR / f"{name}.yaml")
    reward = RewardConfig(**payload["reward"])
    position_cap = PositionCapConfig(**payload.get("position_cap", {}))
    promotion_gate = PromotionGateConfig(**payload.get("promotion_gate", {}))
    selection_score = SelectionScoreConfig(**payload.get("selection_score", {}))
    execution = ExecutionConfig(**payload.get("execution", {}))
    model_payload = dict(payload.get("model", {}))
    if "action_levels" in model_payload:
        model_payload["action_levels"] = tuple(float(v) for v in model_payload["action_levels"])
    if "delta_levels" in model_payload:
        model_payload["delta_levels"] = tuple(float(v) for v in model_payload["delta_levels"])
    model = ModelConfig(**model_payload)
    sampling = SamplingConfig(**payload.get("sampling", {}))
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
        position_cap=position_cap,
        promotion_gate=promotion_gate,
        selection_score=selection_score,
        execution=execution,
        model=model,
        sampling=sampling,
        decision_frequency=int(payload.get("decision_frequency", 1)),
        algorithm=str(payload.get("algorithm", "ppo")),
        separate_actor_critic=bool(payload.get("separate_actor_critic", False)),
        discount_factor=float(payload.get("discount_factor", 0.99)),
        gae_lambda=float(payload.get("gae_lambda", 0.95)),
        ent_coef=float(payload.get("ent_coef", 0.01)),
    )
