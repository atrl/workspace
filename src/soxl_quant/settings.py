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
EXTRA_OBS_DIM = 13
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
    )
