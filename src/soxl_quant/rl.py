from __future__ import annotations

"""Compatibility facade for legacy RL imports.

Real implementations now live under `soxl_quant.policy.*` plus the split
observation / execution / positioning helpers. Keep this module thin so new
work does not accumulate here again.
"""

from .policy.environment import (
    EpisodeConfig,
    EpisodeSummary,
    PositionAction,
    SoxlSimulator,
    SoxlState,
    TrancheLedger,
    build_initial_states,
    build_price_path,
)
from .policy.evaluation import evaluate_split
from .policy.models.moe import QNetwork, RegimeMoENet, SoxlPPOPolicy
from .policy.models.tide import TiDENet
from .policy.policy_builder import actor_action, build_policy
from .policy.reward import ContinuousReward
from .policy.execution import (
    DEFAULT_EXECUTION,
    execution_buy_cooldown_days,
    execution_regime,
    execution_trade_lock_days,
    rebalance_threshold,
)
from .policy.observation import (
    EXTRA_INDEX,
    adapt_observation,
    adapt_observation_for_actor,
    infer_actor_obs_dim,
    make_observation,
    observation_dim_for_profile,
)
from .policy.positioning import (
    BASE_ACTION_LEVELS,
    action_levels_for_profile,
    action_to_position,
    drawdown_position_cap,
    rule_based_position,
)

__all__ = [
    "BASE_ACTION_LEVELS",
    "ContinuousReward",
    "DEFAULT_EXECUTION",
    "EXTRA_INDEX",
    "EpisodeConfig",
    "EpisodeSummary",
    "PositionAction",
    "QNetwork",
    "RegimeMoENet",
    "SoxlPPOPolicy",
    "SoxlSimulator",
    "SoxlState",
    "TiDENet",
    "TrancheLedger",
    "action_levels_for_profile",
    "action_to_position",
    "actor_action",
    "adapt_observation",
    "adapt_observation_for_actor",
    "build_initial_states",
    "build_policy",
    "build_price_path",
    "drawdown_position_cap",
    "evaluate_split",
    "execution_buy_cooldown_days",
    "execution_regime",
    "execution_trade_lock_days",
    "infer_actor_obs_dim",
    "make_observation",
    "observation_dim_for_profile",
    "rebalance_threshold",
    "rule_based_position",
]
