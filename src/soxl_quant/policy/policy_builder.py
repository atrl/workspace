from __future__ import annotations

import random
from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from tianshou.policy import DiscreteSACPolicy, PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from ..config import StrategyConfig
from ..data import FEATURE_COLUMNS, STATIC_INDEX, feature_columns_for_packs, static_index_for_packs
from ..settings import LOOKBACK
from .observation import adapt_observation_for_actor, make_observation
from .modes import ModeState, mode_band, route_mode
from .mode_conditioning import append_mode_one_hot
from .positioning import action_levels_for_profile, rule_based_position, target_position_for_profile
from .environment import (
    EpisodeConfig,
    PositionAction,
    SoxlSimulator,
    SoxlState,
    TrancheLedger,
    build_initial_states,
    build_price_path,
)
from .models.attention import WindowSelfAttentionNet
from .models.gru import GRUNet
from .models.moe import QNetwork, RegimeMoENet, SoxlPPOPolicy
from .models.tide import TiDENet


def _build_net(obs_dim: int, profile: StrategyConfig, model_cfg):
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    static_index = static_index_for_packs(profile.feature.packs, profile.name)
    effective_use_tide = bool(model_cfg.use_tide)
    effective_use_moe = bool(model_cfg.use_regime_router)
    effective_hard_routing = bool(model_cfg.hard_regime_routing)
    if effective_use_tide:
        return TiDENet(
            obs_dim,
            hidden_size=int(model_cfg.tide_hidden_size),
            feature_dim=len(feature_columns),
            observation_window=max(int(model_cfg.observation_window), 1),
            dropout=float(model_cfg.tide_dropout),
            device="cpu",
        )
    if getattr(model_cfg, "use_attention", False):
        return WindowSelfAttentionNet(
            obs_dim,
            feature_dim=len(feature_columns),
            observation_window=max(int(model_cfg.observation_window), 1),
            attention_dim=int(getattr(model_cfg, "attention_dim", 64)),
            attention_heads=int(getattr(model_cfg, "attention_heads", 4)),
            mlp_hidden_sizes=list(profile.hidden_sizes),
            device="cpu",
        )
    if model_cfg.use_gru:
        return GRUNet(
            obs_dim,
            feature_dim=len(feature_columns),
            observation_window=max(int(model_cfg.observation_window), 1),
            gru_hidden_size=int(model_cfg.gru_hidden_size),
            mlp_hidden_sizes=list(profile.hidden_sizes),
            device="cpu",
            dropout=float(getattr(model_cfg, "dropout", 0.0)),
        )
    if effective_use_moe:
        return RegimeMoENet(
            obs_dim,
            hidden_size=64,
            n_experts=int(model_cfg.router_experts),
            device="cpu",
            top_k=int(model_cfg.router_top_k),
            noise_std=float(model_cfg.router_noise_std),
            regime_risk_idx=static_index["regime_risk"],
            hard_regime_routing=effective_hard_routing,
        )
    return Net((obs_dim,), hidden_sizes=list(profile.hidden_sizes), device="cpu")


def build_policy(
    obs_dim: int,
    profile: StrategyConfig,
    use_moe: bool = False,
    hard_regime_routing: bool = False,
    **kwargs,
):
    model_cfg = profile.model
    effective_use_moe = bool(use_moe or model_cfg.use_regime_router)
    if hard_regime_routing:
        model_cfg = replace(model_cfg, hard_regime_routing=True)
    action_levels = action_levels_for_profile(profile)
    n_actions = len(action_levels) if not model_cfg.use_hybrid_overlay else len(model_cfg.delta_levels)

    if profile.algorithm == "sac":
        actor_net = _build_net(obs_dim, profile, model_cfg)
        critic1_net = _build_net(obs_dim, profile, model_cfg)
        critic2_net = _build_net(obs_dim, profile, model_cfg)
        actor = Actor(actor_net, n_actions, device="cpu")
        critic1_raw = Actor(critic1_net, n_actions, softmax_output=False, device="cpu")
        critic2_raw = Actor(critic2_net, n_actions, softmax_output=False, device="cpu")
        critic1 = QNetwork(critic1_raw)
        critic2 = QNetwork(critic2_raw)
        wd = float(getattr(profile, "weight_decay", 0.0))
        actor_optim = torch.optim.Adam(actor.parameters(), lr=profile.lr, weight_decay=wd)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=profile.lr, weight_decay=wd)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=profile.lr, weight_decay=wd)
        policy = DiscreteSACPolicy(
            actor=actor,
            actor_optim=actor_optim,
            critic1=critic1,
            critic1_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=0.005,
            gamma=profile.discount_factor,
            alpha=profile.ent_coef,
            reward_normalization=False,
            estimation_step=1,
            action_space=spaces.Discrete(n_actions),
        )
        return policy, actor

    if profile.separate_actor_critic:
        actor_net = _build_net(obs_dim, profile, model_cfg)
        critic_net = _build_net(obs_dim, profile, model_cfg)
    else:
        shared_net = _build_net(obs_dim, profile, model_cfg)
        actor_net = shared_net
        critic_net = shared_net
    actor = Actor(actor_net, n_actions, device="cpu")
    critic = Critic(critic_net, device="cpu")
    wd = float(getattr(profile, "weight_decay", 0.0))
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=profile.lr, weight_decay=wd)
    policy_cls = SoxlPPOPolicy if effective_use_moe else PPOPolicy
    policy_kwargs = dict(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=profile.discount_factor,
        gae_lambda=profile.gae_lambda,
        max_grad_norm=0.5,
        vf_coef=0.25,
        ent_coef=profile.ent_coef,
        reward_normalization=True,
        recompute_advantage=True,
        action_space=spaces.Discrete(n_actions),
    )
    if effective_use_moe:
        policy_kwargs["moe_lb_coef"] = kwargs.get("moe_lb_coef", 0.01)
    policy = policy_cls(**policy_kwargs)
    return policy, actor


def build_warm_start_dataset(
    initial_states: Iterable[EpisodeConfig],
    profile: StrategyConfig,
    sample_size: int,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (obs, action_idx) supervised dataset that mimics the rule policy.

    For each sampled transition we:
      1. pick a random episode (EpisodeConfig) and a random day inside it
         (after the lookback window).
      2. materialize the same observation the env would see with pos=0, value=cash0.
      3. compute the rule-based position for that obs.
      4. convert rule position into the nearest discrete action index:
           - hybrid overlay: the delta-level closest to 0 (rule is the base)
           - pure discrete:  the action-level closest to rule_pos.

    Resulting dataset teaches the actor to "stay neutral around the rule" as
    its initial policy, which is exactly the prior PPO would otherwise need
    many rollouts to discover.
    """
    rng = random.Random(rng_seed)
    use_hybrid = bool(profile.model.use_hybrid_overlay)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    if use_hybrid:
        levels = np.asarray(profile.model.delta_levels, dtype=np.float32)
        target_value = 0.0
        default_idx = int(np.argmin(np.abs(levels - target_value)))
    else:
        levels = action_levels_for_profile(profile)
        default_idx = -1

    episodes = list(initial_states)
    if not episodes:
        raise ValueError("build_warm_start_dataset requires at least one episode")

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []
    attempts = 0
    max_attempts = sample_size * 8

    cache: dict[str, pd.DataFrame] = {}
    while len(obs_list) < sample_size and attempts < max_attempts:
        attempts += 1
        episode = episodes[rng.randrange(len(episodes))]
        if episode.pkl_path not in cache:
            cache[episode.pkl_path] = pd.read_pickle(episode.pkl_path).iloc[
                episode.start_idx : episode.end_idx
            ].copy()
        df = cache[episode.pkl_path]
        if len(df) <= episode.lookback + 2:
            continue
        idx = rng.randrange(episode.lookback, len(df) - 1)
        price_path = build_price_path(df)
        obs = make_observation(
            df,
            idx,
            pos=0.0,
            value=episode.cash,
            peak_value=episode.cash,
            ledger=TrancheLedger(),
            price_path=price_path,
            asset_idx=episode.asset_idx,
            observation_window=episode.observation_window,
            feature_columns=feature_columns,
        )
        rule_pos = float(rule_based_position(obs))
        if use_hybrid:
            action_idx = default_idx
        else:
            action_idx = int(np.argmin(np.abs(levels - rule_pos)))
        obs_list.append(obs)
        action_list.append(action_idx)

    if not obs_list:
        raise RuntimeError("build_warm_start_dataset produced 0 samples")
    return np.stack(obs_list, axis=0), np.asarray(action_list, dtype=np.int64)


def supervised_pretrain(
    actor: Actor,
    obs_array: np.ndarray,
    action_array: np.ndarray,
    steps: int,
    lr: float,
    batch_size: int,
) -> dict[str, float]:
    """Train ``actor`` with cross-entropy on (obs, action_idx) pairs.

    Returns a dict with final loss / accuracy for logging.
    """
    if steps <= 0 or len(obs_array) == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "steps": 0}

    device = next(actor.parameters()).device
    obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(action_array, dtype=torch.long, device=device)
    optim = torch.optim.Adam(actor.parameters(), lr=float(lr))
    loss_fn = torch.nn.CrossEntropyLoss()
    n = obs_tensor.shape[0]

    last_loss = float("nan")
    last_acc = float("nan")
    for step in range(int(steps)):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        batch_obs = obs_tensor[idx]
        batch_labels = action_tensor[idx]
        out = actor(batch_obs)
        logits = out[0] if isinstance(out, tuple) else out
        loss = loss_fn(logits, batch_labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        last_loss = float(loss.detach().cpu().item())
        with torch.no_grad():
            preds = logits.argmax(-1)
            last_acc = float((preds == batch_labels).float().mean().item())
    return {"loss": last_loss, "accuracy": last_acc, "steps": int(steps)}


def budget_usage_action_label(usage: float, n_actions: int) -> int:
    """Map a continuous budget usage ratio to the nearest discrete action."""

    if n_actions <= 1:
        return 0
    clipped = float(np.clip(usage, 0.0, 1.0))
    levels = np.linspace(0.0, 1.0, int(n_actions), dtype=np.float32)
    return int(np.argmin(np.abs(levels - clipped)))


def budget_usage_action_label_from_position(
    position: float,
    mode: str,
    profile: StrategyConfig,
    n_actions: int,
) -> int:
    """Map an oracle target position back to a budget-usage action label."""

    if profile.mode_router and getattr(profile.mode_router, "enabled", False):
        floor, cap = mode_band(str(mode), profile.mode_router)
    else:
        floor, cap = (0.0, 1.0)
    floor = float(floor)
    cap = float(cap)
    if cap <= floor:
        usage = 0.0
    else:
        usage = (float(position) - floor) / (cap - floor)
    return budget_usage_action_label(usage, n_actions)


def build_budget_usage_oracle_dataset_from_frame(
    df: pd.DataFrame,
    replay_daily: pd.DataFrame,
    profile: StrategyConfig,
    sample_size: int = 2000,
    rng_seed: int = 0,
    asset_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build no-promotion BC labels from a replay daily path.

    This is for representational-capacity diagnosis. Passing held-out replay
    rows is allowed only for a no-promotion upper-bound experiment.
    """

    if len(df) <= LOOKBACK + 2:
        raise ValueError("oracle budget-usage dataset requires a frame longer than LOOKBACK")
    required = {"datetime", "position", "mode"}
    missing = required.difference(replay_daily.columns)
    if missing:
        raise ValueError(f"replay daily missing required columns: {sorted(missing)}")

    label_frame = replay_daily[list(required)].copy()
    label_frame["datetime"] = pd.to_datetime(label_frame["datetime"])
    label_by_date = {
        pd.Timestamp(row["datetime"]).normalize(): (float(row["position"]), str(row["mode"]))
        for _, row in label_frame.iterrows()
    }

    def row_date(idx: int) -> pd.Timestamp:
        if "datetime" in df.columns:
            return pd.Timestamp(df.iloc[idx]["datetime"]).normalize()
        return pd.Timestamp(df.index[idx]).normalize()

    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    target_action_count = len(profile.model.delta_levels) if profile.model.use_hybrid_overlay else len(action_levels_for_profile(profile))
    price_path = build_price_path(df)
    cash = 100_000.0
    peak = cash
    pos = 0.0
    ledger = TrancheLedger()
    obs_list: list[np.ndarray] = []
    action_list: list[int] = []

    for idx in range(LOOKBACK, len(df) - 1):
        date = row_date(idx)
        if date not in label_by_date:
            continue
        oracle_pos, oracle_mode = label_by_date[date]
        obs = make_observation(
            df,
            idx,
            pos,
            cash,
            peak,
            ledger,
            price_path,
            asset_idx=asset_idx,
            observation_window=profile.model.observation_window,
            feature_columns=feature_columns,
        )
        label = budget_usage_action_label_from_position(
            oracle_pos,
            oracle_mode,
            profile,
            target_action_count,
        )
        obs_list.append(obs)
        action_list.append(label)

        next_ret = float(df.iloc[idx + 1]["return"])
        cash *= 1.0 + oracle_pos * next_ret
        peak = max(peak, cash)
        ledger.update(pos, oracle_pos, float(price_path[idx]))
        pos = oracle_pos

    if not obs_list:
        raise RuntimeError("oracle budget-usage dataset produced 0 samples")

    if sample_size > 0 and len(obs_list) > sample_size:
        rng = random.Random(rng_seed)
        indices = sorted(rng.sample(range(len(obs_list)), int(sample_size)))
        obs_list = [obs_list[i] for i in indices]
        action_list = [action_list[i] for i in indices]

    return np.stack(obs_list, axis=0), np.asarray(action_list, dtype=np.int64)


def build_mode_conditioned_budget_usage_oracle_dataset_from_frame(
    df: pd.DataFrame,
    replay_daily: pd.DataFrame,
    profile: StrategyConfig,
    sample_size: int = 2000,
    rng_seed: int = 0,
    asset_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build oracle labels and append the replay mode one-hot to each obs."""

    obs_array, action_array = build_budget_usage_oracle_dataset_from_frame(
        df,
        replay_daily=replay_daily,
        profile=profile,
        sample_size=0,
        rng_seed=rng_seed,
        asset_idx=asset_idx,
    )
    mode_frame = replay_daily[["mode"]].copy()
    conditioned_obs = np.stack(
        [append_mode_one_hot(obs, str(mode)) for obs, mode in zip(obs_array, mode_frame["mode"].tolist())],
        axis=0,
    )
    if sample_size > 0 and len(conditioned_obs) > sample_size:
        rng = random.Random(rng_seed)
        indices = sorted(rng.sample(range(len(conditioned_obs)), int(sample_size)))
        conditioned_obs = conditioned_obs[indices]
        action_array = action_array[indices]
    return conditioned_obs, action_array


def build_budget_usage_warm_start_dataset_from_frame(
    df: pd.DataFrame,
    profile: StrategyConfig,
    expert_actor: Actor,
    expert_profile: StrategyConfig,
    sample_size: int = 2000,
    rng_seed: int = 0,
    asset_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build BC labels that treat an expert fast position as budget usage.

    The rollout stays inside the supplied frame, so callers can pass only the
    train split and avoid teaching from held-out fixed-split windows.
    """

    if len(df) <= LOOKBACK + 2:
        raise ValueError("budget-usage warm-start requires a frame longer than LOOKBACK")

    target_feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    expert_feature_columns = feature_columns_for_packs(expert_profile.feature.packs, expert_profile.name)
    target_action_count = len(profile.model.delta_levels) if profile.model.use_hybrid_overlay else len(action_levels_for_profile(profile))
    target_price_path = build_price_path(df)
    expert_price_path = target_price_path

    target_cash = 100_000.0
    target_peak = target_cash
    target_pos = 0.0
    target_ledger = TrancheLedger()
    target_mode_state = ModeState()

    expert_cash = 100_000.0
    expert_peak = expert_cash
    expert_pos = 0.0
    expert_ledger = TrancheLedger()
    expert_mode_state = ModeState()

    obs_list: list[np.ndarray] = []
    action_list: list[int] = []

    for idx in range(LOOKBACK, len(df) - 1):
        target_obs = make_observation(
            df,
            idx,
            target_pos,
            target_cash,
            target_peak,
            target_ledger,
            target_price_path,
            asset_idx=asset_idx,
            observation_window=profile.model.observation_window,
            feature_columns=target_feature_columns,
        )
        if profile.mode_router and getattr(profile.mode_router, "enabled", False):
            target_mode_state = route_mode(target_obs, target_mode_state, profile.mode_router)

        expert_obs = make_observation(
            df,
            idx,
            expert_pos,
            expert_cash,
            expert_peak,
            expert_ledger,
            expert_price_path,
            asset_idx=asset_idx,
            observation_window=expert_profile.model.observation_window,
            feature_columns=expert_feature_columns,
        )
        if expert_profile.mode_router and getattr(expert_profile.mode_router, "enabled", False):
            expert_mode_state = route_mode(expert_obs, expert_mode_state, expert_profile.mode_router)

        expert_action = actor_action(expert_actor, expert_obs)
        expert_fast_position = target_position_for_profile(
            expert_action,
            expert_obs,
            config=expert_profile.position_cap,
            action_levels=action_levels_for_profile(expert_profile),
            hard_floor=expert_profile.model.hard_floor,
            use_hybrid_overlay=expert_profile.model.use_hybrid_overlay,
            delta_levels=np.asarray(expert_profile.model.delta_levels, dtype=np.float32)
            if expert_profile.model.use_hybrid_overlay
            else None,
            mode_band=mode_band(expert_mode_state.name, expert_profile.mode_router)
            if expert_profile.mode_router and getattr(expert_profile.mode_router, "enabled", False)
            else None,
            position_budget_config=expert_profile.position_budget,
            current_drawdown=max(1.0 - expert_cash / max(expert_peak, 1e-8), 0.0),
            position_mapping=expert_profile.model.position_mapping,
        )
        label = budget_usage_action_label(expert_fast_position, target_action_count)
        obs_list.append(target_obs)
        action_list.append(label)

        target_pos = target_position_for_profile(
            label,
            target_obs,
            config=profile.position_cap,
            action_levels=action_levels_for_profile(profile),
            hard_floor=profile.model.hard_floor,
            use_hybrid_overlay=profile.model.use_hybrid_overlay,
            delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32)
            if profile.model.use_hybrid_overlay
            else None,
            mode_band=mode_band(target_mode_state.name, profile.mode_router)
            if profile.mode_router and getattr(profile.mode_router, "enabled", False)
            else None,
            position_budget_config=profile.position_budget,
            current_drawdown=max(1.0 - target_cash / max(target_peak, 1e-8), 0.0),
            position_mapping=profile.model.position_mapping,
        )

        next_ret = float(df.iloc[idx + 1]["return"])
        target_cash *= 1.0 + target_pos * next_ret
        target_peak = max(target_peak, target_cash)
        target_ledger.update(target_pos, target_pos, float(target_price_path[idx]))

        expert_cash *= 1.0 + expert_fast_position * next_ret
        expert_peak = max(expert_peak, expert_cash)
        expert_ledger.update(expert_pos, expert_fast_position, float(expert_price_path[idx]))
        expert_pos = expert_fast_position

    if not obs_list:
        raise RuntimeError("budget-usage warm-start produced 0 samples")

    if sample_size > 0 and len(obs_list) > sample_size:
        rng = random.Random(rng_seed)
        indices = sorted(rng.sample(range(len(obs_list)), int(sample_size)))
        obs_list = [obs_list[i] for i in indices]
        action_list = [action_list[i] for i in indices]

    return np.stack(obs_list, axis=0), np.asarray(action_list, dtype=np.int64)


def actor_action(actor: Actor, obs: np.ndarray) -> int:
    if isinstance(actor, EnsembleActor):
        return actor.action(obs)
    obs = adapt_observation_for_actor(obs, actor)
    with torch.no_grad():
        out = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        return int(logits.argmax(-1).item())


class EnsembleActor:
    """Aggregate multiple actors by averaging their action probabilities.

    Each underlying actor may expect a different observation dimension (e.g.
    different `observation_window`). The ensemble adapts the observation per
    member via `adapt_observation_for_actor` before forwarding.

    Aggregation method:
    - `mean_logits`: softmax of logits per actor, averaged, then argmax
    - `mean_action`: argmax index per actor, averaged as float and rounded
    """

    def __init__(
        self,
        actors: list[Actor],
        method: str = "mean_logits",
        weights: list[float] | None = None,
    ) -> None:
        if not actors:
            raise ValueError("EnsembleActor requires at least one actor")
        self.actors = list(actors)
        self.method = method
        if weights is None:
            self.weights = [1.0 / len(self.actors)] * len(self.actors)
        else:
            if len(weights) != len(self.actors):
                raise ValueError("weights length must match actors length")
            total = float(sum(weights))
            if total <= 0.0:
                raise ValueError("weights must sum to a positive value")
            self.weights = [float(w) / total for w in weights]
        self.output_dim = getattr(self.actors[0], "output_dim", None)

    def eval(self) -> "EnsembleActor":
        for actor in self.actors:
            actor.eval()
        return self

    def action(self, obs: np.ndarray) -> int:
        if self.method == "mean_action":
            idxs: list[float] = []
            for actor, weight in zip(self.actors, self.weights):
                adapted = adapt_observation_for_actor(obs, actor)
                with torch.no_grad():
                    out = actor(torch.as_tensor(adapted, dtype=torch.float32).unsqueeze(0))
                    logits = out[0] if isinstance(out, tuple) else out
                    idxs.append(float(logits.argmax(-1).item()) * weight)
            return int(round(sum(idxs)))

        probs_accum: torch.Tensor | None = None
        for actor, weight in zip(self.actors, self.weights):
            adapted = adapt_observation_for_actor(obs, actor)
            with torch.no_grad():
                out = actor(torch.as_tensor(adapted, dtype=torch.float32).unsqueeze(0))
                logits = out[0] if isinstance(out, tuple) else out
                probs = torch.softmax(logits, dim=-1) * weight
            if probs_accum is None:
                probs_accum = probs
            else:
                if probs_accum.shape[-1] != probs.shape[-1]:
                    raise ValueError(
                        f"ensemble members have mismatched action dim: "
                        f"{probs_accum.shape[-1]} vs {probs.shape[-1]}"
                    )
                probs_accum = probs_accum + probs
        assert probs_accum is not None
        return int(probs_accum.argmax(-1).item())


__all__ = [
    "EnsembleActor",
    "PositionAction",
    "SoxlPPOPolicy",
    "SoxlSimulator",
    "SoxlState",
    "actor_action",
    "budget_usage_action_label",
    "budget_usage_action_label_from_position",
    "build_budget_usage_oracle_dataset_from_frame",
    "build_mode_conditioned_budget_usage_oracle_dataset_from_frame",
    "build_budget_usage_warm_start_dataset_from_frame",
    "build_initial_states",
    "build_policy",
    "build_warm_start_dataset",
    "supervised_pretrain",
]
