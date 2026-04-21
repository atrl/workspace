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
from .positioning import action_levels_for_profile, rule_based_position
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
        actor_optim = torch.optim.Adam(actor.parameters(), lr=profile.lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=profile.lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=profile.lr)
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
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=profile.lr)
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
    "build_initial_states",
    "build_policy",
    "build_warm_start_dataset",
    "supervised_pretrain",
]
