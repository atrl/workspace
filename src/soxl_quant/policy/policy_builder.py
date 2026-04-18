from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from gymnasium import spaces
from tianshou.policy import DiscreteSACPolicy, PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from ..config import StrategyConfig
from ..data import FEATURE_COLUMNS, STATIC_INDEX
from .observation import adapt_observation_for_actor
from .positioning import action_levels_for_profile
from .environment import PositionAction, SoxlSimulator, SoxlState, build_initial_states
from .models.moe import QNetwork, RegimeMoENet, SoxlPPOPolicy
from .models.tide import TiDENet


def _build_net(obs_dim: int, profile: StrategyConfig, model_cfg):
    effective_use_tide = bool(model_cfg.use_tide)
    effective_use_moe = bool(model_cfg.use_regime_router)
    effective_hard_routing = bool(model_cfg.hard_regime_routing)
    if effective_use_tide:
        return TiDENet(
            obs_dim,
            hidden_size=int(model_cfg.tide_hidden_size),
            feature_dim=len(FEATURE_COLUMNS),
            observation_window=max(int(model_cfg.observation_window), 1),
            dropout=float(model_cfg.tide_dropout),
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
            regime_risk_idx=STATIC_INDEX["regime_risk"],
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


def actor_action(actor: Actor, obs: np.ndarray) -> int:
    obs = adapt_observation_for_actor(obs, actor)
    with torch.no_grad():
        out = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        return int(logits.argmax(-1).item())


__all__ = [
    "PositionAction",
    "SoxlPPOPolicy",
    "SoxlSimulator",
    "SoxlState",
    "actor_action",
    "build_initial_states",
    "build_policy",
]
