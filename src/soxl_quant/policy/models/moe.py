from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from tianshou.policy import PPOPolicy


class QNetwork(torch.nn.Module):
    """Wrapper around Actor to use as Q-network for discrete SAC."""

    def __init__(self, actor_net):
        super().__init__()
        self.net = actor_net

    def forward(self, obs, state=None, info=None):
        out, _state = self.net(obs, state, info)
        return out

    @property
    def device(self):
        return next(self.parameters()).device


class RegimeMoENet(torch.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_size: int = 64,
        n_experts: int = 3,
        device: str = "cpu",
        top_k: int = 1,
        noise_std: float = 0.1,
        shared_hidden_size: int = 16,
        regime_risk_idx: Optional[int] = None,
        hard_regime_routing: bool = False,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.hidden_size = hidden_size
        self.output_dim = hidden_size
        self.device = device
        self.top_k = top_k
        self.noise_std = noise_std
        self.regime_risk_idx = regime_risk_idx
        self.hard_regime_routing = hard_regime_routing
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_experts),
        )
        self.experts = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden_size), torch.nn.ReLU()) for _ in range(n_experts)]
        )
        self.shared_expert = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, shared_hidden_size),
            torch.nn.ReLU(),
        )
        self.output = torch.nn.Linear(hidden_size + shared_hidden_size, hidden_size)
        self._last_lb_loss: Optional[torch.Tensor] = None

    def _load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        f = gate_probs.mean(dim=0)
        p = gate_probs.mean(dim=0)
        return self.n_experts * (f * p).sum()

    def forward(self, obs, state: Any = None, info: Dict[str, Any] | None = None) -> Tuple[torch.Tensor, Any]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        orig_leading = obs.shape[:-1]
        if obs.ndim != 2:
            obs = obs.reshape(-1, obs.shape[-1])
        gate_logits = self.gate(obs)

        if self.training and self.noise_std > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std

        gate_probs_full = torch.softmax(gate_logits, dim=-1)
        self._last_lb_loss = self._load_balance_loss(gate_probs_full) if self.training else None

        if self.top_k >= self.n_experts:
            gate_weights = gate_probs_full
        else:
            topk_vals, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            gate_weights = torch.zeros_like(gate_logits)
            gate_weights.scatter_(-1, topk_indices, torch.softmax(topk_vals, dim=-1))

        if self.hard_regime_routing and self.regime_risk_idx is not None:
            regime_risk = obs[:, self.regime_risk_idx]
            hard_weights = torch.zeros_like(gate_weights)
            for batch_idx in range(gate_weights.shape[0]):
                rr = float(regime_risk[batch_idx])
                if rr < 0.35:
                    hard_weights[batch_idx, 0] = 1.0
                elif rr > 0.65:
                    hard_weights[batch_idx, 2] = 1.0
                else:
                    expert_idx = int(gate_weights[batch_idx].argmax())
                    hard_weights[batch_idx, expert_idx] = 1.0
            gate_weights = hard_weights

        expert_outs = torch.stack([expert(obs) for expert in self.experts], dim=1)
        routed = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        shared = self.shared_expert(obs)
        out = self.output(torch.cat([routed, shared], dim=-1))
        if orig_leading:
            out = out.reshape(*orig_leading, -1)
        return out, state

    def get_load_balance_loss(self) -> Optional[torch.Tensor]:
        return self._last_lb_loss


class SoxlPPOPolicy(PPOPolicy):
    """PPOPolicy 子类，自动将 MoE 的 load-balance loss 加入总损失。"""

    def __init__(self, *args, moe_lb_coef: float = 0.01, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._moe_lb_coef = moe_lb_coef

    def learn(self, batch: "Batch", batch_size: int, repeat: int, **kwargs: Any) -> Dict[str, list[float]]:  # type: ignore[override]
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / (std + self._eps)
                ratio = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss - self._weight_ent * ent_loss

                lb = None
                if hasattr(self.actor, "net"):
                    net = self.actor.net
                    if hasattr(net, "get_load_balance_loss"):
                        lb = net.get_load_balance_loss()
                if lb is not None and isinstance(lb, torch.Tensor):
                    loss = loss + self._moe_lb_coef * lb

                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(self._actor_critic.parameters(), max_norm=self._grad_norm)
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }


__all__ = ["QNetwork", "RegimeMoENet", "SoxlPPOPolicy"]
