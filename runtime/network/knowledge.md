# NETWORK Knowledge

_Auto-generated from memory/ integration on 2026-04-24._
_Total entries: 3._

## 2026-04-18 — RL Policy Module Migration (`dropped`)
Legacy memory note normalized into the repo-wide schema. Date: 2026-04-18 Context: package folders under `src/soxl_quant/` existed, but `src/soxl_quant/policy/*` was still mostly wrapper code pointing back into `src/soxl_quant/rl.py` that m

## 2026-04-22 — Simplified Reward Failure and Refined Direction (`reference`)
simplified_v1 failed: proxy MaxDD 58.5% vs v19 54.8%. Root cause: stripping reward removed necessary guardrails (stale_position, trapped_loser, downside_signal, risk_budget). User wants: cash respected, conflicting gradients reduced, generalization to other 3x ETFs. New approach: remove activity_bonus (forced trading), reduce inactivity_penalty, strengthen risk guardrails, keep proven components. Network returns to [128,64]+GRU-64. Multi-asset joint training is the next frontier after single-asset reward is fixed.

## 2026-04-24 — Learned-router attention oracle distillation (`dropped`)
Swapping the learned-router oracle probe backbone from GRU to attention produced effectively identical policy metrics. This compresses the search space further: same-scale backbone swaps are not enough to close the learned-router gap.
