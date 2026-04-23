---
date: '2026-04-22'
title: "2026-04-22: Rulebase Imitation Bonus \u2014 Full Experiment Series (v1-v7)"
kind: review
profiles: []
status: kept
tags:
- eval-2022
- kept
- review
- walk-forward
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: "Legacy memory note normalized into the repo-wide schema. \u9488\u5BF9\u8DD1\
  \u4E0D\u8FC7rulebase\u7684\u95EE\u9898\uFF0C\u5982\u679C\u5728rulebase\u7684\"\u57CB\
  \u70B9\"\u4E0A\u7ED9\u8D85\u989D\u5956\u52B1\uFF0C\u80FD\u63D0\u9AD8\u5B66\u4E60\
  rulebase\u7684\u80FD\u529B\u5417\uFF1F Imitation bonus DOES improve the model's\
  \ ability to learn rulebase timing, as evidenced by v1's 812% proxy return (vs v16\
  \ ba"
---

# 2026-04-22: Rulebase Imitation Bonus — Full Experiment Series (v1-v7)

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Original Question

针对跑不过rulebase的问题，如果在rulebase的"埋点"上给超额奖励，能提高学习rulebase的能力吗？

## Answer: Yes, but with important caveats

Imitation bonus DOES improve the model's ability to learn rulebase timing, as evidenced by v1's 812% proxy return (vs v16 baseline 632%). However, excessive imitation causes overfitting. The optimal balance requires:
1. A small imitation bonus (0.10-0.15) for dense positive baseline
2. Strong episodic drawdown controls to prevent copying rulebase's high-exposure behavior
3. Participation signals (trend_alignment, opportunity_cost) to prevent excessive conservatism

## Full Results Comparison

| Version | Key Delta | Proxy Return | Proxy MaxDD | test_2022 MaxDD | test_2023_2024 Return | test_2023_2024 MaxDD |
|---------|-----------|-------------|-------------|-----------------|----------------------|---------------------|
| Rulebase | - | 688.6% | 69.9% | 54.2% | 112.5% | 41.6% |
| v16 baseline | - | 632.4% | 50.1% | ~34.6% | - | - |
| **v1** | imitation=0.20 | 812.3% | 51.9% | 32.2% | 19.6% | 38.5% |
| **v2** | imitation=0.10, tight DD | 448.3% | 46.9% | **30.2%** | 34.0% | **29.4%** |
| **v3** | v4 + alpha=3.5 | 776.9% | 58.5% | 34.0% | 50.4% | 36.1% |
| **v4** | v2 + strong participation | 500.8% | 49.4% | 31.3% | **51.8%** | 31.1% |
| **v5** | v4 + episodic=4.0 | 573.7% | 52.0% | 32.4% | 40.7% | 33.1% |
| **v6** | v4 + hard_floor=0.10 | 481.6% | 53.2% | 34.3% | 42.5% | 36.0% |
| **v7** | v2/v4 compromise | 686.3% | 52.2% | 31.9% | 40.4% | 33.3% |

## Key Findings

### 1. Imitation Bonus Effect
- `rule_imitation_bonus=0.20` (v1): Too strong, causes overfitting. Proxy return 812% but OOS weak.
- `rule_imitation_bonus=0.10` (v2/v4): Better balance. Provides dense baseline without excessive copying.

### 2. Drawdown Control Framework
- v2's framework (episode_penalty=3.5, step_penalty=0.30) achieves the best MaxDD control
- Tightening further (v5: episode_penalty=4.0) causes training degeneration and WORSE MaxDD
- Lowering hard_floor (v6: 0.20→0.10) increases trading noise and worsens MaxDD

### 3. Participation Signals Trade-off
- v2: Low participation (trend_alignment=0.25, opportunity=0.05) → best MaxDD, weak returns
- v4: High participation (trend_alignment=0.40, opportunity=0.20) → best returns, MaxDD +1-1.3pp
- There's a fundamental trade-off: more participation → better returns → higher MaxDD

### 4. Training Iterations
- All versions with max_iters=80 showed degeneration after early iterations
- max_iters=20 prevents degeneration and is sufficient

## Best Version: v4

v4 achieves the best overall balance:
- Proxy: 500.8% / 0.86 / 49.4%
- test_2020: 111.6% / 2.67 / 17.3%
- test_2022: -22.7% / -0.85 / 31.3% (vs rulebase -48.2% / 54.2%)
- test_2023_2024: 51.8% / 0.80 / 31.1%

Misses the 30% MaxDD target by only 1-1.3pp, but has the best walk-forward returns.

## Recommendation

For production use, v4 is the recommended candidate. To push MaxDD below 30%, consider:
1. Using v2's model (MaxDD 30.2% / 29.4%) if walk-forward returns of 34% are acceptable
2. Further research on regime-conditional position sizing
3. Alternative: use v4 for bull markets and v2 for bear markets (ensemble approach)

## Code Changes Made

- `src/soxl_quant/policy/environment.py`: Added `prev_rule_pos` to state
- `src/soxl_quant/policy/reward.py`: Added `_moderate_reward` branch with imitation bonus
- `src/soxl_quant/config.py`: Added `use_moderate_reward`, `rule_imitation_bonus`, `rule_decision_bonus`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
