# v10 Expectations

## Promotion Gate Targets

| Check | Threshold | v2 Actual | v8/v9 Actual | v10 Target |
|-------|-----------|-----------|--------------|------------|
| test_2022_max_dd | < 35.0% | 23.6% ✓ | 31.3% ✓ | < 30.0% |
| test_2022_avg_pos | < 0.35 | 0.18 ✓ | 0.27 ✓ | < 0.25 |
| test_proxy_sharpe | > 0.75 | 0.86 ✓ | 0.88 ✓ | > 0.80 |
| test_proxy_max_dd | < 50%* | 50.4% | 53.4% | < 48% |

*Primary optimization target.

## Success Criteria

1. **valid best score > 95** — 必须恢复 v2 级别的学习能力（v8/v9 只有 89.54）
2. **test_proxy_maxdd < 48%** — episode-level penalty 必须产生可见的回撤改善
3. **test_2022_max_dd < 30%** — 保持 2022 控制优势
4. **test return > 450%** — 不能太保守

## Failure Modes

- If valid score < 90: episode-level penalty 与 PPO credit assignment 冲突，需要调整实现方式
- If maxdd > 52%: episode-level penalty 力度不足，需要提高至 4.0+
- If return < 350%: agent 过于保守，需要降低 episode_drawdown_penalty 或提高 threshold
- If 2022_max_dd > 35%: downside asymmetry 未能改善 bear-market 行为
