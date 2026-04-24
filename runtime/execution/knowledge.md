# EXECUTION Knowledge

> 最后更新: 2026-04-24  
> 该方向负责执行层策略：仓位管理、交易频率控制、mode routing 和 overlay。

## 核心结论

1. **Position cap 是最有效的风控工具**: drawdown-based 动态仓位上限（min_cap 从 0.40 起步）比固定仓位更优。
2. **Decision frequency = 3 是 sweet spot**: 降低交易频率到每 3 天一次，2022 MaxDD 从 60.56% 降至 38.4%。
3. **Trade lock（3-day cooldown）有效**: 减少微交易摩擦，RL 交易次数从 1000+ 降至 ~100。
4. **Regime-conditional overlay 有方向性**: vol_stress 特征的 regime gating 改善 stress 态表现，但非万能。
5. **Generic overlay retuning 是重复工作**: stress/bull 阈值微调已被多次验证为边际收益。

## 核心机制

### Drawdown-based Position Cap
```python
position_cap = f(current_drawdown)
# start_drawdown = 5% 开始降仓
# full_drawdown = 25% 达到 min_cap
# min_cap = 0.40 (symmetric floor)
```

### Decision Frequency
```
frequency = 3  # 每 3 个交易日决策一次
# vs frequency = 1: 交易次数从 ~1000 降至 ~300
# vs frequency = 5: 回报损失过大
```

### Trade Lock
```
rebalance_threshold = 0.05  # 5% 偏离才触发
trade_lock_days = 3         # 交易后锁定 3 天
```

### Mode Routing
```
mode = route_mode(regime_risk, anchor_gap_180)
# bull:   regime_risk < 0.3, anchor_gap > 0.01
# stress: regime_risk > 0.6
# base:   中间态
```

## 有效方法 ✅

| 方法 | 关键参数 | 验证结果 |
|------|---------|---------|
| Symmetric floor 40% | min_cap = 0.40 | A+B 突破: 698% return（但 MaxDD 60.56% fail gate） |
| Decision frequency = 3 | freq = 3 | 2022 MaxDD 38.4%，return 降至 314% |
| 3-day trade lock | lock = 3 | 摩擦从 1000+ 笔降至 ~100 笔 |
| Asymmetric threshold | buy=0.02, sell=0.05 | 方向性改善，但未通过 promotion gate |
| Regime-conditional overlay | vol_stress gating | stress 态 IC 改善，test 表现边际提升 |

## 无效方法 ❌

| 方法 | 失败原因 |
|------|---------|
| Regime gate (balanced_regime_gate) | 未通过 fixed-split 2022 gate，avg_pos 仍高 |
| Generic overlay retuning | 仅调整 stress/bull 阈值是重复工作，无新机制 |
| Decision frequency = 1 | 过度交易，摩擦损耗大 |
| Asymmetric threshold v2 | 与 v1 类似，未带来突破 |

## 待验证方向

- [ ] 显式 release/re-entry 状态的 overlay 机制（非阈值微调）
- [ ] 动态 decision frequency（stress 时低频，bull 时高频）
- [ ] 仓位上限与 VIX term structure 联动

---

## 实验记录（参考）

见 memory archive 原始记录，按日期排序。
