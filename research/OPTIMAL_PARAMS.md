# 最优策略参数

## 遗传算法优化结果 (2026-04-05)

### 性能指标
```
总收益:        1603.19%
年化收益:      59.03%
最大回撤:      55.11%
Sharpe:        1.07
Calmar:        1.07
交易次数:      193
最终价值:      $1,703,188
收益倍数:      17.03x
超额收益:      1446.61% (vs 买入持有 156.58%)
```

### 最优参数

```yaml
# 恐慌参数
panic_threshold: -35.2       # 恐慌得分阈值（降低触发）
panic_position: 0.98         # 恐慌时最大仓位

# 趋势参数
trend_position: 0.4382       # 趋势时仓位

# VIX阈值
vix_panic: 20.4              # VIX恐慌阈值（降低）
vix_extreme: 28.5            # VIX极端阈值
vix_greed: 12.7              # VIX贪婪阈值（降低）

# 风控
stop_loss: -0.08             # 硬止损（收紧）
```

### 关键改进

1. **VIX阈值降低** - 触发更频繁，捕捉更多恐慌机会
   - panic: 27.4 → 20.4
   - greed: 18.0 → 12.7

2. **仓位更激进** - 恐慌时满仓
   - panic_position: 94% → 98%

3. **止损更严** - 快速止损
   - stop_loss: -12% → -8%

4. **趋势保守** - 趋势市降低仓位
   - trend_position: 64% → 44%

5. **交易克制** - 减少交易频率
   - 405次 → 193次 (-52%)

### 策略逻辑

```python
if VIX > 28.5:
    # 极端恐慌 - 满仓抄底
    if PanicScore <= -35.2:
        target_position = 0.98
        
elif VIX > 20.4:
    # 恐慌 - 分批建仓
    if PanicScore <= -35.2:
        if current_position < 0.5:
            target_position = 0.6
        elif current_position < 0.7:
            target_position = 0.8
        else:
            target_position = 0.98

elif VIX < 12.7:
    # 贪婪 - 保守趋势
    if TrendBuy:
        target_position = 0.4
    else:
        target_position = 0.2

else:
    # 正常 - 趋势跟踪
    if TrendBuy:
        target_position = 0.4382
    else:
        target_position = 0.3
```

### 对比所有版本

| 策略 | 收益 | Sharpe | 回撤 | 交易 | 倍数 |
|------|------|--------|------|------|------|
| 原始版本 | 146.64% | 0.44 | 49.40% | 405 | 2.47x |
| 手动优化 | 324.97% | 0.61 | 60.76% | 262 | 4.25x |
| **遗传算法** | **1603.19%** | **1.07** | 55.11% | 193 | **17.03x** |

### 使用方法

```python
# 在 qlib_framework_implementation.py 中使用
params = StrategyParams(
    panic_buy_threshold=-35.2,
    panic_position=0.98,
    trend_position=0.4382,
    vix_high=20.4,
    vix_low=12.7,
    stop_loss=-0.08,
)
```
