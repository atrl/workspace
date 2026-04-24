# FACTOR Knowledge

> 最后更新: 2026-04-24  
> 该方向负责特征工程、因子有效性验证（时序 IC）和信号回测。

## 核心结论

1. **核心特征体系**包含 ~50 个时序特征，分 6 类：趋势锚定、短期动量、波动率、回撤/压力、VIX、体制风险。
2. **regime_risk** 是最有效的体制分类器，决定 bull/base/stress 三态路由。
3. **时序 IC 门槛**: mean_ic > 0.03, ic_positive_ratio > 0.55, signal_accuracy > 0.55。
4. **过渡态特征**（transition_v23）实验失败：7 个 stress-release 检测特征导致过拟合，未提升测试集表现。
5. **VIX-free 替代方案**有效：当 VIX 缺失时，realized_kurtosis + jump_variation + vol_stress_score 可替代体制风险计算。

## 特征公式

### 趋势锚定
```
anchor_gap_180 = close / MA(close, 180) - 1
anchor_slope_180 = slope(MA(close, 180), 63)
trend_gap_21   = close / MA(close, 21) - 1
trend_slope_21 = slope(MA(close, 21), 21)
```

### 波动率
```
realized_vol_N = Std(return, N) * sqrt(252)   # N ∈ {10, 21, 63}
vol_of_vol_21  = Std(realized_vol_21, 21)
vol_coef_21    = realized_vol_21 / median(realized_vol_21, 252)
```

### 体制风险（regime_risk）
```
regime_risk =
  0.25 * vix_zscore
+ 0.18 * percentile(realized_vol_21)
+ 0.20 * percentile(drawdown_63)
+ 0.15 * vol_coef_21
+ 0.12 * percentile(loss_streak_21)
+ 0.10 * percentile(downside_ratio_21)
```

### VIX-free 替代（regime_risk_no_vix）
```
regime_risk_no_vix =
  0.30 * percentile(realized_vol_21)
+ 0.25 * percentile(drawdown_63)
+ 0.20 * realized_kurtosis_21.rank(pct)
+ 0.15 * jump_variation_5.rank(pct)
+ 0.10 * vol_stress_score.rank(pct)
```

## 有效方法 ✅

| 方法 | 验证结果 | 状态 |
|------|---------|------|
| regime_risk 体制分类 | 三态切片显示 IC 显著分化，stress 态 IC 最高 | 已固化 |
| VIX-free stress proxy | ablation 通过，VIX 缺失时 regime_risk_no_vix 替代有效 | 已固化 |
| 时序 IC 快筛 | 1d/5d/10d/20d 多 horizon 检验，IC 衰减分析 | 已固化 |
| vol_stress_score | 新增特征，与现有特征正交 (|rho| < 0.80)，IC 显著 | 候选入库 |

## 无效方法 ❌

| 方法 | 失败原因 |
|------|---------|
| transition_v23 特征包 | 7 个 stress-release 检测特征导致训练集过拟合，test_2022 MaxDD 反而恶化 |
| 纯 price-volume 动量 | 与 return 高度相关，未提供增量信息 |

## 待验证方向

- [ ] 跨标的因子稳定性：SOXL 有效的特征在 TQQQ/UPRO 上是否同样有效
- [ ] 高频因子（intraday）引入：当前仅使用日频
- [ ] 宏观因子（利率、信用利差）与 LETF 的 regime 关联

---

## 实验记录（参考）

见 memory archive 原始记录，按日期排序。
