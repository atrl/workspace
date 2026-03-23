# SOXL RL Phase 4 - 研究指导与技术修复计划

## 📚 研究总结 (2026-03-23 15:20)

### 1. DeepScalper 论文要点 (AAAI 2024)

**核心架构**:
1. **Dueling Q-network + Action Branching**
   - 解决大动作空间问题
   - 分离状态价值和动作优势

2. **Hindsight Bonus 奖励函数**
   - 鼓励长期视角（整个交易日）
   - 避免短视决策

3. **Encoder-Decoder 架构**
   - 多模态时序嵌入
   - 整合宏观+微观市场信息

4. **风险感知辅助任务**
   - 平衡收益与风险
   - 关键：避免过拟合

**实验结果**:
- 3年真实数据
- 6个金融期货
- 显著优于 SOTA 基线

---

### 2. Sharpe 计算问题 (关键发现)

**正确计算方法**:

```python
# ❌ 错误（当前代码）
annual_factor = sqrt(252 * 78)  # = 140.2
sharpe = mean_ret / std_ret * annual_factor

# ✅ 正确（研究建议）
# 分钟级数据：6.5小时交易时间
n_periods = 252 * 6.5 * 60  # = 98,280
annual_factor = sqrt(n_periods)  # = 313.4

# 或者：使用每日收益聚合
daily_returns = aggregate_to_daily(minute_returns)
sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
```

**合理范围**:
- HFT策略：3-15（高个位数到低双位数）
- >3.0 可能过拟合
- 当前22.26 → **严重过拟合**

**必须包含**:
- 交易成本（佣金、滑点、冲击成本）
- 风险-free rate（短线可视为0）
- 非正态分布调整（峰度）

---

### 3. 过拟合检测与缓解

**当前问题**:
```
IS Sharpe:  22.26
OOS Sharpe: 13.74
偏差:       38% ⚠️

IS Return:  +214%
OOS Return: +18.9%
偏差:       91% ⚠️⚠️⚠️
```

**根因**:
1. 数据不足（1个月，测试集仅3天）
2. 参数过多（31个特征）
3. 无 Walk-forward 验证
4. 交易成本未完全包含

**解决方案**:

#### A. Walk-Forward 优化 (必需)
```python
# 时间序列交叉验证
for i in range(n_splits):
    train_start = start + i * step
    train_end = train_start + train_window
    test_end = train_end + test_window
    
    # 训练
    model.fit(data[train_start:train_end])
    
    # 测试
    metrics = model.evaluate(data[train_end:test_end])
    
    # 滚动窗口
    start += step
```

#### B. 简化模型
- 特征：31 → 15-20（去除高相关）
- 网络层：减少深度
- 参数：减少调优次数

#### C. 真实成本建模
```python
# 当前：0.1% 固定
commission = 0.001

# 改进：
commission = 0.0005  # 基础佣金
slippage = volume * impact_coefficient  # 滑点
market_impact = abs(position_change) * price * impact_factor
total_cost = commission + slippage + market_impact
```

#### D. 长期数据
- 目标：60天（28,080行）
- 包含多种市场状态（牛市、熊市、震荡）

---

## 🎯 技术修复计划

### Phase 4.3: Sharpe 计算修复 (P0)

**文件**: `src/rl/environment.py`, `scripts/evaluate_*.py`

**修改**:

```python
# environment.py - 奖励函数
def calculate_reward(self):
    if len(self.returns) > 10:
        mean_ret = np.mean(self.returns[-20:])
        std_ret = max(np.std(self.returns[-20:]), 0.001)
        
        # 使用每日收益，而非分钟收益
        daily_return = mean_ret * 78  # 每天78个5分钟
        daily_vol = std_ret * np.sqrt(78)
        
        # Sharpe-like奖励（不年化）
        reward = daily_return / daily_vol * 10
    else:
        reward = portfolio_return * 100
    
    return np.clip(reward, -10, 10)

# evaluate_*.py - Sharpe计算
def calculate_sharpe(returns):
    # 聚合到每日
    n_days = len(returns) // 78
    daily_returns = []
    
    for i in range(n_days):
        day_ret = returns[i*78:(i+1)*78]
        daily_returns.append(np.sum(day_ret))
    
    daily_returns = np.array(daily_returns)
    
    # 年化
    mean_daily = np.mean(daily_returns)
    std_daily = max(np.std(daily_returns), 0.001)
    sharpe = mean_daily / std_daily * np.sqrt(252)
    
    return sharpe
```

**预期**:
- IS Sharpe: 1.5-3.0
- OOS Sharpe: 1.0-2.5
- 偏差 < 30%

---

### Phase 4.4: 数据重建 (P0)

**文件**: `scripts/download_minute_data_v2.py`

**目标**:
- 60天数据（28,080行）
- 6个ETF
- 时间范围：2026-01-20 ~ 2026-03-23

**备选方案**:
- Alpha Vantage API
- IEX Cloud
- 本地历史数据

---

### Phase 4.5: Walk-Forward 验证 (P1)

**文件**: `scripts/train_walk_forward.py` (新建)

**实现**:

```python
def walk_forward_train(data, n_splits=5, train_ratio=0.8):
    """Walk-forward训练"""
    total_len = len(data)
    test_len = total_len // n_splits
    train_len = int(test_len * train_ratio)
    
    results = []
    
    for i in range(n_splits):
        # 分割
        test_start = i * test_len
        test_end = test_start + test_len
        train_start = max(0, test_start - train_len)
        train_end = test_start
        
        train_data = data[train_start:train_end]
        test_data = data[test_start:test_end]
        
        # 训练
        model = train_sac(train_data)
        
        # 测试
        metrics = evaluate(model, test_data)
        results.append(metrics)
    
    # 聚合结果
    mean_sharpe = np.mean([r['sharpe'] for r in results])
    std_sharpe = np.std([r['sharpe'] for r in results])
    
    return {
        'mean_sharpe': mean_sharpe,
        'std_sharpe': std_sharpe,
        'all_results': results
    }
```

---

### Phase 4.6: 模型简化 (P1)

**特征选择**:
```python
# 当前：31个特征
# 目标：15-20个核心特征

selected_features = [
    # 价格动量
    'return_1', 'return_5', 'return_10',
    
    # 波动率
    'volatility_10', 'volatility_20',
    
    # 趋势
    'ma_5_diff', 'ma_10_diff',
    
    # 技术指标
    'rsi_14', 'bb_position', 'atr_14',
    
    # 成交量
    'vol_ratio',
    
    # 动量
    'momentum_5', 'momentum_10',
]

# 去除：
# - 高相关特征（ma_20, bb_mid, bb_upper, bb_lower）
# - 冗余特征（dividends, stock_splits, capital_gains）
```

**网络架构**:
```python
# 当前：默认 MLP
# 改进：简化版

policy_kwargs = dict(
    net_arch=[64, 64],  # 减少层数
    activation_fn=torch.nn.ReLU,
)
```

---

## 📊 成功标准 (修订版)

**基于研究的合理目标**:

```
✅ Sharpe: 1.5-3.0 (非 20+)
✅ OOS偏差: < 30%
✅ Max DD: < 15%
✅ 年化收益: > 20%
✅ Walk-forward稳定性: std(sharpe) < 0.5
```

**对比 V25 基线**:
```
V25:  Sharpe 0.96, Max DD -8.69%
RL:   Sharpe > 1.5, Max DD < 15%, OOS稳定
```

---

## 🔬 研究参考

### DeepScalper (AAAI 2024)
- 标题: "DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities"
- 关键技术: Dueling Q-network, Hindsight Bonus, Encoder-Decoder
- 实验数据: 3年真实数据，6个金融期货
- 引用: https://arxiv.org/abs/2201.09008

### Sharpe Ratio in HFT
- 年化因子: sqrt(252 * 6.5 * 60) = 313.4
- 合理范围: 3-15
- 必须包含: 交易成本、滑点、市场冲击

### Overfitting Mitigation
- Walk-forward优化
- 简化模型（减少参数）
- 真实成本建模
- 长期数据（多种市场状态）

---

## 📋 执行顺序

**P0 - 立即修复**:
1. ✅ Sharpe计算修复
2. ⏳ 数据重建（60天）
3. ⏳ 重新训练

**P1 - 短期优化**:
4. ⏳ Walk-forward验证
5. ⏳ 特征选择（31→15）
6. ⏳ 网络简化

**P2 - 长期改进**:
7. ⏳ DeepScalper架构集成
8. ⏳ Qlib深度集成
9. ⏳ 实盘模拟

---

## 💡 关键建议 (基于研究)

### 1. 停止当前训练
**原因**:
- Sharpe 22.26 严重过拟合
- 评估逻辑错误
- 数据不足

### 2. 修复优先级
```
数据质量 > 评估准确性 > 模型复杂度
```

### 3. 简化优先
- 不要盲目增加复杂度
- DeepScalper的成功在于架构设计，而非参数数量
- 先确保基础正确，再考虑高级特性

### 4. 验证为王
- Walk-forward是必需，非可选
- OOS测试必须严格
- 长期数据才能验证鲁棒性

---

## 🚨 风险警告

**当前状态**: 
- ❌ Sharpe 22.26 (过拟合)
- ❌ OOS收益仅18.9%（暴跌91%）
- ❌ 仓位不稳定（做空→做多）

**建议**: 
**不要部署当前模型到实盘！**

**必须先完成**:
1. Sharpe修复
2. 数据重建
3. Walk-forward验证
4. OOS测试通过

---

**文档创建**: 2026-03-23 15:20
**作者**: simons (研究指导)
**执行者**: clawQuant
**状态**: 📋 待执行
