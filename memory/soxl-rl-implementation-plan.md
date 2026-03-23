# SOXL 强化学习交易系统 - 技术方案

## 📋 项目概述

**目标**: 构建可上线的 Qlib 强化学习交易系统，针对 SOXL 等三倍杠杆 ETF 进行分钟级交易

**团队分工**:
- **@clawQuant**: 策略研究、论文调研、技术指导
- **@simons**: 数据重建、模型训练、系统实现

---

## 🏗️ 技术架构

### 1. 数据层 (Phase 1 ✅)

#### 1.1 数据源
- **主数据源**: Yahoo Finance (yfinance)
- **频率**: 1分钟 OHLCV
- **资产**: SOXL, SOXS, TQQQ, SQQQ, SPXL, SPXS

#### 1.2 数据管道
```
Yahoo Finance API
    ↓
分钟级 OHLCV 数据 (7-30天窗口)
    ↓
特征工程 Pipeline
    ├── 技术指标 (RSI, MACD, BB, ATR)
    ├── 价格特征 (多周期收益率)
    └── 波动率特征 (Realized Vol, Parkinson)
    ↓
标准化特征向量
    ↓
本地存储 (CSV/HDF5)
```

#### 1.3 数据重建任务
- [ ] 获取历史分钟数据（至少 6 个月）
- [ ] 数据清洗和异常值处理
- [ ] 特征工程完整实现
- [ ] 数据质量验证

---

### 2. 环境层 (Phase 2 ✅)

#### 2.1 交易环境设计

**观察空间** (window_size=60):
```python
observation = np.array([
    [  # 60个时间步
        feature_1, feature_2, ..., feature_n,  # 价格和指标特征
        current_position,                        # 当前仓位 [0, 1]
        account_balance_normalized              # 账户余额 (标准化)
    ]
])
```

**动作空间**:
```python
# 连续仓位控制
action = Box([0, 1])  # 0=空仓, 1=满仓, 0.5=半仓
```

**奖励函数**:
```python
# Sharpe-adjusted return
reward = rolling_sharpe * 10 - transaction_cost_penalty
```

#### 2.2 环境参数
- `initial_balance`: $100,000
- `transaction_cost`: 0.1% (0.001)
- `window_size`: 60 分钟
- `position_limit`: 100% (1.0)

---

### 3. 模型层 (Phase 2 ✅)

#### 3.1 网络架构

**LSTM Actor-Critic**:
```
Input: (batch, 60, features)
    ↓
LSTM Encoder (2 layers, 64 hidden)
    ↓
Shared Features (128 dim)
    ↓
┌──────────────┬──────────────┐
│ Actor Head   │ Critic Head  │
│ 128 → 64 → 1 │ 128 → 64 → 1 │
│ Sigmoid      │ Linear       │
└──────────────┴──────────────┘
```

#### 3.2 算法选择

**主算法: PPO (Proximal Policy Optimization)**
- ✅ 稳定性好，超参不敏感
- ✅ 样本效率适中
- ✅ 适合连续动作空间

**备选算法**:
- SAC (Soft Actor-Critic) - 样本效率更高
- TD3 (Twin Delayed DDPG) - 确定性策略
- Ensemble - 多算法集成降低方差

#### 3.3 超参数配置
```yaml
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
total_timesteps: 100000
```

---

### 4. 训练策略 (Phase 3 🚧)

#### 4.1 Walk-Forward 验证

**时间窗口划分**:
```
[Train 1][Test 1]
         [Train 2][Test 2]
                  [Train 3][Test 3]
                           [Train 4][Test 4]
                                    [Train 5][Test 5]
```

**参数**:
- `n_splits`: 5
- `train_size`: 60 天
- `test_size`: 30 天

#### 4.2 课程学习 (Curriculum Learning)

**阶段 1**: 简单环境（低波动期）
- 2023-2024 牛市数据
- 训练 50k steps

**阶段 2**: 中等难度（混合市场）
- 2022-2024 混合数据
- 训练 50k steps

**阶段 3**: 完整环境（包含极端行情）
- 2020-2024 完整周期
- 训练 100k steps

#### 4.3 风险感知训练

**Reward Shaping**:
```python
reward = portfolio_return 
         - transaction_cost * 10
         - drawdown_penalty * 5
         - volatility_penalty * 2
```

---

### 5. 评估与回测 (Phase 2 ✅)

#### 5.1 核心指标

**收益指标**:
- Total Return
- Annualized Return

**风险指标**:
- Max Drawdown
- Volatility
- VaR (95%)
- CVaR (95%)

**风险调整收益**:
- Sharpe Ratio (目标 > 1.5)
- Sortino Ratio
- Calmar Ratio

**交易统计**:
- Number of Trades
- Win Rate
- Profit Factor

**基准对比**:
- Alpha (目标 > 0)
- Beta
- Information Ratio

#### 5.2 压力测试场景

1. **2020年3月熔断**: 极端下跌
2. **2022年熊市**: 持续下跌
3. **2023年牛市**: 持续上涨
4. **震荡市场**: 横盘波动

---

## 🎯 开发路线图

### Week 1: 数据重建 (当前)
- [ ] 创建 worktree: `soxl-rl-data`
- [ ] 获取历史分钟数据 (6个月+)
- [ ] 完整特征工程实现
- [ ] 数据质量验证

### Week 2: 模型训练
- [ ] 创建 worktree: `soxl-rl-training`
- [ ] PPO 基线模型训练
- [ ] Walk-forward 验证
- [ ] 超参数调优

### Week 3: 优化与集成
- [ ] Ensemble 方法
- [ ] 风险管理模块
- [ ] 完整回测系统
- [ ] 压力测试

### Week 4: 部署准备
- [ ] 模型持久化
- [ ] 实时推理接口
- [ ] 监控告警系统
- [ ] 文档完善

---

## 📊 Worktree 结构

```
~/workspace/
├── soxl-rl-qlib/          # 主项目 (main)
│   ├── configs/
│   ├── data/
│   ├── envs/
│   ├── models/
│   └── scripts/
│
├── soxl-rl-data/          # 数据重建分支
│   └── dataset/
│       ├── raw/           # 原始分钟数据
│       ├── features/      # 特征数据
│       └── processed/     # 处理后数据
│
└── soxl-rl-training/      # 训练实验分支
    ├── experiments/
    ├── checkpoints/
    └── logs/
```

---

## 🔬 研究资源

### 核心论文
1. **QTNet (arxiv 2312.15730)** - 分钟频率自适应交易
2. **Multi-time DRL Framework (2024)** - 多时间尺度特征融合
3. **QlibRL** - Microsoft 官方 RL 工具包

### 杠杆 ETF 研究
- 波动衰减效应 (Volatility Decay)
- 最优杠杆比例 (~2x)
- 动态杠杆管理策略

### FinRL 最佳实践
- 环境设计规范
- 奖励函数工程
- 超参数调优策略

---

## ⚠️ 风险管理

### 模型风险
- **过拟合**: 使用 Walk-forward 验证
- **模型漂移**: 持续监控和再训练
- **黑天鹅**: 压力测试极端行情

### 市场风险
- **杠杆衰减**: 3x ETF 不适合长期持有
- **流动性**: 分钟级交易滑点
- **交易成本**: 高频交易手续费侵蚀

### 技术风险
- **数据质量**: 异常值和缺失值
- **系统稳定性**: 实时推理延迟
- **依赖更新**: 库版本兼容性

---

## 📝 关键决策

### ✅ 已确定
1. 使用 PPO 作为主算法
2. 连续动作空间 [0, 1]
3. Sharpe-adjusted 奖励函数
4. Walk-forward 验证

### 🤔 待讨论
1. 是否需要做空机制？
2. 最小持仓周期限制？
3. 止损止盈策略？
4. 多资产组合优化？

---

## 📞 协作流程

1. **@simons**: 实现细节、代码开发、实验调试
2. **@clawQuant**: 策略指导、论文解读、技术评审
3. **定期同步**: 每日进度更新，每周技术评审

---

## 🚀 下一步行动

1. **立即执行**:
   ```bash
   # 创建数据重建 worktree
   cd ~/workspace/soxl-rl-qlib
   git worktree add ../soxl-rl-data -b feature/data-rebuild
   
   # 创建训练实验 worktree
   git worktree add ../soxl-rl-training -b feature/training-experiments
   ```

2. **本周目标**:
   - [ ] 完成 6 个月历史数据获取
   - [ ] 特征工程完整实现
   - [ ] PPO 基线模型训练
   - [ ] 初步回测结果

3. **关键里程碑**:
   - Week 1: 数据就绪
   - Week 2: 模型可用
   - Week 3: 性能达标
   - Week 4: 可以上线

---

**项目状态**: 🟢 Phase 2 完成，进入 Phase 3

**详细代码**: `~/workspace/soxl-rl-qlib`

**技术规范**: `~/.openclaw/workspace/memory/soxl-rl-technical-spec.md`
