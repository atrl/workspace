# SOXL 三倍ETF强化学习交易策略

## 目标
构建可上线的 Qlib 强化学习交易模型，针对 SOXL 等三倍杠杆 ETF 进行分钟级交易

## 核心挑战
1. **杠杆衰减** - 3x ETF 长期持有存在波动损耗
2. **高频数据** - 分钟级数据量大，特征工程复杂
3. **过拟合风险** - 强化学习容易在金融数据上过拟合
4. **交易成本** - 高频交易手续费侵蚀利润

## 研究方向（基于最新论文）

### 1. 算法选择
- **PPO (Proximal Policy Optimization)** - 稳定性好，适合连续动作空间
- **SAC (Soft Actor-Critic)** - 样本效率高，适合低频更新
- **TD3** - 适合确定性策略
- **Ensemble DRL** - 多算法集成降低方差

### 2. 环境设计 (Gym-based)
```python
State Space:
  - 价格特征: OHLCV 分钟数据 (窗口 60-240 分钟)
  - 技术指标: RSI, MACD, Bollinger Bands, ATR
  - 市场微观结构: 成交量分布, 订单流不平衡
  - 波动率: realized vol, GARCH 预测
  - 相关性: 与 SOXX/QQQ 的相对表现

Action Space:
  - 连续: 仓位权重 [0, 1] (做空需考虑T+1)
  - 或离散: {买入, 持有, 卖出}

Reward Function:
  - Sharpe-adjusted return
  - Risk penalty (drawdown, volatility)
  - Transaction cost penalty
```

### 3. 关键论文参考
1. **QTNet (arxiv 2312.15730)** - 分钟频率数据的自适应交易模型
2. **Multi-time DRL Framework (2024)** - 多时间尺度特征融合
3. **QlibRL** - Microsoft 官方 RL 工具包

### 4. 训练策略
- **课程学习**: 先学日级，再学分钟级
- **模仿学习**: 用历史最优策略初始化
- **Risk-aware RL**: 在 reward 中加入 VaR/CVaR 约束
- **Domain Randomization**: 模拟不同市场状态（牛市/熊市/震荡）

### 5. 回测与验证
- Walk-forward validation (避免未来数据泄露)
- Out-of-sample 测试（至少 6 个月）
- 交易成本敏感性分析
- 与 buy-and-hold、动量策略对比

## 技术架构

```
数据层
  └── 分钟级 OHLCV 数据 (Yahoo/Tiingo/自有数据)
  └── 特征工程 (技术指标 + 微观结构)
  
环境层 (Qlib/Gym)
  └── TradingEnv (state/reward/action)
  └── 滑点/手续费模拟
  
模型层
  └── 策略网络 (LSTM/Transformer)
  └── 价值网络 (Critic)
  └── RL 算法 (PPO/SAC/TD3)
  
执行层
  └── 信号生成
  └── 仓位管理
  └── 风控模块
```

## 开发路线图

### Phase 1: 数据与特征 (Week 1)
- [ ] 创建 worktree
- [ ] 分钟级数据获取与清洗
- [ ] 特征工程 pipeline

### Phase 2: 环境搭建 (Week 1-2)
- [ ] Qlib RL 环境配置
- [ ] 自定义 TradingEnv
- [ ] Reward function 设计

### Phase 3: 模型训练 (Week 2-3)
- [ ] 基线模型 (PPO)
- [ ] 超参调优
- [ ] Ensemble 方法

### Phase 4: 验证与部署 (Week 3-4)
- [ ] 回测与压力测试
- [ ] 模型持久化
- [ ] 上线监控

## 风险提示
- 三倍 ETF 不适合长期持有
- 模型在极端行情下可能失效
- 需要持续监控和再训练
