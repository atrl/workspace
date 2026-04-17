---
name: qlib-rl-trading
description: 使用Qlib强化学习框架训练单标的动态仓位管理策略。支持PPO算法，适用于任何单一资产的择时交易（ETF、股票、期货等）。完整工作流：数据准备→环境创建→模型训练→回测验证→极端行情测试。借鉴agentic-alpha的R-G-E-V-D方法论。支持自定义特征、外部数据（VIX、宏观指标等）、风控参数。
---

# Qlib RL Trading Skill

基于Qlib官方RL框架的强化学习交易策略开发。

## 适用场景

- ✅ **单标的择时** - ETF、个股、期货、加密货币等
- ✅ **动态仓位管理** - RL学习何时加仓/减仓/空仓
- ✅ **多时间框架** - 日频、周频、分钟频
- ✅ **自定义特征** - 技术指标、外部数据、基本面因子
- ✅ **极端行情保护** - 自定义风控规则

## 核心工作流 (R-G-E-V-D)

```
Research → Generate → Evaluate → Validate → Distill
```

| 阶段 | 目标 | 产出 |
|------|------|------|
| **Research** | 定义问题、目标、约束 | 任务配置文件 |
| **Generate** | 创建环境、定义奖励函数 | RL环境 + 奖励函数 |
| **Evaluate** | 训练模型、初步回测 | 训练模型 + 基础指标 |
| **Validate** | 极端行情测试、稳定性验证 | 验证报告 |
| **Distill** | 文档化、沉淀经验 | 策略文档 + 经验库 |

---

## 快速开始

### 1. 准备数据

```bash
# 准备训练数据（任意标的）
uv run python scripts/prepare_rl_data.py \
  --ticker <YOUR_TICKER> \
  --start 2020-01-01 \
  --end 2026-04-01

# 可选：添加外部数据（VIX、利率等）
uv run python scripts/prepare_rl_data.py \
  --ticker <YOUR_TICKER> \
  --external-data data/vix.csv,data/rates.csv
```

### 2. 训练PPO模型

```bash
# 基础训练
uv run python scripts/train_ppo.py \
  --ticker <YOUR_TICKER> \
  --config configs/ppo_default.yaml

# 自定义风控参数
uv run python scripts/train_ppo.py \
  --ticker <YOUR_TICKER> \
  --config configs/ppo_custom.yaml \
  --stop-loss 0.10 \
  --max-position 0.80
```

### 3. 回测验证

```bash
# 回测
uv run python scripts/backtest_rl.py \
  --model models/<ticker>_ppo.zip \
  --start 2025-01-01 \
  --end 2026-04-01

# 极端行情测试
uv run python scripts/validate_extreme.py \
  --model models/<ticker>_ppo.zip \
  --events configs/extreme_events.yaml
```

### 4. 对比baseline

```bash
# 与买入持有对比
uv run python scripts/compare_baseline.py \
  --model models/<ticker>_ppo.zip \
  --baseline buy_and_hold \
  --start 2022-01-01
```

---

## Research阶段

### 任务定义

创建 `configs/task_config.yaml`:

```yaml
# 任务配置
task:
  name: "<TICKER>动态仓位管理"
  type: "single_asset_timing"  # 单标的择时
  ticker: "<YOUR_TICKER>"       # 例如: AAPL, SPY, BTC-USD
  
# 目标
objectives:
  primary: "maximize_risk_adjusted_return"  # 最大化风险调整收益
  constraints:
    - "max_drawdown < 30%"
    - "max_position < 95%"
    - "min_position > 0%"
    - "turnover_rate < 5x"  # 年换手率<5倍

# 风控（根据标的特性调整）
risk_controls:
  stop_loss: 0.15           # 15%止损
  take_profit: 0.20         # 20%止盈
  position_limit:
    normal: 0.95            # 正常最大仓位
    high_volatility: 0.5    # 高波动时最大仓位
    # 根据需要添加更多条件

# 外部数据（可选）
external_data:
  - name: "vix"
    file: "data/vix.csv"
    features: ["vix_zscore", "vix_percentile"]
  - name: "macro"
    file: "data/macro.csv"
    features: ["interest_rate", "inflation"]

# 训练配置
training:
  algorithm: "PPO"
  total_timesteps: 100000
  learning_rate: 3e-4
  batch_size: 64
```

---

## Generate阶段

### 1. 环境创建

使用 `scripts/create_simulator.py`:

```python
# 创建通用模拟器
from qlib.rl.simulator import SingleAssetSimulator

simulator = SingleAssetSimulator(
    instrument="<YOUR_TICKER>",
    data=data,
    config=SimulatorConfig(
        initial_cash=100000,
        transaction_cost=0.001,
        max_position=0.95,
        stop_loss=0.15,
        take_profit=0.20,
    ),
)
```

**关键组件**:

```python
# 1. 状态空间 - 可自定义
state_features = {
    # 基础市场特征（必选）
    "market": [
        "return_1d_zscore",      # 1日收益Z-score
        "return_5d_zscore",      # 5日收益Z-score
        "volatility_20d_rank",   # 波动率20日分位数
        "volume_zscore",         # 成交量Z-score
    ],
    
    # 技术指标（可选）
    "technical": [
        "bb_width",             # 布林带宽度
        "rsi_14",               # RSI指标
        "macd_signal",          # MACD信号
    ],
    
    # 外部数据（可选，根据标的添加）
    "external": [
        # 股票ETF: VIX、利率、情绪指标
        # 加密货币: Fear & Greed Index、链上数据
        # 期货: 基差、库存数据
        # 商品: 宏观数据、汇率
    ],
    
    # 组合状态（自动添加）
    "portfolio": [
        "position_ratio",        # 当前仓位比例
        "unrealized_pnl",       # 未实现盈亏
        "cash_ratio",           # 现金比例
    ],
}

# 2. 动作空间
action_space = {
    "type": "continuous",       # 连续动作
    "range": [0.0, 0.95],      # 仓位比例 [0%, 95%]
    "min_trade": 0.05,         # 最小交易阈值
}

# 3. 奖励函数 - 可自定义
reward_function = {
    "step_reward": "portfolio_return - transaction_cost",
    "risk_penalty": "drawdown^2 * lambda",
    "sharpe_bonus": "rolling_sharpe * weight",
    "final_bonus": "total_return * weight",
}
```

### 2. 状态归一化

```python
# scripts/normalize_state.py
class StateNormalizer:
    """状态归一化器 - 避免NaN/Inf"""
    
    def __init__(self, feature_stats: dict):
        self.mean = feature_stats["mean"]
        self.std = feature_stats["std"]
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """归一化状态，裁剪到[-3, 3]"""
        normalized = (state - self.mean) / (self.std + 1e-8)
        return np.clip(normalized, -3.0, 3.0)
```

### 3. 自定义风控

```python
# 根据标的特性添加风控规则
def custom_risk_controls(state, action):
    """自定义风控"""
    
    # 示例1: 高波动限制
    if state.volatility > threshold:
        action = limit_position(action, max=0.5)
    
    # 示例2: 回撤保护
    if state.drawdown > 0.10:
        action = prevent_buy(action)
    
    # 示例3: 外部条件（如VIX）
    if has_external_data("vix") and state.vix > 30:
        action = limit_position(action, max=0.5)
    
    return action
```

---

## Evaluate阶段

### 1. 训练PPO

```bash
# 使用默认配置
uv run python scripts/train_ppo.py --ticker <YOUR_TICKER>

# 自定义配置
uv run python scripts/train_ppo.py \
  --ticker <YOUR_TICKER> \
  --lr 3e-4 \
  --batch-size 64 \
  --timesteps 100000 \
  --clip-range 0.1
```

**训练监控**:
- 使用TensorBoard: `tensorboard --logdir logs/`
- 关键指标: `loss`, `policy_loss`, `entropy`, `approx_kl`, `sharpe`

### 2. 回测指标

```python
# 标准指标输出（通用）
metrics = {
    "return": {
        "total_return": ...,          # 总收益率
        "annual_return": ...,         # 年化收益率
        "excess_return": ...,         # 超额收益
    },
    "risk": {
        "max_drawdown": ...,          # 最大回撤
        "volatility": ...,            # 年化波动率
        "var_95": ...,                # 95% VaR
    },
    "risk_adjusted": {
        "sharpe": ...,                # Sharpe比率
        "sortino": ...,               # Sortino比率
        "calmar": ...,                # Calmar比率
    },
    "trading": {
        "num_trades": ...,            # 交易次数
        "win_rate": ...,              # 胜率
        "avg_hold_days": ...,         # 平均持仓天数
        "turnover_rate": ...,         # 年换手率
    },
}
```

---

## Validate阶段

### 极端行情测试

创建 `configs/extreme_events.yaml`（根据标的自定义）:

```yaml
extreme_events:
  - name: "市场崩盘"
    start: "2020-02-20"
    end: "2020-04-15"
    type: "crash"
    expected: "max_drawdown < 50%"
    
  - name: "熊市期间"
    start: "2022-01-01"
    end: "2022-12-31"
    type: "bear_market"
    expected: "outperform_benchmark > 10%"
    
  - name: "高波动期"
    start: "2025-04-01"
    end: "2025-04-30"
    type: "volatility_spike"
    expected: "recovery_days < 20"
```

运行测试:

```bash
uv run python scripts/validate_extreme.py \
  --model models/<ticker>_ppo.zip \
  --events configs/extreme_events.yaml \
  --report reports/extreme_validation.md
```

### 稳定性测试

```bash
# Rolling窗口测试
uv run python scripts/test_stability.py \
  --model models/<ticker>_ppo.zip \
  --window 60 \
  --metric sharpe
```

---

## Distill阶段

### 文档化

输出到 `docs/rl-strategies/`:

```
docs/rl-strategies/
├── <TICKER>-PPO-YYYY-MM-DD.md      # 策略文档
├── extreme_validation.md             # 极端行情验证
├── baseline_comparison.md            # baseline对比
└── lessons_learned.md                # 经验总结
```

### 经验库

更新到 `memory/rl_experience.md`:

```markdown
## 推荐配置

1. PPO + 状态归一化 + 成本感知动作
2. 学习率 3e-4, clip_range 0.1
3. batch_size 64, 训练100k steps

## 禁止配置

1. 不归一化状态 → NaN/Inf
2. clip_range > 0.2 → 策略崩溃
3. 不设止损 → 回撤失控

## 根据标的类型调整

### 杠杆ETF（如3x ETF）
- 必须添加外部数据（VIX/恐慌指标）
- 降低最大仓位到80%
- 严格止损（10%）
- 高波动时限制仓位到50%

### 蓝筹股（如AAPL、MSFT）
- 可用基础技术指标
- 最大仓位95%
- 适度止损（15%）

### 加密货币
- 添加链上数据、Fear & Greed Index
- 24/7交易，调整时间窗口
- 高波动保护（7日波动>30%限制仓位）
```

---

## 关键门槛

### Evaluate门槛

| 指标 | 门槛 | 说明 |
|------|------|------|
| Sharpe | > 0.5 | 基础有效性 |
| Max DD | < 30% | 风控底线 |
| 训练稳定 | 无NaN | 数值稳定 |

### Validate门槛

| 测试项 | 门槛 | 说明 |
|--------|------|------|
| 极端行情存活 | Max DD < 50% | 不爆仓 |
| 熊市outperform | > 0% | 跑赢benchmark |
| 稳定性 | IC_recent/full > 0.5 | 无严重退化 |

---

## 脚本说明

### 可复用脚本 (`scripts/`)

| 脚本 | 功能 | 输出 |
|------|------|------|
| `prepare_rl_data.py` | 准备数据 | `data/rl/{ticker}.pkl` |
| `create_simulator.py` | 创建环境 | Simulator对象 |
| `train_ppo.py` | 训练PPO | `models/{ticker}_ppo.zip` |
| `backtest_rl.py` | 回测 | 回测报告 |
| `validate_extreme.py` | 极端行情测试 | 验证报告 |
| `compare_baseline.py` | 对比baseline | 对比报告 |
| `test_stability.py` | 稳定性测试 | 稳定性报告 |
| `normalize_state.py` | 状态归一化工具 | 归一化器 |

### 临时脚本

一次性研究脚本放在 `./scripts/temp/`，完成后删除。

---

## 与其他Skills的关系

| Skill | 关系 |
|-------|------|
| `$qlib-single-factor-mining` | SFA提供因子 → RL使用因子作为状态 |
| `$qlib-multi-factor-backtest` | MFA提供baseline → RL对比性能 |
| `$factor-research-review` | Research阶段 → 提取因子构造思路 |

---

## 常见问题

### Q1: 训练出现NaN怎么办？

```python
# 检查清单：
1. 状态归一化：使用 StateNormalizer
2. 奖励裁剪：clip到[-1, 1]
3. 梯度裁剪：max_grad_norm = 0.1
4. 学习率：降低到1e-4
5. 检查数据：是否有NaN/Inf
```

### Q2: 策略交易太频繁？

```python
# 解决方案：
1. 增加最小交易阈值：min_trade = 0.05
2. 增加交易成本：transaction_cost = 0.002
3. 添加换手惩罚：reward -= turnover * 0.01
```

### Q3: 极端行情回撤太大？

```python
# 解决方案：
1. 添加风控层：check_risk_limits()
2. 根据标的添加条件限制：
   - 高波动时限制仓位
   - 大回撤时禁止加仓
   - 添加外部数据条件
3. 强制止损：if unrealized_loss > threshold, force_sell()
```

### Q4: 如何针对不同标的优化？

```python
# 通用建议：
1. 调整特征：根据标的特性选择相关特征
2. 调整风控：根据波动率调整止损/仓位限制
3. 调整成本：根据流动性调整transaction_cost
4. 添加外部数据：如股票加VIX，加密货币加链上数据
```

---

## 参考资料

- `references/qlib_rl_official.md` - Qlib官方RL文档
- `references/ppo_tuning.md` - PPO调参指南
- `references/extreme_events.md` - 极端行情处理
- `references/reward_design.md` - 奖励函数设计

---

## 版本历史

- v1.0 (2026-04-04) - 初始版本，支持SOXL择时
