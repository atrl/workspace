# RL 迁移方案设计

## 当前问题

硬参数策略：
```python
panic_threshold = -35.2
panic_position = 0.98
trend_position = 0.4382
vix_panic = 20.4
vix_extreme = 28.5
vix_greed = 12.7
stop_loss = -0.08
```

问题：
- 参数固定，无法适应市场变化
- 没有学习能力
- 过度依赖历史优化

---

## 方案对比

| 方案 | RL学习内容 | 优点 | 缺点 | 推荐度 |
|------|----------|------|------|--------|
| **方案1: 分层RL** | 仓位大小 (0-100%) | 简单、可控、可解释 | 需要规则信号 | ⭐⭐⭐⭐⭐ |
| **方案2: 参数RL** | 7个参数的值 | 完全自适应 | 动作空间大、难训练 | ⭐⭐ |
| **方案3: 直接RL** | 买入/卖出/持有 | 完全RL | 之前失败过 | ⭐ |
| **方案4: 调整RL** | 参数调整幅度 | 保留基础逻辑 | 复杂度高 | ⭐⭐⭐ |

---

## 推荐方案：分层RL

### 架构设计

```
┌─────────────────────────────────────────┐
│          市场状态 (State)                │
│  - VIX 指数                              │
│  - 收益率                                │
│  - RSI                                   │
│  - 恐慌得分                              │
│  - 当前仓位                              │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
   ┌───▼────┐      ┌───▼────┐
   │规则策略│      │  RL智能体│
   │(信号)  │      │ (仓位)  │
   └───┬────┘      └───┬────┘
       │                │
   买入/卖出         仓位 0-100%
   持有信号
       │                │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │   最终仓位      │
       │ = 信号 × 仓位%  │
       └────────────────┘
```

### 实现

#### 1. 环境设计

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SOXLTradingEnv(gym.Env):
    """
    SOXL 分层RL交易环境

    状态：市场状态
    动作：仓位大小 [0, 1]
    奖励：夏普比率 + 收益
    """

    def __init__(self, df):
        super().__init__()

        self.df = df
        self.current_step = 60  # 从第60天开始

        # 状态空间：11维
        # [vix, vix_zscore, daily_ret, rsi, panic_score,
        #  current_pos, portfolio_return, volatility, ma_ratio, volume_ratio, cash_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

        # 动作空间：仓位大小 [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # 账户状态
        self.cash = 100000
        self.shares = 0
        self.history = []

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 60
        self.cash = 100000
        self.shares = 0
        self.history = [100000]
        return self._get_observation(), {}

    def _get_observation(self):
        """获取当前市场状态"""
        row = self.df.iloc[self.current_step]

        # 计算当前组合状态
        pos_val = self.shares * row['close']
        total = self.cash + pos_val
        pos = pos_val / total if total > 0 else 0

        # 计算历史收益率
        if len(self.history) > 1:
            returns = np.diff(self.history[-20:]) / np.array(self.history[-21:-1])
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            portfolio_return = (total - 100000) / 100000
        else:
            volatility = 0
            portfolio_return = 0

        # 市场特征
        obs = np.array([
            row['vix'] / 50,                    # 归一化 VIX
            row['vix_zscore'],                   # VIX z-score
            row['daily_ret'],                    # 日收益率
            row['RSI'] / 100,                    # 归一化 RSI
            row['PanicScore'] / 100,            # 归一化恐慌得分
            pos,                                 # 当前仓位
            portfolio_return,                    # 组合收益
            volatility,                          # 波动率
            row['MA5'] / row['MA20'],           # 均线比率
            row['volume'] / row['volume_ma20'], # 成交量比率
            self.cash / total,                  # 现金比例
        ], dtype=np.float32)

        # 处理 NaN
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs

    def step(self, action):
        """执行动作（仓位调整）"""

        # 1. 获取规则策略信号
        rule_signal = self._get_rule_signal()

        # 2. RL动作：仓位大小
        rl_position = float(action[0])

        # 3. 最终仓位 = 规则信号 × RL仓位
        target_position = rule_signal * rl_position

        # 4. 执行交易
        row = self.df.iloc[self.current_step]
        pos_val = self.shares * row['close']
        total = self.cash + pos_val
        current_pos = pos_val / total if total > 0 else 0

        # 调整仓位
        if abs(target_position - current_pos) > 0.05:
            delta = (total * target_position - pos_val) / row['close']

            if delta > 10:  # 买入
                cost = delta * row['close'] * 1.001
                if cost <= self.cash:
                    self.shares += delta
                    self.cash -= cost
            elif delta < -10:  # 卖出
                sell = min(-delta, self.shares)
                self.cash += sell * row['close'] * 0.999
                self.shares -= sell

        # 5. 更新状态
        self.current_step += 1
        new_total = self.cash + self.shares * self.df.iloc[self.current_step]['close']
        self.history.append(new_total)

        # 6. 计算奖励
        reward = self._calculate_reward()

        # 7. 检查是否结束
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done, False, {}

    def _get_rule_signal(self):
        """
        规则策略信号（简化版）

        返回：
        - 1.0: 买入信号
        - 0.5: 持有信号
        - 0.0: 卖出信号
        """
        row = self.df.iloc[self.current_step]

        # 恐慌信号
        if row['PanicScore'] <= -35.2:
            if row['vix'] > 28.5:
                return 1.0  # 极端恐慌
            elif row['vix'] > 20.4:
                return 0.9  # 恐慌

        # 趋势信号
        if row['RSI'] < 65 and row['MA5'] > row['MA20']:
            return 0.6  # 上升趋势

        # 贪婪信号
        if row['vix'] < 12.7:
            if row['RSI'] > 65:
                return 0.2  # 超买
            return 0.4

        # 默认
        return 0.5

    def _calculate_reward(self):
        """
        计算奖励

        组合奖励 = 收益 + 风险惩罚 + 交易成本
        """
        if len(self.history) < 2:
            return 0

        # 1. 收益奖励
        daily_return = (self.history[-1] - self.history[-2]) / self.history[-2]
        return_reward = daily_return * 100  # 放大

        # 2. 风险惩罚
        if len(self.history) > 20:
            returns = np.diff(self.history[-20:]) / np.array(self.history[-21:-1])
            volatility = np.std(returns)
            risk_penalty = -volatility * 10
        else:
            risk_penalty = 0

        # 3. 回撤惩罚
        peak = max(self.history)
        drawdown = (peak - self.history[-1]) / peak
        dd_penalty = -drawdown * 50 if drawdown > 0.1 else 0

        # 总奖励
        reward = return_reward + risk_penalty + dd_penalty

        return reward


# ============================================================================
# PPO 训练
# ============================================================================

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

def train_rl_position_agent():
    """训练仓位RL智能体"""

    # 1. 准备数据
    df = pd.read_csv('/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv', parse_dates=['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)

    # 计算特征
    df['daily_ret'] = df['close'].pct_change()
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # VIX 特征
    df['vix_ma20'] = df['vix'].rolling(20).mean()
    df['vix_std20'] = df['vix'].rolling(20).std()
    df['vix_zscore'] = (df['vix'] - df['vix_ma20']) / (df['vix_std20'] + 1e-8)

    # 恐慌得分
    df['PanicScore'] = (
        df['vix_zscore'] * 40 +
        df['vix'].rolling(20).rank(pct=True) * 60 +
        np.where(df['daily_ret'] < 0, np.minimum(df['daily_ret'] / -0.05, 1) * -50, 0)
    )

    df = df.dropna()

    # 2. 创建环境
    env = DummyVecEnv([lambda: SOXLTradingEnv(df)])

    # 3. 创建 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/rl_position/"
    )

    # 4. 训练
    print("="*70)
    print("训练分层RL仓位智能体")
    print("="*70)
    print(f"数据: {len(df)} records")
    print(f"状态空间: 11维")
    print(f"动作空间: 仓位 [0, 1]")
    print()

    model.learn(total_timesteps=100000)

    # 5. 保存模型
    model.save("models/soxl_position_rl")
    print("\n✅ 模型已保存: models/soxl_position_rl.zip")

    return model


# ============================================================================
# 回测 RL 策略
# ============================================================================

def backtest_rl_strategy(model, df):
    """回测RL策略"""

    env = SOXLTradingEnv(df)
    obs, _ = env.reset()

    done = False
    portfolio_values = [100000]

    while not done:
        # RL 预测仓位
        action, _ = model.predict(obs, deterministic=True)

        # 执行
        obs, reward, done, _, _ = env.step(action)

        # 记录
        total = env.cash + env.shares * df.iloc[env.current_step]['close']
        portfolio_values.append(total)

    # 计算指标
    pv = np.array(portfolio_values)
    total_return = (pv[-1] - 100000) / 100000

    daily_returns = np.diff(pv) / pv[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    max_dd = np.max((np.maximum.accumulate(pv) - pv) / np.maximum.accumulate(pv))

    print("\n" + "="*70)
    print("RL 策略回测结果")
    print("="*70)
    print(f"总收益:        {total_return:.2%}")
    print(f"Sharpe:        {sharpe:.2f}")
    print(f"最大回撤:      {max_dd:.2%}")
    print(f"收益倍数:      {pv[-1] / 100000:.2f}x")


if __name__ == '__main__':
    # 训练
    model = train_rl_position_agent()

    # 回测
    df = pd.read_csv('/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv', parse_dates=['date'])
    # ... 特征计算 ...
    backtest_rl_strategy(model, df)
```

---

## 关键设计

### 1. 分层架构

```
规则策略 (信号)  ×  RL智能体 (仓位)  =  最终仓位
     1.0         ×      0.85        =     0.85
     0.5         ×      0.90        =     0.45
     0.0         ×      0.80        =     0.00
```

### 2. 状态空间（11维）

| 维度 | 说明 | 归一化 |
|------|------|--------|
| vix | VIX指数 | / 50 |
| vix_zscore | VIX z-score | 原值 |
| daily_ret | 日收益率 | 原值 |
| RSI | 相对强弱指标 | / 100 |
| PanicScore | 恐慌得分 | / 100 |
| current_pos | 当前仓位 | [0, 1] |
| portfolio_return | 组合收益 | 原值 |
| volatility | 波动率 | 原值 |
| ma_ratio | 均线比率 | MA5/MA20 |
| volume_ratio | 成交量比率 | vol/vol_ma20 |
| cash_ratio | 现金比例 | cash/total |

### 3. 动作空间

```python
# 连续动作：仓位大小 [0, 1]
action_space = spaces.Box(low=0.0, high=1.0, shape=(1,))
```

### 4. 奖励函数

```python
reward = 收益奖励 + 风险惩罚 + 回撤惩罚

# 收益奖励
return_reward = daily_return * 100

# 风险惩罚
risk_penalty = -volatility * 10

# 回撤惩罚（回撤 > 10% 时）
dd_penalty = -drawdown * 50
```

---

## 优势

1. **可控性**: 规则策略提供基础信号，RL只调整仓位
2. **可解释**: 可以看到RL如何调整仓位
3. **稳定性**: 不会完全失控（有规则策略兜底）
4. **学习能力**: RL可以学习市场状态与最优仓位的关系

---

## 下一步

1. 实现环境 `SOXLTradingEnv`
2. 训练 PPO 模型
3. 回测对比（硬参数 vs RL）
4. 分析RL学到了什么
