# SOXL 三倍ETF强化学习交易系统 - 技术规范

## 1. 系统架构

### 1.1 整体架构图
```
┌─────────────────────────────────────────────────────────────┐
│                     数据层 (Data Layer)                      │
├─────────────────────────────────────────────────────────────┤
│  Yahoo Finance API  →  分钟级OHLCV数据  →  本地存储 (HDF5)   │
│                           ↓                                  │
│                    特征工程 Pipeline                         │
│              (技术指标 + 微观结构 + 波动率)                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    环境层 (Environment Layer)                │
├─────────────────────────────────────────────────────────────┤
│  TradingEnv (Gymnasium)                                      │
│  ├── State: 特征向量 + 仓位信息 + 账户状态                   │
│  ├── Action: 连续仓位权重 [0, 1] 或 离散 {买/持/卖}         │
│  └── Reward: Sharpe-adjusted return - transaction cost      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     模型层 (Model Layer)                     │
├─────────────────────────────────────────────────────────────┤
│  策略网络 (Actor)          价值网络 (Critic)                 │
│  ├── LSTM/Transformer      ├── LSTM/Transformer             │
│  ├── 全连接层              ├── 全连接层                      │
│  └── Action输出            └── Value输出                     │
│                                                              │
│  RL算法: PPO / SAC / TD3 / Ensemble                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    执行层 (Execution Layer)                  │
├─────────────────────────────────────────────────────────────┤
│  信号生成  →  仓位管理  →  风控模块  →  订单执行             │
└─────────────────────────────────────────────────────────────┘
```

## 2. 数据规范

### 2.1 数据源
- **主数据源**: Yahoo Finance (yfinance)
- **备选**: Tiingo, Alpha Vantage
- **频率**: 1分钟 OHLCV
- **资产**: SOXL, SOXS, TQQQ, SQQQ, SPXL, SPXS

### 2.2 特征工程

#### 2.2.1 价格特征
```python
# 多周期收益率
returns = {
    'ret_1m': close.pct_change(1),
    'ret_5m': close.pct_change(5),
    'ret_15m': close.pct_change(15),
    'ret_60m': close.pct_change(60),
}

# 对数收益率
log_ret = np.log(close / close.shift(1))
```

#### 2.2.2 技术指标
```python
# RSI (相对强弱指数)
RSI(period=14)

# MACD (移动平均收敛发散)
MACD(fast=12, slow=26, signal=9)

# Bollinger Bands
BBands(period=20, std=2)

# ATR (平均真实波幅)
ATR(period=14)

# 成交量变化率
volume_rate = volume.pct_change()

# VWAP (成交量加权平均价)
vwap = (high + low + close) / 3 * volume
```

#### 2.2.3 波动率特征
```python
# 已实现波动率 (Realized Volatility)
realized_vol = returns.rolling(60).std() * np.sqrt(390)  # 年化

# Parkinson 波动率
parkinson_vol = np.sqrt(
    (1 / (4 * np.log(2))) * (np.log(high / low) ** 2)
).rolling(60).mean() * np.sqrt(390)

# GARCH 预测波动率 (可选)
```

#### 2.2.4 微观结构特征
```python
# 订单流不平衡 (需Level 2数据，可用代理)
flow_imbalance = (close - open) / (high - low)

# 成交量加权价格偏离
vwap_deviation = (close - vwap) / vwap

# 日内模式
hour_of_day = index.hour
minute_of_hour = index.minute
```

### 2.3 数据预处理
```python
# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 或使用滚动标准化
features_rolling_scaled = (features - features.rolling(252).mean()) / features.rolling(252).std()

# 缺失值处理
features = features.fillna(method='ffill').fillna(0)
```

## 3. 环境设计

### 3.1 状态空间 (State Space)
```python
observation_space = Box(
    low=-np.inf, 
    high=np.inf, 
    shape=(window_size, num_features + 2),  # +2 for position and account
    dtype=np.float32
)

# 状态组成:
# - 价格特征 (window_size, num_price_features)
# - 技术指标 (window_size, num_indicators)
# - 当前仓位 (1,)
# - 账户余额 (1,)
```

### 3.2 动作空间 (Action Space)

#### 方案A: 连续动作 (推荐)
```python
action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
# 0 = 空仓, 1 = 满仓, 0.5 = 半仓
```

#### 方案B: 离散动作
```python
action_space = Discrete(3)
# 0 = 卖出, 1 = 持有, 2 = 买入
```

### 3.3 奖励函数 (Reward Function)

#### 基础奖励
```python
def compute_reward(state, action, next_state):
    # 收益率
    portfolio_return = (next_state.portfolio_value - state.portfolio_value) / state.portfolio_value
    
    # 交易成本
    transaction_cost = abs(action - state.position) * cost_rate
    
    # 基础奖励
    reward = portfolio_return - transaction_cost
    
    return reward
```

#### Sharpe-adjusted 奖励 (推荐)
```python
def sharpe_reward(returns, window=60):
    mean_ret = returns.rolling(window).mean()
    std_ret = returns.rolling(window).std()
    sharpe = mean_ret / (std_ret + 1e-8)
    return sharpe
```

#### 风险惩罚奖励
```python
def risk_adjusted_reward(reward, drawdown, volatility):
    # 回撤惩罚
    drawdown_penalty = lambda_dd * max(0, drawdown)
    
    # 波动率惩罚
    volatility_penalty = lambda_vol * volatility
    
    # 总奖励
    adjusted_reward = reward - drawdown_penalty - volatility_penalty
    
    return adjusted_reward
```

### 3.4 环境实现框架
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ETFTradingEnv(gym.Env):
    def __init__(self, data, config):
        super().__init__()
        
        # 数据
        self.data = data
        self.window_size = config.get('window_size', 60)
        
        # 交易参数
        self.initial_balance = config.get('initial_balance', 100000)
        self.transaction_cost = config.get('transaction_cost', 0.001)  # 0.1%
        self.position_limit = config.get('position_limit', 1.0)
        
        # 空间定义
        num_features = data.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, num_features + 2),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
        # 初始化
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        # 执行动作
        old_position = self.position
        self.position = np.clip(action[0], 0, self.position_limit)
        
        # 计算交易成本
        trade_size = abs(self.position - old_position)
        transaction_cost = trade_size * self.portfolio_value * self.transaction_cost
        
        # 获取下一时刻的价格变化
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        price_return = (next_price - current_price) / current_price
        
        # 更新组合价值
        portfolio_return = self.position * price_return
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value -= transaction_cost
        
        # 计算奖励
        reward = self._compute_reward(portfolio_return, transaction_cost)
        
        # 更新步数
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # 记录历史
        self.history.append({
            'step': self.current_step,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'reward': reward
        })
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        # 获取窗口数据
        window_data = self.data.iloc[
            self.current_step - self.window_size:self.current_step
        ].values
        
        # 添加仓位和账户信息
        position_info = np.full((self.window_size, 1), self.position)
        account_info = np.full(
            (self.window_size, 1), 
            self.portfolio_value / self.initial_balance
        )
        
        observation = np.hstack([window_data, position_info, account_info])
        return observation.astype(np.float32)
    
    def _compute_reward(self, portfolio_return, transaction_cost):
        # Sharpe-adjusted reward
        if len(self.history) > 60:
            returns = [h['reward'] for h in self.history[-60:]]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
            return sharpe - transaction_cost * 10  # 放大交易成本惩罚
        else:
            return portfolio_return - transaction_cost * 10
```

## 4. 模型架构

### 4.1 网络结构

#### LSTM-based Policy Network
```python
import torch
import torch.nn as nn

class LSTMPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers, batch_first=True
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]  # 取最后时刻
        
        action = self.actor(features)
        value = self.critic(features)
        
        return action, value
```

#### Transformer-based Network (可选)
```python
class TransformerPolicyNetwork(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.actor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq, batch, feat)
        x = self.transformer(x)
        features = x[-1]  # 最后时刻
        
        action = self.actor(features)
        value = self.critic(features)
        
        return action, value
```

### 4.2 RL算法选择

#### PPO (Proximal Policy Optimization) - 推荐
- **优点**: 稳定性好，超参不敏感
- **缺点**: 样本效率较低
- **适用**: 初期开发，稳定训练

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy", 
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)
model.learn(total_timesteps=100000)
```

#### SAC (Soft Actor-Critic)
- **优点**: 样本效率高，自动探索
- **缺点**: 需要调温度参数
- **适用**: 连续动作，复杂环境

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    verbose=1
)
model.learn(total_timesteps=100000)
```

### 4.3 训练策略

#### 课程学习 (Curriculum Learning)
```python
# 阶段1: 简单环境（低波动期）
env_phase1 = ETFTradingEnv(data=bull_market_data, config=easy_config)
model.learn(total_timesteps=50000)

# 阶段2: 中等难度（混合市场）
env_phase2 = ETFTradingEnv(data=mixed_data, config=medium_config)
model.learn(total_timesteps=50000)

# 阶段3: 完整环境（包含极端行情）
env_phase3 = ETFTradingEnv(data=full_data, config=hard_config)
model.learn(total_timesteps=100000)
```

#### 模仿学习初始化
```python
# 用历史最优策略预训练
from imitation.algorithms import BC

expert_data = load_expert_demonstrations()
bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=expert_data
)
bc_trainer.train(n_epochs=10)

# 然后用RL微调
model = PPO("MlpPolicy", env)
model.policy = bc_trainer.policy
model.learn(total_timesteps=100000)
```

## 5. 回测与验证

### 5.1 Walk-Forward Validation
```python
def walk_forward_validation(data, model_class, config, n_splits=5):
    """
    Walk-forward validation 避免未来数据泄露
    """
    total_length = len(data)
    train_size = total_length // (n_splits + 1)
    test_size = train_size // 2
    
    results = []
    
    for i in range(n_splits):
        # 训练集
        train_start = i * train_size
        train_end = train_start + train_size
        train_data = data.iloc[train_start:train_end]
        
        # 测试集 (紧接训练集)
        test_start = train_end
        test_end = test_start + test_size
        test_data = data.iloc[test_start:test_end]
        
        # 训练
        env = ETFTradingEnv(train_data, config)
        model = model_class("MlpPolicy", env)
        model.learn(total_timesteps=50000)
        
        # 测试
        test_env = ETFTradingEnv(test_data, config)
        obs = test_env.reset()
        total_reward = 0
        
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = test_env.step(action)
            total_reward += reward
            if done:
                break
        
        results.append({
            'fold': i,
            'train_period': (train_start, train_end),
            'test_period': (test_start, test_end),
            'total_reward': total_reward,
            'final_value': test_env.portfolio_value
        })
    
    return results
```

### 5.2 评估指标
```python
def evaluate_performance(history, benchmark_returns):
    """
    计算完整评估指标
    """
    portfolio_values = [h['portfolio_value'] for h in history]
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    metrics = {
        # 收益指标
        'total_return': (portfolio_values[-1] / portfolio_values[0]) - 1,
        'annualized_return': (1 + returns.mean()) ** 252 - 1,
        
        # 风险指标
        'volatility': returns.std() * np.sqrt(252),
        'max_drawdown': calculate_max_drawdown(portfolio_values),
        'var_95': np.percentile(returns, 5),
        'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
        
        # 风险调整收益
        'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252),
        'sortino_ratio': calculate_sortino(returns),
        'calmar_ratio': calculate_calmar(returns),
        
        # 交易统计
        'num_trades': calculate_num_trades(history),
        'win_rate': calculate_win_rate(history),
        'profit_factor': calculate_profit_factor(history),
        
        # 相对基准
        'alpha': calculate_alpha(returns, benchmark_returns),
        'beta': calculate_beta(returns, benchmark_returns),
        'information_ratio': calculate_information_ratio(returns, benchmark_returns)
    }
    
    return metrics
```

### 5.3 压力测试
```python
def stress_test(model, crisis_periods):
    """
    在极端市场条件下测试模型
    """
    results = {}
    
    for period_name, period_data in crisis_periods.items():
        env = ETFTradingEnv(period_data, config)
        obs = env.reset()
        
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            if done:
                break
        
        results[period_name] = {
            'final_value': env.portfolio_value,
            'max_drawdown': calculate_max_drawdown([h['portfolio_value'] for h in env.history]),
            'total_return': env.portfolio_value / config['initial_balance'] - 1
        }
    
    return results
```

## 6. 部署架构

### 6.1 模型持久化
```python
# 保存模型
model.save("models/soxl_ppo_v1.zip")

# 保存配置
import yaml
config = {
    'model_version': 'v1.0',
    'training_date': '2025-03-22',
    'features': ['rsi', 'macd', 'bb', 'atr', 'volume_rate'],
    'hyperparameters': {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64
    }
}
with open('models/config.yaml', 'w') as f:
    yaml.dump(config, f)
```

### 6.2 实时推理
```python
class TradingAgent:
    def __init__(self, model_path, config_path):
        self.model = PPO.load(model_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.feature_pipeline = FeaturePipeline(self.config['features'])
    
    def predict(self, market_data):
        """
        实时预测
        """
        # 特征工程
        features = self.feature_pipeline.transform(market_data)
        
        # 模型推理
        action, _ = self.model.predict(features, deterministic=True)
        
        # 风控检查
        action = self._apply_risk_controls(action)
        
        return action
    
    def _apply_risk_controls(self, action):
        """
        应用风控规则
        """
        # 最大仓位限制
        action = np.clip(action, 0, self.config['max_position'])
        
        # 单日交易次数限制
        # ... 
        
        return action
```

### 6.3 监控与告警
```python
def monitor_performance(agent, market_data, threshold):
    """
    实时监控模型表现
    """
    metrics = calculate_recent_metrics(agent.history)
    
    alerts = []
    
    # 回撤告警
    if metrics['current_drawdown'] > threshold['max_drawdown']:
        alerts.append(f"⚠️ 回撤超限: {metrics['current_drawdown']:.2%}")
    
    # Sharpe下降告警
    if metrics['rolling_sharpe'] < threshold['min_sharpe']:
        alerts.append(f"⚠️ Sharpe下降: {metrics['rolling_sharpe']:.2f}")
    
    # 模型漂移检测
    if detect_model_drift(agent.recent_predictions, agent.historical_predictions):
        alerts.append("⚠️ 检测到模型漂移，建议重新训练")
    
    return alerts
```

## 7. 风险管理

### 7.1 仓位管理
```python
def position_sizing(signal, volatility, account_value, risk_per_trade=0.02):
    """
    基于波动率的仓位管理
    """
    # Kelly Criterion (简化版)
    kelly_fraction = signal / (volatility ** 2)
    kelly_fraction = np.clip(kelly_fraction, 0, 1)
    
    # 限制单笔风险
    max_position = (account_value * risk_per_trade) / volatility
    
    position = min(kelly_fraction, max_position)
    return position
```

### 7.2 止损止盈
```python
class RiskManager:
    def __init__(self, stop_loss=0.05, take_profit=0.10, trailing_stop=0.03):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.highest_price = None
    
    def check_exit(self, current_price, entry_price):
        """
        检查是否触发止损/止盈
        """
        pnl = (current_price - entry_price) / entry_price
        
        # 更新最高价
        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price
        
        # 止损
        if pnl <= -self.stop_loss:
            return 'stop_loss'
        
        # 止盈
        if pnl >= self.take_profit:
            return 'take_profit'
        
        # 移动止损
        trailing_pnl = (current_price - self.highest_price) / self.highest_price
        if trailing_pnl <= -self.trailing_stop and pnl > 0:
            return 'trailing_stop'
        
        return None
```

## 8. 开发计划

### Phase 1: 数据与特征 (Week 1)
- [x] 项目结构搭建
- [ ] 分钟级数据下载器
- [ ] 特征工程 Pipeline
- [ ] 数据存储与加载

### Phase 2: 环境搭建 (Week 1-2)
- [ ] Gym 交易环境
- [ ] 奖励函数设计
- [ ] 环境验证测试

### Phase 3: 模型训练 (Week 2-3)
- [ ] PPO 基线模型
- [ ] 超参调优
- [ ] Ensemble 方法

### Phase 4: 验证与部署 (Week 3-4)
- [ ] Walk-forward 验证
- [ ] 压力测试
- [ ] 模型持久化
- [ ] 上线监控

## 9. 参考文献

1. **QTNet** (arxiv 2312.15730) - 分钟频率自适应交易模型
2. **Multi-time DRL Framework** (2024) - 多时间尺度特征融合
3. **Qlib Documentation** - Microsoft Qlib 官方文档
4. **Stable-Baselines3** - RL算法实现库
5. **FinRL** - 金融强化学习开源库
