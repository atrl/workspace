# SOXL 强化学习交易系统 - 完整技术指导方案

## 🎯 项目目标
构建可上线的 Qlib 强化学习交易系统，针对 SOXL 等三倍杠杆 ETF 进行分钟级交易

**目标指标**:
- Sharpe Ratio > 1.5
- Max Drawdown < 20%
- Annualized Return > 15%
- Win Rate > 55%

---

## 📦 第一步：环境准备

### 1.1 创建 Worktree

```bash
# 进入主项目
cd ~/workspace/soxl-rl-qlib

# 创建数据重建 worktree
git worktree add ../soxl-rl-data -b feature/data-rebuild
cd ../soxl-rl-data

# 创建必要的目录
mkdir -p dataset/{raw,features,processed}
mkdir -p logs
mkdir -p notebooks

# 创建训练实验 worktree
cd ~/workspace/soxl-rl-qlib
git worktree add ../soxl-rl-training -b feature/training-experiments
cd ../soxl-rl-training

# 创建必要的目录
mkdir -p experiments
mkdir -p checkpoints
mkdir -p logs/tensorboard
mkdir -p results
```

### 1.2 虚拟环境设置

```bash
# 创建虚拟环境
cd ~/workspace/soxl-rl-qlib
python3.10 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 验证安装
python -c "import stable_baselines3; import gymnasium; import pandas; print('✅ All dependencies installed')"
```

### 1.3 ta-lib 安装（如果遇到问题）

```bash
# macOS
brew install ta-lib
pip install ta-lib

# 如果失败，尝试：
pip install --no-cache-dir ta-lib

# Ubuntu/Debian
sudo apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

---

## 📊 第二步：数据重建

### 2.1 获取历史数据

**方案 A: Yahoo Finance（推荐用于测试）**
```bash
cd ~/workspace/soxl-rl-data

# 下载 30 天分钟数据
python -m data.downloader \
    --symbols SOXL SOXS TQQQ SQQQ \
    --interval 1m \
    --lookback-days 30 \
    --output-dir ./dataset/raw

# 下载 60 天 5 分钟数据（更多历史）
python -m data.downloader \
    --symbols SOXL SOXS TQQQ SQQQ \
    --interval 5m \
    --lookback-days 60 \
    --output-dir ./dataset/raw
```

**方案 B: 多源数据融合（推荐用于生产）**
```python
# scripts/download_extended_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_extended_history(symbol, start_date, end_date, interval='1m'):
    """
    下载扩展历史数据
    通过多次请求绕过 Yahoo Finance 限制
    """
    all_data = []
    current_start = start_date
    
    while current_start < end_date:
        # Yahoo Finance 限制：1m 数据最多 7 天
        current_end = min(current_start + timedelta(days=7), end_date)
        
        print(f"Downloading {symbol} from {current_start} to {current_end}")
        data = yf.download(
            symbol,
            start=current_start,
            end=current_end,
            interval=interval,
            progress=False
        )
        
        if not data.empty:
            all_data.append(data)
        
        current_start = current_end
    
    if all_data:
        return pd.concat(all_data).drop_duplicates()
    return pd.DataFrame()

# 使用示例
if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6个月
    
    for symbol in ['SOXL', 'SOXS', 'TQQQ', 'SQQQ']:
        data = download_extended_history(symbol, start_date, end_date)
        data.to_csv(f'./dataset/raw/{symbol}_1m_extended.csv')
        print(f"✅ {symbol}: {len(data)} rows")
```

**方案 C: 第三方数据源**
```python
# 使用 Tiingo, Alpha Vantage 或付费数据源
# 推荐数据源：
# - Tiingo (免费额度有限)
# - Alpha Vantage (免费但有限制)
# - Polygon.io (付费，高质量)
# - QuantConnect (免费研究数据)
```

### 2.2 数据质量检查

```python
# notebooks/data_quality_check.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_data_quality(filepath):
    """数据质量检查"""
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    
    print("=" * 60)
    print(f"Data Quality Report: {filepath}")
    print("=" * 60)
    
    # 基本信息
    print(f"\n📊 Basic Info:")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Time span: {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    # 缺失值
    print(f"\n❌ Missing Values:")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")
    
    # 异常值检测
    print(f"\n⚠️  Outliers:")
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            outliers = ((df[col] < q1) | (df[col] > q99)).sum()
            print(f"  {col}: {outliers} potential outliers")
    
    # 连续性检查
    print(f"\n📅 Continuity:")
    df_sorted = df.sort_values('datetime')
    time_diffs = df_sorted['datetime'].diff()
    
    expected_interval = pd.Timedelta(minutes=1)  # 1分钟数据
    gaps = (time_diffs > expected_interval * 2).sum()
    print(f"  Gaps detected: {gaps}")
    
    if gaps > 0:
        print("  ⚠️  Data has gaps, consider interpolation")
    
    # 价格一致性
    print(f"\n💰 Price Consistency:")
    invalid_prices = (df['high'] < df['low']).sum()
    if invalid_prices > 0:
        print(f"  ❌ Invalid high<low: {invalid_prices} rows")
    else:
        print(f"  ✅ All high >= low")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 价格走势
    axes[0, 0].plot(df['datetime'], df['close'])
    axes[0, 0].set_title('Price Over Time')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Close Price')
    
    # 成交量分布
    axes[0, 1].hist(df['volume'], bins=50, edgecolor='black')
    axes[0, 1].set_title('Volume Distribution')
    axes[0, 1].set_xlabel('Volume')
    axes[0, 1].set_ylabel('Frequency')
    
    # 收益率分布
    returns = df['close'].pct_change().dropna()
    axes[1, 0].hist(returns, bins=50, edgecolor='black')
    axes[1, 0].set_title('Returns Distribution')
    axes[1, 0].set_xlabel('Return')
    axes[1, 0].set_ylabel('Frequency')
    
    # 缺失值热图
    axes[1, 1].imshow(df.isna().values.T, aspect='auto', cmap='viridis')
    axes[1, 1].set_title('Missing Values Heatmap')
    axes[1, 1].set_xlabel('Row Index')
    axes[1, 1].set_ylabel('Column')
    
    plt.tight_layout()
    plt.savefig('./dataset/quality_check.png', dpi=150)
    print(f"\n📈 Visualization saved to ./dataset/quality_check.png")
    
    return df

# 运行检查
df = check_data_quality('./dataset/raw/SOXL_1m.csv')
```

### 2.3 特征工程完整实现

```python
# data/advanced_features.py
import pandas as pd
import numpy as np
from typing import List, Optional

class AdvancedFeatureEngineer:
    """高级特征工程"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df = self.df.sort_values('datetime').reset_index(drop=True)
    
    def add_price_features(self) -> 'AdvancedFeatureEngineer':
        """价格特征"""
        # 多周期收益率
        for period in [1, 5, 15, 30, 60]:
            self.df[f'return_{period}m'] = self.df['close'].pct_change(period)
        
        # 对数收益率
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        
        # 价格动量
        for period in [5, 10, 20]:
            self.df[f'momentum_{period}m'] = (
                self.df['close'] / self.df['close'].shift(period) - 1
            )
        
        return self
    
    def add_volatility_features(self) -> 'AdvancedFeatureEngineer':
        """波动率特征"""
        # 已实现波动率
        for period in [10, 20, 60]:
            self.df[f'volatility_{period}m'] = (
                self.df['log_return'].rolling(period).std() * np.sqrt(390 * 252)
            )
        
        # Parkinson 波动率
        self.df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(self.df['high'] / self.df['low']) ** 2)
        ).rolling(60).mean() * np.sqrt(390 * 252)
        
        # ATR (Average True Range)
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        prev_close = close.shift(1)
        
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        for period in [14, 20]:
            self.df[f'atr_{period}m'] = tr.rolling(period).mean()
        
        return self
    
    def add_volume_features(self) -> 'AdvancedFeatureEngineer':
        """成交量特征"""
        # 成交量变化率
        self.df['volume_change'] = self.df['volume'].pct_change()
        
        # 成交量移动平均
        for period in [5, 10, 20]:
            self.df[f'volume_ma_{period}m'] = (
                self.df['volume'].rolling(period).mean()
            )
        
        # 成交量相对强度
        self.df['volume_ratio'] = (
            self.df['volume'] / self.df['volume'].rolling(20).mean()
        )
        
        # VWAP
        self.df['vwap'] = (
            (self.df['high'] + self.df['low'] + self.df['close']) / 3 * 
            self.df['volume']
        ).cumsum() / self.df['volume'].cumsum()
        
        # 价格相对 VWAP
        self.df['price_to_vwap'] = self.df['close'] / self.df['vwap'] - 1
        
        return self
    
    def add_technical_indicators(self) -> 'AdvancedFeatureEngineer':
        """技术指标"""
        try:
            import talib
            
            # RSI
            for period in [7, 14, 21]:
                self.df[f'rsi_{period}'] = talib.RSI(
                    self.df['close'].values, timeperiod=period
                )
            
            # MACD
            macd, signal, hist = talib.MACD(
                self.df['close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            self.df['macd'] = macd
            self.df['macd_signal'] = signal
            self.df['macd_hist'] = hist
            
            # Bollinger Bands
            for period in [10, 20]:
                upper, middle, lower = talib.BBANDS(
                    self.df['close'].values,
                    timeperiod=period,
                    nbdevup=2,
                    nbdevdn=2
                )
                self.df[f'bb_upper_{period}'] = upper
                self.df[f'bb_middle_{period}'] = middle
                self.df[f'bb_lower_{period}'] = lower
                
                # BB 宽度和位置
                self.df[f'bb_width_{period}'] = (
                    (upper - lower) / middle
                )
                self.df[f'bb_position_{period}'] = (
                    (self.df['close'] - lower) / (upper - lower)
                )
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                self.df['high'].values,
                self.df['low'].values,
                self.df['close'].values
            )
            self.df['stoch_k'] = slowk
            self.df['stoch_d'] = slowd
            
            # ADX (趋势强度)
            self.df['adx'] = talib.ADX(
                self.df['high'].values,
                self.df['low'].values,
                self.df['close'].values,
                timeperiod=14
            )
            
            # CCI
            self.df['cci'] = talib.CCI(
                self.df['high'].values,
                self.df['low'].values,
                self.df['close'].values,
                timeperiod=14
            )
            
        except ImportError:
            print("⚠️  ta-lib not installed, using simplified indicators")
            self._add_simple_indicators()
        
        return self
    
    def _add_simple_indicators(self):
        """简化版技术指标（不依赖 ta-lib）"""
        # RSI
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        
        rs = avg_gain / avg_loss
        self.df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = self.df['close'].rolling(20).mean()
        std_20 = self.df['close'].rolling(20).std()
        
        self.df['bb_upper_20'] = sma_20 + 2 * std_20
        self.df['bb_middle_20'] = sma_20
        self.df['bb_lower_20'] = sma_20 - 2 * std_20
    
    def add_time_features(self) -> 'AdvancedFeatureEngineer':
        """时间特征"""
        # 小时和分钟
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['minute'] = self.df['datetime'].dt.minute
        
        # 是否开盘/收盘时段
        self.df['is_market_open'] = (
            (self.df['hour'] >= 9) & (self.df['hour'] < 16)
        ).astype(int)
        
        # 是否收盘前30分钟
        self.df['is_near_close'] = (
            (self.df['hour'] == 15) & (self.df['minute'] >= 30)
        ).astype(int)
        
        # 星期几
        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        
        return self
    
    def add_microstructure_features(self) -> 'AdvancedFeatureEngineer':
        """市场微观结构特征（简化版）"""
        # 订单流不平衡代理
        self.df['flow_imbalance'] = (
            (self.df['close'] - self.df['open']) / 
            (self.df['high'] - self.df['low'] + 1e-8)
        )
        
        # 日内高低点位置
        self.df['intraday_high_pos'] = (
            (self.df['close'] - self.df['low']) / 
            (self.df['high'] - self.df['low'] + 1e-8)
        )
        
        # 价格跳跃
        self.df['price_gap'] = (
            (self.df['open'] - self.df['close'].shift(1)) / 
            self.df['close'].shift(1)
        )
        
        return self
    
    def build(self) -> pd.DataFrame:
        """构建最终特征"""
        # 执行所有特征工程
        self.add_price_features()
        self.add_volatility_features()
        self.add_volume_features()
        self.add_technical_indicators()
        self.add_time_features()
        self.add_microstructure_features()
        
        # 清理 NaN
        self.df = self.df.fillna(method='ffill').fillna(0)
        
        # 移除无穷值
        self.df = self.df.replace([np.inf, -np.inf], 0)
        
        return self.df

# 使用示例
if __name__ == "__main__":
    # 加载原始数据
    df = pd.read_csv('./dataset/raw/SOXL_1m.csv')
    
    # 特征工程
    engineer = AdvancedFeatureEngineer(df)
    features_df = engineer.build()
    
    # 保存
    features_df.to_csv('./dataset/features/SOXL_1m_features.csv', index=False)
    print(f"✅ Generated {len(features_df.columns)} features")
```

---

## 🎮 第三步：环境调试

### 3.1 环境单元测试

```python
# tests/test_trading_env.py
import pytest
import pandas as pd
import numpy as np
from envs.trading_env import ETFTradingEnv

def create_test_data():
    """创建测试数据"""
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'datetime': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000),
        'rsi_14': np.random.uniform(30, 70, 1000),
        'macd': np.random.randn(1000),
        'bb_upper_20': np.random.randn(1000).cumsum() + 105,
        'bb_lower_20': np.random.randn(1000).cumsum() + 95,
    })
    return data

def test_env_initialization():
    """测试环境初始化"""
    data = create_test_data()
    config = {
        'window_size': 60,
        'initial_balance': 100000,
        'transaction_cost': 0.001,
    }
    
    env = ETFTradingEnv(data, config)
    
    assert env.window_size == 60
    assert env.initial_balance == 100000
    assert env.portfolio_value == 100000
    assert env.position == 0.0
    
    print("✅ Environment initialization test passed")

def test_env_reset():
    """测试重置功能"""
    data = create_test_data()
    env = ETFTradingEnv(data)
    
    obs, info = env.reset()
    
    assert obs.shape == (60, data.shape[1] + 2)  # features + position + balance
    assert info['portfolio_value'] == 100000
    
    print("✅ Reset test passed")

def test_env_step():
    """测试步进功能"""
    data = create_test_data()
    env = ETFTradingEnv(data)
    
    obs, _ = env.reset()
    
    # 测试买入动作
    action = np.array([0.5])  # 半仓
    obs, reward, done, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert 'portfolio_value' in info
    assert 'position' in info
    
    print("✅ Step test passed")

def test_full_episode():
    """测试完整回合"""
    data = create_test_data()
    env = ETFTradingEnv(data)
    
    obs, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    while not done:
        action = np.random.uniform(0, 1, 1)  # 随机动作
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps > 1000:  # 安全退出
            break
    
    print(f"✅ Full episode test passed: {steps} steps, total reward: {total_reward:.4f}")

def test_transaction_cost():
    """测试交易成本"""
    data = create_test_data()
    config = {
        'transaction_cost': 0.01  # 1% 高成本
    }
    env = ETFTradingEnv(data, config)
    
    obs, _ = env.reset()
    
    # 频繁交易应该有显著成本
    for _ in range(10):
        action = np.array([1.0])  # 满仓
        obs, _, done, _, _ = env.step(action)
        action = np.array([0.0])  # 空仓
        obs, _, done, _, _ = env.step(action)
        
        if done:
            break
    
    # 频繁交易应该导致资金减少
    assert env.portfolio_value < 100000, "Frequent trading should reduce portfolio"
    
    print(f"✅ Transaction cost test passed: final value {env.portfolio_value:.2f}")

if __name__ == "__main__":
    test_env_initialization()
    test_env_reset()
    test_env_step()
    test_full_episode()
    test_transaction_cost()
    
    print("\n✅ All tests passed!")
```

### 3.2 环境可视化调试

```python
# notebooks/debug_environment.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from envs.trading_env import ETFTradingEnv

# 创建测试环境
data = pd.read_csv('./dataset/features/SOXL_1m_features.csv')
env = ETFTradingEnv(data)

# 测试随机策略
obs, _ = env.reset()
done = False

portfolio_values = [100000]
positions = [0]
rewards = []

while not done:
    # 随机动作
    action = np.random.uniform(0, 1, 1)
    obs, reward, done, truncated, info = env.step(action)
    
    portfolio_values.append(info['portfolio_value'])
    positions.append(info['position'])
    rewards.append(reward)
    
    if len(portfolio_values) > 500:  # 限制步数
        break

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# 组合价值
axes[0].plot(portfolio_values)
axes[0].set_title('Portfolio Value (Random Policy)')
axes[0].set_ylabel('Value ($)')

# 仓位
axes[1].plot(positions)
axes[1].set_title('Position Over Time')
axes[1].set_ylabel('Position (0-1)')

# 奖励
axes[2].plot(rewards)
axes[2].set_title('Rewards')
axes[2].set_ylabel('Reward')
axes[2].set_xlabel('Time Step')

plt.tight_layout()
plt.savefig('./logs/environment_debug.png', dpi=150)
print("✅ Debug visualization saved")
```

---

## 🤖 第四步：模型训练

### 4.1 基线模型训练

```bash
cd ~/workspace/soxl-rl-training

# 基础训练
python scripts/train.py \
    --data ../soxl-rl-data/dataset/features/SOXL_1m_features.csv \
    --config configs/qlib.yaml \
    --output ./experiments/baseline \
    --total-timesteps 50000

# Walk-forward 验证
python scripts/train.py \
    --data ../soxl-rl-data/dataset/features/SOXL_1m_features.csv \
    --config configs/qlib.yaml \
    --output ./experiments/walk_forward \
    --walk-forward \
    --n-splits 5
```

### 4.2 超参数调优

```python
# scripts/hyperparameter_tuning.py
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import numpy as np

def objective(trial):
    """Optuna 目标函数"""
    # 超参数搜索空间
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_epochs = trial.suggest_int('n_epochs', 5, 20)
    gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 0.99)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    
    # 创建环境
    data = pd.read_csv('../soxl-rl-data/dataset/features/SOXL_1m_features.csv')
    from envs.trading_env import ETFTradingEnv
    
    env = ETFTradingEnv(data, {
        'window_size': 60,
        'initial_balance': 100000,
        'transaction_cost': 0.001,
    })
    
    # 训练模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=0
    )
    
    model.learn(total_timesteps=20000)
    
    # 评估
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            break
    
    return total_reward

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:")
print(study.best_params)
print(f"Best value: {study.best_value}")
```

### 4.3 TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir=./logs/tensorboard --port=6006

# 在浏览器中打开
# http://localhost:6006
```

---

## 📈 第五步：回测与评估

### 5.1 完整回测流程

```python
# scripts/comprehensive_backtest.py
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from envs.trading_env import ETFTradingEnv
from scripts.evaluate import evaluate_performance, print_metrics
import matplotlib.pyplot as plt

def comprehensive_backtest(model_path, test_data_path, output_dir):
    """综合回测"""
    
    # 加载模型和数据
    model = PPO.load(model_path)
    test_data = pd.read_csv(test_data_path)
    
    # 创建测试环境
    env = ETFTradingEnv(test_data, {
        'window_size': 60,
        'initial_balance': 100000,
        'transaction_cost': 0.001,
    })
    
    # 运行回测
    obs, _ = env.reset()
    done = False
    
    portfolio_values = [100000]
    positions = [0]
    actions_list = []
    returns = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        positions.append(info['position'])
        actions_list.append(float(action[0]))
        returns.append(info.get('return', 0))
        
        done = done or truncated
    
    # 获取历史
    history = env.get_portfolio_history()
    
    # 计算指标
    metrics = evaluate_performance(history)
    print_metrics(metrics)
    
    # 生成报告
    create_backtest_report(
        portfolio_values,
        positions,
        actions_list,
        returns,
        metrics,
        output_dir
    )
    
    return metrics

def create_backtest_report(portfolio_values, positions, actions, returns, metrics, output_dir):
    """创建回测报告"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3)
    
    # 组合价值
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(portfolio_values, linewidth=2)
    ax1.axhline(y=100000, color='gray', linestyle='--', label='Initial')
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 仓位
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(range(len(positions)), positions, alpha=0.5)
    ax2.set_title('Position Size', fontsize=12)
    ax2.set_ylabel('Position (0-1)')
    ax2.grid(True, alpha=0.3)
    
    # 动作分布
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(actions, bins=50, edgecolor='black', alpha=0.7)
    ax3.set_title('Action Distribution', fontsize=12)
    ax3.set_xlabel('Action Value')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 收益率分布
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(returns, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.axvline(x=np.mean(returns), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(returns):.4f}')
    ax4.set_title('Returns Distribution', fontsize=12)
    ax4.set_xlabel('Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 累积收益
    ax5 = fig.add_subplot(gs[2, 1])
    cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1
    ax5.plot(cumulative_returns.values, linewidth=2)
    ax5.axhline(y=0, color='gray', linestyle='--')
    ax5.set_title('Cumulative Returns', fontsize=12)
    ax5.set_ylabel('Cumulative Return')
    ax5.grid(True, alpha=0.3)
    
    # 关键指标
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    metrics_text = f"""
    Performance Metrics
    ===================
    Total Return: {metrics.get('total_return', 0):.2%}
    Annualized Return: {metrics.get('annualized_return', 0):.2%}
    
    Risk Metrics
    ============
    Volatility: {metrics.get('volatility', 0):.2%}
    Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
    VaR (95%): {metrics.get('var_95', 0):.4f}
    
    Risk-Adjusted Returns
    =====================
    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
    Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
    Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}
    
    Trading Statistics
    ==================
    Number of Trades: {metrics.get('num_trades', 0)}
    Win Rate: {metrics.get('win_rate', 0):.2%}
    Profit Factor: {metrics.get('profit_factor', 0):.2f}
    """
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.savefig(f'{output_dir}/backtest_report.png', dpi=150, bbox_inches='tight')
    print(f"✅ Backtest report saved to {output_dir}/backtest_report.png")
    
    # 保存数据
    pd.DataFrame({
        'portfolio_value': portfolio_values,
        'position': positions,
        'return': [0] + returns
    }).to_csv(f'{output_dir}/backtest_data.csv', index=False)
    
    # 保存指标
    pd.DataFrame([metrics]).to_csv(f'{output_dir}/metrics.csv', index=False)

if __name__ == "__main__":
    comprehensive_backtest(
        model_path='./experiments/baseline/final_model.zip',
        test_data_path='../soxl-rl-data/dataset/features/SOXL_1m_features.csv',
        output_dir='./results/backtest'
    )
```

---

## 🐛 常见问题与解决方案

### 问题 1: 数据不足
**症状**: Yahoo Finance 只能获取 7-30 天的分钟数据

**解决方案**:
```python
# 方案 A: 多次请求拼接
def download_extended_data(symbol, total_days=180):
    """下载扩展历史数据"""
    all_data = []
    
    for start_offset in range(0, total_days, 7):
        end_date = datetime.now() - timedelta(days=start_offset)
        start_date = end_date - timedelta(days=7)
        
        data = yf.download(symbol, start=start_date, end=end_date, interval='1m')
        if not data.empty:
            all_data.append(data)
    
    return pd.concat(all_data).drop_duplicates()

# 方案 B: 使用 5 分钟数据替代
# 5 分钟数据可以获取 60 天

# 方案 C: 使用第三方数据源
# Tiingo, Alpha Vantage, Polygon.io
```

### 问题 2: 模型不收敛
**症状**: 奖励持续为负或不稳定

**解决方案**:
```python
# 1. 降低学习率
learning_rate = 1e-5  # 从 3e-4 降低

# 2. 增加训练步数
total_timesteps = 200000  # 从 100000 增加

# 3. 简化奖励函数
def simple_reward(portfolio_return, transaction_cost):
    # 先用简单的奖励函数
    return portfolio_return - transaction_cost * 5

# 4. 检查特征标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 5. 减小网络规模
policy_kwargs = {
    'net_arch': [32, 32]  # 从 [64, 64] 减小
}
```

### 问题 3: 过拟合
**症状**: 训练集表现好，测试集表现差

**解决方案**:
```python
# 1. 使用 Walk-forward 验证
n_splits = 5
train_test_split = 0.8

# 2. 添加正则化
# 在模型配置中
policy_kwargs = {
    'net_arch': [64, 64],
    'activation_fn': nn.ReLU,
    # 添加 dropout
}

# 3. 减少特征数量
selected_features = [
    'return_1m', 'return_5m',
    'rsi_14', 'macd', 'bb_position_20',
    'volume_ratio', 'volatility_20m'
]

# 4. 早停
from stable_baselines3.common.callbacks import EarlyStoppingCallback

# 5. 增加噪声
def add_noise(obs, noise_level=0.01):
    return obs + np.random.randn(*obs.shape) * noise_level
```

### 问题 4: 交易成本过高
**症状**: 频繁交易导致资金快速减少

**解决方案**:
```python
# 1. 增加交易成本惩罚
transaction_cost = 0.002  # 从 0.001 增加到 0.002

# 2. 在奖励函数中放大惩罚
reward = portfolio_return - transaction_cost * 20

# 3. 添加最小持仓时间限制
class ETFTradingEnv(gym.Env):
    def __init__(self, ...):
        self.min_holding_period = 5  # 最少持有 5 分钟
        self.last_trade_time = 0
    
    def step(self, action):
        # 检查是否满足最小持仓时间
        if self.current_step - self.last_trade_time < self.min_holding_period:
            action = np.array([self.position])  # 保持当前仓位
        else:
            self.last_trade_time = self.current_step
        
        # ...

# 4. 限制交易频率
max_trades_per_day = 10
```

---

## 📚 最佳实践总结

### 1. 数据处理
✅ **DO**:
- 使用多个数据源交叉验证
- 完整的数据质量检查
- 合理的缺失值处理
- 特征标准化

❌ **DON'T**:
- 直接使用原始价格（应使用收益率）
- 忽略数据时间对齐
- 过度特征工程（导致过拟合）

### 2. 环境设计
✅ **DO**:
- 从简单环境开始，逐步增加复杂度
- 真实模拟交易成本
- 使用合理的奖励函数
- 充分测试环境逻辑

❌ **DON'T**:
- 忽略交易成本
- 使用过于复杂的奖励函数
- 动作空间过大

### 3. 模型训练
✅ **DO**:
- 从小模型开始
- 使用 Walk-forward 验证
- 监控训练曲线
- 保存检查点

❌ **DON'T**:
- 一次性训练过多步数
- 忽略过拟合信号
- 使用单一验证集

### 4. 评估与部署
✅ **DO**:
- 使用多个评估指标
- 压力测试极端行情
- 与基准对比
- 持续监控模型表现

❌ **DON'T**:
- 只看总收益
- 忽略风险指标
- 直接上线未测试的模型

---

## 🎯 成功标准

### 基线模型（Week 2）
- [ ] Sharpe Ratio > 1.0
- [ ] Max Drawdown < 30%
- [ ] Win Rate > 50%
- [ ] 与 Buy & Hold 对比

### 优化模型（Week 3）
- [ ] Sharpe Ratio > 1.5
- [ ] Max Drawdown < 20%
- [ ] Win Rate > 55%
- [ ] 跑赢基准 5%+

### 可上线模型（Week 4）
- [ ] Sharpe Ratio > 2.0
- [ ] Max Drawdown < 15%
- [ ] Win Rate > 60%
- [ ] 稳定跑赢基准 10%+
- [ ] 通过压力测试

---

## 📞 获取帮助

**遇到问题时**:
1. 检查本文档的"常见问题"部分
2. 查看代码注释和文档字符串
3. 在 Discord #soxl 频道提问
4. @clawQuant 寻求技术指导

**协作流程**:
- 每日更新进度到 #soxl
- 每周技术评审会议
- 重大决策需要讨论

---

**祝训练顺利！🚀**
