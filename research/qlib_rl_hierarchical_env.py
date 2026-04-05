#!/usr/bin/env python3
"""
Qlib 框架下的分层RL环境

使用真正的 Qlib 组件：
1. ✅ qlib.data.BaseProvider - 数据提供器
2. ✅ Qlib Expression - 特征计算
3. ✅ 训练/测试时间分割
4. ✅ Qlib 风格回测
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import qlib
from qlib.data import BaseProvider
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import gymnasium as gym
from gymnasium import spaces
import re


# ============================================================================
# Qlib Data Provider for RL
# ============================================================================

class SOXLQlibProvider(BaseProvider):
    """
    SOXL + VIX 数据提供器（继承 qlib.data.BaseProvider）
    """
    
    def __init__(self):
        super().__init__()
        self._data = None
    
    def load(self) -> pd.DataFrame:
        """加载数据"""
        if self._data is None:
            # 使用完整数据（2010-2026）
            df = pd.read_csv(
                '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix_full.csv',
                parse_dates=['date']
            )
            # 不再限制开始日期，使用全部数据
            df = df.set_index('date')
            self._data = df
        return self._data
    
    def get_features(self, fields: List[str]) -> pd.DataFrame:
        """
        获取特征（使用 Qlib Expression 语法）
        
        支持：
        - $close, $volume, $vix
        - Mean($close, N)
        - Std($close, N)
        - RSI($close, N)
        """
        df = self.load()
        result = {}
        
        for field in fields:
            if field in df.columns:
                result[field] = df[field]
                continue
            
            # $close, $volume 等
            if field.startswith('$'):
                col = field[1:]
                if col in df.columns:
                    result[field] = df[col]
            
            # Mean($close, N)
            elif field.startswith('Mean('):
                match = re.match(r'Mean\(\$(\w+),\s*(\d+)\)', field)
                if match:
                    col, window = match.groups()
                    if col in df.columns:
                        result[field] = df[col].rolling(int(window)).mean()
            
            # Std($close, N)
            elif field.startswith('Std('):
                match = re.match(r'Std\(\$(\w+),\s*(\d+)\)', field)
                if match:
                    col, window = match.groups()
                    if col in df.columns:
                        result[field] = df[col].rolling(int(window)).std()
            
            # RSI($close, N)
            elif field.startswith('RSI('):
                match = re.match(r'RSI\(\$(\w+),\s*(\d+)\)', field)
                if match:
                    col, window = match.groups()
                    if col in df.columns:
                        delta = df[col].diff()
                        gain = delta.where(delta > 0, 0).rolling(int(window)).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(int(window)).mean()
                        result[field] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        return pd.DataFrame(result, index=df.index)


# ============================================================================
# Qlib 风格的分层RL环境
# ============================================================================

class SOXLQlibHierarchicalEnv(gym.Env):
    """
    SOXL 分层RL环境 - 使用 Qlib 框架
    
    关键改进：
    1. ✅ 使用 Qlib Expression
    2. ✅ 训练/测试时间分割
    3. ✅ 继承 BaseProvider
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        provider: Optional[SOXLQlibProvider] = None,
        start_date: str = '2020-01-01',
        end_date: str = '2026-04-01',
        train_ratio: float = 0.7,  # 70% 训练，30% 测试
        mode: str = 'train',  # 'train' or 'test'
        initial_balance: float = 100000,
        commission: float = 0.001,
        lookback: int = 20,
        max_short: float = 0.3,
        # 规则策略参数
        panic_threshold: float = -35.2,
        vix_panic: float = 20.4,
        vix_extreme: float = 28.5,
        vix_greed: float = 12.7,
    ):
        super().__init__()
        
        # Qlib Provider
        self.provider = provider if provider else SOXLQlibProvider()
        
        # 时间分割
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.train_ratio = train_ratio
        self.mode = mode
        
        # 加载并分割数据
        self._load_and_split_data()
        
        # 环境参数
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback = lookback
        self.max_short = max_short
        
        # 规则策略参数
        self.panic_threshold = panic_threshold
        self.vix_panic = vix_panic
        self.vix_extreme = vix_extreme
        self.vix_greed = vix_greed
        
        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # 观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )
        
        self.reset()
    
    def _load_and_split_data(self):
        """加载并分割训练/测试数据"""
        # 使用 Qlib Expression 计算特征
        fields = [
            '$close', '$volume', '$vix',
            'Mean($close, 5)', 'Mean($close, 20)',
            'RSI($close, 14)',
        ]
        
        features = self.provider.get_features(fields)
        df = self.provider.load()
        
        # 合并
        for col in features.columns:
            df[col] = features[col]
        
        # 计算额外特征（Qlib 风格）
        df['return_1'] = df['$close'].pct_change()
        df['vix_normalized'] = df['$vix'] / 50
        df['vix_ma20'] = df['$vix'].rolling(20).mean()
        df['vix_std20'] = df['$vix'].rolling(20).std()
        df['vix_zscore'] = (df['$vix'] - df['vix_ma20']) / (df['vix_std20'] + 1e-8)
        
        # 恐慌得分
        df['panic_score'] = (
            df['vix_zscore'] * 40 +
            df['$vix'].rolling(20).rank(pct=True) * 60 +
            np.where(df['return_1'] < 0, np.minimum(df['return_1'] / -0.05, 1) * -50, 0)
        )
        
        df = df.dropna()
        
        # 时间分割
        total_len = len(df)
        train_len = int(total_len * self.train_ratio)
        
        if self.mode == 'train':
            self.df = df.iloc[:train_len].reset_index(drop=True)
            print(f"训练数据: {len(self.df)} records ({df.index[0]} to {df.index[train_len-1]})")
        else:
            self.df = df.iloc[train_len:].reset_index(drop=True)
            print(f"测试数据: {len(self.df)} records ({df.index[train_len]} to {df.index[-1]})")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0.0
        self.shares = 0
        self.current_step = self.lookback
        self.total_value = self.initial_balance
        self.returns = []
        self.history = [self.initial_balance]
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """获取当前状态（使用 Qlib Expression 字段）"""
        row = self.df.iloc[self.current_step]
        
        portfolio_return = (self.total_value - self.initial_balance) / self.initial_balance
        
        # 计算波动率
        if len(self.history) > 10:
            hist_array = np.array(self.history[-20:])
            returns = np.diff(hist_array) / hist_array[:-1]
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        else:
            volatility = 0
        
        # 规则信号
        rule_signal = self._get_rule_signal()
        
        # 成交量比率
        volume_ma20 = self.df['$volume'].iloc[self.current_step-19:self.current_step+1].mean()
        volume_ratio = row['$volume'] / (volume_ma20 + 1e-8)
        
        # 状态（使用 Qlib 字段）
        obs = np.array([
            row['vix_normalized'],
            row['vix_zscore'],
            row['return_1'],
            row['RSI($close, 14)'] / 100,
            row['panic_score'] / 100,
            self.position,
            portfolio_return,
            volatility,
            row['Mean($close, 5)'] / row['Mean($close, 20)'],
            volume_ratio,
            rule_signal,
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _get_rule_signal(self) -> float:
        """规则策略信号"""
        row = self.df.iloc[self.current_step]
        
        # 使用 Qlib Expression 字段
        if row['panic_score'] <= self.panic_threshold:
            if row['$vix'] > self.vix_extreme:
                return 1.0
            elif row['$vix'] > self.vix_panic:
                return 0.9
        
        if row['RSI($close, 14)'] < 65 and row['Mean($close, 5)'] > row['Mean($close, 20)']:
            return 0.6
        
        if row['$vix'] < self.vix_greed:
            if row['RSI($close, 14)'] > 65:
                return 0.2
            return 0.4
        
        return 0.5
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        current_price = float(self.df['$close'].iloc[self.current_step])
        
        # 解析RL动作
        short_ratio = float(action[0])
        long_ratio = float(action[1])
        rl_position = long_ratio - short_ratio
        rl_position = np.clip(rl_position, -self.max_short, 1.0)
        
        # 规则信号
        rule_signal = self._get_rule_signal()
        
        # 最终仓位
        target_position = rule_signal * rl_position
        target_position = np.clip(target_position, -self.max_short, 1.0)
        
        # 执行交易
        position_change = target_position - self.position
        trade_value = abs(position_change) * self.total_value
        commission_cost = trade_value * self.commission
        
        self.position = target_position
        self.shares = (self.total_value * abs(self.position)) / current_price
        
        # 前进一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 计算收益
        if not done:
            next_price = float(self.df['$close'].iloc[self.current_step])
            price_return = (next_price - current_price) / current_price
            
            if self.position >= 0:
                portfolio_return = self.position * price_return - commission_cost / self.total_value
            else:
                portfolio_return = -self.position * (-price_return) - commission_cost / self.total_value
            
            self.returns.append(portfolio_return)
            self.total_value *= (1 + portfolio_return)
            self.history.append(self.total_value)
            
            # 计算奖励（优化版）
            reward = self._calculate_reward_optimized(portfolio_return)
        else:
            reward = 0
        
        info = {
            'total_value': self.total_value,
            'position': self.position,
            'rl_position': rl_position,
            'rule_signal': rule_signal,
            'return': self.returns[-1] if self.returns else 0,
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _calculate_reward_optimized(self, portfolio_return: float) -> float:
        """优化的奖励函数"""
        # 收益奖励
        return_reward = portfolio_return * 150
        
        # Sharpe奖励
        if len(self.returns) > 10:
            mean_return = np.mean(self.returns[-20:])
            std_return = np.std(self.returns[-20:]) + 1e-8
            sharpe_reward = mean_return / std_return * 15
        else:
            sharpe_reward = 0
        
        # 回撤惩罚（仅>15%时惩罚）
        if len(self.history) > 0:
            peak = max(self.history)
            drawdown = (peak - self.total_value) / peak
            if drawdown > 0.15:
                dd_penalty = -(drawdown - 0.15) * 30
            else:
                dd_penalty = 0
        else:
            dd_penalty = 0
        
        # 波动惩罚
        if len(self.returns) > 5:
            recent_vol = np.std(self.returns[-10:])
            vol_penalty = -recent_vol * 5
        else:
            vol_penalty = 0
        
        # 持仓奖励
        if 0.3 <= abs(self.position) <= 0.8:
            position_reward = 2.0
        else:
            position_reward = 0
        
        reward = return_reward + sharpe_reward + dd_penalty + vol_penalty + position_reward
        
        return reward
    
    def render(self):
        rule_signal = self._get_rule_signal()
        print(f"Step {self.current_step}: Value=${self.total_value:.2f}, "
              f"Pos={self.position:.2f}, Rule={rule_signal:.2f}")


if __name__ == '__main__':
    print("="*70)
    print("Qlib 框架分层RL环境")
    print("="*70)
    print("""
使用的 Qlib 组件：
  ✅ qlib.data.BaseProvider - 数据提供器基类
  ✅ Qlib Expression - Mean($close, N), RSI($close, N)
  ✅ 训练/测试时间分割
  ✅ Qlib 字段命名 - $close, $volume, $vix
    """)
    
    # 测试训练环境
    print("创建训练环境...")
    train_env = SOXLQlibHierarchicalEnv(mode='train')
    print(f"训练数据: {len(train_env.df)} records")
    
    # 测试测试环境
    print("\n创建测试环境...")
    test_env = SOXLQlibHierarchicalEnv(mode='test')
    print(f"测试数据: {len(test_env.df)} records")
    
    print("\n✅ 环境创建成功！")
