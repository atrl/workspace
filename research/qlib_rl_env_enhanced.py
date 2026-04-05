#!/usr/bin/env python3
"""
增强版 Qlib RL 环境 - 添加 SOXX 特征

新增特征：
1. SOXX 收益率
2. SOXX/SOXL 比率（杠杆效应）
3. SOXX 趋势
4. SOXX RSI
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import gymnasium as gym
from gymnasium import spaces
import re

# 导入 Qlib 基类
import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')
from qlib_rl_hierarchical_env import SOXLQlibProvider


class SOXLQlibProviderEnhanced(SOXLQlibProvider):
    """增强版 Provider - 支持 SOXX 特征"""
    
    def load(self) -> pd.DataFrame:
        """加载完整数据（SOXL + SOXX + VIX）"""
        if self._data is None:
            df = pd.read_csv(
                '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv',
                parse_dates=['date']
            )
            df = df.set_index('date')
            self._data = df
        return self._data
    
    def get_features(self, fields: List[str]) -> pd.DataFrame:
        """
        获取特征（支持 SOXX 相关 Expression）
        
        新增：
        - $soxx_close, $soxx_volume
        - Mean($soxx_close, N)
        - RSI($soxx_close, N)
        - leverage_ratio = SOXL / SOXX
        """
        df = self.load()
        result = {}
        
        for field in fields:
            if field in df.columns:
                result[field] = df[field]
                continue
            
            # SOXL 特征
            if field.startswith('$soxl_'):
                col = field[6:]  # 去掉 $soxl_
                if col in df.columns:
                    result[field] = df[col]
            
            # SOXX 特征
            elif field.startswith('$soxx_'):
                col = field[6:]  # 去掉 $soxx_
                if col in df.columns:
                    result[field] = df[col]
            
            # VIX 特征
            elif field == '$vix':
                result[field] = df['vix']
            
            # 杠杆比率
            elif field == 'leverage_ratio':
                result[field] = df['soxl_close'] / df['soxx_close']
            
            # SOXL 收益率
            elif field == 'soxl_return':
                result[field] = df['soxl_close'].pct_change()
            
            # SOXX 收益率
            elif field == 'soxx_return':
                result[field] = df['soxx_close'].pct_change()
            
            # SOXL 均线
            elif field.startswith('Mean($soxl_close'):
                match = re.match(r'Mean\(\$soxl_close,\s*(\d+)\)', field)
                if match:
                    window = int(match.group(1))
                    result[field] = df['soxl_close'].rolling(window).mean()
            
            # SOXX 均线
            elif field.startswith('Mean($soxx_close'):
                match = re.match(r'Mean\(\$soxx_close,\s*(\d+)\)', field)
                if match:
                    window = int(match.group(1))
                    result[field] = df['soxx_close'].rolling(window).mean()
            
            # SOXL RSI
            elif field.startswith('RSI($soxl_close'):
                match = re.match(r'RSI\(\$soxl_close,\s*(\d+)\)', field)
                if match:
                    window = int(match.group(1))
                    delta = df['soxl_close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                    result[field] = 100 - (100 / (1 + gain / (loss + 1e-8)))
            
            # SOXX RSI
            elif field.startswith('RSI($soxx_close'):
                match = re.match(r'RSI\(\$soxx_close,\s*(\d+)\)', field)
                if match:
                    window = int(match.group(1))
                    delta = df['soxx_close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                    result[field] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        return pd.DataFrame(result, index=df.index)


class SOXLQlibHierarchicalEnvEnhanced(gym.Env):
    """
    增强版 Qlib RL 环境 - 包含 SOXX 特征
    
    状态空间：16维（原来11维 + 5维新特征）
    
    新增特征：
    1. SOXX 收益率
    2. SOXX RSI
    3. 杠杆比率（SOXL/SOXX）
    4. SOXX 趋势
    5. 基准偏离度
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        provider: Optional[SOXLQlibProviderEnhanced] = None,
        mode: str = 'train',
        split_date: str = '2020-01-01',
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
        
        self.provider = provider if provider else SOXLQlibProviderEnhanced()
        self.split_date = pd.Timestamp(split_date)
        self.mode = mode
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback = lookback
        self.max_short = max_short
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
        
        # 观察空间：16维
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )
        
        self._load_data()
        self.reset()
    
    def _load_data(self):
        """加载并分割数据"""
        # Qlib Expression 字段
        fields = [
            '$soxl_close', '$soxl_volume', '$vix',
            '$soxx_close', '$soxx_volume',
            'Mean($soxl_close, 5)', 'Mean($soxl_close, 20)',
            'Mean($soxx_close, 5)', 'Mean($soxx_close, 20)',
            'RSI($soxl_close, 14)',
            'RSI($soxx_close, 14)',
            'leverage_ratio',
            'soxl_return',
            'soxx_return',
        ]
        
        features = self.provider.get_features(fields)
        df = self.provider.load()
        
        # 合并
        for col in features.columns:
            df[col] = features[col]
        
        # 计算额外特征
        df['vix_normalized'] = df['vix'] / 50
        df['vix_ma20'] = df['vix'].rolling(20).mean()
        df['vix_std20'] = df['vix'].rolling(20).std()
        df['vix_zscore'] = (df['vix'] - df['vix_ma20']) / (df['vix_std20'] + 1e-8)
        
        # 恐慌得分
        df['panic_score'] = (
            df['vix_zscore'] * 40 +
            df['vix'].rolling(20).rank(pct=True) * 60 +
            np.where(df['soxl_return'] < 0, np.minimum(df['soxl_return'] / -0.05, 1) * -50, 0)
        )
        
        # 基准偏离度（SOXL vs SOXX）
        df['benchmark_deviation'] = (
            df['soxl_return'] - df['soxx_return']
        ) / (df['soxx_return'].abs() + 1e-8)
        
        # 杠杆比率变化
        df['leverage_ratio_change'] = df['leverage_ratio'].pct_change()
        
        df = df.dropna()
        
        # 分割
        if self.mode == 'train':
            self.df = df[df.index < self.split_date].reset_index(drop=True)
            print(f"训练集: {len(self.df)} records")
        else:
            self.df = df[df.index >= self.split_date].reset_index(drop=True)
            print(f"测试集: {len(self.df)} records")
    
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
        """获取16维状态"""
        row = self.df.iloc[self.current_step]
        
        portfolio_return = (self.total_value - self.initial_balance) / self.initial_balance
        
        # 波动率
        if len(self.history) > 10:
            hist_array = np.array(self.history[-20:])
            returns = np.diff(hist_array) / hist_array[:-1]
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        else:
            volatility = 0
        
        rule_signal = self._get_rule_signal()
        
        # 成交量比率
        soxl_vol_ma20 = self.df['soxl_volume'].iloc[self.current_step-19:self.current_step+1].mean()
        soxl_vol_ratio = row['soxl_volume'] / (soxl_vol_ma20 + 1e-8)
        
        # 16维状态
        obs = np.array([
            # 原有11维
            row['vix_normalized'],               # 0
            row['vix_zscore'],                   # 1
            row['soxl_return'],                  # 2
            row['RSI($soxl_close, 14)'] / 100,  # 3
            row['panic_score'] / 100,           # 4
            self.position,                        # 5
            portfolio_return,                     # 6
            volatility,                           # 7
            row['Mean($soxl_close, 5)'] / row['Mean($soxl_close, 20)'],  # 8
            soxl_vol_ratio,                       # 9
            rule_signal,                          # 10
            
            # 新增5维（SOXX相关）
            row['soxx_return'],                  # 11: SOXX收益率
            row['RSI($soxx_close, 14)'] / 100,  # 12: SOXX RSI
            row['leverage_ratio'] / 3.0,         # 13: 杠杆比率（归一化）
            row['Mean($soxx_close, 5)'] / row['Mean($soxx_close, 20)'],  # 14: SOXX趋势
            row['benchmark_deviation'],          # 15: 基准偏离度
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _get_rule_signal(self) -> float:
        """规则策略信号（不变）"""
        row = self.df.iloc[self.current_step]
        
        if row['panic_score'] <= self.panic_threshold:
            if row['vix'] > self.vix_extreme:
                return 1.0
            elif row['vix'] > self.vix_panic:
                return 0.9
        
        if row['RSI($soxl_close, 14)'] < 65 and row['Mean($soxl_close, 5)'] > row['Mean($soxl_close, 20)']:
            return 0.6
        
        if row['vix'] < self.vix_greed:
            if row['RSI($soxl_close, 14)'] > 65:
                return 0.2
            return 0.4
        
        return 0.5
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作（不变）"""
        current_price = float(self.df['soxl_close'].iloc[self.current_step])
        
        short_ratio = float(action[0])
        long_ratio = float(action[1])
        rl_position = long_ratio - short_ratio
        rl_position = np.clip(rl_position, -self.max_short, 1.0)
        
        rule_signal = self._get_rule_signal()
        target_position = rule_signal * rl_position
        target_position = np.clip(target_position, -self.max_short, 1.0)
        
        position_change = target_position - self.position
        trade_value = abs(position_change) * self.total_value
        commission_cost = trade_value * self.commission
        
        self.position = target_position
        self.shares = (self.total_value * abs(self.position)) / current_price
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        if not done:
            next_price = float(self.df['soxl_close'].iloc[self.current_step])
            price_return = (next_price - current_price) / current_price
            
            if self.position >= 0:
                portfolio_return = self.position * price_return - commission_cost / self.total_value
            else:
                portfolio_return = -self.position * (-price_return) - commission_cost / self.total_value
            
            self.returns.append(portfolio_return)
            self.total_value *= (1 + portfolio_return)
            self.history.append(self.total_value)
            
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
        """奖励函数（不变）"""
        return_reward = portfolio_return * 150
        
        if len(self.returns) > 10:
            mean_return = np.mean(self.returns[-20:])
            std_return = np.std(self.returns[-20:]) + 1e-8
            sharpe_reward = mean_return / std_return * 15
        else:
            sharpe_reward = 0
        
        if len(self.history) > 0:
            peak = max(self.history)
            drawdown = (peak - self.total_value) / peak
            if drawdown > 0.15:
                dd_penalty = -(drawdown - 0.15) * 30
            else:
                dd_penalty = 0
        else:
            dd_penalty = 0
        
        if len(self.returns) > 5:
            recent_vol = np.std(self.returns[-10:])
            vol_penalty = -recent_vol * 5
        else:
            vol_penalty = 0
        
        if 0.3 <= abs(self.position) <= 0.8:
            position_reward = 2.0
        else:
            position_reward = 0
        
        reward = (
            return_reward +
            sharpe_reward +
            dd_penalty +
            vol_penalty +
            position_reward
        )
        
        return reward
    
    def render(self):
        rule_signal = self._get_rule_signal()
        print(f"Step {self.current_step}: Value=${self.total_value:.2f}, "
              f"Pos={self.position:.2f}, Rule={rule_signal:.2f}")


if __name__ == '__main__':
    print("="*70)
    print("增强版 Qlib RL 环境 - 包含 SOXX 特征")
    print("="*70)
    print("""
新增特征（5维）：
  1. SOXX 收益率
  2. SOXX RSI
  3. 杠杆比率（SOXL/SOXX）
  4. SOXX 趋势
  5. 基准偏离度

状态空间：11维 → 16维

Qlib 组件：
  ✅ qlib.data.BaseProvider
  ✅ Qlib Expression (Mean, RSI)
  ✅ 自定义 Expression (leverage_ratio)
    """)
    
    # 测试
    env = SOXLQlibHierarchicalEnvEnhanced(mode='train')
    obs, _ = env.reset()
    
    print(f"观察空间: {obs.shape}")
    print(f"动作空间: {env.action_space}")
    print()
    print("状态向量（16维）:")
    print(f"  [0-10]  原有特征（SOXL+VIX）")
    print(f"  [11]    SOXX收益率: {obs[11]:.4f}")
    print(f"  [12]    SOXX RSI:   {obs[12]:.4f}")
    print(f"  [13]    杠杆比率:   {obs[13]:.4f}")
    print(f"  [14]    SOXX趋势:   {obs[14]:.4f}")
    print(f"  [15]    基准偏离:   {obs[15]:.4f}")
    print()
    print("✅ 环境创建成功！")
