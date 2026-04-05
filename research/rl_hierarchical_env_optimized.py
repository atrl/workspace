#!/usr/bin/env python3
"""
分层RL环境 - 优化版本

关键优化：
1. 调整奖励函数：降低回撤惩罚（50→15）
2. 限制做空范围：最多做空30%
3. 增加持仓奖励：鼓励合理持仓
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces


class SOXLHierarchicalEnvOptimized(gym.Env):
    """
    SOXL 分层RL交易环境 - 优化版
    
    优化：
    1. 奖励函数调整
    2. 限制做空
    3. 持仓奖励
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data_path: str = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv',
        initial_balance: float = 100000,
        commission: float = 0.001,
        lookback: int = 20,
        # 规则策略参数
        panic_threshold: float = -35.2,
        vix_panic: float = 20.4,
        vix_extreme: float = 28.5,
        vix_greed: float = 12.7,
        # 优化参数
        max_short: float = 0.3,  # 最大做空比例
    ):
        super().__init__()
        
        # 加载数据
        self.df = pd.read_csv(data_path, parse_dates=['date'])
        self.df = self.df[self.df['date'] >= '2020-01-01'].reset_index(drop=True)
        
        # 计算特征
        self._calculate_features()
        
        # 环境参数
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback = lookback
        
        # 规则策略参数
        self.panic_threshold = panic_threshold
        self.vix_panic = vix_panic
        self.vix_extreme = vix_extreme
        self.vix_greed = vix_greed
        
        # 优化参数
        self.max_short = max_short
        
        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # 观察空间
        self.feature_cols = [
            'vix_normalized',
            'vix_zscore',
            'return_1',
            'rsi_14',
            'panic_score',
            'current_position',
            'portfolio_return',
            'volatility',
            'ma_ratio',
            'volume_ratio',
            'rule_signal',
        ]
        
        n_features = len(self.feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        self.reset()
    
    def _calculate_features(self):
        """计算特征"""
        df = self.df
        
        df['return_1'] = df['close'].pct_change()
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['ma_ratio'] = df['MA5'] / df['MA20']
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        df['vix_normalized'] = df['vix'] / 50
        df['vix_ma20'] = df['vix'].rolling(20).mean()
        df['vix_std20'] = df['vix'].rolling(20).std()
        df['vix_zscore'] = (df['vix'] - df['vix_ma20']) / (df['vix_std20'] + 1e-8)
        
        df['panic_score'] = (
            df['vix_zscore'] * 40 +
            df['vix'].rolling(20).rank(pct=True) * 60 +
            np.where(df['return_1'] < 0, np.minimum(df['return_1'] / -0.05, 1) * -50, 0)
        )
        
        self.df = df.dropna().reset_index(drop=True)
    
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
        """获取当前状态"""
        row = self.df.iloc[self.current_step]
        
        portfolio_return = (self.total_value - self.initial_balance) / self.initial_balance
        
        if len(self.history) > 10:
            hist_array = np.array(self.history[-20:])
            returns = np.diff(hist_array) / hist_array[:-1]
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        else:
            volatility = 0
        
        rule_signal = self._get_rule_signal()
        
        obs = np.array([
            row['vix_normalized'],
            row['vix_zscore'],
            row['return_1'],
            row['rsi_14'] / 100,
            row['panic_score'] / 100,
            self.position,
            portfolio_return,
            volatility,
            row['ma_ratio'],
            row['volume_ratio'],
            rule_signal,
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _get_rule_signal(self) -> float:
        """规则策略信号"""
        row = self.df.iloc[self.current_step]
        
        if row['panic_score'] <= self.panic_threshold:
            if row['vix'] > self.vix_extreme:
                return 1.0
            elif row['vix'] > self.vix_panic:
                return 0.9
        
        if row['rsi_14'] < 65 and row['MA5'] > row['MA20']:
            return 0.6
        
        if row['vix'] < self.vix_greed:
            if row['rsi_14'] > 65:
                return 0.2
            return 0.4
        
        return 0.5
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        current_price = float(self.df['close'].iloc[self.current_step])
        
        # 解析RL动作
        short_ratio = float(action[0])
        long_ratio = float(action[1])
        rl_position = long_ratio - short_ratio
        
        # ============================================================
        # 优化1：限制做空范围
        # ============================================================
        rl_position = np.clip(rl_position, -self.max_short, 1.0)
        
        # 规则信号
        rule_signal = self._get_rule_signal()
        
        # 最终仓位（分层）
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
            next_price = float(self.df['close'].iloc[self.current_step])
            price_return = (next_price - current_price) / current_price
            
            if self.position >= 0:
                portfolio_return = self.position * price_return - commission_cost / self.total_value
            else:
                portfolio_return = -self.position * (-price_return) - commission_cost / self.total_value
            
            self.returns.append(portfolio_return)
            self.total_value *= (1 + portfolio_return)
            self.history.append(self.total_value)
            
            # ============================================================
            # 优化2：调整奖励函数
            # ============================================================
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
        """
        优化的奖励函数
        
        关键调整：
        1. 降低回撤惩罚：50 → 15
        2. 降低波动惩罚：20 → 5
        3. 增加持仓奖励：鼓励合理持仓
        """
        # 1. 收益奖励（提高权重）
        return_reward = portfolio_return * 150  # 从100提高到150
        
        # 2. Sharpe奖励
        if len(self.returns) > 10:
            mean_return = np.mean(self.returns[-20:])
            std_return = np.std(self.returns[-20:]) + 1e-8
            sharpe_reward = mean_return / std_return * 15  # 从10提高到15
        else:
            sharpe_reward = 0
        
        # 3. 回撤惩罚（大幅降低）
        if len(self.history) > 0:
            peak = max(self.history)
            drawdown = (peak - self.total_value) / peak
            # 优化：从50降到15，且只在回撤>15%时惩罚
            if drawdown > 0.15:
                dd_penalty = -(drawdown - 0.15) * 30  # 之前：-drawdown * 50
            else:
                dd_penalty = 0
        else:
            dd_penalty = 0
        
        # 4. 波动惩罚（大幅降低）
        if len(self.returns) > 5:
            recent_vol = np.std(self.returns[-10:])
            vol_penalty = -recent_vol * 5  # 从20降到5
        else:
            vol_penalty = 0
        
        # 5. 新增：持仓奖励（鼓励合理持仓）
        if 0.3 <= abs(self.position) <= 0.8:
            # 仓位在合理范围内，给予奖励
            position_reward = 2.0
        else:
            position_reward = 0
        
        # 总奖励
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
    print("分层RL环境 - 优化版测试")
    print("="*70)
    print("""
优化点：
  1. 奖励函数调整
     - 回撤惩罚：50 → 15（且仅>15%时惩罚）
     - 波动惩罚：20 → 5
     - 收益奖励：100 → 150
     - 新增持仓奖励
  
  2. 限制做空
     - 最大做空：30%（之前无限制）
    """)
    
    env = SOXLHierarchicalEnvOptimized()
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Max short: {env.max_short}")
    print()
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  RL Position: {info['rl_position']:.2f} (limited to -0.3)")
        print(f"  Final Pos:   {info['position']:.2f}")
        print(f"  Reward:      {reward:.4f}")
        print()
        
        if done:
            break
