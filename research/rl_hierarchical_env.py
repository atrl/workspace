#!/usr/bin/env python3
"""
分层RL交易环境 - 参考之前的设计

核心改进：
1. ✅ 使用之前的动作空间设计 [short_ratio, long_ratio]
2. ✅ 支持做空（仓位 -1 到 1）
3. ✅ 分层架构：规则信号 × RL仓位
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces


class SOXLHierarchicalTradingEnv(gym.Env):
    """
    SOXL 分层RL交易环境
    
    架构：
    - 规则策略：提供买卖信号 (基于恐慌指数)
    - RL智能体：调整仓位大小 [-1, 1]
    - 最终仓位 = 规则信号 × RL仓位
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data_path: str = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv',
        initial_balance: float = 100000,
        commission: float = 0.001,
        lookback: int = 20,
        # 规则策略参数（硬参数）
        panic_threshold: float = -35.2,
        vix_panic: float = 20.4,
        vix_extreme: float = 28.5,
        vix_greed: float = 12.7,
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
        
        # ============================================================
        # 动作空间（参考之前设计）
        # ============================================================
        # [short_ratio, long_ratio] ∈ [0, 1] × [0, 1]
        # 最终仓位 = long_ratio - short_ratio ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # ============================================================
        # 观察空间
        # ============================================================
        # 状态特征：11维
        self.feature_cols = [
            'vix_normalized',       # VIX归一化
            'vix_zscore',           # VIX z-score
            'return_1',             # 1日收益率
            'rsi_14',               # RSI
            'panic_score',          # 恐慌得分
            'current_position',     # 当前仓位
            'portfolio_return',     # 组合收益
            'volatility',           # 波动率
            'ma_ratio',             # 均线比率
            'volume_ratio',         # 成交量比率
            'rule_signal',          # 规则信号（新增）
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
        
        # 收益率
        df['return_1'] = df['close'].pct_change()
        
        # 均线
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['ma_ratio'] = df['MA5'] / df['MA20']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        # 成交量
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        # VIX 特征
        df['vix_normalized'] = df['vix'] / 50
        df['vix_ma20'] = df['vix'].rolling(20).mean()
        df['vix_std20'] = df['vix'].rolling(20).std()
        df['vix_zscore'] = (df['vix'] - df['vix_ma20']) / (df['vix_std20'] + 1e-8)
        
        # 恐慌得分
        df['panic_score'] = (
            df['vix_zscore'] * 40 +
            df['vix'].rolling(20).rank(pct=True) * 60 +
            np.where(df['return_1'] < 0, np.minimum(df['return_1'] / -0.05, 1) * -50, 0)
        )
        
        # 删除 NaN
        self.df = df.dropna().reset_index(drop=True)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0.0  # -1 到 1
        self.shares = 0
        self.current_step = self.lookback
        self.total_value = self.initial_balance
        self.returns = []
        self.history = [self.initial_balance]
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """获取当前状态"""
        row = self.df.iloc[self.current_step]
        
        # 计算组合状态
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
        
        # 状态向量
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
        
        # 处理 NaN
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _get_rule_signal(self) -> float:
        """
        规则策略信号
        
        返回：
        - 1.0: 强烈买入（极端恐慌）
        - 0.5: 中性
        - 0.0: 强烈卖出（贪婪）
        """
        row = self.df.iloc[self.current_step]
        
        # 恐慌信号
        if row['panic_score'] <= self.panic_threshold:
            if row['vix'] > self.vix_extreme:
                return 1.0  # 极端恐慌
            elif row['vix'] > self.vix_panic:
                return 0.9  # 恐慌
        
        # 趋势信号
        if row['rsi_14'] < 65 and row['MA5'] > row['MA20']:
            return 0.6  # 上升趋势
        
        # 贪婪信号
        if row['vix'] < self.vix_greed:
            if row['rsi_14'] > 65:
                return 0.2  # 超买
            return 0.4
        
        # 默认
        return 0.5
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作
        
        动作：[short_ratio, long_ratio] ∈ [0, 1] × [0, 1]
        RL仓位：long_ratio - short_ratio ∈ [-1, 1]
        最终仓位：规则信号 × RL仓位
        """
        current_price = float(self.df['close'].iloc[self.current_step])
        
        # ============================================================
        # 1. 解析RL动作
        # ============================================================
        short_ratio = float(action[0])
        long_ratio = float(action[1])
        rl_position = long_ratio - short_ratio
        rl_position = np.clip(rl_position, -1.0, 1.0)
        
        # ============================================================
        # 2. 获取规则信号
        # ============================================================
        rule_signal = self._get_rule_signal()
        
        # ============================================================
        # 3. 计算最终仓位（分层架构）
        # ============================================================
        # 最终仓位 = 规则信号 × RL仓位
        # 例如：规则信号 1.0 × RL仓位 0.8 = 0.8
        #      规则信号 0.3 × RL仓位 1.0 = 0.3
        target_position = rule_signal * rl_position
        target_position = np.clip(target_position, -1.0, 1.0)
        
        # ============================================================
        # 4. 执行交易
        # ============================================================
        position_change = target_position - self.position
        
        # 交易成本
        trade_value = abs(position_change) * self.total_value
        commission_cost = trade_value * self.commission
        
        # 更新仓位
        self.position = target_position
        self.shares = (self.total_value * abs(self.position)) / current_price
        
        # ============================================================
        # 5. 前进一步
        # ============================================================
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # ============================================================
        # 6. 计算收益
        # ============================================================
        if not done:
            next_price = float(self.df['close'].iloc[self.current_step])
            price_return = (next_price - current_price) / current_price
            
            # 组合收益（考虑做空）
            if self.position >= 0:
                # 多头
                portfolio_return = self.position * price_return - commission_cost / self.total_value
            else:
                # 空头
                portfolio_return = -self.position * (-price_return) - commission_cost / self.total_value
            
            self.returns.append(portfolio_return)
            
            # 更新总价值
            self.total_value *= (1 + portfolio_return)
            self.history.append(self.total_value)
            
            # ============================================================
            # 7. 计算奖励
            # ============================================================
            reward = self._calculate_reward(portfolio_return)
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
    
    def _calculate_reward(self, portfolio_return: float) -> float:
        """
        计算奖励（增强版）
        
        奖励 = 收益奖励 + Sharpe奖励 + 回撤惩罚 + 波动惩罚
        """
        # 1. 收益奖励
        return_reward = portfolio_return * 100
        
        # 2. Sharpe-like 奖励
        if len(self.returns) > 10:
            mean_return = np.mean(self.returns[-20:])
            std_return = np.std(self.returns[-20:]) + 1e-8
            sharpe_reward = mean_return / std_return * 10
        else:
            sharpe_reward = 0
        
        # 3. 回撤惩罚
        if len(self.history) > 0:
            peak = max(self.history)
            drawdown = (peak - self.total_value) / peak
            dd_penalty = -drawdown * 50 if drawdown > 0.1 else 0
        else:
            dd_penalty = 0
        
        # 4. 波动惩罚
        if len(self.returns) > 5:
            recent_vol = np.std(self.returns[-10:])
            vol_penalty = -recent_vol * 20
        else:
            vol_penalty = 0
        
        # 总奖励
        reward = return_reward + sharpe_reward + dd_penalty + vol_penalty
        
        return reward
    
    def render(self):
        rule_signal = self._get_rule_signal()
        print(f"Step {self.current_step}: Value=${self.total_value:.2f}, "
              f"Pos={self.position:.2f}, Rule={rule_signal:.2f}")


# ============================================================================
# 测试环境
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("分层RL交易环境测试")
    print("="*70)
    print("""
架构：
  规则策略 (信号) × RL智能体 (仓位) = 最终仓位
  
动作空间（参考之前设计）：
  [short_ratio, long_ratio] ∈ [0, 1] × [0, 1]
  RL仓位 = long_ratio - short_ratio ∈ [-1, 1]
  
状态空间（11维）：
  vix, vix_zscore, return, rsi, panic_score,
  current_pos, portfolio_return, volatility,
  ma_ratio, volume_ratio, rule_signal
    """)
    
    env = SOXLHierarchicalTradingEnv()
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print()
    
    # 随机测试
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Action:      short={action[0]:.2f}, long={action[1]:.2f}")
        print(f"  RL Position: {info['rl_position']:.2f}")
        print(f"  Rule Signal: {info['rule_signal']:.2f}")
        print(f"  Final Pos:   {info['position']:.2f}")
        print(f"  Reward:      {reward:.4f}")
        print(f"  Value:       ${info['total_value']:.2f}")
        print()
        
        if done:
            break
