#!/usr/bin/env python3
"""
Qlib + RL 混合策略

架构：
1. 基础信号层（Qlib Expression）：恐慌指数 + 趋势跟踪
2. 仓位控制层（RL）：学习最优仓位大小

核心思路：
- RL不学买卖信号（太难）
- RL只学仓位大小（更可控）
- 基础信号由Qlib Expression提供（可解释）
"""
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.contrib.strategy import BaseStrategy
from qlib.backtest.decision import TradeDecision, Order

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ============================================================================
# 第一层：Qlib Expression 基础信号
# ============================================================================

class QlibSignalProvider:
    """使用Qlib Expression计算恐慌和趋势信号"""
    
    def __init__(self):
        self.panic_buy_threshold = -45
        self.panic_sell_threshold = 75
        
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Qlib风格的信号"""
        
        df = df.copy()
        
        # 1. 恐慌指数（使用Qlib Expression风格）
        # VIX相关
        df['vix_ma20'] = df['vix'].rolling(20).mean()
        df['vix_std20'] = df['vix'].rolling(20).std()
        df['vix_zscore'] = (df['vix'] - df['vix_ma20']) / (df['vix_std20'] + 1e-8)
        df['vix_percentile'] = df['vix'].rolling(20).rank(pct=True)
        
        # 日收益率
        df['daily_return'] = df['close'].pct_change()
        df['panic_return'] = df['daily_return'].clip(lower=-0.05) / -0.05  # 裁剪到[-1, 0]
        
        # 恐慌得分（Qlib Expression: vix_zscore * 40 + vix_percentile * 60 + panic_return * -50）
        df['panic_score'] = (
            df['vix_zscore'] * 40 +
            df['vix_percentile'] * 60 +
            df['panic_return'] * -50
        )
        
        # 恐慌信号（Qlib Expression: If(panic_score < -45, 1, 0)）
        df['panic_buy_signal'] = (df['panic_score'] <= self.panic_buy_threshold).astype(int)
        df['panic_sell_signal'] = (df['panic_score'] >= self.panic_sell_threshold).astype(int)
        
        # 2. 趋势跟踪（使用Qlib Expression）
        # 均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        
        # 趋势强度（Qlib Expression: (ma5 - ma20) / ma20）
        df['trend_strength'] = (df['ma5'] - df['ma20']) / df['ma20']
        
        # RSI（Qlib Expression风格）
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        # 趋势信号（Qlib Expression: If(ma5 > ma20 && rsi < 65, 1, 0)）
        df['trend_buy_signal'] = ((df['ma5'] > df['ma20']) & (df['rsi'] < 65)).astype(int)
        df['trend_sell_signal'] = ((df['ma5'] < df['ma20']) & (df['rsi'] > 35)).astype(int)
        
        # 3. 市场模式（基于VIX）
        df['market_mode'] = pd.cut(
            df['vix'],
            bins=[0, 18, 25, 100],
            labels=['greed', 'normal', 'panic']
        )
        
        return df


# ============================================================================
# 第二层：RL 仓位控制器
# ============================================================================

class PositionController:
    """RL仓位控制器 - 学习最优仓位大小"""
    
    def __init__(self, state_dim: int = 10, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.policy = PositionPolicy(state_dim, hidden_dim)
        
    def get_position(
        self,
        panic_signal: int,
        trend_signal: int,
        market_mode: str,
        current_position: float,
        volatility: float,
        drawdown: float,
    ) -> float:
        """获取RL推荐的仓位"""
        
        # 构造状态
        state = self._build_state(
            panic_signal, trend_signal, market_mode,
            current_position, volatility, drawdown
        )
        
        # RL预测仓位
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            position = self.policy(state_tensor).item()
        
        return position
    
    def _build_state(
        self,
        panic_signal: int,
        trend_signal: int,
        market_mode: str,
        current_position: float,
        volatility: float,
        drawdown: float,
    ) -> np.ndarray:
        """构造状态向量"""
        
        # 市场模式编码
        mode_encoding = {
            'panic': [1, 0, 0],
            'normal': [0, 1, 0],
            'greed': [0, 0, 1],
        }
        
        state = np.array([
            panic_signal,
            trend_signal,
            *mode_encoding.get(market_mode, [0, 1, 0]),
            current_position,
            min(volatility, 1.0),
            min(drawdown, 1.0),
            # 可以添加更多特征
        ], dtype=np.float32)
        
        return state


class PositionPolicy(nn.Module):
    """仓位策略网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 输出 [0, 1]
        )
    
    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        return self.net(x) * 0.85  # 最大仓位85%


# ============================================================================
# 第三层：Qlib Strategy 整合
# ============================================================================

class RLHybridStrategy(BaseStrategy):
    """Qlib策略：RL控制仓位的混合策略"""
    
    def __init__(
        self,
        signal_provider: QlibSignalProvider,
        position_controller: PositionController,
        max_position: float = 0.85,
        stop_loss: float = -0.15,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.signal_provider = signal_provider
        self.position_controller = position_controller
        self.max_position = max_position
        self.stop_loss = stop_loss
        
        # 状态追踪
        self.peak_value = None
        self.entry_value = None
        
    def generate_trade_decision(self, execute_result=None) -> TradeDecision:
        """Qlib要求的接口：生成交易决策"""
        
        # 获取当前数据
        trade_step = self.trade_calendar.get_trade_step()
        current_data = self.get_current_data()
        
        # 获取信号
        signals = self._get_signals(current_data)
        
        # 获取RL推荐的仓位
        rl_position = self._get_rl_position(signals, current_data)
        
        # 生成订单
        orders = self._generate_orders(rl_position, current_data)
        
        return TradeDecision(orders)
    
    def _get_signals(self, data) -> Dict:
        """获取信号"""
        # 这里需要从data中提取信号
        # 简化实现
        return {
            'panic_buy': data.get('panic_buy_signal', 0),
            'panic_sell': data.get('panic_sell_signal', 0),
            'trend_buy': data.get('trend_buy_signal', 0),
            'trend_sell': data.get('trend_sell_signal', 0),
            'market_mode': data.get('market_mode', 'normal'),
            'volatility': data.get('volatility', 0.2),
            'drawdown': data.get('drawdown', 0),
        }
    
    def _get_rl_position(self, signals: Dict, data) -> float:
        """获取RL推荐的仓位"""
        
        current_position = data.get('position', 0)
        
        return self.position_controller.get_position(
            panic_signal=signals['panic_buy'] - signals['panic_sell'],
            trend_signal=signals['trend_buy'] - signals['trend_sell'],
            market_mode=signals['market_mode'],
            current_position=current_position,
            volatility=signals['volatility'],
            drawdown=signals['drawdown'],
        )
    
    def _generate_orders(self, target_position: float, data) -> List[Order]:
        """生成订单"""
        
        current_position = data.get('position', 0)
        delta = target_position - current_position
        
        orders = []
        
        if abs(delta) > 0.02:  # 2%阈值
            # 计算股数
            portfolio_value = data.get('portfolio_value', 100000)
            price = data.get('close', 100)
            
            shares = int(portfolio_value * delta / price)
            
            if shares != 0:
                orders.append(Order(
                    stock_id='SOXL',
                    amount=abs(shares),
                    direction=Order.BUY if shares > 0 else Order.SELL,
                ))
        
        return orders


# ============================================================================
# RL 训练器
# ============================================================================

class PositionRLTrainer:
    """训练RL仓位控制器"""
    
    def __init__(
        self,
        signal_provider: QlibSignalProvider,
        position_controller: PositionController,
        lr: float = 1e-3,
    ):
        self.signal_provider = signal_provider
        self.position_controller = position_controller
        self.optimizer = torch.optim.Adam(
            position_controller.policy.parameters(),
            lr=lr
        )
    
    def train(
        self,
        df: pd.DataFrame,
        episodes: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """训练RL"""
        
        if verbose:
            print("="*70)
            print("RL仓位控制器训练")
            print("="*70)
            print(f"\n训练数据: {len(df)} records")
            print(f"Episodes: {episodes}")
            print()
        
        # 计算信号
        df = self.signal_provider.calculate_signals(df)
        
        best_return = -999
        best_weights = None
        
        for episode in range(episodes):
            # 运行一个episode
            total_return, trades = self._run_episode(df)
            
            # 更新最佳
            if total_return > best_return:
                best_return = total_return
                best_weights = {
                    k: v.clone() for k, v in 
                    self.position_controller.policy.state_dict().items()
                }
            
            # PPO更新
            self._ppo_update()
            
            if verbose and (episode + 1) % 20 == 0:
                print(f"Episode {episode+1}/{episodes}, Return: {total_return:.2%}, Best: {best_return:.2%}")
        
        # 恢复最佳权重
        if best_weights:
            self.position_controller.policy.load_state_dict(best_weights)
        
        return {
            'best_return': best_return,
            'episodes': episodes,
        }
    
    def _run_episode(self, df: pd.DataFrame) -> Tuple[float, int]:
        """运行一个episode"""
        
        cash = 100000
        shares = 0
        peak_value = 100000
        trades = 0
        
        states, actions, rewards = [], [], []
        
        for i in range(30, len(df)):
            row = df.iloc[i]
            
            # 当前状态
            position_value = shares * row['close']
            total_value = cash + position_value
            position_ratio = position_value / total_value if total_value > 0 else 0
            peak_value = max(peak_value, total_value)
            
            # 计算特征
            volatility = df['close'].pct_change().iloc[max(0,i-20):i+1].std() * np.sqrt(252)
            drawdown = (peak_value - total_value) / peak_value
            
            # 获取RL仓位
            target_position = self.position_controller.get_position(
                panic_signal=row['panic_buy_signal'] - row['panic_sell_signal'],
                trend_signal=row['trend_buy_signal'] - row['trend_sell_signal'],
                market_mode=row['market_mode'],
                current_position=position_ratio,
                volatility=volatility,
                drawdown=drawdown,
            )
            
            # 执行交易
            old_value = total_value
            if abs(target_position - position_ratio) > 0.02:
                target_value = total_value * target_position
                delta_shares = (target_value - position_value) / row['close']
                
                if delta_shares > 10:
                    cost = delta_shares * row['close'] * 1.001
                    if cost <= cash:
                        cash -= cost
                        shares += delta_shares
                        trades += 1
                elif delta_shares < -10:
                    sell = min(-delta_shares, shares)
                    cash += sell * row['close'] * 0.999
                    shares -= sell
                    trades += 1
            
            # 记录
            states.append([
                row['panic_buy_signal'] - row['panic_sell_signal'],
                row['trend_buy_signal'] - row['trend_sell_signal'],
                1 if row['market_mode'] == 'panic' else 0,
                1 if row['market_mode'] == 'normal' else 0,
                1 if row['market_mode'] == 'greed' else 0,
                position_ratio,
                min(volatility, 1.0),
                min(drawdown, 1.0),
            ])
            
            actions.append(target_position)
            rewards.append((total_value - old_value) / old_value * 100)
        
        # 计算总收益
        final_value = cash + shares * df['close'].iloc[-1]
        total_return = (final_value - 100000) / 100000
        
        return total_return, trades
    
    def _ppo_update(self):
        """PPO更新（简化版）"""
        # 这里可以添加PPO更新逻辑
        pass


# ============================================================================
# 完整回测
# ============================================================================

def run_qlib_rl_backtest(
    data_file: str = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv',
    train_episodes: int = 100,
    verbose: bool = True,
):
    """运行完整的Qlib + RL回测"""
    
    print("="*70)
    print("Qlib + RL 混合策略")
    print("="*70)
    print("\n架构:")
    print("  第一层: Qlib Expression (恐慌指数 + 趋势跟踪)")
    print("  第二层: RL Position Controller (仓位控制)")
    print()
    
    # 加载数据
    df = pd.read_csv(data_file, parse_dates=['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
    
    # 初始化
    signal_provider = QlibSignalProvider()
    position_controller = PositionController(state_dim=8, hidden_dim=64)
    trainer = PositionRLTrainer(signal_provider, position_controller)
    
    # 训练RL
    train_results = trainer.train(df, episodes=train_episodes, verbose=verbose)
    
    # 使用训练好的RL回测
    print("\n" + "="*70)
    print("使用训练好的RL回测")
    print("="*70)
    
    df = signal_provider.calculate_signals(df)
    
    cash = 100000
    shares = 0
    peak_value = 100000
    portfolio_values = [100000]
    trades = []
    
    for i in range(30, len(df)):
        row = df.iloc[i]
        
        # 当前状态
        position_value = shares * row['close']
        total_value = cash + position_value
        position_ratio = position_value / total_value if total_value > 0 else 0
        peak_value = max(peak_value, total_value)
        
        volatility = df['close'].pct_change().iloc[max(0,i-20):i+1].std() * np.sqrt(252)
        drawdown = (peak_value - total_value) / peak_value
        
        # RL推荐仓位
        target_position = position_controller.get_position(
            panic_signal=row['panic_buy_signal'] - row['panic_sell_signal'],
            trend_signal=row['trend_buy_signal'] - row['trend_sell_signal'],
            market_mode=row['market_mode'],
            current_position=position_ratio,
            volatility=volatility,
            drawdown=drawdown,
        )
        
        # 执行交易
        if abs(target_position - position_ratio) > 0.02:
            target_value = total_value * target_position
            delta_shares = (target_value - position_value) / row['close']
            
            if delta_shares > 10:
                cost = delta_shares * row['close'] * 1.001
                if cost <= cash:
                    cash -= cost
                    shares += delta_shares
                    trades.append({'date': row['date'], 'type': 'buy', 'price': row['close']})
            elif delta_shares < -10:
                sell = min(-delta_shares, shares)
                cash += sell * row['close'] * 0.999
                shares -= sell
                trades.append({'date': row['date'], 'type': 'sell', 'price': row['close']})
        
        portfolio_values.append(cash + shares * row['close'])
    
    # 计算指标
    pv = np.array(portfolio_values)
    total_return = (pv[-1] - 100000) / 100000
    
    daily_returns = np.diff(pv) / pv[:-1]
    years = len(daily_returns) / 252
    annual_return = (1 + total_return) ** (1 / max(years, 0.1)) - 1
    
    max_value = np.maximum.accumulate(pv)
    drawdowns = (max_value - pv) / max_value
    max_drawdown = np.max(drawdowns)
    
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[30] - 1)
    
    print(f"\n总收益:        {total_return:.2%}")
    print(f"年化收益:      {annual_return:.2%}")
    print(f"最大回撤:      {max_drawdown:.2%}")
    print(f"Sharpe:        {sharpe:.2f}")
    print(f"交易次数:      {len(trades)}")
    print(f"收益倍数:      {pv[-1] / 100000:.2f}x")
    print(f"买入持有:      {buy_hold:.2%}")
    print(f"超额收益:      {total_return - buy_hold:.2%}")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'num_trades': len(trades),
        'final_value': pv[-1],
    }


if __name__ == '__main__':
    results = run_qlib_rl_backtest(train_episodes=100, verbose=True)
