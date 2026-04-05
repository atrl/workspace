#!/usr/bin/env python3
"""
混合策略：恐慌指数 + 趋势跟踪

核心思路：
1. VIX > 25: 恐慌模式 → 恐慌指数策略（逆向抄底）
2. 18 < VIX < 25: 正常模式 → 趋势跟踪（顺势而为）
3. VIX < 18: 贪婪模式 → 趋势跟随（吃趋势）
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Trade:
    type: str
    date: datetime
    price: float
    shares: int
    position: float
    reason: str
    strategy: str


class HybridStrategy:
    """混合策略：恐慌指数 + 趋势跟踪"""
    
    def __init__(
        self,
        panic_buy_threshold: float = -50,
        panic_sell_threshold: float = 80,
        panic_max_position: float = 0.70,
        trend_ma_fast: int = 5,
        trend_ma_slow: int = 20,
        trend_max_position: float = 0.50,
        vix_high: float = 25.0,
        vix_low: float = 18.0,
        max_total_position: float = 0.75,
        stop_loss: float = -0.15,
    ):
        self.panic_buy_threshold = panic_buy_threshold
        self.panic_sell_threshold = panic_sell_threshold
        self.panic_max_position = panic_max_position
        self.trend_ma_fast = trend_ma_fast
        self.trend_ma_slow = trend_ma_slow
        self.trend_max_position = trend_max_position
        self.vix_high = vix_high
        self.vix_low = vix_low
        self.max_total_position = max_total_position
        self.stop_loss = stop_loss
    
    def calculate_panic_score(self, df: pd.DataFrame, i: int) -> float:
        """计算恐慌得分"""
        if i < 20:
            return 0
        
        vix = df['vix'].iloc[i]
        daily_ret = (df['close'].iloc[i] / df['close'].iloc[i-1] - 1) if i > 0 else 0
        
        vix_rolling = df['vix'].iloc[max(0,i-20):i+1]
        vix_mean = vix_rolling.mean()
        vix_std = vix_rolling.std() + 1e-8
        vix_zscore = (vix - vix_mean) / vix_std
        
        vix_percentile = vix_rolling.rank(pct=True).iloc[-1]
        
        panic_score = (
            vix_zscore * 40 +
            vix_percentile * 60 +
            min(daily_ret / -0.05, 1) * -50 if daily_ret < 0 else 0
        )
        
        return panic_score
    
    def calculate_trend_signal(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """计算趋势信号"""
        if i < max(self.trend_ma_fast, self.trend_ma_slow):
            return 'hold', 0
        
        close = df['close'].iloc[i]
        ma_fast = df['close'].iloc[i-self.trend_ma_fast+1:i+1].mean()
        ma_slow = df['close'].iloc[i-self.trend_ma_slow+1:i+1].mean()
        
        # 趋势强度
        trend_strength = (ma_fast - ma_slow) / ma_slow
        
        # RSI
        delta = df['close'].diff().iloc[max(0,i-14):i+1]
        gain = delta.where(delta > 0, 0).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # 综合信号
        if ma_fast > ma_slow and rsi < 70:
            return 'buy', min(trend_strength * 10, 1.0)
        elif ma_fast < ma_slow and rsi > 30:
            return 'sell', min(abs(trend_strength) * 10, 1.0)
        else:
            return 'hold', 0
    
    def determine_market_mode(self, vix: float) -> str:
        """判断市场模式"""
        if vix > self.vix_high:
            return 'panic'
        elif vix < self.vix_low:
            return 'greed'
        else:
            return 'normal'
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """运行混合策略回测"""
        
        if verbose:
            print("="*70)
            print("混合策略回测：恐慌指数 + 趋势跟踪")
            print("="*70)
            print(f"\n参数配置:")
            print(f"  恐慌指数: 买入阈值={self.panic_buy_threshold}, 最大仓位={self.panic_max_position:.0%}")
            print(f"  趋势跟踪: MA{self.trend_ma_fast}/{self.trend_ma_slow}, 最大仓位={self.trend_max_position:.0%}")
            print(f"  VIX阈值: 恐慌>{self.vix_high}, 贪婪<{self.vix_low}")
            print(f"  总仓位限制: {self.max_total_position:.0%}")
            print(f"  止损: {self.stop_loss:.0%}\n")
        
        cash = 100000
        shares = 0
        avg_price = 0
        peak_value = 100000
        
        trades = []
        portfolio_values = [100000]
        
        panic_trades = 0
        trend_trades = 0
        
        for i in range(30, len(df)):
            row = df.iloc[i]
            price = row['close']
            vix = row.get('vix', 20)
            
            # 当前状态
            position_value = shares * price
            total_value = cash + position_value
            position_ratio = position_value / total_value if total_value > 0 else 0
            peak_value = max(peak_value, total_value)
            
            total_return = (total_value - 100000) / 100000
            
            # 风控检查
            if total_return < self.stop_loss and shares > 0:
                cash += shares * price * 0.999
                trades.append(Trade('sell', row['date'], price, int(shares), 0, 
                                   f'止损({total_return:.1%})', 'risk'))
                shares = 0
                avg_price = 0
                peak_value = total_value
                portfolio_values.append(cash)
                continue
            
            # 判断市场模式
            mode = self.determine_market_mode(vix)
            
            # 计算信号
            panic_score = self.calculate_panic_score(df, i)
            trend_signal, trend_strength = self.calculate_trend_signal(df, i)
            
            # 根据模式选择策略
            target_position = position_ratio
            strategy = 'hold'
            
            if mode == 'panic':
                # 恐慌模式：逆向抄底
                if panic_score <= self.panic_buy_threshold and position_ratio < self.panic_max_position:
                    if position_ratio < 0.3:
                        target_position = 0.3
                    elif position_ratio < 0.5:
                        target_position = 0.5
                    else:
                        target_position = self.panic_max_position
                    strategy = 'panic'
                
                elif panic_score >= self.panic_sell_threshold and position_ratio > 0:
                    target_position = max(0, position_ratio - 0.3)
                    strategy = 'panic'
            
            else:
                # 正常/贪婪模式：趋势跟踪
                if trend_signal == 'buy' and position_ratio < self.trend_max_position:
                    target_position = min(position_ratio + trend_strength * 0.2, self.trend_max_position)
                    strategy = 'trend'
                
                elif trend_signal == 'sell' and position_ratio > 0:
                    target_position = max(0, position_ratio - trend_strength * 0.3)
                    strategy = 'trend'
                    
                    # 趋势止损
                    if avg_price > 0 and (price - avg_price) / avg_price < -0.05:
                        target_position = 0
            
            # 总仓位限制
            target_position = min(target_position, self.max_total_position)
            
            # 执行交易
            if abs(target_position - position_ratio) > 0.02:
                target_value = total_value * target_position
                delta_shares = (target_value - position_value) / price
                
                if delta_shares > 10:
                    cost = delta_shares * price * 1.001
                    if cost <= cash:
                        total_cost = avg_price * shares + price * delta_shares
                        shares += delta_shares
                        avg_price = total_cost / shares
                        cash -= cost
                        
                        trades.append(Trade('buy', row['date'], price, int(delta_shares),
                                          target_position, f'{mode}模式', strategy))
                        
                        if strategy == 'panic':
                            panic_trades += 1
                        elif strategy == 'trend':
                            trend_trades += 1
                
                elif delta_shares < -10:
                    sell = min(-delta_shares, shares)
                    cash += sell * price * 0.999
                    shares -= sell
                    
                    trades.append(Trade('sell', row['date'], price, int(sell),
                                      target_position, f'{mode}模式', strategy))
                    
                    if strategy == 'panic':
                        panic_trades += 1
                    elif strategy == 'trend':
                        trend_trades += 1
            
            portfolio_values.append(cash + shares * price)
        
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
        calmar = annual_return / max_drawdown if max_drawdown > 0 else sharpe
        
        buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[30] - 1)
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe': sharpe,
            'calmar': calmar,
            'num_trades': len(trades),
            'panic_trades': panic_trades,
            'trend_trades': trend_trades,
            'final_value': pv[-1],
            'buy_hold_return': buy_hold_return,
            'trades': trades,
        }
        
        if verbose:
            print("\n" + "="*70)
            print("回测结果")
            print("="*70)
            print(f"总收益:        {total_return:.2%}")
            print(f"年化收益:      {annual_return:.2%}")
            print(f"最大回撤:      {max_drawdown:.2%}")
            print(f"波动率:        {volatility:.2%}")
            print(f"Sharpe:        {sharpe:.2f}")
            print(f"Calmar:        {calmar:.2f}")
            print("-"*70)
            print(f"交易次数:      {len(trades)}")
            print(f"  恐慌指数:    {panic_trades}")
            print(f"  趋势跟踪:    {trend_trades}")
            print("-"*70)
            print(f"最终价值:      ${pv[-1]:,.0f}")
            print(f"收益倍数:      {pv[-1] / 100000:.2f}x")
            print("-"*70)
            print(f"买入持有:      {buy_hold_return:.2%}")
            print(f"超额收益:      {total_return - buy_hold_return:.2%}")
        
        return results


def optimize_hybrid_params(df: pd.DataFrame) -> Dict:
    """参数优化"""
    
    print("="*70)
    print("参数优化：混合策略")
    print("="*70)
    
    best_result = {'total_return': -999}
    
    from itertools import product
    
    key_params = {
        'panic_buy_threshold': [-40, -50],
        'panic_max_position': [0.60, 0.70],
        'trend_max_position': [0.40, 0.50],
        'vix_high': [23, 25],
        'vix_low': [16, 18],
        'max_total_position': [0.70, 0.75],
        'stop_loss': [-0.10, -0.12],
    }
    
    total = 1
    for v in key_params.values():
        total *= len(v)
    
    print(f"搜索空间: {total} 组合\n")
    
    count = 0
    for params in product(*key_params.values()):
        count += 1
        
        strategy = HybridStrategy(
            panic_buy_threshold=params[0],
            panic_max_position=params[1],
            trend_max_position=params[2],
            vix_high=params[3],
            vix_low=params[4],
            max_total_position=params[5],
            stop_loss=params[6],
        )
        
        results = strategy.run_backtest(df, verbose=False)
        
        # 综合评分：收益 + Sharpe - 回撤
        score = results['total_return'] + results['sharpe'] * 0.5 - results['max_drawdown'] * 0.5
        
        if score > best_result.get('score', -999):
            best_result = {
                'total_return': results['total_return'],
                'sharpe': results['sharpe'],
                'max_drawdown': results['max_drawdown'],
                'num_trades': results['num_trades'],
                'panic_trades': results['panic_trades'],
                'trend_trades': results['trend_trades'],
                'score': score,
                'params': dict(zip(key_params.keys(), params)),
            }
        
        if count % 20 == 0:
            print(f"进度: {count}/{total} ({count/total*100:.0f}%), 当前最优: {best_result['total_return']:.2%}")
    
    print("\n" + "="*70)
    print("最优参数")
    print("="*70)
    for k, v in best_result['params'].items():
        print(f"  {k}: {v}")
    print(f"\n最优结果:")
    print(f"  总收益: {best_result['total_return']:.2%}")
    print(f"  Sharpe: {best_result['sharpe']:.2f}")
    print(f"  最大回撤: {best_result['max_drawdown']:.2%}")
    print(f"  交易次数: {best_result['num_trades']} (恐慌{best_result['panic_trades']}, 趋势{best_result['trend_trades']})")
    
    return best_result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv')
    parser.add_argument('--optimize', action='store_true')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.data, parse_dates=['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
    
    if args.optimize:
        best = optimize_hybrid_params(df)
        print("\n\n使用最优参数运行回测:")
        strategy = HybridStrategy(**best['params'])
        results = strategy.run_backtest(df, verbose=True)
    else:
        strategy = HybridStrategy()
        results = strategy.run_backtest(df, verbose=True)
