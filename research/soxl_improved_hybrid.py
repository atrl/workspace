#!/usr/bin/env python3
"""
改进版混合策略：简单 + 有效

核心思路：
1. 恐慌模式(VIX>25): 70%仓位，恐慌指数策略
2. 正常模式(18<VIX<25): 60%仓位，趋势跟踪
3. 贪婪模式(VIX<18): 40%仓位，趋势跟随

关键改进：
- 减少交易频率（5%阈值）
- 更清晰的模式切换
- 更宽松的止损
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
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
    vix: float


class ImprovedHybrid:
    """改进版混合策略"""
    
    def __init__(
        self,
        # 恐慌参数
        panic_buy_threshold: float = -45,
        panic_sell_threshold: float = 75,
        panic_position: float = 0.70,  # 恐慌时仓位
        
        # 趋势参数
        trend_fast: int = 5,
        trend_slow: int = 20,
        trend_position_normal: float = 0.60,  # 正常时仓位
        trend_position_greed: float = 0.40,  # 贪婪时仓位
        
        # VIX阈值
        vix_panic: float = 25.0,
        vix_greed: float = 18.0,
        
        # 交易参数
        trade_threshold: float = 0.05,  # 5%触发交易
        stop_loss: float = -0.15,
    ):
        self.panic_buy_threshold = panic_buy_threshold
        self.panic_sell_threshold = panic_sell_threshold
        self.panic_position = panic_position
        
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.trend_position_normal = trend_position_normal
        self.trend_position_greed = trend_position_greed
        
        self.vix_panic = vix_panic
        self.vix_greed = vix_greed
        
        self.trade_threshold = trade_threshold
        self.stop_loss = stop_loss
    
    def get_mode(self, vix: float) -> Tuple[str, float]:
        """判断市场模式 + 目标仓位"""
        if vix > self.vix_panic:
            return 'panic', self.panic_position
        elif vix < self.vix_greed:
            return 'greed', self.trend_position_greed
        else:
            return 'normal', self.trend_position_normal
    
    def calculate_panic_score(self, df: pd.DataFrame, i: int) -> float:
        """计算恐慌得分"""
        if i < 20:
            return 0
        
        vix = df['vix'].iloc[i]
        daily_ret = (df['close'].iloc[i] / df['close'].iloc[i-1] - 1)
        
        vix_rolling = df['vix'].iloc[i-20:i+1]
        vix_zscore = (vix - vix_rolling.mean()) / (vix_rolling.std() + 1e-8)
        vix_percentile = vix_rolling.rank(pct=True).iloc[-1]
        
        score = vix_zscore * 40 + vix_percentile * 60
        if daily_ret < 0:
            score += min(daily_ret / -0.05, 1) * -50
        
        return score
    
    def calculate_trend(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """计算趋势"""
        if i < self.trend_slow:
            return 'hold', 0
        
        ma_f = df['close'].iloc[i-self.trend_fast+1:i+1].mean()
        ma_s = df['close'].iloc[i-self.trend_slow+1:i+1].mean()
        
        trend = (ma_f - ma_s) / ma_s
        
        # RSI
        delta = df['close'].diff().iloc[max(0,i-14):i+1]
        gain = delta.where(delta > 0, 0).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        if ma_f > ma_s and rsi < 65:
            return 'up', min(abs(trend) * 10, 1.0)
        elif ma_f < ma_s and rsi > 35:
            return 'down', min(abs(trend) * 10, 1.0)
        else:
            return 'hold', 0
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """回测"""
        
        if verbose:
            print("="*70)
            print("改进版混合策略")
            print("="*70)
            print(f"\n策略配置:")
            print(f"  恐慌模式(VIX>{self.vix_panic}): 仓位{self.panic_position:.0%}, 逆向抄底")
            print(f"  正常模式({self.vix_greed}<VIX<{self.vix_panic}): 仓位{self.trend_position_normal:.0%}, 趋势跟踪")
            print(f"  贪婪模式(VIX<{self.vix_greed}): 仓位{self.trend_position_greed:.0%}, 趋势跟随")
            print(f"\n风控:")
            print(f"  止损: {self.stop_loss:.0%}")
            print(f"  交易阈值: {self.trade_threshold:.0%}\n")
        
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
            
            # 止损
            if total_return < self.stop_loss and shares > 0:
                cash += shares * price * 0.999
                trades.append(Trade('sell', row['date'], price, int(shares), 0,
                                 f'止损({total_return:.1%})', 'risk', vix))
                shares = 0
                avg_price = 0
                peak_value = cash
                portfolio_values.append(cash)
                continue
            
            # 判断模式
            mode, base_position = self.get_mode(vix)
            
            # 计算信号
            panic_score = self.calculate_panic_score(df, i)
            trend_dir, trend_strength = self.calculate_trend(df, i)
            
            # 目标仓位
            target_position = position_ratio
            
            if mode == 'panic':
                # 恐慌模式：逆向
                if panic_score <= self.panic_buy_threshold:
                    target_position = base_position
                    strategy = 'panic'
                elif panic_score >= self.panic_sell_threshold and position_ratio > 0:
                    target_position = max(0, position_ratio - 0.3)
                    strategy = 'panic'
                else:
                    strategy = 'hold'
            
            else:
                # 正常/贪婪：趋势
                if trend_dir == 'up' and position_ratio < base_position:
                    target_position = base_position
                    strategy = 'trend'
                elif trend_dir == 'down' and position_ratio > 0.1:
                    target_position = max(0.1, position_ratio - 0.2)
                    strategy = 'trend'
                else:
                    strategy = 'hold'
            
            # 执行交易
            if abs(target_position - position_ratio) > self.trade_threshold:
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
                                         target_position, f'{mode}', strategy, vix))
                        
                        if strategy == 'panic':
                            panic_trades += 1
                        elif strategy == 'trend':
                            trend_trades += 1
                
                elif delta_shares < -10:
                    sell = min(-delta_shares, shares)
                    cash += sell * price * 0.999
                    shares -= sell
                    
                    trades.append(Trade('sell', row['date'], price, int(sell),
                                     target_position, f'{mode}', strategy, vix))
                    
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
            print(f"  恐慌策略:    {panic_trades}")
            print(f"  趋势策略:    {trend_trades}")
            print("-"*70)
            print(f"最终价值:      ${pv[-1]:,.0f}")
            print(f"收益倍数:      {pv[-1] / 100000:.2f}x")
            print("-"*70)
            print(f"买入持有:      {buy_hold_return:.2%}")
            print(f"超额收益:      {total_return - buy_hold_return:.2%}")
        
        return results


def param_search(df: pd.DataFrame):
    """参数搜索"""
    
    print("="*70)
    print("参数搜索")
    print("="*70)
    
    best = {'score': -999}
    
    from itertools import product
    
    params = {
        'panic_buy_threshold': [-40, -45, -50],
        'panic_position': [0.65, 0.70, 0.75],
        'trend_position_normal': [0.55, 0.60, 0.65],
        'trade_threshold': [0.04, 0.05, 0.06],
    }
    
    total = 1
    for v in params.values():
        total *= len(v)
    
    print(f"搜索空间: {total} 组合\n")
    
    count = 0
    for p in product(*params.values()):
        count += 1
        
        strategy = ImprovedHybrid(
            panic_buy_threshold=p[0],
            panic_position=p[1],
            trend_position_normal=p[2],
            trade_threshold=p[3],
        )
        
        results = strategy.run_backtest(df, verbose=False)
        
        # 评分：收益 - 回撤*0.5
        score = results['total_return'] - results['max_drawdown'] * 0.5
        
        if score > best['score']:
            best = {
                'score': score,
                'params': dict(zip(params.keys(), p)),
                'results': results,
            }
        
        if count % 20 == 0:
            print(f"进度: {count}/{total}, 当前最优收益: {best['results']['total_return']:.2%}")
    
    print("\n" + "="*70)
    print("最优参数")
    print("="*70)
    for k, v in best['params'].items():
        print(f"  {k}: {v}")
    
    return best


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv')
    parser.add_argument('--search', action='store_true')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.data, parse_dates=['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
    
    if args.search:
        best = param_search(df)
        print("\n用最优参数运行:")
        strategy = ImprovedHybrid(**best['params'])
        results = strategy.run_backtest(df, verbose=True)
    else:
        strategy = ImprovedHybrid()
        results = strategy.run_backtest(df, verbose=True)
