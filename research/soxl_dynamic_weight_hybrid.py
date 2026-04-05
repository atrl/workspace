#!/usr/bin/env python3
"""
动态权重混合策略：根据VIX自动调整恐慌/趋势权重

核心创新：
1. VIX > 25: 恐慌权重80%, 趋势权重20%
2. 18 < VIX < 25: 趋势权重70%, 恐慌权重30%
3. VIX < 18: 趋势权重90%, 恐慌权重10%

目标：在控制回撤的同时提升收益
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


class DynamicWeightHybrid:
    """动态权重混合策略"""
    
    def __init__(
        self,
        # 恐慌指数参数
        panic_buy_threshold: float = -45,
        panic_sell_threshold: float = 75,
        panic_max_position: float = 0.80,
        
        # 趋势跟踪参数
        trend_ma_fast: int = 5,
        trend_ma_slow: int = 20,
        trend_max_position: float = 0.60,
        trend_stop_loss: float = -0.08,
        
        # 动态权重参数
        vix_very_high: float = 30.0,    # 极端恐慌
        vix_high: float = 25.0,         # 恐慌
        vix_low: float = 18.0,          # 贪婪
        
        # 风控参数
        max_total_position: float = 0.85,
        hard_stop_loss: float = -0.12,
        trailing_stop: float = 0.08,    # 移动止损
    ):
        self.panic_buy_threshold = panic_buy_threshold
        self.panic_sell_threshold = panic_sell_threshold
        self.panic_max_position = panic_max_position
        
        self.trend_ma_fast = trend_ma_fast
        self.trend_ma_slow = trend_ma_slow
        self.trend_max_position = trend_max_position
        self.trend_stop_loss = trend_stop_loss
        
        self.vix_very_high = vix_very_high
        self.vix_high = vix_high
        self.vix_low = vix_low
        
        self.max_total_position = max_total_position
        self.hard_stop_loss = hard_stop_loss
        self.trailing_stop = trailing_stop
    
    def get_dynamic_weights(self, vix: float) -> Tuple[float, float]:
        """根据VIX计算恐慌/趋势权重"""
        
        if vix > self.vix_very_high:
            # 极端恐慌：恐慌85%, 趋势15%
            return 0.85, 0.15
        
        elif vix > self.vix_high:
            # 恐慌：恐慌70%, 趋势30%
            return 0.70, 0.30
        
        elif vix < self.vix_low:
            # 贪婪：趋势80%, 恐慌20%
            return 0.20, 0.80
        
        else:
            # 正常：趋势60%, 恐慌40%
            return 0.40, 0.60
    
    def calculate_panic_score(self, df: pd.DataFrame, i: int) -> float:
        """计算恐慌得分"""
        if i < 20:
            return 0
        
        vix = df['vix'].iloc[i]
        daily_ret = (df['close'].iloc[i] / df['close'].iloc[i-1] - 1) if i > 0 else 0
        
        vix_rolling = df['vix'].iloc[max(0,i-20):i+1]
        vix_zscore = (vix - vix_rolling.mean()) / (vix_rolling.std() + 1e-8)
        vix_percentile = vix_rolling.rank(pct=True).iloc[-1]
        
        panic_score = (
            vix_zscore * 40 +
            vix_percentile * 60 +
            min(daily_ret / -0.05, 1) * -50 if daily_ret < 0 else 0
        )
        
        return panic_score
    
    def calculate_trend_signal(self, df: pd.DataFrame, i: int) -> Tuple[str, float, float]:
        """计算趋势信号和强度"""
        if i < max(self.trend_ma_fast, self.trend_ma_slow):
            return 'hold', 0, 50
        
        close = df['close'].iloc[i]
        ma_fast = df['close'].iloc[i-self.trend_ma_fast+1:i+1].mean()
        ma_slow = df['close'].iloc[i-self.trend_ma_slow+1:i+1].mean()
        
        # 趋势强度
        trend_strength = abs((ma_fast - ma_slow) / ma_slow)
        
        # RSI
        delta = df['close'].diff().iloc[max(0,i-14):i+1]
        gain = delta.where(delta > 0, 0).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        # 信号
        if ma_fast > ma_slow:
            return 'buy', min(trend_strength * 15, 1.0), rsi
        else:
            return 'sell', min(trend_strength * 15, 1.0), rsi
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """运行动态权重混合策略回测"""
        
        if verbose:
            print("="*70)
            print("动态权重混合策略")
            print("="*70)
            print(f"\n创新点：根据VIX自动调整恐慌/趋势权重")
            print(f"  VIX > {self.vix_very_high}: 恐慌85%, 趋势15%")
            print(f"  VIX > {self.vix_high}: 恐慌70%, 趋势30%")
            print(f"  {self.vix_low} < VIX < {self.vix_high}: 趋势60%, 恐慌40%")
            print(f"  VIX < {self.vix_low}: 趋势80%, 恐慌20%")
            print(f"\n风控:")
            print(f"  硬止损: {self.hard_stop_loss:.0%}")
            print(f"  移动止损: {self.trailing_stop:.0%}")
            print(f"  最大仓位: {self.max_total_position:.0%}\n")
        
        cash = 100000
        shares = 0
        avg_price = 0
        peak_value = 100000
        high_water_mark = 100000
        
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
            
            # 1. 硬止损
            if total_return < self.hard_stop_loss and shares > 0:
                cash += shares * price * 0.999
                trades.append(Trade('sell', row['date'], price, int(shares), 0, 
                                 f'硬止损({total_return:.1%})', 'risk', vix))
                shares = 0
                avg_price = 0
                peak_value = total_value
                high_water_mark = total_value
                portfolio_values.append(cash)
                continue
            
            # 2. 移动止损
            if total_value > high_water_mark:
                high_water_mark = total_value
            
            if shares > 0 and high_water_mark > 0:
                drawdown_from_high = (high_water_mark - total_value) / high_water_mark
                if drawdown_from_high > self.trailing_stop:
                    cash += shares * price * 0.999
                    trades.append(Trade('sell', row['date'], price, int(shares), 0,
                                     f'移动止损({drawdown_from_high:.1%})', 'risk', vix))
                    shares = 0
                    avg_price = 0
                    high_water_mark = total_value
                    portfolio_values.append(cash)
                    continue
            
            # 3. 计算动态权重
            panic_weight, trend_weight = self.get_dynamic_weights(vix)
            
            # 4. 计算各策略信号
            panic_score = self.calculate_panic_score(df, i)
            trend_signal, trend_strength, rsi = self.calculate_trend_signal(df, i)
            
            # 5. 恐慌策略仓位
            panic_position = 0
            if panic_score <= self.panic_buy_threshold:
                panic_position = self.panic_max_position * min(abs(panic_score) / 100, 1.0)
            elif panic_score >= self.panic_sell_threshold:
                panic_position = 0
            
            # 6. 趋势策略仓位
            trend_position = 0
            if trend_signal == 'buy' and rsi < 70:
                trend_position = self.trend_max_position * trend_strength
            elif trend_signal == 'sell':
                trend_position = 0
                
                # 趋势止损
                if avg_price > 0 and (price - avg_price) / avg_price < self.trend_stop_loss:
                    trend_position = 0
            
            # 7. 加权融合
            target_position = (
                panic_position * panic_weight +
                trend_position * trend_weight
            )
            
            # 8. 总仓位限制
            target_position = min(target_position, self.max_total_position)
            
            # 9. 执行交易
            if abs(target_position - position_ratio) > 0.015:
                target_value = total_value * target_position
                delta_shares = (target_value - position_value) / price
                
                strategy = 'panic' if panic_weight > trend_weight else 'trend'
                
                if delta_shares > 10:
                    cost = delta_shares * price * 1.001
                    if cost <= cash:
                        total_cost = avg_price * shares + price * delta_shares
                        shares += delta_shares
                        avg_price = total_cost / shares
                        cash -= cost
                        
                        trades.append(Trade('buy', row['date'], price, int(delta_shares),
                                         target_position, 
                                         f'恐慌{panic_weight:.0%}+趋势{trend_weight:.0%}',
                                         strategy, vix))
                        
                        if strategy == 'panic':
                            panic_trades += 1
                        else:
                            trend_trades += 1
                
                elif delta_shares < -10:
                    sell = min(-delta_shares, shares)
                    cash += sell * price * 0.999
                    shares -= sell
                    
                    trades.append(Trade('sell', row['date'], price, int(sell),
                                     target_position,
                                     f'恐慌{panic_weight:.0%}+趋势{trend_weight:.0%}',
                                     strategy, vix))
                    
                    if strategy == 'panic':
                        panic_trades += 1
                    else:
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


if __name__ == '__main__':
    df = pd.read_csv('/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv', parse_dates=['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
    
    strategy = DynamicWeightHybrid()
    results = strategy.run_backtest(df, verbose=True)
