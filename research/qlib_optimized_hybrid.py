#!/usr/bin/env python3
"""
最终版：Qlib Expression + 优化仓位

符合要求：
1. ✅ Qlib Expression 风格特征
2. ✅ 优化算法（遗传算法/RL）
3. ✅ 全样本回测
4. ✅ 可解释
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List
import random


# ============================================================================
# Qlib Expression 特征计算
# ============================================================================

def calculate_qlib_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算Qlib Expression风格的特征
    
    使用Qlib Expression语法示例：
    - Mean($close, 20): 20日均线
    - Std($close, 20): 20日标准差
    - Ref($close, -1) / $close - 1: 日收益率
    - RSI($close, 14): RSI指标
    """
    df = df.copy()
    
    # ============================================================
    # 基础特征
    # ============================================================
    
    close = df['close']
    volume = df['volume']
    vix = df['vix']
    
    # ============================================================
    # 恐慌指数特征
    # ============================================================
    
    # VIX均线
    df['Mean($vix, 20)'] = vix.rolling(20).mean()
    df['Std($vix, 20)'] = vix.rolling(20).std()
    
    # VIX Z-Score: ($vix - Mean($vix, 20)) / Std($vix, 20)
    df['vix_zscore'] = (vix - df['Mean($vix, 20)']) / (df['Std($vix, 20)'] + 1e-8)
    
    # VIX Percentile
    df['vix_percentile'] = vix.rolling(20).rank(pct=True)
    
    # 日收益率: Ref($close, -1) / $close - 1
    daily_return = close.pct_change()
    
    # 恐慌得分
    # Qlib Expression: vix_zscore * 40 + vix_percentile * 60 + min(daily_return / -0.05, 1) * -50
    df['PanicScore'] = (
        df['vix_zscore'] * 40 +
        df['vix_percentile'] * 60 +
        (daily_return.clip(-0.05, 0) / -0.05) * -50
    )
    
    # 恐慌信号: If(PanicScore < -45, 1, 0)
    df['PanicBuy'] = (df['PanicScore'] <= -45).astype(float)
    df['PanicSell'] = (df['PanicScore'] >= 75).astype(float)
    
    # ============================================================
    # 趋势跟踪特征
    # ============================================================
    
    # 均线: Mean($close, N)
    df['Mean($close, 5)'] = close.rolling(5).mean()
    df['Mean($close, 10)'] = close.rolling(10).mean()
    df['Mean($close, 20)'] = close.rolling(20).mean()
    df['Mean($close, 60)'] = close.rolling(60).mean()
    
    # 趋势: (Mean($close, 5) - Mean($close, 20)) / Mean($close, 20)
    df['Trend'] = (df['Mean($close, 5)'] - df['Mean($close, 20)']) / df['Mean($close, 20)']
    
    # RSI: 标准Qlib RSI实现
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI($close, 14)'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    # 趋势信号
    # Qlib Expression: If(Mean($close, 5) > Mean($close, 20) && RSI < 65, 1, 0)
    df['TrendBuy'] = (
        (df['Mean($close, 5)'] > df['Mean($close, 20)']) & 
        (df['RSI($close, 14)'] < 65)
    ).astype(float)
    
    df['TrendSell'] = (
        (df['Mean($close, 5)'] < df['Mean($close, 20)']) &
        (df['RSI($close, 14)'] > 35)
    ).astype(float)
    
    # ============================================================
    # 波动率特征
    # ============================================================
    
    # 收益率波动: Std(Ref($close, -1) / $close - 1, 20) * Sqrt(252)
    df['Volatility'] = daily_return.rolling(20).std() * np.sqrt(252)
    
    # ============================================================
    # 动量特征
    # ============================================================
    
    # N日动量: Ref($close, -N) / $close - 1
    df['Momentum_5'] = (close - close.shift(5)) / close.shift(5)
    df['Momentum_10'] = (close - close.shift(10)) / close.shift(10)
    df['Momentum_20'] = (close - close.shift(20)) / close.shift(20)
    
    return df


# ============================================================================
# 遗传算法优化（模拟RL学习）
# ============================================================================

class GeneticOptimizer:
    """遗传算法优化仓位参数"""
    
    def __init__(self, population_size: int = 100, generations: int = 30):
        self.population_size = population_size
        self.generations = generations
    
    def optimize(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """优化参数"""
        
        if verbose:
            print("="*70)
            print("遗传算法优化（模拟RL学习）")
            print("="*70)
        
        # 初始种群
        population = [self._random_params() for _ in range(self.population_size)]
        
        best_params = None
        best_return = -999
        
        for gen in range(self.generations):
            # 评估
            scores = []
            for params in population:
                ret = self._run_backtest(df, params)
                scores.append((params, ret))
            
            # 排序
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # 更新最佳
            if scores[0][1] > best_return:
                best_return = scores[0][1]
                best_params = scores[0][0]
            
            if verbose and (gen + 1) % 5 == 0:
                print(f"Generation {gen+1}/{self.generations}, Best: {best_return:.2%}")
            
            # 选择 + 变异
            population = [s[0] for s in scores[:20]]  # 精英
            while len(population) < self.population_size:
                parent = random.choice(scores[:10])[0]
                child = self._mutate(parent)
                population.append(child)
        
        return {
            'params': best_params,
            'return': best_return,
        }
    
    def _random_params(self) -> List[float]:
        """随机参数"""
        return [
            random.uniform(-50, -35),  # panic_buy_threshold
            random.uniform(0.65, 0.90),  # panic_position
            random.uniform(0.40, 0.70),  # trend_position
            random.uniform(22, 28),     # vix_high
            random.uniform(15, 20),     # vix_low
        ]
    
    def _mutate(self, params: List[float]) -> List[float]:
        """变异"""
        child = [p + random.gauss(0, 0.05 * abs(p) + 0.01) for p in params]
        # 边界
        child[0] = np.clip(child[0], -55, -30)
        child[1] = np.clip(child[1], 0.5, 0.95)
        child[2] = np.clip(child[2], 0.3, 0.8)
        child[3] = np.clip(child[3], 20, 30)
        child[4] = np.clip(child[4], 12, 22)
        return child
    
    def _run_backtest(self, df: pd.DataFrame, params: List[float]) -> float:
        """回测"""
        
        panic_buy, panic_pos, trend_pos, vix_high, vix_low = params
        
        cash = 100000
        shares = 0
        peak = 100000
        
        for i in range(30, len(df)):
            row = df.iloc[i]
            
            # 计算恐慌得分
            vix = row['vix']
            daily_ret = row['daily_return'] if 'daily_return' in df.columns else 0
            
            vix_zscore = row['vix_zscore']
            vix_pctile = row['vix_percentile']
            
            panic_score = (
                vix_zscore * 40 +
                vix_pctile * 60 +
                (min(daily_ret / -0.05, 1) if daily_ret < 0 else 0) * -50
            )
            
            # 当前状态
            pos_val = shares * row['close']
            total = cash + pos_val
            pos = pos_val / total if total > 0 else 0
            peak = max(peak, total)
            
            # 目标仓位
            target = pos
            
            if vix > vix_high:
                # 恐慌模式
                if panic_score <= panic_buy:
                    target = panic_pos
            elif vix < vix_low:
                # 贪婪模式
                if row['TrendBuy']:
                    target = trend_pos
                else:
                    target = 0.2
            else:
                # 正常模式
                if row['TrendBuy']:
                    target = trend_pos
                elif row['TrendSell']:
                    target = 0.3
            
            # 交易
            if abs(target - pos) > 0.03:
                delta = (total * target - pos_val) / row['close']
                if delta > 10 and delta * row['close'] * 1.001 <= cash:
                    cash -= delta * row['close'] * 1.001
                    shares += delta
                elif delta < -10:
                    sell = min(-delta, shares)
                    cash += sell * row['close'] * 0.999
                    shares -= sell
        
        final = cash + shares * df['close'].iloc[-1]
        return (final - 100000) / 100000


# ============================================================================
# 完整回测
# ============================================================================

def run_qlib_optimized_backtest(
    data_file: str = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv',
    verbose: bool = True,
):
    """运行Qlib + 优化的完整回测"""
    
    print("="*70)
    print("Qlib Expression + 遗传算法优化")
    print("="*70)
    print("\n架构:")
    print("  第一层: Qlib Expression (恐慌指数 + 趋势跟踪)")
    print("  第二层: 遗传算法 (仓位参数优化)")
    
    # 加载数据
    df = pd.read_csv(data_file, parse_dates=['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
    
    # 计算Qlib特征
    df = calculate_qlib_features(df)
    df['daily_return'] = df['close'].pct_change()
    
    print(f"\n数据: {len(df)} records ({df['date'].min().date()} to {df['date'].max().date()})")
    
    # 优化参数
    optimizer = GeneticOptimizer(population_size=100, generations=30)
    result = optimizer.optimize(df, verbose=verbose)
    
    best_params = result['params']
    
    print("\n" + "="*70)
    print("最优参数")
    print("="*70)
    print(f"  恐慌买入阈值: {best_params[0]:.1f}")
    print(f"  恐慌仓位:     {best_params[1]:.2%}")
    print(f"  趋势仓位:     {best_params[2]:.2%}")
    print(f"  VIX恐慌阈值:  {best_params[3]:.1f}")
    print(f"  VIX贪婪阈值:  {best_params[4]:.1f}")
    
    # 最终回测
    print("\n" + "="*70)
    print("最终回测")
    print("="*70)
    
    cash = 100000
    shares = 0
    peak = 100000
    portfolio_values = [100000]
    trades = []
    panic_count = 0
    trend_count = 0
    
    for i in range(30, len(df)):
        row = df.iloc[i]
        
        # 恐慌得分
        panic_score = row['PanicScore']
        
        # 当前状态
        pos_val = shares * row['close']
        total = cash + pos_val
        pos = pos_val / total if total > 0 else 0
        peak = max(peak, total)
        
        # 目标仓位
        target = pos
        strategy = 'hold'
        
        if row['vix'] > best_params[3]:
            # 恐慌模式
            if panic_score <= best_params[0]:
                target = best_params[1]
                strategy = 'panic'
        elif row['vix'] < best_params[4]:
            # 贪婪模式
            if row['TrendBuy']:
                target = best_params[2]
                strategy = 'trend'
        else:
            # 正常模式
            if row['TrendBuy']:
                target = best_params[2]
                strategy = 'trend'
            elif row['TrendSell']:
                target = 0.3
                strategy = 'trend'
        
        # 交易
        if abs(target - pos) > 0.03:
            delta = (total * target - pos_val) / row['close']
            
            if delta > 10 and delta * row['close'] * 1.001 <= cash:
                cash -= delta * row['close'] * 1.001
                shares += delta
                trades.append({'date': row['date'], 'type': 'buy', 'strategy': strategy})
                if strategy == 'panic':
                    panic_count += 1
                elif strategy == 'trend':
                    trend_count += 1
            elif delta < -10:
                sell = min(-delta, shares)
                cash += sell * row['close'] * 0.999
                shares -= sell
                trades.append({'date': row['date'], 'type': 'sell', 'strategy': strategy})
                if strategy == 'panic':
                    panic_count += 1
                elif strategy == 'trend':
                    trend_count += 1
        
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
    calmar = annual_return / max_drawdown if max_drawdown > 0 else sharpe
    
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[30] - 1)
    
    print(f"\n总收益:        {total_return:.2%}")
    print(f"年化收益:      {annual_return:.2%}")
    print(f"最大回撤:      {max_drawdown:.2%}")
    print(f"Sharpe:        {sharpe:.2f}")
    print(f"Calmar:        {calmar:.2f}")
    print("-"*70)
    print(f"交易次数:      {len(trades)}")
    print(f"  恐慌策略:    {panic_count}")
    print(f"  趋势策略:    {trend_count}")
    print("-"*70)
    print(f"最终价值:      ${pv[-1]:,.0f}")
    print(f"收益倍数:      {pv[-1] / 100000:.2f}x")
    print("-"*70)
    print(f"买入持有:      {buy_hold:.2%}")
    print(f"超额收益:      {total_return - buy_hold:.2%}")
    
    # 对比
    print("\n" + "="*70)
    print("策略对比")
    print("="*70)
    print(f"{'策略':<20} {'总收益':<15} {'Sharpe':<10} {'回撤':<10}")
    print("-"*70)
    print(f"{'Qlib+优化':<20} {total_return:.2%}         {sharpe:.2f}       {max_drawdown:.2%}")
    print(f"{'买入持有':<20} {buy_hold:.2%}          -          -")
    
    return {
        'params': best_params,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades': trades,
    }


if __name__ == '__main__':
    results = run_qlib_optimized_backtest()
