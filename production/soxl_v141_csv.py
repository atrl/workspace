#!/usr/bin/env python3
"""
SOXL V14.1 - CSV 版本（临时）
不依赖 Qlib Provider，直接使用 CSV 数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Part 1: 数据加载（CSV 直接读取）
# ============================================================

def load_data_from_csv():
    """从 CSV 加载 SOXL 数据"""
    workspace = Path('/Users/rziiiii/.openclaw/workspace')
    
    # 尝试多个可能的路径
    possible_paths = [
        workspace / 'data' / 'symbols' / 'SOXL.csv',
        workspace / 'qlib_data' / 'csv' / 'soxl.csv',
        Path.home() / '.qlib' / 'qlib_data' / 'csv' / 'soxl.csv',
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ 找到数据: {path}")
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            print(f"   数据量: {len(df)} 条")
            print(f"   时间范围: {df.index[0]} 至 {df.index[-1]}")
            return df
    
    print(f"❌ 未找到数据文件")
    print(f"   尝试路径: {[str(p) for p in possible_paths]}")
    return None

# ============================================================
# Part 2: 因子计算（Qlib Expression 风格）
# ============================================================

def calculate_factors(df):
    """计算因子（Qlib Expression 风格）"""
    data = df.copy()
    
    # 趋势因子
    data['Mean($close, 5)'] = data['close'].rolling(5).mean().shift(1)
    data['Mean($close, 20)'] = data['close'].rolling(20).mean().shift(1)
    data['Mean($close, 60)'] = data['close'].rolling(60).mean().shift(1)
    
    # 动量因子
    data['Ref($close, 5) / $close - 1'] = data['close'].pct_change(5).shift(1)
    
    # 波动率因子
    ret = data['close'].pct_change()
    data['Std($close / Ref($close, 1) - 1, 20)'] = ret.rolling(20).std().shift(1)
    data['Std($close / Ref($close, 1) - 1, 60)'] = ret.rolling(60).std().shift(1)
    
    # 成交量因子
    data['Mean($volume, 20)'] = data['volume'].rolling(20).mean().shift(1)
    
    return data

# ============================================================
# Part 3: V14.1 信号计算
# ============================================================

def calculate_signal_v141(df):
    """V14.1 信号（趋势权重 0.55）"""
    # 趋势因子
    trend = (
        (df['close'] > df['Mean($close, 5)']).astype(float) * 0.2 +
        (df['close'] > df['Mean($close, 20)']).astype(float) * 0.3 +
        (df['close'] > df['Mean($close, 60)']).astype(float) * 0.3 +
        (df['Mean($close, 5)'] > df['Mean($close, 20)']).astype(float) * 0.2
    )
    
    # 动量因子
    momentum = -df['Ref($close, 5) / $close - 1']
    
    # 波动率因子
    vol_ratio = df['Std($close / Ref($close, 1) - 1, 20)'] / (df['Std($close / Ref($close, 1) - 1, 60)'] + 1e-9)
    vol_factor = 1 - vol_ratio.clip(0.5, 2)
    
    # 成交量确认
    vol_conf = (df['volume'] / (df['Mean($volume, 20)'] + 1e-9) > 1.5).astype(float) * 0.5 + 0.5
    
    # V14.1 综合信号
    signal = (
        trend * 0.55 +           # V14.1: 0.55 (从 0.50)
        momentum * 0.30 +
        vol_factor * 0.15
    ) * vol_conf
    
    return signal.clip(-1, 1)

# ============================================================
# Part 4: Kelly 仓位管理
# ============================================================

def kelly_position(returns, window=50, max_kelly=0.40):
    """Kelly 公式"""
    if len(returns) < window:
        return 0.20
    
    recent = returns[-window:]
    wins = recent[recent > 0]
    losses = recent[recent < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.20
    
    win_rate = len(wins) / len(recent)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    win_loss_ratio = avg_win / (avg_loss + 1e-9)
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    kelly = kelly * 0.5
    
    vol = np.std(recent)
    if vol > 0.03:
        kelly *= 0.7
    
    return np.clip(kelly, 0.10, max_kelly)

# ============================================================
# Part 5: 回测引擎
# ============================================================

def backtest_v141(df, signal, initial_balance=100000, cost=0.001):
    """V14.1 回测"""
    
    close = df['close']
    ret = close.pct_change()
    
    balance = initial_balance
    position = 0.0
    peak_value = initial_balance
    
    trade_returns = []
    current_kelly = 0.20
    
    last_position = 0.0
    
    portfolio_values = [initial_balance]
    positions = [0.0]
    
    # 8级回撤保护
    dd_levels = [-0.04, -0.06, -0.08, -0.10, -0.12, -0.14, -0.16, -0.18]
    dd_mults = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    vol_series = df['Std($close / Ref($close, 1) - 1, 20)']
    
    for i in range(60, len(df)):
        current_signal = signal.iloc[i]
        
        kelly_limit = current_kelly
        
        current_value = balance
        if current_value > peak_value:
            peak_value = current_value
        
        current_dd = (current_value - peak_value) / peak_value
        
        dd_mult = 1.0
        for level, mult in zip(dd_levels, dd_mults):
            if current_dd < level:
                dd_mult = mult
                break
        
        current_vol = vol_series.iloc[i]
        if pd.notna(current_vol):
            ann_vol = current_vol * np.sqrt(252)
            if ann_vol > 0.80:
                vol_mult = 0.4
            elif ann_vol > 0.70:
                vol_mult = 0.6
            else:
                vol_mult = 1.0
        else:
            vol_mult = 1.0
        
        target_position = current_signal * kelly_limit * dd_mult * vol_mult
        target_position = np.clip(target_position, -kelly_limit, kelly_limit)
        
        # EMA 平滑
        target_position = 0.7 * target_position + 0.3 * last_position
        last_position = target_position
        
        current_price = close.iloc[i]
        
        if i < len(df) - 1:
            next_price = close.iloc[i + 1]
            
            position_change = abs(target_position - position)
            trade_cost = position_change * cost
            
            price_return = (next_price - current_price) / current_price
            portfolio_return = target_position * price_return - trade_cost
            
            balance *= (1 + portfolio_return)
            position = target_position
            
            trade_returns.append(portfolio_return)
            portfolio_values.append(balance)
            positions.append(position)
            
            if len(trade_returns) > 50:
                current_kelly = kelly_position(np.array(trade_returns))
    
    # 统计
    pv = np.array(portfolio_values)
    rets = np.diff(pv) / (pv[:-1] + 1e-9)
    
    total_return = pv[-1] / pv[0] - 1
    sharpe = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)
    
    cummax = np.maximum.accumulate(pv)
    dd = (pv - cummax) / cummax
    max_dd = dd.min()
    
    win_rate = (rets > 0).sum() / len(rets)
    
    position_changes = np.abs(np.diff(positions))
    trade_count = (position_changes > 0.01).sum()
    
    avg_position = np.mean(np.abs(positions))
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trade_count': trade_count,
        'final_kelly': current_kelly,
        'avg_position': avg_position
    }

# ============================================================
# Part 6: 主流程
# ============================================================

def main():
    print("=" * 60)
    print("SOXL V14.1 - CSV 版本（趋势权重 +5%）")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_data_from_csv()
    if df is None:
        return None
    
    # 2. 计算因子
    print(f"\n📊 计算因子...")
    df = calculate_factors(df)
    print(f"✅ 因子计算完成")
    
    # 3. 计算信号
    signal = calculate_signal_v141(df)
    
    # 4. 回测
    stats = backtest_v141(df, signal)
    
    print(f"\n📈 V14.1 结果:")
    print(f"  总收益: {stats['total_return']:.2%}")
    print(f"  Sharpe: {stats['sharpe']:.2f}")
    print(f"  Max DD: {stats['max_dd']:.2%}")
    print(f"  胜率: {stats['win_rate']:.2%}")
    print(f"  交易: {stats['trade_count']}")
    print(f"  平均仓位: {stats['avg_position']:.2%}")
    print(f"  最终 Kelly: {stats['final_kelly']:.2%}")
    
    # 对比
    print(f"\n📊 对比 V14:")
    print(f"  V14:   Sharpe 0.46, Max DD 17.4%, 平均仓位 12.3%")
    print(f"  V14.1: Sharpe {stats['sharpe']:.2f}, Max DD {stats['max_dd']:.2%}, 平均仓位 {stats['avg_position']:.2%}")
    
    # 目标检查
    print(f"\n🎯 目标检查:")
    if stats['sharpe'] >= 0.50:
        print(f"  ✅ Sharpe 达标 ({stats['sharpe']:.2f} >= 0.50)")
    else:
        print(f"  ❌ Sharpe 未达标 ({stats['sharpe']:.2f} < 0.50)")
    
    if stats['max_dd'] >= -0.20:
        print(f"  ✅ Max DD 达标 ({stats['max_dd']:.2%} >= -20%)")
    else:
        print(f"  ❌ Max DD 未达标 ({stats['max_dd']:.2%} < -20%)")
    
    return stats

if __name__ == "__main__":
    main()
