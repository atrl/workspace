#!/usr/bin/env python3
"""
SOXL Qlib V14.1 - 微调版（目标 Sharpe 0.50）
基于 V14 (Sharpe 0.46, Max DD 17.4%) 微调

微调参数：
- 趋势权重: 0.50 → 0.55 (+0.05)
- 动量权重: 0.30 (保持)
- 波动率权重: 0.15 (保持)

目标:
- Sharpe > 0.50
- Max DD < 20%
"""

import qlib
from qlib.data import D
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Part 1: Qlib 初始化
# ============================================================

def init_qlib():
    """初始化 Qlib"""
    possible_paths = [
        Path('/Users/rziiiii/.openclaw/workspace/qlib_data'),
        Path(__file__).parent.parent / 'qlib_data',
    ]
    
    qlib_data_path = None
    for path in possible_paths:
        if path.exists():
            qlib_data_path = path
            break
    
    if qlib_data_path is None:
        print(f"❌ Qlib 数据不存在")
        return False
    
    try:
        qlib.init(provider_uri=str(qlib_data_path), expression_cache=None, dataset_cache=None)
        print(f"✅ Qlib 初始化成功: {qlib_data_path}")
        return True
    except Exception as e:
        print(f"❌ Qlib 初始化失败: {e}")
        return False

# ============================================================
# Part 2: Qlib Expression 因子
# ============================================================

def get_qlib_features():
    """Qlib Expression 因子"""
    return [
        '$close',
        '$volume',
        'Mean($close, 5)',
        'Mean($close, 20)',
        'Mean($close, 60)',
        'Ref($close, 5) / $close - 1',
        'Mean($volume, 20)',
        'Std($close / Ref($close, 1) - 1, 20)',
        'Std($close / Ref($close, 1) - 1, 60)',
    ]

# ============================================================
# Part 3: Kelly 仓位管理
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
# Part 4: V14.1 信号计算（微调版）
# ============================================================

def calculate_signal_v141(df):
    """
    V14.1 信号（趋势权重 +5%）
    
    V14 参数:
    - trend_weight = 0.50
    - momentum_weight = 0.30
    - vol_weight = 0.15
    
    V14.1 微调:
    - trend_weight = 0.55 (+0.05)
    - momentum_weight = 0.30 (保持)
    - vol_weight = 0.15 (保持)
    """
    # 趋势因子
    trend = (
        (df['$close'] > df['Mean($close, 5)']).astype(float) * 0.2 +
        (df['$close'] > df['Mean($close, 20)']).astype(float) * 0.3 +
        (df['$close'] > df['Mean($close, 60)']).astype(float) * 0.3 +
        (df['Mean($close, 5)'] > df['Mean($close, 20)']).astype(float) * 0.2
    )
    
    # 动量因子
    momentum = -df['Ref($close, 5) / $close - 1']
    
    # 波动率因子
    vol_ratio = df['Std($close / Ref($close, 1) - 1, 20)'] / (df['Std($close / Ref($close, 1) - 1, 60)'] + 1e-9)
    vol_factor = 1 - vol_ratio.clip(0.5, 2)
    
    # 成交量确认
    vol_conf = (df['$volume'] / (df['Mean($volume, 20)'] + 1e-9) > 1.5).astype(float) * 0.5 + 0.5
    
    # V14.1 综合信号（微调权重）
    signal = (
        trend * 0.55 +           # V14: 0.50 → V14.1: 0.55
        momentum * 0.30 +        # 保持
        vol_factor * 0.15        # 保持
    ) * vol_conf
    
    return signal.clip(-1, 1)

# ============================================================
# Part 5: 回测引擎
# ============================================================

def backtest_v141(df, signal, initial_balance=100000, cost=0.001):
    """V14.1 回测"""
    
    close = df['$close']
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
    print("SOXL Qlib V14.1 - 微调版（趋势权重 +5%）")
    print("=" * 60)
    
    if not init_qlib():
        print("\n❌ 无法初始化 Qlib")
        return None
    
    instruments = ['SOXL']
    features = get_qlib_features()
    
    print(f"\n📊 加载 Qlib 数据...")
    
    try:
        df = D.features(instruments=instruments, fields=features,
                       start_time='2020-01-01', end_time='2026-03-19')
        print(f"✅ 数据: {len(df)} 条")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None
    
    # 计算信号
    signal = calculate_signal_v141(df)
    
    # 回测
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
    print(f"\n📊 对比:")
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
