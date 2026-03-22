#!/usr/bin/env python3
"""
SOXL Qlib V2 - Kelly + 多级风控
优化：Max DD 控制 + Sharpe 提升
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
    # 使用绝对路径
    qlib_data_path = Path('/Users/rziiiii/.openclaw/workspace/qlib_data/bin')
    
    if not qlib_data_path.exists():
        print(f"❌ Qlib 数据不存在: {qlib_data_path}")
        return False
    
    try:
        qlib.init(provider_uri=str(qlib_data_path), expression_cache=None, dataset_cache=None)
        print(f"✅ Qlib 初始化成功")
        return True
    except Exception as e:
        print(f"❌ Qlib 初始化失败: {e}")
        return False

# ============================================================
# Part 2: Qlib Expression 因子
# ============================================================

def get_qlib_features():
    """Qlib Expression 因子（无 Lookahead Bias）"""
    return [
        '$close',
        '$volume',
        'Mean($close, 5)',
        'Mean($close, 20)',
        'Mean($close, 60)',
        'Ref($close, 5) / $close - 1',
        'Mean($volume, 20)',
        'Std($close / Ref($close, 1) - 1, 20)',  # 20日波动率
    ]

# ============================================================
# Part 3: Kelly 仓位管理
# ============================================================

def kelly_position(returns, window=50, max_kelly=0.40):
    """Kelly 公式（保守版）"""
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
    
    # 半 Kelly
    kelly = kelly * 0.5
    
    # 波动率调整
    vol = np.std(recent)
    if vol > 0.03:
        kelly *= 0.7
    
    return np.clip(kelly, 0.10, max_kelly)

# ============================================================
# Part 4: 信号计算（V25 核心）
# ============================================================

def calculate_signal(df):
    """计算 V25 风格信号"""
    trend = (
        (df['$close'] > df['Mean($close, 5)']).astype(float) * 0.2 +
        (df['$close'] > df['Mean($close, 20)']).astype(float) * 0.3 +
        (df['$close'] > df['Mean($close, 60)']).astype(float) * 0.3 +
        (df['Mean($close, 5)'] > df['Mean($close, 20)']).astype(float) * 0.2
    )
    
    rev = -df['Ref($close, 5) / $close - 1']
    vol_conf = (df['$volume'] / (df['Mean($volume, 20)'] + 1e-9) > 1.5).astype(float) * 0.5 + 0.5
    
    signal = trend * 0.5 + rev * 0.3 + vol_conf * 0.2
    return signal.clip(-1, 1)

# ============================================================
# Part 5: 回测引擎（V2 风控）
# ============================================================

def backtest_v2(df, signal, initial_balance=100000, cost=0.001):
    """V2 回测：Kelly + 多级风控"""
    
    close = df['$close']
    ret = close.pct_change()
    
    # 状态变量
    balance = initial_balance
    position = 0.0
    peak_value = initial_balance
    
    # Kelly 状态
    trade_returns = []
    current_kelly = 0.20
    
    # 信号平滑
    last_position = 0.0
    
    # 结果记录
    portfolio_values = [initial_balance]
    positions = [0.0]
    
    # 8级回撤保护
    dd_levels = [-0.04, -0.06, -0.08, -0.10, -0.12, -0.14, -0.16, -0.18]
    dd_mults = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    # 波动率列
    vol_series = df['Std($close / Ref($close, 1) - 1, 20)']
    
    # 遍历
    for i in range(60, len(df)):
        current_signal = signal.iloc[i]
        
        # Kelly 限制
        kelly_limit = current_kelly
        
        # 回撤保护
        current_value = balance
        if current_value > peak_value:
            peak_value = current_value
        
        current_dd = (current_value - peak_value) / peak_value
        
        dd_mult = 1.0
        for level, mult in zip(dd_levels, dd_mults):
            if current_dd < level:
                dd_mult = mult
                break
        
        # 波动率限制
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
        
        # 目标仓位
        target_position = current_signal * kelly_limit * dd_mult * vol_mult
        target_position = np.clip(target_position, -kelly_limit, kelly_limit)
        
        # EMA 平滑
        target_position = 0.7 * target_position + 0.3 * last_position
        last_position = target_position
        
        # 执行交易
        current_price = close.iloc[i]
        
        if i < len(df) - 1:
            next_price = close.iloc[i + 1]
            
            # 交易成本
            position_change = abs(target_position - position)
            trade_cost = position_change * cost
            
            # 收益
            price_return = (next_price - current_price) / current_price
            portfolio_return = target_position * price_return - trade_cost
            
            # 更新
            balance *= (1 + portfolio_return)
            position = target_position
            
            # 记录
            trade_returns.append(portfolio_return)
            portfolio_values.append(balance)
            positions.append(position)
            
            # 更新 Kelly
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
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trade_count': trade_count,
        'final_kelly': current_kelly
    }

# ============================================================
# Part 6: 主流程
# ============================================================

def main():
    print("=" * 60)
    print("SOXL Qlib V2 - Kelly + 多级风控")
    print("=" * 60)
    
    if not init_qlib():
        return
    
    instruments = ['SOXL']
    features = get_qlib_features()
    
    print(f"\n📊 加载 Qlib 数据...")
    
    try:
        df = D.features(instruments=instruments, fields=features,
                       start_time='2020-01-01', end_time='2026-03-19')
        print(f"✅ 数据: {len(df)} 条")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 计算信号
    signal = calculate_signal(df)
    
    # V2 回测
    stats = backtest_v2(df, signal)
    
    print(f"\n📈 Qlib V2 结果:")
    print(f"  总收益: {stats['total_return']:.2%}")
    print(f"  Sharpe: {stats['sharpe']:.2f}")
    print(f"  Max DD: {stats['max_dd']:.2%}")
    print(f"  胜率: {stats['win_rate']:.2%}")
    print(f"  交易: {stats['trade_count']}")
    print(f"  最终 Kelly: {stats['final_kelly']:.2%}")
    
    # 对比
    print(f"\n📊 对比 V1:")
    print(f"  V1: Sharpe 0.33, 收益 36.43%, Max DD -61.90%")
    print(f"  V2: Sharpe {stats['sharpe']:.2f}, 收益 {stats['total_return']:.2%}, Max DD {stats['max_dd']:.2%}")
    
    return stats

if __name__ == "__main__":
    main()
