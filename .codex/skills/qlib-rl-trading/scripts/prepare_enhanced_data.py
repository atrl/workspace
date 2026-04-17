#!/usr/bin/env python3
"""
Prepare Enhanced RL Data - 使用完整历史数据

数据源：
1. ohlcv_master.csv (2010-2026, 16年数据)
2. VIX 数据
3. 更多技术指标特征
"""
import argparse
import pickle
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np


def load_soxl_full(data_dir: str = "/Users/rziiiii/.openclaw-gemini/workspace/data") -> pd.DataFrame:
    """加载完整的SOXL数据（2010-2026）"""
    
    master_file = Path(data_dir) / "ohlcv_master.csv"
    
    # 读取全部数据
    df = pd.read_csv(master_file, parse_dates=['date'])
    
    # 过滤SOXL
    df = df[df['instrument'] == 'SOXL'].copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Loaded SOXL: {len(df)} records ({df['date'].min()} to {df['date'].max()})")
    
    return df


def load_vix_data(workspace_dir: str = "/Users/rziiiii/.openclaw/workspace") -> pd.DataFrame:
    """加载VIX数据"""
    
    vix_file = Path(workspace_dir) / "data/vix.csv"
    
    if vix_file.exists():
        df = pd.read_csv(vix_file, parse_dates=['Date'])
        df = df.rename(columns={'Date': 'date', 'Close': 'vix'})
        df = df[['date', 'vix']]
        print(f"Loaded VIX: {len(df)} records")
        return df
    
    print("VIX data not found, will use HV proxy")
    return None


def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算高级技术指标"""
    df = df.copy()
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # 1. 收益率特征
    for period in [1, 5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)
        df[f'return_{period}d_zscore'] = (
            (df[f'return_{period}d'] - df[f'return_{period}d'].rolling(60).mean()) / 
            (df[f'return_{period}d'].rolling(60).std() + 1e-8)
        )
    
    # 2. 波动率特征
    returns = close.pct_change()
    for period in [5, 10, 20, 60]:
        vol = returns.rolling(period).std() * np.sqrt(252)
        df[f'volatility_{period}d'] = vol
        df[f'volatility_{period}d_rank'] = vol.rolling(120).rank(pct=True)
    
    # 3. 成交量特征
    for period in [5, 10, 20]:
        df[f'volume_{period}d_ma'] = volume.rolling(period).mean()
        df[f'volume_zscore_{period}d'] = (
            (volume - df[f'volume_{period}d_ma']) / 
            (volume.rolling(period).std() + 1e-8)
        )
    
    # 4. 均线特征
    for period in [5, 10, 20, 60, 120]:
        df[f'ma_{period}'] = close.rolling(period).mean()
        df[f'price_to_ma{period}'] = (close - df[f'ma_{period}']) / df[f'ma_{period}']
    
    # 5. RSI
    for period in [6, 14, 28]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # 6. MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 7. 布林带
    for period in [10, 20]:
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        df[f'bb_upper_{period}'] = ma + 2 * std
        df[f'bb_lower_{period}'] = ma - 2 * std
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / ma
        df[f'bb_position_{period}'] = (close - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-8)
    
    # 8. ATR (Average True Range)
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / close
    
    # 9. 动量指标
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = close / close.shift(period) - 1
        df[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
    
    # 10. 均线斜率
    for period in [5, 10, 20]:
        ma = df[f'ma_{period}']
        df[f'ma{period}_slope'] = (ma - ma.shift(5)) / ma.shift(5)
    
    # 11. 价格位置
    for period in [20, 60]:
        rolling_high = high.rolling(period).max()
        rolling_low = low.rolling(period).min()
        df[f'price_position_{period}d'] = (close - rolling_low) / (rolling_high - rolling_low + 1e-8)
    
    return df


def merge_vix(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """合并VIX数据"""
    if vix_df is None:
        # 使用HV作为VIX代理
        df['vix'] = df['volatility_20d'] * 100  # 近似
        df['vix_zscore'] = (
            (df['vix'] - df['vix'].rolling(60).mean()) / 
            (df['vix'].rolling(60).std() + 1e-8)
        )
        df['vix_percentile_20d'] = df['vix'].rolling(20).rank(pct=True)
        return df
    
    df = df.merge(vix_df, on='date', how='left')
    
    # 填充缺失的VIX数据
    df['vix'] = df['vix'].fillna(method='ffill')
    
    # VIX特征
    df['vix_zscore'] = (
        (df['vix'] - df['vix'].rolling(60).mean()) / 
        (df['vix'].rolling(60).std() + 1e-8)
    )
    df['vix_percentile_20d'] = df['vix'].rolling(20).rank(pct=True)
    df['vix_percentile_60d'] = df['vix'].rolling(60).rank(pct=True)
    
    # 恐慌指数（简化版）
    df['panic_score'] = np.clip((df['vix'] - 20) / 10, -2, 3)
    
    return df


def prepare_enhanced_rl_data(
    ticker: str = "SOXL",
    output_dir: str = "data/rl",
    start: str = "2010-01-01",
    end: str = "2026-12-31",
) -> Dict:
    """准备增强版RL数据"""
    
    print(f"Preparing enhanced RL data for {ticker}...")
    
    # 1. 加载SOXL数据
    df = load_soxl_full()
    
    # 2. 过滤日期
    df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
    print(f"Filtered: {len(df)} records")
    
    # 3. 计算高级特征
    df = calculate_advanced_features(df)
    print("Calculated advanced features")
    
    # 4. 加载并合并VIX
    vix_df = load_vix_data()
    df = merge_vix(df, vix_df)
    print("Merged VIX data")
    
    # 5. 定义特征列（选择最重要的特征）
    feature_cols = [
        # 收益率
        'return_1d_zscore',
        'return_5d_zscore',
        'return_10d_zscore',
        
        # 波动率
        'volatility_20d_rank',
        'volatility_60d_rank',
        
        # 成交量
        'volume_zscore_20d',
        
        # 均线
        'price_to_ma5',
        'price_to_ma20',
        'price_to_ma60',
        
        # RSI
        'rsi_6',
        'rsi_14',
        
        # MACD
        'macd_hist',
        
        # 布林带
        'bb_width_20',
        'bb_position_20',
        
        # ATR
        'atr_ratio',
        
        # 动量
        'momentum_10',
        'momentum_20',
        
        # 均线斜率
        'ma5_slope',
        'ma20_slope',
        
        # 价格位置
        'price_position_20d',
        'price_position_60d',
        
        # VIX
        'vix_zscore',
        'vix_percentile_20d',
        'vix_percentile_60d',
        'panic_score',
    ]
    
    # 6. 计算归一化统计量
    norm_stats = {}
    for col in feature_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                norm_stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                }
    
    print(f"Calculated normalization stats for {len(norm_stats)} features")
    
    # 7. 保存数据
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data = {
        'df': df,
        'feature_cols': feature_cols,
        'norm_stats': norm_stats,
        'ticker': ticker,
        'start': str(df['date'].min()),
        'end': str(df['date'].max()),
    }
    
    output_file = output_path / f"{ticker.lower()}_enhanced.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {output_file}")
    
    # 8. 保存CSV
    csv_file = output_path / f"{ticker.lower()}_enhanced.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved CSV to {csv_file}")
    
    # 9. 打印统计
    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)
    print(f"Records:     {len(df)}")
    print(f"Date Range:  {df['date'].min()} to {df['date'].max()}")
    print(f"Features:    {len(feature_cols)}")
    print(f"Years:       {(df['date'].max() - df['date'].min()).days / 365:.1f}")
    print("="*60)
    
    return data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='SOXL')
    parser.add_argument('--output-dir', default='data/rl')
    parser.add_argument('--start', default='2010-01-01')
    parser.add_argument('--end', default='2026-12-31')
    
    args = parser.parse_args()
    
    prepare_enhanced_rl_data(
        ticker=args.ticker,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
    )
