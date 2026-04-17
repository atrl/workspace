#!/usr/bin/env python3
"""
Prepare RL Training Data - Qlib Native

准备强化学习训练数据，包含：
1. 基础OHLCV数据
2. 技术指标特征
3. 外部数据（VIX）
4. 归一化统计量
"""
import argparse
import pickle
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import numpy as np


def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """加载价格数据"""
    # 尝试多个可能的路径
    possible_paths = [
        Path(f"data/soxl_with_vix.csv"),  # workspace/data/
        Path(f"data/{ticker.lower()}_with_vix.csv"),
        Path(f"data/symbols/{ticker}.csv"),
        Path(f"../data/soxl_with_vix.csv"),  # skills/qlib-rl-trading/../data/
        Path(f"../../data/soxl_with_vix.csv"),  # skills/../data/
        Path(f"/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv"),  # 绝对路径
    ]
    
    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path, parse_dates=['date'])
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            return df
    
    raise FileNotFoundError(f"No data found for {ticker}. Tried: {[str(p) for p in possible_paths]}")


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    df = df.copy()
    close = df['close']
    volume = df['volume'] if 'volume' in df.columns else df.get('volume', 1)
    
    # 收益率特征
    df['return_1d'] = close.pct_change(1)
    df['return_5d'] = close.pct_change(5)
    df['return_20d'] = close.pct_change(20)
    
    # Z-score化（避免尺度问题）
    df['return_1d_zscore'] = (df['return_1d'] - df['return_1d'].rolling(60).mean()) / (df['return_1d'].rolling(60).std() + 1e-8)
    df['return_5d_zscore'] = (df['return_5d'] - df['return_5d'].rolling(60).mean()) / (df['return_5d'].rolling(60).std() + 1e-8)
    
    # 波动率特征
    returns = close.pct_change()
    df['volatility_5d'] = returns.rolling(5).std() * np.sqrt(252)
    df['volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
    df['volatility_60d'] = returns.rolling(60).std() * np.sqrt(252)
    
    # 波动率分位数（避免归一化问题）
    df['volatility_20d_rank'] = df['volatility_20d'].rolling(60).rank(pct=True)
    
    # 成交量特征
    if 'volume' in df.columns:
        df['volume_zscore'] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-8)
        df['volume_change'] = volume.pct_change()
    else:
        df['volume_zscore'] = 0
        df['volume_change'] = 0
    
    # 均线偏离
    df['ma_5'] = close.rolling(5).mean()
    df['ma_20'] = close.rolling(20).mean()
    df['ma_60'] = close.rolling(60).mean()
    df['price_to_ma5'] = (close - df['ma_5']) / df['ma_5']
    df['price_to_ma20'] = (close - df['ma_20']) / df['ma_20']
    df['price_to_ma60'] = (close - df['ma_60']) / df['ma_60']
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['bb_upper'] = df['ma_20'] + 2 * df['volatility_20d'] * df['ma_20'] / np.sqrt(252)
    df['bb_lower'] = df['ma_20'] - 2 * df['volatility_20d'] * df['ma_20'] / np.sqrt(252)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma_20']
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    return df


def add_external_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """添加外部特征（VIX等）"""
    df = df.copy()
    
    # 如果数据中已有VIX
    if 'vix' in df.columns:
        vix = df['vix']
        df['vix_zscore'] = (vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-8)
        df['vix_percentile_20d'] = vix.rolling(20).rank(pct=True)
        
        # 恐慌指数（简化版）
        df['panic_score'] = np.clip((vix - 20) / 10, -2, 3)
    else:
        # 使用历史波动率作为VIX替代
        df['vix_zscore'] = df['volatility_20d_zscore'] = (
            (df['volatility_20d'] - df['volatility_20d'].rolling(60).mean()) / 
            (df['volatility_20d'].rolling(60).std() + 1e-8)
        )
        df['vix_percentile_20d'] = df['volatility_20d_rank']
        df['panic_score'] = np.clip(df['volatility_20d_rank'] - 0.5, -1, 2)
    
    return df


def calculate_normalization_stats(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """计算归一化统计量"""
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                }
    return stats


def prepare_rl_data(
    ticker: str,
    start: str,
    end: str,
    output_dir: str = "data/rl",
    include_vix: bool = True,
) -> Dict:
    """准备RL数据"""
    print(f"Preparing RL data for {ticker} ({start} to {end})...")
    
    # 1. 加载数据
    df = load_price_data(ticker, start, end)
    print(f"  Loaded {len(df)} records")
    
    # 2. 计算技术指标
    df = calculate_technical_features(df)
    print("  Calculated technical features")
    
    # 3. 添加外部特征
    if include_vix:
        df = add_external_features(df, ticker)
        print("  Added external features (VIX)")
    
    # 4. 定义特征列
    feature_cols = [
        # 市场特征
        'return_1d_zscore', 'return_5d_zscore',
        'volatility_20d_rank',
        'volume_zscore',
        'bb_width', 'bb_position',
        'rsi_14',
        
        # 外部特征
        'vix_zscore', 'vix_percentile_20d', 'panic_score',
    ]
    
    # 5. 计算归一化统计量
    norm_stats = calculate_normalization_stats(df, feature_cols)
    print(f"  Calculated normalization stats for {len(norm_stats)} features")
    
    # 6. 保存数据
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data = {
        'df': df,
        'feature_cols': feature_cols,
        'norm_stats': norm_stats,
        'ticker': ticker,
        'start': start,
        'end': end,
    }
    
    output_file = output_path / f"{ticker.lower()}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved to {output_file}")
    
    # 7. 保存CSV（便于检查）
    csv_file = output_path / f"{ticker.lower()}_features.csv"
    df.to_csv(csv_file, index=False)
    print(f"  Saved features to {csv_file}")
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare RL training data")
    parser.add_argument('--ticker', required=True, help="Ticker symbol (e.g., SOXL)")
    parser.add_argument('--start', default='2020-01-01', help="Start date")
    parser.add_argument('--end', default='2026-04-01', help="End date")
    parser.add_argument('--output-dir', default='data/rl', help="Output directory")
    parser.add_argument('--include-vix', action='store_true', help="Include VIX features")
    
    args = parser.parse_args()
    
    data = prepare_rl_data(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        include_vix=args.include_vix,
    )
    
    print("\nData preparation complete!")
    print(f"Features: {len(data['feature_cols'])}")
    print(f"Records: {len(data['df'])}")
