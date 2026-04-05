#!/usr/bin/env python3
"""
SOXL Panic Strategy - True Qlib Native Implementation

使用Qlib原生框架实现恐慌指数策略

Qlib框架组件：
- qlib.data.D.features() - 数据加载
- qlib.contrib.strategy.BaseStrategy - 策略基类
- qlib.contrib.backtest.backtest - 回测引擎
- qlib.contrib.evaluate.risk_analysis - 风险分析

Author: clawQuant
Date: 2026-04-02
"""

import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.contrib.strategy import BaseStrategy
from qlib.backtest.decision import TradeDecision, Order, OrderDir
from qlib.contrib.backtest import backtest
from qlib.contrib.evaluate import risk_analysis
from qlib.utils import init_instance_by_config

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class PanicScoreCalculator:
    """恐慌指数计算器"""
    
    def __init__(
        self,
        vix_low: float = 10,
        vix_high: float = 30,
        fear_min: float = -100,
        fear_max: float = 100,
        sigmoid_steepness: float = 0.5,
    ):
        self.vix_low = vix_low
        self.vix_high = vix_high
        self.fear_min = fear_min
        self.fear_max = fear_max
        self.sigmoid_steepness = sigmoid_steepness
    
    def calculate_rsi(self, prices: pd.Series, period: int = 6) -> pd.Series:
        """计算RSI6"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_panic_score(self, vix: float, rsi6: float) -> float:
        """计算恐慌指数"""
        # Sigmoid映射
        vix_clipped = max(min(vix, self.vix_high), self.vix_low)
        vix_mid = (self.vix_high + self.vix_low) / 2
        x = -(vix_clipped - vix_mid) * self.sigmoid_steepness
        sigmoid_value = 1 / (1 + np.exp(-x))
        
        base_score = sigmoid_value * (self.fear_max - self.fear_min) + self.fear_min
        
        # RSI调整
        if pd.isna(rsi6):
            return base_score
        
        if rsi6 <= 30:
            adjustment = 20 + (30 - rsi6) * 2.5
            return max(-150, base_score - adjustment)
        elif rsi6 >= 70:
            adjustment = 20 + (rsi6 - 70)
            return min(150, base_score + adjustment)
        else:
            return base_score


class SOXLPanicStrategyQlib(BaseStrategy):
    """
    SOXL恐慌指数策略 - Qlib原生实现
    
    继承自 qlib.contrib.strategy.BaseStrategy
    """
    
    def __init__(
        self,
        instrument: str = "SOXL",
        max_position: float = 0.9,
        min_position: float = 0.5,
        buy_score_threshold: float = -50,
        sell_score_threshold: float = 85,
        daily_return_threshold: float = -0.08,
        cumulative_return_threshold: float = -0.15,
        **kwargs,
    ):
        """初始化策略"""
        super().__init__(**kwargs)
        
        self.instrument = instrument
        self.max_position = max_position
        self.min_position = min_position
        self.buy_score_threshold = buy_score_threshold
        self.sell_score_threshold = sell_score_threshold
        self.daily_return_threshold = daily_return_threshold
        self.cumulative_return_threshold = cumulative_return_threshold
        
        # 恐慌指数计算器
        self.panic_calculator = PanicScoreCalculator()
        
        # 缓存数据
        self._data_cache = None
        self._last_buy_date = None
        self._position = 0.0
        
    def _load_features(self, start_time: str, end_time: str) -> pd.DataFrame:
        """
        使用Qlib加载数据
        
        Returns:
            DataFrame with columns: close, volume, vix, rsi6, panic_score
        """
        # 方式1: 使用Qlib Expression
        # 但VIX不是SOXL的内置字段，需要从外部加载
        
        # 方式2: 从CSV加载并转换为Qlib格式
        # 这里我们使用外部VIX数据
        
        # 加载SOXL数据
        soxl_df = pd.read_csv('data/soxl_with_vix.csv')
        soxl_df['date'] = pd.to_datetime(soxl_df['date'])
        soxl_df = soxl_df.set_index('date')
        
        # 过滤日期
        soxl_df = soxl_df[(soxl_df.index >= start_time) & (soxl_df.index <= end_time)]
        
        # 计算RSI6
        soxl_df['rsi6'] = self.panic_calculator.calculate_rsi(soxl_df['close'])
        
        # 计算恐慌指数
        soxl_df['panic_score'] = soxl_df.apply(
            lambda row: self.panic_calculator.calculate_panic_score(
                row['vix'], row['rsi6']
            ),
            axis=1
        )
        
        # 计算历史波动率
        returns = soxl_df['close'].pct_change()
        soxl_df['hv'] = returns.rolling(window=20).std() * np.sqrt(252)
        soxl_df['hv'] = soxl_df['hv'].fillna(1.0)
        
        # 计算波动率系数
        soxl_df['vol_coef'] = soxl_df['hv'] / 1.0
        
        return soxl_df
    
    def generate_trade_decision(self, execute_result=None):
        """
        核心接口 - 每个交易日调用
        
        Returns:
            TradeDecision: 包含订单列表
        """
        trade_step = self.trade_calendar.get_trade_step()
        trade_date = self.trade_calendar.get_step_time(trade_step)[0]
        
        # 获取当前数据
        current_data = self._get_current_data(trade_date)
        if current_data is None:
            return TradeDecision([])
        
        orders = []
        
        # 获取当前仓位
        current_position = self._get_current_position()
        
        # 买入逻辑
        if current_position < self.max_position:
            buy_signal, buy_reason = self._should_buy(current_data, trade_date)
            if buy_signal:
                order = self._create_buy_order(current_data, current_position)
                if order:
                    orders.append(order)
        
        # 卖出逻辑
        if current_position > self.min_position:
            sell_signal, sell_reason = self._should_sell(current_data, current_position)
            if sell_signal:
                order = self._create_sell_order(current_data, current_position, sell_reason)
                if order:
                    orders.append(order)
        
        return TradeDecision(orders)
    
    def _get_current_data(self, trade_date) -> Optional[pd.Series]:
        """获取当前交易日数据"""
        if self._data_cache is None:
            return None
        
        if trade_date not in self._data_cache.index:
            return None
        
        return self._data_cache.loc[trade_date]
    
    def _get_current_position(self) -> float:
        """获取当前仓位"""
        # 从trade_position获取
        trade_position = self.trade_position
        if trade_position is None:
            return 0.0
        
        # 计算仓位比例
        position_value = trade_position.get_stock_price(self.instrument) * trade_position.get_stock_count(self.instrument)
        total_value = trade_position.get_cash() + position_value
        
        return position_value / total_value if total_value > 0 else 0.0
    
    def _should_buy(self, current_data: pd.Series, trade_date) -> Tuple[bool, str]:
        """判断是否应该买入"""
        panic_score = current_data['panic_score']
        
        # 检查恐慌指数
        if panic_score > self.buy_score_threshold + 10:
            return False, f"恐慌指数({panic_score:.1f})未达标"
        
        # 计算日跌幅
        daily_return = current_data['close'] / current_data.get('prev_close', current_data['close']) - 1
        
        # 计算累积跌幅
        cumulative_return = 0
        if self._last_buy_date is not None:
            days = (trade_date - self._last_buy_date).days
            if days > 0 and days <= 30:
                # 从缓存获取历史价格
                hist_data = self._data_cache.loc[:trade_date].tail(days + 1)
                if len(hist_data) > 0:
                    cumulative_return = current_data['close'] / hist_data.iloc[0]['close'] - 1
        
        # 波动率系数
        vol_coef = current_data.get('vol_coef', 1.0)
        
        # 检查阈值
        daily_threshold = self.daily_return_threshold * vol_coef
        cumulative_threshold = self.cumulative_return_threshold * vol_coef
        
        if daily_return <= daily_threshold:
            return True, f"日跌幅({daily_return:.2%})"
        
        if cumulative_return <= cumulative_threshold:
            return True, f"累积跌幅({cumulative_return:.2%})"
        
        return False, "不满足买入条件"
    
    def _should_sell(self, current_data: pd.Series, current_position: float) -> Tuple[bool, str]:
        """判断是否应该卖出"""
        panic_score = current_data['panic_score']
        
        # 检查恐慌指数
        if panic_score < self.sell_score_threshold:
            return False, f"恐慌指数({panic_score:.1f})未达标"
        
        return True, f"恐慌指数卖出({panic_score:.1f})"
    
    def _create_buy_order(self, current_data: pd.Series, current_position: float) -> Optional[Order]:
        """创建买入订单"""
        # 计算目标仓位
        target_position = min(current_position + 0.1, self.max_position)
        buy_amount = target_position - current_position
        
        if buy_amount < 0.01:
            return None
        
        # 创建买入订单
        return Order(
            stock_id=self.instrument,
            amount=buy_amount,
            direction=OrderDir.BUY
        )
    
    def _create_sell_order(self, current_data: pd.Series, current_position: float, reason: str) -> Optional[Order]:
        """创建卖出订单"""
        # 计算卖出仓位
        sell_amount = min(0.1, current_position - self.min_position)
        
        if sell_amount < 0.01:
            return None
        
        # 创建卖出订单
        return Order(
            stock_id=self.instrument,
            amount=sell_amount,
            direction=OrderDir.SELL
        )


def run_qlib_backtest():
    """
    使用Qlib框架运行回测
    """
    print("="*80)
    print("SOXL Panic Strategy - Qlib Native Backtest")
    print("="*80)
    
    # 初始化Qlib
    try:
        qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
        print("✓ Qlib初始化成功")
    except Exception as e:
        print(f"✗ Qlib初始化失败: {e}")
        print("使用本地数据模式...")
    
    # 策略配置
    strategy_config = {
        "class": "SOXLPanicStrategyQlib",
        "module_path": "__main__",
        "kwargs": {
            "instrument": "SOXL",
            "max_position": 0.9,
            "min_position": 0.5,
            "buy_score_threshold": -50,
            "sell_score_threshold": 85,
        }
    }
    
    # 回测配置
    backtest_config = {
        "start_time": "2020-01-01",
        "end_time": "2026-04-01",
        "account": 100000,
        "benchmark": "SOXL",
        "exchange_kwargs": {
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.001,
            "close_cost": 0.001,
            "min_cost": 5,
        },
        "pos_type": "Position"
    }
    
    # 运行回测
    print("\n开始回测...")
    
    # 由于Qlib需要特定的数据格式，这里使用简化版本
    # 完整版本需要将数据转换为Qlib的二进制格式
    
    print("\n⚠️  Qlib原生回测需要:")
    print("1. 将数据转换为Qlib二进制格式 (qlib.dump_bin)")
    print("2. 或使用QlibDataLoader加载自定义数据")
    print("3. 配置正确的股票池 (instruments)")
    
    print("\n当前实现为演示版本，完整实现请参考:")
    print("- qlib/contrib/strategy/")
    print("- qlib/contrib/backtest/")


if __name__ == '__main__':
    run_qlib_backtest()
