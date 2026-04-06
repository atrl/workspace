#!/usr/bin/env python3
"""
最简单的 qlib 官方回测演示

使用 qlib 最基础的 API：
1. qlib.init()
2. D.features()
3. qlib.backtest.backtest()
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import TradeDecisionWO, Order
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest import backtest
from stable_baselines3 import DQN

print("="*70)
print("最简单的 qlib 官方回测演示")
print("="*70)

# 1. 初始化 Qlib
qlib.init(provider_uri="/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary")
print("✅ Qlib 初始化成功")

# 2. 简单策略 - 每天买入10% SOXL
class SimpleBuyStrategy(BaseStrategy):
    """简单的定期买入策略"""
    
    def __init__(self, instrument="SOXL", **kwargs):
        super().__init__(**kwargs)
        self.instrument = instrument
        self.bought = False
    
    def generate_trade_decision(self, execute_result=None):
        """生成交易决策"""
        trade_step = self.trade_calendar.get_trade_step()
        total_steps = self.trade_calendar.get_trade_len()
        
        # 最后一天不交易
        if trade_step >= total_steps - 1:
            return TradeDecisionWO([], self)
        
        # 第一天买入100股，之后持有
        if not self.bought:
            trade_date = self.trade_calendar.get_step_time(trade_step)[0]
            next_date = self.trade_calendar.get_step_time(trade_step + 1)[0] if trade_step + 1 < total_steps else trade_date
            
            order = Order(
                stock_id=self.instrument,
                amount=100,
                direction=Order.BUY,
                start_time=trade_date,
                end_time=next_date
            )
            self.bought = True
            return TradeDecisionWO([order], self)
        
        return TradeDecisionWO([], self)

# 3. 配置回测
start_time = "2020-01-02"
end_time = "2020-03-31"  # 只测试3个月
benchmark = "SOXL"
account = 100000

exchange_kwargs = {
    "freq": "day",
    "deal_price": "close",
    "open_cost": 1.001,
    "close_cost": 1.001,
    "min_cost": 5,
}

print(f"\n✅ 回测时间: {start_time} to {end_time}")
print(f"✅ 初始资金: ${account}")
print(f"✅ 基准: {benchmark}")

# 4. 运行回测
print("\n" + "="*70)
print("运行 qlib 官方回测引擎")
print("="*70)

try:
    strategy = SimpleBuyStrategy(instrument="SOXL")
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True
    )
    
    print("✅ 开始回测...")
    
    portfolio_metrics, indicator_metrics = backtest(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        executor=executor,
        benchmark=benchmark,
        account=account,
        exchange_kwargs=exchange_kwargs
    )
    
    print("\n" + "="*70)
    print("✅ Qlib 官方回测完成!")
    print("="*70)
    
    print(f"\n组合指标:")
    for key, value in portfolio_metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\n交易指标:")
    for key, value in indicator_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n✅✅✅ 成功使用 qlib 官方回测系统！")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("完成")
print("="*70)
