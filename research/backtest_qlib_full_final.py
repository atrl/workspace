#!/usr/bin/env python3
"""
完整的 qlib 官方回测系统 + DQN 模型（2020-2026）

使用 qlib 官方 API：
1. ✅ qlib.init()
2. ✅ D.features()
3. ✅ BaseStrategy
4. ✅ SimulatorExecutor
5. ✅ backtest.backtest()
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
print("完整的 qlib 官方回测系统 + DQN 模型（2020-2026）")
print("="*70)

# 1. 初始化 Qlib
qlib.init(provider_uri="/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary")
print("✅ Qlib 初始化成功")

# 2. 定义 RL 策略
class RLStrategy(BaseStrategy):
    """使用 DQN 模型的 qlib 策略"""
    
    def __init__(self, model_path, instrument="SOXL", **kwargs):
        super().__init__(**kwargs)
        
        self.model = DQN.load(model_path)
        self.instrument = instrument
        print(f"✅ 加载 DQN 模型: {model_path}")
    
    def generate_trade_decision(self, execute_result=None):
        """qlib 核心接口：生成交易决策"""
        
        # 获取当前交易日
        trade_step = self.trade_calendar.get_trade_step()
        total_steps = self.trade_calendar.get_trade_len()
        
        # 最后两天不交易（避免日历索引越界）
        if trade_step >= total_steps - 2:
            return TradeDecisionWO([], self)
        
        trade_date = self.trade_calendar.get_step_time(trade_step)[0]
        next_date = self.trade_calendar.get_step_time(trade_step + 1)[0]
        
        # 加载历史数据
        df = D.features(
            instruments=[self.instrument],
            fields=['$close', '$volume', '$vix'],
            start_time=pd.Timestamp(trade_date) - pd.Timedelta(days=100),
            end_time=trade_date
        )
        
        if df.empty:
            return TradeDecisionWO([], self)
        
        # 提取数据
        close = df['$close'].values
        volume = df['$volume'].values
        vix = df['$vix'].values
        
        # 构建观察（17维）
        ret = (close[-1] - close[-2]) / close[-2] if close[-2] > 0 else 0
        ma5 = np.mean(close[-5:])
        ma20 = np.mean(close[-20:])
        
        obs = np.array([
            ret,
            ma5 / ma20 - 1 if ma20 > 0 else 0,
            ret,
            np.std(close) * np.sqrt(252) if len(close) > 1 else 0,
            vix[-1],
            (vix[-1] - np.mean(vix)) / (np.std(vix) + 1e-8) if len(vix) > 0 else 0,
            np.mean(vix),
            0.5,
            0,
            np.log(volume[-1] / (np.mean(volume) + 1e-8)) if len(volume) > 0 and np.mean(volume) > 0 else 0,
            0,
            0,
            trade_step / total_steps,
            0, 0, 0, 0
        ], dtype=np.float32).reshape(1, -1)
        
        # 预测动作
        action, _ = self.model.predict(obs, deterministic=True)
        target_pos = int(action.item()) * 0.1
        
        # 生成订单
        order = Order(
            stock_id=self.instrument,
            amount=target_pos,
            direction=Order.BUY if target_pos > 0 else Order.SELL,
            start_time=trade_date,
            end_time=next_date
        )
        
        return TradeDecisionWO([order], self)

# 3. 配置回测
start_time = "2020-01-02"
end_time = "2026-03-31"  # 提前2天结束，避免日历索引越界
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
    strategy = RLStrategy(
        model_path="models/qlib_rl_v2/dqn_model.zip",
        instrument="SOXL"
    )
    
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
    
    # 提取关键指标
    if isinstance(portfolio_metrics, pd.DataFrame):
        final_value = portfolio_metrics['account'].iloc[-1]
        total_return = (final_value / account - 1) * 100
        
        print(f"\n最终资产: ${final_value:,.2f}")
        print(f"总收益: {total_return:.2f}%")
        print(f"收益倍数: {final_value / account:.2f}x")
    
    print(f"\n交易次数: {len(indicator_metrics)}")
    
    print("\n✅✅✅ 成功使用 qlib 官方回测系统 + DQN 模型!")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("完成")
print("="*70)
