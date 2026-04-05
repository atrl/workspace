#!/usr/bin/env python3
"""
分层RL训练脚本

参考之前的动作空间设计：
- 动作：[short_ratio, long_ratio] ∈ [0, 1] × [0, 1]
- RL仓位 = long_ratio - short_ratio ∈ [-1, 1]
- 最终仓位 = 规则信号 × RL仓位
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from rl_hierarchical_env import SOXLHierarchicalTradingEnv


def train_hierarchical_rl(
    total_timesteps: int = 200000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    eval_freq: int = 10000,
    save_dir: str = 'models/hierarchical_rl',
):
    """
    训练分层RL智能体
    """
    
    print("="*70)
    print("训练分层RL智能体")
    print("="*70)
    print(f"""
配置：
  - 动作空间: [short_ratio, long_ratio]
  - 状态空间: 11维
  - 学习率: {learning_rate}
  - 总步数: {total_timesteps:,}
  - 保存目录: {save_dir}
    """)
    
    # 1. 创建环境
    train_env = DummyVecEnv([lambda: SOXLHierarchicalTradingEnv()])
    eval_env = DummyVecEnv([lambda: SOXLHierarchicalTradingEnv()])
    
    # 2. 创建模型
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device='auto',
    )
    
    # 3. 创建回调
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best/",
        log_path=f"{save_dir}/logs/",
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    
    # 4. 训练
    print("开始训练...")
    model.learn(
        total_timesteps=total_timesteps,
        # callback=eval_callback,  # 简化：去掉callback
        # progress_bar=True,  # 简化：去掉进度条
    )
    
    # 5. 保存最终模型
    model.save(f"{save_dir}/final_model")
    print(f"\n✅ 模型已保存: {save_dir}/final_model.zip")
    
    return model


def backtest_rl_strategy(model, env) -> Dict:
    """
    回测RL策略
    """
    
    obs, _ = env.reset()
    done = False
    
    portfolio_values = [100000]
    positions = []
    rule_signals = []
    rl_positions = []
    
    while not done:
        # RL预测
        action, _ = model.predict(obs, deterministic=True)
        
        # 执行
        obs, reward, done, truncated, info = env.step(action)
        
        # 记录
        portfolio_values.append(info['total_value'])
        positions.append(info['position'])
        rule_signals.append(info['rule_signal'])
        rl_positions.append(info['rl_position'])
    
    # 计算指标
    pv = np.array(portfolio_values)
    total_return = (pv[-1] - 100000) / 100000
    
    daily_returns = np.diff(pv) / pv[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    
    max_dd = np.max((np.maximum.accumulate(pv) - pv) / np.maximum.accumulate(pv))
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final_value': pv[-1],
        'positions': positions,
        'rule_signals': rule_signals,
        'rl_positions': rl_positions,
        'portfolio_values': pv,
    }


def backtest_hardcoded_strategy(env) -> Dict:
    """
    回测硬参数策略（基准）
    """
    
    obs, _ = env.reset()
    done = False
    
    portfolio_values = [100000]
    
    # 使用硬参数策略
    while not done:
        # 获取规则信号
        rule_signal = env._get_rule_signal()
        
        # 硬参数：直接用规则信号作为仓位
        # 转换为RL动作
        if rule_signal >= 0.5:
            # 做多
            action = np.array([0.0, rule_signal])
        else:
            # 减仓
            action = np.array([1 - rule_signal, 0.0])
        
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(info['total_value'])
    
    # 计算指标
    pv = np.array(portfolio_values)
    total_return = (pv[-1] - 100000) / 100000
    
    daily_returns = np.diff(pv) / pv[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    
    max_dd = np.max((np.maximum.accumulate(pv) - pv) / np.maximum.accumulate(pv))
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final_value': pv[-1],
        'portfolio_values': pv,
    }


def compare_strategies(rl_results: Dict, hardcoded_results: Dict):
    """
    对比策略
    """
    
    print("\n" + "="*70)
    print("策略对比")
    print("="*70)
    print()
    
    print(f"{'指标':<20} {'硬参数策略':<15} {'RL策略':<15} {'改进':<15}")
    print("-"*70)
    
    # 总收益
    print(f"{'总收益':<20} {hardcoded_results['total_return']:>13.2%} "
          f"{rl_results['total_return']:>13.2%} "
          f"{rl_results['total_return'] - hardcoded_results['total_return']:>+13.2%}")
    
    # Sharpe
    print(f"{'Sharpe':<20} {hardcoded_results['sharpe']:>13.2f} "
          f"{rl_results['sharpe']:>13.2f} "
          f"{rl_results['sharpe'] - hardcoded_results['sharpe']:>+13.2f}")
    
    # 最大回撤
    print(f"{'最大回撤':<20} {hardcoded_results['max_dd']:>13.2%} "
          f"{rl_results['max_dd']:>13.2%} "
          f"{rl_results['max_dd'] - hardcoded_results['max_dd']:>+13.2%}")
    
    # 最终价值
    print(f"{'最终价值':<20} ${hardcoded_results['final_value']:>11,.0f} "
          f"${rl_results['final_value']:>11,.0f} "
          f"${rl_results['final_value'] - hardcoded_results['final_value']:>+11,.0f}")
    
    # 收益倍数
    print(f"{'收益倍数':<20} {hardcoded_results['final_value']/100000:>13.2f}x "
          f"{rl_results['final_value']/100000:>13.2f}x "
          f"{(rl_results['final_value'] - hardcoded_results['final_value'])/100000:>+13.2f}x")
    
    print("="*70)
    
    # 分析RL行为
    if len(rl_results['positions']) > 0:
        print("\n" + "="*70)
        print("RL 行为分析")
        print("="*70)
        
        positions = np.array(rl_results['positions'])
        rule_signals = np.array(rl_results['rule_signals'])
        rl_positions = np.array(rl_results['rl_positions'])
        
        print(f"平均仓位:           {np.mean(positions):.2f}")
        print(f"仓位标准差:         {np.std(positions):.2f}")
        print(f"平均规则信号:       {np.mean(rule_signals):.2f}")
        print(f"平均RL调整:         {np.mean(rl_positions):.2f}")
        print()
        
        # RL是否在调整？
        print(f"RL调整范围:         [{np.min(rl_positions):.2f}, {np.max(rl_positions):.2f}]")
        print(f"RL调整幅度:         {np.std(rl_positions):.2f}")
        
        # RL是否在做空？
        short_ratio = np.sum(rl_positions < 0) / len(rl_positions)
        print(f"RL做空比例:         {short_ratio:.2%}")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    
    # 1. 训练
    model = train_hierarchical_rl(
        total_timesteps=200000,
        learning_rate=3e-4,
        save_dir='models/hierarchical_rl',
    )
    
    # 2. 回测RL策略
    print("\n" + "="*70)
    print("回测 RL 策略")
    print("="*70)
    
    env = SOXLHierarchicalTradingEnv()
    rl_results = backtest_rl_strategy(model, env)
    
    # 3. 回测硬参数策略
    print("\n" + "="*70)
    print("回测硬参数策略（基准）")
    print("="*70)
    
    env = SOXLHierarchicalTradingEnv()
    hardcoded_results = backtest_hardcoded_strategy(env)
    
    # 4. 对比
    compare_strategies(rl_results, hardcoded_results)
    
    # 5. 保存结果
    import json
    
    results = {
        'rl': {
            'total_return': rl_results['total_return'],
            'sharpe': rl_results['sharpe'],
            'max_dd': rl_results['max_dd'],
            'final_value': rl_results['final_value'],
        },
        'hardcoded': {
            'total_return': hardcoded_results['total_return'],
            'sharpe': hardcoded_results['sharpe'],
            'max_dd': hardcoded_results['max_dd'],
            'final_value': hardcoded_results['final_value'],
        }
    }
    
    with open('models/hierarchical_rl/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ 训练完成！结果已保存到 models/hierarchical_rl/")
