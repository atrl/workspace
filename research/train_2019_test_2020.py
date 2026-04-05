#!/usr/bin/env python3
"""
训练：2010-2019 | 测试：2020-2026

目标：测试模型在未来6年的泛化能力

训练集：2010-03-11 到 2019-12-31 (约9年)
测试集：2020-01-02 到 2026-04-02 (约6年)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

from qlib_rl_hierarchical_env import SOXLQlibHierarchicalEnv, SOXLQlibProvider
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd

print("="*70)
print("训练：2010-2019 | 测试：2020-2026")
print("="*70)
print("""
目标：测试模型在未来6年的泛化能力

时间分割：
  训练集：2010-2019 (约9年, 2479条)
  测试集：2020-2026 (约6年, 1562条)
  
测试集包含：
  - 2020年 COVID 暴跌与反弹
  - 2021年 牛市
  - 2022年 熊市
  - 2023-2024 反弹
  - 2025-2026 波动
    """)

# ============================================================
# 1. 自定义环境（2019年前训练，2020年后测试）
# ============================================================

class SOXLQlibHierarchicalEnvSplit(SOXLQlibHierarchicalEnv):
    """自定义分割的环境"""
    
    def __init__(
        self,
        mode: str = 'train',  # 'train' or 'test'
        split_date: str = '2020-01-01',  # 分割点
        **kwargs
    ):
        self.split_date = pd.Timestamp(split_date)
        self.mode = mode
        super().__init__(mode=mode, **kwargs)
    
    def _load_and_split_data(self):
        """加载并按 2019/2020 分割"""
        # 使用 Qlib Expression 计算特征
        fields = [
            '$close', '$volume', '$vix',
            'Mean($close, 5)', 'Mean($close, 20)',
            'RSI($close, 14)',
        ]
        
        features = self.provider.get_features(fields)
        df = self.provider.load()
        
        # 合并
        for col in features.columns:
            df[col] = features[col]
        
        # 计算额外特征
        df['return_1'] = df['$close'].pct_change()
        df['vix_normalized'] = df['$vix'] / 50
        df['vix_ma20'] = df['$vix'].rolling(20).mean()
        df['vix_std20'] = df['$vix'].rolling(20).std()
        df['vix_zscore'] = (df['$vix'] - df['vix_ma20']) / (df['vix_std20'] + 1e-8)
        
        # 恐慌得分
        df['panic_score'] = (
            df['vix_zscore'] * 40 +
            df['$vix'].rolling(20).rank(pct=True) * 60 +
            np.where(df['return_1'] < 0, np.minimum(df['return_1'] / -0.05, 1) * -50, 0)
        )
        
        df = df.dropna()
        
        # 按 2019/2020 分割
        if self.mode == 'train':
            # 训练集：2010-2019
            self.df = df[df.index < self.split_date].reset_index(drop=True)
            print(f"训练集: {len(self.df)} records ({df[df.index < self.split_date].index.min()} to {df[df.index < self.split_date].index.max()})")
        else:
            # 测试集：2020-2026
            self.df = df[df.index >= self.split_date].reset_index(drop=True)
            print(f"测试集: {len(self.df)} records ({df[df.index >= self.split_date].index.min()} to {df[df.index >= self.split_date].index.max()})")


# ============================================================
# 2. 创建环境
# ============================================================

print("创建环境...")
print()

# 训练环境（2010-2019）
train_env = DummyVecEnv([lambda: SOXLQlibHierarchicalEnvSplit(mode='train')])

# 测试环境（2020-2026）
test_env = SOXLQlibHierarchicalEnvSplit(mode='test')

print()
print("="*70)
print("训练配置")
print("="*70)
print(f"训练数据: 2010-2019 ({len(train_env.envs[0].df)} records)")
print(f"测试数据: 2020-2026 ({len(test_env.df)} records)")
print(f"训练/测试比: {len(train_env.envs[0].df) / len(test_env.df):.2f}x")
print()

# ============================================================
# 3. 创建 PPO 模型
# ============================================================

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=0,
)

# ============================================================
# 4. 训练
# ============================================================

print("开始训练（200k步）...")
model.learn(total_timesteps=200000)
print("✅ 训练完成")
print()

# ============================================================
# 5. 测试（2020-2026，完全未见过的数据）
# ============================================================

print("="*70)
print("测试：2020-2026（完全未见过的6年数据）")
print("="*70)

obs, _ = test_env.reset()
test_values = [100000]
test_positions = []
test_returns = []
test_dates = []

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    
    test_values.append(info['total_value'])
    test_positions.append(info['rl_position'])
    test_returns.append(info['return'])
    
    if test_env.current_step < len(test_env.df):
        test_dates.append(test_env.df.iloc[test_env.current_step]['date'] if 'date' in test_env.df.columns else test_env.current_step)

# 计算测试指标
test_values = np.array(test_values)
test_return = (test_values[-1] - 100000) / 100000
test_daily_returns = np.diff(test_values) / test_values[:-1]
test_sharpe = np.mean(test_daily_returns) / (np.std(test_daily_returns) + 1e-8) * np.sqrt(252)
test_max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

print(f"总收益:      {test_return:.2%}")
print(f"Sharpe:      {test_sharpe:.2f}")
print(f"最大回撤:    {test_max_dd:.2%}")
print(f"最终价值:    ${test_values[-1]:,.0f}")
print(f"收益倍数:    {test_values[-1]/100000:.2f}x")
print()

test_pos = np.array(test_positions)
print(f"平均仓位:    {np.mean(test_pos):.2f}")
print(f"做空比例:    {np.sum(test_pos < 0) / len(test_pos):.2%}")
print(f"仓位范围:    [{np.min(test_pos):.2f}, {np.max(test_pos):.2f}]")

# ============================================================
# 6. 对比：训练集 vs 测试集
# ============================================================

print()
print("="*70)
print("对比：训练集 vs 测试集")
print("="*70)

# 训练集回测
train_env_single = SOXLQlibHierarchicalEnvSplit(mode='train')
obs_train, _ = train_env_single.reset()
train_values = [100000]

done_train = False
while not done_train:
    action_train, _ = model.predict(obs_train, deterministic=True)
    obs_train, reward_train, done_train, truncated_train, info_train = train_env_single.step(action_train)
    train_values.append(info_train['total_value'])

train_values = np.array(train_values)
train_return = (train_values[-1] - 100000) / 100000
train_daily_returns = np.diff(train_values) / train_values[:-1]
train_sharpe = np.mean(train_daily_returns) / (np.std(train_daily_returns) + 1e-8) * np.sqrt(252)
train_max_dd = np.max((np.maximum.accumulate(train_values) - train_values) / np.maximum.accumulate(train_values))

print(f"{'指标':<15} {'训练集 (2010-2019)':>20} {'测试集 (2020-2026)':>20}")
print("-"*70)
print(f"{'总收益':<15} {train_return:>18.2%} {test_return:>18.2%}")
print(f"{'Sharpe':<15} {train_sharpe:>18.2f} {test_sharpe:>18.2f}")
print(f"{'最大回撤':<15} {train_max_dd:>18.2%} {test_max_dd:>18.2%}")
print(f"{'收益倍数':<15} {train_values[-1]/100000:>18.2f}x {test_values[-1]/100000:>18.2f}x")
print("="*70)

# ============================================================
# 7. 过拟合检查
# ============================================================

print()
print("="*70)
print("过拟合检查")
print("="*70)

sharpe_ratio = test_sharpe / train_sharpe if train_sharpe > 0 else 0
return_ratio = test_return / train_return if train_return > 0 else 0

print(f"Sharpe比 (测试/训练):   {sharpe_ratio:.2f}")
print(f"收益比 (测试/训练):     {return_ratio:.2f}")
print()

if sharpe_ratio < 0.5:
    print("❌ 严重过拟合（测试Sharpe < 训练Sharpe * 0.5）")
elif sharpe_ratio < 0.8:
    print("⚠️ 警告：可能存在过拟合（测试Sharpe < 训练Sharpe * 0.8）")
elif sharpe_ratio > 1.2:
    print("✅ 测试表现优于训练（可能市场环境更适合）")
else:
    print("✅ 泛化能力良好")

# ============================================================
# 8. 与买入持有对比
# ============================================================

print()
print("="*70)
print("对比：策略 vs 买入持有（2020-2026）")
print("="*70)

# 计算买入持有收益
df_test = test_env.df
buy_hold_return = (df_test['$close'].iloc[-1] / df_test['$close'].iloc[0] - 1)
buy_hold_value = 100000 * (1 + buy_hold_return)

print(f"{'策略':<20} {'总收益':>15} {'Sharpe':>10} {'最终价值':>15}")
print("-"*70)
print(f"{'RL策略 (2010-2019训练)':<20} {test_return:>13.2%} {test_sharpe:>10.2f} ${test_values[-1]:>13,.0f}")
print(f"{'买入持有':<20} {buy_hold_return:>13.2%} {'-':>10} ${buy_hold_value:>13,.0f}")
print("="*70)

if test_return > buy_hold_return:
    print(f"✅ 策略跑赢买入持有 {(test_return - buy_hold_return):.2%}")
else:
    print(f"❌ 策略跑输买入持有 {(test_return - buy_hold_return):.2%}")

# ============================================================
# 9. 保存模型
# ============================================================

import pathlib
pathlib.Path("models/qlib_rl_2019").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_2019/ppo_model")

print()
print("="*70)
print("✅ 训练完成！")
print("="*70)
print(f"模型已保存: models/qlib_rl_2019/ppo_model.zip")
print()
print("关键发现：")
print(f"  训练集 (2010-2019): {train_return:.2%}, Sharpe {train_sharpe:.2f}")
print(f"  测试集 (2020-2026): {test_return:.2%}, Sharpe {test_sharpe:.2f}")
print(f"  过拟合检查: Sharpe比 {sharpe_ratio:.2f} {'✅' if sharpe_ratio >= 0.8 else '⚠️'}")
print()
print("测试集包含：")
print("  - 2020年 COVID 暴跌与反弹")
print("  - 2021年 牛市") 
print("  - 2022年 熊市")
print("  - 2023-2026 波动期")
