# Qlib RL Trading Skill

通用Qlib强化学习交易策略开发框架，支持任何单标的的动态仓位管理。

## 适用场景

- ✅ **股票/ETF择时** - SPY, QQQ, SOXL等
- ✅ **个股交易** - AAPL, TSLA等
- ✅ **期货交易** - 商品、股指期货
- ✅ **加密货币** - BTC, ETH等
- ✅ **外汇交易** - 主要货币对

## 安装依赖

```bash
pip install pyqlib torch tensorboard
```

## 快速开始

### 1. 准备数据

```bash
# 准备任意标的数据
python scripts/prepare_rl_data.py \
  --ticker <YOUR_TICKER> \
  --start 2020-01-01 \
  --end 2026-04-01
```

### 2. 训练模型

```bash
# 训练
python scripts/train_ppo.py \
  --data data/rl/<ticker>.pkl \
  --output-dir models/<ticker>
```

### 3. 回测

```bash
# 回测
python scripts/backtest_rl.py \
  --model models/<ticker>/ppo_model.zip \
  --data data/rl/<ticker>.pkl
```

## 针对不同标的的建议

### 杠杆ETF
- 使用 `ppo_safe.yaml`
- 添加外部数据（VIX）
- 严格止损（10%）

### 蓝筹股
- 使用 `ppo_default.yaml`
- 可满仓（95%）
- 适度止损（15%）

### 加密货币
- 更高交易成本（0.5%）
- 24/7市场（调整max_steps）
- 添加链上数据

## 配置说明

见 `configs/` 目录：
- `ppo_default.yaml` - 默认配置
- `ppo_safe.yaml` - 高波动标的
- `extreme_events.yaml` - 极端事件测试

---

**版本**: v1.0
**更新日期**: 2026-04-04
**维护者**: clawQuant
