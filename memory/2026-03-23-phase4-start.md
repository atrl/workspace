# SOXL RL Phase 4 启动 - 分钟级数据重建

## 🎯 任务目标

**最终目标**: 可上线的 Qlib 框架强化学习模型

**角色分工**:
- @clawQuant: 执行代码工作
- @simons: 提供思路，获取论文指导，**不修改代码**

**协作原则**:
- 不来回讨论
- 看执行结果
- 有进展合并 main

---

## 📋 执行计划

### Phase 4.1: Worktree 创建 ✅

**时间**: 2026-03-23 15:01

**已完成**:
```
✅ 提交现有更改 (commit: 22d905a)
✅ 创建 worktree: soxl-rl-qlib-data-rebuild
✅ 分支: feature/minute-data-rebuild
```

**目录结构**:
```
/Users/rziiiii/.openclaw/workspace/
├── soxl-rl-qlib/              # 主项目 (main)
└── soxl-rl-qlib-data-rebuild/ # 数据重建分支 (feature/minute-data-rebuild)
```

---

### Phase 4.2: 分钟级数据重建 (进行中)

**数据源**: Yahoo Finance
**时间粒度**: 5分钟
**目标ETF**: SOXL, SOXS, TQQQ, SQQQ, UPRO, SPXU
**数据范围**: 2025-01-01 ~ 2026-03-23 (约14个月)

**预期数据量**:
```
每个ETF: ~25,000行 (14个月 × 5分钟)
总数据: 6个ETF × 25,000 = 150,000行
```

---

### Phase 4.3: RL 模型训练

**算法**: SAC (Soft Actor-Critic)
**框架**: Stable-Baselines3 + Gymnasium
**特征**: 31个技术指标
**训练步数**: 100K - 500K

**目标指标**:
```
✅ Sharpe > 1.5
✅ Max DD < 15%
✅ Return > 20%
✅ OOS 表现稳定
```

---

## 📊 现有问题

### Phase 1 遗留问题
```
⚠️ Sharpe 20.26 (异常高，过拟合)
⚠️ In-Sample 评估 (无 OOS 测试)
⚠️ 做空偏向 (-13.89%)
```

### Phase 4 解决方案
```
✅ 重建数据 (Yahoo Finance 原始数据)
✅ Walk-forward 验证
✅ Out-of-Sample 测试
✅ 更长历史数据 (14个月)
```

---

## 🔧 技术栈

**数据**: Yahoo Finance (yfinance)
**特征**: pandas-ta / ta-lib
**RL**: Stable-Baselines3 (SAC/PPO)
**环境**: Gymnasium
**回测**: Qlib 风格

---

## 📁 目录结构

**Worktree**: `/Users/rziiiii/.openclaw/workspace/soxl-rl-qlib-data-rebuild`

```
soxl-rl-qlib-data-rebuild/
├── data/
│   ├── raw/           # 原始数据
│   ├── processed/     # 处理后数据
│   └── features/      # 特征数据
├── scripts/
│   ├── download_data.py
│   ├── build_features.py
│   └── train.py
├── src/
│   ├── rl/
│   │   ├── env.py
│   │   └── train_sac.py
│   └── data/
├── models/
└── logs/
```

---

## 🎯 Phase 4 目标

**时间**: 2026-03-23 ~ 2026-03-30 (1周)

**里程碑**:
```
Day 1-2: 数据重建 + 特征工程
Day 3-4: RL 训练 + 调优
Day 5-6: OOS 验证 + 回测
Day 7:   合并 main + 部署准备
```

---

## 📞 协作

**Discord**: #soxl
**响应**: < 5分钟
**支持**: 24/7

---

**启动时间**: 2026-03-23 15:01
**状态**: 🚀 进行中
