# SOXL RL Phase 4 执行记录

## 时间线

### 15:01 - 项目启动
- ✅ 创建 worktree: `soxl-rl-qlib-data-rebuild`
- ✅ 分支: `feature/minute-data-rebuild`
- ✅ 提交现有更改

### 15:04 - 10K步训练完成
```
✅ Sharpe: 9.17 (异常高)
✅ Return: +54.49%
✅ Max DD: -1.68%
✅ Mean Position: 10.51% (做多偏向)
```

**问题识别**:
- Sharpe 9.17 异常高（目标: 1.5-3.0）
- 可能源因：
  1. 评估年化因子过大 (`sqrt(252*78)` ≈ 140x)
  2. 数据量太少（1个月）
  3. 过拟合

### 15:06 - 数据重建准备
- ✅ 创建 `download_minute_data_v2.py` (增强版下载脚本)
- ⏳ Yahoo Finance API 不稳定
- 📋 待API恢复后下载 60 天数据

### 15:10 - 100K步训练启动
- 🔄 训练中（预计 5-10 分钟）
- PID: 47374

---

## 发现的技术问题

### 1. Sharpe 计算问题
```python
# 当前代码 (train_sac.py:80)
sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 78)
```

**问题**:
- `sqrt(252 * 78)` = 140.2，放大倍数过大
- 如果 returns 是分钟级收益率，这个公式理论正确
- 但如果数据量少，std 很小，会导致 Sharpe 爆炸

**可能解决方案**:
1. 增加 `std` 的最小阈值（如 0.001）
2. 使用对数收益率
3. 限制 Sharpe 范围（如 [-5, 5]）

### 2. 数据量不足
- 现有: 1,560 行 × 6 ETF = 9,360 行（1个月）
- 目标: 60 天 × 78 行/天 = 4,680 行/ETF
- 需要: Yahoo Finance API 恢复

### 3. Yahoo Finance API 不稳定
- `yf.Ticker().history()` 频繁失败
- 可能需要：
  1. 增加重试逻辑 ✅ (已在 v2 脚本中)
  2. 使用其他数据源
  3. 使用本地历史数据

---

## 待办事项

### 高优先级
- [ ] 等待 100K 训练完成
- [ ] 检查 100K 训练结果
- [ ] 运行 OOS 测试
- [ ] 修复 Sharpe 计算问题

### 中优先级
- [ ] 数据重建（60天）
- [ ] Walk-forward 验证
- [ ] 特征工程优化

### 低优先级
- [ ] LSTM 网络实现
- [ ] Qlib 深度集成
- [ ] 实盘模拟测试

---

## 文件变更

### 新建文件
- `memory/2026-03-23-phase4-start.md` - Phase 4 启动文档
- `memory/2026-03-23-phase4-progress.md` - 本文件
- `soxl-rl-qlib-data-rebuild/scripts/download_minute_data_v2.py` - 数据下载脚本

### 修改文件
- 无

---

## Git 状态

**主项目** (`soxl-rl-qlib`):
```
Branch: main
Last commit: 22d905a (chore: 添加回测验证脚本)
Uncommitted: logs/train_100k*.log
```

**Worktree** (`soxl-rl-qlib-data-rebuild`):
```
Branch: feature/minute-data-rebuild
Last commit: 50d3169 (docs: 添加README和项目说明)
Uncommitted: scripts/download_minute_data_v2.py
```

---

## 下一步行动

1. **立即**: 等待 100K 训练完成
2. **接下来**: 分析结果，决定是否需要：
   - 修复 Sharpe 计算
   - 重新训练
   - 进行 OOS 测试
3. **稍后**: 数据重建 + 更长训练

---

**更新时间**: 2026-03-23 15:11
**状态**: 🔄 训练中
