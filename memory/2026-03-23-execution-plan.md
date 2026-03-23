# SOXL RL Phase 4 - 执行计划

## 📋 当前状态 (2026-03-23 15:15)

### 训练进度
```
🔄 100K步训练: 24.3K/100K (24%)
   - FPS: 306
   - 预计完成: 15:20-15:25 (5-10分钟)

✅ 10K步基线:
   - Sharpe: 9.17 ⚠️ (异常高)
   - Return: +54.49%
   - Max DD: -1.68%
   - Mean Position: 10.51% (做多偏向)
```

### 数据状态
```
✅ SOXL 5分钟特征: 1,540行
📅 时间范围: 2026-02-23 09:30 ~ 2026-03-20 15:55
⚠️ 数据量: 约1个月（不足）
⚠️ Yahoo Finance API: 不稳定
```

### Git 状态
```
主项目 (main):
  - 未提交: logs/, models/sac_soxl_10k.zip

Worktree (feature/minute-data-rebuild):
  - 新文件: scripts/download_minute_data_v2.py
  - 未提交
```

---

## 🎯 Phase 4 执行计划

### Phase 4.1: 基线验证 (进行中)

**目标**: 验证现有框架 + 识别问题

**步骤**:
1. ✅ Worktree 创建
2. ✅ 10K步基线训练
3. 🔄 100K步训练 (24%完成)
4. ⏳ 结果评估
5. ⏳ OOS 测试
6. ⏳ 问题诊断

**预期问题**:
- Sharpe 异常高 (9.17) → 过拟合
- 数据量少 (1个月) → 样本不足
- 无 OOS 测试 → 无法验证泛化

---

### Phase 4.2: 数据重建 (待定)

**目标**: 获取高质量分钟级数据

**方案 A: Yahoo Finance (优先)**
```bash
# 等待API恢复后执行
cd /Users/rziiiii/.openclaw/workspace/soxl-rl-qlib-data-rebuild
python3 scripts/download_minute_data_v2.py
```

**预期数据**:
- 6个ETF × 60天 × 78行/天 = 28,080行
- 时间范围: 2026-01-20 ~ 2026-03-23

**方案 B: 本地数据源**
- Alpha Vantage API
- IEX Cloud
- 已有历史数据

---

### Phase 4.3: 模型优化 (待定)

**目标**: 解决过拟合，提升泛化能力

**优化方向**:

1. **评估指标修复**
```python
# 当前问题: Sharpe 年化因子过大
sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 78)

# 修复方案:
std = max(np.std(returns), 0.001)  # 最小阈值
sharpe = np.clip(np.mean(returns) / std * np.sqrt(252 * 78), -5, 5)
```

2. **Walk-forward 验证**
```python
# 训练集: 前80%
# 验证集: 后20%
# 滚动窗口: 每周重新训练
```

3. **正则化**
- 增加网络 dropout
- 减少网络层数
- 增加训练噪声

4. **特征选择**
- 减少特征数量 (31 → 15-20)
- 去除高相关特征
- 增加市场因子

---

### Phase 4.4: Qlib 集成 (待定)

**目标**: 将RL模型集成到Qlib框架

**步骤**:
1. 创建 Qlib 策略包装器
2. 配置 Qlib 回测引擎
3. 对比 V25 基线
4. 部署准备

---

## 🚨 关键风险

### P0 - 立即处理
- ⚠️ Sharpe 9.17 异常高 → 必须修复评估逻辑
- ⚠️ Yahoo Finance API 不稳定 → 需要备用数据源

### P1 - 中期处理
- ⚠️ 数据量少 → 数据重建
- ⚠️ 无 OOS 测试 → Walk-forward 验证

### P2 - 长期优化
- ⚠️ 做空偏向 → 奖励函数调整
- ⚠️ 特征过多 → 特征选择

---

## 📊 成功标准

**最终目标**:
```
✅ Sharpe: 1.5 - 3.0 (非异常高)
✅ Max DD: < 15%
✅ Return: > 20%
✅ OOS 表现: 与 IS 接近 (偏差 < 30%)
✅ Walk-forward: 稳定盈利
```

**对比 V25 基线**:
```
V25: Sharpe 0.96, Max DD -8.69%
RL:  目标 Sharpe > 1.5, Max DD < 15%
```

---

## 📞 协作分工

**@clawQuant** (执行者):
- 代码实现
- 训练执行
- 结果分析
- 技术问题解决

**@simons** (指导者):
- 策略思路
- 论文指导
- 结果评估
- 方向决策

**协作原则**:
- 不来回讨论
- 看执行结果
- 有进展合并 main

---

## 📅 时间线

**Day 1 (今天)**:
- [x] Worktree 创建
- [x] 10K步基线
- [ ] 100K步训练完成
- [ ] 问题诊断

**Day 2-3**:
- [ ] 数据重建
- [ ] 评估修复
- [ ] Walk-forward 实现

**Day 4-5**:
- [ ] 模型优化
- [ ] OOS 测试
- [ ] 结果验证

**Day 6-7**:
- [ ] Qlib 集成
- [ ] 对比 V25
- [ ] 部署准备

---

## 🔧 技术栈

**数据**:
- Yahoo Finance (yfinance)
- Pandas
- 5分钟K线

**特征**:
- 31个技术指标
- 收益率、波动率、MA、RSI、布林带、ATR

**RL**:
- Stable-Baselines3 (SAC)
- Gymnasium
- Sharpe-like 奖励

**环境**:
- Python 3.14
- CPU 训练

---

**更新时间**: 2026-03-23 15:15
**状态**: 🔄 等待 100K 训练完成
**下一步**: 结果评估 + 问题诊断
