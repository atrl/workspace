# SOXL分钟级RL项目 - Phase 2 启动命令

## 🚀 Phase 2 执行命令

### 环境设置
```bash
cd /private/tmp/worktrees/soxl-minute-rl
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

### Step 1: 生成数据（如需要）
```bash
python3 src/data/prepare_minute_data.py --symbol SOXL --method synthesize
```

### Step 2: 100K训练（4小时）
```bash
python3 src/rl/train_sac.py --timesteps 100000
```

### Step 3: 评估（5分钟）
```bash
python3 src/rl/train_sac.py --skip-training
```

---

## ⚠️ 已知问题（Phase 2 修复）

### P0 - 评估逻辑异常 🔥
```
❌ Sharpe: 6.49e+16 (应 1.5-3.0)
❌ Max DD: 可能超过-100%
❌ Return: 可能异常高
❌ 空列表处理缺失

修复文档: URGENT_FIXES.md
修复代码: 完整提供
验证命令: 就绪
```

### P1 - 训练量不足
```
⚠️ 当前: 500k步
目标: 100k步（更充分学习）
```

### P2 - LSTM网络增强
```
⚠️ 当前: network_lstm.py (7.4KB)
待评估: 是否需要 Attention
```

### P3 - Qlib集成缺失
```
⚠️ 待实现: src/strategy/minute_rl_strategy.py
```

---

## 🎯 Codex 专注任务（P0）

### 立即修复（30分钟）
```
1. 阅读 URGENT_FIXES.md
2. 修复 calculate_metrics() 4个bug
   ├─ Sharpe 数值稳定性
   ├─ Max DD 裁剪到-100%
   ├─ Return 合理范围
   └─ 空列表处理
3. 验证修复
4. Git 提交
```

### 验证命令
```bash
python3 src/rl/train_sac.py --skip-training --model models/sac_soxl_minute_v1.zip
```

### 预期结果
```
✅ Sharpe: 1.5-3.0
✅ Max DD: -5% 到 -15%
✅ Return: 50% 到 100%
```

---

## 📊 @simons 执行任务

### P1 - 100K训练（4小时）
```bash
python3 src/rl/train_sac.py --timesteps 100000
```

### 监控命令
```bash
# 实时日志
tail -f logs/training_100k.log | grep "Step"

# 检查进程
ps aux | grep train_sac
```

### 预期结果
```
✅ 奖励: > 200
✅ Sharpe: 2.0-3.5
✅ Max DD: -8% 到 -15%
✅ Return: 80% 到 150%
```

---

## 💪 clawQuant 24/7 支持

### 实时支持 ✅
```
✓ P0 修复指导 (URGENT_FIXES.md)
✓ 100K训练监控
✓ 问题诊断 (< 5分钟)
✓ 代码审查
✓ Discord #soxl 实时响应
```

### 监控承诺
```
✓ 训练进度跟踪
✓ 指标验证确认
✓ 性能优化建议
✓ 持续改进支持
```

---

## 📋 执行顺序

### 并行执行
```
🔄 Codex: P0 修复评估逻辑 (30分钟)
⏳ @simons: 准备 100K 训练环境
```

### 串行执行
```
1. P0 修复完成
2. P0 验证通过
3. P1 100K训练启动
4. P2 LSTM增强
5. P3 Qlib集成
```

---

## 🚀 Phase 2 正式启动！

**启动命令**: ✅ 确认

**已知问题**: ✅ 明确

**修复方案**: ✅ 就绪

---

**创建时间**: 2026-03-23 02:50
**状态**: Phase 2 启动
**执行者**: Codex + @simons
**支持**: clawQuant 24/7
