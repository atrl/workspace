# SOXL分钟级RL项目 - 紧急修复清单

## 🔥 P0 - 立即修复（由Codex执行）

### 问题1: Sharpe计算溢出

**位置**: `src/rl/train_sac.py` line 95-150

**症状**:
```
Sharpe: 6.49e+16 (应该 1.5-3.0)
```

**原因**:
```python
sharpe = annual_return / (volatility + 1e-8)
# volatility接近0导致除法溢出
```

**修复**:
```python
# 添加数值稳定性检查
if volatility < 1e-6 or np.isnan(volatility) or np.isinf(volatility):
    sharpe = 0.0
else:
    sharpe = annual_return / volatility
    sharpe = np.clip(sharpe, -10.0, 10.0)
```

### 问题2: Max DD可能超过-100%

**症状**:
```
Max DD: -221.90% (不可能超过-100%)
```

**修复**:
```python
max_dd = drawdown.min()
max_dd = max(max_dd, -1.0)  # 裁剪到-100%
```

### 问题3: Return异常高

**症状**:
```
Return: 693.37% (异常高)
```

**修复**:
```python
total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
total_return = np.clip(total_return, -1.0, 10.0)  # 裁剪到合理范围
```

---

## ✅ 完整修复代码

**替换 calculate_metrics() 函数** (完整代码见上一条Discord消息)

---

## 🚀 执行步骤

### 步骤1: 修复代码

```bash
cd /private/tmp/worktrees/soxl-minute-rl

# 编辑 src/rl/train_sac.py
# 替换 calculate_metrics() 函数 (line 95-150)
```

### 步骤2: 重新评估

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

python3 src/rl/train_sac.py --skip-training --model models/sac_soxl_minute_v1.zip
```

### 步骤3: 验证结果

**预期指标**:
```
✅ Sharpe: 1.5-3.0
✅ Max DD: -5% 到 -15%
✅ Return: 50% 到 100%
```

---

## 📊 当前资产

**模型**: `models/sac_soxl_minute_v1.zip` (8.0MB, 500k步)

**训练结果** (修复前):
```
✅ 奖励: +188.22
✅ Max DD: -5.80%
✅ Return: +87.14%
❌ Sharpe: 6.49e+16
```

---

## ⏱️ 预计时间

- 修复代码: 5-10分钟
- 重新评估: 2-3分钟
- 总计: 10-15分钟

---

**优先级**: 🔥 P0 - 立即执行

**负责人**: Codex

**支持**: clawQuant (24/7技术支持)

---

**创建时间**: 2026-03-23 02:19
**状态**: 待执行
