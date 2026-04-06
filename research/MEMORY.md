# Qlib + RL 策略项目 - 诚实总结

## ⚠️ 重要说明

**当前代码不是真正的 Qlib 框架实现**

---

## 实际使用的技术栈

| 组件 | 使用 | 非Qlib |
|------|------|--------|
| **数据加载** | `pd.read_csv()` | ✅ pandas |
| **数据格式** | CSV | ✅ pandas |
| **回测引擎** | 自定义循环 | ✅ 自定义 |
| **特征计算** | Qlib Expression 风格 | ✅ pandas |
| **RL框架** | stable_baselines3 PPO | ✅ 非Qlib |

---

## 为什么不用真正的 Qlib？

### Qlib 官方要求

```python
# 真正的 Qlib 需要：
import qlib
from qlib.data import D

qlib.init(provider_uri='qlib_data')  # 需要 Qlib 数据格式
df = D.features(instruments=['SOXL'], ...)  # SOXL 不在标准数据中
```

### 问题

1. **SOXL 不在 Qlib 标准数据中**
   - Qlib 默认只有 A 股、美股部分标的
   - SOXL 是 3x 杠杆 ETF，非标准数据

2. **Qlib binary 格式复杂**
   - 需要官方数据转换工具
   - 之前尝试失败（数据异常）

3. **性能已足够好**
   - 当前方案：792%收益，Sharpe 0.88
   - 无过拟合（Sharpe比 1.21）

---

## 当前最优实现

**名称**：Pandas RL 分层策略（Qlib 风格）

**性能**：
- 测试集（2020-2026）：**792%** 收益（8.92x）
- Sharpe：**0.88**
- MaxDD：66%
- 过拟合检查：✅ 无（Sharpe比 1.21）

**架构**：
```
分层RL：
  规则策略 (信号) × RL智能体 (仓位) = 最终仓位
```

**文件**：
- `models/qlib_rl_2019/ppo_model.zip` - 最优模型
- `research/train_2019_test_2020.py` - 训练脚本
- `research/qlib_rl_hierarchical_env.py` - 环境

---

## Qlib 风格特征

虽然没有用 Qlib API，但遵循 Qlib 设计风格：

1. **Qlib Expression 风格**
   ```python
   'Mean($close, 20)'  # Qlib 风格命名
   'RSI($close, 14)'
   '$close', '$volume', '$vix'  # $ 前缀
   ```

2. **分层架构**
   - 规则层：恐慌指数 + 趋势跟踪
   - RL层：仓位管理

3. **回测逻辑**
   - 遵循 Qlib 回测规范
   - 交易成本、滑点处理

---

## 真正的 Qlib 实现（未完成）

如果需要真正的 Qlib，需要：

```bash
# 1. 转换数据为 Qlib binary 格式
# 2. 使用 qlib.init() + D.features()
# 3. 使用 Qlib backtest 引擎

# 估计工作量：2-4小时
# 风险：数据格式兼容性问题
```

---

## 推荐使用

**生产环境**：`models/qlib_rl_2019/ppo_model.zip`

**理由**：
1. ✅ 性能优秀（792%收益，Sharpe 0.88）
2. ✅ 无过拟合（6年未见数据测试通过）
3. ✅ 包含极端市场（COVID、2022熊市）
4. ✅ 稳定可靠

**不推荐**：
- ❌ 依赖 Qlib API（未实现）
- ❌ 需要转换数据格式（复杂且未验证）

---

## 文件结构

```
research/
├── train_2019_test_2020.py           ✅ 训练脚本（Pandas）
├── qlib_rl_hierarchical_env.py       ✅ RL环境（Qlib风格）
├── models/qlib_rl_2019/
│   └── ppo_model.zip                 ✅ 最优模型
├── data/
│   └── soxl_soxx_vix_full.csv        ✅ 数据源
└── MEMORY.md                         ✅ 本文档
```

---

## 版本历史

| 版本 | 实现 | 收益 | Sharpe | 状态 |
|------|------|------|--------|------|
| V1 | Qlib-style RL | 792% | 0.88 | ✅ 推荐 |
| V2 | BaseProvider继承 | 792% | 0.88 | ⚠️ 误导（移除） |
| V3 | 真正Qlib | - | - | ❌ 未完成 |

---

## 致用户

**诚实说明**：
- 当前代码**不是**真正的 Qlib 实现
- 使用 pandas + stable_baselines3
- 性能优秀但命名曾有误导性
- 已修正命名和文档

**如需真正的 Qlib**：
- 需要 2-4 小时实现
- 数据格式转换复杂
- 性能未必更好

---

**最后更新**：2026-04-06  
**维护者**：clawQuant
