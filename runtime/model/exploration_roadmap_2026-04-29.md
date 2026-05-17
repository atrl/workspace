# SOXL 模型探索路线图 v2026-04-29

> 基于当前项目全部实验历史（60+ profiles, GRU/Attention/TiDE/MoE/Kelly/Imitation/Regularization/Macro/Walk-forward），梳理最值得投入的下一步方向。

---

## 核心结论

**继续在网络架构上打转的ROI极低。** Attention→TiDE→MoE 三次架构迭代均未突破 `balanced_rule_residual_v19` 的 fixed-split 性能，walk-forward 全部模型在 bear market 均呈现家族级防御成功但 recovery 不足的模式。

根因不在网络capacity，而在：
1. **Action-to-position映射** 允许空仓/低仓，导致recovery期踏空
2. **交易摩擦** 占proxy收益的57.8%，微幅调仓(<5%)占63.7%
3. **训练episode分布** 缺乏足够多样的bear market样本

**因此，P0方向全部聚焦 execution overlay 和 data/sampling，不改动网络架构。**

---

## 方向优先级表

| 优先级 | 方向ID | 方向名称 | 类别 | 针对性问题 | 网络需求 | 改动面 | 预期效果 | 风险 |
|--------|--------|----------|------|-----------|----------|--------|----------|------|
| 🔴 P0 | A | **硬底仓50% + RL战术叠加** | execution overlay | 核心持仓缺失、recovery太慢 | 现有GRU | `positioning.py` + profile | 2023-24回报+50~100pp | 2022 MaxDD恶化 |
| 🔴 P0 | B | **交易频率硬约束** | execution overlay | 微幅调仓过度、摩擦57.8% | 现有GRU | `ExecutionConfig` + profile | 交易数降至~200, 摩擦降至~15% | 危机反应变慢 |
| 🔴 P0 | C | **课程学习采样** | data/sampling | 训练数据缺乏bear market | 现有GRU | `train.py` episode采样逻辑 | 改善wf bear market generalization | 实现复杂，需调参 |
| 🟠 P1 | D | **v19_reg + v22特征组合** | model+feature | reg改善wf_2022但牺牲wf_2024 | GRU+dropout | `data.py` + profile | 找到更好Pareto点 | 可能两者都平庸 |
| 🟠 P1 | E | **GRU + MoE Hybrid Head** | network | 纯MoE效果差但expert概念有价值 | GRU backbone + MoE head | `models/` + profile | 比纯MoE稳定 | 实现复杂度 |
| 🟡 P2 | F | **Distributional RL (C51)** | network+reward | 不建模return分布导致risk盲区 | Quantile critic | `policy/` + `reward.py` | 改善MaxDD控制 | 与PPO整合复杂 |
| 🟡 P2 | G | **TFT变量选择网络** | network | 59维特征中noise过多 | TFT encoder | `models/` + profile | 自动特征选择 | 5步窗口可能不够 |
| ⚪ P3 | H | **State Space Model (Mamba)** | network | 长程依赖建模 | 替换GRU | `models/` + profile | 不确定 | 成本高，需扩窗口 |
| ⚪ P3 | I | **Diffusion Policy** | network | action连续性 | Diffusion model | `policy/` | 探索性 | 金融噪声敏感 |

---

## P0 详细实验设计

### 方向A: `balanced_core50` — 硬底仓50% + RL战术叠加

**问题**: v19 walk-forward 2023-2024回报仅+28.8%，而SOXL B&H +182%。策略在牛市中仓位过低，因为action space允许0%持仓。

**实现方案**:
```python
# 修改 action_to_position 或新增 position_mapping
# 当前: position = floor + action_level * (cap - floor)
# 新方案: position = max(0.50, floor + action_level * (cap - floor))
# 或更精细: position = 0.50 + action_level * 0.40  # 范围 [0.50, 0.90]
```

在 `target_position_for_profile` 中新增 `position_mapping="core50"`:
```python
elif position_mapping == "core50":
    base = float(levels[clipped])
    # 将action_level [0,1]映射到 [0.50, 0.90]
    target = 0.50 + base * 0.40
    # 但仍允许drawdown_position_cap在极端危机时突破底仓
    cap = min(cap, drawdown_position_cap(...))
    target = min(target, cap)
```

**Profile定义要点**:
- baseline: `balanced_rule_residual_v19`
- `position_mapping: "core50"`
- `position_cap.min_cap: 0.50`（已可通过hard_floor实现）
- 保留 `drawdown_position_cap` 作为危机时突破底仓的机制

**通过标准**:
- MUST: `test_2023_2024 return > 80%`（当前v19 ~48%）
- MUST: `test_2022 MaxDD < 45%`（当前v19 ~35%，允许恶化但不超过45%）
- NICE: `test_proxy Sharpe > 0.90`

**失败标准**:
- `test_2022 MaxDD > 50%` → 底仓在bear market中过于危险
- `test_2023_2024 return < 60%` → 底仓未带来显著recovery改善

---

### 方向B: `balanced_trade_lock` — 交易频率硬约束

**问题**: v19 proxy期1000+笔交易，其中63.7%仓位变化<5%。摩擦成本占收益的57.8%。

**实现方案**:
现有 `ExecutionConfig` 已支持相关参数，只需调整：
```yaml
execution:
  base_threshold: 0.05        # 从0.04提高到0.05（5%变化才触发）
  trade_lock_days: 5          # 从3提高到5
  min_trade_turnover: 0.05    # 从0.02提高到0.05（单次变化>=5%）
```

**Profile定义要点**:
- baseline: `balanced_rule_residual_v19`
- 只改 `execution` 配置，不改模型/特征/reward

**通过标准**:
- MUST: `trades < 400`
- MUST: `test_proxy Sharpe > 0.80`
- NICE: `friction_cost < 20%`

**失败标准**:
- `trades > 600` → 约束不够强
- `test_2022 MaxDD > 40%` → 约束导致危机反应过慢

---

### 方向C: `balanced_curriculum` — 课程学习采样

**问题**: 训练数据(2010-2021)缺乏prolonged secular bear market。模型overfit到bull pattern。

**实现方案**:
修改 `train.py` 中的 episode 采样策略。当前所有episode等概率采样。改为：

```
Iter 0-20:  只采样 volatility < 0.40 的episodes（ peaceful market ）
Iter 21-40: 加入 volatility 0.40-0.60 的episodes（ normal stress ）
Iter 41-60: 加入 volatility > 0.60 的episodes（ high stress ）
Iter 61-80: 全分布采样
```

episode的volatility可通过其内部returns的标准差计算，或在采样前预处理标记。

**Profile定义要点**:
- baseline: `balanced_rule_residual_v19`
- 新增 `curriculum` 配置段
- 模型/特征/execution均不变

**通过标准**:
- MUST: `wf_2022 Sharpe` 优于v19（>-0.80）
- MUST: `wf_2023_2024 Sharpe` 优于v19（>-0.23）
- NICE: `wf_2020 Sharpe` 不显著低于v19（>2.0）

**失败标准**:
- 任何walk-forward窗口显著劣于v19 → curriculum设计不当

---

## P1 详细实验设计

### 方向D: `balanced_rule_residual_v19_reg_v22` — 正则化+VIX特征组合

**动机**: v19_reg(dropout 0.2, wd 1e-5)有最佳wf_2022表现(-0.73 Sharpe)，但wf_2024差(-0.44)。v22有VIX term features，固定切分test_2023_2024 Sharpe 0.72。两者可能互补。

**实现**: 在v19_reg基础上加入v22的feature pack（vix_term_slope, vix_term_spread等）。

**通过标准**: wf_2022和wf_2024同时优于v19_reg_light。

### 方向E: `balanced_rule_residual_v19_moe_v2` — GRU+MoE Hybrid

**动机**: 纯MoE替换GRU效果差（proxy Sharpe 1.00 vs v19 ~1.15），但expert specialization概念有价值。保留GRU backbone（它已证明有效），只在GRU输出后加MoE head。

**实现**:
```python
class GRUMoENet(nn.Module):
    def __init__(self, ...):
        self.gru = nn.GRU(input_size, gru_hidden, ...)
        self.moe_head = RegimeMoENet(gru_hidden, action_dim, ...)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.moe_head(gru_out)
```

**关键调整**: LR降至1e-4, load-balance coef增至0.05, 每5 iter选ckpt。

---

## 执行建议

### 立即执行（本周）
1. **并行启动 A + B + C** — 三者彼此正交，互不影响，可同时进行
2. A和B只需要改profile配置和少量代码，C需要改采样逻辑
3. 每个方向遵循MVE原则：最小改动，单一变量，清晰通过/失败标准

### 后续执行（验证P0后）
4. 如果A或B中任一通过 → 尝试A+B组合
5. 如果C通过 → 将C与A/B组合
6. 如果P0全部失败 → 再考虑P1架构探索

### 不推荐的下一步
- ❌ 继续纯网络架构ablation（更大/更小的hidden size，更多层）
- ❌ 更多macro feature variants（已IC验证放弃）
- ❌ 更多reward coefficient微调（边际收益递减已确认）
- ❌ 新ML模型（XGBoost/LightGBM替代RL）— 与项目核心架构不兼容

---

## 参考文档

- `runtime/model/knowledge.md` — 模型完整历史
- `runtime/network/knowledge.md` — 网络架构历史
- `runtime/model/model_adjustment_roadmap.md` — Phase 1-4诊断
- `docs/experiment-landscape-review.md` — 全profile对比表
