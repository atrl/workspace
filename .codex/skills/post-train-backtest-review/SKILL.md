---
name: post-train-backtest-review
description: 在每次训练或回测结束后，对最新模型与上一版本/基线模型做系统化复盘。用于比较 `artifacts/models/*_summary.json`、区分表现好的阶段和表现差的阶段、分析切片差异与可能原因、检查是否存在过拟合或风格漂移，并给出下一步实验方向。适用于 SOXL harness 中的训练后评估、checkpoint 选择复核、回测退化诊断、feature 归因复盘和 promotion gate 前的人类可读总结。
---

# Post-Train Backtest Review

在这个仓库里，训练完成不等于任务完成。必须把“这次为什么更好 / 更差、好坏发生在哪些年份、可能是哪类特征和行为造成的、下一步该做什么”写成可复用的结论。

## Core Rule

- 先比较，再解释；不要先想故事。
- 先看切片，再看总回报；总收益高不代表可推广。
- 把 `2022` 当作默认评估窗口，不要把它当作可随意优化的训练目标。
- 每次复盘都要同时区分：
  - 整体变好
  - 哪些年份或窗口变好
  - 哪些年份或窗口变差
  - 代价是什么
  - 代价是否值得

## Inputs

优先使用仓库已有产物，而不是口头回忆：

- 主输入：
  - `artifacts/models/*_summary.json`
- 可选补充：
  - `runtime/feature_analysis/summary.json`
  - `runtime/generalization/detailed/*.csv`
  - `memory/*.md`
  - `codex_exec_plans/active/*.md`

仓库里最重要的比较字段通常是：

- `valid`
- `selection_history`
- `slices.test_2020`
- `slices.test_2022`
- `slices.test_2023_2024`
- `test_proxy`
- `promotion`

## Default Workflow

### 1. Pick the comparison pair

始终明确这三个角色：

- `latest`: 最新训练结果
- `previous`: 上一个可接受 checkpoint，或当前生产/主分支基线
- `baseline family`: 规则策略、旧 RL 版本、或同 profile 上一轮实验

如果用户没有指定比较对象，默认优先级：

1. 同 profile 的上一版 summary
2. 当前主分支最佳 summary
3. 规则基线或重要 milestone

### 2. Generate a first-pass diff

优先运行 skill 自带脚本：

```bash
python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py \
  --latest artifacts/models/<latest>_summary.json \
  --previous artifacts/models/<previous>_summary.json \
  --output runtime/post_train_review/<label>.md
```

如果你还有特征归因产物，一并传入：

```bash
python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py \
  --latest artifacts/models/<latest>_summary.json \
  --previous artifacts/models/<previous>_summary.json \
  --latest-features runtime/feature_analysis/<latest>/summary.json \
  --previous-features runtime/feature_analysis/<previous>/summary.json \
  --output runtime/post_train_review/<label>.md
```

### 3. Classify good windows and bad windows

必须按切片写结论，不要只写一句“整体提升”。

最低要求：

- `test_2020`: 是否保住强趋势行情的捕获能力
- `test_2022`: 是否改善下跌/高压窗口的去风险能力
- `test_2023_2024`: 是否因为过度防守而损失后期趋势收益
- `test_proxy`: 总体风险收益是否仍然站得住

建议把窗口分成三类：

- 明确变好：回报/Sharpe 改善，且回撤没有明显恶化
- 代价型变好：回报改善，但回撤或暴露显著上升
- 明确变差：回报与 Sharpe 下滑，或回撤恶化且没有补偿

### 4. Explain with mechanism, not adjectives

优先从下面这些“可被代码改变”的机制解释，不要只写“市场不同”：

- 平均仓位是否上升/下降
- 2022 暴露是否过高，说明去风险不足
- 2023-2024 变差且仓位下降，说明策略可能过度保守
- `selection_history` 最优迭代是否过早出现，说明需要更强 early stopping
- validation 提升但 OOS 切片恶化，说明存在 checkpoint 选择偏差
- `promotion.fixed_split` 某一项刚好从通过变成不通过，说明改动触碰了硬门槛

### 5. Add feature-level evidence only when it changes the decision

如果需要讨论“特征差异”，优先比较：

- `top_shap`
- `top_ic`

只回答两个问题：

1. 哪些特征的重要性明显上升或下降
2. 这种变化是否与窗口表现变化一致

不要把 feature list 当结论本身。特征分析必须回到行为解释，例如：

- 风险代理特征上升，通常对应更强去风险
- 趋势/锚点特征上升，通常对应更高趋势持仓
- 短周期反转特征上升，若交易频率同步变高，通常意味着更容易微调过度

如需更深入的特征解释，先运行仓库已有脚本：

```bash
qlib_venv/bin/python scripts/feature_analysis.py
```

### 6. End with next actions, not just diagnosis

复盘必须落到“下一步怎么试”。默认只给 1-3 个高价值动作，并明确预期。

优先使用下面的映射：

- `2022` 改善但 `2023-2024` 变差：
  - 说明过度防守；优先放松 floor/cap、降低危机态惩罚、检查卖出阈值是否过高
- `2023-2024` 改善但 `2022` 恶化：
  - 说明风险约束太弱；优先检查 drawdown cap、压力状态路由、avg_pos
- validation 很强但 OOS 变差：
  - 优先复核 checkpoint 选择、减少训练轮数、增强 walk-forward 决策权重
- 总收益提升但回撤急剧放大：
  - 说明风格漂移到更高杠杆暴露；优先检查 floor、rebalance threshold、卖出惩罚是否过弱
- 切片好坏分化明显：
  - 先判断是 regime specialization 还是过拟合，再决定是否做 mixture / gate / state routing

## Review Output Contract

最终复盘至少要包含这些段落：

1. 总结论
2. 变好的窗口
3. 变差的窗口
4. 机制解释
5. 特征差异（如果有）
6. 下一步实验

如果没有足够证据解释，不要强行解释，直接写：

- 证据不足
- 还需要什么额外产物
- 为什么当前不能下结论

## Repo-Specific Guardrails

- 不要只用 `valid` 决定模型是否更优。
- 不要因为 `test_proxy` 上升就忽略 `test_2022` 风险恶化。
- 不要把 `avg_pos` 当成单向好坏指标；它是风格暴露指标，必须结合窗口解释。
- 不要把 `selection_history` 中的最佳 iter 当成稳定结论；要看是否早熟、是否 plateau、是否后续退化。
- 任何“改进”都应该回写到 workpad 或 memory，而不是只留在聊天里。

## Current Strategy Improvement Plan

这个 skill 不只是要“会复盘”，还要把当前仓库最值得做的改进方向说清楚。默认把当前策略分成两条线：

- `canonical balanced`
  - 当前默认最佳通用 qlib 策略
  - 目标：保住 `2022` gate，同时尽量抬高 `2023-2024` 和总收益
- `balanced_floor40_sym`
  - 当前高收益但高风险分支
  - 目标：只有在明确接受更激进风格时才继续

### P0: 主线优先级

当前默认优先顺序：

1. 保住 canonical `balanced` 的 `2022` 风控通过状态
2. 提升 `2023-2024` 收益和总 `test_proxy`
3. 再考虑追求极高总回报分支

如果一项改动让 `2022` gate 从通过变成失败，默认视为主线退化。

### P1: 针对 canonical `balanced` 的优先改进

这是当前最值得继续优化的路线。

优先方向：

- 提升 `2023-2024` 而不破坏 `2022`
  - 优先看牛市放宽逻辑是否过于保守
  - 但不要直接抬高全局 floor；优先做更精细的 regime-conditioned 放宽
- 强化 checkpoint 选择
  - 如果 validation 提升但 fixed-split 退化，优先改选择逻辑而不是继续训更久
  - 默认优先检查 `20/40/60/80` 的 `selection_history`
- 压缩无效交易
  - 当前 trade count 仍明显高于规则策略
  - 优先用更高 `rebalance_threshold`、最小 turnover gate、trade lock/cooldown，而不是再加更多 reward bonus

### P2: 针对 `balanced_floor40_sym` 的补救顺序

如果用户明确要求继续这条分支，默认补救顺序如下：

1. 先修 `2022` 去风险，而不是继续加训练轮数
2. 先压 `avg_pos`，再谈总收益
3. 先看危机态/高风险态仓位控制，再看普通趋势收益

优先检查：

- `drawdown_position_cap`
- stress / bear regime routing
- crisis floor 是否过高
- bull floor 是否在风险切换后仍滞留过久
- `avg_pos` 在 `test_2022` 是否高于 `0.35`

默认不要做的事：

- 不要先把 `max_iters` 再拉长
- 不要因为 validation 高就继续追这条线
- 不要把 `2020` / `2023-2024` 的上涨收益当成足够理由忽视 `2022`

### P3: 现阶段最可能有价值的实验

如果用户让你直接给下一轮实验，默认优先从下面选，不要平均分配精力：

1. `canonical balanced` 上做更细粒度的 stress de-risking
  - 目标：保持 `2022` gate 通过，同时减少对 `2023-2024` 的压制
2. `canonical balanced` 上做交易频率压缩
  - 目标：减少微调交易，提高 Sharpe，避免靠更高暴露换收益
3. checkpoint / early-stopping 改进
  - 目标：避免“validation 更高但 OOS 更差”的选择偏差

如果用户问“为什么不是先继续 floor40 分支”，默认回答：

- 因为它目前已经证明自己是高暴露换收益的分支
- 当前最缺的不是更多上涨暴露，而是更稳的主线提升
- 在现有 harness 下，主线最佳策略仍应以 `promotion.eligible=true` 为硬约束

## References

- 研究与行业实践摘要：读 `references/research-notes.md`
- 自动化对比脚本：运行 `scripts/compare_backtests.py`
