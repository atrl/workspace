---
name: post-train-backtest-review
description: 在每次训练或回测结束后，对最新模型与上一版本/基线模型做系统化复盘。用于比较 `runtime/model/card/*/*_summary.json`、区分表现好的阶段和表现差的阶段、分析切片差异与可能原因、检查是否存在过拟合或风格漂移，并给出下一步实验方向。适用于 SOXL harness 中的训练后评估、checkpoint 选择复核、回测退化诊断、feature 归因复盘和 promotion gate 前的人类可读总结。
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
  - `runtime/model/card/*/*_summary.json` (所有实验卡片目录下的 summary JSON)
- 可选补充：
  - `runtime/feature_analysis/summary.json`
  - `runtime/generalization/detailed/*.csv`
  - `runtime/<direction>/knowledge.md`
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
  --latest runtime/model/card/<name>/<latest>_summary.json \
  --previous runtime/model/card/<name>/<previous>_summary.json \
  --output runtime/model/card/<label>.md
```

如果你还有特征归因产物，一并传入：

```bash
python .codex/skills/post-train-backtest-review/scripts/compare_backtests.py \
  --latest runtime/model/card/<name>/<latest>_summary.json \
  --previous runtime/model/card/<name>/<previous>_summary.json \
  --latest-features runtime/feature_analysis/<latest>/summary.json \
  --previous-features runtime/feature_analysis/<previous>/summary.json \
  --output runtime/model/card/<label>.md
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

当 current/paper model 总体 OOS 仍然为正、但近期超额或回撤明显失败时，先运行 poor-window 诊断，而不是直接重训：

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/diagnose_oos_poor_windows.py \
  --daily-csv runtime/production/card/<source>/<daily>.csv \
  --profile <profile> \
  --output-dir runtime/<direction>/card/<label> \
  --oos-start 2020-01-02 \
  --window-size 63 \
  --target-excess 30 \
  --target-max-dd 15
```

输出的 `poor_windows.csv` 用来定位坏窗口，`feature_differences.csv` 只作为机制假设来源。
不要把高 `delta_z` 特征直接当成可上线因子；先走 factor IC、单信号回测和 walk-forward gate。

当连续两个以上 execution/model-selection 诊断都没过 promotion gate 时，不要继续做单卡阈值微调。
先做一个 runtime notebook 聚合多个卡片的 `poor_windows.csv` 与
`feature_differences.csv`：按时间簇、目标失败类型、平均仓位、MaxDD、excess gap 和
top feature group 判断是 underparticipation、drawdown control、还是 selection bias。
只有聚合结论指向明确机制后，才开下一条参数实验或训练分支。

可复用聚合入口：

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/aggregate_oos_poor_windows.py \
  --input-dir runtime/<direction>/card/<label_a> --label <label_a> \
  --input-dir runtime/<direction>/card/<label_b> --label <label_b> \
  --selection-json runtime/<direction>/card/<label_a>/selection.json --selection-label <label_a> \
  --selection-json runtime/<direction>/card/<label_b>/selection.json --selection-label <label_b> \
  --output-dir runtime/model_selection/card/<aggregate_label>/aggregate \
  --target-excess 30 \
  --target-max-dd 15
```

输出的 `poor_window_clusters.csv` 负责决定下一步机制；`feature_delta_clusters.csv`
只负责生成待 IC / 单信号 / walk-forward 验证的特征假设。
如果传入 `--selection-json`，脚本还会写出 `gate_gap_attribution.csv` 与
`gate_gap_profile_summary.csv`。这些表用于判断每个 profile/window 是
`joint_tradeoff`、`drawdown_control`、`underparticipation`、`trend_capture`、
`near_miss_drawdown` 还是 `selection_bias`，并给出下一步机制建议。

当最新分支只有某个 walk-forward 窗口 near miss，但 `test_proxy`、`wf_2020`
或 `wf_2022` 仍是 `joint_tradeoff` / `drawdown_control` 时，不要把 near miss
当 promotion 证据；下一步应优先做 walk-forward-weighted multi-objective
selection/training objective，或 state-conditioned participation + drawdown guard。
不要继续加重同类全局 reward penalty。

如果已有 `walkforward_daily_screen.csv` / `strict_daily_target_screen.csv`
等候选表，优先在 runtime notebook 中调用：

```python
from src.soxl_quant.app.walkforward_selection import (
    score_walkforward_candidates,
    summarize_walkforward_selection,
)
```

`score_walkforward_candidates()` 用 `+30pp` 和 `<=15% MaxDD` 对每个窗口做
weighted gap ranking。若没有任何候选 `full_gate=True`，不要把分数最低者当
promotion；只能把它作为 closest diagnostic reference，并进入
walk-forward-weighted multi-objective training/selection 设计。

当 walk-forward weighted selector、poor-window aggregate、以及路径级
rebound/downtrend router 都没有过 full gate 时，不要继续扩大路径阈值网格。
在 runtime notebook 中把三类证据交给 handoff 逻辑：

```python
from src.soxl_quant.app.training_objective_handoff import (
    derive_training_objective_handoff,
    render_training_objective_handoff,
)
```

`derive_training_objective_handoff()` 会把最近候选的 MaxDD/relative gap、
共享 poor-window cluster、feature group consensus、以及 router 是否已经
`diagnostic_no_promotion` 合并成下一轮训练目标。如果结果要求
`train_state_conditioned_multi_objective`，下一步应进入训练时/模型选择时的
状态条件多目标，而不是再做 path-level threshold router search。

进入状态条件多目标前，先用已有 daily path 做一个 notebook-first objective
screen，确认最接近的老师路径和仍然失败的项：

```python
from src.soxl_quant.app.state_conditioned_objective import (
    daily_path_metrics,
    slice_daily_path_metrics,
    score_state_conditioned_objective,
    summarize_state_conditioned_objective,
)
```

这个 screen 必须同时输出 window metrics、cluster metrics、objective scores
和 `selection.json`。如果没有 `objective_pass=True`，结论仍然是训练目标 /
checkpoint-selection target，不允许 runtime promotion。

当 objective screen 已经落成 per-date target table，并且 target-alignment
screen 仍没有候选通过 window gate 时，不要停在“最接近老师路径”的结论。
继续在 runtime notebook 中把 alignment scores 转成 retained-checkpoint /
candidate selection objective：

```python
from src.soxl_quant.app.state_objective_training_targets import (
    score_state_objective_checkpoint_candidates,
    summarize_state_objective_checkpoint_selection,
)
```

`score_state_objective_checkpoint_candidates()` 默认按
`checkpoint_objective_loss` 排序，弱化 legacy validation score，显式惩罚
target alignment loss、walk-forward gate loss、rebound MAE、risk-cap
violation 和 target coverage 缺失。若 `window_gate` 或 `coverage_gate` 未
同时通过，结论只能是下一轮 checkpoint scoring / BC warm-start / PPO reward
shaping 的训练配方，不允许 runtime promotion。

当 state-objective BC warm-start 的全局 label accuracy / fold accuracy
通过，但 rebound 或 downtrend 等稀疏状态切片 accuracy 明显失败时，不要把它
标成 ready-for-PPO。结论应是“global fit 可用、稀疏状态未学会”，下一步先做
rebound-weighted BC、状态加权采样或 fold-local loss weighting，再进入
walk-forward PPO。`selection.json` 至少要记录 `state_slice_fit_gate`、
`rebound_accuracy`、`next_training_requires_rebound_weighting`，并继续保持
`promotion_allowed=false`。

如果 rebound-weighted BC 已经做过确定性权重筛选，且 rebound rows 少于
20、`rebound_accuracy` 没有提升或仍低于 0.60，不要继续扩大同类权重网格。
下一步应改成 target enrichment、rebound 状态标签扩展、或 state-conditioned
loss/sampling objective；继续提高同一个 slice weight 只会把 downtrend/global
fit 当作“进步”而掩盖 rebound 仍未学会。

如果 target enrichment 已经把 rebound rows 扩到接近 20，且原始 rebound
slice accuracy 仍没有提升，不要继续调邻近半径或 target floor。此时问题不是
稀疏标签数量或目标强度，而更可能是当前 observation/action 表面对 rebound
状态不可分；下一步应进入 state-conditioned loss、显式 rebound state head、
或模型结构/特征表达，而不是更多邻近日期扩展。

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
- 在使用 retained checkpoint actor 或重新导出的 daily replay 做推广判断前，先做 replay parity check：比较 summary JSON、已保存 daily CSV、当前 actor replay 的 return、MaxDD、avg_pos 和日期范围。若不一致，只能作为诊断，不能作为 promotion 证据。
- 在 notebook 里评分 counterfactual replay 时，传给 metrics 的 frame 必须只包含
  `datetime` 和单一目标 `position` 列。不要在包含原始 `position` 的宽表上
  直接把 `oracle_position` / `base_position` rename 成 `position`；重复列会让
  position 变成二维数组，导致指标报错或更隐蔽的误算。
- 如果连续的 BC / label diagnostics（例如 logistic、tree、triple-barrier
  action labels）能提高平均超额，但仍不能把 key-window MaxDD 压到目标以内，
  不要继续换分类器或做阈值扫参。先看 `selected_signal_audit.csv` 的
  false positive / sparse-action 结构；若收益改善伴随 2022 或关键窗口回撤扩大，
  下一步应转向 explicit drawdown-state / sequence features 或 execution-layer
  drawdown budget，而不是再开一个同类 BC sweep。
- 如果 fixed-signal + global realized-equity drawdown budget 只能小幅降低
  MaxDD，却把平均超额打回门槛以下，说明全局 cap 过钝。下一步应测试
  regime/state-conditioned budget activation（例如仅在 broken-anchor/high-vol
  状态触发）或更细的 execution budget，而不是继续降低全局 start/full
  drawdown 阈值。
- 任何“改进”都应该回写到 workpad 或 memory，而不是只留在聊天里。

## Generic Quant Optimization Framework

量化策略优化应该先通用，再实例化。默认把优化问题拆成 5 类，而不是直接跳到某个参数：

1. **信号质量**
  - 预测/动作是否真正区分了好环境和坏环境
  - 现象：高换手但没有稳定 alpha；valid 好但不同窗口表现分裂
2. **风险预算映射**
  - 风险状态是否被正确翻译成仓位、杠杆、止损、cap
  - 现象：熊市持仓过高；上涨期过度防守；收益主要来自暴露漂移
3. **交易频率与摩擦**
  - 策略是否在支付太多无效交易成本
  - 现象：trade count 高、Sharpe 低、回测对交易成本敏感
4. **模型选择偏差**
  - 是否选中了 validation 上好看、但 OOS 不稳的 checkpoint
  - 现象：selection history 上升，但 fixed-split / walk-forward 退化
5. **执行层覆盖**
  - 当前问题是 actor 本身，还是 actor 没问题但 execution/risk overlay 不对
  - 现象：换一个 cap / floor / execution rule，结果明显变化

### Default Optimization Order

如果没有额外上下文，默认按这个顺序排查：

1. 先排除模型选择偏差
2. 再排除执行/风控 overlay 是否已经足够解释问题
3. 再看交易频率和成本
4. 最后才考虑新增特征或重训

原因：

- 很多“模型问题”其实是 checkpoint 选错
- 很多“需要重训”的问题，其实先改 execution/risk 映射就能验证方向
- 新特征和更长训练成本最高，也最容易引入额外过拟合

### Counterfactual Rule

如果一个建议可以通过 **固定 actor + 改 execution/risk 规则** 来测试，先做 counterfactual backtest，不要立刻重训。

典型可先回测验证的建议：

- 调整 `position_cap`
- 调整 bull / bear regime cap
- 调整最小仓位 / 最大仓位
- 调整 trade gate / rebalance threshold
- 调整 cooldown / trade lock

只有当 counterfactual backtest 显示“方向正确但幅度不够”时，再进入重训。

## Historical-Best Feature Recall

如果用户要求“分析历史最优策略哪里好、哪里差、该加什么特征”，不要停在收益对比。必须把 **历史最优基线** 本身拆成“擅长状态”和“失效状态”，然后把这两类状态映射回已有特征和缺失特征。

默认流程：

1. 选定当前历史最优基线
  - 通常是当前 canonical summary，例如 `runtime/model/card/balanced/soxl_balanced_summary.json`
2. 先从 summary / detailed backtest 中定位：
  - 最好的防守窗口
  - 最好的趋势捕获窗口
  - 最差的反弹错失窗口
  - 最差的震荡 / 再启动窗口
3. 读取已有 feature attribution：
  - `runtime/feature_analysis/summary.json`
  - `runtime/feature_analysis/shap_importance.csv`
  - `runtime/feature_analysis/ic.csv`
4. 回答这 3 个问题：
  - 历史最优策略已经会看什么
  - 历史最优策略不会看什么
  - 下一轮应该补的是哪类状态特征，而不是哪一个孤立因子

### Required Output Shape

至少明确写出：

- 好的地方：
  - 哪些窗口明显跑赢
  - 当时主要是什么特征状态
- 差的地方：
  - 哪些窗口明显落后
  - 当时已有特征为什么不够
- 特征召回：
  - 当前最重要的 5-10 个特征到底在描述什么市场状态
- 缺口判断：
  - 缺的是趋势、风险、交易摩擦，还是状态转换特征

## Trade-Quality And Negative-Kelly Audit

如果用户问的是“模型为什么会做负期望的 T”“Kelly / Expectancy 为负说明学到了什么”，不要只重复总收益或 Sharpe。必须把 **训练 reward** 和 **交易级后果** 对齐起来看。

### Core interpretation rule

- 先区分：
  - actor 是否学到了正的总体路径控制
  - actor 是否学到了正的交易级 edge
- `Kelly` / `Expectancy` 为负，不等于“模型没学到东西”。
- 更常见的含义是：
  - 模型学到了组合路径层面的 reward
  - 但没有学到高质量、可支付摩擦成本的短线 / 做 T edge
- 在这个仓库里，默认怀疑的不是“模型失灵”，而是：
  - reward 对齐错位
  - 执行阈值太低
  - 微幅调仓过多
  - 某些持仓周期段实际上是负期望

### Repo workflow

默认按这个顺序做，不要跳步：

1. 先确认 summary 层是否已经说明 actor 值得保留
  - 看 `promotion`
  - 看 `slices.test_2022`
  - 看 `test_proxy`
2. 再看交易级归因，而不是直接对 Kelly 下结论
  - 优先使用：
    - `runtime/diagnosis/phase1_phase2_combined.json`
    - `runtime/diagnosis/soxl_trade_attribution.md`
  - 如果缺少产物，运行：

```bash
qlib_venv/bin/python scripts/diagnosis_phase1_phase2.py
```

  - 如果需要把结论固化成训练后审计报告，运行：

```bash
qlib_venv/bin/python scripts/trade_quality_audit.py \
  --summary runtime/model/card/<profile>/<profile>_summary.json \
  --diagnosis runtime/diagnosis/phase1_phase2_combined.json \
  --profile <profile> \
  --output runtime/model/card/<profile>_trade_quality_audit.md
```

3. 按两种分层检查负 edge 的位置
  - `by_holding_period`
  - `by_pos_change`
4. 把负 edge bucket 映射回 reward / execution 机制
  - 看 `src/soxl_quant/policy/reward.py`
  - 看 profile 中的：
    - `reward`
    - `execution`
    - `position_cap`
5. 先做固定 actor 的 counterfactual，再决定要不要重训

### Minimum questions to answer

至少回答下面这些问题：

1. 负 `Kelly` / `Expectancy` 是出现在：
  - 全部交易
  - 仅微幅调仓
  - 仅 `3-10d`
  - 仅特定压力窗口
2. 这些负 edge 交易是否数量很多
  - 如果数量小，可能只是噪声
  - 如果数量大，说明策略行为被系统性鼓励
3. 这些交易对总 PnL 的贡献是否低于摩擦成本
4. 它们更像是：
  - 信号弱
  - 风险控制动作太碎
  - reward 在鼓励“勤奋交易”
  - execution 阈值太低

### Mapping negative Kelly back to mechanism

优先用下面这套映射，不要泛泛而谈：

- `micro_<5%` 很弱：
  - 优先怀疑再平衡阈值过低、trade lock 不够、actor 对微小状态变化过敏
- `mid_3_10d` 为负：
  - 优先怀疑做 T 逻辑在震荡里被反复打脸，reward 允许太多中短期反转
- 整体 `Kelly` 为正但某些 bucket 为负：
  - 说明策略主要靠少数长持仓赚钱，噪声交易在稀释 edge
- `Sharpe` 还可以但交易级 `Expectancy` 很差：
  - 说明收益主要来自仓位路径控制，不是来自高质量逐笔交易
- `test_2022 avg_pos` 改善但交易级 `Kelly` 更差：
  - 说明风险动作变多了，但动作质量不高

### Reward-alignment diagnosis

如果用户问“负期望时模型学到了什么奖励”，必须明确说出：

- 当前 reward 优化的是：
  - 单步组合收益
  - drawdown / downside / risk budget
  - inactivity / opportunity cost
  - trend alignment / trapped loser / stale position
- 这不等于直接优化：
  - 每笔交易的 `Expectancy`
  - 每类做 T 的净 alpha
  - 成本后的 round-trip edge

因此默认结论模板应该是：

- 模型学到了 `aggregate path reward`
- 但没有充分学到 `per-trade expectancy`
- 当前负 edge bucket 更像是仓位轨迹控制的副产物，而不是独立 alpha 来源

### Default counterfactual order

当负 Kelly 集中在交易碎片化时，默认先做这些 fixed-actor 验证：

1. 提高最小调仓阈值
2. 增加 `trade_lock` / `cooldown`
3. 过滤 `<5%` 微幅调仓
4. 抑制 `3-10d` 内的来回反向交易
5. 调整 bear / bull cap，而不是先改 reward

只有当 fixed-actor 结果说明：

- 方向正确但幅度不够

才进入下一轮重训，并把训练目标改成更偏交易质量，例如：

- round-trip 结算型 reward
- 微调仓惩罚
- short-hold flip 惩罚
- 成本后 alpha 奖励

### Output contract for Kelly review

如果这一步被触发，最终复盘里要额外包含：

1. 哪些交易 bucket 为负期望
2. 它们的数量占比
3. 它们的摩擦成本影响
4. 它们对应的 reward / execution 机制
5. 下一步优先做 fixed-actor 还是重训

## Optional External Analysis Stack

如果仓库内 summary 和 trade-attribution 还不够，再补外部工具，但不要让外部工具替代 repo 主验证链。

- `vectorbt`
  - 用途：trade-level forensics、逐笔分层、持仓时长 / PnL / cost sensitivity 统计
  - 最适合回答“哪类交易在亏钱”
- `QuantStats`
  - 用途：HTML tear sheet、rolling Sharpe、drawdown、月度热图
  - 最适合做训练后的人类可读汇报
- `pyfolio` / `pyfolio-performance`
  - 用途：传统收益路径和暴露分析
  - 作为补充，不要替代本仓库的 promotion gate
- `Qlib`
  - 继续作为主训练 / 主回测 / 主 promotion 来源
  - 不把外部分析结果当成 promotion 的唯一依据

推荐组合：

- `Qlib` 负责训练与 gate
- 仓库内 trade attribution 脚本负责第一层诊断
- `vectorbt` 负责更细的交易级法医分析
- `QuantStats` 负责面向人的报告输出

### Feature Recall Heuristics

默认把当前 top features 映射成 4 类状态语言，而不是只报名字：

- 趋势确认：
  - `anchor_gap_*`
  - `trend_gap_*`
  - `trend_slope_*`
  - `relative_strength_*`
- 压力持续：
  - `drawdown_*`
  - `cum_drop_*`
  - `below_anchor_ratio_*`
  - `days_since_high_*`
  - `loss_streak_*`
- 波动 / 恐慌：
  - `vix_*`
  - `realized_vol_*`
  - `downside_*`
  - `vol_stress_score`
  - `regime_risk`
- 路径依赖 / decay：
  - `tracking_error_*`
  - `decay_*`
  - `jump_variation_*`

如果一个基线在“持续熊市防守”很好，但在“暴跌后快速反弹”很差，默认优先怀疑：

- 缺少 stress-release 特征
- 缺少 rebound confirmation 特征
- 缺少 regime persistence / regime release 特征

而不是先加更多通用 momentum。

## Effective Feature Optimization Methods

这个 skill 不能只给 reward / cap 调参建议。默认还要提出 1-3 个 **可执行的特征优化方向**，并说明为什么它们比继续调 reward 更值得先试。

优先级顺序：

1. **Transition features**
  - 用于识别“还在压力里”还是“压力正在解除”
  - 例子：
    - `regime_risk_delta_5`
    - `vix_zscore_delta_5`
    - `downside_ratio_10_delta_5`
    - `vol_stress_score_delta_5`
2. **Persistence / hazard features**
  - 用于识别某种 regime 已持续多久，以及延续概率
  - 例子：
    - `days_above_stress_threshold`
    - `days_below_anchor`
    - `stress_cluster_size`
3. **Cross-horizon disagreement features**
  - 用于识别短期反转和长期结构是否冲突
  - 例子：
    - 短周期反弹为正，但 `anchor_gap_180` 仍显著为负
    - 短期波动快速回落，但 `drawdown_63` 仍很深
4. **Post-shock rebound features**
  - 用于识别暴跌后的再进入窗口
  - 例子：
    - `days_since_20d_low`
    - `rebound_from_20d_low`
    - `positive_day_ratio_5_after_deep_drop`

### Feature Proposal Guardrails

- 不要把“再加几个类似的 momentum / vol 因子”当成有效特征优化。
- 新特征必须尽量正交，优先补状态盲区，而不是堆叠相关因子。
- 每个特征建议都要绑定一个失败窗口，例如：
  - `2020-04` 暴跌反弹漏抓
  - `2022-07` 熊市反弹漏抓
  - `2023-01` 趋势重启识别太慢
- 如果一个建议可以先通过 execution overlay 验证方向，就先做 counterfactual，再决定是否重训。

## Quant Reality Audit

训练后复盘不能只看收益、Sharpe、MaxDD。默认还要补一次“量化现实性”检查，防止策略只在乐观假设里成立。

最低要检查这 5 项：

1. **未来数据 / 泄漏**
  - 特征是否只使用当下及过去可得信息
  - 训练选择是否直接或间接使用了 `2022` / test 标签
  - 如果结论依赖 test window 调参，必须明确标注为“诊断”，不能伪装成无泄漏选择
2. **训练-回测成本一致性**
  - 训练内成本假设和 qlib backtest 成本假设是否同向
  - 如果训练假设明显乐观于回测，优先修正训练奖励或执行层
3. **交易频率 / 摩擦鲁棒性**
  - trade count 是否异常偏高
  - 在更高成本下是否快速失真
  - 如果对 15-20 bps 成本极敏感，优先压换手，不要先加特征
4. **执行可实现性**
  - lot size、最小佣金、close-price 成交假设是否过于理想
  - 高波动窗口是否存在“看起来能交易、实际上难成交”的风险
5. **容量 / 冲击成本**
  - 当前仓位切换是否过大
  - 若策略依赖频繁大幅调仓，必须把冲击成本列为未覆盖风险

仓库内默认可先用固定 actor 做成本敏感性检查：

```bash
python scripts/cost_sensitivity_backtest.py \
  --profile balanced \
  --model runtime/model/card/balanced/soxl_balanced_actor.pt \
  --output runtime/model/card/balanced_cost_sensitivity.json
```

解释规则：

- 成本上升后收益小幅下降但排名不变：
  - 说明策略对摩擦相对稳健
- 成本上升后 Sharpe / return 快速塌陷：
  - 说明当前优化重点应转向交易压缩或更强 execution gate
- stress window 已差，成本敏感性又高：
  - 优先排除该分支，不要继续重训加特征

## Auto-Optimization Rule

skill 不能只给“建议列表”，还要默认决定下一轮最便宜、信息增益最高的实验。

默认顺序：

1. 先做固定 actor counterfactual
2. 再做量化现实性检查
3. 只有方向仍成立，才开真实重训

默认决策树：

- 如果 execution/risk overlay 已经能解释大部分变化：
  - 下一轮继续 overlay 优化，不重训
- 如果 counterfactual 方向正确，但真实重训只带来小幅改善：
  - 下一轮优先改 checkpoint / model selection，而不是继续微调同一个参数
- 如果成本敏感性很高：
  - 下一轮优先降换手、增大 rebalance gate、提高 transaction_cost / turnover_penalty
- 如果无泄漏风险没有关干净：
  - 先修 harness，再停止性能优化
- 如果 canonical baseline 仍明显更优：
  - aggressive branch 降级为参考分支，不再当默认升级目标

## Continuity Discipline

不要每轮都新建 memory 文件。默认规则：

- **当前线程内的实验过程**：
  - 写入 `codex_exec_plans/active/...`
- **跨会话仍值得保留的方法、决策、长期结论**：
  - 才写入 `runtime/<direction>/knowledge.md`
- **只是一次局部 sweep 或单轮失败尝试**：
  - 优先并入已有 memory 记录，不单独新建文件

如果这次工作的新增信息不满足“以后还会反复用到”，就不要新增 memory。

### Optimization Mapping

看到下面这些现象时，默认优先考虑对应方向：

- `2022` / stress window 恶化，`avg_pos` 升高：
  - 优先查风险预算映射，不先查特征
- `2023-2024` 变差且 trade count 很高：
  - 优先查交易频率压缩，不先查更多 reward bonus
- validation 提升但 OOS 退化：
  - 优先查 checkpoint 选择和 early stopping
- 总收益提升但 MaxDD 急剧放大：
  - 优先查暴露/仓位映射，而不是把它当成“alpha 提升”
- 改 execution overlay 就能明显改变结果：
  - 说明当前最便宜的下一步是规则层优化，而非立即重训

## Current SOXL Snapshot

下面这部分只是当前仓库实例，不是通用原则本身。

默认把当前策略分成两条线：

- `canonical balanced`
  - 当前默认最佳通用 qlib 策略
  - 目标：保住 `2022` gate，同时尽量抬高 `2023-2024` 和总收益
- `balanced_floor40_sym`
  - 当前高收益但高风险分支
  - 目标：只有在明确接受更激进风格时才继续

### Current Priority

当前默认优先顺序：

1. 保住 canonical `balanced` 的 `2022` 风控通过状态
2. 提升 `2023-2024` 收益和总 `test_proxy`
3. 再考虑追求极高总回报分支

如果一项改动让 `2022` gate 从通过变成失败，默认视为主线退化。

### Current Repo-Specific Directions

- 对 canonical `balanced`
  - 优先做更精细的 stress de-risking
  - 优先做交易频率压缩
  - 优先改 checkpoint / early stopping
- 对 `balanced_floor40_sym`
  - 先修 `2022` 去风险，不先加训练轮数
  - 先压 `avg_pos`，再谈总收益
  - 先看危机态/高风险态仓位控制，再看普通趋势收益

### Current Example

在当前仓库里，`balanced_floor40_sym` 的一个快速 counterfactual 例子是：

- 单独降低 `bull_min_cap` 几乎不改变 fixed-split 结果
- 单独收紧 `bear_min_cap` 才会对 `2022` 暴露和回撤产生可见影响

这类现象说明：

- 当前瓶颈更可能在 stress de-risking，而不是牛市放宽参数
- 先做 counterfactual rule sweep，比直接重训更划算

## References

- 研究与行业实践摘要：读 `references/research-notes.md`
- 自动化对比脚本：运行 `scripts/compare_backtests.py`
