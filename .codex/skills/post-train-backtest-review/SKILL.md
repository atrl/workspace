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
  - 通常是当前 canonical summary，例如 `artifacts/models/soxl_balanced_summary.json`
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
  --model artifacts/models/soxl_balanced_actor.pt \
  --output runtime/post_train_review/balanced_cost_sensitivity.json
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
  - 才写入 `memory/...`
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
