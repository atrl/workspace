# Post-Train Review Research Notes

这些参考不是让你照抄术语，而是帮助你在训练后复盘时避免两类常见错误：

1. 把单次好看的 backtest 当成稳定改进
2. 只看总收益，不看切片、门槛和风格漂移

## 1. Bailey et al. - Probability of Backtest Overfitting

来源：

- David H. Bailey, Jonathan M. Borwein, Marcos Lopez de Prado, Qiji Jim Zhu
- "The Probability of Backtest Overfitting" (2015)
- https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf

对这个 skill 最重要的启发：

- 选出来的 in-sample 最优策略，可能在 out-of-sample 连中位数都跑不过。
- 训练后复盘不能只写“最新 checkpoint 分数最高”，还要写：
  - 这个最优点是否只在 validation 上成立
  - OOS 切片是否同步改善
  - 是否只是从很多候选里挑中了一个幸存者

落地到本仓库：

- 每次训练后至少比较 `valid`、`test_proxy`、`test_2022`、`test_2023_2024`
- 如果 validation 提升而切片退化，把它当成潜在 selection bias，而不是改进

## 2. Lopez de Prado - Deflated Sharpe Ratio

来源：

- Marcos Lopez de Prado
- "Deflating the Sharpe Ratio"
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2465675

对这个 skill 最重要的启发：

- Sharpe 会被多重测试和非正态收益分布夸大。
- 当你比较很多 reward、floor、gate、feature 组合时，名义 Sharpe 不能直接当结论。

落地到本仓库：

- 不要因为某次 run 的 Sharpe 高了就忽视 `2022` drawdown 或 `avg_pos`
- 报告里应该注明这是“在当前候选集中的相对结果”，不是无偏统计真值

## 3. AQR - Can Machines Learn Finance?

来源：

- Bryan Kelly, Seth Pruitt, Yinan Su
- AQR working paper / Alternative Thinking summary
- https://www.aqr.com/-/media/AQR/Documents/Alternative-Thinking/AQR-Alternative-Thinking-2Q19-Can-Machines-Learn-Finance.pdf

对这个 skill 最重要的启发：

- 模型选择的关键不只是灵活度，而是是否强调可靠的 out-of-sample 表现。
- Cross-validation 和“as-if out-of-sample”测试，是约束过拟合的核心。

落地到本仓库：

- 复盘应优先强调 fixed-split + walk-forward，而不是单次 full test
- 如果训练后结论无法跨窗口复现，应优先降复杂度或收紧选择标准

## 4. Man Group - Backtesting

来源：

- Man Group / Man Institute
- "Backtesting" (2015)
- https://www.man.com/insights/backtesting

对这个 skill 最重要的启发：

- 机构实践里会对 backtest 的 Sharpe 做折扣，因为研究过程本身包含数据挖掘。
- 换句话说，研究者默认怀疑“看起来很好”的结果，而不是默认相信。

落地到本仓库：

- 训练后总结要默认寻找代价和漏洞：
  - 收益上升是否来自更高暴露
  - drawdown 是否恶化
  - promotion gate 是否被破坏

## 5. AAMDRL - Walk-forward analysis for DRL

来源：

- "AAMDRL: Augmented Asset Management with Deep Reinforcement Learning" (2020)
- https://arxiv.org/abs/2010.08497

对这个 skill 最重要的启发：

- DRL 交易系统需要重复 train-test 的 walk-forward 分析，而不是只给一段静态回测。

落地到本仓库：

- 如果最新模型和旧模型差异很大，但还没有 walk-forward 证据，不要下强结论
- 复盘里要明确：结论是 fixed-split 结论，还是 fixed-split + walk-forward 结论

## 6. How Companies Commonly Review a New Model

虽然公开披露通常不会给出完整内部模板，但从论文、白皮书和机构研究文章看，训练后比较大体遵循同一顺序：

1. 先看是否通过硬门槛
2. 再看分窗口表现是否一致
3. 再看收益是怎么换来的
4. 最后才决定继续放大、继续调参，还是回滚

把这个顺序落实到本仓库，就是：

1. `promotion.fixed_split`
2. `test_2020` / `test_2022` / `test_2023_2024`
3. `avg_pos`、`max_dd`、checkpoint 选择、特征归因
4. 只保留 1-3 个下一步动作
