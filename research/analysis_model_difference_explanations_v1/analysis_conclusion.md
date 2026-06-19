## 中文结论

- 做了什么：本 notebook 使用 `src/analysis.build_model_difference_explanations`，把 `logistic_clf` 作为基准，对 QuantBench smoke 回测的全区间、regime、固定窗口和 window-regime 分段生成模型差异解释表。
- 核心指标：共生成 `19` 条解释；signal 分布为 `{'baseline_defends': 6, 'segment_winner': 10, 'small_gap': 3}`；decision 分布为 `{'block_or_rework_loser': 6, 'monitor': 3, 'promote_segment_candidate': 10}`。
- regime 差异：最高优先级分段是 `late_test::downtrend`，winner=`alstm_mse`，loser=`logistic_clf`，相对收益 spread=`47.8991`。
- 是否晋升：不晋升。本实验是 analysis 框架验证，只说明解释表可以作为 dashboard/notebook review 输入，不证明任何 smoke 模型可上线。
- 风险：输入来自 smoke 回测，样本短且不代表完整 QuantBench 生产表现；解释表只做分段诊断，不替代成本、滑点、压力窗口和 production health gate。
- 下一步：将 `model_difference_explanations.csv` 接入 dashboard view/filter 或 schedule 通知摘要，让 review 时直接看到每个 regime/window 的 winner、loser 和建议动作。
