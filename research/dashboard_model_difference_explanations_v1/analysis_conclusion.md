## 中文结论

- 做了什么：本 notebook 将 `model_difference_explanations.csv` 接入 `src/dashboard`，生成 dashboard export、view model 和静态 HTML。
- 核心指标：dashboard 有 `12` 个模型卡，差异解释 `19` 条；top signal=`segment_winner`，top decision=`promote_segment_candidate`。
- regime 差异：最高优先级分段是 `late_test::downtrend`，HTML 中已经有 `Model Difference Explanations` 面板、explanation signal filter 和 difference decision filter。
- 是否晋升：不晋升。本实验只验证 dashboard 展示链路，不证明任何 QuantBench smoke 模型可进入 production。
- 风险：输入来自 smoke 回测和研究解释表，样本短；dashboard 只展示 review 线索，不能替代 objective、backtest、production health 和 paper execution gate。
- 下一步：把该面板纳入 schedule refresh 产物，让定时 dashboard export 自动携带模型差异解释。
