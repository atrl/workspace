# Stage1 历史研究 Goal：短活动说明

## 目标

仅交易 SOXL、SOXX、现金，使用项目现有 Qlib/NAS 数据研究因子和连续配置，
在完整自身状态的原生 Qlib 账户中取得扣费后稳定 SOXX 超额。
合同 v2 的 0/2/5bp 财富、可用盘中与收盘回撤、CVaR5、完整年度、
126 交易日滚动窗口及区块不确定性门必须由同一冻结候选全部通过。
`historical_research_passed` 当前为 false。历史通过仍不是独立前向或生产晋升。
具体门槛见 `docs/three_asset_research_contract.md`。

## 当前证据和下一任务

截至 2026-11-12，已有来源与多因子局部门陆续关闭，0个新完整账户。
[动作相关风险/价值检验](../research/stage1_shock_action_marginal_value_risk_20261112/conclusion.md)
虽使2020风险MSE点值略降，但系数方向、价值校准及行动价值门失败。
保存的同锚点高SOXL相对较高SOXX价值在248日中预测正241日、
实际正137日，错误看多108日。2020季度只作事后归因。

新增[前日分钟路径来源](../research/stage1_intraday_jump_path_source_20261112/conclusion.md)
在2018–2022有1055/1056个有效次日来源，与旧逐日跳跃和已用日内波动
控制不近重复；但其[固定配对价值检验](../research/stage1_intraday_jump_relative_value_20261112/conclusion.md)
在2020只微幅改善MSE点值，区块下界、系数方向及费用后行动门失败，
0个候选账户。来源新颖不等于可交易信息。pi2状态策略使用过2018–2020
开发数据，预留动作不是独立市场样本；历史真实分钟首次送达仍未认证。

非价格来源的[H.8同锚点价值检验](../research/stage1_h8_relative_value_triage_20261112/conclusion.md)
零拟合：2020条件相关为负且区间不跨零，但2019区间跨零，固定跨年门失败；
历史首次网络送达也未认证。CFTC持仓集中度在2019仅198日/42个报告期
有条件来源、原始版本仍不具严格PIT资格，不能据旧正风险关联直接拟合。

下一项仍限现有 Qlib/NAS，寻找**实质不同的决策前经济机制**，同时
解释现金机会与SOXL相对选择/尾损，并冻结同状态整股费用后行动到完整
账户的传递。不能靠已关闭路线改名、翻号、换频率或按2020季度选开关。
方法与旧详细结论分别在 `docs/research_failure_to_feature_sop.md` 和原卡。

客户端 Goal 2026-11-12 已直接读回 active；科学状态仍未通过。
