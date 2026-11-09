# 冲击集中度 × 自身回撤：风险信息通过，行动未传递

**SOXX 判定：历史费用后稳定超额未取得。** 这是一项现有数据、同状态物理动作的风险信息筛选，未运行连续三资产完整账户或 0/2/5bp SOXX 门。

前一完整交易日 SOXX 21 日绝对冲击集中度，与决策时已接收自身回撤相乘；对照同时含两者主效应、十项自身状态、两项动作和报价费率。Qlib 来源覆盖 2018–2020 的 558 日；pi2 保存的未来路径风险标签在 2020 年 247/248 日可用。“历史风险已知”另有不同含义，不作为本次过滤。2018–2019 的 303 日、15150 个已成熟拟合动作训练；2020 的 248 日、三种预留物理动作/锚点评价。

候选 2020 按日等权风险 MSE **0.00093118**，低于加性对照 **0.00100673** 和训练日均常数 **0.01705433**；对照减候选的 10 会话区块下界 **+3.43e−7**，标准化交互系数 **+0.02642**，方向符合冻结假说。因此这个同状态局部**风险信息门通过**。但是三种预留整股行动的最小预测风险排序在 248 日内 **0 日改变**；候选交互置零也 0 日改变，5 个选中锚点的结果未知。冻结行动门失败，局部风险改善和费用后价值增量均为零。固定三行动排序版本关闭，不调符号、门限或动作集救援。该门只对这三种 held 行动与这个选择规则有效；没有同窗 Q2 风险 critic 比较或完整 actor/account 验证。

实现收据：首次 DatasetH 分组设置错误在拟合前停止；第二次两次拟合完成后在索引名对齐处失败，没有产生科学评分/Recorder；精确修复后重放同一冻结计划，保存检查点再完成两次拟合。因此实际求解器调用合计 **4** 次，最终模型 **2** 个，没有隐藏消耗。原始错误与修复范围见 `attempt1_failure.json`、`attempt2_failure.json` 和 `repair_plan.json`。

原生 Qlib Recorder `be415a07e7024c9483790172823d4900` NAS 归档 SHA `07582f2ba6e56ae6ccd4684b25c714cdb6bcb17df0f2e07848d75124be71e232`；限定恢复读取 result、predictions、action_rows、candidate_model 通过。这个因子只能保留为条件风险信息线索；要成为配置方案，还须有事前冻结的风险与费用后价值共同目标、真正改变的合法行动、与既有 Q2 critic 的匹配增量，以及自身完整账户 SOXX 门。

作用域补充：The pi2 state/policy that generated these saved action labels was itself trained using 2018-2020 development data. Held denotes held physical action IDs, not a market out-of-sample policy. The 2020 risk information pass is conditional development evidence only.
