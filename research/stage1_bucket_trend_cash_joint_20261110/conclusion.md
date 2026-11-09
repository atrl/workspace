# 桶卖压 × 前期趋势：固定现金价值组合结果

**SOXX 判定：未取得费用后稳定超额，历史 Goal 未通过。** 本卡是现有 Qlib/Recorder 源上的固定股数 H5 局部现金买入价值检验，没有连续三资产自身账户，也没有 0/2/5bp 完整策略结果。

来源门通过：桶卖压代理来自已完成分钟，前期 SOXX20 日趋势来自上一完整会话；2019/2020 分别有 252/231 个来源可用日。2018–2019 的 295 日、107725 条成熟行按日等权训练两个固定 Ridge；2020 的 253 来源日中 244 日有成熟标签。

2020 年按日等权 MSE：主效应对照 2.20605e−5、交互候选 2.21836e−5、训练日均常数 2.25570e−5。交互候选劣于对照；标准化交互系数 −2.150e−4，与事前正号相反，10 会话区块的对照减候选损失下界 −2.991e−7。首个正分行动相对对照改变 25 日，单独置零候选交互只改变 12 日；两模型行动差不能全算交互的直接作用。可配对 244 日的局部增量均值 +5.01e−6，但区块下界 −2.919e−5；另有 9 日选中结果未知。候选对 HOLD 的正局部值不构成对照增量或 SOXX 账户超额。信息门和行动门均失败，固定版本关闭，不调符号、阈值、窗口或年份。2021+ 目标未读取。

瓶颈是交互未改善跨年校准，也未形成稳定、完整可知的资金动作增量。桶量仅为 raw close×volume 的价格分类代理，不是订单流；本卡仍使用旧调整价父本的固定动作标签，历史真实分钟首收和完整账户反馈未认证。原生 Qlib Recorder `fbc6a5d8414442aba56c41e2d4c3a00e` 已在 NAS 归档 `532fe62401929aba42eb757c16bb72ad3d4041156366917cb36ca041985ff01f`，限定恢复读取 result、predictions、action_rows 通过。下一研究应查一个不同于现有价格/量组合的决策前机制；不能用本卡微调获利。

目标读取范围说明：Saved Qlib label object was deserialized with its full sealed year range, then filtered to 2018-2020 before any target computation, fit or metric. The result field 2021plus_target_outcomes_accessed=false means no 2021+ target analysis; it does not mean those bytes were never loaded.
