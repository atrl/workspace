# 路径质量 × 产品跟踪波动：固定双因子结果

**SOXX 判定：未取得超额，历史 Goal 未通过。** 本卡只检验同一逐层点时父本的次日等名义 beta、整股 Tiered 局部选择价值；没有连续三资产模型或完整 0/2/5bp 账户。

前一完整交易日的 SOXX 路径方向效率与 SOXL 跟踪误差波动，在 2018–2020 的 558 个来源日均可得，来源秩相关 −0.0397。冻结的两个主效应 Ridge 与新增交互 Ridge 各拟合一次：2018–2019 有 303 个成熟训练日，2020 有 241 个成熟测试日。交互候选的 MSE 为 8.0736e−9，高于对照 7.9027e−9；交互系数 +1.052e−6，与预定负号相反。

2020 年首个来源意向有 242 日，其中 227 日两臂整股结果可评价、15 日未知。交互只改变 2 日的 SOXL/SOXX 选择；相比对照的局部费用后增量均值 −6.00e−7，10 会话区块 2.5% 下界 −1.893e−6。固定信息门和行动门均失败，关闭这个交互版本，不改变符号、窗口、alpha、阈值或评估年救援。2021+ 标签结果未读取。

这个失败定位为交互没有稳定地改善预测，也几乎没有传到整股资产选择。来源历史价格为调整价代理，首次真实接收时点未认证；训练标签按事后两臂可成交筛选，现场意向及连续账户尚未验证。原生 Qlib Recorder `a48bca19964b4e54b9d04a4674fcb84a` 已上传 NAS，归档 SHA `4ee90af1abf244b8e3ac50af7b801202ca83baa9d392eb1becdca5c0248b17ec`；限定恢复读取 result、predictions、action_rows 通过。下一机制转向现金部署价值与既有盘中代理的条件关系，先做来源/旧路线去重。

目标读取范围说明：Saved Qlib label object was deserialized with its full sealed year range, then filtered to 2018-2020 before any target computation, fit or metric. The result field 2021plus_target_outcomes_accessed=false means no 2021+ target analysis; it does not mean those bytes were never loaded.
