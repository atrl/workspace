# SOXL 模拟盘恢复记录（2026-09-28）

本次恢复原富途 `SIMULATE` 周期 `stage1_2026-08-03_v2`，不重置账户。起始本金为8月3日的10万美元；20笔成交截至8月12日，之后持仓未调：SOXL 200股、SOXX 98股、现金32,959.769美元。

9月28日17:00:13.647–17:00:15.610上海时间只读查询，账户权益115,802.529美元，总收益 **15.802529%**。原16:15观察保留；现金、持仓和20笔成交完全不变，517.80美元权益变化仅来自市场估值。用户确认“没有加减资金，也没有再次重置”。按9月25日正式收盘价、使用查询时现金重估为119,372.409美元（19.372409%），不是独立历史结单。停机后收益来自持仓，不能当作模型持续运行收益。现金差额30.524美元为未分配净投资现金流；股息、费用及到账日仍未知。

证据入口：[实际收益审计 notebook](../production/stage1_recovery_20260928/stage1_recovery_20260928.ipynb)、[绑定原始快照哈希的确认](../runtime/soxl/stage1_futu_sim/recovery_evidence/20260928/net_cash_adjustment_confirmation.json)。大明细和原始快照以 notebook 指向的 Qlib Recorder 为准。

旧冻结权重缺失，用户已授权按固定 `pit_only_aux_005` 重训，产生新身份，不冒称恢复旧权重：[重训 notebook](../research/pit_constituent_auxiliary_training_allocation_retrained_20260928/pit_constituent_auxiliary_training_allocation_retrained_20260928.ipynb)、[冻结产物 manifest](../research/pit_constituent_auxiliary_training_allocation_retrained_20260928/frozen_bundle_manifest.json)。HOLD 保持 C=0.10，仅开发期验证原四阈值，风险门不变；[净费用有界验证](../research/stage1_open_action_value_net_fees_bounded_20260928/conclusion.md)中四个阈值均未通过全部门。原固定0.50候选的连续权重调仓日由1713降至1457（减少14.94%），开发期10bp收益115.04%至170.74%，最大回撤29.93%至30.08%，CVaR仍较差。用户明确选择“仅在模拟盘试运行新候选，保留未晋升标记”；[授权](../production/stage1_futu_sim_v1/paper_trial_authorization_20260928.json)只对应该结果，不能用于其他模型或实盘。

已修复：终态部分成交漏记、重复执行与账本现金误判；标签按实际可得时间过滤，开盘不读首小时未来特征；候选使用自身持仓与风控状态。费用逐腿扣现金，再按净资产归一持仓，修复“扣收益却不扣现金权重”。净费用复验已经完成；研究门保持失败，模拟授权没有改写研究结论。模型、数据、Recorder均按manifest精确保留到Git，防止再次只剩摘要而丢失权重。

启动验收：

1. 核对 manifest 哈希、数据截止、原20笔成交、现金确认作用范围；新现金差异不可沿用旧确认。
2. 通过账本、调度、费用守恒测试，再执行 `production/run_stage1_futu_sim.sh --slot start-now --dry-run`，确认账户、整数股计划、HOLD及恢复信号语义。
3. 校验常驻调度和OpenD恢复任务；本次仅在明确授权的模拟试验中运行，等待常规交易时段的新鲜行情；未验证的未来成交不算已完成。

[模拟试验入口](../production/stage1_hold_paper_trial_20260928/conclusion.md)已实际执行只读验收：hash、账户、QLib数据通过，盘前正确返回`needs_execution_mark`，手动启动为`armed`。17:20上海时间已验证OpenD行情/交易登录有效，两个LaunchAgent均已注册、首次退出码0、每60秒检查。下一次开盘检查为9月28日21:31上海时间；当前尚无本次恢复后的新成交。

环境快照：[sim-runtime-20260928.txt](../requirements/sim-runtime-20260928.txt)。128项核心针对性检查及4项dashboard检查通过；4项依赖丢失旧权重的历史比较未运行，新模型用新的冻结产物与验收独立验证。
