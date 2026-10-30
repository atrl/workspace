# 固定股数单一买单：五会话参考与向量化身份通过

**SOXX判定：历史Goal未通过。** 本卡只对新单笔执行合同做同状态五会话机械核验，零新模型、零候选账户；合同v2的SOXX 0/2/5bp财富、风险、年/126日窗和区块门未运行。按冻结[计划](plan.json)选择五个普通早盘、13:00半日市、2020恢复前/后及旧多单来源案例，使用逐层点时f696+b472 Qlib父本、双ETF原价/调整价分钟与原价整股Tiered费率。执行见[notebook](stage1_v5_fixed_share_path_parity_20261030.ipynb)，具体订单和哈希见[`audit.json`](audit.json)。

在完成分钟只用当时NAV、现金和原价收盘锁定SOXX或SOXL整股数，下一bar**仅尝试这一买单**。同一父本开盘状态复制HOLD与EVENT，两支事件后五会话锁定，不调用旧b472/旧beta上限；恢复后以已结算原生股数、现金、费用计数和之前高水位继续。五例EVENT均只有一笔指定资产买单，另一资产股数差0。逐事件参考[`stage1_capfree_lock_episode.py`](../../src/backtest/stage1_capfree_lock_episode.py)与向量化[`stage1_capfree_lock_vectorized.py`](../../src/analysis/stage1_capfree_lock_vectorized.py)的HOLD/EVENT**终端NAV、可用mark最大回撤、股数、现金、Tiered费用和月度计数最大差全部0**。原2018-10-15的附带卖SOXL例，在新版本中仅保留买SOXX。合成决策时不足一股的EVENT为明确无成交，与HOLD终端/回撤相同。

首次notebook执行在Qlib读取前因计划缺现有检查点代码身份键而停止，保留[`attempt1_preflight_failure.json`](attempt1_preflight_failure.json)及原计划；只补键后按原五案例做4次Qlib读取，无收益或年份调整。旧完整目标权重Recorder保留其原动作作用域，本卡**不能**借用其五会话费后价值、2021/2022留出结论或声称超额。下一批先事前冻固定原价股数动作的全开发日历来源/整股支持、资金不足/no-share状态、五会话终端成熟和预算；来源门过后再一次生成Qlib Recorder新标签并核NAS，然后按完整日分组/五会话purge执行分钟信息门。真实券商拒单/部分成交及历史分钟首收仍`needs_data`。
