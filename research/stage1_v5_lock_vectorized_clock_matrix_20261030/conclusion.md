# 向量化五会话双臂：四时钟九动作机械身份通过

**SOXX判定：历史Goal未通过。** 本卡零新拟合、零候选资金账户，合同v2的0/2/5bp财富、回撤、CVaR、年度、126日窗及区块门未测。固定复用[四例事件时钟卡](../stage1_v5_capfree_lock_clock_matrix_20261030/conclusion.md)的2018半日市、2019普通早盘及2020恢复前/后来源事件，加入SOXL等beta买入、SOXX买入和半日市双向SOXL卖出＋SOXX买入，共**9个动作臂**。源Qlib、父本Recorder与逐层点时模型身份先核SHA；结果见[notebook](stage1_v5_lock_vectorized_clock_matrix_20261030.ipynb)与[`audit.json`](audit.json)。

[`stage1_capfree_lock_vectorized.py`](../../src/analysis/stage1_capfree_lock_vectorized.py)把五会话Qlib分钟和次开盘标价按真实顺序编译。恢复后事件从10:32原生已结算股数/现金及Tiered计数起，`start_at=10:33`，初始高水位包含此前q0分钟路径；恢复前两分支共同锁定并取消之后的旧恢复。2018-11-23实际闭市13:00。9个动作臂各自HOLD/EVENT的**终端NAV、可用mark最大回撤、股数、现金、费用和月度Tiered股数计数**对修正逐事件引擎的最大差全部 **0**；合成零整股目标也保留与HOLD相同的路径。当前会话09:48之前的无效bar若不进入决策后路径，不会无故使整日失格；后四个会话仍要求完整有效。

这是**机械身份**，不是九次收益检验或全部分钟路径标签。10:30–10:32的事件/恢复先后仍`unknown`；2021/2022价值列未用，真实首次分钟接收、券商成交与调整价独立应收分红仍未认证。下一步冻结一批覆盖所有决策前意向的HOLD／SOXL／SOXX五会话数值标签：future fill与路径完整只决定标签资格，no-fill/unknown保留；按完整日分组并以五会话期限purge。批量结果须写Qlib Recorder、验证NAS收据/限定读回，且不能把固定0.10两臂值冒充最终连续三资产接收器。
