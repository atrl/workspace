# Stage1 纸面前向观察：代码身份与来源新鲜度

只读检查密封 release 的 **12** 份代码、当前 Git 原字节、已登记调度、Qlib 15 分钟日历与 NAS 最新已验证快照。当前两份代码 SHA-256 不符，精确补丁保存在 `code_diff.patch`。`stage1_open_action_value.py` 仅新增可选 `state_transition` 参数；纸面调用没有传入该参数，仍走原默认转移。`intraday_recovery_direct_allocation.py` 则在 provider 声明 `$is_complete` 时，只采用首四条都完整的首小时 bar，这是对未来不完整来源的**实际语义变化**，不能通过改 manifest 哈希跳过审核。

当前 provider 的确声明 `$is_complete`。用 Qlib `D.calendar`/`D.features` 对 10 月 19–21 日 SOXL/SOXX 共 **6 个标的日、24 条首小时 bar** 的限定检查，完整标志全为 1；这只说明这些已完成日的过滤是空操作，**不证明将来的部分 bar 与旧 release 等价**。本轮截至 **2026-10-23T01:05:05.842513+00:00**，该 Qlib provider 日历及 NAS 最新已验证 `us_semis_intraday` 快照都只到 **10 月 21 日**，尚不足以重建 10 月 22 日首小时。数据更新何时到达需按实际后续收据确认，不能从一次观察断言调度故障。

`launchctl` 有 `com.soxl.stage1.regime.paper` 登记；先前 10 月 22 日运行日志已记载 release 哈希拦截。此次没有启动、重启或更新纸面 release，也没有模型拟合、账户、订单、SOXX 成绩或合格前向日。要恢复未来观察，须分别完成：①对首小时完整性语义做独立审核并重新密封 release；②确保当次决策前可用的已验证分钟来源进入 Qlib provider；③按冻结前向计划确认双账户、时间与风险门。已有未通过的 Stage1 纸面基线即使恢复运行，也不能冒充新候选跑赢 SOXX。
