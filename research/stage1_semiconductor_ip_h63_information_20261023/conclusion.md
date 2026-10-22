# Stage1 美联储半导体工业生产 H63 方案：拟合前关闭

结论：`HOLD_DUPLICATE_CLOSE_BEFORE_FIT`。Qlib 无拟合预检确认 2,956/882 条成熟样本和来源完整，不能据此重跑已关闭的因子。

此前 [`stage1_semiconductor_ip_information_20261022`](../stage1_semiconductor_ip_information_20261022/conclusion.md) 已用同一 `ip_mom_pct_same_vintage` 与原 Stage1 39 项，在复用历史上完成两次 Ridge 和一次区块检验；新增系数 −0.00359828，成对 MSE 改善 −0.0001121125，两半段和下界均为负。H63 只改变预测期限并删除发布年龄/基年控制，不构成新的实体产出机制。

本版初始拟合代码还继承 H5 变量名和 3014/940 字面断言；机械修复及原无拟合凭据均已记录，未运行修复后的拟合。新增模型拟合 0、Qlib 写入 Recorder 0、原生账户 0。

下一步从冻结 Stage1 的全区间现金拖累和有效保护对比出发，选择另一个可事前观测的独立机制；不以 H63、符号或控制项调整补救该工业生产路线。
