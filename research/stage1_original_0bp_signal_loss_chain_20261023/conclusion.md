# 原 Stage1 0bp 信号至损失诊断 v1：来源日历边界停止

首轮只读 notebook 在 Qlib 日历校验处停止。账户复盘 provider 从 2023-01-04 开始，而冻结原 Stage1 信号帧有 2023-01-03 至 2026-10-07 共 944 日。原 Stage1 39 项所用的 daily_prices provider 包含全部 944 日。未写新 Recorder、未拟合、未跑账户、未重放。v2 只更换诊断 Qlib 日历来源，保留全部源对象、账户损益及分析定义。
