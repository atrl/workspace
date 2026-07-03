# 中文结论

- 做了什么：读取 high-chop LGB/XGB 模型的最新 target_position、qlib 日线、Futu regular/extended 1m，检查 close-to-close 信号在收盘、盘后和次日开盘路径上的执行风险。
- 7/2 结论：2026-07-02 target_position=0.0005，属于 near_cash；当天 RTH open-to-close=-17.22%、RTH low-from-open=-22.97%、盘后首笔相对 RTH close=-0.24%、盘后 MAE=-0.81%。这说明修复日线后模型没有在 7/2 满仓承受暴跌。
- 近期窗口：2026-06-26 到 2026-07-02 全部 5 个交易日 near_cash，max_abs_target=0.0005；这能解释为什么 7/2 没吃大回撤，但不能证明满仓/空仓切换的实盘能力。
- 是否晋升：execution_not_proven；close-to-close 回测仍需要在固定信号生成时间、盘后/次日开盘成交假设、滑点和分钟 MAE/MFE 上继续验证。
- qlib recorder：dad036329bc94f82b0fee93d658f7bb1
