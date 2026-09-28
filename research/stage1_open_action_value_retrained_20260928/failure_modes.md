# Failure modes

- 未通过：development_return_above_baseline
- 未通过：stress_25bps_return_above_baseline
- 未通过：max_drawdown_not_worse
- 未通过：cvar5_not_worse
- 该候选来自已反复使用的开发窗口，尚无 clean unseen holdout。
- 激活前必须核对回放与运行时使用同一自身账本状态和开盘/恢复决策函数。
- outer 只能作为 reused diagnostic，不能作为晋升依据。
