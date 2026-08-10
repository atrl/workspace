# Failure modes

- 开发门全部通过。
- 该候选来自已反复使用的开发窗口，尚无 clean unseen holdout。
- shadow Stage1 状态与实际执行持仓分离，生产接入前必须验证状态账本同步。
- outer 只能作为 reused diagnostic，不能作为晋升依据。
