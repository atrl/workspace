# qlib recorder 模型差异 Review

## 中文结论

- 共生成 7 条 review 项，信号分布：{'strength': 3, 'weakness': 4}。
- 是否晋升：`promote_candidate` 只能作为分段候选；若同一模型存在 `block_or_rework`，应先处理压力区间弱点。
- Review 顺序按相对收益差、稳定性差、回撤差和换手效率综合排序。

| rank | model | segment | signal | decision | note |
| ---: | --- | --- | --- | --- | --- |
| 1 | gcn_mse | all | weakness | block_or_rework | gcn_mse 在 all 弱于基准 ridge_mse：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 2 | gcn_mse | early | strength | promote_candidate | gcn_mse 在 early 强于基准 ridge_mse：相对收益和稳定性改善，可作为该分段的晋升候选。 |
| 3 | gcn_mse | stress | weakness | block_or_rework | gcn_mse 在 stress 弱于基准 ridge_mse：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 4 | gcn_mse | stress_window | weakness | block_or_rework | gcn_mse 在 stress_window 弱于基准 ridge_mse：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 5 | gcn_mse | stress_window::stress | weakness | block_or_rework | gcn_mse 在 stress_window::stress 弱于基准 ridge_mse：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 6 | gcn_mse | bull | strength | promote_candidate | gcn_mse 在 bull 强于基准 ridge_mse：相对收益和稳定性改善，可作为该分段的晋升候选。 |
| 7 | gcn_mse | early::bull | strength | promote_candidate | gcn_mse 在 early::bull 强于基准 ridge_mse：相对收益和稳定性改善，可作为该分段的晋升候选。 |
