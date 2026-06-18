# SOXL 模型差异 Review Briefing

## 中文结论

- 共生成 175 条 review 项，信号分布：{'risk': 3, 'strength': 25, 'weakness': 147}。
- 是否晋升：`promote_candidate` 只能作为分段候选；若同一模型存在 `block_or_rework`，应先处理压力区间弱点。
- Review 顺序按相对收益差、稳定性差、回撤差和换手效率综合排序。

| rank | model | segment | signal | decision | note |
| ---: | --- | --- | --- | --- | --- |
| 1 | alstm_mse | stress | weakness | block_or_rework | alstm_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 2 | autoformer_mse | stress | weakness | block_or_rework | autoformer_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 3 | darnn_mse | stress | weakness | block_or_rework | darnn_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 4 | dlinear_mse | stress | weakness | block_or_rework | dlinear_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 5 | fedformer_mse | stress | weakness | block_or_rework | fedformer_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 6 | gat_mse | stress | weakness | block_or_rework | gat_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 7 | gcn_mse | stress | weakness | block_or_rework | gcn_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 8 | gru_mse | stress | weakness | block_or_rework | gru_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 9 | hawkes_gru_mse | stress | weakness | block_or_rework | hawkes_gru_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 10 | hypergraph_mse | stress | weakness | block_or_rework | hypergraph_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 11 | lstm_mse | stress | weakness | block_or_rework | lstm_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 12 | market_graph_mse | stress | weakness | block_or_rework | market_graph_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 13 | mlp_mixer_mse | stress | weakness | block_or_rework | mlp_mixer_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 14 | patch_tst_mse | stress | weakness | block_or_rework | patch_tst_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 15 | rgcn_mse | stress | weakness | block_or_rework | rgcn_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 16 | ridge_ic | stress | weakness | block_or_rework | ridge_ic 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 17 | ridge_mse | stress | weakness | block_or_rework | ridge_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 18 | rsr_mse | stress | weakness | block_or_rework | rsr_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 19 | sequence_lag_ridge_mse | stress | weakness | block_or_rework | sequence_lag_ridge_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
| 20 | sfm_mse | stress | weakness | block_or_rework | sfm_mse 在 stress 弱于基准 logistic_clf：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。 |
