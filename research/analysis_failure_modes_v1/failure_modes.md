# SOXL 模型失败模式

## 中文结论

- 共识别 4 条失败模式，以下列出 severity 最高的 4 条。
- 诊断优先级按相对收益落后、稳定性下降、回撤扩大和换手增加综合排序。

| model | segment | failure_mode | severity | review_note |
| --- | --- | --- | ---: | --- |
| gcn_mse | stress | return_stability_lag | 78.0462 | gcn_mse 在 stress 同时落后基准 ridge_mse 的相对收益和稳定性，优先检查该 regime/window 的信号适配。 |
| gcn_mse | stress_window | return_stability_lag | 78.0462 | gcn_mse 在 stress_window 同时落后基准 ridge_mse 的相对收益和稳定性，优先检查该 regime/window 的信号适配。 |
| gcn_mse | stress_window::stress | return_stability_lag | 78.0462 | gcn_mse 在 stress_window::stress 同时落后基准 ridge_mse 的相对收益和稳定性，优先检查该 regime/window 的信号适配。 |
| gcn_mse | all | return_stability_lag | 45.1654 | gcn_mse 在 all 同时落后基准 ridge_mse 的相对收益和稳定性，优先检查该 regime/window 的信号适配。 |
