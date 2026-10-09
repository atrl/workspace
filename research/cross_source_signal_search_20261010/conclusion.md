# 跨数据源因果信号搜索（中文结论）
- 做了什么：在 qlib 原生链路上，用 60 条预登记因果暴露规则，覆盖 5 类数据源（macro_vol_20261010, soxl_daily, soxl_intraday_t1, soxx_constituents, soxx_daily）；阈值用扩展窗口分位，无收益预测、无拟合参数。
- 选择：仅使用 development ['2014-01-02', '2022-12-30'] 的稳定超额（收益+回撤+Sharpe 同超）。
- 选中信号：None。
- development 稳定超额信号：无。
- 决定：`no_stable_excess_across_sources`；promotion_eligible=False。
- 风险：诊断搜索可能过拟合；命中必须经独立前向窗口验证；成分股历史止于 2026-07-02；SOXL 杠杆衰减只由真实价格体现；本结果不构成生产或实盘证明。
- qlib Recorder：1d61099221694021bed04d7fc3c7d45f；as_of=2026-07-02；qlib docs：['https://qlib.readthedocs.io/en/latest/component/data.html', 'https://qlib.readthedocs.io/en/latest/component/recorder.html']。
