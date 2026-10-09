# 日内早盘动量稳定超额研究（中文结论）

- 做了什么：仅用 qlib 原生 15 分钟 provider，把开盘 60 分钟（09:30–10:30）可观测路径映射到当日 10:30–收盘的 SOXL/SOXX/现金仓位；7 条预登记阈值规则，无收益预测、无拟合参数、无隔夜仓；选择仅使用 development ['2020-01-02', '2022-12-30']。
- 核心指标（选中 early_momentum_l2）：复用区间 ['2023-01-03', '2026-09-25'] 收益 27.92%（SOXX 47.07%），最大回撤 -61.37%（SOXX -35.12%），Sharpe 0.36（SOXX 0.58）。
- 新前向区间 ['2026-09-26', '2026-10-07']：收益 -1.31%（SOXX 0.41%），回撤 -3.71%，Sharpe -1.57。
- 是否晋升：False；决定 `no_stable_intraday_excess_in_registered_family`。
- 风险：10:30 决策价可成交性未验证（无买卖价差/排队/延迟）；仅交易日内余下时段；前向样本短；本结果不构成生产或实盘证明。
- 下一步：只有 development、复用诊断与高成本压力同时通过稳定超额才登记冻结参数与独立前向 shadow。
- qlib Recorder：90422ab9e8504b8b864d0d67c957f448；provider_uri=/Users/rziiiii/dev/workspace/data/qlib/soxl_intraday_15min；as_of=2026-10-07；qlib docs：['https://qlib.readthedocs.io/en/latest/component/data.html', 'https://qlib.readthedocs.io/en/latest/component/recorder.html']。
