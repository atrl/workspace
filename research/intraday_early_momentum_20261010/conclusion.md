# 日内早盘动量稳定超额研究（中文结论）

- 做了什么：仅用 qlib 原生 15 分钟 provider，把开盘 60 分钟（09:30–10:30）可观测路径映射到当日 10:30–收盘的 SOXL/SOXX/现金仓位；7 条预登记阈值规则，无收益预测、无拟合参数、无隔夜仓；选择仅使用 development ['2020-01-02', '2023-12-29']。
- 核心指标（选中 soxx_momentum_l20）：复用区间 ['2024-01-02', '2026-06-30'] 收益 14.45%（SOXX 10.74%），最大回撤 -32.26%（SOXX -35.12%），Sharpe 0.34（SOXX 0.29）。
- 新前向区间 ['2026-07-01', '2026-10-07']：收益 -7.90%（SOXX -7.21%），回撤 -14.47%，Sharpe -1.30。
- 是否晋升：False；决定 `no_stable_intraday_excess_in_registered_family`。
- 稳健性：选中信号在相邻阈值上的条件超额符号不稳定，说明该边缘极薄且可能来自少量样本；在 25bp 成本下不能通过稳定超额。
- 风险：10:30 决策价可成交性未验证（无买卖价差/排队/延迟）；仅交易日内余下时段；前向样本短；换手高，边缘可能不能承受更宽价差；本结果不构成生产或实盘证明。
- 下一步：只有 development、复用诊断与高成本压力同时通过稳定超额，且相邻阈值不翻转，才登记冻结参数与独立前向 shadow。
- qlib Recorder：b0e963a5695c40f49bb64934d820df25；provider_uri=/Users/rziiiii/dev/workspace/data/qlib/soxl_intraday_15min；as_of=2026-10-07；qlib docs：['https://qlib.readthedocs.io/en/latest/component/data.html', 'https://qlib.readthedocs.io/en/latest/component/recorder.html']。
