# SOXL 稳定超额前沿研究（中文结论）

- 做了什么：在 qlib 原生链路上执行 9 条预登记因果暴露规则（含 SOXX 基准对照）；无收益预测、无拟合 alpha、零模型拟合；选择仅使用 development ['2016-01-04', '2022-12-30']。
- 核心指标（选中 trend_vol_switch_v35）：复用区间 ['2023-01-03', '2026-09-25'] 收益 348.00%（SOXX 397.60%），最大回撤 -68.48%（SOXX -41.67%），Sharpe 0.93（SOXX 1.33）。
- 新前向区间 ['2026-09-26', '2026-10-08']：收益 0.25%（SOXX 0.25%），最大回撤 -4.45%（SOXX -4.45%），Sharpe 0.41。
- regime/窗口差异：选中策略平均 SOXL 权重 0.000、平均现金权重 0.000；滚动年正超额 3/4。
- 是否晋升：False；决定 `no_stable_excess_in_registered_family`。前向同时超过 SOXX 收益与回撤：False。
- 风险：前向样本短；无收益预测；SOXL 杠杆衰减仅由真实价格体现；诊断与前向区间不参与选择；本结果不构成生产或实盘证明。
- 下一步：无收益预测的日频风险/趋势规则不能产生稳定超额。后续应转向已被证明的日内机制（早期会话状态 + 现金时点控制），并冻结结构等待新的连续前向证据。
- qlib Recorder：f0d3bdc1e5b541d2a188c6a4486417aa；provider_uri=/Users/rziiiii/dev/workspace/data/qlib/soxl；as_of=2026-10-08；qlib docs：['https://qlib.readthedocs.io/en/latest/component/data.html', 'https://qlib.readthedocs.io/en/latest/component/recorder.html']。
