# 中文结论

- 做了什么：停止围绕 Stage1 30% SOXL 上限做小幅日内调参，改成低频核心结构。SOXX 收盘价相对 140/150/160 日均线的三个独立状态时钟各投一票；每个时钟上穿均线 8% 后持 SOXL，跌回均线后降为 SOXX，三票均值形成 0/33%/67%/100% SOXL 权重。信号只用当日收盘已知数据，下一交易日开盘执行。
- 状态和成本修复：2013 年开始预热并连续运行一次状态机，开发/outer 边界不重置；显式回放按真实目标权重变化计双边换手，另由 qlib SignalRecord/PortAnaRecord 生成独立投组证据。
- 开发期 2016–2022：候选 10bp 累计收益 1072.05%，SOXL 买入持有 509.16%，超额 562.89 个百分点；MaxDD -64.13% 对 SOXL -91.29%；调仓日 26。
- 复用 outer 2023–截至 2026-08-11：候选 10bp 累计收益 1522.71%，SOXL 1309.43%，超额 213.28 个百分点；MaxDD -64.53% 对 SOXL -88.66%；调仓日 11。50bp 压力仍保存在 benchmark sidecar 与 recorder。
- qlib 原生投组对账：同一 outer 的账户收益 1476.24%，原生 SOXL benchmark 1273.88%，超额 202.36 个百分点，MaxDD -69.42%。相对显式 open-to-open 回放的收益差 -46.47 个百分点，来自 qlib 的逐日账户、持仓取整和交易估值口径；两个口径均保留，不混算。
- 与生产 Stage1 的同窗比较（2023-01-03–2026-07-21）：趋势核心 1712.06%，Stage1 543.34%。这回答了此前“只超过一点”的问题：Stage1 的 30% SOXL 上限在强牛市结构上先天限制收益，低频杠杆档位模型才是量级更大的方向。
- 反证：同一冻结参数在 TQQQ/UPRO/SPXL/TECL 的复用 outer 全部未跑赢各自 3x 买入持有，cross-asset gate=False。说明 SOXL outer 超额带半导体周期特异性，不能视为普适稳健性。
- 是否晋升：not_promoted_outer_contaminated_and_cross_asset_gate_failed。不改生产模拟盘；原因不是收益不足，而是 outer 已在研发中被查看且跨资产外层失败。该卡是强 retrospective challenger，不是干净可晋升模型。
- 风险：即便改善，历史 MaxDD 仍约 65%；趋势反转、跳空和长时间震荡会产生滞后损失。下一步必须预注册参数并积累新的 forward holdout，或引入截至信号日可得的半导体周期 PIT 信息后重新训练。
- qlib recorder：d7a793983b554983858d978795ab57f0。完整预测、标签、positions、portfolio report、手工逐日回放和目标权重只在 recorder，卡根保持 slim。
- 不构成买卖建议。
