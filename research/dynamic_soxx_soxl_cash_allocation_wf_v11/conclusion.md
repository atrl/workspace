# 中文结论

- 做了什么：新增 v11 no-template SOXX core + ML regime SOXL satellite；最终仓位不读取、不缩放、不复用 template position。
- 模型来源：使用 XGB/LGBM/avg_xgb_lgb 预测 SOXL 机会、SOXX 环境和 tail risk；核心 SOXX 仓由趋势/波动/VIX risk gate 控制。
- 基准：SOXX open-to-open return=413.6565%，maxDD=-43.0919%。
- v11 nested selected：return=43.3092%，maxDD=-13.7952%，excess=-370.3473%。
- 结论：去掉 template 后没有通过目标；它证明当前 alpha 不够，不是实盘候选。
- 失败原因：为了把回撤压到 10% 附近，SOXX core 参与率必须大幅降低；但这样会系统性错过 SOXX 牛市上涨。SOXL satellite 的 ML 信号不足以补回这部分收益。
- 是否晋升：不晋升；decision=not_promoted_no_template_regime_satellite_failed_target。
- qlib recorder：e8c75a35d57648668cd5f5b95007f526
- 不构成买卖建议。
