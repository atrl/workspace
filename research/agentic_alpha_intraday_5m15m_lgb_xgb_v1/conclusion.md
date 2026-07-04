# 中文结论

- 做了什么：把 SOXL/SOXX 1m 缓存正式重采样成 qlib 5min/15min provider，并通过 qlib.init、D.features、DatasetH/DataHandlerLP、R.start、LGBMClassifier 和 XGBClassifier 重跑下一根 bar open-to-close 盘中模型。
- 核心指标：validation 选择 15min / xgb，valid return_pct=42.2473、test return_pct=20.4127、test max_drawdown_pct=-7.3140、test IC=0.0032。
- regime/window 差异：5min 与 15min 都使用 purged split（purge_bars=1）和 2024/2025/2026 walk-forward；月度稳定性与成本敏感度见 monthly_stability.csv、cost_sensitivity.csv 和 backtest_walk_forward_metrics.csv。
- 是否晋升：不晋升；15min 有弱正信号但月度集中、IC 很小，5min validation 不稳定，尚不能证明盘中实盘能力。
- 风险：执行假设是 bar 收盘出信号、下一根 bar 开盘进、下一根 bar 收盘出，未证明真实 spread、排队成交、PDT、滑点扩张和实时延迟。
- 下一步：若继续推进，应接入实时分钟 replay 和成交模拟，扩大成本压力，并把 15min xgb/consensus 作为候选而非生产信号。
- qlib recorder：6f57d668e1f44b388a6c45cbb2f3637c
