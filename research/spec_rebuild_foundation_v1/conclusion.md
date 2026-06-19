# 中文结论

- 做了什么：使用 qlib calendar 和 canonical SOXL panel 跑通 `src/` 基础分析切片。
- 核心指标：生成 window、regime、model comparison 三组 sidecar，top model 为 mean_return_smoke。
- regime 差异：产物按 VIX、趋势、回撤、波动率、成交量冲击和牛熊/震荡切片，供后续模型稳定性分析复用。
- 是否晋升：不晋升；这是框架烟测卡，不是生产策略。
- 风险：样本和模型都只是基础验证，不能作为交易信号。
- 下一步：接入真实 QuantBench model registry、objective promotion score 和 production paper/sim notebook。
