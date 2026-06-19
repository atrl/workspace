# License Review

## 中文结论

- GitHub 来源：https://github.com/quantopian/alphalens，仓库 license 复核为 Apache-2.0。
- 只保存 URL、自有摘要、结构化假设和本仓库运行产物；未复制 README 正文，未 vendoring Alphalens 或第三方实现代码。
- Alphalens 的 returns analysis、information coefficient analysis、turnover analysis、grouped analysis 只作为因子诊断方法假设。
- 本 notebook 只验证本仓库已有 `src.factor.analyze_factor_suite`、`compute_factor_ic`、`compute_factor_correlation` 和分层收益输出契约。
- 后续如要直接依赖 Alphalens runtime，需要单独复核依赖版本、维护状态和 Apache-2.0 notice 要求。
