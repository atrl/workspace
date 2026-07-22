# 中文结论

## 跑了什么

用 qlib SOXL/SOXX/SOXS 日线 provider 做两层审计。第一层在 2011–2022 开发期比较固定机械规则与持有至少 3 日、最多 10% SOXS 的事后 oracle；第二层对相对收益、现金优势、未来波动与五日尾部事件做 2016–2022 逐年 walk-forward。2023–2026 只做诊断，不参与任何选择。

## 关键结果

10bp 开发期通过全部主门槛的机械规则数为 0；不可交易 oracle 为 2（不含 SOXS 1，允许最多 10% SOXS 1，因此 SOXS 没有扩大通过数量）。机械规则中上行捕获最高的是 `fixed_soxl_soxx_0.40`，为 59.65%，但 MaxDD -72.51%、最差五日 -35.37%。基础特征稳定目标为 `['cash_advantage_1d', 'future_vol_5d']`；加入 SOXS 后为 `['cash_advantage_1d', 'cash_advantage_5d', 'future_vol_5d']`；SOXS 独有稳定目标为 `['cash_advantage_5d']`。

## 决定、风险与下一步

`needs_iteration_feasible_oracle_and_some_stable_targets_but_no_tradable_rule_pass`。oracle 使用未来收益，只回答约束是否存在可行域，不能当作策略；机械规则通过也不等于可晋升。现金按 0 收益处理，SOXS 是每日重置产品，且 2023–2026 已被观察。只有开发期存在可行域且标签稳定，下一步才值得设计单一、冻结的 allocation model，并进入新的 sealed-forward paper window。

不构成买卖建议。
