# 结论

## 跑了什么

复用 `trainable_regime_policy_nested_wf_v1` 的 nested outer 目标仓位，但丢弃旧 `strategy_return`，重新使用 qlib SOXL/SOXX 开盘价按 close(t) 信号、next open 到 following open 收益计算。固定测试非对称财富锁盈和 beta-preserving SOXL/SOXX 梯形配置；outer 2023–2026 不参与控制器参数选择。

## 核心结果

10bp 主方案 `smart_leverage_asymmetric_lock_beta_ladder`：总收益 419.57%，CAGR 61.00%，MaxDD -52.50%，单日最差 -14.19%，CVaR5 -5.48%，滚动五日最差 -24.31%，SOXL 平均仓位 14.22%，上涨捕获 36.48%。正式全门槛通过数为 0。

## 增量效果

相对干净重算的 `nested_opportunity_clean_soxl`，主方案收益变化 -207.91 个百分点，MaxDD 改善 +9.78 个百分点，单日最差改善 +1.46 个百分点，CVaR5 改善 +1.71 个百分点，上涨捕获变化 -11.23 个百分点。

## 决定

`research_reject_smart_leverage_controller_failed_required_gates`。这是论文启发的诊断实现，不是原论文忠实复刻；当前 qlib store 缺少 PIT T-bill/T-bond，债券状态为 `needs_data_pit_tbill_tbond_not_in_current_qlib_store`。此外该架构是在看过既有 outer 结果后提出，任何看似通过的结果仍需方案冻结后的未来 paper-sim 确认。

## 风险与下一步

SOXX 当前成分股特征仍有 survivorship 风险；本卡只复用来源模型的目标仓位，不把旧盘中回放收益作为证据。下一步只在结果显示尾部改善且参与度未坍塌时，才值得补 PIT 债券并训练路径级 LFNN；否则停止该方向。

不构成买卖建议。
