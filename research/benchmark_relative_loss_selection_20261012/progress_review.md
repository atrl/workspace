# 最近进展复核（2026-10-12）

研究证据比此前更可信：时点、标签成熟、费用独立路径、Qlib原生现金/数量对账与失败封存均已有明确证据。但当前没有同时达到扣费后SOXX稳定超额与回撤门的策略。

- 层归因：冻结原始配置在开发期相对SOXX财富1.2006/1.0747，在复用诊断期降为0.8144/0.7758（10/25bp）；原始预测的泛化不足。风险投影减少回撤，也减少上涨暴露，不能靠移除风险门宣称解决。
- H20：55次拟合、8个原生场景后，组合退回SOXX；预测MSE未超越各自prior，路线关闭。
- VXN/VIX：100次拟合、12个原生场景。复用诊断门通过，开发稳定门与匹配控制组信息/组合增量未通过；SOXL未被使用，路线关闭。
- 原始季度指引：110次拟合、12个原生场景。原始文本/保守发布时点资格得到改善，超额预测和净组合增量未通过；累计正超额集中于少数年份，全部场景SOXL数量为零，路线关闭。
- 活跃SIM仍是2026-09-28 paper trial身份，开发期MaxDD/CVaR比较门失败。当前raw observed与settled日期混有旧回放，不能当作独立合格前向。独立paper有近期事件记录；完整认证资格仍是needs_data，原始计数见progress_snapshot.json。

本轮有限优化：只改变选模型的评价口径，比较平均两腿归一化MSE与H5收益差MSE。内层模型完全复用，只有选择改变的年份允许一次外层成熟样本重拟合；费用、期限、风险与候选集合固定。若失败封存，不继续重搜同一路线。研究入口是本卡notebook，预测、模型、组合明细与原生报告归Recorder。

父结论：../stage1_uniform_layer_attribution_20261010/conclusion.md、../h20_matched_forecast_execution_20261010/conclusion.md、../vxn_vix_information_increment_20261011/conclusion.md、../nvda_original_guidance_information_20261011/conclusion.md。进展来自本次文件与运行状态复核，旧聊天不作为持久状态。
