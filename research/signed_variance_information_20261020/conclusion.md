# SOXX 负半方差份额风险信息筛查

状态：待 root GO；尚无本卡实际数值、模型拟合或新 Recorder。

唯一 notebook 调用 `src.workflow.signed_variance_information.run(root=ROOT)`。先封存版本、精确来源和有界 XNYS 日历；使用完整 RTH 一分钟变动构造 RV 与负半方差份额。2019–2022 共同合格行拟合固定 B/C OLS，训练残差估计 smearing；模型与精确 22 日尾部关闭读回后才读取 2023+ 数值。held 任何目标或滞后缺失立即 needs_data，不压缩日期或执行统计检验。

唯一门槛为 q 系数、held 配对 QLIKE 改善均值及固定区块下界均大于零。全部 held 年度均须报告。复用历史的风险信息通过仍不是独立验证、费用后 SOXX 超额或生产晋升；失败版本闭合，不搜索符号、窗口、频率或标签。实际结果和下一步在唯一授权执行后同步。
