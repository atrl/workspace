# Quant-Arena 方法迁移：Stage1 冻结模型诊断结论

- 执行：核对三个固定 Qlib Recorder、944 个原生日历会话；新拟合 0、新账户 0；诊断 Recorder `0bf6c7fa718f4db6b05c161bf38fe58b`。
- 0bp（仍扣 IBKR Tiered 费用）账户：Stage1 财富 3.2969，SOXX 4.8059，相对财富 0.6860。
- 年度：2023: relative wealth 0.8104, complete=False；2024: relative wealth 1.0098, complete=True；2025: relative wealth 0.8624, complete=True；2026: relative wealth 0.9720, complete=False。126日窗口 819 个，正超额比例 31.1%。
- 信号到决策目标平均 L1 距离 0.1855；源风险权重均值 59.2%，决策目标 51.8%，退出现金 48.2%。这些不能解读为可消除损失。
- 缺口：无独立逐日 replay 对象，不能在本卡重建七项精确归因；成交后即时权重未知，退出持仓不可替代。
- 决策：`diagnostic_only_no_model_change_no_promotion`。论文未提供与当前三资产同任务的新信息；现役 SIM 不变。风险：复用历史、来源 09:47 与 SIM 09:31 时钟不同、未来数据首次可用性仍需另证。下一步是只对新增且 PIT 合格的特征冻结信息门，再做完整 0bp 账户；本卡不是新候选或晋升证据。
