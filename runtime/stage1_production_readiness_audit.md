# Stage1 上线就绪度更正（2026-10-09）

**当前不能晋升。** 本文替换此前“只差 0.11pp”“只是 Recorder 治理问题”的评估与行动建议。原稿保留在 Git 历史中；此前用短窗口 PASS 推翻长窗口失败、据已查看窗口调整尾部规则或门槛的建议均撤回。

## 当前运行版本与真实失败

当前身份来自 `production/stage1_futu_sim_v1/runtime_release.json`，由
`src.workflow.stage1_runtime_release.load_runtime_release` 校验来源、对象哈希和试运行授权：

- release：`stage1_hold_net_fees_20260928_paper_trial_v1`
- allocation Recorder：`f696f656b3d44e868588e09a5d97c905`
- action Recorder：`b472a458fc9042b4905094fddeaebd33`
- `research_promotion_status=not_promoted`；SIMULATE 试运行授权不等于研究晋升。

现役 action 开发期有两个真实失败门：

| 指标 | 现役 action | 自身基线 | 判定 |
|---|---:|---:|---|
| MaxDD | −30.083% | −29.931% | 劣于基线 |
| CVaR5 | −3.539% | −3.485% | 劣于基线 |

`development_gate_pass=false`。不能用旧生产摘要、别的候选或别的窗口替代现役版本的风险证据。

## 旧生产摘要与短窗诊断不能互相替代

旧生产卡 `summary.json` 引用的 `f6b09330...` / `e9c97a32...` 不是当前 release。
其 2023-01-04 至 2026-08-07 的收益和 worst5d −12.11% 是未独立复现的旧摘要，不能称为当前可用优势。

可原生读取的旧 Recorder `454e94c9...` / `76a6fe6a...` 覆盖
2023-01-03 至 2026-06-26、873 日。它们各自窗口的四项风险指标通过，只能作为旧窗口诊断；
身份、窗口、成本与完整日历没有完成同口径匹配，不能确认或推翻旧生产失败。
缺失的原 Recorder 不得通过新 replay 冒充恢复。

权威更正卡：`research/stage1_gate_recheck_20261010/`，
`decision=legacy_diagnostic_no_recheck_verdict`、`production_gate_override=false`。

## 三类前向记录分开报告

1. **现役 SIM**：`completed_slots` 原始日期与账本 settled marks 均不等于合格前向。
   污染时段、换版本、决策时序、完整结算、基准配对及来源身份尚须核验，资格为 `needs_data`。
2. **旧谱系**：缺失的历史 Recorder 与重复查看的 outer 窗口不能补充前向样本。
3. **独立 regime paper**：四本 candidate/baseline × 10/25bp 账本是另一套冻结候选，
   不能合并到现役 SIM 的前向证据。2026-10-09 开盘结算后，每本有 6 条已结算记录，
   这仍只是观察计数，不是资格认证。

独立 regime paper 的既有观察计划要求至少 60 个完整配对交易日、每类区间至少 15 日、
至少 3 次区间转换，并验证数据质量与双成本下的绩效、风险和换手。
该计划属于对应的冻结候选，不自动授权现役 SIM。原审计中的 13/17 天数字不得称为合格前向。

权威更正卡：`research/stage1_history_audit_20261010/`。
本卡为只读诊断，无签发成功资格的分支。

## 后续工作与停机条件

- 保持现有封存的独立 paper 观察，按其 release 身份核验事件、结算与配对基准。
  它的验证结果只属于自身候选，不能据此解除另一个现役版本的风险门。
- 现役 SIM 的 MaxDD/CVaR 失败继续保留。下一轮机制改进必须先冻结假设、有限对照、
  成本与时间边界，验证自身基线的改进；既有 outer 只能作为重复使用的历史诊断。
- 任何候选均须完成对应的研究、风险和独立前向门才能重新评估晋升。
  缺失来源、时序不明、样本不足或实际失败，分别报告 `needs_data` 或 `not_promoted`。
- 夜盘 `night_session_signal_20261010` 已查实跨越开盘的小时 bar 泄漏。
  原高 IC 与原生对账声明不构成有效策略证据；旧工作流已隔离。
  如继续该方向，须另行核验细粒度可用时点、完整会话、往返费用与同账户 Qlib 对账，
  才能登记新的研究；不能继续对原两年样本调参来恢复原资格。

本轮修复审计与证据完整性，不改变现役配置，不签发晋升，也不证明新的 alpha。
