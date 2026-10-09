# Stage1 feasible-action 训练域对照（中文结论）
- 做了什么：同一统一 vintage 冻结 coherent 输入上，1 个预声明候选（feasible 域）与 all-state 对照；超参、label fee、minrows、硬修复规则全不变；不调参、不搜阈值。
- plan sha `78f053b70475`；控制复现（filter=all == 封存 control）：True。
- legacy identity parity（original frame + same legacy models/spec vs 源 17，逐日）：passed。若一致，则 uniform 与源 17 的 active/no_hold 差异来自**输入 vintage 敏感性**（legacy state feedback），不能笼统说“价格影响近零”：economic control 在同价路径变化小只是其特性，不等于所有策略路径不敏感。
- baseline 身份：`no_hold_reference`=禁用开盘 HOLD 动作层、始终执行合法上游目标；`active_hold_baseline`=legacy rules-only；`cash_reference`=显式全现金（**不是** no_hold）。
- 前提校正：实证 feasible 比例 0.8610，非法行 label 方差更低、极端 top-decile 仅约 5% —— “非法样本主导极值”的强假设**不成立**；本卡是预声明合法性对照，**不能**把过滤说成已知解决噪声。
### 结果（按 segment × cost 分组）
- control_all_state development_walk_forward @10bp：200.46% / MaxDD -29.03% / CVaR -3.06%
- control_all_state development_walk_forward @25bp：160.83% / MaxDD -30.18% / CVaR -3.08%
- control_all_state outer_reused_diagnostic @10bp：362.93% / MaxDD -26.48% / CVaR -4.77%
- control_all_state outer_reused_diagnostic @25bp：324.74% / MaxDD -26.85% / CVaR -4.69%
- candidate_feasible development_walk_forward @10bp：213.48% / MaxDD -29.79% / CVaR -3.35%
- candidate_feasible development_walk_forward @25bp：300.49% / MaxDD -29.48% / CVaR -3.51%
- candidate_feasible outer_reused_diagnostic @10bp：163.17% / MaxDD -26.26% / CVaR -4.45%
- candidate_feasible outer_reused_diagnostic @25bp：256.31% / MaxDD -26.65% / CVaR -4.67%
- active_hold_baseline development_walk_forward @10bp：114.59% / MaxDD -30.08% / CVaR -3.49%
- no_hold_reference development_walk_forward @10bp：118.19% / MaxDD -29.93% / CVaR -3.47%
- cash_reference development_walk_forward @10bp：0.00% / MaxDD 0.00% / CVaR 0.00%
- active_hold_baseline development_walk_forward @25bp：96.72% / MaxDD -30.09% / CVaR -3.59%
- no_hold_reference development_walk_forward @25bp：102.23% / MaxDD -30.13% / CVaR -3.58%
- cash_reference development_walk_forward @25bp：0.00% / MaxDD 0.00% / CVaR 0.00%
- active_hold_baseline outer_reused_diagnostic @10bp：219.49% / MaxDD -26.92% / CVaR -4.80%
- no_hold_reference outer_reused_diagnostic @10bp：220.43% / MaxDD -26.92% / CVaR -4.81%
- cash_reference outer_reused_diagnostic @10bp：0.00% / MaxDD 0.00% / CVaR 0.00%
- active_hold_baseline outer_reused_diagnostic @25bp：75.14% / MaxDD -32.38% / CVaR -4.42%
- no_hold_reference outer_reused_diagnostic @25bp：66.12% / MaxDD -36.44% / CVaR -4.40%
- cash_reference outer_reused_diagnostic @25bp：0.00% / MaxDD 0.00% / CVaR 0.00%
### SOXX buy-hold（真实成本，无终值假设卖出）
- development_walk_forward：gross 320.42% / net10 320.00% / net25 319.37%
- outer_reused_diagnostic：gross 398.59% / net10 398.09% / net25 397.34%
### 50:50 实际再平衡费用对照
- development_walk_forward @10bp：net 581.38%
- development_walk_forward @25bp：net 555.26%
- outer_reused_diagnostic @10bp：net 1017.13%
- outer_reused_diagnostic @25bp：net 988.73%
- native：status=`passed`，boundary=`2018-12-31`；target convention：{'control_all_state': 'post_fee_desired', 'candidate_feasible': 'post_fee_desired', 'active_hold_baseline': 'pre_fee_account', 'no_hold_reference': 'pre_fee_account', 'cash_reference': 'trivial_never_trades'}；commission/1：{10: 0.001, 25: 0.0025}。control/candidate 默认 postFee 完整对账；active/no_hold 为 pre_fee_account，待根接口 `target_convention` 就绪后完整复跑，现标 blocked 并保留精确原因（非不可验证）。
- 报告纪律：未超过 gross SOXX 不宣称稳定超额；失败项如实保留。
- 子 Recorder：e4d888a42c914bf69660ab78a35ba6e7；保存全部 annual models/scaler/legality flag/maturity/predictions/labels/native reports/positions。
