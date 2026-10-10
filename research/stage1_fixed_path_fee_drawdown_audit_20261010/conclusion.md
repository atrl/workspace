# 固定路径费用-回撤审计（Step A，诊断；0 model/0 strategy/0 orders）
- 相同实际目标/flag路径，controller 不重跑；funding=`pre_fee_account`；阈值 0.22（旧 controller 假设）。
- native：passed 12/12（inherited 4/4，recomputed 8/8）；plan sha `955ade829af4`。
- 源 native 复核（冻结输入 source_qa.json，sha `62321d200537`）：父 recorder `6c6fac4f2d974545a4ca8ed8b149a17c`，20 checks，all_passed=True，max ending-weight error 7.258868006676522e-09。
### fixed0 对照源 actual（回答 stop_rule）
- development_walk_forward::10：源 actual 484 日，fixed0 324 日，交集 324 日（占 actual 0.6694214876033058），仅 fixed0 0，仅 actual 160。
- development_walk_forward::25：源 actual 469 日，fixed0 150 日，交集 150 日（占 actual 0.31982942430703626），仅 fixed0 0，仅 actual 319。
- outer_reused_diagnostic::10：源 actual 47 日，fixed0 46 日，交集 46 日（占 actual 0.9787234042553191），仅 fixed0 0，仅 actual 1。
- outer_reused_diagnostic::25：源 actual 446 日，fixed0 304 日，交集 304 日（占 actual 0.6816143497757847），仅 fixed0 0，仅 actual 142。
### 各 case（actual mask = 源 emergency_active；hypothetical 按 settlement cost）
- development_walk_forward::10::0（recomputed_settlement，native=True）：trueMaxDD -0.2908；actual 484，hypothetical 324，交集 324（占比 0.6694214876033058）；base DD_before 与源吻合=False。
- development_walk_forward::10::10（source_inherited，native=True）：trueMaxDD -0.2993；actual 484，hypothetical 484，交集 484（占比 1.0）；base DD_before 与源吻合=True。
- development_walk_forward::10::25（recomputed_settlement，native=True）：trueMaxDD -0.3242；actual 484，hypothetical 593，交集 484（占比 1.0）；base DD_before 与源吻合=False。
- development_walk_forward::25::0（recomputed_settlement，native=True）：trueMaxDD -0.2565；actual 469，hypothetical 150，交集 150（占比 0.31982942430703626）；base DD_before 与源吻合=False。
- development_walk_forward::25::10（recomputed_settlement，native=True）：trueMaxDD -0.2751；actual 469，hypothetical 253，交集 253（占比 0.5394456289978679）；base DD_before 与源吻合=False。
- development_walk_forward::25::25（source_inherited，native=True）：trueMaxDD -0.3021；actual 469，hypothetical 469，交集 469（占比 1.0）；base DD_before 与源吻合=True。
- outer_reused_diagnostic::10::0（recomputed_settlement，native=True）：trueMaxDD -0.2666；actual 47，hypothetical 46，交集 46（占比 0.9787234042553191）；base DD_before 与源吻合=False。
- outer_reused_diagnostic::10::10（source_inherited，native=True）：trueMaxDD -0.2692；actual 47，hypothetical 47，交集 47（占比 1.0）；base DD_before 与源吻合=True。
- outer_reused_diagnostic::10::25（recomputed_settlement，native=True）：trueMaxDD -0.2732；actual 47，hypothetical 53，交集 47（占比 1.0）；base DD_before 与源吻合=False。
- outer_reused_diagnostic::25::0（recomputed_settlement，native=True）：trueMaxDD -0.3212；actual 446，hypothetical 304，交集 304（占比 0.6816143497757847）；base DD_before 与源吻合=False。
- outer_reused_diagnostic::25::10（recomputed_settlement，native=True）：trueMaxDD -0.3388；actual 446，hypothetical 333，交集 333（占比 0.7466367713004485）；base DD_before 与源吻合=False。
- outer_reused_diagnostic::25::25（source_inherited，native=True）：trueMaxDD -0.3644；actual 446，hypothetical 446，交集 446（占比 1.0）；base DD_before 与源吻合=True。
- 机制：费用只扣一次；10bp→25bp 是 15bp 步长，其余为净 NAV→drawdown_before→risk 的状态反馈。
- actual 与 hypothetical 不混用；不作可加 DD 因果分解，不唯一归因市场与费差。
- 限制：外窗为复用历史；不晋升（notpromoted）；Step B 未授权；下一机制须另预登记与独立验证。
