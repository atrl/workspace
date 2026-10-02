# 当前目标表达能力与可复用实现（只读）

当前代码 `src/model/benchmark_anchor_utility.py:99–117` 计算

`target(x,h)=q(x)+alpha_H(x)*h`，

其中 `q=alpha_L*e_L+alpha_C*e_C+(1-alpha_L-alpha_C-alpha_H)*e_S`。输入隐层只依赖标准化市场特征，持仓仅进入最后的 HOLD 专家；三个非负 theta 行的全局总和不超过1。

固定一个市场状态 x，若 `alpha_H<1`，精确不交易的条件只有 `h=q/(1-alpha_H)`。这是持仓单纯形中的一个固定点，不能表示同一市场下围绕目标的一片持仓区域都维持不动。`alpha_H`接近1可以放慢调整，但一般仍产生非零订单；因此每日交易并不直接等于费用训练代码失效。一个特定历史序列也可能恰好走过固定点，这不构成不交易区域的表达能力。

由于基函数在[0,1]，`alpha_j(x)<=sum_k(theta_jk)`。若 `alpha_H=1`，HOLD行就要消耗全部全局预算，SOXL和CASH行必为0。按有限参数下严格介于0和1之间的sigmoid实数定义，HOLD的全部系数须在常数基上，成为处处全HOLD；浮点sigmoid饱和不能恢复已经为0的其他两行。由相同SOXX持仓出发，要在状态x1全仓SOXL、另一个状态x2全现金，分别要求两个行预算至少1，总和至少2，违反总预算≤1。因此不只是优化没走到某些组合，它们本来就在此模型类之外。

这些是结构性事实，不证明更大表达能力会改善真实净超额。全局预算原先提供了有限参数下的精确SOXX和简单可验证合法权重；任何后续表达改变仍需保留这些验收性质，并继续完整费用与原有风险门。

## 仓库已有积累

- `src/model/integrated_allocation.py:126–150` 的 `IntegratedAllocationPolicy` 已将真实盘前3资产持仓与市场特征拼接输入网络，`rollout_policy`使用自己费用后、收益标记后的状态；测试中已有状态时序、费用资金与独立集合持仓验证。可以复用状态契约及核对，而不是重新造会计。该版本输出softmax，有限logits不能达到精确边界或自动提供HOLD区域；Phase1 `stable_excess_integration_20261002` 历史门失败，且基于后来发现有价格口径问题的旧源，不能把原高收益当作有效alpha证据。
- `src/workflow/pit_constituent_auxiliary_training_allocation.py:973` 的 `_holdings_aware_turnover` 已修复“目标差值代替真实订单”的问题。旧卡 `holdings_aware_pit_allocation_v1` Recorder77a5c2bb26f34fe0bf6e19d45f15a2a5，在2016–2022、10bp选择诊断候选，交易602/1762天、MaxDD−57.69%，未晋升。此处“持仓感知”主要指换手会计，并不等于网络读实际持仓。它仍采用EWM及rebalance_threshold（同文件1312–1338），以及soft_trade_event惩罚（872行）；不能把减少交易证明成无阈值的学得HOLD区域，也不能直接复用旧的近似扣费 `gross - cost*turnover` 取代本轮精确扣费。
- `src/model/benchmark_anchor_utility.py:120–136` 已有非负L1球投影及其测试，可复用投影数学/可行性验收，但它当前投影的是全局参数，不是状态相关的动作单纯形激活。
- 已对当前工作树及 `git grep HEAD -- src tests` 检查 `sparsemax|entmax`，未找到现成实现。名称含`sparse_execution`的 `src/workflow/soxl_compounding_regime_sparse_execution.py` 是固定2%/5% no-trade band、最短持有等旧规则实验（42–43、850–867行），不是稀疏概率映射；其结论明确提示固定band/dwell会丢弃有效日频目标变化。

本次只做结构证明与复用盘点，没有实现替代模型、交易带、阈值或新增拟合。费用诊断也显示近期保存持仓在扣费前已经落后；增强HOLD表达能力不能被预先当成充分修复。
