# H5 日夜差分监督：条件性后续对照建议

登记日：2026-10-18。状态：`conditional_proposal_before_real_price_label_read`。
这是来源资格通过后才可能进入独评的唯一监督机制笔记，**不是 fit GO，也不是最终冻结训练计划**。
来源资格仍优先；本笔记不改 [source-only 草案](../day_night_label_source_20261018/source_label_design_draft.json)，
不授权价格读取、真实标签构造、模型实现、训练或账户回放。
本次作者仅复核静态源码和已核验文献，未读取真实 price、label、prediction 对象。

## 唯一假说与价格语义

保持原 Stage1 的输入、交易映射和主目标，在同一个共享表示上增加一个连续监督任务。
B 学同一 H5 的总相对回报 S；C 学隔夜与日内相对分量之差 Z。
假说仅是：端点总回报可能掩盖互相抵消的分量，差分监督可能约束原共享表示。
它不增加外生信息，不直接预测或切换一个新交易时段，也没有改善长期现金或胜过 SOXX 的实证。

沿用来源草案的唯一端点：d0=T+1，d1…d5 为随后五个原生共同 session，d5=T+6。
对 i∈{SOXL, SOXX}：

```text
D_i = sum(j=0..4) [log C(i,dj) - log O(i,dj)]
N_i = sum(j=0..4) [log O(i,dj+1) - log C(i,dj)]
Drel = D_SOXL - D_SOXX
Nrel = N_SOXL - N_SOXX
S = Drel + Nrel
Z = Nrel - Drel
```

这里是 **distribution-adjusted research-price plane**：同日共同调整因子 f_t 在日内 D 中抵消，
相邻日的 f_(t+1)/f_t 进入隔夜 N。不能称为 raw-price-only，也不能等同真实分红现金流及费用账户。
来源资格须先证明 O/C 共同基础与内容身份；旧 stitch 比例接近 1 不是该证明。
若新 S/Z 使用与原目标不同的冻结 vintage，必须显式保留该身份；原 H1/H5 数组完全不动。
S 也不是原 daily-reset 27/63/10 参考组合 alpha 的精确复制。

本方案不分别训练 D、N 再将预测相加。相同 X、正则和样本权重的线性估计算子 A 满足
`A*D + A*N = A*(D+N)`，这不构成新的总回报预测机制。本方案比较的是 S 与 Z 对共享表示施加的监督。

## 同网络与同预算 B/C

原网络在 [PitAuxiliaryAllocationNet](../../src/workflow/pit_constituent_auxiliary_training_allocation.py#L281)
用 16 个 common 输入形成 8 维 `tail_encoder` 状态；allocation 同时接收原 39 个 primary 输入和该状态。
建议只在此 8 维状态上新增一个 `Linear(8, 1)` 回归头。B/C 均新增 9 参数，均为 982 参数，
原 973 参数及 allocation、tail、alpha 头保持原结构。容量匹配只指 B 与 C，不声称相对原 A 没有增加容量。
新头只用于训练；其预测不进入持仓、风险状态、阈值或执行订单。

原 [DomainHeadAllocator](../../src/model/stage1_domain_heads.py#L10) 是克隆主/辅助资产 BCE 输出映射；
本建议不采用该分头，也不继承 PG、detach、fullpath/STE 等干预。
新回归任务只用主训练 cohort，不给 20741 个原辅助资产样本另造 S/Z。

两臂先按原 seed 初始化原网络，再以不推进 RNG 状态的方式加入零 weight、零 bias 新头。
需在后续合成合同中确认 B/C 全部初始 tensors 相同、原初始输出相同、原 RNG 路径保留。
这项笔记不运行该合同。

原 39/16 输入、scaler、3014 主训练行、20741 辅助训练行、seeds 71/172/273、300 epochs、
原辅助抽样序列及固定终态均保留。若以后另获完整批准，拟合预算唯一为 3 seeds × 2 臂 = 6 fits；
不选 seed、epoch、窗口、方向或模型。当前没有任何训练授权。

## 唯一尺度与损失建议

对于各臂的目标 Y（B 为 S，C 为 Z），仅在同一原 3014 个成熟训练行上，用 float64 计算
`mu_Y = mean(Y)`、`sigma_Y = std(Y, ddof=1)`，再将 `(Y-mu_Y)/sigma_Y` 一次转换至原训练 dtype。
两个目标使用同一算法，各自的常数跨三 seeds 复用并记录；不使用预测期或评价期数据。
这是匹配监督尺度，不把目标的方差大小作为两臂的隐含损失权重。
不截尾、不 winsorize、不翻转符号、不增加 epsilon 救援；标准差为零、非有限或标准化结果非有限即停止该监督版本资格。

```text
L_B = L_original + 0.02 * 0.5 * mean((regression_head - standardized_S)^2)
L_C = L_original + 0.02 * 0.5 * mean((regression_head - standardized_Z)^2)
```

均值覆盖全部相同主训练行，不另加样本权重。
0.02 借用原 `primary_tail_loss_weight` 的数值，是一项需在看到标签前审定的单一设计选择；
外部论文没有推荐该权重，更没有证明它适合本账户。此处不登记其他系数或损失备选。
原 allocation loss、两个 primary BCE、原 auxiliary weighted BCE 及其系数不变。
原 [fit](../../src/model/minute_pit_allocator.py#L161) 的 AdamW、原抽样、一次 combined backward 和
一次全参数 `globalclip(2)` 保留，不改为分开裁剪或引入梯度投影。

**直接回传路径不等于完整更新影响范围。** 新回归损失的计算图接到新头与共享 encoder，
没有直接使用 allocation 私有层。零初始化新头使第一步该损失对 encoder 的直接梯度为 0，
但新增 head 的梯度仍参与原全局裁剪范数；若范数触发裁剪，系数可影响包括 allocation 私有参数在内的所有更新。
之后共享表示的变化也会改变原目标的前向与梯度，AdamW 状态进一步沿训练路径演化。
因此不能声称干预只影响 shared 表示、私有梯度在整个训练中严格不变，或 allocation/optimizer 方向受保护。
保留原一次全局裁剪；后续完整 B/C 对照评估的是整个监督包，包括它与裁剪、优化器和原账户控制器的相互作用。

## H1/H5 与成熟时序不变

[MinutePitAllocator.fit](../../src/model/minute_pit_allocator.py#L154) 的 allocation returns 是 H1，
由 [原 loader](../../src/workflow/soxs_defensive_hurdle.py#L134) 的
`open.shift(-2)/open.shift(-1)-1` 构造，覆盖 T+1 开盘至 T+2 开盘。
tail/cash target 与 relative alpha 是 H5，后者还使用原 27/63/10 参考组合定义。
所有原目标仍由 [fit 的共同成熟筛选](../../src/model/minute_pit_allocator.py#L127) 接收，
不因为 allocation 是 H1 就放宽 H5 成熟钟或扩大 cohort。

新 S/Z 必须覆盖同一原 3014 行，且不能早于原 d5 09:31 America/New_York 的 H5 成熟约定；
原 cutoff=2023-01-01、train_end=2022-12-30 及 signal 末日 2022-12-21 保留。
原成熟钟是历史研究可用性约定，不额外认证实际首次接收时间。
新标签缺失不能通过删行、缩短窗口、换成下一可得开盘或挑年份解决。
全部未来 O/C 只可作为成熟后的训练标签，不成为当次 09:29 输入。
日级 open 也不是原 09:47 账户的实际成交价；本方案不改变该映射或声称解决执行标签误差。

## 证据边界与可证伪结果

[Lou/Polk/Skouras, A tug of war](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf)
与 [Hendershott/Livdan/Rösch, Asset pricing: A tale of night and day](https://faculty.haas.berkeley.edu/hender/CAPMday-night.pdf)
支持日夜分量异质性的研究动机，不支持本神经网络损失或 SOXL/SOXX H5 的有效性。
前者主开价使用首半小时 VWAP，不能回填为本项目 09:29 可见输入；股票截面证据也不等于 ETF 时序可预测性。
社区发现路径、论文限制和已关闭路线见 [原 triage](overnight_intraday_task_triage.md)。

新头只接收 16 common 输入，现有信息未必能够识别杠杆 ETF 的相对 Z；这一限制不靠改接收层救援。
即使 Z 可预测，其与 S 的抵消关系、共享训练干扰或原内生风险状态仍可能使完整账户无改善，甚至增加现金。
因此没有现金必降、negative transfer 已证实、尾风险改善或 SOXX 超额提升的主张。

来源/标签资格失败则不训练该源版本。若后续获独评和 owner GO，完整方案仍须事前冻结精确对象身份、
上述尺度/损失/初始化、有限预算及 [三资产研究合同](../../docs/three_asset_research_contract.md) 的全部经济门。
B/C 必须在相同成本、独立 own-state 下形成完整账户，并与 SOXX、原 Stage1 及强参考比较；
不能以训练 MSE、分类诊断、现金下降或仅胜 matched B 宣称成功，也不能用中间结果选择是否展示账户。
费用后超额、MaxDD/CVaR、完整年度/126 滚动、配对区间和独立前向要求不因本监督任务而降低。
失败关闭固定版本，不调权重、head 位置、裁剪、seed 或控制器来挽救它。

本次交付仅为本 Markdown 笔记；没有真实价格/标签读取、特征生成、fit、账户或新市场 HTTP。
