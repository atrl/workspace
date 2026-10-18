# 固定二阶 ETF 路径面积：接收层与来源可实施性

2026-10-18。**可进入独立、有界的来源／action-label 桥接设计；尚非 fit-ready，也没有 GO。**
它可以诚实定义为同一时间网格上的已报告分钟成交价路径统计量，无需先证明真实套利偏差。
本轮只读 registry、静态代码、小型来源/支持量 metadata，以及既有三个外部 URL；没有搜索、价格/标签读取、面积计算或拟合。

## 去重与主张边界

[campaign 的 backup_candidate](../three_asset_campaign.json)仍登记该固定二阶面积为待资格备选；本轮未找到已执行的等价实验。
[分钟权重](../minute_value_weight_20261017/conclusion.md)及 [SSL](../minute_self_supervised_20261017/conclusion.md)
失败的是完整日内路径编码后 H5 探针，未验证本固定前 15 分钟面积在开盘 action 层的条件增量。
已闭 opening gap/return15、二者交互和 ROD 尾盘窗口也不包含这个同窗顺序量；不能据此称旧路线成功或重开它们。

只沿用已登记的 [社区线索](https://www.mql5.com/en/articles/23548)、
[Futter/Horvath/Wiese 原论文](https://arxiv.org/abs/2308.15135)及 [作者仓库](https://github.com/owenfutter/signature_trading)。
论文提供将路径依赖纳入策略的框架，不证明这里的 SOXL/SOXX 单一面积、H1 action 或完整费用账户有效。
本候选不是完整 Sig-Trading 复现，不增加阶数、时间通道、滞后通道、窗口或深度模型。
这仍是已复用历史上的新研究假说，不是新独立市场证据。

## 唯一定义与充分匹配的低阶控制

仅用 raw NONE 的 09:30 开盘和随后至 **09:45 已闭合**的 15 个分钟 close。
两资产必须具有相同的完整分钟网格；令 `x_j=P_SOXX,j/P_SOXX,0−1`、`y_j=P_SOXL,j/P_SOXL,0−1`，从 `(0,0)` 出发：

```text
A = 0.5 * Σ(j=1..15) [x_(j-1) * Δy_j − y_(j-1) * Δx_j]
```

固定 SOXX 为 x、SOXL 为 y；不取绝对值、不反号、不除以观察后挑出的波动量。
这是分段线性路径的反对称二阶面积。
数学上，先走 `(a,0)` 再走 `(0,b)` 与反序路径拥有相同端点、端点二次项、增量平方和及协变差，面积却分别为 `ab/2` 与 `−ab/2`。
这证明“顺序可以不同”，不证明真实数据有可用增量。

B/C 共用八个控制：`X,Y,X²,XY,Y²,ΣΔx²,ΣΔxΔy,ΣΔy²`，其中 `X=x_15,Y=y_15`。
再共用一个严格 prefix-ready mask；第十个新增槽，B 为中性 0、C 为 A。
两个模型同形、同预算、同训练期变换；只用训练域拟合尺度，unknown 的原始 A/控制保持 NaN，
输入编码为中性必须配共同 mask，不能解释为“已知面积零”。

理想简单收益关系 `y=a*x` 时面积严格为零，减去 `3*x` 不改变面积，不能借“残差化”制造新量。
若用 log 坐标的确定函数 `y=f(x)`，连续面积又可退化为端点函数并混入离散化效应，故本候选保持上述 simple-return 定义。
不把两 ETF 当作固定 3:1 平价套利；当前基准及历史版本也不能凭名称假定不同。

## 已有来源足以提出该测量，不等于本轮已核完值

| 只读来源身份 | 现有证据与限制 |
|---|---|
| 原 NAS NONE snapshot | `18f1fb3f17907932ddd13881aea9199369181a92c9b6c77395d0b255100c68c3`；原分钟计划起点 `2018-10-12`，as_of `2026-10-09`。 |
| 原生 raw-minute provider | `data/qlib/minute_information_pit_allocator_20261012/core_1min`；manifest SHA `e2d97bd4dc00cdb231f7aa926799efcbc6f0d613ca197fbe04e86db1f074bf4e`，本轮只核 metadata hash；字段含 open/close/price_valid。 |
| 已存原件内容身份 | [full-day source_inventory](../intraday_full_day_expansion_20261012/source_inventory.json)：SOXL NONE parquet SHA `94b2ee6bf16f76d2e3e453f45a8db533869a2fe6303e90c47505a61647d0dc2a`；SOXX SHA `9133d01608d534d753015fa043d1fc538ccd12afbab8aaa1a01daa0f64448561`。未读取这些价格文件。 |

应从该固定 raw NONE provider 通过 Qlib 取值。另一条 full-day `soxl_intraday_1min` 是 published affine research plane，
包含非零加法项的资格限制，不能因同属分钟数据就把 adjusted close 换入本定义。
manifest 匹配也不等于本轮已重新核对全部 bin 内容；未来桥接须绑定实际读取对象/字段/内容 hash，不能换当前 dirty provider。
现有数据原则上不需要新市场 HTTP、付费、期权、报价簿或真实交易数据采购。

**同步网格不是同时成交。** 分钟 close 可对应两只 ETF 不同的最后成交时刻，价差、零成交或价格离散也可能主导 A。
这些是测量解释与稳健性限制：若所有用到的 bar 均已闭合，尚不会自动形成未来信息，亦不要求补 tick/quote 才允许条件研究。
但若源实际用到 09:45 起始、09:46 才结束的 bar，或把以后补来的完整日状态当已知，则是真正时钟违约。

## 接入 09:45 action，不能倒填 09:29 allocation

原 raw-known09:45 → decision09:46 → fill09:47 路径见
[`replay_stage1_raw`](../../src/backtest/stage1_domain_heads_raw.py)。
当前定义对应右标 09:31…09:45 的 15 根闭合 bar，加第一根的 09:30 open；沿用历史 +60 秒可用性约定在 09:46 接收。
这不是实盘 first receipt 认证。不得使用 native09:46 的 close，也不得用 09:45 路径末价充当 09:47 成交价。

新 mask 只能由该 prefix 的存在性、重复/非法值和当时的 source validity 决定。
**禁止复用** [`completed_session_features`](../../src/data/minute_allocation_features.py) 的 full-session `minute_core_ready`：
它检查整个 390/210 分钟日，下午缺失会把未来信息带入早盘 mask。
新 prefix 缺失时保留 unknown，不补价、不删训练或账户日。

建议唯一接收处是开盘 EXECUTE/HOLD action 模型：原 39 primary /16 common /3014 allocation cohort、
三 seed 原 allocator 和目标流均保持并复用；不重训成 09:46 allocation，也不删掉 2011–2018 年原行。
原 action 静态合同有 22 个输入，加入上述 10 槽后 B/C 都为 32 槽；可沿用同一固定 logistic pipeline、
原 C/class weighting/random_state/执行阈值及 mandatory-derisk 守卫，不新增 A×动作方向、量级权重或新控制器。
相同名义参数量不代表 B 的常零槽有相同有效自由度；C/B 只检验预登记面积输入包的增量，不把它描述为纯微观经济因果。

## 关键剩余桥接：标签与 causal OOF 支持

不能直接拿 [`stage1_open_action_value._build_training_frame`](../../src/workflow/stage1_open_action_value.py#L301)
的旧标签给新输入使用：原记录为 09:31 decision、next09:30 退出/next09:31 reset 成熟，另有旧 25bp proxy。
09:45 路径已经覆盖该旧目标的一部分已发生收益；若不明确换到 post09:47 两分支结果，就不能称为本 action 的未来优势标签。
`minute_net_action_value_20261013` 又是 10:31 恢复任务，不能直接代替这张开盘标签。

现成 [tangent_router_labels](../tangent_router_labels_20261015/summary.json) 的 metadata 为：
756 个 reference sessions、**260 个成熟 action rows；2022 年前只有 154，低于旧 minimum 200**。
这正是 campaign 中 `cost_sensitive_action_loss` 备用方向所面对的相同支持量不足。
Recorder `1bf6500a803b4fb8bc20d181ed93ba55`，NAS `00c4d795bfbf293daf9c3a55e8608f97ca78dd4c8360eee9f3ace27d871861e5`；
其 `analysis/labels` SHA `1d4f97e8e132a4ce48fd297a63ae1b3fd47b82089812c5f34633eab74ee669cc`，本轮未反序列化。
而且其 teacher 是 **tangent_linear OOF + deterministic-risk always-execute shadow**，训练时关闭两个 learned controllers；
260 行不是原完整 Stage1 的通用 label 库，不是 3014 个 allocation 样本，也不是全策略反事实收益。

因此，正式实验前的最小工作是单独冻结：

1. 原 allocator **结构/特征/损失不变**的逐年 causal OOF 目标与共同 shadow state 来源；先核哪些既有 Recorder 可精确复用。
   缺失的早期 fold 可另列有限生成预算，所有 scaler、auxiliary observations 和成熟标签严格止于该 fold 的过去。
   **不得用 2022 全样本模型反推早年 teacher**；那会让训练信息进入早年 action 标签/状态。
2. 同一 shadow 状态上的 EXECUTE/HOLD 两分支，使用真实已存 09:47 entry→next09:47、同一 next-target reset、
   当前预登记 Tiered/费用标签合同与成熟钟；冻结它作为局部优势而非整个未来策略价值的含义。
   B/C 使用完全相同标签、同一 action 成熟行集及 source-ready mask，不能按面积挑 action 行。
3. 实际逐年支持量、prefix 分辨率/全零或常数情况、输入与标签身份的训练域资格报告。
   不降低旧 200 门，不把 154 称为开发通过；若需新的逐年 OOF，必须先由 owner/独评给单独有界预算，
   不能自动因旧 OOF 不足永久关闭本机制，也不能承诺补 fold 必会增加足够有效行。

这一桥接只需要已有数据上的源/标签与 OOF 工程，当前没有授权执行。

## 完整反证标准

若采用该唯一规格，拟合的是 B/C 两个固定 action 头，原 allocation 与恢复模型复用；额外 OOF 生成不能隐含算进“仅两 fits”，
必须单列。每臂在 0/2/5bp 形成独立 own-state/数量/现金/风险/费用路径，共六个新原生账户，不能共用一个路径事后扣费用。
旧 Stage1、旧 B_shared、SOXX、strong9097 等参照按精确同成本合同复用；additional27baa 只报告。
全部 SOXX 净超额、匹配 B、强参考、MaxDD/CVaR、完整年度/126 滚动、配对区间和充分独立前向门不放宽。

若机械资格证明面积在固定源上退化为常数，或在事前数值合同下仅有不可辨的舍入残差，应结束该固定版本；不能仅凭可能有微观噪声就先否决。若有变化却不能改善完整经济门，
同样关闭，不搜索阶数、坐标、方向、bar 频率或控制器阈值。真实数据中非零 A、分类更好或现金更低都不是 alpha 证明。
结论是 **可证伪且不需先证套利，但须先做正确 clock/label/OOF 桥接**；本笔记不赋予 source、fit、账户或生产 GO。
