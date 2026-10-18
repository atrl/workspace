# 路径面积动作研究：原结构 OOF 来源盘点

2026-10-18，root 只读小型 summary/manifest、notebook 和静态实现。没有反序列化模型、预测、价格或标签，没有生成 OOF、面积或账户；此盘点不授权拟合。

承接 [路径面积接收层说明](signature_path_readiness.md)。固定面积只进入 09:46 action → 09:47 fill，不能进入 09:29 allocation；同窗完整性 mask 只看早盘 prefix。

## 可以少做的重复工作

原 `minute_information_pit_allocator_20261012` Recorder `68e31a0a0fad44e7ba1d1f7d856c7a0d` 的 manifest 当前 SHA 为 `ac90270cf66164e5fb9bca62b50ab86eeab7733bd0146799c45d97937f94e82a`，NAS archive `ee328cab292f178bf3582ef574fb411561c93b82545cd3bb90842ecd9a9a24f1`。
它登记了以下 9 个 base 年度模型：

| Artifact | 登记的 SHA256 |
|---|---|
| `models/base_2020_71` | `f9bef1bfed5815a632efd33ded5e98cef2cb090b41b670e75786da038a202a40` |
| `models/base_2020_172` | `9e6d354b91a12600a8f0eeaedc541a314cc9e0675bb29ab0330b7061e67065e9` |
| `models/base_2020_273` | `7bf1451562409592c27d4deebde083ddf0816aa1c3ddc56122fec373d79a011a` |
| `models/base_2021_71` | `9a02f1a267462914ca82bf4a3879c7b7e222ea0d862ba2b4146d558afff65d37` |
| `models/base_2021_172` | `65f87112bf60e36236f06d444cc4b0d4f94b0611d5ca67eb820898fa5f58140c` |
| `models/base_2021_273` | `3992909cb15da24a8d7b10dd9380722052a9833b5fe87a629b66cbf4c877c7bc` |
| `models/base_2022_71` | `d226915b5062436a6ef106f70ca054c8ffd9662d2268528dbd6605d6df798b44` |
| `models/base_2022_172` | `dd69fea76a11df330f2f6bee12bd06d1a98593e9a918432ee12955e1403f3728` |
| `models/base_2022_273` | `973923cc39c076b3b4342435d74e236d0900e263084873d42e5b90c0c0f93b8e` |

这证明存在可核对的保存身份，尚未证明对象当前可加载、原训练 cutoff/scaler/auxiliary 状态合格或其预测可直接复用。
源码 `src/workflow/minute_pit_allocator.py` 的年度 base 分支使用 seed+year、20 个原生日历 session purge，截止该年的 1 月 1 日；outer 模型另以 2022-12-30 训练终点保存。
未来来源资格应先检查这些固定对象与其训练 audit，而非重训这 9 个模型。不得用 outer 的 2022 全样本模型反推 2018–2022 年早期决策。

## 必须保留的区别

- `tangent_router_oof_20261015` 用上述 base 的结构/训练边界重新训练的是 tangent allocation head；其 9 个 tangent 模型不等于原 base 模型。
- `tangent_router_labels_20261015` 的 teacher 为 tangent OOF，加训练用 deterministic-risk/always-execute shadow，并关闭两个 learned controllers。已有 260 个成熟 action rows、2022 前 154 个；154 低于原 200 开发门槛，不可降低或把 allocation 的 3014 行冒充 action 样本。
- 该标签 notebook 已有 2bp、Tiered、公共 reset 的局部两分支实现线索；需核具体 09:47 时钟/公司行为/现金账本版本后才可复用代码。它不能无条件替代新 base teacher 的状态、动作资格或标签数，也不能视为完整策略未来价值。
- 新训练 teacher 若使用原 base OOF，既有 tangent 标签不能直接复制。要从相同冻结 shadow 策略重建两臂共用的 post09:47 EXECUTE/HOLD 局部优势，保留原 active Stage1、匹配对照、SOXX、强参考及完整 own-state 成本账户的经济比较。

## 可能需要新增的最小范围，尚未授权

原 manifest 未列 2018、2019 base fold。若独立冻结的 bridge 需要覆盖 raw-minute 来源自 2018-10-12 起的全部训练期，可预登记这两年各三 seed、至多六次补充 OOF 拟合，并复用已资格的 2020–2022 九模型；不要在看支持量/表现后选择起始年或 seed。
“至多六次”只是来源补全候选预算，不是已获 GO，也不保证成熟 action rows 会达到 200。后续 B/C 动作头拟合和财务账户预算必须单列，不能藏在 OOF 名下。

最容易漏掉的前视点在辅助样本：`pit_allocation_retrain._training_auxiliary` 先把原生观察截到每 fold 的截止日，再构建 forward labels；不能从 2022/2026 生成的辅助标签中筛早期索引，并用 `_auxiliary_dataset` 的统一 `observations_end` 字段伪装成早期可用。
该 Dataset wrapper 自身会写传入的 observations_end，这不是独立成熟证明。新早期 fold 需按过去原始观察重建，并保留 PIT-only 过滤和真实可用性；缺 PIT 辅助支持时不能把 `union_backfill_pretrain` 当已知历史成分。空辅助支持的实际处理须在拟合前冻结并审查，不能临时更换来源或权重。

主标签仍是原 H1 分配和 H5 joint maturity，所有 scaler/标签/辅助观察须守各 fold 的过去边界。结构/超参选自已复用研究历史，所以即使逐年拟合时间合格，也不称历史独立验证。

下一步是一个独立的、无性能选择的 bridge 预登记：固定原结构 teacher/模型与数据白名单、逐 fold 截止/成熟条件、09:47 局部分支现金与 reset 规则、source-ready mask、支持量门槛和全部预算；再做实际来源资格。本文不触发模型/账户执行，也不重开任何已关闭的 tangent/旧 opening/标签对齐实验。
