# 2026-11-06 活动研究入口与代码精简收据

## 已核对的工作区

实施前只有主工作树 `main`；工作区已有 605 项修改、362 项未跟踪。
`research/three_asset_campaign.json` 的 11 月增补属于既有未提交工作，
已在改写前逐字保存。此次不重置、不清理这些无关 WIP。

## 活动上下文

| 活动文件 | 原字节数 | 新字节数 | 原字节快照 SHA-256 |
|---|---:|---:|---|
| `AGENTS.md` | 73,435 | 8,589 | `bf68c78ed2a42fb1ee78587885a7b1c7e87dd5efefa32ed58ebaccc8b7071c2c` |
| `research/three_asset_campaign.json` | 271,688 | 2,874 | `19afbe4e8662edec459c180c36b165d4066236d5c2dbd5b77814ff8b8f65f288` |
| `research/active_research_index.json` | 60,233 | 1,057 | `a170a2ec7222b7c7ace8b62ca983dc617c6531af7b727afdd7b28d2672077353` |

原字节分别在 `docs/archive/AGENTS_pre_compaction_20261106.md`、
`research/campaign_history/three_asset_campaign_pre_compaction_20261106.json` 和
`research/campaign_history/active_research_index_pre_compaction_20261106.json`。
来源主 checkout HEAD 为 `ad9b94854fabca45cd247beba5b4db6f10a2061e`；
campaign 快照还包含该提交之后的既有未提交增补。原研究卡与 Recorder 未重写。
默认上下文减少约 393 KB；历史快照仍在仓库，因此这项主要减少模型读取量。

## 删除与保留

删除无生产调用方且活动 campaign 缺字段时已报错的旧监控链：
`.codex/agents/research_monitor.toml`、`docs/lean_runtime_interfaces.md`、
`src/core/research_monitor.py`、`src/core/research_runtime.py` 和
`tests/test_research_runtime.py`。其原字节可从上述来源提交恢复。
短任务由任务所有者按真实进程句柄和阶段结束收据观察，不再常驻模型巡检。

对已关闭的 `research/cushion_target_tiered_20261013`，删除本地
`qlib_recorder` 的 46 个文件、228,665,691 字节。该固定版本已失败；
NAS 完整归档 SHA 为
`25cacfe72ff08b875be6d6bbb69e5312c56877c8e27778cd921bfc29a0ec8950`，
此前 34 个选定对象限定读回通过。卡片、notebook、摘要和后续计划的
归档身份仍在；新增 `retention.json` 保留逐文件大小/SHA、少量元数据及
恢复路径。再次读取 Qlib 对象前须从 NAS 恢复，不能把摘要当本地 Recorder。
删除前再次通过 SSH 核了远端归档的完整 SHA。

对已关闭的 `research/minute_net_action_value_20261013`，同样在逐卡
引用、工作区、归档 SHA 与 44 个原生/模型对象读回核验后，删除本地
`qlib_recorder` 的 57 个文件、77,397,602 字节；`retention.json`
保存逐文件 SHA、少量元数据和 NAS 恢复位置。两卡合计实际释放
306,063,293 字节本地明细。

研究目录约 17 GB；最大的三个目录是尚未提交的 2026-11-04/05
own-state、continuous-policy 和 H8 Recorder（约 1.7/1.1/1.0 GB）。
它们不能因固定模型失败就整体删除：部分承载新方法、来源和限定读回证据。
旧 `ibkr_tiered_minute_diagnosis_20261013` 约 728 MB，仍是 Tiered
会计/结构诊断参考，也保留。删除前统计的 Recorder 中 416 个零字节文件主要是
`code_cached.txt`；删它们几乎不省空间，却会改变原生记录的文件清单。
其余约 300 MB 的 `code_diff/code_status/code_cached` 快照分散在 Recorder，
需要逐卡核对引用、关闭状态和 NAS 恢复，不能按文件名批删。
大于 50 MB 的旧卡筛查中，另有 NAS 归档的 IBKR Tiered 诊断、
Stage1 分钟信息比较、完整路径 Tiered 父本和分钟 PIT 配置器仍被
后续计划、工作流或测试作为实质对照引用，保留本地对象；
无 NAS 对应读回的旧大卡也保留。

后续删除候选须逐卡满足：固定结果已关闭、保留结论/摘要/notebook、没有
生产或冻结路径引用、不是既有 WIP、原字节和 SHA 有恢复路径，并确认
NAS 收据与限定读回或明确可接受的本地证据缩减。未满足者保持原位，
不把“已归档”当作“已清理”。
