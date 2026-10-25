# 低成本持续研究运行规则

2026-10-20用户确认：P2与整理并行，最低模型成本优先，复用现有Qlib，
历史最小证据归档后精确清理。科学目标以研究合同v2为准。

## 模型与任务包

项目配置默认负责人Sol/medium、实施Luna/medium、监控Luna/low、独评Sol/high。
复杂问题经具体证据升级，最终科学争议才按需使用Astra；不默认所有任务ultra。
配置只影响支持该配置的后续调用，现有线程/UI覆盖不能称已改变。实际spawn应明确
model和reasoning，记录返回代理名及请求参数；若服务不返回实际模型则标未核验。

新代理默认独立上下文（工具支持时fork_turns=none），只传目标、冻结身份、范围、
接口、验收和返回格式。任务包目标≤8k token，工具输出默认≤2k，回报≤800字。
同一时刻最多3个子代理：两个实施槽和一个监控/评审槽；共享文件由一名代理负责。
两次定向修复未解决才升级；关键会计、时序或数据泄漏问题可立即升级。

## 监控与消耗

`python -m src.core.research_monitor`只观察已有PID/启动身份、退出收据和进度。
程序按60秒采样；状态变化才输出，廉价代理做异常摘要。没有事件时不唤醒强模型。
工具只能轮询时沿原句柄低频观察，记录实际开销；静默或心跳过期不触发重启。

`src.core.research_runtime.aggregate_usage`读取明确指定线程的usage记录，不读提示词。
输入、缓存输入、输出分列，推理输出是输出子项；不得累加每条累计thread计数。
模型/角色/阶段仅在事件或派发收据确有依据时归属，未知不按时间邻近猜测。
当前Goal统计和rollout计数不等价，API标价估算也不是账户实际账单。
首次实施不设总Goal硬token预算；对同类任务比较非缓存输入、输出、调用数、
返工次数与记录到的模型价格情景，科学门槛不随降耗而降低。

## 一次冻结、有限评审

每个科学阶段保留一次设计评审、一次受影响实现检查、一次结果评估。
未变的来源/实现不重复审查或冷恢复；交付不重训。失败固定实验只做一次必要归因。
源读取、拟合、账户等实际执行前记录owner授权和有限预算；用户已授权的范围内
由研究负责人落实，不反复向用户索要许可。科学结果不能由执行者自签。

## 活动入口与归档

当前入口为`research/three_asset_campaign.json`，旧大状态文件移入campaign_history。
当前phase从阶段收据投影，Goal、代理、进程、科学状态分别显示；不再维护两份
可写的current_task。旧状态仅供追溯，不参与调度。

旧SOP和合同原字节在docs/archive，闭合实验通过索引退出活动搜索。删除前核查
运行入口、import、notebook、冻结清单及pickle路径；有引用的旧代码不重签hash。
未证实可删时保留并说明。禁止批量git clean/reset或删除652项原WIP。

## Goal客户端边界

Goal属于客户端线程。研究文件不能代替客户端active；恢复/更新后须get_goal读回。
当前Stage1历史研究Goal的完成口径与逐轮选择规则见
[`stage1_historical_goal.md`](stage1_historical_goal.md)。历史通过只能记录
`historical_research_passed`，独立前向及生产晋升属于后续合同门槛，不等待
新窗口才结束此历史Goal。
当前工具若不支持恢复，记录客户端限制，继续已明确授权的工作，不伪造状态、
不清除历史或创建替代Goal来绕过限制。阶段完成与科学目标达成分别报告。

官方依据：[子代理](https://learn.chatgpt.com/docs/agent-configuration/subagents)、
[Goal](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)。
