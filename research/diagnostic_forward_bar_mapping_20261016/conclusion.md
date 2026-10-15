# 固定已保存响应的端点映射结论

固定比较通过：2026-07-29 SOXL/SOXX 各 26 个完整 15 分钟窗口，按 (end−15min,end] 聚合 open/high/low/close/volume，52×5 个值全部精确相等（rtol=0、atol=1e−7）。未内连接删除缺失窗口，未搜索偏移或日期。

2026-10-14 两份已保存 NONE 响应共 780 行，原 time_key 直接作为声明 interval_end，interval_start=end−1min；未额外 +1min。隔离 Qlib 的 390 个日历端点、780 行 OHLCV 与原数组 float32 转换结果逐值相等。SOXL/SOXX 原值至 float32 最大绝对量化分别为 3.8086e−6、3.02734e−5。

最初固定比较已成功，但在 active Recorder 中初始化新 Qlib provider 触发 RecorderInitializationError。获准的一次软件续接在 Recorder 上下文外只使用已保存 payload 完成读回，并追加 native_repair_1 namespace。原 22 个 artifact hash 全部不变，原失败 summary、workflow/source 与 manifest 单独保留。旧 registered_artifact_count=1 是原实现固定值；新 native_repair_registered_artifact_count 记录实际对象数。

证据边界：old K1M 的 QFQ 有原 manifest 证明；direct K15M 只有 path/hash，其 autype、SDK、receipt 未认证。因此这是同日保存价量的经验聚合一致，不是已认证同调整接口，也不是旧 QFQ 与新 NONE 价格等价。固定比较支持声明的已完成历史分钟 end-label 映射，不证明所有日期/vendor 语义、live prefix 可用时间或不可修订 finality。

本次只解除固定已保存 completed-history end-label/Qlib 读回门。原 start-grid 探测失败不回填，未修改原三执行模块；0 网络、0 模型/fit/replay、0 前向天数。daily collector 角色映射、实际未来 receipt→seal→later quote 会话及 pilot 仍未完成，不可据此宣称策略达标或直接启动。

下一步：把已审定 end-label 合同以新代码身份接入前向适配器，明确剩余 daily source-role gate 后再独立审定完整会话激活；不重新采集本轮数据。

独立评估接受本限定结果；负责人从NAS空目录恢复核对29个对象，全部哈希一致。源响应和原失败对象均保持原身份。
