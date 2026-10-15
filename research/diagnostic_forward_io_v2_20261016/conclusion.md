# 前向 IO v2 离线实现结论

新身份已实现采集、原公式特征构造、封存模型提案、目标意图、后续报价、12 个原股现金/应收账本、持久日志及终端 Qlib 校账的有限流程。原三模块、旧计划、冻结模型及历史失败结论未修改；本轮未启动真实 pilot。

唯一 focused fixture 最终通过：真实已保存 Yahoo SPY 响应 3226 行通过严格 parser；其余 HTTP 结构为既有 Qlib 值构造的明确 synthetic fixture（含 synthetic high/low 包络、SOXS synthetic volume），不是对未来实际供应商响应的认证。14 个 synthetic 日线请求通过新解析与隔离 provider，原公式得到 39+16 个特征，9 次 mock prediction（实际模型调用0），12账本完成9个slot和5个价格事件，12份原生校账通过。真实网络/模型/fit/前向天数全部0；真实SEC/Futu transport及完整run_pilot时序尚未现场验证。

价格口径明确改变：main/common/multi-pair 使用同一 Yahoo query2 全历史 quote OHLC×同payload adjclose/close；volume 是未乘价格调整因子的 vendor-reported share-volume，历史split口径依vendor版本，不声称physical原股数。原execution面保留 NasdaqOHLC×Yahoo ratio，读取固定bootstrap prefix（12525行）并严格校验刷新重叠。原prefix385行高低包络微小差异未修复，依据原builder仅消费open/close，资格限定实际用字段；新增payload OHLC和重叠全4字段校验不变。

Futu end标签采用已独立核验的固定已完成历史映射，end保留time_key/start=end−1min；并不证明不可修订finality。每个已注册slot用真实receipt和独立后续quote，日志先封存再报价；后续slot失败保留此前已执行账本。native使用独立自有持仓/现金/应收/费用核对，逻辑event grid只做会计排序，实际receipt/每symbol报价时间保留，不能从该稀疏grid宣称年化风险。

实际预算已削减为14日线+2live-open+1SEC=17HTTP，Futu最多15、allocator最多9、recovery最多9，未用7个旧Nasdaq额度不重分配。禁止自动fallback、重试、分页、订阅及真实交易。注册需要实际code/endpoint/event/bootstrap字节hash和固定session/deadline；当前只有implementation许可。

下一步是独立复核这版完整代码并在最早2026-10-19另行签发精确pilot许可。届时必须取得真实bundle、模型proposal、封存target、laterquote与12native对账，才能取得新的前向区间证据；任何源/事件/时钟失败具体记录并停止资格，不以synthetic验证替代。第一pilot本身仍不能证明稳定超额SOXX。

独立代码复核后的两次窄修复已追加 implementation_repair_1/2，原 full-chain 源码与输出保持原hash：每个 LIVE_OPEN child 按自己的 receipt 验证未来时间，完整依赖和映射证据字节绑定，Qlib先初始化再读源，初始化失败落terminal状态，且HTTP只允许固定Ledger调用。受影响验证复用原12native结果，最后同一case1passed/2.44s，0新真实模型/网络/native重放。修复代码不宣称已经现场执行过真实HTTP/Futu。

最终SEC守卫同时识别8-K/A、10-Q/A、10-K/A，未知修订同样update_pending；仅纯本地守卫验证，未重跑完整链、native或模型。补丁保存sec_amendment_guard_repair，之前所有artifact hash不变。

独立评估接受离线实现与最后修订报告守卫；负责人完成NAS归档并从空目录恢复核验23个对象，哈希全部一致。归档传输属于交付动作，不是研究行情采集。当前没有pilot许可或后台观察进程。
