# 固定四日交易日历代理：来源草案

目前只写来源/知识时钟草案和notebook骨架，本卡未导入官方原字节、未生成特征或读取收益，0拟合、0账户，没有新的绩效结果，不晋升。

唯一候选窗口为前月预计最后交易日与本月已确认开市前三日。它不是机构现金截止、养老金流量或DTC/银行结算机制。历史规则/公告来源原12份加单独获准的最多5份补证，累计最多17份；无法证明已知性的区间保留unknown，不用今天的日历补早年，也不把2026-10-09数据尾端当月末。

重要未决：原训练字段在09:29，而Stage1研究决策在09:46；当日开市在09:46被确认的信息不能配给已从09:30开始的原训练收益。当前只报告两时钟知识覆盖，不改标签、offset或接收模型。

来源完整性与各年/月known覆盖仍待source-only执行和独评。下一步先冻结官方inventory与有限请求合同，再取得明确源执行GO。

入口：[source_plan_draft.json](source_plan_draft.json)、[source_document_inventory.json](source_document_inventory.json)。

同轮root单独授权的12份文档发现已由external代理完成一次：8个HTML200、4个SEC PDF403，0重试，原文cache仍在临时目录，等待inventory hash绑定。原12请求和inventory保持原样；root另授权external追加最多5个新原文URL/5次显式HTTP补固定缺年，0retry/redirect，禁止重抓旧403。累计上限17；后续source qualification仍只可复用成功cache，新HTTP为0。Nasdaq例外公告不自动认证NYSE/Arca同步闭市；原文缺口和venue缺口均保留unknown。

补证入口：[source_document_extension_1.json](source_document_extension_1.json)（已完成独立receipt；SHA450bada2…29b1dc）。web发现查询与显式原文HTTP分别记账；是否补齐全部knowledge覆盖尚未确定，不按找到的文档数量宣称合格。当前草案仍未frozen，来源native/特征/模型/账户执行未授权。

补证结果已绑定：exec39737 exit0，5个新URL中4份ICE年历原文成功、1份BusinessWire超时；累计17次显式HTTP、12份成功raw、4个SEC403和1个timeout，0retry/redirect。2014–2020及2023–2025相关普通年历覆盖得到补充，但不得据此宣称全历史资格；2011–2013、SEC原文、NYSE/Arca临时闭市及vintage仍有缺口。发现但未抓取的2021-12-27新年历不纳入raw corpus。所有来源native/ledger/feature/fit/account仍为0，后续只可在独评source-onlyplan+ownerGO后cache-only资格检查。
