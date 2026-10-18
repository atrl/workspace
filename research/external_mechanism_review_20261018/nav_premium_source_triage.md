# Issuer NAV / exchange close 折溢价：既有来源路径去重

2026-10-18。裁决：**`existing_source_path_excluded_do_not_reopen`**；若有新增来源证据，最低缺口为
`needs_matched_market_price_basis_and_publication_clock`。没有登记新模型、特征或账户。
本轮按“先检查已关闭机制”分支结束，外部搜索为 **0 组、0 查询**，没有网页或市场数据集下载。
当前日夜监督实验及其尚未审定账户分数不参与本裁决。

## 相同点与现有证据

[campaign 的 issuer_NAV_basis 排除项](../three_asset_campaign.json#L2435)已明确：现有 issuer workbook
缺少同期 market price，不得拼入不相关 close 来制造 NAV basis。
[冻结来源清单](../stage1_finra_short_mix_20261018/source_inventory.json#L98)保留同一被拒来源：
`soxx/issuer_fund_download.bin`，SHA256
`1932baf722d6170cee585fe9cc918ee60ca70c591d0ff40ee7010e3020f4e1af`。
已登记的 Historical 表有 8928 行，仅 `As Of / NAV per Share / Ex-Dividends / Shares Outstanding`，
没有同期 market price 或 premium/discount 字段。本轮没有重读该 workbook。

该缺口恰好阻断“issuer NAV 对 exchange close 折溢价”的分子/分母匹配；给它换名为 tracking error、
NAV dislocation，或取两只 ETF 的折溢价差，都不会补齐缺失字段与时钟。
这不是一次完整 premium 策略已经失败的证明，也不能把旧份额流实验的结论当成折溢价检验。

[既有 flow 来源质量侧记](../etf_flow_information_20261013/source_quality.json)与
[其冻结计划](../etf_flow_information_20261013/plan.json)登记的 SOXX 表覆盖 2001-07-10 至 2026-10-08，
包含休市日；这不是合格 premium 序列覆盖。历史 publication、first receipt、无后续修订均未认证。
旧 flow 使用“两原生 session 后 09:31”的诊断可用性假设；它既不是公布证据，也不能直接改称当日 09:29 已知。
该清单没有建立 SOXL 匹配 NAV/市价历史，故不能把 SOXX 表的长度推广为双 ETF 资格。
源质量侧记中的既有拆股样例数值未用于本次推断或计算。

已封存材料的发行人入口为 [iShares SOXX](https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf)
与 [Direxion 半导体杠杆 ETF](https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs)。
这两个地址只是旧来源计划中的官方定位，本轮未重新访问，**不声称当前下载字段、历史范围或公开访问条件已经改善**。
由于已有相同排除项，本轮没有另找社区回测或论文来替代缺失的来源证明。

## 只有以下新增证据才构成不同来源条件

1. **实际字段与配对口径。** 两只 ETF 各自具备同一 session、同一每股单位和币种的 issuer NAV，
   以及定义明确的市场价格或官方 premium。须分清收盘拍卖价、最后成交价、BBO 中点及官方网页的
   “Market Price”；不能默认这些都等于 exchange close。新文件需有可冻结字节身份、字段说明和实际覆盖，
   无需先假定同一 vendor，但跨源必须证明同口径，不能仅凭日期相同相除。
2. **估值时点与可用时点分开。** NAV valuation time、价格取样时点、对外公布时点、下载/接收时点及修订规则
   分别记录。欲用于下一次 09:29，两个值都须有支持其在该时刻前可用的证据；只写 `As Of` 日期不足。
   当代下载或当前网页不能回填历史已知性。若只能支持 retrospective/conditional 研究，必须保留该限制，
   不认证历史 first vintage；是否准入该有界来源版本仍需另外审查。
3. **分红与拆股基础。** 同期 NAV 和价格须处在相同实际份额平面，明确 ex-date、拆股生效时点及历史重列方式。
   不得将未来分红/拆股调整后的 total-return close 除以当时 per-share NAV，或把来源特有历史归一化误认成溢价。
   公司行动表存在本身不证明两个字段已匹配；当前 day/night 来源的共同调整证明也不能自动移植至 issuer NAV。
4. **有界公开获取。** 先以元数据、字段说明及明确历史下载入口证明无需付费、auth 或期权资料，并登记范围缺口。
   本轮没有认证任何新的 SOXL/SOXX 历史下载入口；也不授权试拉全历史、补缺年份或计算 premium。

## 条件性信息与反证边界

若上述测量合格，发行人独立估值相对成交价格的偏离，在定义上并非单纯对现有价格字段做另一种变换，
因此**有可能**提供原 16/39 输入未包含的测量信息。但这不等于统计独立或可预测的净收益；NAV 也可能只是
同一市场收盘信息的另一种表达，表观差异可能来自报价滞后、舍入、价差或基础不一致。

唯一待证伪的经济链条仍是“可见的相对溢价反映暂时价格压力，随后部分收敛”。若价差已在下一开盘前收敛，
或只反映底层资产估值追赶、不可成交报价、不同估值时钟，则不能支持原 09:47 执行账户。
即使存在短时收敛，也没有证据保证它覆盖原 H1 allocation/H5 分类任务，或改善长期现金与 SOXX 净超额。
以后若另获来源资格和正式设计授权，必须在同信息可用性、相同输入容量/训练预算的配对控制下检验固定增量，
并保留原完整费用账户与 SOXX/匹配对照/强参考的所有经济门；本轮不选择模型、符号、阈值或窗口。

无法补齐配对价格、共同基础或开盘前可用时钟，则继续关闭这个来源版本；不通过任意加一日滞后、
拼接现有 adjusted close、换字段名称或借用当前网页来重开。来源合格也只会产生候选输入资格，不会认证 alpha。

本次仅写本笔记，读取既有 campaign、计划、来源清单/质量侧记和静态代码；没有反序列化价格、预测或标签对象，
没有计算收益、生成特征、执行模型/账户，也没有修改当前实验或旧卡。
