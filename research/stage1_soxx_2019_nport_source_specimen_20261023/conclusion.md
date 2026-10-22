# SOXX 2019 NPORT-P 原 XML 样本：身份与字段合格

唯一冻结的 2019-09-30 SOXX NPORT-P `primary_doc.xml` 返回 HTTP 200，原件 **38,831 字节**，SHA-256 `951d162d4e996b74a7e1818e13c4c118c348320a95e54a72a4158a20c88d0e73`。原始 XML 与 HTTP 头已归档 NAS，归档 SHA-256 `4a5fc73692fcf3b3d43171a2dc156f98c2bb753c53d5ab0e07595ea193503308`，远端哈希和原字节定点恢复通过。

XML 内 `cik=0001100663`、`seriesId=S000004354`、`classId=C000012084`，**`repPdDate=2019-09-30`** 与 SEC filing index 的报告期一致；另有 `repPdEnd=2020-03-31` 和 `dateSigned=2019-10-29`，两者都不当作公开可用时间。34 个 `invstOrSec` 条目包含证券名、CUSIP、股数、市值 `valUSD`、净资产占比 `pctVal` 和资产类别。初版结构审查漏识 `repPdDate` 字段，失败边界和零网络修正另存，未重新请求原件。

决定：`one_original_NPORT_XML_identity_report_date_and_holdings_schema_qualified`。这只证明一个 2019 来源样本的字段语义与身份，尚未计算持仓集中度或验证其余年份。1 个 XML GET、0 市场收益、0 拟合、0 账户。后续数值提取必须遵循 SEC [N-PORT 字段说明](https://www.sec.gov/files/nport_readme.pdf)中 `PERCENTAGE` 相对**基金净资产**的口径，不能直接把它当普通股内部权重。
