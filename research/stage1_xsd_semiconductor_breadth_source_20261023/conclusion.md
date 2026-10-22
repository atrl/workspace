# XSD 与 SOXX 半导体领导结构源预检

State Street 的 XSD 跟踪改良等权半导体指数，2006 年成立；指数提供与 SOXX 不同的行业内部权重视角。本卡只验证历史源覆盖，不定义或筛选特征。

Yahoo 固定历史请求各返回 4,218 个有效日收盘，2010-01-04 至 2026-10-09，日期无重复；XSD 有 67 次分红和 1 次拆分，SOXX 有 65 次分红和 1 次拆分。两份原始响应、头文件与接收时钟已归档至 NAS，并完成远端 SHA-256 与逐文件回读。NAS 归档 SHA-256：`fbcf034d8a1b0b17a32ec2f6ac290e4c0a2605a4efc83371bb0941e308a2c52c`。

下一阶段先核对拆分与分红调整是否在当时可知、映射到冻结 Stage1 的 Qlib 信号日，并与原 39 项中的行业/交叉宽度特征做训练期冗余检查。只有不同且来源时间合格，才冻结单坐标 H5 信息检验；通过后才单独冻结 0bp 原生账户。

本卡 0 拟合、0 账户，没有 SOXX 超额结论。历史调整收盘及公司行动是今日回溯快照，首次公开和修订时钟待核实。

来源：[State Street XSD](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-semiconductor-etf-xsd)，[S&P 半导体精选行业指数](https://www.spglobal.com/spdji/en/indices/equity/sp-semiconductors-select-industry-index/)。
