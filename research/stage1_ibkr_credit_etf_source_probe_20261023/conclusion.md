# IBKR HYG 长历史来源限定探针

连通的 IBKR 只读合同搜索确认美股 ARCA `HYG` 的精确合同 ID `43652089`。按冻结计划仅请求一次 `ONE_DAY`、4,000 根日线、常规时段并包含公司行动，连接器返回 **`Step count more than 1000 is not allowed`**，价格柱 **0 条**。该连接器的本接口没有历史结束日期参数；不能靠这一请求取得覆盖 2011 年 Stage1 起点的 HYG 序列。按停止规则没有把请求改成 1,000 根后重试，也没有请求 LQD。

完整结构化连接器响应原字节 SHA-256 `4b5c3cc5736e400ea7fbb3d21b2c4cdc1ad4902026eb57358ad0dcfacbe14672` 保存于忽略 Git 的来源快照；Yanxuan NAS 归档 SHA-256 `8ab80397505f25daf0781d88000215542535a7e359415e9bca43a4f45973a98a` 已远端校验并限定恢复。先前 Futu HYG 2011 单日也返回零行；两次限定探针都不能证明其他提供商没有历史数据。

本卡 1 次只读历史请求、0 SOXX 结果读取、0 拟合、0 原生账户、0 券商写操作。决定：`connected_IBKR_HYG_daily_history_4000_bar_request_rejected_max1000_no_2011_source`。HYG/LQD 信用风险偏好机制仍需可分页到 2011 年且有历史可用时点/分红处理的独立来源；不能用近 1,000 日回填训练期或先跑 0bp 账户。
