# Stage1 Futu SOXX 日换手率：原始来源与 Qlib 覆盖门

以只读 Futu OpenAPI `US.SOXX` 日 K 线、明确 `AuType.NONE` 和五页固定预算取得当日已完成换手率。原响应 CSV 与请求收据按哈希备份 NAS，限定恢复逐件一致；随后建成独立原生 Qlib 日频来源。没有新闻数据、未来 SOXX 结果、模型拟合或资金账户。

Stage1 原日历训练 3020/3020 日有值，复用历史评价 946/946 日有值；缺日 []。数值范围 0.00219–0.93093。四项来源门 4/4，决定 `conditional_source_coverage_pass_first_public_vintage_unknown`。

历史每日首次公开数值与供应商修订版本未核证；来源覆盖不证明特征增益或费用后 SOXX 超额。若继续，须另冻原39项受控信息检验，并在完整 Stage1-parent 自身状态0bp账户之前解决时点来源边界。源 NAS SHA `13d2984554cda99416dc5f393efa4cd7e31f487c4988a0336816dbed1aea9bd7`；Qlib Recorder `ff416be4fcf447409a45349287d1e4ca` NAS SHA `3ac89dbb4bc8c8d170462036af8b32eb8dd7f34a518b462497c332ccd653d721`。
