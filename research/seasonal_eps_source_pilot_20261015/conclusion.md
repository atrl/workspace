# 原始公告季节性EPS输入原型：拆股检查通过，尚不可训练

本轮0拟合、0回测、未查看事件后收益。区别于earnings_enhancement_20261005的供应商EPS/分析师一致预期快照，本方向拟核查同财政季度的季节性盈利预测误差。它不是市场一致预期，也不能称实际超预期或已验证alpha。[相关研究定义](https://www.nber.org/system/files/working_papers/w30860/revisions/w30860.rev1.pdf)使用拆股调整后的同比EPS变化及过去八季度变化的标准差；该论文结果不等于SOXX/SOXL策略有效。

固定核对2021年四拆一和2024年十拆一两处边界，选择各自拆股后Q2报告及上一年同季度报告，共四份官方历史公告的当前HTML副本：FY2021Q2、FY2022Q2、FY2024Q2、FY2025Q2。四份发行人网页均HTTP200并保存原始字节/哈希；本地SEC直接读取样例403未得到原件，不把该错误页视为财务证据。没有绕过登录或修改全局网络策略。

解析限定GAAP diluted EPS，核对财政季度标题、GAAP表与正文数值；不混Non-GAAP或基本EPS，拒绝不相邻的年度同季度配对。原显示小数精度保留，Qlib仅登记事件参考字段，training_input_qualified始终为0，不作向日频交易特征的前向填充。

| 对比 | 上年原报EPS | 当期EPS | 未调拆股的差 | 拆股后同比差 | 当期表内可比差 |
|---|---:|---:|---:|---:|---:|
| FY2021Q2→FY2022Q2，4拆1 | 0.99 | 0.94 | -0.05 | +0.6925 | +0.69 |
| FY2024Q2→FY2025Q2，10拆1 | 2.48 | 0.67 | -1.81 | +0.422 | +0.42 |

拆股倍数和完成日期来自对应当期公告正文；换算后的上年数值与当期表内上年数值的±半显示单位区间相交，说明观察差异与显示舍入相容，不证明不存在任何其他重述。表中换算小数是算术派生值，不能当作发行人报告了更高测量精度。若忽略股本口径，两例均会把增长错写为下降。

来源：[FY2021Q2](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2021)、[FY2022Q2](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2022)、[FY2024Q2](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2024)、[FY2025Q2](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2025/)。

决策needs_full_corpus_and_vintage_clock。当前页面副本不等于历史首版；日期不等于精确公开时刻，旧SEC API/index时钟分歧亦未由此解决。仅四份公告不足以构建连续同比变化或八个既往变化的尺度；若严格排除当前变化，需至少13个连续季度的原值/比较关系、已知拆股处理与口径审计。尚未计算SUE、跨公司排名或ETF合成，也没有宣称单一NVDA代表完整历史SOXX成分。

下一步先做完整季度原件/口径和时点资格，再冻结唯一指标与对照；不能把已有供应商失败路线换名字重跑，不能拿当期重新列示的上年数值冒充此前已知预期。任何收益检验仍须时间隔离和独立证据。生产不变，稳定净SOXX超额未完成。

1项聚焦合同测试通过；实际四公告解析、原生Qlib读回及NAS冷恢复逐对象哈希通过。原始HTML与精确字符串事实在Recorder，瘦卡只保留事件摘要与两组口径检查。
