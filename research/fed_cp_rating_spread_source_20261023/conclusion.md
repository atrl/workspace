# Fed商业票据等级利差：两次定额传输均未取得原文

拟议信号是联储公开的30日非金融A2/P2与AA票据利率差，不同于已失败的SOFR−IORB和HYG−LQD。联储[CP说明](https://www.federalreserve.gov/releases/cp/about.htm)描述同日、同期限利率来源与通常次日发布，但不保证历史每次实际钟点；[FRED A2/P2](https://fred.stlouisfed.org/series/RIFSPPNA2P2D30NB/)和[FRED AA](https://fred.stlouisfed.org/series/RIFSPPNAAD30NB)是当前回溯入口，不是原始发布版本。

冻结的两次默认传输均以curl 92结束，0 HTTP响应/报头/正文；其空响应字节及收据已归档NAS并限定恢复。另冻的同URL HTTP/1.1窄修复各请求一次，均以curl 28、约30秒无字节结束。修复的NAS归档PUT也以curl 28超时，随后只读SSH路径校验失败；这四份本地空响应保留，**未证明**修复归档落盘。两次修复都在来源解析前停止，累计4次请求、0条数值、0 Qlib映射、0拟合、0账户。

决定：`needs_original_Fed_CP_transport_after_two_bounded_protocols_no_fit`。这是本机传输和归档状态，不是票据利差信息门失败，也不能据FRED网页的今天数值回填2011–2022。固定传输版本关闭，不继续改端点、协议或日期救援；需独立授权的新有限来源路径或外部网络状态变化。SOXX总Goal继续走其他不依赖该来源的支线。
