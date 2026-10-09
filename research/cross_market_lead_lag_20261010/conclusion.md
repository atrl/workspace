# 跨市场领先-滞后研究（中文结论）

- 新数据：`data/raw/external/asia_semis_20261010_v1`，Yahoo Finance 日线收盘（2011-10-10 至 2026-10-08），含台湾加权/台积电/韩国 SK 海力士/KOSPI/日经 与美股 SOXL/SOXX/SMH/NVDA/TSM/AMD/ASML。
- 方法：亚洲盘在美股同一交易所本地日开盘前收盘；只把严格滞后（t-1 信号对 t 美股）当作预测证据，同日相关只作同期共振记录。预测 IC 门槛 |0.03|。
- 亚洲市场 IC：twii(同+0.173,滞后+0.003)；tsmc_tw(同+0.148,滞后-0.012)；skhynix(同+0.181,滞后-0.000)；kospi(同+0.205,滞后-0.008)；nikkei(同+0.160,滞后+0.026)。
- 美股参照 IC：smh(同+0.984,滞后-0.085)；nvda(同+0.754,滞后-0.062)；tsm(同+0.750,滞后-0.062)；amd(同+0.645,滞后-0.041)；asml(同+0.790,滞后-0.057)；soxx(同+0.997,滞后-0.089)。
- 判定为可预测（forecast_textbook）的市场：无。
- development 稳定超额：无。
- 决定：`no_tradeable_cross_market_forecast`；promotion_eligible=False。
- 风险：仅日线收盘、未保留亚洲盘中时刻；同日共振不可交易，不计为预测能力；无收益预测与拟合参数；本结果不构成生产或实盘证明。
- qlib Recorder：1b2cb672588c4bbc9cf276ccfa2a401e；qlib docs：['https://qlib.readthedocs.io/en/latest/component/data.html', 'https://qlib.readthedocs.io/en/latest/component/recorder.html']。
