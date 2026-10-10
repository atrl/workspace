# TSMC YoY 加速度 5-session 信息初筛（0 model，中文结论）
- 固定机制：`TSMC reported YoY acceleration (current yoy - previous consecutive yoy), fixed positive expected sign`；固定 horizon=5 session；固定正方向；无 threshold/year/horizon/sign 搜索。
- primary stat=`signed_time_series_spearman`（时序 Spearman(score, spread5d)，非两资产横截面 IC）；secondary 仅 SOXL/SOXX label IC/Pearson。
- 决定：`end_fixed_monthly_information_route_no_parameter_search`；gate：{'scope': 'development', 'min_valid_n': 40, 'pass': False, 'valid_n': 84, 'primary_ic': -0.09476945515335523, 'ci95_lower': -0.3267771733589285}。
### 分窗口
- development：valid_n=84；primaryIC=-0.09476945515335523；CI=[-0.3267771733589285,0.11720173354532991]；secondary SOXL/SOXX IC=-0.10028754842884441/-0.10351740485981879；unknown={}。
- outer_reused_diagnostic：valid_n=45；primaryIC=0.08999571806636376；CI=[-0.18806876633289557,0.3870118414702069]；secondary SOXL/SOXX IC=0.08887571279028161/0.09388279520100172；unknown={}。
- 源/价格 SHA：source 329811ed6fa9 / price 48ec582f748d（frozen_verified=True）。
- 边界：raw association 初筛；未证明超越 price/vol、无经济 alpha、不可上线；CI 跨 0 表示证据不足，而非证明不存在信息；后续仍需 priced-vol control 与 10/25bp economic/native 及独立前向。
- 本 fixed route 因证据不足而停止；next action=end_fixed_monthly_information_route_no_parameter_search（不调 sign/horizon/threshold）。
- outer=reused_diagnostic，不作资格/选择；unknown 全保留不填0；每 period 仅 1 独立观察。
- 子 Recorder：62851a0ab548440aa4961627c8b897b4。
