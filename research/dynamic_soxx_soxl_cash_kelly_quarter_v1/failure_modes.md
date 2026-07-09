# Failure Modes

- kelly_estimator_noise: rolling Kelly uses recent realized source-template returns; noisy edge estimates can change exposure after the regime has already moved.
- source_template_dependency: this card sizes the v2 LGB/XGB-confirmed regime template artifact; it does not retrain an independent signal.
- cap_dependency: the max Kelly scale is a fixed risk-control cap; without the cap, daily Kelly fractions can be unstable.
- benchmark_scope: current evidence is versus SOXX open-to-open and the v10 fixed-risk-budget reference, not versus live execution.
- selected_return=349.0079%, selected_maxDD=-8.6907%, SOXX_return=413.6565%, v10_fixed_return=458.1614%.
