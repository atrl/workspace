# Failure Modes

- long_short_time_series_momentum: SOXL/SOXX trend signals could not create a MaxDD<20% path; the best return setting stayed below the production baseline and still breached drawdown.
- crisis_alpha_overlay: adding a negative crisis sleeve did not transfer diversified-futures crisis-alpha behavior to single-name leveraged semiconductor ETF exposure.
- data_boundary: this card uses full production-window qlib SOXL/SOXX/VIX daily data through 2026-06-09 and therefore closes the branch on the comparable window.
- next_action: move away from single-asset trend-following overlays; prioritize new orthogonal data such as longer option-risk history, capital-flow history, or model-family changes.
