# Failure Modes

- 5min branch has unstable validation evidence; positive 2026 test is not enough for promotion.
- 15min branch is positive in validation and 2026 test, but monthly returns are concentrated and IC is small.
- Intraday simulation is next-bar open-to-close and uses fixed bps costs; live spread, queue, PDT, and borrow/funding frictions are not proven.
- qlib PortAnaRecord is not used for this high-frequency single-asset execution probe; large predictions and labels are recorder-only artifacts.
