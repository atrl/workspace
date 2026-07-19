# SOXL Exhaustion, Structure, and Fragility Implementation Plan

1. Add synthetic contract tests for bounded, one-session-lagged, future-invariant
   path-exhaustion and leverage-fragility factors.
2. Add synthetic tests for the fixed five-name bellwether structure frame and
   ensure the fixed-proxy limitation is explicit.
3. Implement factor builders under `src/factor/` and reuse the existing
   volatility-scaled breakdown label and price-only benchmark.
4. Add workflow tests for pre-2023 selection, conditional incremental lift,
   intraday coverage blocking, and slim-card recorder policy.
5. Implement a qlib-native workflow using `D.features`, `DatasetH`, and
   `qlib.workflow.R` recorder evidence.
6. Execute the research notebook top-to-bottom and synchronize conclusion and
   sidecars from executed results.
7. Independently recompute headline outer metrics, run focused and full tests,
   inspect the diff, commit, and push the branch.
