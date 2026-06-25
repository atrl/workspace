# Failure Modes

- CogAlpha-style screening is implemented as deterministic Codex-compatible candidate evaluation, not a full autonomous multi-generation agent loop.
- Alpha158 and generated factors are selected on train+validation only; 2026 remains a holdout report window.
- Single-asset SOXL has much weaker cross-sectional evidence than the paper datasets, so IC/RankIC are time-series approximations.
- The selected throttle can still trail constrained_weight_reference if extra factors reduce participation during strong trend regimes.
