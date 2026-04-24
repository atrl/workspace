# hierarchical online budget generalization

This diagnostic keeps the fast `imitation_v4` actor fixed and changes only the deployable runtime policy shape.

- `actor_runtime`: current single-actor budget-usage runtime with the existing mode router.
- `hier_rule_daily`: online hierarchical composer with rule mode refresh every day.
- `hier_rule_slow5`: same composer, but slow mode refresh every 5 trading days.
- `hier_learned_daily` / `hier_learned_slow5`: same structure using the learned mode classifier.

| ETF | actor proxy | rule daily proxy | rule slow5 proxy | learned daily proxy | learned slow5 proxy | best |
| --- | --- | --- | --- | --- | --- | --- |
| SOXL | 178.2% / 0.969 / 22.6% / pos 0.187 | 120.6% / 0.822 / 22.6% / pos 0.175 | 123.3% / 0.877 / 22.8% / pos 0.164 | 415.2% / 1.082 / 29.1% / pos 0.277 | 195.3% / 0.741 / 36.6% / pos 0.282 | hier_learned_daily |
| TQQQ | 56.4% / 0.521 / 23.4% / pos 0.274 | 58.7% / 0.587 / 18.3% / pos 0.251 | 74.7% / 0.688 / 18.0% / pos 0.244 | 168.6% / 0.902 / 30.9% / pos 0.353 | 214.4% / 0.990 / 25.9% / pos 0.364 | hier_learned_slow5 |
| UPRO | 39.0% / 0.484 / 22.4% / pos 0.300 | 50.6% / 0.630 / 17.9% / pos 0.279 | 67.0% / 0.737 / 17.6% / pos 0.283 | 112.2% / 0.868 / 23.7% / pos 0.378 | 131.9% / 0.929 / 18.3% / pos 0.387 | hier_learned_slow5 |
| SPXL | 24.2% / 0.339 / 23.3% / pos 0.301 | 38.1% / 0.507 / 18.4% / pos 0.279 | 68.8% / 0.758 / 13.8% / pos 0.276 | 102.1% / 0.822 / 23.3% / pos 0.376 | 134.3% / 0.952 / 17.2% / pos 0.384 | hier_learned_slow5 |
| TECL | 51.4% / 0.476 / 25.5% / pos 0.247 | 47.6% / 0.492 / 21.0% / pos 0.223 | 78.9% / 0.699 / 23.0% / pos 0.220 | 152.5% / 0.772 / 35.2% / pos 0.352 | 201.1% / 0.896 / 31.4% / pos 0.354 | hier_learned_slow5 |

## Interpretation

A deployable improvement here must beat `actor_runtime` on proxy participation without giving back the 2022 drawdown compression. If the hierarchical variants only reshuffle risk while staying low-return, the missing capability is still the fast actor's participation policy, not the router cadence.
