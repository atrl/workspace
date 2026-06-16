## Conclusion

Decision: `official_replay_metric_aligned_checkpoint_bc_near_miss_no_promotion`.

No candidate is promotable. Aggregated official replay candidates: `23`; full official two-window gate candidates: `0`; near-miss candidates: `2`; final-retention-ready candidates: `0`.

Selected diagnostic reference: `short_volshock_official_replay_screen_v1::short_volshock_cap05_hit2` from `short_volshock_action_cap_overlay`. It is selected because it is the lowest-loss official replay near-miss, not because it can be retained. Its fold metrics have min relative return `29.817pp` and max MaxDD `15.107%`; missing final required windows: `wf_2020`.

The next model step should use the selected short-volshock near-miss as a metric-aligned checkpoint/BC teacher reference, add `wf_2020` coverage and strict current-YTD retention checks, and keep actor retention blocked until the final gate is proven. Dashboard interaction work remains a separate front-end track; it should consume the same freshness and model-difference metadata instead of implying that baseline is an updated actor.
