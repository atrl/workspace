# Release16 Failed Fold Daily-Path Diagnostic

- Decision: `release16_failed_folds_require_participation_target_revision_no_promotion`.
- Metric basis: `official_daily_path_rebased_eval_window`; direct comparison source `release16_cb99_rebound_checkpoint_screen_v1`.
- Failed folds remain return-only: True; PPO/checkpoint screen produced no gated candidate on failed folds: True.
- wf_2020: selected `bc_initialization` return 43.06%, rel -15.03pp, MaxDD 14.96%, return gap to +30pp relative 45.03pp.
- wf_2020: SOXLBacktest reference gate-pass count 0; best return reference `legacy_default` return 220.78% / MaxDD 54.28%.
- wf_2020: low-position up-day screen found 74 days; selected prev-position mean 0.159, risk-cap reference prev-position mean 0.306.
- wf_2020: target share >=50% is 24.615%; share >=70% is 0.850%.
- wf_2023_2024: selected `bc_initialization` return 4.35%, rel 4.71pp, MaxDD 5.62%, return gap to +30pp relative 25.29pp.
- wf_2023_2024: SOXLBacktest reference gate-pass count 0; best return reference `wf_pass_ytd18_15_15_02` return 4.51% / MaxDD 1.06%.
- wf_2023_2024: low-position up-day screen found 85 days; selected prev-position mean 0.079, risk-cap reference prev-position mean 0.032.
- wf_2023_2024: target share >=50% is 4.497%; share >=70% is 2.910%.
- Next action: `build_release16_failedfold_participation_target_overlay_then_rerun_checkpoint_screen`; keep actor retention blocked until the revised target reruns all three walk-forward folds and YTD/dashboard gates.

Sidecars are written in this card directory. This diagnostic does not promote or retain any actor.
