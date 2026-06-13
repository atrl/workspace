# Retraining Gate Gap Handoff V1

- Decision: `run_full_walkforward_cap10_joint_contract_bc_checkpoint_training_with_qlib_buy_sell_reference`.
- Promotion: blocked; this card does not retain or promote a new actor.
- Strict retention scope: 0 replayable model rows plus 1 SOXLBacktest reference row; no replayable model passes complete WF + MaxDD/return alignment.
- SOXLBacktest qlib reference: `legacy_default` keeps the buy/sell timing baseline with YTD `132.077977%`, but worst WF MaxDD is `84.918053%` and is not promotable.
- SOXLBacktest risk-floor probe: `wf_pass_ytd18_15_15_02` passes the <20% WF drawdown screen with MaxDD `18.514585%`, but YTD is only `18.308910%`; use it as a floor reference, not a model.
- Walk-forward weighted diagnostic: `recovery_reentry_conditional_vol_target_v1` is closest by weighted loss, but still fails full gate with max WF MaxDD `26.139320%`; next mechanism is `walk_forward_weighted_multi_objective_training`.
- Best available training start remains cap10 joint-contract BC/checkpoint path. wf_2022 selected MaxDD `18.455088%`, rel `36.419124pp`; wf_2023_2024 selected MaxDD `13.597475%`, rel `30.501116pp`.
- Critical stop rule: do not lengthen PPO when wf_2022 MaxDD drift vs BC exceeds `2pp`; observed PPO-selected drift was `18.820132pp`.
- Dashboard scope: local dashboard/server only; Vercel is excluded. The next retained candidate must export per-model feature columns, feature snapshots, BUY/SELL feature summaries, position, action, and price fields before it can be exposed as runnable in v2.
- Next action: run full walk-forward training with wf_2020 coverage, qlib SOXLBacktest buy/sell reference scoring, and dashboard replay export; keep actor retention blocked until all gates pass.
