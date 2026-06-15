# Model Retention Frontier

- Rows scanned: `170`.
- Completed handoffs: `133`.
- Recommended next source: `release16_constrained_checkpoint_repair_screen_v1`.
- Recommended next action: `train_reward_constrained_short_training_candidate_with_relative_floor_and_drawdown_gap_loss_then_official_three_fold_replay`.
- Primary blocker: `target_revision_ready`.
- Retention gate: MaxDD `< 20.0`, WF relative return `>= 30.0` pp, and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
