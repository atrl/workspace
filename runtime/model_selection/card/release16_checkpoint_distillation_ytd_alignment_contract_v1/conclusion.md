# release16_checkpoint_distillation_ytd_alignment_contract_v1

Decision: `release16_checkpoint_distillation_ytd_alignment_contract_no_actor_retention`.

No actor is retained and promotion is blocked. The source train-label contract passes scope audit, but checkpoint candidates still do not pass all walk-forward windows and there is no single retained actor with YTD replay. YTD SOXLBacktest position rows are saved only as a recipe/guard, not direct labels.

Key evidence:
- Distillation target rows: 3731
- Scope audit blocked groups: 0
- YTD mean absolute position gap: 0.247463
- YTD underparticipation rows: 20
- Selected checkpoint-objective diagnostic candidate: `bc_initialization`
- Promotion allowed: `False`

Next action: train a single retained-actor candidate only after adding official daily path objective loss, recipe-only YTD participation alignment, and PPO-vs-BC drift stops; then rerun full fixed, walk-forward, YTD, and dashboard replay gates.
