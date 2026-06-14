# release16 state-head post-fullprobe objective screen conclusion

Decision: `release16_state_head_post_fullprobe_objective_screen_no_actor_retention`.

The latest full state-head actor is not retainable. It still fails the fixed-test MaxDD gate, YTD return alignment against both SOXLBacktest and current-best floor, and walk-forward stability gates.

The best diagnostic candidate by current fold evidence is `checkpoint_bc_initialization`. It is useful only as a checkpoint/objective teacher because checkpoint paths are fold-local and have no single retained actor or YTD replay.

YTD underparticipation rows where SOXLBacktest position >=35% and full actor position <=15%: 20. Mean full position is 0.085 vs SOXLBacktest 0.162.

Candidate-screen metrics are recomputed from official daily paths for comparability; see full_probe_metric_basis_reconciliation.csv for the reported full-probe worker-metric deltas.

Next action: build a new checkpoint-distillation / target contract with YTD SOXLBacktest position-alignment loss and a PPO-vs-BC drawdown drift stop, then run a constrained replay before any retained actor training.
