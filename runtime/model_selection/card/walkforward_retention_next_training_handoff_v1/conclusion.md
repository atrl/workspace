## Conclusion

Decision: `no_retained_walkforward_model_route_to_metric_aligned_checkpoint_bc_screen`.

No final walk-forward model is retained: `final_retained_count=0` and `strict_passing_profile_count=0`. The SOXLBacktest qlib path remains a buy/sell timing reference baseline, not a promotable actor.

Dashboard freshness is split intentionally: latest market data is `2026-06-15`, safe qlib replay end is `2026-06-12`, baseline replay end is `2026-06-15`, actor replay end is `2026-06-12`, and actor latest signal date is `2026-06-15`.

The latest drawdown-drift short-training branch should not be extended as-is: best checkpoint iter is `0.0`, post-init score improved is `False`, and all recorded checkpoints still fail the 20% drawdown target: `True`.

Next action: run `official_replay_metric_aligned_checkpoint_bc_screen_v1` before any new PPO branch, using official daily replay metrics and SOXLBacktest buy/sell timing deltas as the checkpoint/BC selection contract.
