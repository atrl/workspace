# v14 Design Notes

## Clean Ablation: GRU Alone vs GRU + Penalty

v12/v13 proved GRU + terminal penalty achieves MaxDD 48.8%. But v13 showed that reducing penalty strength had zero effect — the GRU architecture itself seems to be the dominant factor.

### v14: GRU Only, No Episode Penalty

- **episode_drawdown_penalty: 0.0** — completely remove terminal penalty
- **downside_asymmetry_coef: 1.0** — no asymmetry
- **Keep GRU + observation_window: 3** — the only change from v2 baseline is architecture

### Hypothesis

If v14 achieves MaxDD < 50%, the GRU's temporal memory alone is sufficient for drawdown control. The episode penalty was never necessary — the MLP's inability to track drawdown state was the root cause all along.

If v14 achieves MaxDD > 52%, the episode penalty is still required even with GRU.

### Expected Outcome

- If GRU alone controls drawdown → future work should focus on architecture improvements (larger GRU, LSTM, attention)
- If GRU alone fails → the penalty + GRU combination is required
