# Goal

Continue the active SOXL agent loop in this repository until the next highest-priority task is completed with proof of work in the repo.

# Context

This repo's active contract is in `AGENTS.md`.
The active execution plan is `codex_exec_plans/active/2026-04-10-soxl-agent-loop.md`.
The current system already has:

- qlib-native SOXL training and backtest flow
- paper replay runtime with a broker adapter contract
- IB semi-auto scaffolding
- promotion gates wired into saved model summaries and runtime blocking
- isolated walk-forward worker plumbing in `src/soxl_quant/cli.py`

The most important remaining near-term task is to validate and harden the walk-forward harness so promotion is deterministic and the last window no longer gets lost to Python 3.14 / qlib shutdown noise.

Current true repo progress, which should override any stale assumptions:

- `position_cap` config is fully wired through training, slice evaluation, backtest, saved bundles, and replay
- promotion gates are already implemented in training summaries and semi-auto runtime blocking
- isolated walk-forward worker plumbing already exists in `src/soxl_quant/cli.py`
- artifact-tag isolation for walk-forward worker outputs has been added so `wf_*` runs no longer overwrite the canonical fixed-split actor/summary

This means the next rotation should not re-implement promotion gates or broker abstractions. It should validate the harness and refresh model evidence.

# Success Criteria

- [ ] Read startup context from `AGENTS.md`, `SOUL.md`, `USER.md`, today's and yesterday's memory, and the active plan before acting
- [ ] Execute exactly one meaningful SOXL loop step
- [ ] Prefer the current highest-priority task from the active plan instead of inventing side quests
- [ ] Prioritize a real validation of the isolated walk-forward path before further strategy tuning
- [ ] Leave proof of work in the repo:
- [ ] Update the active plan with command, result, and keep/drop decision
- [ ] Update today's memory with the important result

# Constraints

- SOXL only; do not expand to other symbols
- Respect the repo workflow in `AGENTS.md`
- Do not create a parallel workflow outside `AGENTS.md` and `codex_exec_plans/`
- Prefer improving the package under `src/soxl_quant/`
- If running experiments, do not optimize directly on held-out `2022` labels beyond the documented evaluation framework
- If a run fails, document the failure and tighten the harness instead of blindly retrying
- Avoid destructive git commands
- The preferred immediate command is:
- `python3 soxl_qlib_strategy_v2.py --profiles balanced --walk-forward --skip-data-refresh --seed 20260411`
- The key question to answer is:
- does the isolated worker harness now produce a deterministic `wf_2023_2024` result and a stable promotion verdict in `artifacts/models/soxl_balanced_summary.json`?
