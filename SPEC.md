# SPEC.md

## Project

Maintain a compact, repeatable `SOXL` qlib research workspace.

The project is successful when future signal experiments can iterate by editing
notebook cards and card-local artifacts while leaving framework code stable.

## In Scope

- SOXL-only research and backtesting.
- Canonical data and qlib calendar/store files under `data/`.
- Reusable abstractions under `soxl_backtest/`.
- Executable experiment evidence under `runtime/<direction>/card/<experiment>/`.
- Focused tests for framework behavior, reference parity, and architecture
  boundaries.

## Out of Scope

- Production broker execution.
- Scheduler or dashboard applications.
- Legacy RL training/model-management surfaces.
- Scattered one-off scripts as the primary experiment workflow.
- Multi-ticker optimization outside SOXL support data.

## Layering

Keep these separate:

1. data loading and qlib calendar alignment
2. signal or reference-policy generation
3. position/backtest accounting
4. artifact persistence and notebook conclusion

`soxl_backtest` owns reusable code. Runtime notebooks own experiment-specific
logic and conclusions.
