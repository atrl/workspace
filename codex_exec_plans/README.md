# Codex Execution Plans

`codex_exec_plans/` is the live task and handoff surface. It should stay compact enough for agents to read before every substantial loop.

## Read Order

1. `active/current.md` — current goal, active acceptance criteria, latest evidence, next action.
2. Any other short active topic workpad under `active/`.
3. `docs/research_knowledge/index.md` — mechanism-specific research memory.
4. `memory/index.json` plus `scripts/memory_search.py` — durable decisions and historical notes.
5. `runtime/` evidence only after the index points to a relevant output.

## Size Budget

Active workpads are not archives. Keep each active Markdown file at or below `300` lines. Move completed loops to `completed/YYYY-MM/` and link them from `active/current.md` when still relevant.

## Completed Threads

Completed or abandoned workpads keep the detailed command/evidence trail. They are searchable history, not the default startup surface.
