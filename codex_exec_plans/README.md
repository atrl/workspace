# Codex Execution Plans

`codex_exec_plans/` is the live task and handoff surface. It should stay compact enough for agents to read before every substantial loop.

## Read Order

1. `active/current.md` — current goal, active acceptance criteria, latest evidence, next action.
2. Any other short active topic workpad under `active/`.
3. `docs/research_knowledge/index.md` — mechanism-specific research memory.
4. `memory/index.json` plus `scripts/memory_search.py` — durable decisions and historical notes.
5. `runtime/index.json` only after the category index points to a relevant output.

## Size Budget

Active workpads are not archives. Keep each active Markdown file at or below `300` lines. Summarize completed loops into `memory/` and `docs/research_knowledge/` instead of storing long historical workpads here.

## Completed Threads

Completed or abandoned threads are retrieved through `memory/index.json`, not through historical files under `codex_exec_plans/`.
