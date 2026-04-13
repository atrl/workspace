# Codex Execution Plans

This directory is the execution harness for SOXL research.

## Layout

- `active/`: plans currently being worked
- `completed/`: plans that finished or were superseded

## Per-plan requirements

Each plan should include:

- Goal
- Current baseline
- Hypotheses
- Validation harness
- Exit criteria
- Decision log
- Latest result
- Next action

## Rules

- One active plan per main optimization thread.
- Update the active plan before or immediately after each validation run.
- If a direction is abandoned, record why and move the plan to `completed/`.
- Do not rely on chat history as the primary experiment log.
