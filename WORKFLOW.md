---
tracker:
  kind: linear
  api_key: $LINEAR_API_KEY
  project_slug: "soxl-harness-4cd6b62a198a"
  active_states:
    - Todo
    - In Progress
    - Rework
    - Human Review
    - Merging
  terminal_states:
    - Done
    - Canceled
    - Duplicate
polling:
  interval_ms: 5000
workspace:
  root: ~/code/symphony-workspaces
hooks:
  after_create: |
    git clone --depth 1 git@github.com:atrl/workspace.git .
  before_remove: |
    git status --short
agent:
  max_concurrent_agents: 10
  max_turns: 20
codex:
  command: codex --config shell_environment_policy.inherit=all --config model_reasoning_effort=xhigh --model gpt-5.3-codex app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
---

You are working on a Linear issue `{{ issue.identifier }}`

{% if attempt %}
Continuation context:

- This is retry attempt #{{ attempt }} because the issue is still in an active state.
- Resume from the current workspace state instead of restarting from scratch.
- Do not repeat already-completed investigation or validation unless needed for
  new code changes.
- Do not end the turn while the issue remains in an active state unless you are
  blocked by missing required permissions, secrets, or tools.
{% endif %}

Issue context:
Identifier: {{ issue.identifier }}
Title: {{ issue.title }}
Current status: {{ issue.state }}
Labels: {{ issue.labels }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Instructions:

1. This is an unattended orchestration session. Never ask a human to perform
   routine follow-up actions.
2. Only stop early for a true blocker such as missing required auth,
   permissions, or tools. If blocked, record it in the workpad comment and move
   the issue according to workflow.
3. Work only in the provided repository copy. Do not touch other paths.
4. Final reporting should contain completed actions and blockers only.

## Prerequisite

The workflow expects Linear access through Symphony's tracker integration, and
repo skills may also rely on the `linear_graphql` tool when available.

## Default posture

- Start by determining the issue's current status, then follow the matching flow.
- Start every task by opening the tracking workpad comment and bringing it up to
  date before doing new implementation work.
- Spend extra effort up front on planning and verification design before
  implementation.
- Reproduce first: confirm the current behavior or issue signal before changing
  code so the fix target is explicit.
- Treat a single persistent Linear comment as the source of truth for progress.
- Use that single workpad comment for progress and handoff notes; do not create
  scattered status comments.
- Operate autonomously end-to-end unless blocked by missing requirements,
  secrets, or permissions.
- Move status only when the matching quality bar is met.

## Related repo context

- `AGENTS.md` is the top-level operating contract.
- `SPEC.md` defines project intent and invariants.
- This repository is for `SOXL` only.
- Durable proof of work must still be written back into the repo, especially
  under `codex_exec_plans/` and `memory/`.

## Status map

- `Backlog` -> out of scope for this workflow; do not modify.
- `Todo` -> queued; immediately transition to `In Progress` before active work.
- `In Progress` -> implementation actively underway.
- `Human Review` -> validated and waiting on human review.
- `Merging` -> approved and ready for merge / landing flow.
- `Rework` -> reviewer requested changes; planning and implementation required.
- `Done` -> terminal state; no further action required.

## Step 0: Determine current issue state and route

1. Fetch the current issue state from the provided Linear issue context.
2. Route to the matching flow:
   - `Backlog` -> stop and wait for a human to move it to `Todo`
   - `Todo` -> immediately move to `In Progress`, then ensure bootstrap workpad
     comment exists, then start execution flow
   - `In Progress` -> continue execution flow from the current workpad comment
   - `Human Review` -> wait for feedback or approval; do not start new work
     unless the issue moves back to an active implementation state
   - `Merging` -> only perform merge / landing actions
   - `Rework` -> resume planning and implementation from the workpad comment
   - `Done` -> do nothing and shut down
3. Check current repo state before acting:
   - branch
   - `git status`
   - current `HEAD`
4. If issue state and repo/workpad reality disagree, update the workpad comment
   first, then proceed with the safest flow.

## Step 1: Start/continue execution

1. Find or create a single persistent workpad comment for the issue.
2. Search for the marker header `## Codex Workpad`.
3. Reuse the existing workpad comment if present. Do not create multiple active
   workpad comments.
4. Immediately reconcile the workpad before new edits:
   - check off items that are already done
   - expand or fix the plan so it is comprehensive for current scope
   - ensure acceptance criteria are explicit
   - ensure validation is explicit
5. Before implementing, capture a concrete reproduction or baseline signal and
   record it in the workpad comment.
6. Treat the workpad comment as the active execution checklist.

## Step 2: Execution phase

1. Implement against the workpad checklist and keep it current as reality changes.
2. Check off completed items and add newly discovered tasks in the correct
   section.
3. Run the validation required for the actual scope.
4. Prefer targeted proof that directly demonstrates the behavior you changed.
5. If the task exposes a harness weakness, improve the harness and record that
   improvement as part of the result.
6. Re-check acceptance criteria before considering the issue ready for review.
7. Write durable proof of work back to the repo:
   - update the relevant workpad file under `codex_exec_plans/` when needed
   - update `memory/YYYY-MM-DD.md` with the important continuity note
   - update any affected docs or skills when reusable workflow knowledge changed

## Step 3: Review handling

1. Update the workpad comment with final:
   - commands run
   - validation or failure evidence
   - keep / drop decision
   - blockers, if any
2. Only move the issue to `Human Review` when acceptance criteria and validation
   are explicitly satisfied.
3. When the issue is in `Human Review`, wait for feedback instead of doing new
   work.
4. If review feedback requires changes, move the issue to `Rework` and continue
   from the same workpad comment.
5. When the issue is in `Merging`, perform only merge / landing work.
6. Move the issue to `Done` only when the work is accepted and the durable repo
   record is complete.

## Blocked-access escape hatch

Use this only when completion is blocked by missing required tools, auth, or
permissions that cannot be resolved in-session.

When blocked, record in the workpad comment:

- what is missing
- why it blocks required validation or completion
- the exact action needed to unblock

Do not mark work blocked for ordinary debugging, noisy warnings, or because a
first attempt failed.

## Required proof of work

A loop is not complete until the repository contains the result.

Every substantial iteration should leave behind:

- goal or hypothesis
- command or procedure executed
- validation evidence or failure evidence
- keep / drop / next-action decision

If work touches execution or deployment, also record whether it changed:

- signal generation
- position sizing
- broker execution
- reconciliation or monitoring
