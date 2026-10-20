# Lean research runtime interfaces

These helpers observe research progress; they do not launch, restart, or stop
work. The campaign owner remains the authority for goal state and experiment
scope. Qlib Recorder artifacts remain the authority for research evidence.

```python
from src.core.research_runtime import project_task, observe_job, aggregate_usage

task = project_task(receipts="receipts.json", run_state="run_state.json")
job = observe_job(pid=8421, start_identity="linux:123456", progress_path="run_state.json")
usage = aggregate_usage("session.jsonl")
```

`TaskProjection` keeps `goal_state`, `job_state`, and `scientific_state`
independent. Explicit completed phase receipts override stale pending fields in
`run_state.json`; the mismatch is surfaced in `issues`. Missing receipts remain
unknown. Receipt files may be lists of normalized `{phase, status}` objects, a
single such object, or an object with a `receipts` list. A state-file `running`
claim becomes `unverified_running`; only `observe_job` can verify a live PID.
The projection does not infer scientific success from process success.

`observe_job` uses the supplied PID and expected OS start identity, explicit exit
receipt, progress-file content signature, and heartbeat timestamp. An exit
receipt is trusted only when both its PID and start identity match the expected
handle. A live PID without an expected start identity is
`running_unverified_identity`; PID reuse is reported separately. A state file
by itself never establishes that a process is alive. Heartbeat staleness is an
observation, not a restart instruction.

`aggregate_usage` reads JSONL `token_usage_record` rows. It counts response
`usage` once per `response_id`, with cumulative `turn_token_usage` deltas only
when response-level usage is absent. Cached input and reasoning output remain
reported subsets; model and role are omitted unless a future source explicitly
provides them. It reports token counts and explicitly does not infer billing.
Each invocation uses `aggregation_mode: full_scan` and rereads the complete
source. Response IDs prevent duplicate counting, and cumulative counters are
compared with all response-level usage already counted for that turn, including
response records that arrive between cumulative records. Appends and
truncations are reflected without a persisted byte offset or duplicate
checkpoint. A partial trailing JSONL line is ignored in that scan and will be
re-read on the next invocation.

Read-only CLI, one snapshot:

```bash
PYTHONPATH=. python -m src.core.research_monitor --state run_state.json \
  --receipts receipts.json --pid 8421 --usage session.jsonl --once
```

Bounded watch (maximum 120 observations; default interval 60 seconds):

```bash
PYTHONPATH=. python -m src.core.research_monitor --state run_state.json \
  --watch --interval 60 --iterations 10
```

The CLI prints only changed structured summaries. It does not print source file
contents, prompts, or credentials.

For launched jobs, the owner records `make_job_handle(pid, phase=..., job_id=...)`
immediately after `Popen`; it captures the OS process-start identity. On completion,
`make_exit_receipt(handle, exit_code)` binds the exit to that same process. Keep
these receipts outside guarded experiment directories. The monitor reads them
through `--campaign research/three_asset_campaign.json`; it does not need the
owner's tool session handle. A missing live PID plus a matching exit receipt is a
terminal result, not an invitation to restart.
