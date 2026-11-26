# Task brief, log entry and done list

## Brief

Settle these in your head or in two lines before you change anything. Write them down only when the task is large.

| Field | Content |
|---|---|
| Goal | The result the user wants to see, not the edit |
| Not in scope | Files, rows, devices and data that stay untouched |
| Identity | Row, Python index, physical qubit, bias |
| Inputs | Which values are measured now, and which are seeds |
| Acceptance | The gate numbers that decide pass or miss |
| Time | Expected duration, and the bound at which you stop |
| End state | Parked hardware, files that must exist |

If one field cannot be filled from the request, `STATE.md` or the code, and it is on the ask list in `RUN_ENVELOPE.md`, ask once. Otherwise decide and record.

## Log entry

Logs live in `docs/log/`, one file per month, named like `2026-11.md`. Append at the end. Never open an old month to append. If a month file passes 60 KB, continue in a file with a letter, like `2026-11-b.md`.

Each entry is one heading line and at most 12 lines of text.

```text
## <YYYY-MM-DD> <HH.MM> <kind> <row and what, in a few words>
- Asked. <what the user asked, with row, index, physical qubit and bias>
- Ran. <script, run flags, sub-study name>
- Result. <the numbers with units, each with pass or miss against its gate>
- Label. <one of the six evidence labels>
- Data. <path of the data set folder>
- Config. <file and key, old to new, unit> or none
- Next. <the one next step>
```

`<kind>` is one of `result`, `miss`, `failure`, `decision`, `harness`. The time is written with a dot, like `14.30`.

Writing rules for the log and for `STATE.md`.

- Plain words with spaces between them. Plain ASCII.
- Lines under 400 characters. Shorter is better.
- Round numbers to the digits that matter. Full precision stays in the HDF5 and the config.
- No process ids, no hashes, no slide counts, no restating of the parked state unless it changed. The one process id that is written down is the long run you started, in `STATE.md`, while it runs.
- Never write ACTIVE, RUNNING or queued into the log. A live state belongs only in `STATE.md`.
- A failure entry adds two lines, `Cause` (or best guess, marked as a guess) and `Layer` (task, context, environment, verification or state).

## Done list

- The goal is met, or the miss is recorded with its number.
- User changes and raw data are untouched.
- Code follows `EXPERIMENT_CONTRACTS.md`.
- `python tools/harness_check.py` passes.
- The diff holds no stray config or path change.
- After hardware work the run folder check passed and the hardware is parked.
- Every config change is reported as old to new with unit, row and evidence.
- `STATE.md` is current and one log entry is written.
