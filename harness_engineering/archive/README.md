# Archive

Old harness files, kept whole so nothing is lost. They are history, not instructions.

| File | What it is | Size |
|---|---|---|
| `PUCQ4_STATE_through_2026-11-25.md` | Every state block from Aug to Nov 2026. Was `harness_engineering/PUCQ4_STATE.md`. | 285 KB |
| `FAILURE_MODES_through_2026-11-25.md` | Full text of every recorded failure. Was `harness_engineering/FAILURE_MODES.md`. | 77 KB |

Two more archives live in `docs/archive/`. They are the old task log and the old measurement index.

## Rules

1. Never read an archive file from top to bottom. Search it for a word, a date or a run label, then read only that range.
2. `../LESSONS.md` holds line numbers into these files.
3. Words such as ACTIVE, RUNNING or queued in an archive describe the past. They are never the current state. The current state is `../STATE.md`.
4. Numbers in an archive are seeds. Read live values from the config files.
5. Do not append to archive files. New state goes to `../STATE.md`. New log entries go to `docs/log/`.
6. The two big files are not valid UTF-8. Do not rewrite them.
