# The harness

These files tell a coding agent how to work in this repo. `AGENTS.md` in the repo root is loaded at the start of every chat. It names the card to read for each task. Nothing here allows a hardware run by itself.

## Files

| File | Holds | Changes |
|---|---|---|
| `../AGENTS.md` | The rules that always apply, and the task table | rarely |
| `STATE.md` | Current state, blocker, next step, where the data is | after every result, overwritten |
| `RUN_ENVELOPE.md` | What may run without asking, the fixed limits, the ask list, one-off permissions | only on the user's word |
| `SSF_CHECK.md` | Steps for the one SSF check on M1 that the user allowed on 25 Nov 2026. Temporary | once, after the check |
| `MEASUREMENT_PLAYBOOK.md` | Gates, evidence labels, stop rule, order of work | when the user changes a rule |
| `PUCQ4_DEVICE.md` | Qubit map, flux and RF facts, device targets, deck rules | rarely |
| `EXPERIMENT_CONTRACTS.md` | Folder structure, raw data, class and script structure | rarely |
| `REPOSITORY_MAP.md` | Where classes, config keys, labels and tools are | when code moves |
| `ENVIRONMENT_AND_SAFETY.md` | Hardware boundary, Python on the lab PC, long runs, Yokogawa steps | rarely |
| `VERIFICATION.md` | The check steps and what each tool proves | rarely |
| `LESSONS.md` | One line per lesson, searched by symptom | one line per new lesson |
| `TASK_TEMPLATE.md` | Brief, log entry format, done list | rarely |
| `SOURCES.md` | Which source wins, device and design sources | rarely |
| `manifest.json` | Machine readable list of files, size limits and gates | with the files |
| `../.codex/hooks.json` | A Codex hook. After a chat is compacted it prints `AGENTS.md` and `STATE.md` again | rarely |
| `archive/` | The old state and failure files, whole | never |
| `../docs/log/` | Task log, one file per month | appended |
| `../docs/MEASUREMENT_INDEX.md` | One line per campaign | one line per campaign |

## Upkeep rules

1. One home per rule. Before you add a rule, search for it. If it exists, change it there. Never copy a rule into a second file.
2. State is overwritten, history is appended. `STATE.md` never grows by dated blocks. The log gets one short entry.
3. Every file has a size limit in `manifest.json`. `tools/harness_check.py` fails when a file passes it. When a file is full, cut or move text. Do not raise the limit to make room for history.
4. Plain ASCII, plain words with spaces, short lines.
5. When a rule replaces an older one, delete the old wording everywhere and add its key phrase to `superseded_phrases` in `manifest.json`. The check then fails if the old wording comes back.
6. A repeated failure is a harness defect. Find which layer failed (task, context, environment, verification, state), fix the narrowest one, add one line to `LESSONS.md`.
7. A gate, a limit or a permission changes only on the user's word, with the date.
8. After any harness change run `python tools/harness_check.py`.
9. `AGENTS.md` is read once per chat and cut off without warning above 32 KiB. Keep it far below. A change to it reaches the agent only in a new chat.
10. The hook in `.codex/hooks.json` does nothing until a person has reviewed and trusted it on that machine (the `/hooks` command in the Codex CLI, or the hooks settings in the app). Any change to the file needs a new review. If the hook is not trusted or the Codex surface has no hooks, the rule at the top of `AGENTS.md` still applies. Read `STATE.md` again after a compaction.

## Why it is built this way

- The agent reads little. The root file is about 6 KB. A measurement task adds 30 to 50 KB of cards. Before the tune-up of 25 Nov 2026 the same task was told to read about 350 KB.
- Rules that used to be relaxed after a miss are now numbers in one table, with a stop rule.
- The folder structure is checked by a tool (`tools/check_run_folder.py`), not by prose.
- Sweep settings are checked against the gates by a tool (`tools/check_config_rules.py`). Whether an SSF target is reachable at all is estimated by a tool (`tools/estimate_ssf.py`).
- The five layers of the Walking Labs lecture are still covered. Task is `TASK_TEMPLATE.md`. Context is `REPOSITORY_MAP.md`, `EXPERIMENT_CONTRACTS.md` and `PUCQ4_DEVICE.md`. Environment is `ENVIRONMENT_AND_SAFETY.md` and `RUN_ENVELOPE.md`. Verification is `VERIFICATION.md` and the tools. State is `STATE.md`, `LESSONS.md` and the log.
