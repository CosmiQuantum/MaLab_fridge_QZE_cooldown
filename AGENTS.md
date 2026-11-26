# Rules for coding agents in this repo

This repo runs QICK measurements on superconducting qubits in the lab and analyses the data. Running an experiment script is a lab operation on real hardware. It is never a software test.

The detailed harness starts at `harness_engineering/README.md`. This file holds the rules that always apply and tells you which card to read.

## Start of every chat, and again after your context was compacted

1. Read `harness_engineering/STATE.md`. It is short. It holds the current state, the open blocker and the next step.
2. Find your task below and read only the cards named there.
3. Never read a file under `archive/` from top to bottom. Search it.

| Task | Read in `harness_engineering/` |
|---|---|
| Measure, calibrate or diagnose on hardware | `RUN_ENVELOPE.md`, `MEASUREMENT_PLAYBOOK.md`, the identity map in `PUCQ4_DEVICE.md`, Before a run and Long runs in `ENVIRONMENT_AND_SAFETY.md`, the overrides in `REPOSITORY_MAP.md` |
| Write or change acquisition code | `EXPERIMENT_CONTRACTS.md`, `REPOSITORY_MAP.md` |
| Find a class, config key, label or tool | `REPOSITORY_MAP.md` |
| Python setup on the lab PC, Yokogawa steps | `ENVIRONMENT_AND_SAFETY.md` |
| Something failed or looks wrong | `LESSONS.md`, searched by symptom |
| Analyse saved data, build a report or deck | `VERIFICATION.md`, deck rules in `PUCQ4_DEVICE.md` |
| Change the harness itself | `README.md` |

## Rules that always hold

Hardware

1. Touch hardware only for a measurement the user asked for in this chat, and only inside `harness_engineering/RUN_ENVELOPE.md`. Hardware means running or importing an acquisition script, QICK or Pyro, the Yokogawas, the attenuators, or writing to the `M:` drive.
2. Never import `round_robin_benchmark.py` or any other acquisition script. They start hardware at import. Read them as text.
3. Never guess a device parameter, port, frequency, gain, attenuation or path. A value from config, a deck, a log or an archive is a seed until a fresh measurement confirms it. A limit in `STATE.md` or `RUN_ENVELOPE.md` always beats a config value.
4. One acquisition at a time. After any crash or kill, read back both Yokogawas and bring them to 0 A with output off.

Structure

5. Reuse the existing classes, QICK programs, plots and `Data_H5` saving. First try run settings of an existing script. A new pulse sequence is a copy of the nearest `section_*.py` with the same interface. No second framework, no new raw data format, and no new helper script when an existing script with its run flags does the job.
6. Every run saves into `<data root>/<run_name>/<device_name>/<study>/<sub_study>/<time stamp>/` with exactly `optimization/`, `study_data/` and `documentation/` inside. Raw data goes to `study_data/Data_h5/<label>/`, plots and notes go to `documentation/`. Never add, drop or rename a level.
7. One sub-study per sweep, with a name a person can read. Check the path length before launch with `python tools/check_run_folder.py plan`.
8. Name every qubit with row, Python index and physical qubit. Physical Q4 is M5, index 4. M4, index 3, is not physical Q4.
9. Run `git status` before you edit. Changes you did not make belong to the user. Commit or push only when asked.

Science

10. The gates are fixed numbers. Only the user changes them.

| Stage | Gate |
|---|---|
| Qspec | FWHM at or below 0.5 MHz, at least 3 acquired points across the line |
| SSF | histogram contrast above 50 percent, never converted to another metric |
| T2R and T2E | at least 8 acquired points per fringe |
| Relaxation delay | 5 to 10 times the measured T1 |
| Readout gain | below the onset of resonator movement |

11. A miss is written down as a miss. Never relax, rename or reinterpret a gate, and never invent a new acceptance label or review to pass.
12. A fitted number, a saved plot or a finished script is not a result. Look at the raw data, the sampling and a repeat.
13. Two attempts per gate and row, then stop that stage, record it and tell the user what you need. When a value that used to pass now fails, first rerun the last validated setting unchanged. If a limit in `RUN_ENVELOPE.md` forbids that setting, ask.
14. Never start a stage that depends on a failed gate, unless the user asked for that stage by name. Then label its result `measured-below-gate`.

State and tokens

15. After every result, pass or miss, run `python tools/check_run_folder.py check` on the new folder, overwrite `harness_engineering/STATE.md`, and append one entry of at most 12 lines to the month file in `docs/log/` (format in `harness_engineering/TASK_TEMPLATE.md`).
16. Search before you read. Never read a file over 40 KB whole. `REPOSITORY_MAP.md` tells you where things are.
17. Write for the next agent in plain words with spaces, plain ASCII, short lines. No hashes, and no process ids or ACTIVE and RUNNING labels outside `STATE.md`.
18. Do not copy rules or state into other files. Each rule has one home.

## Ask or decide

The user wants to give one request and then be left alone. Decide alone and record it when the choice is inside the envelope. That covers which script and flags, the order of stages, prerequisite stages, sweep ranges, reps, names, and writing validated values into config.

Ask once, in one short message with the number or choice you need, for the items on the ask list in `RUN_ENVELOPE.md`.
The main ones are any nonzero Yoko current, readout gain above a ceiling, a missing device parameter, a new folder level or data format, a changed gate, a run over about 6 hours, deleting data, and a conflict between two of the user's rules.
While you wait, carry on with the parts that do not depend on the answer.

## Checks

```powershell
python tools/harness_check.py
git diff --check
```

The first command parses every Python file without importing it and checks the harness files. If plain `python` fails on the lab PC, use the Conda form in `ENVIRONMENT_AND_SAFETY.md`.

## Done means

1. The goal is met, or the miss is recorded with its number and what was tried.
2. `python tools/harness_check.py` passes and the diff holds no stray config or path change.
3. After hardware work the run folder check passed, the hardware is parked, and both facts are in `STATE.md`.
4. Offline checks are never reported as hardware validation.
5. Every changed config value is reported as old to new with unit, row and evidence.
