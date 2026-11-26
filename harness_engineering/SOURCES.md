# Sources

## Where a fact comes from, strongest first

When two sources disagree, the higher one wins.

1. A fresh instrument readback, or raw HDF5 from a run you can name, with its saved config.
2. The values the code will use now (`system_config.py`, `expt_config.py`, the chosen script and its overrides).
3. `STATE.md`.
4. The characterisation deck and the upstream driver.
5. Comments, archives, old logs, search seeds and memory of the chat.

A value in the code is not automatically validated. A validated value does not change the code until someone writes it there on purpose.

This order is about facts. It is not about limits. A limit in `STATE.md` or `RUN_ENVELOPE.md` always beats a config value. A config gain above a ceiling is never used.

The six evidence labels are defined in `MEASUREMENT_PLAYBOOK.md`.

For a `validated-measurement` record the date, row with index and physical qubit, Yoko state, sweep and averaging, readout and pulse settings, the fitted value with unit, the repeat evidence, and the raw and plot paths. If a field is missing, say so. Never fill it from another run.

## Device and driver

- Characterisation deck on the lab PC. `C:\Users\Ma Quantum Lab\Downloads\PUCQ4 Initial Characterization (1).pptx`. It maps physical Q4 on line D4 to M5 and physical Q6 on line A5 to M6, gives M4 only a 5575 MHz seed with no physical number, and shows current axes from -10 to +10 mA. The folder `_ppt_extract/` in the repo is a working copy of that deck, not a source of truth.
- Chip design record. Named `Device04V2.py` in `docs/q5-q6-profile-debug-plan.md`. It gives the intended qubit to resonator order. Intent is not a confirmed assignment.
- Upstream Yokogawa GS200 driver. https://github.com/ma-quantumlab/malab/blob/main/instruments/voltsource.py . The local `malab_yokogawa_gs200.py` keeps the same interface with a socket and added current guards.

## Harness design

- Walking Labs, Lecture 01 on why capable agents still fail. https://walkinglabs.github.io/learn-harness-engineering/en/lectures/lecture-01-why-capable-agents-still-fail/ . Its five layers are task, context, environment, verification and state.
- Codex reads `AGENTS.md` once at the start of a session, from the repo root down to the start folder, and stops at 32 KiB in total. A later edit needs a new session. That is why the root file is short and the cards are read on demand.
