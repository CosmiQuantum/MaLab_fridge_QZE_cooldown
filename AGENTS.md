# Repository guide for coding agents

## Purpose

This repository controls QICK-based superconducting-qubit experiments and analyzes
the resulting data. Treat experiment execution as hardware operation, not as an
ordinary software test.

The main orchestration entry point is `round_robin_benchmark.py`. Experiment
implementations are the numbered `section_*.py` files. Shared experiment settings
live in `expt_config.py` and `system_config.py`; PUCQ4-specific settings live in
`pucq4_config.py`.

## Harness entry point

The detailed, repository-specific harness starts at
[`harness_engineering/README.md`](harness_engineering/README.md). This root file
keeps the rules that must always be visible; the harness folder supplies routed
context, task templates, environment setup, verification ladders, persistent
PUCQ4 state, and diagnosed failure modes.

Before substantive work:

1. Read the mandatory safety and source-of-truth sections in the harness index.
2. Follow its task-type routing table rather than loading every document blindly.
3. For PUCQ4 configuration or hardware work, read
   `harness_engineering/PUCQ4_STATE.md` before interpreting a qubit number.
4. Define acceptance criteria with `harness_engineering/TASK_TEMPLATE.md`.
5. If a task fails, classify the failure using
   `harness_engineering/FAILURE_MODES.md` and record the durable lesson in
   `docs/harness-log.md`.

## Safety boundaries

- Never run experiment scripts, connect to Pyro/QICK/Visdom, change attenuators,
  or write to the `M:` data drive unless the user explicitly asks for a lab run.
- Importing `round_robin_benchmark.py` is unsafe: its module-level code creates
  directories and begins a potentially long hardware loop.
- Do not guess device parameters, qubit frequencies, gains, attenuations, output
  paths, or enabled `run_flags`. Ask when the requested change depends on them.
- Preserve raw data and user experiment changes. Do not delete or rewrite data
  artifacts, and inspect `git status` before editing.
- Keep hardware-free verification separate from hardware validation. Never claim
  that an offline check proves an experiment works on the instrument.

## Working conventions

- Target Python 3.9 syntax unless the project interpreter is deliberately updated.
- Prefer small, localized edits. The experiment scripts contain substantial
  module-level configuration, so avoid broad formatting or refactoring.
- Keep units and indexing explicit. Qubit numbers used in filenames/UI may be
  one-based while Python list indices are zero-based; preserve the local convention.
- When changing a configuration value, state the old value, new value, affected
  qubits/experiment, and physical unit in the handoff.
- Do not add secrets, host credentials, or generated measurement data to Git.

## Required structure for new experiment code

Use the existing files as templates; do not invent a second experiment framework.
For a new numbered experiment, copy the closest `section_*.py` by experiment type.
For a new orchestration script, copy `round_robin_benchmark.py` and retain its
phase ordering. Analysis, migration, and small maintenance utilities are exempt
from the experiment-class rules unless they operate hardware.

### Experiment modules (`section_*.py`)

- Expose a high-level experiment/measurement class with both `__init__` and
  `run`. Keep hardware acquisition inside `run`, not at class definition time.
- Match the established constructor vocabulary where applicable:
  `QubitIndex`, `number_of_qubits`, `outerFolder`, `round_num`, `signal`,
  `save_figs`, and `experiment`. Preserve capitalization because callers use it.
- Store those inputs on `self`, including `self.outerFolder`, `self.round_num`,
  `self.save_figs`, and `self.experiment`. Keep the existing return-value style of
  the nearest analogous experiment so the round-robin collector can consume it.
- Set `self.expt_name`, then obtain experiment settings through
  `expt_cfg[self.expt_name]` or `add_qubit_experiment(...)`, matching the nearest
  existing implementation. Use the passed `QICK_experiment` object's
  `readout_cfg` and `qubit_cfg`; do not create parallel config dictionaries or
  embed replacement device configuration in the class.
- Keep low-level `AveragerProgramV2` program classes separate from the high-level
  wrapper, following the nearest existing `section_*.py` file.
- Gate plot output with `self.save_figs`. Build paths with
  `os.path.join(self.outerFolder, ...)`, create an experiment-specific plot
  subfolder using the established helper, and retain filenames containing round,
  one-based displayed qubit number, timestamp, and experiment name when the
  analogous file does so.
- Do not save the round-robin HDF5 dataset from a section class. Return results and
  configuration to the orchestrator; the orchestrator saves them through
  `Data_H5`.
- Standalone `pucq4_*.py` acquisition scripts are orchestrators for their own
  run. They must use `pucq4_config.create_data_dict` and
  `pucq4_config.save_h5`/`Data_H5`, retain `Dates`, `Round Num`, `Batch Num`,
  `Exp Config`, and `Syst Config`, and save under
  `subStudyDataFolder/Data_h5/<dataset label>/`. Do not introduce new NPZ/CSV
  raw-data formats for these scripts.
- This lab computer still fails on generated Windows paths longer than 260
  characters. Before a hardware run, calculate the longest intended artifact
  path (including the timestamped `Data_H5` filename) and shorten the sub-study
  or dataset label if needed. After acquisition, verify that both the expected
  plot and HDF5 file exist; a saved plot alone does not prove raw data persisted.

### Round-robin orchestration scripts

Retain the existing sequence and naming scheme:

1. Imports, including experiment classes, `Data_H5`, `QICK_experiment`, and values
   from `expt_config`.
2. `Run Configurations`, including `run_flags`, qubit selection, save/plot options,
   run metadata, and per-device values.
3. `Data Saving Setup`, with this hierarchy:
   `<data root>/<run_name>/<device_name>/<study>/<sub_study>/<timestamp>/`, then
   `optimization/`, `study_data/`, and `documentation/`.
4. Logging under `documentation/` and notes saved as `sub_study_notes.txt`.
5. Per-experiment key lists and object-array dictionaries created with
   `create_data_dict`.
6. The round/qubit loop: create `QICK_experiment`, select the qubit-specific
   `readout_cfg`/`qubit_cfg` values, gate each experiment with `run_flags`, call its
   `run`, collect results plus `Exp Config` and `Syst Config`, then release objects.
7. Batch saving through `Data_H5(subStudyDataFolder, ..., batch_num, save_r)` using
   the established dataset labels, followed by dictionary reinitialization.

New scripts must receive or construct the same folder variables rather than save
plots or data to ad-hoc relative directories. A different data root, hierarchy,
dataset label, or class/run interface requires explicit user approval and must be
called out in the task log.

## Verification

Safe default check (parses every tracked Python file without importing it):

```powershell
& 'C:\Users\Ma Quantum Lab\anaconda3\python.exe' tools/harness_check.py
```

The check also reviews newly added `section_*.py` and `round_robin*.py` files for
the structural contracts above. These checks are intentionally limited to new
files so they do not demand unrelated rewrites of historical scripts.

Before running any Python command, use the interpreter configured for this PyCharm
module rather than assuming the path above is still current.

For each task, also inspect the focused diff:

```powershell
git diff --check
git diff -- <changed-files>
```

Hardware validation is a separate, user-authorized step. Record the exact script,
run flags, device, qubits, sweep bounds, attenuations, output location, and abort
condition before starting it.

## Definition of done

A coding task is complete only when:

1. Its requested behavior and non-goals are explicit.
2. The safe offline verifier passes.
3. The focused diff contains no accidental configuration or data-path changes.
4. Hardware validation is either completed with recorded results or clearly marked
   as not run and requiring an authorized lab session.
5. New repository knowledge or a repeated failure mode is captured in the
   relevant `harness_engineering/` document and `docs/harness-log.md`.

## PUCQ4 calibration acceptance rules

- PUCQ4 has exactly two flux-tunable qubits: physical Q4 on Yoko 3
  (`192.168.1.73`, DC line D4, resonator row M5) and physical Q6 on Yoko 4
  (`192.168.1.77`, DC line A5, resonator row M6). Treat the other qubits as
  fixed-frequency unless new device documentation explicitly supersedes this.
- At zero current, verify both Yokogawas are in current mode at 0 A before a
  benchmark. Use the PowerPoint characterization range as the hard boundary for
  later current sweeps; never extrapolate the current range.
- Tune qspec drive gain for a clean, distinguishable line near 0.1-0.2 MHz FWHM
  and calibrated peak population around 0.3-0.5. Use enough frequency points to
  resolve each peak with several samples. Do not modify the underlying qspec
  QICK program to achieve this.
- Tune Gaussian sigma so amplitude Rabi has just over half an oscillation and a
  fitted pi amplitude in the 0.6-0.7 DAC-gain window. Re-run Rabi after any qspec
  frequency or sigma change, then propagate the measured pi amplitude to all
  shared configs.
- Optimize single-shot readout by alternating: (1) length at fixed gain/offset,
  choosing the shortest length at the fidelity plateau, and (2) a 2D resonator
  gain/frequency-offset sweep at that length. Repeat until improvement is no
  longer significant. If qubit frequency, qspec power, sigma, or pi amplitude
  changes materially, revalidate SSF.
- Final PUCQ4 evidence must come from one-pass `round_robin_benchmark.py` output
  with its normal calibrated population plots and canonical HDF5 hierarchy.
  Required plots are res spec, qspec, Rabi, SSF, T1, T2R, and T2E for every
  physically assigned qubit. T2 sampling must provide at least four points per
  fringe.
- Maintain a reusable PowerPoint in the `pucq4_first_light` run folder. Slides
  begin with a zero-Yoko-current device-map table listing each physical qubit,
  its repository/resonator index, attached resonator frequency, qubit frequency,
  T1, T2R, and T2E (with units and one-based/zero-based conventions explicit).
  After the Yoko characterization, include a corresponding current-sweep summary
  table for the two flux-tunable qubits. Populate both tables only from validated
  round-robin/current-sweep results, not seeds or diagnostic fit candidates. Slides
  group all qubits by experiment (res spec, qspec, Rabi, SSF, T1, T2R, T2E),
  followed by final readout-length and gain/frequency optimization slides.
  Annotate relevant configurations and use the plots generated by round robin.
- For Yoko 3/Q4 and Yoko 4/Q6 current characterization, add current-versus-
  frequency qspec and res-spec maps, current-versus-gain Rabi maps, the selected
  sigma-versus-current calibration (pi amplitude 0.6-0.7), and T1/T2R/T2E maps.
  Save current-indexed sigmas in shared configuration for reuse and add the
  final maps plus acquisition settings to the reusable PowerPoint.
