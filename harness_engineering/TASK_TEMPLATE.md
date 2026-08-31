# Task specification and Definition of Done

Resolve this brief before changing files. A short answer is fine, but no field
that can materially change the result should remain implicit.

## Task brief

- Goal: the observable outcome, not merely the edit.
- Non-goals: behavior, files, devices, and data that must remain untouched.
- Files and systems in scope:
- Inputs and provenance: current config, raw run, deck seed, or user-supplied
  value.
- Index convention: physical qubit, displayed `Q#`, resonator `M#`, and Python
  index.
- Units: MHz, GHz, us, A, mA, DAC gain, dB, and sample count as applicable.
- Authority: offline only, or an explicitly authorized lab operation.
- Expected outputs: file paths, plots, datasets, table entries, or console
  evidence.
- Acceptance criteria: objective observations and commands that prove success.
- Abort or rollback condition:

## Hardware authorization envelope

A request to edit code is not authorization to run it. Before any lab operation,
record all of the following in commentary or the task log:

- exact script and whether importing it is safe;
- device, physical qubit, resonator row, and Python index;
- `run_flags`, number of rounds, reps, and selected qubits;
- frequency/current/gain/delay bounds and physical units;
- attenuator values, readout point, and relevant pulse lengths;
- intended data root and longest generated Windows artifact path;
- expected duration and progress/health signal;
- abort condition and safe final hardware state.

Changing Yoko current needs an explicit current target/range and ramp plan. A
generic instruction to “test it” never grants that authority.

## Definition of Done template

- [ ] Requested behavior and non-goals are explicit.
- [ ] Existing user changes and raw data were preserved.
- [ ] Local class, config, plotting, and saving contracts are followed.
- [ ] `tools/harness_check.py` passes in the configured Python environment.
- [ ] `git diff --check` passes and the focused diff was reviewed.
- [ ] Offline evidence is not represented as hardware validation.
- [ ] If authorized, the exact hardware run completed inside its envelope, or
      the partial result and safe final state were recorded.
- [ ] Expected plot and HDF5 files both exist; their configuration/provenance is
      recoverable.
- [ ] Any config change is reported as old value -> new value, affected row,
      experiment, and unit.
- [ ] A durable discovery or repeat failure was added to the harness and log.

## Diagnostic entry template

- Symptom:
- Reproduction and evidence:
- Layer: task specification / context provision / execution environment /
  verification feedback / state management.
- Root cause or best bounded hypothesis:
- Harness change:
- Re-run result:
- Remaining uncertainty and next evidence required:

“The fit returned a number” is not an acceptance criterion. State the quality
gate: acquired bounds, reproducibility, residuals/uncertainty, physical
plausibility, and plot/raw-data persistence as applicable.
