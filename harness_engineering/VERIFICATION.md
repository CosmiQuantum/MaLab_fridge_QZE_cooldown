# Verification and feedback

Verification closes the gap between “the agent thinks it is done” and observable
correctness. Use the highest authorized rung; do not skip the safe lower rungs.

## Verification ladder

### 0. Specification and state

- Resolve goal, non-goals, scope, units, indexing, authority, and acceptance
  criteria.
- Inspect `git status` before edits.
- Read current source values and the relevant state/log entry.

### 1. Hardware-safe static check

Use the configured PyCharm Python environment as described in
`ENVIRONMENT_AND_SAFETY.md`, then run:

```powershell
tools/harness_check.py
```

The checker parses tracked and new Python files as Python 3.9 without importing
them. It detects syntax/conflict markers, checks new experiment/orchestration
structure, validates `harness_engineering/manifest.json`, checks all required
harness files, and confirms that `AGENTS.md` points to the harness.

It does not connect to QICK, validate a pulse program, verify dependencies, or
prove a fit/plot is scientifically correct.

### 2. Focused change review

```powershell
git diff --check
git diff -- <changed files>
```

Because harness files may initially be untracked, also inspect their full content
or use `git diff --no-index -- NUL <file>` on Windows. Confirm there are no
accidental run-flag, frequency, gain, attenuation, path, or raw-data changes.

### 3. Pure-logic verification

When a changed behavior can be isolated from hardware, test the pure function or
analysis logic with synthetic inputs. Never import an acquisition orchestrator to
reach a helper. Extracting pure logic requires a separate, scoped code change and
must preserve the experiment interface.

### 4. Artifact verification

For analysis of an existing run:

- cite the raw HDF5 path and dataset label;
- confirm plot axes/units and that requested calibration was applied;
- compare fitted center/time with acquired bounds and raw structure;
- inspect residuals/uncertainty/repeatability rather than accepting a scalar;
- distinguish missing data, rejected data, and an actual physical null result.

### 5. Authorized hardware validation

Run only inside the recorded hardware envelope. Record actual script, Git/config
state, flags, rows, bounds, reps/rounds, attenuators, output folder, duration,
abort events, and final instrument state. Verify both plots and HDF5 output before
starting a dependent experiment.

## PUCQ4 scientific acceptance gates

- Resonator spectroscopy: a reproducible, unsaturated feature with enough points
  to locate the center/readout slope; state whether the result is high-power,
  low-power, or punch-out.
- Qubit spectroscopy: use the unchanged repository QICK program. The current user
  target is a distinguishable calibrated-population peak around 0.3-0.5 with
  approximately 0.1-0.2 MHz FWHM and several acquired points across the line.
  Reproduce a new/shifted candidate before updating shared config.
- Rabi: just over half an oscillation in the default plot and a fitted pi
  amplitude near 0.6-0.7 DAC gain. A sigma or qfreq change requires a new Rabi.
- T1/T2R/T2E: a resolved decay/fringe, physically plausible bounded fit, and
  sufficient delay range. T2 sampling must give at least four points per fringe;
  current 0.2 us spacing at 0.5 MHz detuning gives ten.
- Single shot: optimize readout length and gain/frequency iteratively. Choose the
  shortest length at the fidelity plateau, then repeat the 2D sweep. Accept only
  stable repeated fidelity; a numerical maximum on a sparse/noisy grid is not
  enough.
- Final characterization: one pass of the normal round-robin chain, canonical
  population plots, complete HDF5/config provenance, and no diagnostic override
  left active.

## Change propagation checks

- qfreq change -> re-run qspec fit and Rabi.
- sigma or pi-amplitude change -> re-run Rabi, then T1/T2 and SSF as relevant.
- readout frequency/gain/length change -> re-run SSF and confirm all population
  calibrated experiments still look correct.
- Yoko current change -> re-find resonator/qfreq as planned, select the
  current-specific sigma/pi amplitude, and never reuse zero-current coherence as
  though it were measured at that bias.

## Completion claim

State which rungs passed and which did not run. “Offline verifier passed” and
“hardware validated” are deliberately different claims. If a requested endpoint
depends on unavailable authorization or new external evidence, report it as
pending rather than declaring success.
