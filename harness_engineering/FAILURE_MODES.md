# Diagnostic loop and known failure modes

When something fails, first attribute it to one of the five harness layers. Fix
the narrowest layer, re-run, and record the result. Do not compensate for a
missing rule by increasing power, scan width, repetitions, or model effort
without evidence.

## Layer checklist

| Layer | Ask | Typical repair |
| --- | --- | --- |
| Task specification | Are outcome, non-goals, units, indexing, authority, and acceptance objective? | Fill `TASK_TEMPLATE.md`; make the hardware envelope bounded. |
| Context provision | Did the agent see the right architecture/config/data convention? | Update the repo map/contract; link the exact analogous file. |
| Execution environment | Is the IDE SDK activated, dependency/runtime correct, and hardware state known? | Fix setup or preflight; do not alter experiment physics to mask it. |
| Verification feedback | Can success/failure be observed independently? | Add a safe check, positive control, raw-data check, or fit quality gate. |
| State management | Did a prior result lose provenance or become a false fact? | Label evidence, update PUCQ4 state, and log rejected approaches. |

## Recorded failure patterns

### Unsafe import starts work

- Symptom: importing an orchestration module creates directories or begins a lab
  loop.
- Root cause: substantial module-level execution in round-robin scripts and
  `QICK_experiment` construction.
- Prevention: inspect as text; the offline checker uses `ast` and never imports
  project modules.

### Python starts but NumPy cannot load DLLs

- Layer: execution environment.
- Root cause: direct use of the Conda interpreter without activating its runtime.
- Prevention: query the PyCharm SDK immediately before Python commands and use
  the activated invocation recorded in `ENVIRONMENT_AND_SAFETY.md`.

### Qspec shows ringing/oscillations

- Layer: context/configuration, then verification.
- Root cause: coherent sidelobes from a 20 us rectangular drive at excessive
  spectroscopy gain, not a need to replace the QICK program.
- Prevention: tune low-power config and inspect calibrated population/linewidth;
  keep `PulseProbeSpectroscopyProgram` unchanged.

### Qspec fit is outside the sweep or has negative width

- Layer: verification/analysis.
- Root cause: array index used as a frequency seed and unconstrained width.
- Prevention: seed with acquired MHz, bound center to acquired range, require
  positive width, and reject a scalar fit that conflicts with raw data.

### Resonator scan is unexpectedly slow

- Layer: context.
- Root cause: res spec sweeps frequency in Python; putting hundreds of averages
  in QICK `rounds` causes a network round trip at every point. Qspec instead uses
  a hardware frequency loop.
- Prevention: preserve equivalent averaging with res-spec `reps` high and
  `rounds=1`; do not generalize this rule to experiments with different loops.

### Plot exists but raw HDF5 does not

- Layer: verification/environment.
- Root cause: generated Windows path exceeded 260 characters; acquisition and
  plot completed before HDF5 creation failed.
- Prevention: preflight full generated filename, shorten labels, then separately
  assert plot and HDF5 existence before continuing.

### Physical Q4 is confused with displayed Q4

- Layer: task specification/state.
- Root cause: one-based UI names, zero-based list indices, resonator rows, and
  physical device numbers are not equivalent on PUCQ4.
- Prevention: identify all dimensions. Physical Q4 is M5/index 4; M4/index 3 is
  unassigned.

### Rejected M4 candidates reappear as facts

- Layer: state management.
- Root cause: a deck seed or one-point/high-power fluctuation was copied into a
  numeric six-row config and later read as measured truth.
- Prevention: `NaN` in measured-results state, explicit `seed`/`placeholder`
  labels where numeric arrays require a value, and the bounded stop rule in
  `PUCQ4_STATE.md`.

### Blind scanning consumes time without information

- Layer: verification/state.
- Root cause: repeated scans changed power/span but did not target a remaining
  hypothesis or include a positive control.
- Prevention: list the hypothesis each run can falsify, require a control, stop
  when plausible QICK-side explanations are exhausted, and request new external
  evidence.

### Single-shot “optimum” is unstable or poor

- Layer: task specification/verification.
- Root cause: accepting one grid maximum without repeated plateau behavior, or
  failing to re-optimize after upstream qfreq/sigma/readout changes.
- Prevention: alternate length and gain/frequency sweeps, repeat points, choose
  the shortest plateau length, and validate in the normal round-robin chain.

### Shared config and per-run overrides disagree

- Layer: context/state.
- Root cause: `system_config.py`, `pucq4_config.py`, optimization arrays, and
  orchestration overrides can all contain different stages of calibration.
- Prevention: inspect the actual execution path, label starting points versus
  accepted results, and report propagation with old/new/unit/evidence.

## Recording a new failure

Use the diagnostic template in `TASK_TEMPLATE.md`. Include the failed command or
run path, evidence, responsible layer, harness change, and re-run outcome. If the
root cause is still uncertain, record bounded hypotheses and the next evidence
that would distinguish them; do not convert the best guess into a rule.
