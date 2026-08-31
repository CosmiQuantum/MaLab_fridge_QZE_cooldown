# Repository map and context

## Control and data flow

```text
system_config.py + expt_config.py
              |
              v
round_robin_benchmark.py  --selects-->  section_*.py high-level wrapper
              |                              |
              |                              v
              |                        AveragerProgramV2 / QICK
              v
     result dictionaries
              |
              v
section_008_save_data_to_h5.py::Data_H5
              |
              v
<timestamp>/study_data/Data_h5/<dataset label>/*.h5

Plots and notes --------------------------------> <timestamp>/documentation/
Optimization artifacts ------------------------> <timestamp>/optimization/
```

`round_robin_benchmark.py` is the primary orchestration template, but it has
module-level execution. Importing it can create folders, connect to lab services,
and begin a long run. Inspect it as text; never import it for an offline check.

## Main components

- `system_config.py`: `QICK_experiment`, hardware channels, readout config, and
  qubit config. Lists are generally in resonator-row/Python-index order.
- `expt_config.py`: sweep bounds, points, reps, rounds, delays, and experiment
  registry. `add_qubit_experiment`/the established builder specializes list
  values for one index.
- `section_001...section_011...py`: low-level QICK program plus high-level
  experiment wrapper. The wrapper's `run` method is the round-robin interface.
- `build_state.py` and `build_task.py`: established config-specialization helpers;
  they import QICK and are not needed by the syntax-only verifier.
- `section_008_save_data_to_h5.py`: canonical `Data_H5` writer. It adds
  `Data_h5/<data_type>/` below the folder passed by the orchestrator.
- `round_robin_benchmark.py`: complete phase ordering, result collection,
  calibrated plotting, and batch saving. Its current values may be a diagnostic
  setup rather than a reusable full benchmark; inspect every run control.
- `pucq4_config.py`: PUCQ4 constants and folder/data helpers for standalone
  `pucq4_*.py` orchestrators.
- `pucq4_00...pucq4_06...py`: board check and early characterization utilities.
  They are hardware scripts, not offline tests.
- `Combined_Optimization_Scriptsv2.py`: alternating single-shot readout length
  and gain/frequency optimization. Its arrays are starting points until a run is
  accepted and propagated.
- `analysis_*.py` and `qze_nbar_analysis/`: analysis paths. Confirm whether an
  individual script has module-level I/O before running it.
- `malab_yokogawa_gs200.py`: local standard-library copy of the MaLab GS200
  interface. Instantiating the class opens a socket immediately.
- `docs/harness-log.md`: chronological evidence and failed-approach log.
- `tools/harness_check.py`: safe static verifier; parses Python without imports.

## Indexing and naming

Never use “Q4” alone in a task or handoff. State all relevant identities:

- physical qubit number from the chip documentation;
- resonator row `M1` through `M6`;
- Python index `0` through `5`;
- displayed filename/UI number, usually `Q{index + 1}`.

For PUCQ4, physical Q4 is on M5/Python index 4, while displayed Q4/Python index
3 is M4 and has no validated physical-qubit assignment. The detailed table is
in [PUCQ4_STATE.md](PUCQ4_STATE.md).

## Configuration ownership

- Experiment classes consume the passed `QICK_experiment` object's
  `readout_cfg` and `qubit_cfg`; do not add a parallel device-config system.
- Generic sweep shapes live in `expt_config.py`.
- Shared device values live in `system_config.py`.
- Standalone PUCQ4 defaults/helpers live in `pucq4_config.py`, but accepted
  round-robin values must be propagated deliberately to the shared configs.
- Per-run overrides in an orchestrator win at execution time. Always inspect
  them before claiming that a shared config value was used.
- Comments and deck values are context, not measured truth. Use the evidence
  labels defined in `SOURCES.md`.
