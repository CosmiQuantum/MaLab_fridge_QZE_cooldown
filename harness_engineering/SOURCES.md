# Sources and evidence labels

## Harness-design sources

- Walking Labs, “Lecture 01. Strong Models Don't Mean Reliable Execution”:
  https://walkinglabs.github.io/learn-harness-engineering/en/lectures/lecture-01-why-capable-agents-still-fail/
  (read 2026-08-28). Rules implemented here: make requirements explicit, expose
  implicit conventions, stabilize the environment, provide verification, persist
  cross-session state, diagnose failures by layer, and use a command-verifiable
  Definition of Done.
- OpenAI, “Custom instructions with AGENTS.md”:
  https://developers.openai.com/codex/agent-configuration/agents-md
  (read 2026-08-28). The root `AGENTS.md` is the always-loaded entry point;
  project instruction discovery is hierarchical and has a default combined size
  limit, so detailed routed context lives in this folder.

## Device and driver sources

- Local characterization deck:
  `C:\Users\Ma Quantum Lab\Downloads\PUCQ4 Initial Characterization (1).pptx`.
  The extracted copy under `_ppt_extract/` is a temporary working artifact, not a
  versioned source of truth. The deck maps physical Q4/D4 to M5 and physical
  Q6/A5 to M6, gives M4 only a `5575 MHz` seed with unknown physical number, and
  shows current axes from -10 to +10 mA.
- Upstream GS200 interface:
  https://github.com/ma-quantumlab/malab/blob/main/instruments/voltsource.py
  The local `malab_yokogawa_gs200.py` keeps the relevant public interface with a
  standard-library socket implementation and added current guards.
- Executable repo sources: `system_config.py`, `expt_config.py`,
  `round_robin_benchmark.py`, `pucq4_config.py`, the relevant `section_*.py`, and
  `section_008_save_data_to_h5.py`.
- Chronological run/failure evidence: `docs/harness-log.md`. It contains exact
  `M:` output paths and diagnostic settings; raw measurements remain outside Git.

## Evidence labels

Use one of these labels whenever a number can be mistaken for calibration:

- `executable-config`: the value code will currently use after per-run overrides;
  not automatically validated.
- `validated-measurement`: reproducible run with bounds/configuration, accepted
  plot/fit quality, and persisted raw data unless an explicit save limitation is
  stated.
- `deck-seed`: characterization/reference value used to choose a search region.
- `rejected-candidate`: observed fit/fluctuation that failed repeatability,
  bounds, control, or physical-quality gates.
- `acceptance-target`: desired result such as qspec linewidth or Rabi pi window;
  not a measured value.

For a validated measurement, record date, physical/index identity, Yoko state,
sweep and averaging, attenuation/readout/pulse config, fitted value and units,
quality/repeat evidence, and raw/plot paths. If any field is unavailable, state
the limitation instead of filling it from a different run.
