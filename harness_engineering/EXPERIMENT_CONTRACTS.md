# Experiment, plotting, and data contracts

## New or changed `section_*.py` modules

Copy the nearest experiment of the same type. Do not introduce a second
framework.

- Keep the low-level `AveragerProgramV2` class separate from the high-level
  experiment/measurement wrapper.
- The high-level class exposes both `__init__` and `run`; acquisition stays in
  `run`, never at class-definition time.
- Preserve constructor vocabulary where applicable: `QubitIndex`,
  `number_of_qubits`, `outerFolder`, `round_num`, `signal`, `save_figs`, and
  `experiment`. Store the passed values on `self`.
- Set `self.expt_name`; specialize settings through `expt_cfg[self.expt_name]`
  or the existing `add_qubit_experiment(...)` path used by the nearest class.
- Consume the passed `QICK_experiment.readout_cfg` and `.qubit_cfg`. Do not
  hard-code a replacement device dictionary inside the class.
- Preserve the nearest class's return shape so the round-robin collector and
  HDF5 keys still match.
- Gate every plot save with `self.save_figs`. Build paths from
  `self.outerFolder` with `os.path.join` and the existing plot-folder helper.
- Preserve filename provenance: round, one-based displayed qubit, timestamp,
  and experiment name where the analogous class includes them.
- A section class returns results/configuration; it does not independently save
  the round-robin HDF5 batch.

The qspec low-level program is the existing `PulseProbeSpectroscopyProgram` in
`section_004_qubit_spec_ge.py`. Calibration tasks may change config, sweep, fit,
or wrapper plotting as authorized, but must not silently replace that QICK pulse
program.

## Round-robin orchestration

Retain the established phase order:

1. Imports for experiment wrappers, `Data_H5`, `QICK_experiment`, and
   `expt_config` values.
2. Run controls: `run_flags`, selected indices, `n`, save/plot behavior,
   metadata, attenuators, and per-device overrides.
3. Folder hierarchy:
   `<data root>/<run>/<device>/<study>/<sub-study>/<timestamp>/`, containing
   `optimization/`, `study_data/`, and `documentation/`.
4. `sub_study_notes.txt` and the isolated round-robin logger under
   `documentation/`.
5. Per-experiment key lists and object-array dictionaries from
   `create_data_dict`.
6. Round/qubit loop: create `QICK_experiment`, select the row-specific readout
   and qubit values, gate each experiment with `run_flags`, call `run`, collect
   results plus `Exp Config` and `Syst Config`, then release objects.
7. Batch save with `Data_H5(subStudyDataFolder, ..., batch_num, save_r)`, using
   established labels, followed by dictionary reinitialization.

For the final characterization loop, `n=1` means one pass. It does not make a
diagnostic set of `run_flags` correct: all flags, selected rows, and sub-study
metadata must still be reviewed explicitly.

## Standalone PUCQ4 acquisition scripts

Standalone numbered `pucq4_*.py` scripts are their own orchestrators. They must:

- use `pucq4_config.setup_data_folders` or construct the exact same hierarchy;
- use `pucq4_config.create_data_dict` and `save_h5`/`Data_H5`;
- retain `Dates`, `Round Num`, `Batch Num`, `Exp Config`, and `Syst Config`;
- put raw data below `subStudyDataFolder/Data_h5/<dataset label>/`;
- put plots, logs, and notes in `documentation/`;
- avoid NPZ/CSV as a replacement raw-acquisition format.

Analysis-only exports may use another format when explicitly requested, but
must cite the source HDF5 and must not overwrite it.

## Windows artifact-path contract

This lab computer has failed when a generated HDF5 path exceeds 260 characters.
Pre-compute the longest complete filename, including:

`<timestamp>_<dataset>_results_batch_<n>_Num_per_batch<n>.h5`

Shorten only descriptive sub-study/dataset labels before acquisition. After the
run, verify both the expected plot and HDF5 file. A plot alone is not evidence
that raw data persisted.

## Config and plot handoff

For every changed calibration value, report:

- file and key;
- old value -> new value;
- physical qubit, resonator row, and Python index;
- units and experiment affected;
- run path/evidence used;
- whether downstream Rabi, SSF, or coherence must be revalidated.

Keep final round-robin plots on the established calibrated-population y-axis for
qspec, Rabi, T1, T2R, and T2E. Diagnostic plots belong in a separate temporary
script/folder and do not replace the default saved plots.
# Report and PowerPoint plot layout

- Never set both an arbitrary width and height when inserting a saved plot into
  a PowerPoint. Preserve the plot's original aspect ratio so axes, labels, and
  text are not warped.
- Fit each plot inside its allotted slide cell and center it, leaving whitespace
  on the unconstrained dimension when the source aspect ratio differs.
- Reusable report generators must enforce this behavior programmatically; do
  not rely on a later manual PowerPoint correction.
