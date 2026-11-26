# Verification

Use the highest step you are allowed to use, and never skip the lower ones. In the final message say which steps passed and which did not run.

## Steps

| Step | What | Command or action |
|---|---|---|
| 0 | State and brief | Read `STATE.md`. Goal, non-goals, row and index, units and acceptance are written down. `git status` looked at. |
| 1 | Offline check | `python tools/harness_check.py` |
| 2 | Diff review | `git diff --check` and `git diff -- <changed files>`. No stray change to run flags, frequency, gain, attenuation, path or raw data. |
| 3 | Pure logic | Test a pure function with made up input. Never import an acquisition script to reach a helper. |
| 4 | Saved data | For analysis of an existing run. Name the HDF5 path and label, check axes and units, compare the fit with the acquired range and the raw trace. |
| 5 | Hardware | Only inside the envelope. Then `python tools/check_run_folder.py check "<data_set folder>" --expect "<labels>"` in the Conda form, and compare the saved `Exp Config` and `Syst Config` with what you asked for. |

## What the offline check does

`tools/harness_check.py` parses every Python file as Python 3.9 without importing it. It finds syntax errors and merge markers. It checks new acquisition scripts against the saving contract. It checks the harness files for size, plain ASCII, broken file references, superseded phrases and the log entry format. It contacts no hardware.

It does not validate a pulse program, a dependency, a fit or a plot.

## What the run folder check does

`tools/check_run_folder.py check` verifies the three subfolders, that raw data sits only under `study_data/Data_h5/<label>/`, the file names, path lengths, the notes file, that plots exist, and (when `h5py` is installed) that the HDF5 files hold real arrays and the standard keys.

It needs `h5py`, so run it in the Conda form. Without `h5py` it ends with `CHECK INCOMPLETE` and that does not count as a pass. A NOTE about missing standard keys must be looked at. Folders written by `Combined_Optimization_Scriptsv2.py` and the `longitudinal_qze_005` scripts fail this check by design.

It does not judge the science. The gates in `MEASUREMENT_PLAYBOOK.md` do that.

## What the config rule check does

`tools/check_config_rules.py` reads the sweep settings the round robin would use and reports every row whose T2 sampling, relaxation delay or qspec step misses a gate. It is offline. It cannot see environment overrides or values fitted during a run, so the saved `Exp Config` stays the truth after a run.

## Words to use

- "Offline check passed" and "validated on hardware" are different claims. Never use one for the other.
- "Measured" means raw data was saved and looked at. "Validated" means it also passed every gate with a repeat.
- A result that depends on a permission or on evidence you do not have is "pending".
