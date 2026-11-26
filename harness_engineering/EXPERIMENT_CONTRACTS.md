# Code and data saving contract

Everything new must look like what is already here. Copy the nearest working example and change as little as possible. `REPOSITORY_MAP.md` tells you which example is nearest.

## 0. What to build

Take the first line that fits.

1. Run settings and documented overrides of an existing script. No new code.
2. An existing measurement repeated over a value that has an override. A controller (section 5). No new class, key or label.
3. A value that has no override. A standalone `pucq4_NN_<what>.py` script that reuses the wrapper class. Nearest example is `pucq4_12_m4_search.py`.
4. A new pulse sequence. A new `section_*.py` (section 3).

## 1. Folder structure

Every acquisition saves into this tree. The names and the number of levels are fixed.

```text
<data root>/<run_name>/<device_name>/<study>/<sub_study>/<data_set>/
    optimization/                    folder handed to QICK_experiment, pre-optimisation output, length Rabi
    study_data/Data_h5/<label>/      raw HDF5 batches, nothing else
    documentation/                   sub_study_notes.txt, RR_script.log, json run records, all plots
```

| Level | Variable in the scripts | Value now |
|---|---|---|
| data root | literal in `round_robin_benchmark.py`, `DATA_ROOT` in `pucq4_config.py` | `M:/_Data/20250822 - Olivia` |
| run | `run_name` | `pucq4_run_started_Oct_7` |
| device | `device_name` | `PUCQ4` |
| study | `study` | `find_resonators_sandbox` in the round robin, `qze_005` for Zeno |
| sub-study | `sub_study` | for example `rr6_0mA` |
| data set | `data_set` | time stamp from `strftime("%Y-%m-%d_%H-%M-%S")` |

Folder variables keep these exact names. `dataSetFolder`, `optimizationFolder`, `subStudyDataFolder` (the `study_data` folder), `studyDocumentationFolder` (the `documentation` folder).

Rules.

1. Do not build this tree by hand in a new script. A standalone script calls `pucq4_config.setup_data_folders(study, sub_study, substudy_txt_notes)` and uses the folder dictionary it returns. A round robin style script keeps the folder block of `round_robin_benchmark.py` unchanged.
2. `data_set` is always the time stamp. Never a fixed name, so a rerun can never write into an old folder.
3. One sub-study per sweep or per question. Not one per grid point. A repeat of the same thing is a new time stamp inside the same sub-study.
4. Name a sub-study so a person can read it. Pattern is what, rows, bias, date, for example `ssf_opt_m1_0mA_1124`. Up to about 40 characters fits under the path limit.
5. Reuse an existing study. In the round robin the study is the literal `find_resonators_sandbox` and has no override. Another study name is a code edit. Ask first.
6. Plots go under `documentation/` in the subfolder that the section class makes. Raw data goes under `study_data/Data_h5/`. The one other raw data place is `optimization/Data_h5/` (length Rabi and pre-optimisation). Never put a plot in `study_data` or raw data in `documentation`.
7. Extra files that explain a run (plans, reviews, status json, launcher and its log) go into that run's `documentation/` folder. Never leave the only copy in a temp folder or a private workspace.
8. Windows fails at 260 characters. Before launch run `python tools/check_run_folder.py plan --study <study> --sub-study <sub_study>`. `round_robin_benchmark.py` has its own guard for this. Keep it.
9. Never rename, move or rewrite an existing run folder or an old path in a document.

## 2. Raw data

- Raw data is saved only through `Data_H5` from `section_008_save_data_to_h5.py`.

```python
saver = Data_H5(subStudyDataFolder, data_dict, batch_num, save_r)
saver.save_to_h5('t1_ge')
del saver
```

- That writes `study_data/Data_h5/<label>/<time stamp>_<label>_results_batch_<batch_num>_Num_per_batch<save_r>.h5` with one group `Q<index + 1>` per qubit.
- Make a new `Data_H5` object for every save. A second save on the same object nests the folders.
- The dictionary comes from `create_data_dict(keys, save_r, list_of_all_qubits)`. Every key list holds `Dates`, `Round Num`, `Batch Num`, `Exp Config` and `Syst Config`. `Exp Config` is `copy.deepcopy(expt_cfg)`. `Syst Config` is the config that `run()` returned.
- Use the labels that exist (list in `REPOSITORY_MAP.md`). A new label needs the user's yes and is added to that list.
- Standalone `pucq4_NN_*.py` scripts use `pucq4_config.create_data_dict` and `pucq4_config.save_h5`. Two differences to know.
  - `pucq4_config.create_data_dict` takes a qubit count as third argument, where the round robin one takes a list.
  - `pucq4_config.save_h5` stores arrays as numbers, where the round robin stores the default text form. Readers differ, so never mix both forms under one label in one sub-study.
- Never add another raw format. No `np.save`, `np.savez`, CSV, pickle or JSON for acquired data, and no direct `h5py.File` writes. An analysis export in another format is fine when the user asks for it and it names its source HDF5.
- Save raw data before any fit. A fit error must never lose acquired data.
- Never overwrite or delete raw data.

## 3. Section classes (`section_*.py`)

A new pulse sequence is a copy of the nearest section file of the same type. Keep what that file does.

- Keep the low level `AveragerProgramV2` program class apart from the wrapper class. Do not edit existing program classes.
- The wrapper has `__init__` and `run`. Hardware is touched only inside `run`.
- Keep the constructor of the class you copied. The classes differ. Qspec, length Rabi, T1, T2R and T2E use `QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_figs, experiment`, then keyword options. With no model use that order. Pass everything after `round_num` by keyword. `QubitIndex` is zero based.
- Store the arguments on `self`. Set `self.expt_name` to a key of `expt_cfg`. Build the config the way `section_007_T1_ge.py` does, with `all_qubit_state` and `add_qubit_experiment`.
- A new `expt_cfg` key also needs its sweep branch in `build_task.py`. Without it the sweep is missing.
- Use the passed `experiment.readout_cfg` and `experiment.qubit_cfg`. No second device dictionary and no path literal inside a section class.
- `run` returns its results with `self.config` in the same place as the class you copied. The orchestrator stores that as `Syst Config`.
- Plots are inside `if self.save_figs`. Keep the folder and file name code of the class you copied. Each class has its own folder, for example `res_spec_ge_plots`, `qubit_spec_ge_plots`, `ss_repeat_meas_ge/Q<n>`, `T1_ge`. With no model use the T1 form, folder `os.path.join(self.outerFolder, self.expt_name)` and file `R_<round>_Q_<index + 1>_<time stamp>_<expt_name>_q<index + 1>.png`.
- A section class does not save HDF5. It returns data and the orchestrator saves it.
- Keep the calibrated population y axis on qspec, Rabi, T1, T2R and T2E plots. It comes from `run(scaling=True)`, which also changes what `run` returns. For T1 that is 13 values with the config as the seventh.

## 4. Orchestration scripts

Keep the phase order of `round_robin_benchmark.py`.

1. Imports of the wrappers, `Data_H5`, `QICK_experiment`, `expt_config` values.
2. Run settings. `run_flags`, chosen rows, `n`, save and plot switches, notes, bias.
3. The folder block from section 1, then `sub_study_notes.txt` and the `RR_script.log` logger.
4. Key lists and `create_data_dict` dictionaries.
5. The loop over rounds and rows. Call `apply_pucq4_flux_experiment_config` once and `apply_pucq4_round_robin_profile` per row before a wrapper is built. Build `QICK_experiment(optimizationFolder, ...)`, pick the row values, check `run_flags`, build the wrapper with `studyDocumentationFolder`, call `run`, collect results with `Exp Config` and `Syst Config`, delete the objects.
6. Batch save with `Data_H5`, then rebuild the dictionaries.

Adding a measurement to the round robin needs five things. A `run_flags` key, a key list, its dictionary in both `create_data_dict` blocks (start and rebuild), a collect block and a save block.

## 5. Controllers for a queue of runs

A controller starts existing scripts one after another. It holds no acquisition, fitting or saving code of its own.

- For a few runs, start single row runs by hand, one after another. That needs no controller.
- The existing controllers `pucq4_07_flux_campaign.py` to `pucq4_11_sigma_coherence_maps.py` serve only flux work on M5 and M6, and they point at the old `pucq4_first_light` paths. Do not reuse them as they are.
- What to copy from them is the child launch. `activated_subprocess_env` and `run_round_robin` in `pucq4_07_flux_campaign.py` start `round_robin_benchmark.py` as a child with environment overrides, the Conda DLL folders and a log. `pucq4_08_single_sweep_qspec_maps.py` to `pucq4_11` add a JSON manifest, `--plan-only`, and a rerun that skips finished points.
- Do not copy their sub-study line. They set a new `PUCQ4_SUB_STUDY` for every grid point and stage. A new controller sets one sub-study for the whole sweep.
- The guard pattern is `pucq4_13_avoided_crossing.py` with `tools/pucq4_crossing_guard.py`. A checked JSON plan, a deadline, a `STOP` file, and sources parked at the end.
- A new controller is a numbered `pucq4_NN_<what>.py` in the repo root, kept in git. It first calls `pucq4_config.setup_data_folders`. Its own time stamp folder holds the child logs, the manifest and the plots across points in `documentation/`. Its stop file is `documentation/STOP`. It has a deadline.
- Throwaway debug scripts are deleted when the question is answered. They are never committed with a `_tmp` name.

## 6. Do not copy from these

They work, but they do not follow the contract.

| File | Why not |
|---|---|
| `Combined_Optimization_Scriptsv2.py` | in use for SSF optimisation, but it builds its own folders and writes HDF5 with `h5py` directly |
| `sweep_ADC_attenuation.py`, `punch_out_test.py` | hand built folders and missing levels. The first also writes with `h5py` directly |
| `round_robin_benchmark_qspec_simple.py`, `round_robin_benchmark_res_spec_simple.py` | old run and device names, old key lists |
| `longitudinal_qze_*.py`, `active_reset_test_*.py` | old run and device names, or a `data_set` that is not a plain time stamp |
| `pucq4_08_cross_readout_qspec.py`, `_tmp_m4_readout_debug.py` | hand built roots, no notes file |
| `pucq4_13_avoided_crossing.py` | good guard pattern, but it builds its own folder tree and log name |
| `section_003_punch_out_ge_mux.py`, `section_006p5_length_rabi_ge.py` | save HDF5 or use a fixed old plot folder inside the class |

Changing one of these on request is fine. Keep its structure, and do not spread its pattern.

The run folder check fails by design on folders written by `Combined_Optimization_Scriptsv2.py` and by the `longitudinal_qze_005` scripts. Record that as expected. Never move, rename or add files to make it pass.

## 7. Reporting a config change

For every changed calibration value give file and key, old value to new value, unit, row with index and physical qubit, bias, the run path that supports it, and what must be measured again.
