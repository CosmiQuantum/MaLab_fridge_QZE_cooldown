# Repository map

Use this card to find a class, a config key or a label without opening the big files. Many scripts are 50 to 150 KB. Search them, never read them whole.

## Flow

```text
system_config.py + expt_config.py
        |
round_robin_benchmark.py  -->  section_*.py wrapper  -->  QICK program
        |
   result dictionaries  -->  Data_H5 (section_008_save_data_to_h5.py)
        |
<data_set>/study_data/Data_h5/<label>/*.h5      plots and notes in <data_set>/documentation/
```

`round_robin_benchmark.py` runs at module level. Importing it makes folders on `M:` and connects to QICK. Read it as text.

## Measurement catalog

`round_robin_benchmark.py` runs a measurement when its key in `run_flags` is true. Only `res_spec` is true by default. `PUCQ4_RUN_FLAGS` (comma list) replaces the whole set. A name in that list that is not a key is dropped without an error.

M5 and M6 need `q_spec` in the same pass, or `PUCQ4_QFREQ_MHZ`, for every pulse stage. The script stops otherwise.

| Measurement | `run_flags` key | File | Wrapper class | `expt_cfg` key | HDF5 label |
|---|---|---|---|---|---|
| Time of flight | `tof` | `section_001_time_of_flight.py` | `TOFExperiment` | `tof` | `tof` |
| Resonator spec | `res_spec` | `section_002_res_spec_ge_mux.py` | `ResonanceSpectroscopy` | `res_spec` | `res_ge` |
| Qubit spec | `q_spec` | `section_004_qubit_spec_ge.py` | `QubitSpectroscopy` | `qubit_spec_ge` | `qspec_ge` |
| Single shot (SSF) | `ss` | `section_005_single_shot_ge.py` | `SingleShot` | `Readout_Optimization` | `ss_ge` |
| Amplitude Rabi | `rabi` | `section_006_amp_rabi_ge.py` | `AmplitudeRabiExperiment` | `power_rabi_ge` | `rabi_ge` |
| Length Rabi | `len_rabi` | `section_006p5_length_rabi_ge.py` | `LengthRabiExperiment` | `length_rabi_ge` | `len_rabi` (saved in `optimization/`) |
| T1 | `t1` | `section_007_T1_ge.py` | `T1Measurement` | `T1_ge` | `t1_ge` |
| T2 Ramsey | `t2r` | `section_009_T2R_ge.py` | `T2RMeasurement` | `Ramsey_ge` | `t2_ge` |
| Ramsey frequency correction | `t2r_correction` | `section_009_T2R_ge.py` | `T2RMeasurement` with `correction=True` | `Ramsey_ge_correction` | `t2_ge_correction_1` |
| T2 echo | `t2e` | `section_010_T2E_ge.py` | `T2EMeasurement` | `SpinEcho_ge` | `t2e_ge` |

The qspec QICK program is `PulseProbeSpectroscopyProgram` in `section_004_qubit_spec_ge.py`. It must not be changed.

Other pieces.

| Need | Where |
|---|---|
| Punch out | `pucq4_04_punch_out.py` (standalone), class `PunchOut` in `section_003_punch_out_ge_mux.py`, and the round robin override `PUCQ4_RES_POWER_CHARACTERIZATION_MAX` |
| Readout gain and frequency search | `GainFrequencySweep.run_sweep` in `section_005_single_shot_ge.py` |
| Zeno versions | `section_004_qubit_spec_ge_zeno.py`, `section_006_amp_rabi_ge_zeno.py`, `section_007_T1_ge_IBM_zeno.py`, `section_007_T1_ge_zeno_loop.py`, `section_009_T2R_ge_zeno.py`, `section_010_T2E_ge_zeno.py`, run by the `longitudinal_qze_*.py` scripts |
| Stark shift maps | `QZEStyleResStarkShift2D` and `ResStarkShift2DAdapted` in `section_004_qubit_spec_ge.py`, `section_004_qubit_spec_ge_dual_stark_drive.py` |
| e-f and f-h levels | `section_002_res_spec_ef.py`, `section_004_qubit_spec_ef.py`, `section_006_amp_rabi_ef.py`, `section_005_single_shot_ef.py`, `section_007_T1_ef.py`, `section_006_amp_fh.py`, `section_004_qubit_spec_fh_V2.py`, `section_011_qubit_temperatures_efRabipt3.py` |
| Early PUCQ4 bring-up | `pucq4_00_check_board.py` to `pucq4_06_qspec_gain_calibration.py`. Standalone, with their own labels `punch_out` and `qspec_gain` |
| Controllers for a queue of runs | `pucq4_07_flux_campaign.py`, `pucq4_08_single_sweep_qspec_maps.py`, `pucq4_09`, `pucq4_10` and `pucq4_11_sigma_coherence_maps.py` run the round robin as a child per stage, for flux work on M5 and M6 only. `pucq4_12_m4_search.py` and `pucq4_13_avoided_crossing.py` take a JSON plan |
| Yoko readback without changing anything | `source_readbacks` in `pucq4_12_m4_search.py`. Read it before use |
| SSF optimisation as a child | `Combined_Optimization_Scriptsv2.py`, set with `PUCQ4_OPT_*` environment values |
| Q2 Zeno long run | `longitudinal_qze_005_ge_ramsey.py`. It runs at module level |
| Overnight Ramsey drift | `round_robin_ramsey_drift.py`, a copy of the round robin. All six rows at the 24 Nov working points, startup calibration, then Ramsey until 06.00 or until `documentation/STOP`. It refuses `PUCQ4_*` overrides. `--fast-hours` gives a short mode. It runs at module level. Results with `tools/analyze_overnight_ramsey.py` |
| Yokogawa driver | `malab_yokogawa_gs200.py`. Making the object opens a socket at once |
| Config builders | `add_qubit_experiment` in `build_task.py`, `all_qubit_state` in `build_state.py` |

Labels written by `round_robin_benchmark.py` are `tof`, `res_ge`, `qspec_ge`, `res_fh`, `rabi_ge`, `len_rabi`, `ss_ge`, `res_ef`, `qspec_ef`, `rabi_ef`, `rabi_fh`, `qspec_fh`, `q_temperatures`, `t1_ge`, `t2_ge`, `t2_ge_correction_1`, `rabi_ge_corrected`, `t2e_ge`, `ef_ss`, `SS_gef`, `test_act`.

Other scripts keep their own labels. The Zeno long run writes for example `Res`, `QSpec`, `Rabi`, `T1_ge_base` and `T1_ge_zeno`. `pucq4_12_m4_search.py` writes `Res_ge` and `QSpec_ge`.

By reading the code (not by running it), these run flags look out of order. `ss_gef`, `ef_ss`, `ef_Rabi`, `fh_rabi`, `test_act`, and `tof` together with `res_spec`. Check the block before you switch one on.

## Where each calibration value lives

Rows M1 to M4 come from `_PUCQ4_FIXED_CONFIG_ROWS` in `system_config.py`. Rows M5 and M6 come from the bias table `PUCQ4_FLUX_PULSE_CALIBRATION` in `system_config.py` (also named `PUCQ4_FLUX_CALIBRATION`), keyed `Yoko3_Q4_M5` and `Yoko4_Q6_M6`, then the current in mA. `resolve_pucq4_flux_config` joins the two.

| Value | Key for M1 to M4 | Key for M5 and M6 |
|---|---|---|
| Resonator frequency | `res_freq_ge` | `res_base_MHz` |
| Readout gain | `res_gain_ge` | `readout_gain` |
| Readout length | `res_length_ge` | `readout_length_us` |
| Readout offset | `res_freq_offset_ge` | `readout_offset_MHz` with `readout_frequency_MHz` |
| Qubit frequency | `qubit_freq_ge` | `qfreq_MHz` |
| Qspec drive gain and length | `qubit_gain_ge`, `qubit_length_ge` | `qspec_gain`, `qspec_length_us` |
| Gaussian sigma | `sigma` | `sigma_us` |
| Pi amplitude | `pi_amp` | `pi_amp` |
| Readout gain cap at zero bias | `PUCQ4_ZERO_CURRENT_READOUT_GAIN_MAX`, one value per row. Only the round robin checks it, only at zero bias, and it does not look at the length | same tuple |

Sweeps, reps, rounds and relaxation delays live in `expt_config.py`. The plain `expt_cfg` literals there are NOT what the round robin runs. The function `pucq4_round_robin_profile` in `expt_config.py` overrides them per row and bias.
It sets `qubit_spec_ge`, `power_rabi_ge`, `Readout_Optimization`, `T1_ge`, `Ramsey_ge` and `SpinEcho_ge`. For `res_spec`, `tof`, `length_rabi_ge` and `Ramsey_ge_correction` the literal is what runs.

There is no time step key for T2. The step is `(stop - start) / (steps - 1)`, and the detuning is `ramsey_freq`.

Order in which a value is decided at run time, last one wins.

1. Literal in `expt_config.py` or `system_config.py`.
2. `apply_pucq4_flux_experiment_config`.
3. `apply_pucq4_round_robin_profile` for the row.
4. A `PUCQ4_*` environment override.
5. A value fitted earlier in the same pass (qspec center, Rabi pi amplitude).

So always read what the run saved to learn what really ran. Sweeps, reps and delays are in `Exp Config`. Readout gain, length and tone, qubit frequency, sigma and pi amplitude are in `Syst Config`. Active overrides are in `acquisition_profile.json`.

`pucq4_config.py` holds older copies of some zero bias values (`RES_FREQS_MEASURED`, `RES_READOUT_OFFSETS`, `RES_GAIN`, `QUBIT_FREQS_MEASURED_0MA`). They differ from `system_config.py` for M2. The round robin reads `system_config.py`. Only the standalone `pucq4_NN` scripts read `pucq4_config.py`.

## Environment overrides of the round robin

The script has about 80 `PUCQ4_*` names. List them with a search for `PUCQ4_` in `round_robin_benchmark.py`. The common ones are below.

Every override is one value for all rows in `PUCQ4_QS`. Rows differ, so run one row per run. The study has no override. It is the literal `find_resonators_sandbox`.

| Group | Names |
|---|---|
| What runs | `PUCQ4_RUN_FLAGS`, `PUCQ4_QS` (rows), `PUCQ4_SUB_STUDY`, `PUCQ4_SUBSTUDY_NOTES` |
| Bias | `PUCQ4_YOKO3_CURRENT_MA`, `PUCQ4_YOKO4_CURRENT_MA` |
| Resonator scan | `PUCQ4_RES_START_MHZ`, `PUCQ4_RES_STEP_MHZ`, `PUCQ4_RES_STEPS`, `PUCQ4_RES_REPS`, `PUCQ4_RES_POWER_CHARACTERIZATION_MAX` |
| Readout point | `PUCQ4_READOUT_FREQ_MHZ`, `PUCQ4_READOUT_OFFSET_MHZ`, `PUCQ4_READOUT_GAIN`, `PUCQ4_READOUT_GAIN_MAX`, `PUCQ4_READOUT_LENGTH_US` |
| Qspec | `PUCQ4_QFREQ_MHZ`, `PUCQ4_QSPEC_START_MHZ`, `PUCQ4_QSPEC_STOP_MHZ`, `PUCQ4_QSPEC_HALF_SPAN_MHZ`, `PUCQ4_QSPEC_STEPS`, `PUCQ4_QSPEC_REPS`, `PUCQ4_QSPEC_GAIN`, `PUCQ4_QSPEC_LENGTH_US` |
| Rabi | `PUCQ4_SIGMA_US`, `PUCQ4_PI_AMP`, `PUCQ4_RABI_START`, `PUCQ4_RABI_STOP`, `PUCQ4_RABI_STEPS`, `PUCQ4_RABI_REPS`, `PUCQ4_RABI_ROUNDS` |
| SSF | `PUCQ4_SSF_SHOTS`, `PUCQ4_SSF_RELAX_DELAY_US`, `PUCQ4_SSF_DURATION_HOURS` |
| Coherence | `PUCQ4_T1_`, `PUCQ4_T2R_` and `PUCQ4_T2E_` with `START_US`, `STOP_US`, `STEPS`, `REPS`, `ROUNDS`, `RELAX_DELAY_US`. Also `PUCQ4_T2R_DETUNING_MHZ` and `PUCQ4_T2E_DETUNING_MHZ` |
| Saving | `PUCQ4_PHASE_CHECKPOINTS` (save after each stage). Works only with one row, one pass, `save_r` 1 and the standard stages, not with timed SSF |

Use `PUCQ4_READOUT_FREQ_MHZ` for the readout tone, not `PUCQ4_RES_FREQ_MHZ`. Do not add new names without the user's yes.

## Tools that are safe and reusable

| Tool | Use |
|---|---|
| `tools/harness_check.py` | Offline check of all Python files and of the harness itself |
| `tools/check_run_folder.py` | `plan` before a run, `check` after a run |
| `tools/check_config_rules.py` | Offline check of T2 sampling, relaxation delay and qspec step against the gates |
| `tools/estimate_ssf.py` | Offline estimate of SSF at another readout gain or length, and of the gain a target would need |
| `tools/check_res_spec_config.py` | Offline check that the profile keeps the resonator settings |
| `tools/inspect_qspec_h5.py` | Short raw IQ summary of one qspec HDF5 |
| `tools/inspect_ssf_optimization.py` | Short summary of SSF optimisation HDF5 files |
| `tools/analyze_qspec_iq_series.py` | Joint IQ diagnostics for saved qspec gain sweeps |
| `tools/ssf_progress.py` | Status file writes that cannot stop an acquisition |
| `tools/build_pucq4_first_light_ppt.py`, `tools/build_pucq4_chip_summary.py` | Deck builders |

The other `tools/build_*`, `tools/audit_*` and `tools/check_pucq4_*` files were made for one past report each. Read one only when you repeat that report.

## Cheap lookups

- A class. Search `^class ` in the one section file.
- What a run flag does. Search `run_flags[` with the key in `round_robin_benchmark.py`. Both quote styles are used.
- The folder block. Search `Data saving info` and `#Folders` in `round_robin_benchmark.py`.
- A config value. Search the key name in `system_config.py`. Skip `_PUCQ4_FLUX_CALIBRATION_HISTORY` at the top, it is not used at run time. The live table starts at `PUCQ4_FLUX_PULSE_CALIBRATION = {`. Search `def pucq4_round_robin_profile` in `expt_config.py`.
- The HDF5 writer is short. `section_008_save_data_to_h5.py` lines 10 to 140.
- Load names differ from save labels. `load_from_h5` wants names such as `Res_ge`, `QSpec_ge`, `T1_ge`, `T2`, `T2E`, and takes the path of one `.h5` file.
