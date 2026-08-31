# Harness task and diagnostic log

Use one short entry per agent-assisted task. This file is persistent memory for
facts that should survive chat sessions; do not put credentials or raw data here.

## Task template

### YYYY-MM-DD — task name

- Goal:
- Non-goals:
- Files in scope:
- Acceptance criteria:
  - [ ] Observable behavior or output is specified.
  - [ ] New experiment code follows the nearest existing `section_*.py` class,
        `run`, config, return-value, and plot-saving conventions.
  - [ ] New orchestration code retains the round-robin folder hierarchy,
        configuration flow, result collection, and `Data_H5` batch-saving pattern.
  - [ ] `tools/harness_check.py` passes.
  - [ ] Focused diff was reviewed.
  - [ ] Hardware validation was run, or explicitly marked not run.
- Hardware run authorized: no
- If authorized: device, qubits, run flags, sweep bounds, attenuations, output path,
  expected duration, and abort condition:
- Result:
- Verification evidence:
- Follow-up:

## Failure template

Record failures under the task that exposed them.

- Symptom:
- Layer: task specification / context / environment / verification / state
- Root cause:
- Harness change that prevents recurrence:
- Re-run result:

## Initial repository observations

- `round_robin_benchmark.py` executes at import time, writes under `M:`, and can run
  up to a large configured round count. It must not be used as an offline test.
- The repository currently has no dependency manifest or automated test suite.
- Offline syntax parsing is therefore the first verification rung; simulation,
  unit tests around extracted pure logic, and explicit hardware smoke tests are
  useful later rungs rather than claims made by the initial harness.

### 2026-08-27 — PUCQ4 resonator scan and script alignment

- Goal: align PUCQ4 characterization scripts with the standard class/config/save
  conventions and confirm all six resonators are visible.
- Hardware run authorized: yes, by the user.
- Run: `pucq4_02_res_spec.py`; 8900–9080 MHz, 0.2 MHz step, gain 0.9,
  10 us integration, 1000 reps, one round; control 7140–7300 MHz.
- Result: six of six dips resolved at 8919.4, 8951.6, 8974.8, 9000.0,
  9014.4, and 9058.8 MHz. Depths were 58.3–97.0%.
- Output: `M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/`
  `pucq4_first_light/res_spec/2026-08-27_17-01-14/`.
- Limitation: coarse, high-power centers; a ~0.02 MHz low-power refinement is
  still required before treating them as dressed readout frequencies.

### 2026-08-27 — Q6 at zero flux current and HDF5 migration

- Goal: set Q6's Yoko 4 bias to 0 A, estimate Q6 from the characterization
  deck, locate the transition, and migrate standalone PUCQ4 acquisition output
  to the round-robin HDF5 structure.
- Hardware run authorized: yes, by the user.
- Bias evidence: Yokogawa GS210 `90Z114062` at `192.168.1.77`; current mode,
  10 mA range, programmed level 0.000000000 A, output off after ramp/verify.
- PowerPoint expectation: Q6 is approximately 6.5-6.6 GHz at 0 mA.
- Coarse run: M6 readout at 9058.8 MHz; 6200-6800 MHz drive, 4 MHz nominal
  spacing, gain 0.5, 10 us pulse, 16,384 averages; candidate 6522.1 MHz.
- Refinement: 6480-6560 MHz, 1 MHz nominal spacing, same drive and averaging;
  center 6523.5 MHz with a depressed response across neighboring points.
- Result: record Q6 f_ge at 0 mA as approximately 6523 MHz. A lower-power fitted
  spectroscopy run is still appropriate for a precision linewidth/center.
- Save-format change: standalone PUCQ4 raw acquisition now uses
  `study_data/Data_h5/<dataset>/` via `Data_H5`, including dates, batch/round,
  experiment configuration, and system configuration. NPZ raw saving removed.

### 2026-08-27 — PUCQ4 zero-current qubit search and benchmark setup

- Goal: keep both known Yokogawa sources at 0 A, locate all six transitions,
  retain PUCQ4 data in the round-robin hierarchy, and configure the benchmark
  for resonator and qubit spectroscopy only.
- Hardware run authorized: yes, by the user.
- Bias evidence: Yoko 3/Q4 (`192.168.1.73`) and Yoko 4/Q6
  (`192.168.1.77`) were verified in current mode at 0.000000000 A, 10 mA
  range, with outputs off.
- Measured zero-current centers, in readout-mode order: M1 5515.3 MHz,
  M2 5439.6 MHz, M3 5557.5 MHz, M5 7083.6 MHz, and M6 6523.5 MHz.
  Coarse and fine scans agreed for M2, M3, M5, and M6; M1's fine scan had a
  less stable baseline and should receive a lower-power precision fit.
- Unresolved: M4 produced no reproducible line in 5.37–5.77 GHz at gain 0.5,
  5.45–5.70 GHz at gain 1.0, or a 0.25 MHz repeat around a one-point artifact.
  Its benchmark start remains the deck estimate of 5575 MHz and is explicitly
  not included in `QUBIT_FREQS_MEASURED_0MA`.
- Benchmark setup: all six modes selected; only `res_spec` and `q_spec` enabled;
  resonator centers `[8919.4, 8951.6, 8974.8, 9000.0, 9014.4, 9058.8]` MHz;
  slope offsets `[-0.4, -0.4, -0.2, 0.4, 0.2, 0.2]` MHz; qubit starting centers
  `[5515.3, 5439.6, 5557.5, 5575.0, 7083.6, 6523.5]` MHz.
- HDF5 correction: `pucq4_03_qubit_hunt.py` now stores acquired I and Q arrays
  instead of empty placeholders.
- Hardware validation of `round_robin_benchmark.py`: not run; setup only, because
  importing or executing it starts the full hardware/data loop.

### 2026-08-27 — PUCQ4 low-power calibration and coherence (M4 excluded)

- Goal: choose the lowest repeatable spectroscopy gain giving roughly
  0.5–1 MHz FWHM, calibrate a 0–1 amplitude-Rabi pi pulse, and measure T1,
  Ramsey T2*, and echo T2 for the validated PUCQ4 modes. M4 was explicitly
  excluded at the user's direction.
- Hardware run authorized: yes, by the user. Both Yokogawas remained at 0 A
  with their outputs off; no bias changes were made during these runs.
- Repeatable spectroscopy results (two repeats each), reported as center MHz,
  drive gain, and the two fitted FWHM values in MHz: M1 5517.6834, 0.03,
  0.586/0.533; M2 5438.7153, 0.05, 0.573/0.520; M3 5557.3371, 0.08,
  0.963/0.967; M5 7084.5557, 0.05, 0.779/0.710; M6 6522.4983, 0.03,
  0.615/0.597.
- Spectroscopy output: `M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/`
  `PUCQ4/pucq4_first_light/qspec_gain_repeatability_0mA/2026-08-27_19-24-42/`.
- Rabi calibration: sigma 0.10 us for all calibrated modes; pi amplitudes M1
  0.3492950, M2 0.6486907, M3 0.7983886, M5 0.3692547, M6 0.4091741 DAC
  units. The sweep was 0–1 gain, 101 points, 200 reps, one round.
- Primary coherence run: one round on M1, M2, M3, M5, M6; T1 0.01–500 us,
  200 points, 300 reps; Ramsey/echo 0–150 us, 100 points, 100 reps. Output:
  `M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/`
  `pucq4_first_light/t1_t2r_t2e_0mA/2026-08-27_19-39-42/`.
- Reliable T1 values (us): M1 21.97 ± 1.15, M2 8.85 ± 1.54,
  M3 33.70 ± 2.82, M5 13.59 ± 1.22, M6 5.28 ± 1.38.
- Reliable Ramsey T2* values (us): M1 15.08 ± 3.74, M2 13.63 ± 3.15
  (0–40 us refinement), M3 41.43 ± 9.41, M5 13.42 ± 5.36.
  M6 remained unresolved in both 0–40 us and 0–5 us/4 MHz/400-rep traces;
  its optimizer returned nonphysical or unconstrained values. Its 0.60 MHz
  spectroscopy linewidth suggests an order-of-magnitude T2* near 0.5 us, but
  that is not recorded as a time-domain measurement.
- Reliable echo T2 values (us): M1 31.92 ± 5.60, M3 49.76 ± 11.96,
  M5 21.82 ± 7.43, M6 6.22 ± 2.59 (0–40 us refinement). M2 echo remained
  noise dominated; the initial 17.54 ± 6.76 us fit is not accepted as reliable.
- Refinement output: `M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/`
  `PUCQ4/pucq4_first_light/t2r_t2e_m2_m6_refine_0mA/2026-08-27_20-10-13/`
  and `t2r_m6_short_refine_0mA/2026-08-27_20-31-09/`.
- Harness result: the round-robin remains configured for one T1/T2*/T2 run on
  `[0, 1, 2, 4, 5]`; all other experiment flags are false. The standard data
  hierarchy and `Data_H5` batching were retained.

### 2026-08-27/28 — PUCQ4 ringing removal and one-pass characterization

- Hardware run authorized: yes. Both Yokogawas remained at 0 A with outputs
  off; no bias changes were made.
- Root cause of the qspec ringing: coherent sidelobes from the 20 us rectangular
  spectroscopy pulse at the former 0.03-0.08 gains. The QICK qspec program was
  not changed. Clean minimum working gains were M1 0.001, M2 0.010, M3 0.002,
  M5 0.001, and M6 0.002 DAC units.
- Final qspec centers: M1 5517.70, M2 5438.73, M3 5557.34, M5 7084.50, and
  M6 6522.57 MHz. Output folders: `res_qspec_low_gain_0mA/2026-08-27_23-02-21`,
  `qspec_low_gain_refine_0mA/2026-08-27_23-13-16`, and
  `qspec_low_gain_refine_0mA/2026-08-27_23-18-20`.
- Refined resonator centers: M1 8919.850, M2 8951.650, M3 8975.000,
  M4 8999.950, M5 9014.700, and M6 9059.450 MHz.
- Sigma values M1-M6 are 0.050, 0.093, 0.114, 0.100, 0.053, and 0.058 us.
  Validated pi amplitudes are M1 0.6886101, M2 0.7385094, M3 0.7085699,
  M5 0.7185497, and M6 0.7584691 DAC units; M4 remains the uncalibrated 0.7
  placeholder. Rabi output: `rabi_pi_near_0p7_0mA/2026-08-27_23-20-27`.
- High-resolution coherence output:
  `coherence_high_resolution_0mA/2026-08-27_23-28-08`. T1 values M1, M2,
  M3, M5, M6 were 22.69, 9.58, 36.15, 5.89, and 5.08 us. Ramsey T2* values
  M1, M2, M3, M5 were 22.54, 15.12, 22.22, and 6.24 us. Echo T2 values M1,
  M2, M3, M5, M6 were 37.24, 16.35, 52.95, 12.11, and 4.05 us.
- M6 Ramsey required a dedicated 0-10 us, 201-point, 2 MHz detuning,
  1000-rep run and gave T2*=1.57 us. Output:
  `t2r_m6_short_high_resolution_0mA/2026-08-27_23-36-56`.
- M4 was not accepted. A 5375-5775 MHz search and two repeats near a candidate
  at 5702 MHz produced noise rather than a reproducible line. In the final
  10000-rep trace the fitter returned 5686.33 MHz, outside the acquired
  5700.47-5704.47 MHz interval. M4 therefore remains `NaN` in measured results
  and uses 5575 MHz only as a deck-derived search seed.
- Final benchmark state: `n=1`; res spec, qspec, Rabi, T1, Ramsey, and echo are
  enabled for `[0, 1, 2, 4, 5]`. Default plotting and `Data_H5` saving remain
  unchanged. Single shot is disabled because all six qubits were not
  successfully calibrated.

### 2026-08-28 — Q3/Q6 fit repair and physical-Q4 mapping

- Both Yokogawas were reverified in current mode at exactly 0 A, 10 mA range,
  with outputs off before acquisition.
- The scaled qspec analysis incorrectly used `argmax`'s array index as a MHz
  initial guess and permitted negative linewidths. It now uses the acquired
  frequency and bounds the fitted center to the sweep with positive width. The
  underlying QICK spectroscopy program was not changed.
- Q3 was reconfirmed at 5557.3278 MHz. Its qspec plot is clean and the fit no
  longer reports negative FWHM.
- Q6 required a +/-25 MHz search at gain 0.01 and 5000 reps. It was confirmed
  at 6522.4767 MHz with 0.3 MHz FWHM. Output:
  `q6_wide_qspec_confirmation_0mA/2026-08-28_08-41-41`.
- PowerPoint mapping correction: physical Q4/DC line D4 is connected to M5;
  physical Q6/A5 is connected to M6. M4 is an unassigned resonator row, not the
  physical Q4. A disposable diagnostic using the standard repository classes
  measured M5 at 9014.950 MHz and physical Q4 at 7084.3641 MHz with 0.165 MHz
  FWHM, using the +0.20 MHz readout offset. The diagnostic script was deleted
  after use. Output: `temporary_physical_q4_m5_debug_0mA/2026-08-28_08-30-39`.
- Downstream output for Q3, physical Q4/M5, and Q6/M6:
  `q3_q4_q6_downstream_validation_0mA/2026-08-28_08-48-42`.
- Clean fitted values: Q3 pi 0.68861, T1 42.90 us, T2* 29.57 us, echo T2
  89.27 us; physical Q4/M5 pi 0.67863, T1 10.01 us, T2* 4.97 us, echo T2
  11.30 us; Q6 T1 3.06 us. Single-shot fidelities were only 0.247, 0.262, and
  0.106 respectively, so readout discrimination is not yet optimized.
- Q6's 50 us T2 traces were rejected as noise-dominated. With sigma shortened
  from 0.058 to 0.052 us, its pi amplitude became 0.71855. Targeted 0-10 us,
  201-point, 2 MHz, 1000-rep measurements gave clean T2*=1.38 us and echo T2
  5.65 us. Output: `q6_rabi_short_t2_refinement_0mA/2026-08-28_09-05-55`.
- Final reusable benchmark state is one pass on index `[4]`, the physical-Q4/M5
  row, with res spec, qspec, Rabi, single shot, T1, Ramsey, and echo enabled.
- The final integrated physical-Q4/M5 pass measured resonator M5 at 9014.950 MHz,
  physical Q4 at 7084.7191 MHz, pi amplitude 0.6686504, T1 7.91 us, T2* 4.74 us,
  and echo T2 13.60 us. Ramsey and echo fringes were resolved. Single-shot
  fidelity was only 0.258 and still needs separate readout optimization. Output:
  `physical_q4_m5_full_validation_0mA/2026-08-28_09-12-29`.
- The characterization deck lists M4 at 8999.95 MHz with a 5575 MHz qubit
  reference but leaves its physical Q# unknown. A 4.075-7.075 GHz M4 search was
  noise-only. A diagnostic-power candidate at 5584.20 MHz did not reproduce at
  lower power, and its standard amplitude-Rabi trace was noise-dominated. The
  candidate was rejected; M4 remains an unvalidated 5575 MHz search seed and
  must not be counted as a calibrated sixth qubit.

### 2026-08-28 — bounded final M4 search and stop decision

- Goal: test the remaining plausible QICK-side explanations for the missing
  transition assigned to resonator row M4, then stop blind scanning if it did
  not reproduce. M4 means resonator row M4 / repository zero-based index 3 /
  displayed Q4; it is not physical flux-tunable Q4, which is on M5.
- Hardware run authorized: yes, by the user. The previously verified Yoko 3 and
  Yoko 4 state remained 0 A with outputs off; neither source was contacted or
  changed in these diagnostics. Every run used resonator DAC attenuators 10 and
  15 dB, qubit DAC attenuators 5 and 4 dB, ADC attenuation 17 dB, and the
  unchanged `PulseProbeSpectroscopyProgram` from
  `section_004_qubit_spec_ge.py`. Abort condition was any QICK/Pyro connection,
  buffer, timing, or acquisition error.
- Deck evidence retained: M4 resonator 9.000 GHz, `f_ge` seed 5575 MHz,
  `f_gf/2` 5466 MHz, anharmonicity 218 MHz, physical-qubit number unknown, and
  a reported VNA linewidth around 20-30 MHz. The deck does not establish that
  its row was measured at the present QICK attenuation/readout/bias conditions.
- Low-power resonator result: gain 0.03, 20 us, 8996-9004 MHz in 0.02 MHz
  steps, 3000 reps. M4 had a real dip at 8999.980 MHz and maximum measured
  magnitude slope at 9000.260 MHz. Output:
  `temporary_m4_low_power_resonator_slope/2026-08-28_13-12-18`.
- Focused deck-frequency test: readout 9000.26 MHz at gain 0.03/20 us;
  qubit drive 5550-5600 MHz in 0.1 MHz steps at gain 0.03/80 us, 5000 reps.
  The trace was noise-only. Output:
  `temporary_m4_low_power_vna_qspec_0mA/2026-08-28_13-18-14`.
- Readout-offset tests: (1) zero-offset readout at 8999.98 MHz, gain 0.30/10 us,
  qubit sweep 5480-5600 MHz in 0.2 MHz steps, gain 0.03/80 us, 4096 reps; and
  (2) slope readout at 9000.40 MHz with the same settings. Both were noise-only.
  Outputs: `temporary_m4_zero_offset_vna_qspec_0mA/2026-08-28_13-28-48`
  and `temporary_m4_high_snr_slope_qspec_0mA/2026-08-28_13-38-30`.
- Two-dimensional readout-parking test: readout 8999.4-9000.8 MHz in 0.2 MHz
  steps at gain 0.30/10 us versus qubit drive 5450-5700 MHz in 1 MHz steps at
  gain 0.03/80 us, 2048 reps. All eight rows were flat; no coherent vertical
  feature appeared. Raw HDF5 and plot are under
  `temporary_m4_resonator_qspec_2d_0mA/2026-08-28_13-48-30`.
- Cross-resonator control: all M1-M6 readouts used their established slope
  points with gain 0.30/10 us while the same 5500-5625 MHz, 0.5 MHz-step qubit
  sweep used gain 0.03/80 us and 2048 reps. M1 produced a clear 5518.00 MHz
  response at about 8 local sigma, proving that the program, drive channel,
  frequency axis, and analysis could recover a real transition. M4 remained
  flat. A 5575 MHz fluctuation on M5 was only about 1.3 sigma and was rejected.
  Output: `temporary_m4_cross_resonator_qspec_0mA/2026-08-28_14-08-02`.
- Final high-resolution combination: M4 low-power readout at 9000.26 MHz,
  gain 0.03/25 us; qubit drive 5525-5625 MHz at 0.05 MHz spacing, gain 0.30,
  40 us pulse, 2048 reps, one round. All 2001 points completed normally, but
  raw IQ and the minimally detrended projection were noise-only with no feature
  at 5575 MHz or elsewhere in the band. Plot:
  `temporary_m4_high_resolution_low_readout_qspec_0mA/`
  `2026-08-28_14-21-41/documentation/m4_high_resolution_qspec.png`.
- Save failure: the final plot saved, but the generated HDF5 path was 268
  characters and `h5py` failed at the Windows 260-character path limit after
  acquisition. Raw arrays from that final run were therefore not persisted.
  `_tmp_m4_readout_debug.py` was retained because M4 was not found, and its
  future sub-study/dataset labels were shortened. `AGENTS.md` now requires a
  full generated-path-length preflight and post-run plot-plus-HDF5 verification.
- Conclusion: simple readout-center error, zero versus slope offset, readout
  gain/punch-out, insufficient qubit gain, 20-80 us pulse length, skipped narrow
  bins near 5575 MHz, and a resonator-row swap in the 5500-5625 MHz region have
  all been tested without a reproducible M4 line. Earlier gain 0.5-1.0 and
  4.075-7.075 GHz searches remain rejected as recorded above. No candidate was
  promoted to `system_config.py`, `expt_config.py`, or round robin.
- Stop decision: the user requested that blind work stop if it was circling.
  No further hardware acquisitions were started. Resume only with new external
  evidence: the original VNA trace and exact bias/attenuation/port routing, an
  independently repeated two-tone/VNA measurement through M4, or a confirmed
  device-map correction. Until then, M4/index 3 is an unassigned resonator with
  a 5575 MHz search seed, not a validated sixth qubit.

### 2026-08-28 — modular tutorial-derived repository harness

- Goal: implement the five harness-defense layers from Walking Labs Lecture 01
  as durable, repo-specific context, environment, verification, and state files;
  point the root agent guide to them; and preserve everything learned during the
  PUCQ4 campaign without promoting guesses to calibration.
- Non-goals: no experiment/config calibration changes, QICK/Pyro/Visdom/Yoko
  connection, attenuator change, `M:` write, or hardware validation.
- Result: added the `harness_engineering/` index, task template, repository map,
  experiment/data contract, environment/safety guide, verification ladder,
  consolidated PUCQ4 state, failure-mode diagnostic guide, evidence sources, and
  a machine-readable manifest. `AGENTS.md` now routes agents through the harness.
- State guard captured: the live `round_robin_benchmark.py` is still configured
  as a rejected M4/index-3 Rabi diagnostic and must be deliberately reset before
  any full characterization run. M4 remains an unassigned row with a deck seed,
  not a validated sixth qubit.
- Executable feedback: `tools/harness_check.py` now validates the manifest,
  required documents, exact five defense-layer names, authorization guard, and
  root `AGENTS.md` link in addition to its import-free Python checks.
- Verification: configured PyCharm Conda Python 3.9.13; offline harness check
  passed for 107 Python files; manifest/document checks passed; `git diff
  --check` reported no whitespace errors (only existing LF/CRLF warnings).
- Hardware validation: not run and not required for this documentation/harness
  task.

### 2026-08-28 — five-mode zero-current calibration and report

- Hardware authorization: explicit. Both Yokos remained in current mode at
  exactly 0 A with outputs off throughout the zero-current campaign.
- Scope: validated rows M1, M2, M3, M5/physical Q4, and M6/physical Q6. M4/index
  3 remained excluded as unresolved.
- Full round-robin run: `rr_full5_optimized_0mA/2026-08-28_16-21-09`, one pass,
  res spec -> qspec -> Rabi -> SSF -> T1 -> T2R -> T2E, exit code 0.
- Focused acceptance reruns: M5/M6 qspec and Rabi at
  `rr_m5_m6_qspec_rabi_followup_0mA/2026-08-28_16-55-52`; M6 qspec gain at
  `rr_m6_qspec_gain_followup_0mA/2026-08-28_17-05-21`; M5/M6 coherence at
  `rr_m5_m6_coherence_followup_0mA/2026-08-28_17-10-46`.
- Accepted qspec gains are M1 .001, M2 .007, M3 .002, M5 .004, M6 .006. M5 and
  M6 final plots show about 0.1 MHz FWHM and reach about 0.30 calibrated
  population without the old square-pulse ringing.
- Accepted sigmas are .050, .101, .124, .055, and .062 us; resulting pi gains
  are .679, .649, .659, .649, and .649 for M1/M2/M3/M5/M6.
- Three alternating SSF length and gain/frequency rounds converged to lengths
  `[10,5,12,5,3]` us, gains `[.48,.56,.41,.33,.60]`, and offsets
  `[-.52,-.40,-.08,+.44,-.04]` MHz. Final RR fidelities were only
  `[58.2,34.3,37.2,17.4,21.5]%`; this is an experimentally observed readout
  limitation, not a claim of high fidelity.
- Accepted coherence values (us): M1 31.45/16.53/37.13, M2
  8.77/11.71/15.88, M3 45.79/42.85/77.22, M5 7.63/5.28/11.26, M6
  3.34/1.53/4.19 for T1/T2R/T2E respectively.
- Report: generated
  `pucq4_first_light/PUCQ4_first_light_characterization.pptx` using the reusable
  analysis-only `tools/build_pucq4_first_light_ppt.py`. Coherence section plots
  remain analog I/Q because the existing round-robin classes do not emit
  population-calibrated coherence figures.
- Flux reference: the source deck confirms exactly two tunable devices and a
  plotted -10 to +10 mA range. No flux sweep was started in this checkpoint.

### 2026-08-28 — PowerPoint aspect-ratio correction

- User requirement: saved experiment plots must never be stretched to fill a
  slide cell because that warps axes and text.
- Updated `tools/build_pucq4_first_light_ppt.py` to contain and center every
  image while preserving its source aspect ratio. Added the rule to
  `harness_engineering/EXPERIMENT_CONTRACTS.md` for all future report edits.

### 2026-08-28 — authorized PUCQ4 flux-campaign envelope

- Scope: Yoko3/physical Q4/M5 and Yoko4/Q6/M6 only, at the nine deck currents
  from -10 through +10 mA. The other source must stay at 0 A/output off.
- Controller: `pucq4_07_flux_campaign.py`, resumable via a manifest under the
  normal `pucq4_first_light/yoko_flux_campaign` data root.
- Each point uses a bounded locator, then normal narrow res/qspec/Rabi, bounded
  sigma retries targeting pi gain 0.60-0.70, followed by res/T1/T2R/T2E.
- Abort condition: failed subprocess, missing/invalid fit, locator outside its
  explicit window, source verification failure, or pi gain outside the target
  after three corrections. Cleanup always ramps the active source to 0 A at
  0.5 mA/s and disables output.

### 2026-08-28 — flux campaign hardware checkpoint

- Completed clean Q4 chains at `0`, `-2.5`, and `-5 mA`; exact values are in
  `harness_engineering/PUCQ4_STATE.md` and the resumable M-drive manifest.
- Wide diagnostics were necessary to reject false narrow-window fits at
  `-5 mA` and `-7.5 mA`; accept broad fits only after full-trace inspection.
- At `-7.5 mA`, the real line is `6423.240 MHz`, but SSF remained `0.1031`
  after wide rescue. The next dependency is raw-I/Q Rabi calibration before
  SSF, not another unchanged readout sweep.
- All failed/aborted passes invoked cleanup; both Yokos were checked at
  `0.0 A`, output off. Remaining sweep points and PowerPoint slides are not
  claimed complete.

### 2026-08-28 — flux checkpoint added to characterization deck

- Regenerated `PUCQ4_first_light_characterization.pptx` with eight new slides:
  a Q4/M5 current-point table, qspec, Rabi, T1, T2R, T2E, readout-length, and
  gain/frequency optimization evidence for the completed 0, -2.5, and -5 mA
  points. The -7.5 mA locator-only result and its SSF blocker are labeled as
  incomplete; remaining Q4 and all Q6 current points are not represented as
  measurements.
- All plot images use contained, centered placement. Automated inspection found
  18 slides, 64 pictures, and maximum relative image-aspect error below `9e-7`,
  so plot text and axes are not stretched.

### 2026-08-28 — resolved Q4 at -7.5 mA

- Fixed the Rabi/SSF circular dependency by adding a 1200-average, unscaled
  raw-I/Q Rabi calibration before readout optimization. At `-7.5 mA`, sigma
  changed from the `0.110 us` seed to `0.05911 us`; accepted pi gain is
  `0.64869`, readout is `7 us / 0.9267 / -0.16 MHz`, and SSF improved from the
  former `0.1031` blocker to `0.666`.
- Accepted coherence is T1 `19.09 us`, T2R `1.86 us`, and T2E `6.01 us`.
  Values were added to `system_config.PUCQ4_FLUX_CALIBRATION` and the reusable
  PowerPoint was regenerated with a dedicated resolved-result slide.
- At `-10 mA` and `+2.5 mA`, narrow qspec traces were visibly noise/failed fits;
  downstream Rabi was stopped rather than accepting aliases. The controller now
  requests and prints qspec FWHM and rejects fit errors, out-of-window centers,
  and invalid linewidths before starting Rabi.
- After both interrupted diagnostic paths, Yoko3 and Yoko4 were independently
  verified at `0.0 A` with output off.

### 2026-08-29 — flux campaign recovery checkpoint

- Fixed raw qspec fit-failure handling so `section_004_qubit_spec_ge.py` always
  returns an FWHM slot (`NaN` on failure) without changing its QICK program.
  The campaign can now reject a failed raw fit and continue into bounded
  readout bootstrap instead of crashing on a three-value/four-value mismatch.
- Q4 completed new full chains at `+7.5 mA` and `+10 mA`. Accepted values are:
  `+7.5`: qfreq `7022.3193 MHz`, sigma `0.05445 us`, pi `0.65867`, readout
  `6 us / 0.9033 / -0.24 MHz`, SSF `0.6913`; `+10`: qfreq `6841.5394 MHz`,
  sigma `0.05911 us`, pi `0.62873`. Q4 is complete at every deck current except
  `-10 mA`, where two expanded readout searches remained below 9% fidelity.
- Q6 completed full chains at `0 mA` and `+2.5 mA`. At `+2.5 mA`, accepted
  qfreq is `6842.1315 MHz`, sigma `0.05800 us`, pi `0.60877`, resonator base
  `9060.46 MHz`, and readout is `12 us / 0.7967 / -0.52 MHz` with SSF `0.5753`.
- Critical Q6 recovery rule: nonzero-current raw Rabi must use the proven Q6
  readout seed, not the historical `3 us / 0.60 / -0.04 MHz` defaults. The
  working seed is currently `12 us / 0.7967 / -0.52 MHz`. With the defaults,
  noise-only Rabi fits produced stable-looking aliases.
- Raw-Rabi sigma correction is damped as
  `sigma *= sqrt(pi_amp / 0.65)` to avoid jumping between fit aliases. Never
  accept a pi value solely because it is in `0.60-0.70`; visually require a
  resolved oscillation whose fit amplitude is meaningful relative to scatter.
- Normalize Lorentzian FWHM sign before validation because the model is
  symmetric in gamma. Continue rejecting widths below `0.02 MHz`, outside the
  scan, or above the stage bound.
- Q6 unresolved current points: `-2.5 mA` has a narrow qspec response near
  `6122.8 MHz` but noise-dominated Rabi and <9% SSF; `+5 mA` candidates near
  `7025.2` and `7034.5 MHz` collapse to unphysical narrow fits; `+7.5 mA`
  candidates near `7156-7157 MHz` produced 8.5 kHz or 139 MHz narrow-fit
  failures. Q6 `-10`, `-7.5`, `-5`, and `+10 mA` remain unmeasured in this
  checkpoint. Do not claim the full flux sweep or final PowerPoint complete.
- All interrupted and failed paths executed controller cleanup. Final
  independent checks: Yoko3 `0.0 A / output False`; Yoko4
  `0.0 A / output False`.

### 2026-08-29 — Q6 negative/+10 mA rejection evidence

- Q6 `-5 mA`: a ±100 MHz diagnostic locator found `5587.7655 MHz` with a
  `0.4814 MHz` fitted width, but a recentered low-power acquisition collapsed
  to an unphysical `0.00175 MHz` width. The feature was not accepted.
- Q6 `-7.5 mA`: the locator found `5061.7466 MHz`, but low-power qspec returned
  `0.00787 MHz`. Two 1200-average raw-I/Q Rabi traces were visibly noise-only;
  both returned the identical `0.8482879` pi-amplitude alias despite a sigma
  change. The retry branch was stopped under the quality abort condition.
- Q6 `+10 mA`: the narrow locator completed, but low-power qspec returned
  `0.00788 MHz`. Its first 1200-average raw-I/Q Rabi trace was likewise
  noise-dominated, so the automatic retry was stopped and no calibration was
  propagated.
- Q6 `-10 mA`: the deck-centered narrow locator returned an unphysical
  `0.00276 MHz` width. A bounded 4343-4543 MHz diagnostic scan found a broad
  feature at `4391.6689 MHz` (`0.9435 MHz`), and a recentered narrow scan
  reproduced it well enough to reach Rabi. The 1200-average response was
  monotonic rather than oscillatory and again hit the `0.8482879` boundary
  alias, so this is spectroscopy-only evidence and not an accepted calibration.
- A failed qspec path can create an HDF5 file whose qspec datasets contain the
  string placeholder `"None"`; file existence alone is therefore not evidence
  that raw sweep arrays persisted. Treat this as a data-persistence failure and
  validate dataset contents before using failed-point files in aggregate maps.
- The campaign controller now records `KeyboardInterrupt` points as `aborted`
  before re-raising, preventing operator quality stops from leaving stale
  `running` manifest entries. Both Yokos were independently rechecked afterward:
  current mode, 10 mA range, `0.0 A`, output off.

### 2026-08-30 — Q6 -10 mA current-specific sigma calibration

- The user authorized treating the recovered -10 mA response as Rabi and
  adjusting sigma without changing the zero-Yoko Q6 sigma (`0.062 us`).
- Four 1200-average raw-I/Q Rabi measurements used current-specific sigmas
  `0.0708`, `0.0808813`, `0.0913046`, and `0.0986291 us`. Fitted pi amplitudes
  were `0.8482879`, `0.8283281`, `0.7584691`, and `0.6886101`, respectively.
  The final raw-I/Q result meets the required `0.6-0.7` window.
- The exact pair is stored separately in
  `system_config.PUCQ4_FLUX_PULSE_CALIBRATION["Yoko4_Q6_M6"][-10.0]` and the
  zero-current device arrays were not changed.
- Full-chain validation remains incomplete: post-readout qspec returned
  `4377.9192 MHz` with `0.81095 MHz` FWHM (just above the `0.8 MHz` bound), and
  the corresponding optimized-readout Rabi returned the `0.8482879` boundary
  fit. The saved pulse entry is therefore marked `revalidation_required`, not
  promoted into the accepted full-point calibration or used for coherence.
- Controller cleanup and an independent query confirmed both sources in current
  mode, 10 mA range, exactly 0 A, with outputs off.

### 2026-08-30 — Q6 -7.5 mA bounded Rabi-only recovery

- The normal recentered locator at `5061.7466 MHz` again collapsed to an
  unphysical `0.00440 MHz` width. A guarded `--rabi-only` campaign mode was
  added for established-frequency pulse recovery; it still uses the normal
  round-robin res/Rabi HDF5 path and source cleanup, but never marks the full
  current point complete.
- Four 1200-average traces tested sigma/pi pairs `0.0986291 us / 0.5488921`,
  `0.0906341 us / 0.3792346`, `0.0692292 us / 0.7784289`, and approximately
  `0.07575 us / 0.2794360`. The non-monotonic branch switching exhausted the
  bounded retry set without entering `0.6-0.7`.
- Visual inspection showed noise-dominated I/Q at every sigma; fitted curves
  were smaller than point scatter. No sigma was promoted to system config, and
  the provisional controller seed was removed. Numeric HDF5 remains useful as
  rejected evidence in the aggregate Rabi map.
- Both sources were independently rechecked afterward: current mode, 10 mA
  range, exactly 0 A, output off.

### 2026-08-29 — renewed full-band M4 search with corrected Nyquist zones

- User explicitly requested another empirical search without new device
  evidence. Hardware envelope: M4/index 3, Yokos held at 0 A/output off,
  resonator DAC attenuators 10/15 dB, qubit DAC attenuators 5/4 dB, ADC 17 dB,
  unchanged `PulseProbeSpectroscopyProgram`, and persisted plot plus HDF5 after
  every acquisition.
- Board preflight confirmed QICK 0.2.363, both drive generators present, 9 GHz
  resonators in DAC zone 2/ADC zone 5, and the qubit-DAC boundary at 4915.2 MHz.
  The old wide scan's lower segment had used zone 2 and was not a valid exclusion.
- Corrected surveys used 1 us pulses and explicit zones: 1.000-4.910 GHz in
  zone 1, 4.916-9.700 GHz in zone 2, plus a gain-1.0 repeat over the expected
  4.916-7.100 GHz band. No M4 response exceeded 1.22 local sigma.
- A six-readout cross-resonator survey over 4.916-7.100 GHz recovered the known
  M1 transition near 5518 MHz and M5 transition near 7085 MHz as positive
  controls, but no sixth transition. Artifacts are under
  `M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/`
  `pucq4_first_light/tmp_m4_hr_qspec_0mA/2026-08-29_21-12-58` through
  `2026-08-29_21-42-29`.
- Result: no candidate was accepted or propagated. Frequency-only, readout-row
  mapping, insufficient coarse-bin width, and weak-drive hypotheses are now
  experimentally rejected at zero current. The remaining diagnosis is a dead,
  uncoupled, unobservable, or incorrectly routed sixth device, which cannot be
  repaired by further configuration-only spectroscopy.

### 2026-08-30 — Q6 -5 mA bounded Rabi-only recovery rejected

- At the established diagnostic candidate `5587.7655 MHz`, four 1200-average
  raw-I/Q measurements tested sigma/pi pairs `0.0986291 us / 0.8482879`,
  `0.1126730 us / 0.3492950`, `0.0825961 us / 0.5788317`, and
  `0.0779433 us / 0.7285296`.
- The bounded series never entered the required `0.6-0.7` pi-amplitude window.
  The last trace was visibly scatter-dominated with no resolved oscillation, so
  no interpolation retry was justified. The provisional `-5 mA` sigma seed was
  removed and no entry was added to `PUCQ4_FLUX_PULSE_CALIBRATION`.
- The manifest records the pulse calibration as failed. Both Yokogawas were
  independently verified afterward in current mode at `0.0 A`, outputs off.

### 2026-08-30 — Q6 +5 mA Rabi-only fit rejected visually

- A guarded 1200-average raw-I/Q run at `+5.0 mA`, `7027.0 MHz`, and
  `sigma=0.058001 us` returned the nominally in-window scalar
  `pi_amp=0.6886101`. Both the Rabi and resonator plot/HDF5 artifacts persisted.
- The saved I/Q trace contains no resolved oscillation: fitted modulation is
  smaller than the point scatter and repeats the known `0.6886101` fit alias.
  It therefore fails the scientific Rabi gate despite passing the numeric
  window. The manifest pulse entry is marked failed, the provisional `+5 mA`
  controller seed was removed, and no system-config calibration was added.
- Controller cleanup completed; independent readback confirmed both sources in
  current mode on the 10 mA range at `0.0 A`, outputs off.

### 2026-08-30 — Q6 cross-readout diagnostic

- Added `pucq4_08_cross_readout_qspec.py`, which keeps the physical-Q6 drive on
  Python index 5 while acquiring the same raw narrow qspec through selected
  resonator rows. It saves one canonical HDF5 plus a row-versus-frequency map
  and always returns Yoko4 to zero/output off.
- Positive control at `+2.5 mA`, `6842.1315 MHz`, gain `0.01`, 1200 reps showed
  a broad `7.67`-local-sigma response only through M6. This validates the
  diagnostic and the expected readout mapping.
- The identical six-row acquisition at `+5 mA`, centered `7027.0 MHz`, found no
  coherent feature: row maxima M1-M6 were `0.65, 1.20, 1.34, 2.16, 0.80, 1.17`
  local sigma. This rejects a simple readout-row migration in 7025-7029 MHz.
- Both runs persisted PNG and HDF5 artifacts under `cross_ro_q6_<current>/`.
  Independent cleanup readback found both sources at `0.0 A`, outputs off.
- A targeted M6 repeat centered on the earlier high-power `7025.2 MHz`
  candidate, using its original qspec gain `0.03`, returned only `0.77` local
  sigma. The candidate is not reproducible and must not seed Rabi/coherence.
- A bounded 6987-7067 MHz M6 locator at gain `0.1` exposed a new broad feature
  near `7062.75 MHz`. It reproduced at gain `0.03`, then at gain `0.006` with
  3000 reps and 5 kHz spacing. The final Lorentzian fit is
  `7062.76592 +/- 0.00125 MHz`, FWHM `0.17924 +/- 0.00428 MHz`, amplitude
  `8.38` local sigma, and residual RMS `0.43` sigma. This is the accepted +5 mA
  qspec center/gain for subsequent Rabi; earlier 7025/7027/7034 seeds are
  rejected.
- Corrected-frequency raw-I/Q Rabi trials used sigma/pi pairs
  `0.062 us / 0.5688519`, `0.0580009 us / 0.5987914`, and
  `0.0556693 us / 0.6287310`. The final I and Q traces visibly resolve the same
  half-oscillation and meet the `0.6-0.7` window. The exact current-specific
  pair is stored in `PUCQ4_FLUX_PULSE_CALIBRATION["Yoko4_Q6_M6"][5.0]`; the
  zero-current sigma remains `0.062 us`. Status is `revalidation_required`
  until the normal readout/coherence chain passes.
## 2026-08-30: Q6/Yoko4 +5 mA full-chain validation

- Corrected low-power qspec: 7062.771244 MHz, 0.175556 MHz FWHM at gain 0.006.
- Current-specific Gaussian sigma: 0.0556693 us; post-readout pi amplitude: 0.618751.
- Readout: 12.0 us, gain 0.9167, -0.76 MHz offset, SSF 0.4910.
- Coherence fits: T1 5.48 us, T2R 2.77 us, T2E 8.00 us. The plotted Ramsey and echo traces are resolved and their sampling exceeds four points per fringe.
- All calibration, alternating readout, post-readout, and coherence stages saved plots and canonical HDF5. Both Yokogawas were independently verified afterward in current mode, 0.01 A range, 0 A, output off.
- Promoted the +5 mA row into `PUCQ4_FLUX_CALIBRATION`; the zero-current six-row sigma and pi arrays remain unchanged.
## 2026-08-30: Q6/Yoko4 +7.5 mA corrected locator

- A 7107-7207 MHz scan using the static M6 readout was featureless (1.64 local sigma).
- Prior res-spec evidence showed the current-shifted M6 resonator at 9061.58 MHz and readout at 9061.06 MHz. Repeating the same broad scan at that readout recovered an isolated line near 7185.7 MHz (5.56 local sigma).
- Low-power acceptance scan at gain 0.006, 3,000 reps, and 5 kHz spacing fit 7185.658453 MHz with 0.137623 MHz FWHM (9.57 local sigma; center uncertainty 0.001996 MHz).
- The former 7157.242 MHz seed is rejected. The measured frequency and low-power gain now seed bounded current-specific sigma tuning.
- Q6 +7.5 mA raw-I/Q Rabi at the corrected 7185.658453 MHz line accepted sigma 0.058001 us on its first bounded attempt, with fitted pi amplitude 0.638711. Both quadratures show a resolved half oscillation. Stored as a current-specific provisional seed pending the full post-readout/coherence chain; zero-current arrays were not changed.
## 2026-08-30: Q6/Yoko4 +7.5 mA full-chain validation

- Final qspec: 7185.659478 MHz with 0.145713 MHz FWHM; sigma 0.058001 us and pi amplitude 0.638711.
- Readout: resonator base 9061.53 MHz, length 11.0 us, gain 0.8334, offset -0.04 MHz, SSF 0.4740.
- Coherence: T1 5.48 us, T2R 4.13 us, T2E 9.35 us; plotted Ramsey/echo fringes are resolved and satisfy the sampling requirement.
- Every stage saved its plot and canonical HDF5. Both Yokogawas independently verified at 0 A/output off afterward. Promoted to the current-indexed shared calibration without modifying zero-current arrays.
## 2026-08-30: Q6/Yoko4 +10 mA corrected locator

- At measured M6 readout 9061.16 MHz, contiguous high-power windows 7010-7110 and 7110-7210 MHz were featureless (1.49 and 1.04 local sigma), rejecting the old 7153/7162 MHz candidates.
- The 7210-7310 MHz window recovered an isolated line near 7211.5 MHz (5.82 local sigma).
- Low-power acceptance at gain 0.006, 3,000 reps, and 5 kHz spacing fit 7211.438328 MHz with 0.149861 MHz FWHM (12.10 local sigma; center uncertainty 0.001740 MHz).
- This measured frequency/gain now seeds bounded current-specific sigma tuning. Zero-current arrays remain unchanged.
- Q6 +10 mA bounded raw-I/Q tuning accepted sigma 0.0511568 us on retry 2, with pi amplitude 0.628731 at the corrected 7211.438328 MHz line. The Q quadrature shows a clean resolved half oscillation. Stored as a current-specific provisional seed pending full validation; zero-current arrays were not changed.
## 2026-08-31: Q6/Yoko4 +10 mA full-chain validation

- Final qspec 7211.443554 MHz with 0.119489 MHz FWHM; sigma 0.0511568 us and pi amplitude 0.628731.
- Readout: resonator base 9061.68 MHz, 12.0 us, gain 0.8722, offset -0.16 MHz, SSF 0.4710.
- Coherence: T1 5.35 us, T2R 6.35 us, T2E 9.56 us; Ramsey/echo plots have resolved, sufficiently sampled fringes.
- Every stage saved plots and canonical HDF5. Both sources independently verified at 0 A/output off. Promoted to current-indexed shared calibration without changing zero-current arrays.
## 2026-08-31: Q6/Yoko4 -2.5 mA corrected locator

- At current-specific M6 readout 9058.11 MHz, a 6072-6172 MHz high-power scan found the true transition near 6103.9 MHz (6.95 local sigma), rejecting the earlier unstable 6121.9-6124.3 MHz candidates.
- Low-power gain tests gave 0.2939 MHz FWHM at 0.006, 0.2520 MHz at 0.004, 0.2602 MHz at 0.003, and 0.23265 MHz at gain 0.002 with 4,000 reps. The final accepted conservative seed is 6103.846687 MHz at gain 0.002 (4.29 local sigma); it is slightly above the preferred 0.1-0.2 MHz band but lower gain would be marginal.
- This corrected frequency now seeds bounded current-specific sigma tuning; zero-current arrays remain unchanged.
- Q6 -2.5 mA bounded raw-I/Q tuning accepted sigma 0.0605405 us on retry 3, with pi amplitude 0.608771 at the corrected 6103.846687 MHz line. The I quadrature shows a resolved half oscillation. Stored as a current-specific provisional seed pending full validation; zero-current arrays were not changed.

## 2026-08-31: Q6/Yoko4 -2.5 mA full-chain validation

- Final qspec was 6103.796753 MHz with 0.24612 MHz FWHM; the current-specific sigma remained 0.0605405 us and pi amplitude 0.608771.
- Readout optimization selected resonator base 9057.78 MHz, 12.0 us, gain 0.7967, +0.08 MHz offset, and SSF 0.4270.
- Coherence fits are visibly resolved: T1 5.03 us, T2R 1.08 us, and T2E 4.46 us. Every stage saved canonical plots and HDF5.
- Promoted the row to shared current-indexed calibration. The zero-current arrays were unchanged, and both sources were independently verified at 0 A/output off.

## 2026-08-31: Q6/Yoko4 -5 mA corrected pulse and readout blocker

- A 5537.8-5637.8 MHz high-power diagnostic isolated the Q6 response near 5587.9 MHz. A 2 MHz low-power scan at gain 0.006 and 3,000 repetitions confirmed the same region.
- Bounded Rabi tuning accepted current-specific sigma 0.0671428759 us and pi amplitude 0.6586705732 after two retries; the saved I trace resolves just over half an oscillation. The zero-current arrays were not modified.
- The normal full chain measured raw qspec FWHM 0.82564 MHz, then exhausted two alternating length/gain-frequency readout passes and its wider rescue sweep. SSF remained 0.085 before corrected Rabi and 0.080 afterward, below the hard 0.20 floor, so coherence was not run and the row remains unvalidated.
- Cleanup completed and independent readback found Yoko3 and Yoko4 both in current mode at 0.0 A with outputs off.

## 2026-08-31: remaining negative-current acceptance blockers

- Q6/Yoko4 at -7.5 mA: the corrected transition is near 4840 MHz. A raw-I/Q
  Rabi trial reached sigma 0.1222528 us / pi 0.6985900, but the full chain
  rejected post-readout qspec (unstable linewidth, including 1.15587 MHz).
- Q6/Yoko4 at -10 mA: raw qspec resolves near 4376.4 MHz and the provisional
  pulse is sigma 0.0986291 us / pi 0.6886101. Repeated strict full chains
  rejected post-readout population qspec (0.82883 MHz, failed fit, 0.00249 MHz
  noise spike, and 7.97607 MHz noise fit). No coherence values are accepted.
- Q4/Yoko3 at -10 mA exposed a resonator-selection defect: M5 is a peak near
  9012.87 MHz, while the historical argmin selector chose the 9017.8 MHz sweep
  edge. Added opt-in peak selection only for this device/current. With the
  corrected readout, qspec passed at 5984.071123 MHz with 0.173861 MHz FWHM.
- Q4 -10 mA raw-I/Q tuning reached sigma 0.0451969 us / pi 0.6985900, stored
  only as current-specific provisional calibration. The trace remains noisy and
  both the original and calibrated-pulse readout rescues stayed below the 0.20
  floor (0.083 and 0.077), so no SSF/coherence result is accepted.
- Every failed/aborted run executed guarded cleanup. Independent readback after
  hardware work found both sources in current mode at 0.0 A with outputs off.
