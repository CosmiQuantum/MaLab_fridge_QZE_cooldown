# PUCQ4 round-robin validation closeout

Updated 2026-09-09T21:50:53 local. Status: **closed_with_gaps**.

The requested time limit ended this campaign with the remaining gaps below. Acceptance counts describe the recorded reproduction checks; separate spectroscopy tuning targets remain explicit.

- [Validation PowerPoint (149 slides)](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/PUCQ4_round_robin_validation_20260908.pptx>)
- [Every attempt and selected audit](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/PUCQ4_round_robin_validation_20260908_artifacts/validation_manifest.json>)
- [Final software, provenance and source-state checks](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/PUCQ4_round_robin_validation_20260908_artifacts/closeout_checks.json>)

- [Archived plans, execution logs, source baselines and helper scripts](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/PUCQ4_round_robin_validation_20260908_artifacts/exec_0909>)

## Completed evidence and remaining coverage

| Evidence | Accepted | Remaining / findings |
| --- | --- | --- |
| Selected zero-current complete chains | 5 / 6 | M1 |
| Physical Q4 / M5 / index 4 / Yoko 3: nonzero currents | 5 / 8 | -10.0, -7.5, -5.0 mA |
| Physical Q6 / M6 / index 5 / Yoko 4: nonzero currents | 5 / 8 | -10.0, -7.5, -5.0 mA |
| M1 endpoint control | review | Yoko 3 +10.0 mA; Yoko 4 +0.0 mA |
| M2 endpoint control | accepted | Yoko 3 +10.0 mA; Yoko 4 +0.0 mA |
| M3 endpoint control | unmeasured | Yoko 3 +0.0 mA; Yoko 4 -10.0 mA |
| M4 endpoint control | unmeasured | Yoko 3 +0.0 mA; Yoko 4 -10.0 mA |

Latest physical Q4 / M5 / Yoko 3 -5 mA pass: **review**. [Complete-pass audit](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/rrv5_m050a/2026-09-09_21-22-47/documentation/validation_M5.json>). qspec: frequency shift exceeds reference linewidth
Its one exact spectroscopy repeat differs by 0.126506 MHz; the repeat limit is 0.015666 MHz. Agreement: False. The original reference difference remains recorded.

## Final all-six default pass

Run: [2026-09-09_19-23-29](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/rr6_0mA/2026-09-09_19-23-29>).
No PUCQ4 environment overrides; one normal pass with res spec, qspec, Rabi, SSF, T1, T2R and T2E. 3 / 6 complete chains pass the recorded gates after review. All final-run outcomes remain in the deck.

| Row / index | Final audit | Findings |
| --- | --- | --- |
| M1 / 0 | [review](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/rr6_0mA/2026-09-09_19-23-29/documentation/validation_M1.json>) | t2_ge: coherence outside comparison band; t2_ge: independent fit requires review; t2_ge: structured oscillations remain in fit residuals; t2e_ge: coherence outside comparison band |
| M2 / 1 | [review](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/rr6_0mA/2026-09-09_19-23-29/documentation/validation_M2.json>) | SSF: contrast fell more than the comparison tolerance |
| M3 / 2 | [review](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/rr6_0mA/2026-09-09_19-23-29/documentation/validation_M3.json>) | t1_ge: coherence outside comparison band |
| M4 / 3 | [pass](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/rr6_0mA/2026-09-09_19-23-29/documentation/validation_M4.json>) | all recorded gates pass |
| M5 / 4 | [pass](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/rr6_0mA/2026-09-09_19-23-29/documentation/validation_M5.json>) | all recorded gates pass |
| M6 / 5 | [pass](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/rr6_0mA/2026-09-09_19-23-29/documentation/validation_M6.json>) | all recorded gates pass |

Exact-setting closeout confirmations: [review and original records](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/PUCQ4_round_robin_validation_20260908_artifacts/final_default_confirmation_review.json>).

- M3 T1: 31.4602 versus 37.9150 us; difference 6.4548 us exceeds the 5.2031 us repeat limit. The original full-pass finding remains; no promotion.
- M2 SSF: 0.26780 versus 0.28530; difference 0.01750 passes the 0.03000 repeat limit. The historical comparison and alternating-readout-optimization gap remain; no promotion.

M1's two predetermined 180 us diagnostics gave 31.038 +/- 0.774 us and 19.474 +/- 1.153 us. Their 11.565 us difference exceeded the 4.167 us repeat limit. Both individual fits passed quality checks; neither diagnostic was promoted. The final default Ramsey trace is assessed separately above.

Spectroscopy targets are FWHM 0.1–0.2 MHz and calibrated peak population 0.3–0.5. See [all measured target deviations](<M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/PUCQ4_round_robin_validation_20260908_artifacts/spectroscopy_target_review.json>). A resolved, repeatable spectrum is not automatically a completed tuning optimization.

## Configuration and reuse

Use `round_robin_benchmark.py` with its normal seven flags and one pass. Set YOKO3_CURRENT_MA (physical Q4 / M5) and YOKO4_CURRENT_MA (physical Q6 / M6) to a documented grid point. The script selects shared pulse/readout and acquisition profiles, remeasures spectroscopy before Rabi/coherence, and retains the calibrated absolute readout tone. Current validation uses one nonzero source at a time. Check the coverage table before using an unvalidated combination.

Defaults finish at zero current. M1–M6 correspond to displayed Q1–Q6 and Python indices 0–5. Physical identities for M1–M4 remain unassigned. Nominal attenuation arguments remain resonator 10/15 dB, qubit 5/4 dB and ADC 17 dB; these are not new attenuator readbacks.

Both Yokogawas were finally verified in current mode at 0 A with outputs off. The protected experiment sources and both reference decks match their task-start hashes. Offline verification is recorded separately from the instrument results.

## Recorded changes

All values below are task-start to final values. Measurement errors and source runs are in the deck/audits. Very small differences can reflect retained numeric precision.

| Row / current | Setting | Old | New | Unit | Source file |
| --- | --- | --- | --- | --- | --- |
| M1 / 0 mA | res_freq_ge | 8919.85 | 8919.77 | MHz | system_config.py |
| M2 / 0 mA | res_freq_ge | 8951.65 | 8951.61 | MHz | system_config.py |
| M3 / 0 mA | res_freq_ge | 8975.0 | 8974.8 | MHz | system_config.py |
| M4 / 0 mA | res_freq_ge | 8999.94 | 8999.92 | MHz | system_config.py |
| M3 / 0 mA | res_gain_ge | 0.41 | 0.95 | DAC gain | system_config.py |
| M3 / 0 mA | res_length_ge | 12.0 | 9.0 | us | system_config.py |
| M3 / 0 mA | res_freq_offset_ge | -0.08 | -0.21 | MHz | system_config.py |
| M4 / 0 mA | res_freq_offset_ge | -0.21 | -0.22999999999956344 | MHz | system_config.py |
| M1 / 0 mA | qubit_freq_ge | 5517.69373 | 5517.691628432926 | MHz | system_config.py |
| M2 / 0 mA | qubit_freq_ge | 5438.71351 | 5438.710887054151 | MHz | system_config.py |
| M3 / 0 mA | qubit_freq_ge | 5557.33447 | 5557.335688295514 | MHz | system_config.py |
| M4 / 0 mA | qubit_freq_ge | 5594.10936235 | 5594.541902899418 | MHz | system_config.py |
| M1 / 0 mA | pi_amp | 0.6786303 | 0.681529634377 | DAC gain | system_config.py |
| M2 / 0 mA | pi_amp | 0.6486907 | 0.666605627785 | DAC gain | system_config.py |
| M3 / 0 mA | pi_amp | 0.6586706 | 0.666605627785 | DAC gain | system_config.py |
| M4 / 0 mA | pi_amp | 0.651681621193 | 0.6417322834645669 | DAC gain | system_config.py |
| M5 / -2.5 mA | qfreq_MHz | 6945.02924848 | 6945.1129971731425 | MHz | system_config.py |
| M5 / -2.5 mA | pi_amp | 0.646706952 | 0.6467069523286333 | DAC gain | system_config.py |
| M5 / -2.5 mA | res_base_MHz | 9013.74 | 9013.7 | MHz | system_config.py |
| M5 / -2.5 mA | readout_offset_MHz | 0.32 | 0.3599999999987631 | MHz | system_config.py |
| M5 / -2.5 mA | T1_us | 12.662962 | 12.843022492027956 | us | system_config.py |
| M5 / -2.5 mA | T2R_us | 3.491655 | 3.3305882611408713 | us | system_config.py |
| M5 / -2.5 mA | T2E_us | 9.683989 | 9.688102646269144 | us | system_config.py |
| M5 / -2.5 mA | qspec_FWHM_MHz | None | 0.11948517573545973 | MHz | system_config.py |
| M5 / -2.5 mA | qspec_FWHM_error_MHz | None | 0.0017745327919961704 | MHz | system_config.py |
| M5 / -2.5 mA | qspec_reps | None | 5000 | averages | system_config.py |
| M5 / -2.5 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M5 / -2.5 mA | qspec_steps | None | 401 | points | system_config.py |
| M5 / -2.5 mA | SSF | None | 0.5926 | contrast | system_config.py |
| M5 / -2.5 mA | requires_qspec_each_pass | None | True | boolean | system_config.py |
| M5 / +0.0 mA | qfreq_MHz | 7084.2807751 | 7084.548595174133 | MHz | system_config.py |
| M5 / +0.0 mA | pi_amp | 0.646706952 | 0.6467069523286333 | DAC gain | system_config.py |
| M5 / +0.0 mA | res_base_MHz | 9014.74 | 9014.66 | MHz | system_config.py |
| M5 / +0.0 mA | readout_frequency_MHz | 9014.94 | 9014.86 | MHz | system_config.py |
| M5 / +0.0 mA | readout_offset_MHz | 0.2 | 0.2000000000007276 | MHz | system_config.py |
| M5 / +0.0 mA | T1_us | 12.440386 | 13.125145177243246 | us | system_config.py |
| M5 / +0.0 mA | T2R_us | 6.123946 | 6.098595743922029 | us | system_config.py |
| M5 / +0.0 mA | T2E_us | 13.850536 | 14.335001803438807 | us | system_config.py |
| M5 / +0.0 mA | qspec_FWHM_MHz | None | 0.10225077122014355 | MHz | system_config.py |
| M5 / +0.0 mA | qspec_FWHM_error_MHz | None | 0.0015882665134117476 | MHz | system_config.py |
| M5 / +0.0 mA | SSF | None | 0.5118 | contrast | system_config.py |
| M5 / +2.5 mA | qfreq_MHz | 7143.68161402 | 7143.883429322838 | MHz | system_config.py |
| M5 / +2.5 mA | pi_amp | 0.651681621 | 0.6467069523286333 | DAC gain | system_config.py |
| M5 / +2.5 mA | res_base_MHz | 9015.04 | 9014.68 | MHz | system_config.py |
| M5 / +2.5 mA | readout_frequency_MHz | 9014.56 | 9014.82 | MHz | system_config.py |
| M5 / +2.5 mA | readout_length_us | 5.0 | 4.0 | us | system_config.py |
| M5 / +2.5 mA | readout_gain | 0.93 | 0.95 | DAC gain | system_config.py |
| M5 / +2.5 mA | readout_offset_MHz | -0.48 | 0.14 | MHz | system_config.py |
| M5 / +2.5 mA | T1_us | 8.626213 | 10.196045128561797 | us | system_config.py |
| M5 / +2.5 mA | T2R_us | 6.4704 | 7.605553477536421 | us | system_config.py |
| M5 / +2.5 mA | T2E_us | 13.620759 | 15.566037333880487 | us | system_config.py |
| M5 / +2.5 mA | qspec_FWHM_MHz | None | 0.11932360664032048 | MHz | system_config.py |
| M5 / +2.5 mA | qspec_FWHM_error_MHz | None | 0.001401703175142955 | MHz | system_config.py |
| M5 / +2.5 mA | qspec_reps | None | 5000 | averages | system_config.py |
| M5 / +2.5 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M5 / +2.5 mA | qspec_steps | None | 401 | points | system_config.py |
| M5 / +2.5 mA | SSF | None | 0.5841 | contrast | system_config.py |
| M5 / +5.0 mA | qfreq_MHz | 7122.14833311 | 7122.257671006202 | MHz | system_config.py |
| M5 / +5.0 mA | pi_amp | 0.651681621 | 0.6516816211926998 | DAC gain | system_config.py |
| M5 / +5.0 mA | readout_offset_MHz | -0.48 | -0.47999999999956344 | MHz | system_config.py |
| M5 / +5.0 mA | T1_us | 5.681098 | 5.1744804110233265 | us | system_config.py |
| M5 / +5.0 mA | T2R_us | 1.303506 | 1.1310400931072198 | us | system_config.py |
| M5 / +5.0 mA | T2E_us | 8.89044 | 8.360387540413047 | us | system_config.py |
| M5 / +5.0 mA | qspec_FWHM_MHz | None | 0.6243436691610823 | MHz | system_config.py |
| M5 / +5.0 mA | qspec_FWHM_error_MHz | None | 0.005091870061888435 | MHz | system_config.py |
| M5 / +5.0 mA | qspec_reps | None | 5000 | averages | system_config.py |
| M5 / +5.0 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M5 / +5.0 mA | qspec_steps | None | 401 | points | system_config.py |
| M5 / +5.0 mA | SSF | None | 0.5782 | contrast | system_config.py |
| M5 / +7.5 mA | qfreq_MHz | 7022.13834876 | 7022.162967054066 | MHz | system_config.py |
| M5 / +7.5 mA | pi_amp | 0.651681621 | 0.6516816211926998 | DAC gain | system_config.py |
| M5 / +7.5 mA | res_base_MHz | 9014.04 | 9013.96 | MHz | system_config.py |
| M5 / +7.5 mA | readout_frequency_MHz | 9013.8 | 9013.800000000001 | MHz | system_config.py |
| M5 / +7.5 mA | readout_offset_MHz | -0.24 | -0.1599999999980355 | MHz | system_config.py |
| M5 / +7.5 mA | T1_us | 7.214108 | 12.757370732627168 | us | system_config.py |
| M5 / +7.5 mA | T2R_us | 3.767455 | 4.038153856007287 | us | system_config.py |
| M5 / +7.5 mA | T2E_us | 8.62431 | 10.907325682518627 | us | system_config.py |
| M5 / +7.5 mA | qspec_FWHM_MHz | None | 0.1223185584832736 | MHz | system_config.py |
| M5 / +7.5 mA | qspec_FWHM_error_MHz | None | 0.0018248357161659372 | MHz | system_config.py |
| M5 / +7.5 mA | qspec_reps | None | 5000 | averages | system_config.py |
| M5 / +7.5 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M5 / +7.5 mA | qspec_steps | None | 401 | points | system_config.py |
| M5 / +7.5 mA | SSF | None | 0.7133 | contrast | system_config.py |
| M5 / +10.0 mA | qfreq_MHz | 6841.22106308 | 6841.484578285427 | MHz | system_config.py |
| M5 / +10.0 mA | pi_amp | 0.646706952 | 0.6516816211926998 | DAC gain | system_config.py |
| M5 / +10.0 mA | res_base_MHz | 9013.09 | 9013.11 | MHz | system_config.py |
| M5 / +10.0 mA | readout_offset_MHz | -0.12 | -0.1400000000012369 | MHz | system_config.py |
| M5 / +10.0 mA | T1_us | 15.896839 | 15.426027649345263 | us | system_config.py |
| M5 / +10.0 mA | T2R_us | 2.139733 | 2.4383651838558325 | us | system_config.py |
| M5 / +10.0 mA | T2E_us | 7.97947 | 8.333998671208258 | us | system_config.py |
| M5 / +10.0 mA | qspec_FWHM_MHz | None | 0.15045271857763148 | MHz | system_config.py |
| M5 / +10.0 mA | qspec_FWHM_error_MHz | None | 0.0024709538824901063 | MHz | system_config.py |
| M5 / +10.0 mA | qspec_reps | None | 5000 | averages | system_config.py |
| M5 / +10.0 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M5 / +10.0 mA | qspec_steps | None | 401 | points | system_config.py |
| M5 / +10.0 mA | SSF | None | 0.6431 | contrast | system_config.py |
| M5 / +10.0 mA | requires_qspec_each_pass | None | True | boolean | system_config.py |
| M5 / +10.0 mA | qspec_frequency_min_MHz | None | 6841.457614725295 | MHz | system_config.py |
| M5 / +10.0 mA | qspec_frequency_max_MHz | None | 6841.583959579264 | MHz | system_config.py |
| M6 / -2.5 mA | qfreq_MHz | 6103.73607274 | 6103.73104802091 | MHz | system_config.py |
| M6 / -2.5 mA | pi_amp | 0.651681621 | 0.6516816211926998 | DAC gain | system_config.py |
| M6 / -2.5 mA | res_base_MHz | 9057.78 | 9057.9 | MHz | system_config.py |
| M6 / -2.5 mA | readout_length_us | 12.0 | 10.0 | us | system_config.py |
| M6 / -2.5 mA | readout_gain | 0.7967 | 0.8 | DAC gain | system_config.py |
| M6 / -2.5 mA | readout_offset_MHz | 0.08 | -0.039999999999054126 | MHz | system_config.py |
| M6 / -2.5 mA | T1_us | 5.91692 | 6.143175070153691 | us | system_config.py |
| M6 / -2.5 mA | T2R_us | 1.173581 | 1.0157470309560712 | us | system_config.py |
| M6 / -2.5 mA | T2E_us | 4.150655 | 4.2917276961703585 | us | system_config.py |
| M6 / -2.5 mA | qspec_FWHM_MHz | None | 0.3281951157831831 | MHz | system_config.py |
| M6 / -2.5 mA | qspec_FWHM_error_MHz | None | 0.013409378770879705 | MHz | system_config.py |
| M6 / -2.5 mA | qspec_reps | None | 40000 | averages | system_config.py |
| M6 / -2.5 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M6 / -2.5 mA | qspec_steps | None | 101 | points | system_config.py |
| M6 / -2.5 mA | SSF | None | 0.32474 | contrast | system_config.py |
| M6 / +0.0 mA | qfreq_MHz | 6522.511432598 | 6522.489876246844 | MHz | system_config.py |
| M6 / +0.0 mA | pi_amp | 0.646706952329 | 0.6467069523286333 | DAC gain | system_config.py |
| M6 / +0.0 mA | readout_frequency_MHz | 9058.48 | 9058.460000000001 | MHz | system_config.py |
| M6 / +0.0 mA | readout_offset_MHz | -0.4 | -0.41999999999825377 | MHz | system_config.py |
| M6 / +0.0 mA | T1_us | 6.435511765 | 5.571980204657388 | us | system_config.py |
| M6 / +0.0 mA | T2R_us | 1.803285514 | 1.7803606627046535 | us | system_config.py |
| M6 / +0.0 mA | T2E_us | 5.466466197 | 5.168960727869256 | us | system_config.py |
| M6 / +0.0 mA | qspec_FWHM_MHz | 0.211476902 | 0.20019179587582014 | MHz | system_config.py |
| M6 / +0.0 mA | qspec_FWHM_error_MHz | 0.010673046 | 0.012556762587316085 | MHz | system_config.py |
| M6 / +0.0 mA | SSF | 0.37234 | 0.35378 | contrast | system_config.py |
| M6 / +2.5 mA | qfreq_MHz | 6841.87974285 | 6841.775497163998 | MHz | system_config.py |
| M6 / +2.5 mA | pi_amp | 0.651681621 | 0.6516816211926998 | DAC gain | system_config.py |
| M6 / +2.5 mA | res_base_MHz | 9060.46 | 9060.08 | MHz | system_config.py |
| M6 / +2.5 mA | readout_frequency_MHz | 9059.94 | 9059.98 | MHz | system_config.py |
| M6 / +2.5 mA | readout_length_us | 12.0 | 22.0 | us | system_config.py |
| M6 / +2.5 mA | readout_gain | 0.7967 | 0.8 | DAC gain | system_config.py |
| M6 / +2.5 mA | readout_offset_MHz | -0.52 | -0.1000000000003638 | MHz | system_config.py |
| M6 / +2.5 mA | T1_us | 7.218997 | 6.79621105927428 | us | system_config.py |
| M6 / +2.5 mA | T2R_us | 2.288311 | 2.320281858189406 | us | system_config.py |
| M6 / +2.5 mA | T2E_us | 6.164389 | 6.533448775271403 | us | system_config.py |
| M6 / +2.5 mA | qspec_FWHM_MHz | None | 0.2052599851296157 | MHz | system_config.py |
| M6 / +2.5 mA | qspec_FWHM_error_MHz | None | 0.0020320259945249414 | MHz | system_config.py |
| M6 / +2.5 mA | qspec_reps | None | 5000 | averages | system_config.py |
| M6 / +2.5 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M6 / +2.5 mA | qspec_steps | None | 401 | points | system_config.py |
| M6 / +2.5 mA | SSF | None | 0.6575 | contrast | system_config.py |
| M6 / +5.0 mA | qfreq_MHz | 7062.53901877 | 7062.612211354989 | MHz | system_config.py |
| M6 / +5.0 mA | pi_amp | 0.646706952 | 0.6516816211926998 | DAC gain | system_config.py |
| M6 / +5.0 mA | res_base_MHz | 9060.98 | 9060.78 | MHz | system_config.py |
| M6 / +5.0 mA | readout_offset_MHz | -0.76 | -0.5600000000013097 | MHz | system_config.py |
| M6 / +5.0 mA | T1_us | 6.308351 | 7.120905647510946 | us | system_config.py |
| M6 / +5.0 mA | T2R_us | 3.342414 | 2.4521516277292186 | us | system_config.py |
| M6 / +5.0 mA | T2E_us | 7.630265 | 7.949307821673911 | us | system_config.py |
| M6 / +5.0 mA | qspec_FWHM_MHz | None | 0.17117461028187456 | MHz | system_config.py |
| M6 / +5.0 mA | qspec_FWHM_error_MHz | None | 0.002246002837143701 | MHz | system_config.py |
| M6 / +5.0 mA | qspec_reps | None | 5000 | averages | system_config.py |
| M6 / +5.0 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M6 / +5.0 mA | qspec_steps | None | 401 | points | system_config.py |
| M6 / +5.0 mA | SSF | None | 0.4262 | contrast | system_config.py |
| M6 / +7.5 mA | qfreq_MHz | 7185.39063657 | 7185.513939668361 | MHz | system_config.py |
| M6 / +7.5 mA | pi_amp | 0.651681621 | 0.6467069523286333 | DAC gain | system_config.py |
| M6 / +7.5 mA | res_base_MHz | 9061.53 | 9061.47 | MHz | system_config.py |
| M6 / +7.5 mA | readout_offset_MHz | -0.04 | 0.020000000000436557 | MHz | system_config.py |
| M6 / +7.5 mA | T1_us | 5.60195 | 5.45179527299386 | us | system_config.py |
| M6 / +7.5 mA | T2R_us | 4.927458 | 5.7231990564627235 | us | system_config.py |
| M6 / +7.5 mA | T2E_us | 8.03301 | 9.238789592754308 | us | system_config.py |
| M6 / +7.5 mA | qspec_FWHM_MHz | None | 0.13651071888404784 | MHz | system_config.py |
| M6 / +7.5 mA | qspec_FWHM_error_MHz | None | 0.0029425361220824365 | MHz | system_config.py |
| M6 / +7.5 mA | qspec_reps | None | 5000 | averages | system_config.py |
| M6 / +7.5 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M6 / +7.5 mA | qspec_steps | None | 401 | points | system_config.py |
| M6 / +7.5 mA | SSF | None | 0.433 | contrast | system_config.py |
| M6 / +10.0 mA | qfreq_MHz | 7211.22085715 | 7211.413079440341 | MHz | system_config.py |
| M6 / +10.0 mA | pi_amp | 0.641732283 | 0.6417322834645669 | DAC gain | system_config.py |
| M6 / +10.0 mA | res_base_MHz | 9061.68 | 9061.54 | MHz | system_config.py |
| M6 / +10.0 mA | readout_offset_MHz | -0.16 | -0.020000000000436557 | MHz | system_config.py |
| M6 / +10.0 mA | T1_us | 6.303632 | 6.67982963731644 | us | system_config.py |
| M6 / +10.0 mA | T2R_us | 4.448974 | 6.114422314936588 | us | system_config.py |
| M6 / +10.0 mA | T2E_us | 9.648699 | 9.700775749042306 | us | system_config.py |
| M6 / +10.0 mA | qspec_FWHM_MHz | None | 0.17805935715636778 | MHz | system_config.py |
| M6 / +10.0 mA | qspec_FWHM_error_MHz | None | 0.0028538062848888603 | MHz | system_config.py |
| M6 / +10.0 mA | qspec_reps | None | 5000 | averages | system_config.py |
| M6 / +10.0 mA | qspec_rounds | None | 1 | rounds | system_config.py |
| M6 / +10.0 mA | qspec_steps | None | 401 | points | system_config.py |
| M6 / +10.0 mA | SSF | None | 0.4358 | contrast | system_config.py |
| M1 / 0 mA | VNA_res | 8919.85 | 8919.77 | MHz | expt_config.py |
| M2 / 0 mA | VNA_res | 8951.65 | 8951.61 | MHz | expt_config.py |
| M3 / 0 mA | VNA_res | 8975.0 | 8974.8 | MHz | expt_config.py |
| M4 / 0 mA | VNA_res | 8999.94 | 8999.92 | MHz | expt_config.py |
| M5 / 0 mA | VNA_res | 9014.74 | 9014.66 | MHz | expt_config.py |
| M1 / 0 mA | VNA_qubit | 5517.7 | 5517.691628432926 | MHz | expt_config.py |
| M2 / 0 mA | VNA_qubit | 5438.72 | 5438.710887054151 | MHz | expt_config.py |
| M3 / 0 mA | VNA_qubit | 5557.33 | 5557.335688295514 | MHz | expt_config.py |
| M4 / 0 mA | VNA_qubit | 5594.10936235 | 5594.541902899418 | MHz | expt_config.py |
| M5 / 0 mA | VNA_qubit | 7084.2807751 | 7084.548595174133 | MHz | expt_config.py |
| M6 / 0 mA | VNA_qubit | 6522.511432598 | 6522.489876246844 | MHz | expt_config.py |
| M1 / 0 mA | RES_FREQS_MEASURED | 8919.85 | 8919.77 | MHz | pucq4_config.py |
| M2 / 0 mA | RES_FREQS_MEASURED | 8951.65 | 8951.61 | MHz | pucq4_config.py |
| M3 / 0 mA | RES_FREQS_MEASURED | 8975.0 | 8974.8 | MHz | pucq4_config.py |
| M4 / 0 mA | RES_FREQS_MEASURED | 8999.94 | 8999.92 | MHz | pucq4_config.py |
| M5 / 0 mA | RES_FREQS_MEASURED | 9014.74 | 9014.66 | MHz | pucq4_config.py |
| M3 / 0 mA | RES_READOUT_OFFSETS | -0.08 | -0.21 | MHz | pucq4_config.py |
| M4 / 0 mA | RES_READOUT_OFFSETS | -0.21 | -0.22999999999956344 | MHz | pucq4_config.py |
| M5 / 0 mA | RES_READOUT_OFFSETS | 0.2 | 0.2000000000007276 | MHz | pucq4_config.py |
| M6 / 0 mA | RES_READOUT_OFFSETS | -0.4 | -0.41999999999825377 | MHz | pucq4_config.py |
| M1 / 0 mA | QUBIT_FREQS_MEASURED_0MA | 5517.69373 | 5517.691628432926 | MHz | pucq4_config.py |
| M2 / 0 mA | QUBIT_FREQS_MEASURED_0MA | 5438.71351 | 5438.710887054151 | MHz | pucq4_config.py |
| M3 / 0 mA | QUBIT_FREQS_MEASURED_0MA | 5557.33447 | 5557.335688295514 | MHz | pucq4_config.py |
| M4 / 0 mA | QUBIT_FREQS_MEASURED_0MA | 5594.10936235 | 5594.541902899418 | MHz | pucq4_config.py |
| M5 / 0 mA | QUBIT_FREQS_MEASURED_0MA | 7084.2807751 | 7084.548595174133 | MHz | pucq4_config.py |
| M6 / 0 mA | QUBIT_FREQS_MEASURED_0MA | 6522.511432598 | 6522.489876246844 | MHz | pucq4_config.py |
| All six | Rows selected by default | M1,M2,M3,M5,M6 | M1--M6 | rows | round_robin_benchmark.py |
| All six | Coherence population scaling | off | on | boolean | round_robin_benchmark.py |
| All six | Readout after res scan | new dip + old offset | saved tone; measured offset | MHz | round_robin_benchmark.py |
| Yoko3 +10 mA | Post-ramp settling | 0 | 600 | s | round_robin_benchmark.py |
| M6 / +10 mA | Ramsey phase detuning | 0.5 | 0.015398310214337 | MHz | expt_config.py |
| M6 / +10 mA | Ramsey fitter frequency guess | automatic FFT | 0.04 (free fit) | MHz | expt_config.py |
| M6 / +10 mA | Ramsey population fit starts | automatic | phase0; amplitude1; offsets0.5/0.5 | rad / population | expt_config.py |
| All rows/currents | Acquisition profiles | run overrides | persistent profiles | see settings tables | expt_config.py |
| M6 / -2.5 mA | Qspec points | 401 | 101 | points | expt_config.py |
| M6 / -2.5 mA | Qspec repetitions | 5000 | 40000 | averages / point | expt_config.py |
| M6 / -2.5 mA | State calibration and SSF shots | 10000 | 50000 | shots / state | expt_config.py |
| M6 / -2.5 mA | Ramsey repetitions | 1200 | 2400 | averages / point / round | expt_config.py |
| M6 / -2.5 mA | Echo repetitions | 1200 | 2400 | averages / point / round | expt_config.py |
