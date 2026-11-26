# Measurement documentation home

Start here for every measurement task. Procedure:
[measurement playbook](../harness_engineering/MEASUREMENT_PLAYBOOK.md).
Architecture and saving: [contracts](../harness_engineering/EXPERIMENT_CONTRACTS.md).
Device mapping and accepted evidence: [PUCQ4 state](../harness_engineering/PUCQ4_STATE.md).
Chronological decisions and failures: [harness log](harness-log.md).

Existing handoffs (check dates, device state, source configs and raw evidence
before treating any values as current):
[Q5/Q6 handoff](Q5_Q6_NEXT_CHAT_HANDOFF.md),
[round-robin validation](pucq4-round-robin-validation.md),
[closeout](pucq4-round-robin-closeout.md),
[avoided crossing](pucq4-avoided-crossing.md).

## Active campaign record

No new acquisition or calibration was performed when this index was created.
At the next authorized measurement task, maintain one current record here with:

- Device, physical qubits/resonator rows/Python indices, bias and timestamp.
- Canonical run documentation path/link, report/PowerPoint and embedded plots.
- Accepted calibration table and exact HDF5/config provenance; rejected
  candidates and unresolved issues clearly separated.
- Qspec FWHM, punch-out ceiling, SSF, T1, fringe sampling, relaxation multiple,
  reps/rounds, elapsed time and repeatability evidence.
- Config propagation status, current safe hardware state and exact next step.

Keep detailed plots and notes in each run's existing documentation folder;
link them here rather than creating competing reports or copying raw data.
Use `![Experiment, qubit, bias and outcome](path-to-existing-plot)` to embed
plots where supported, with a direct artifact link otherwise. This index is
the single navigation point; the run folders remain the artifact source.

## Current status -- November20 after user-requested stop

Q2/M2/index1 Zeno is stopped; no restart queued. The required existing SSF
histogram contrast >50% was not recovered. Fresh pulse checks and101 readout
points yielded repeated candidate9.35-10.61%, baseline7.43-11.29%. FreshT1
4.067+/-.209us,R2.942; sigma.097us envelope.388us. No calibration promotion.
Detailed report and normalplots/raw references:
`M:\_Data\20250822 - Olivia\pucq4_run_started_Oct_7\PUCQ4\qze_005\ssro1120b\2026-11-20_16-27-12\documentation\recovery_review\RESULTS.txt`
Pulse/power/T1 hierarchy: same runroot/PUCQ4/find_resonators_sandbox/ssf1120b.
Fresh .30 powerqualified at7us; .40 fails predefinedshift criterion. Candidate
2us readout remains diagnostic and requires same-length qualification before
promotion. Native startup now savesSSF thenrejects<=.50 ornonfinitecontrast.
Final tProc stopped,clockslocked,Yokos0A/OFF. Resume from report and
harness-log; do not start another weak-readout longrun. Readout chain cause
remains unresolved; constructor attenuation values are not hardware readbacks.

## 2026-11-20 other-row fresh SSF check
User requested SSFs for the other five rows. Existing round_robin_benchmark.py,
flags ss, three independent repetitions per M1/M3/M4/M5/M6,20000 shots/state,
readout gain .10DAC, reset250us M1/M3 and100us others. Historical low-power
settings; M3 uses Nov19 validated carrier/pi/readout. No new qspec/Rabi today;
these are current-setting diagnostics, not accepted optimized benchmarks.
Means(range), histogram contrast percent: M1 14.955(14.800-15.075),
M3 9.892(9.435-10.240), M4 4.412(4.040-4.655),
M5/physicalQ4 9.855(9.520-10.215), M6/physicalQ6 1.190(.775-1.515).
All fail >50% performance target. Preparation staleness and readout limits
remain unresolved; low contrast does not establish qubit absence or root cause.
No config changes, no Zeno restart. All15 normal plots/HDF5 verified with finite
20k IQ/state and metadata. Final tProc stopped/clocks locked,Yokos0A/OFF.
Output: M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/
find_resonators_sandbox/ssall1120. Final run2026-11-20_22-00-12 contains
 documentation/ssf_all_review/RESULTS.txt and exact controller/config snapshots.
IDE controller:ssall1120/check.py. Offline149file verifier passed.
Failure classification: verification feedback; performance gate failed, acquisition
succeeded. Preserve distinction between SSF contrast and assignment accuracy.

## Active all-row SSF optimization -- November20 23:41
User authorized higher readout gain to improve all six rows. Preserve earlier
no-resonance-shift constraint; no permission to use above-shift readout received.
Calibration/power data: PUCQ4/find_resonators_sandbox/ssup1120 under the existing
pucq4_run_started_Oct_7 root. Native SSF optimization: ssopt1120, then ssfine1120.
IDE controller directory: codex/ssup1120. Initial pipeline PID68776, queued finer
gain PID26584, post-readout preparation retry PID30552, offline report PID48992.
Inspect current JSON statuses/processes before hardware access; queued processes
wait and do not overlap acquisition. Do not rerun labels or overwrite raw data.
At this checkpoint M1 initial sweep complete: candidate14.780%, paired baseline
14.712%; no significant recovery. M3 optimization running, M4/M5/M6/M2 follow.
Fresh M1/M3/M5 Rabi pi .68650/.66163/.63178 with R2 .987/.988/.968; valid
half-oscillations, yet low SSF. M4 weak line gets independent narrow retry;
M2 and M6 qspec unresolved (no accepted new carrier); optimize historical seeds
then retry spectroscopy/Rabi at selected readout. Do not label seed-based SSF
as validated preparation. Q2 retry preserves persisted RR gain cap .24 DAC.
Original-length power ceilings (operational30kHz tolerance, not exact punchout):
M1 .1/8us, M2 .4/7us, M3 .2/13us, M4 .1/5us, M5 .1/6us, M6 .1/45us.
Shorter lengths independently qualified. Finer stage tests ceiling+.05DAC with
low/high/low/high resonance confirmation before any high-gain SSF. Failed higher
scans are preserved, not accepted. Zero currents/outputOFF, unchanged attenuators.
SSF is histogram contrast, not assignment accuracy. Zeno remains stopped; no
background script promotes shared configs or restarts it. All acquired candidates
require three20k-shot confirmations. Full results are pending, not a success claim.
Resume from pipeline_status.json, fine_pipeline_status.json, retry_status.json,
report_status.json. On completion RESULTS.txt/summary.json and diagnostic deck
are written to run documentation/campaign_review. report_status.json identifies
final deck. Visual review and any configuration promotion remain separate steps.
Failure class so far: performance/verification; hardware acquisition succeeds,
but >50% target has not been demonstrated. Local19 scripts parse as Python3.9.

Nov20 23:45 accepted intermediate SSF evidence: M3 initial optimization complete,
mean17.2733% (16.945-17.750%), paired baseline9.2284% (8.945-9.535%),3x20000
shots/state each. Gain .1->.2DAC, RO8973.517677->8973.584344MHz, length13us
unchanged; current local preparation5551.348462MHz,sigma.124us,pi.661631DAC.
All75HDF5/75normalplots verified for finite sample counts and exact saved settings;
median native confirmation plot visually reviewed, histogram17.12%, visibly
shifted but substantially overlapping clouds. Improvement is real at these
conditions but far below50%. No sharedconfigpromotion; finer-gain stage remains
queued. M1 audit28HDF5/28plots passed. Pipeline nowM4 refine/optimization,
other queues waiting normally. QICK clockslocked afterM3; hardware remains in
use byM4, so do not describe entire campaign as stopped or completed.

November21 14:29 user explicitly requested stop everything. Current ssup1120 acquisition, fine-gain, retry and report processes had already exited; no Python processes created since November1 were present. STOP marker placed in campaign workspace to cancel pending continuation. Fresh hardware stop acknowledged by QICK stop_tproc; clocks locked. Both Yoko3/Yoko4 verified CURR,0A,outputsOFF. Receipt IDE/codex/ssup1120/user_stop1121.json. Zeno remains stopped. No further measurement or automatic restart authorized; preserve all saved data. Historic IDE/analysis processes were not terminated.

2026-11-21 afternoon: Six fresh zero-bias 0.10 DAC resonator surveys saved under PUCQ4/find_resonators_sandbox/rt1121a, 14:36:50 through 14:39:23. Grid minima M1..M6 = 8919.202, 8951.004, 8974.358, 8999.226, 9013.925, 9058.445 MHz (40 kHz spacing). Source mode CURR/0A/off verified, survey cleanup clocks locked. User confirmed unchanged RF chain. IDE workspace rt1121a contains raw/plot receipts. Next: bounded power, preparation and SSF optimization via continue_run.py; check live stage before any restart. No calibration promotion.

## 2026-11-24 PUCQ4 six-row zero-bias reoptimization

Fresh low-power resonator surveys for M1 through M6 are under the October 7 run, PUCQ4/find_resonators_sandbox/zb1124a. Each of the six timestamped runs has the native resonator plot and res_ge HDF5. The matching IDE controller folder is zb1124a. Both sources read CURR, 0 A, outputs off, 10 mA range and 30 V limit before the survey. All six original acquisitions finished and survey cleanup reported locked clocks. Power qualification uses the fitted minimum and a resolved baseline at each length. M5 original depth SNR was 7.69 at 2000 reps. Use baseline_m*.json for the higher-averaging review and subsequent power qualification. No shared calibration promoted.

The Nov 21 fine queue stopped on missing readout_m5.json because M5's power reference drifted and the preceding pipeline skipped M5. The missing file records no acquisition. Review each row's power eligibility before scheduling downstream runs.

All six 0.10 DAC references passed the notch fit (R2 > 0.95 and depth SNR >= 10). M4 and M5 used independent 6000-rep retries. Fitted magnitude minima M1 through M6 in MHz were 8919.180, 8950.954, 8974.355, 8999.258, 9014.038 and 9058.503. These minima anchor power qualification. Do not promote them as SSF-optimal readout frequencies. All eight acquisitions saved normal plots and HDF5. The baseline controller stopped the tProc, confirmed locked clocks and the final acquisition verified both sources 0 A/off. Fresh power characterization then started under the same sub-study. Inspect zb1124a/power_status.json and the live process before attempting any continuation.

Nov 24 completed original-length resonator power qualification, zb1124a/power_review.json. At M1/M2/M3/M4/M6 (8/7/13/5/45 us), the measured operational ceilings were 0.20/0.30/0.20/0.20/0.10 DAC. The ordinary M2 gain limit remains 0.24 DAC. M5 at 6 us had two resolved 0.10 DAC reference minima of 9014.062807 and 9014.021698 MHz in a low/high/low/high repeat, a 41.1 kHz spread exceeding 30 kHz. Neither gain passed that sequence. All five eligible rows entered the qubit-preparation queue. M5 requires a stable low reference before dependent acquisition. No shared configuration promoted.

M1 preparation in zb1124a: m1_fine coarse line 0.10315 MHz FWHM, confirmed by m1_drive0 at 5515.346774 MHz, FWHM 0.10552 MHz and fitted peak population 0.497. The native calibrated-population plots show a single resolved line and just over half a Rabi period. Sigma 0.050 us, fitted pi gain 0.68153 DAC, Rabi R2 0.9958. At this preliminary point the subsequent single-shot histogram contrast was only 25.855%, below 50%. Await the alternating length/gain/frequency search and independent confirmations before shared changes. M3 repeat line FWHM 0.09635 MHz, peak 0.480, center 5551.354075 MHz. Coarse width 0.10869 MHz. The repeat width is within one acquired 5 kHz bin of 0.1 MHz; retain the measured width and use its independently repeated center for the Rabi follow-up.

M1 first zero-bias readout search, zbopt24a/2026-11-24_14-11-14, saved 67 SSF HDF5 and 67 native plots. At 8 us, 0.20 DAC, 8918.442916 MHz, independent 20k-shot confirmations gave 26.645%, 25.750%, 25.785% (mean 26.06%). Paired baseline mean 13.085%. Shorter lengths at 2 and 4 us qualified ceilings of 0.30 and 0.10 DAC. No SSF promotion. Fine gain and a longer length from the historical measured range remain eligible only after same-length power qualification.

M3 Rabi at the independently confirmed 5551.354075 MHz, sigma 0.124 us: measured pi 0.65168 DAC, R2 0.9916, preliminary SSF 19.42%. Original-length and alternate-length optimization zbopt24a/2026-11-24_14-30-03 gave three independent 20k-shot/state contrasts of 19.605%, 19.525%, 19.015% at 8974.058219 MHz, 13 us, 0.20 DAC. Paired baseline mean 10.184%. The 19.382% candidate mean misses the 50% acceptance gate; finer power qualification and repeat checks remain queued.

M4/plot Q4 repeated qspec in zb1124a/m4_refine_qs and acquired a resolved joint-IQ center of 5594.629158 MHz (0.19757 MHz FWHM); the independent standard population plot shows 5594.632 MHz, 0.202 MHz FWHM and peak about 0.41. Rabi R2 0.9598, pi gain 0.64671 DAC, preliminary SSF 9.70%. Native readout search zbopt24a/2026-11-24_14-58-54 saved 69 points; three paired confirmations at 8999.104811 MHz/0.20 DAC/5 us averaged 10.207%, versus 5.192% baseline. Both 4 us and 8 us qualified at 0.40 DAC and warrant independent readout checks. No shared change. M6 and M2 preparation still requires fresh qspec confirmation.

M6/physical Q6 preliminary readout run zbopt24a/2026-11-24_15-22-46, 34 points. The initial qspec had raw vector SNR 1.05 and state-reference separation SNR 2.55, so pulse preparation remains unverified. At 8 us/0.20 DAC/9059.283020 MHz, three 20k-shot confirmations using the historical pulse gave 3.795%, 4.375%, 3.135% histogram contrast; mean 3.768%, paired baseline 1.483%. The 2 and 4 us power references did not qualify. Retry spectroscopy at the requalified readout, including the historical 0.010 DAC discovery and 64000-rep narrow check if needed.

M2 post-coarse narrow confirmation recovered in zb1124a/m2_refine_qs: joint-IQ center 5438.511117 MHz, FWHM 0.15658 MHz, vector SNR 6.47, local R2 0.666. Standard calibrated plot reports 5438.504 MHz and 0.174 MHz FWHM with fitted peak about 0.31. Fresh Rabi using sigma 0.097 us gave pi 0.68650 DAC and R2 0.9833; preliminary single-shot contrast 11.885%. Continue readout-length and bounded gain/frequency qualification. Prior unresolved status is superseded for the preparation at this readout point.

M2 first readout optimization zbopt24a/2026-11-24_15-44-29, 69 acquired points. At 8949.955019 MHz, 7 us and 0.24 DAC, three 20k-shot confirmations gave 12.225%, 13.845%, 13.180% contrast (mean 13.083%, paired baseline 12.335%). No accepted SSF change. The first pass is complete on all five power-qualified rows. Fine-gain and alternate-length checks are serialized before the remaining preparation retries.

M1 fine readout, zbfine24a/2026-11-24_16-01-28. Repeated power checks qualified 0.25 DAC at both 8 and 10 us. At 8 us/8918.242916 MHz, three candidate contrasts were 31.595%, 31.270%, 31.155% (mean 31.340%, paired 0.20-DAC baseline 25.923%). A 2 us/0.30 DAC diagnostic gave 19.75%. The fixed-gain 10 us screen gave 33.4%, below the 8 us screening maximum. The confirmed 8 us point fails the 50% criterion; no shared readout update.

Fine readout repeats on M3 and M4 also remained below 50%. M3 at 13 us, 0.25 DAC, 8973.858219 MHz gave 24.320%, 24.816%, 25.215% (mean 24.784%, paired baseline 19.343%). M4 at 8 us, 0.40 DAC, 8999.438145 MHz gave 17.740%, 19.375%, 17.325% (mean 18.147%, paired baseline 9.840%). M4's neighboring 7 us comparison used 0.40 DAC after fresh low/trial/low/trial qualification. This was shorter than the 8 us candidate, although the archived controller labeled it longer because it indexed from the initial 5 us setting. Actual lengths are saved per point. Fine run receipts: zbfine24a/2026-11-24_16-14-12 and zbfine24a/2026-11-24_16-29-05.

M2 and M6 fine-stage outcomes: M2/0.24 DAC/7 us/8950.088352 MHz gave 10.267% (paired baseline 10.918%); the preceding confirmed candidate mean of 13.083% remains higher. M6/0.20 DAC/8 us/9059.283020 MHz gave 4.262% (paired baseline 4.032%) with historical preparation and little separation. Its 0.25 DAC attempt failed the 30 kHz power qualification. M6 post-readout spectroscopy is diagnostic until independently confirmed. Fine-stage files are in zbfine24a; exact points and repeats are in zb1124a/fine_gain_m2.json and fine_gain_m6.json.

Offline audit after the five paired readout searches checked 442 native SSF HDF5 records and 442 matching plots. It verified finite ground/excited I/Q shot arrays and exact shot counts, saved fidelity and configuration against the per-point records. M1 97 points, M2 83, M3 99, M4 115 and M6 48. M5 had no SSF points in the first campaign because its reference drifted. In M6's 250 MHz discovery scan the usual 1.5 MHz local joint-fit half-window contains fewer than 20 samples at 0.25 MHz spacing. Reanalysis of the saved HDF5 with a 15-grid-step half-window returned only SNR 3.23 and R2 0.183 for its highest matched feature. See m6_wide_refit.json. No broad-scan carrier accepted.

M6/physical Q6 post-readout recovery at 9059.283020 MHz, gain 0.20 DAC, 8 us: two weak qspec scans agreed near 6405.16 MHz. The 64000-rep window around the historical carrier showed a rising feature at its upper edge; it did not bracket the line. A new centered 0.010 DAC/100 us scan, 101 points x 16000 reps, gave 6405.163622 MHz, FWHM 0.24272 MHz, peak population 0.299. Independent 0.006 DAC/100 us confirmation, 101 points x 32000 reps, gave 6405.151978 MHz, FWHM 0.29392 MHz, peak 0.388, SNR 7.70. Both widths pass 0.5 MHz; the narrower PUCQ4 target remains unmet. New Rabi pi 0.65168 DAC at sigma 0.05559 us, R2 0.8386. Three fresh-pulse SSFs were 3.895%, 4.075%, 4.445%. The carrier and preparation are now supported at this readout, but contrast remains below the acceptance gate. Exact records in zb1124a/m6_centered_review.json and per-scan HDF5/plots. Reoptimize readout with the new carrier, then check reduced drive.

Physical Q4/M5 recovered in separate zb5r24a runs. At 8 us, a stable 0.10 DAC reference qualified; fitted baseline dip 9014.035634 MHz, R2 0.964, SNR 13.47. The old 6 us qualification remains rejected. M5 qspec reproduced at 7065.003268 MHz, FWHM 0.16452 MHz, fitted peak population 0.469, drive 0.006 DAC/100 us. Fresh Rabi with sigma 0.056940825 us gave pi 0.61686 DAC and R2 0.977. Preliminary SSF was 10.781% at the recovery readout. Native readout optimization then began under zb5opt24a/2026-11-24_17-52-53. All six rows now have measured zero-bias qspec and Rabi results; final readout/coherence acceptance is separate.

November 24 zero bias, physical Q4/M5. The initial 8 us, 0.10 DAC, 9013.695634 MHz candidate averaged 11.629% contrast in three confirmations. The fine search averaged 11.112% versus 11.767% for its paired baseline. Keep the initial point for the independent preparation check. See zb5r24a/readout_m5.json and fine_gain_m5.json. M6 readout optimization with the newly measured 6405.151978 MHz carrier started after M5 stopped.

Physical Q6/M6, new-pulse readout (zb6opt24a and zb6fine24a). After length and gain/frequency searches, the 8 us, 0.20 DAC, 9058.849687 MHz point averaged 7.052% contrast (6.660% to 7.521%) versus 6.407% paired baseline. Its fine-stage improvement was within two paired standard errors. The preceding broader grid averaged 5.587%. All 79 fresh-pulse search records passed raw IQ, contrast, configuration, and plot-count audit. With M5's 44 records and the preceding 442-point audit, 565 saved readout points have passed those checks.

## 2026-11-24 zero-bias closeout

All six PUCQ4 rows completed resonator spectroscopy, qubit spectroscopy, amplitude Rabi, alternating readout searches and independent 20,000-shot readout checks at zero current. Values below are diagnostic. The frequency is the preparation carrier from joint IQ. Widths and peak populations come from the native population fit. The optimization mean uses three candidate confirmations.

| Row (index) | Physical | Carrier MHz | FWHM MHz | Peak population | Pi gain DAC | Optimization mean | Fresh SSF |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 (0) | unassigned | 5515.347298 | 0.1090 | 0.484 | 0.6815 | 31.34% | 31.57% |
| M2 (1) | unassigned | 5438.508261 | 0.1945 | 0.249 | 0.6865 | 13.08% | 12.00% |
| M3 (2) | unassigned | 5551.353725 | 0.0993 | 0.471 | 0.6567 | 24.78% | 23.39% |
| M4 (3) | unassigned | 5594.634433 | 0.1739 | 0.422 | 0.6417 | 18.15% | 16.00% |
| M5 (4) | Q4 | 7065.025317 | 0.1659 | 0.476 | 0.6169 | 11.63% | 12.08% |
| M6 (5) | Q6 | 6405.160106 | 0.2875 | 0.349 | 0.6368 | 7.05% | 5.28% |

M4 failed the 0.45 DAC trial at 8 us. Stop at the measured 0.40 DAC ceiling. M6's 0.0045 DAC drive trial gave 0.284 MHz joint-IQ width and 0.281 peak population. Returning to 0.006 DAC gave 0.284 MHz joint-IQ width, 0.349 peak and Rabi R2 0.923. M2's independent population peak at the selected readout was 0.249. M6 remains wider than the preferred device target. No row passed >50% SSF. Final one-pass coherence remains blocked and no settings were promoted. Next diagnostic step is to verify the physical readout chain against its last documented working configuration before repeating the best M1 reference. Both sources verified current mode, zero amperes, outputs off at closeout. QICK tProc stopped with clocks locked.

The 19-slide dated handoff is in pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/PUCQ4_zero_bias_20261124.pptx. The evidence JSON, native plot and HDF5 links, optimization figures, and per-stage receipts are in pucq4_run_started_Oct_7/PUCQ4/find_resonators_sandbox/zb1124a/2026-11-24_13-17-41/documentation/zero_bias_review on M:. The two report copies share SHA-256 c1f3604a44a4bf80c58d3f2eb476c76bc68ca06d55420ea4b0d29348bf48ec5b.

November 25 overnight Ramsey drift, active diagnostic acquisition. Oct_7/PUCQ4/find_resonators_sandbox/ramsey_drift/2026-11-25_21-48-10 started at 21:48 local; both sources verified CURR/0 A/OFF. All six November 24 readout points, fresh startup res/qspec/Rabi, repeated fixed-carrier Ramsey. M1 initial dip 8919.11982 MHz, operating tone 8918.24291575 MHz. Analysis writes documentation/ramsey_offsets.csv and .png after t2_ge batches, reports conditional fit errors, checks alternate delay subsets, and averages up to four neighboring traces per row for weak fringes. First Ramsey batch and full overnight coverage pending.
