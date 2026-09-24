## September 16 - publication completed with explicit user authorization

The user explicitly requested saving the PowerPoint with the previous reports.
Publication succeeded and was hash-verified: 195-slide report at
`M:\_Data\20250822 - Olivia\pucq4_run_started_Aug_3\PUCQ4\pucq4_first_light\PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx`.
SHA256: `D55F2C5180EE970DE78A49D965B3C86101684377C5DAFC94BB3C0D859E27094B`.
The previous 176-slide report is preserved at
`FIRST_LIGHT/ac_review/recovery_20260915/prior_176_slide_report.pptx` with its
original hash. The same archive contains 265 verified evidence files and
`publication.json`. All earlier publication-pending statements are historical.
Scientific status remains unresolved: diagnostic progress, no accepted full
crossing map or J. This publication task contacted no hardware.

# Q5/Q6 recovery plan - September 15, renewed execution

## Terminal outcome (supersedes historical ACTIVE checkpoints below)

The recovery diagnostics completed; the prerequisite for the full map failed.
No acquisition or monitor from this task remains active. Both guards and monitors
exited successfully, with both expected Yokogawas verified at 0 A/output off,
current mode, 10 mA range, 30 V compliance. Acquisition completion is not scientific
acceptance. Stage C was correctly not launched.

- A-B-A: all four starting/return-bias detections passed, both detections at
  -5.652 mA failed. M6 return shifted 67.725 kHz and initial cross-readout centers
  differed 54.450 kHz, exceeding their 50 kHz limits.
- Counterbalanced timing control: all 64 blocks completed in 961.734 seconds.
  M6 held-out t=9.684 passed; M2 t=3.945 failed the unchanged t>=5 requirement.
  Early/late center checks passed, but the cross-readout comparison cannot pass
  when one line fails independent detection. Do not accept its fitted M2 center.
- Audits passed for 148 new canonical HDF5 files and 78 acquisition plots across
  both runs. Eight acquisition/QICK/configuration source hashes were unchanged.

The hurdle is now better bounded: switching readout did not restore the lower
line at the failed bias, and shortening the timing separation did not produce
reliable M2 detection at the control bias. Neither experiment establishes the
microscopic mechanism. More averaging, weaker gates, or another unchanged map
are not supported next steps.

### Remaining prerequisites and next decision

1. Obtain a PUCQ4-specific measured resonator/physical-qubit routing or packaging
   record. Existing design intent and the author's Q5? label do not establish
   physical Q5 identity or a separately addressable microwave path.
2. Before a new full map, demonstrate fresh paired-line visibility and repeat
   stability on both sides of the visibility gap, with an independently
   justified drive/readout intervention. The current results do not identify a
   validated intervention; do not invent device settings or relaunch these plans.
3. The upper branch at the failed B bias was not measured with drive ON in the
   old stopped map or in this lower-line diagnostic. A future branch-selectivity
   test can address that coverage gap, but a null lower control must remain an
   explicit limitation. It cannot by itself authorize a coupled fit.
4. Only then consider the gap-bracketing map described below, retaining all
   original fit and repeat gates and labeling its minimum as model-inferred.

This closes the saved diagnostic branches with negative results; it does not
complete the original scientific objective (paired map, minimum splitting, J,
physical Q5 identity, or weak-response mechanism).

### Report and verification closeout

Local completed report: IDE/q5q6_interleave_0915/recovery_final.pptx (195 slides),
SHA256 D55F2C5180EE970DE78A49D965B3C86101684377C5DAFC94BB3C0D859E27094B.
All 195 slides rendered in native PowerPoint with zero layout issues; the 19
added slides were visually reviewed. Historical 176 slides/notes and 141 media
hashes were preserved. The safe verifier passed for 147 Python files, with no
imports or hardware access; git diff --check passed. Seven A-B-A decision checks,
six aggregation/control checks and the signal/baseline reconstruction checks
passed before the relevant measurements.

Automatic approval review rejected archival and replacement of the report on M:
because it requires explicit publication authorization. The publication script
did not execute; the published report remains the previous 176-slide version.
The completed report and evidence remain safely in the two IDE recovery folders.
A publication question is pending; do not bypass the rejection. Publication is
separate from the unresolved scientific acceptance described above.

## Objective, authority and scope

The user requests a saved plan and its execution to overcome the latest Q5/Q6
failures, using previous work to improve the probability of success. Resolve
repeatable paired spectroscopy across bias, minimum splitting/resonance bias/J,
experimental Q5 identity, independent drive routing, and the weak-response
mechanism where evidence permits. This is authorization for the bounded lab
investigation required by this plan; it does not supply missing wiring facts.
Older round-robin gaps are outside this task. Preserve all existing data, rejected
results, shared calibration, attenuators and underlying QICK programs.

## Starting evidence and why the strategy changes

- The prospective ac_prof2 pilot passed four windows and both center repeats.
- ac_pmap failed the first lower window (frozen t=2.764); its conditional center
  moved 0.131 MHz from prediction. Held-out discovery still gave t=4.746 < 5.
- ac_dmap already allowed unknown-center discovery and retained 16000 validation
  repetitions; its first lower window failed t=3.963. Neither map completed a
  paired bias or repeat control. These failed plans will not be relaunched.
- Earlier central/readout-specific upper tests and a 64000-average escalation
  did not establish repeatable visibility. New thresholds or more averages alone
  are not a recovery strategy.
- Ground-state resonator scans and selected M2 readout settings were unchanged.
  This does not test excited-state discrimination or prove a readout mechanism.
- Exact PUCQ4 design/DXF establish intended Q5/M2, but the original measurement
  author marked Q5?; experimentally confirmed identity and separate wiring are
  missing. Shared DAC1 is the documented microwave drive.

## A. Separate the failure causes using existing evidence

1. Recheck current source and process state; preserve unrelated legacy processes.
2. Compare every complete ac_prof2/ac_pmap/ac_dmap window, including failures,
   using reconstructed canonical raw IQ and saved frozen analyses. Quantify
   absolute line amplitude, per-shot/block noise, direction, width, center,
   ON/OFF baseline changes, and sensitivity to profile mismatch. Compare equal
   shot budgets explicitly; do not compare differently normalized t values as
   physical amplitudes. Verify raw normalization and metadata for every record.
3. Inspect temporal block behavior with independent ON/OFF partitions and assess
   whether apparent bias dependence is distinguishable from elapsed time.
   Existing sequential ON/OFF acquisitions cannot identify this causally.
4. Inspect documented wiring/design provenance not already ruled out; distinguish
   design mapping, actual measured mapping, and independent drive connectivity.
5. Save diagnostic tables/plots, source hashes, limitations and an explicit
   decision for the next experiment. No retrospective result becomes a newly
   accepted line. Do not rewrite prior metric records.

## B. Test the identified bottleneck before another map

The next measurement must change one justified aspect of the protocol and test
it against a control. Candidate interventions, conditional on Stage A:

- If temporal ON/OFF baseline or signal variation dominates, use multiple short,
  balanced/interleaved OFF/ON acquisitions at a held bias and an A-B-A bias return
  to separate time/order from current. Retain the same total shot budget and
  existing drive envelope. Validate aggregation/provenance offline first.
- If frequency/profile mismatch dominates, improve search placement from a
  separate discovery acquisition and validate on new data; do not learn and
  test on the same samples or repeat the failed discovery map unchanged.
- If a documented readout response change dominates, test a bounded readout
  discrimination comparison with measured pulse settings; a ground-state dip
  alone cannot justify optimizing excited-state discrimination.
- If no supported mechanism emerges, do not launch an arbitrary scan. Complete
  source review and identify the precise experimental or wiring prerequisite.

Freeze the selected diagnostic's exact current/frequency/settings, independent
controls, sample budget, family correction, analysis hashes, decision rules and
abort conditions in this document and a fresh JSON plan BEFORE acquisition.
Require a positive control and repeatability at both the successful pilot bias
and a failed-map bias before allowing a full map. A failed diagnostic ends its
branch; no automatic averaging escalation or retry-until-pass.

## C. Conditional paired map and scientific assessment

Only after B passes, freeze a seven-unique-bias paired map with at least two
points on each side of the anticipated minimum, early shifted-grid and final
temporal controls. Use the validated protocol and matched settings. Retain
t>=5, clean independent references, dependence/repeat/conditional-fit gates,
and the existing coupled-model rank, bounds, J uncertainty, residual and slope
alternative gates. Preserve a failed model; do not tune gates to obtain J.
Independent physical Q5 confirmation remains a separate requirement.

## Hardware boundaries and preflight

Use the existing guarded acquisition architecture and canonical HDF5 hierarchy.
Physical Q6=M6/index5/Yoko4/A5; physical Q4=M5/index4/Yoko3/D4 remains zero/off.
M2=index1 is intended-design Q5, experimentally unconfirmed. No undocumented port.
Y4 stays within the established -5.8..-4.8 mA crossing range, 0.5 mA/s ramps,
10 mA range and 30 V compliance; no invented settling requirement. Spectroscopy
stays within 5300..5660 MHz, gain <=0.03, 20 us pulse and 100 us relaxation.
No attenuation change (nominal 10/15, 5/4, 17 dB; setters disabled).
Record exact narrower bounds, repetitions, duration and maximum Windows path
before each run. Fresh expected-serial/mode/range/zero/off readbacks and no
competing acquisition are mandatory. Abort source/acquisition/save/path/deadline
or scientific-control failures; save returned data, then verify both sources at
0 A/output off. Plot AND canonical HDF5 must exist.

## D. Verification, reporting and completion

- Query the configured PyCharm interpreter before Python commands; use Python3.9.
- Run hardware-safe harness and focused diff checks; exercise any new scientific
  analysis with meaningful synthetic controls before real-data interpretation.
- Save all diagnostic results locally in the IDE workspace; acquisition data go
  only to fresh canonical first-light substudies. No raw files enter Git.
- Update this plan, Q5/Q6 handoff and chronological log after accepted/rejected
  evidence. Update the existing report if new scientific evidence is obtained,
  preserving historical slides/notes/media and verifying native rendering.
- Completion requires actual accepted evidence for each scientific endpoint or
  an explicit unresolved prerequisite. A completed plan/report is not scientific
  closure. Do not describe partial execution as full success.

## Execution record

- Plan saved before new analysis or hardware acquisition. Initial process listing
  found no live September15 pilot/map controller; legacy processes are preserved.
- Stage A is beginning; no new hardware acquisition has started.

### Stage A result and frozen Stage B decision

New descriptive split-data analysis reconstructs all saved means. Lower signal
amplitudes (ADC units) are pilot A1 0.08780 +/-0.01151, pilot A2
0.08049 +/-0.01063, fixed-map 0.05200 +/-0.01142, discovery-map
0.02975 +/-0.00733. These are discovery-selected, validation-evaluated profile
amplitudes, not new accepted lines. Equivalent single-shot profile noise stays
approximately 0.60..0.75 ADC units, rather than rising enough to explain the
response loss. Center/width uncertainty and elapsed time remain confounders.
Earlier M6 central lower signal is 0.14419 +/-0.01280 ADC units (descriptive
t=11.267 at 16000 repetitions/state). That is a concrete motivation for a
complementary-readout diagnostic, without changing drive power or averaging.
Outputs: IDE workspace q5q6_recovery_0915/{budget,m6_budget}/budget.json.

The saved six-window experiment is q5q6_recovery_0915/aba.json, with spec.json
freezing the analyzer/acquisition hashes and exact controls. All eight existing
acquisition source hashes match ac_prof2. The new diagnostic decision logic
passed seven positive/negative/completeness checks. No acquisition code changed.

| Visit | Y4 mA | Readout order | Lower-line window MHz |
| --- | --- | --- | --- |
| A1 | -5.6566 | M6, M2 | 5430.7..5432.7 |
| B | -5.652 | M2, M6 | 5431.4..5433.4 |
| A2 | -5.6566 | M6, M2 | 5430.7..5432.7 |

Every window: 101 points, 16000 repetitions per OFF/ON state, gain0.03,
20us pulse,100us relaxation,one round. Row-specific calibrated readout tracking
is unchanged. Root FIRST_LIGHT/ac_roaba/<timestamp>/study_data/Data_h5; maximum
197-character artifact path. Expected about50minutes,7200s deadline+180s guard.
Exact existing controller/guard: pucq4_13_avoided_crossing.py and
tools/pucq4_crossing_guard.py. No old plan is being relaunched.

Assessment uses unchanged unknown-center discovery on64 blocks and independent
validation on64 blocks (8000 held-out repetitions); t>=5,family6 signp<.01,
OFFp>.01,no dependence,conditional-fit quality. First M6 A1 positive-control
failure or invalid reference/dependence requests boundarySTOP. Expected weak B
detections are retained to allow the planned return A, but never promoted.
Lower-protocol readiness requires all three M6 windows and both M2 A windows,
plus same-row A return and cross-readout A center agreement within
min(0.1,max(0.05,3combinedSE))MHz. M2 B may fail detection, explicitly recorded.
Upper-branch and repeated-B validation remain prerequisites for a new full map.
Sequential readouts can demonstrate a usable protocol but cannot alone establish
a microscopic mechanism. No claim of a dark state follows from this experiment.

Relevant primary background, not device-specific proof: Majer et al., Nature
449 (2007), https://qudev.phys.ethz.ch/static/website-2008-2015/content/courses/QSIT07/pdfs/Majer07.pdf,
demonstrates that shared-drive symmetry can suppress a transition at a crossing.
Consequently a visibility failure must not be called absence of an eigenmode;
the present PUCQ4 mechanism still needs its own discriminating evidence.

### Live checkpoint: first A visit

M6 A1 passed: held-out t13.796670, center5431.706905MHz,
FWHM.165508MHz. M2 A1 passed detection: t6.380312,
center5431.761356MHz,FWHM.165525MHz. Cross-readout centers differ54.450kHz,
exceeding the predeclared50kHzlimit. This failure is retained; the A-B-A sequence
continues for the planned diagnostic return, not an automatic map. Four-quarter
independent-reference descriptive fits have no simple monotonic trend explaining
that difference. Results saved in q5q6_recovery_0915/quarters_2. No drift or
readout mechanism is established. Current acquisition is at B; do not relaunch.

### B outcome and conditional next diagnostic (prepared, not launched)

Both B readouts fail: M2 t2.175437; M6 t2.231347 and conditional line-shape
quality fails. The alternate-readout protocol therefore does NOT solve the
central B visibility problem. Complete the predeclared A return to distinguish
local current dependence from a general temporal loss. Do not accept either B
fit, and do not launch the previously conditional full map.

The first A cross-readout mismatch remains54.450kHz. Exploratory200-replicate
independent-block bootstrap gives center SD7.09kHz(M6) and12.72kHz(M2), and does
not erase that mismatch or alter its original failed gate. Both measurements
were separated by several minutes; within-record quarters have no simple
monotonic trend that establishes the cause.

Prepared fresh experiment q5q6_interleave_0915/interleave.json tests this specific
timing confound, after the current guard terminates and sources pass preflight:

- One held Y4=-5.6566mA, Y3zero/off. Same lower window5430.7..5432.7MHz/101points.
-32 cycles, each measures M6 and M2; order alternates(6,2)/(2,6).
-500reps per OFF/ON block: total16000reps/state/readout, exactly the current
  per-readout budget, not an averaging escalation. Gain.03,20us,100us unchanged.
- Existing controller/guard/QICK programs unchanged; one readout scan per row.
  Canonical FIRST_LIGHT/ac_roint/<timestamp>/study_data/Data_h5,197charmax.
  Expected16minutes,3600s+180guard; same ramp/mode/range/compliance/cleanup.
- All64 records required; each row's32 blocks are predeclared, equal-weight,
  same-bias/same-tone/same-frequency, individually saved and normalized. They
  form128 chronological subblocks with64 disjoint discovery/validation blocks.
  This groups a single prospectively designed acquisition; no failed historical
  visit is pooled. t>=5,family2 signp<.01,OFFp>.01,no dependence,fit quality;
  early/late and cross-readout centers must meet the unchanged capped50kHz-or3SE
  rule. No incomplete block selection. Stop on hardware/save/deadline/monitor
  errors or failed complete-group controls; no automatic retry.
- Six offline checks passed: known signal/center, equal-weight reconstruction,
  missing/reordered blocks and mixed tones rejected, null rejected.

If A returns but B remains dark in BOTH readouts, a future map may test a
predeclared visibility gap rather than requiring detection at its center.
Synthetic tests of the UNCHANGED coupled_fit gates passed200/200 trials using
seven points bracketing a gap and10/30kHz independent center noise. This only
establishes numerical identifiability under the synthetic model. Actual paired
visibility on both sides, timing/center controls and all original model gates
remain necessary. Such a minimum would be model-inferred, not directly sampled.
No microscopic dark-state mechanism or physical Q5 identity follows from this.

### September15,16:29 - A-B-A terminal; counterbalanced timing test active

ac_roaba finished2619.157s/exit0; guard43468/child54148 and monitor exited.
Bothsourcesverified expectedserial/CURR/.01A/0A/off/30V. FinalM2A passedt6.451957,
center5431.794255MHz. A-B-A scientificstatusunresolved: bothB detectionsfail,
M6Areturn67.725kHz andinitialcross-readout54.450kHz exceed50kHz. Finalcross-readout
19.625kHz andM2Areturn32.899kHz pass. All18HDF5/12plots audited; staged187-slide
reportpreserves176oldslides/notes and141mediahashes. Notpublishedyet.

Newac_roint/2026-09-15_16-29-21 ACTIVE underguard21004/child45620,
acquisition54503,monitor79643. PID45620 isreused,not thehistoricdiscoveryguard.
Freshpreflightpassedbothsources/allfrozenhashes. ExactoneheldA/32cycles/500reps
perblock/alternatingM6M2 order in q5q6_interleave_0915/interleave.json andspec.
16000totalreps/state/readout,101points,gain.03,20us,100us,197charpath,
3600s+180guard,zerooffcleanup. All64blocksrequired;noearlyselectiveacceptance.
147-fileharness andsix offlineaggregation/controlchecks passed. No sharedconfig,
attenuator,acquisitioncontroller orQICKprogramchange. Publishedreportstill176slides.

Readout covariate review: all three A-B-A visits selected M6 9056.43 MHz and
M2 8951.21 MHz, with unchanged gains/lengths. Thus no tracked-tone change
accompanied the B visibility loss within that run. The new short-block run's
tracker selected M6 9056.38 MHz (50 kHz lower); M2 remained unchanged. This is
recorded as a between-run covariate: a successful new protocol would not prove
that timing alone caused any improvement. No tracked setting was overridden.
