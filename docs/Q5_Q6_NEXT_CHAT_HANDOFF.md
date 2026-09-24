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

# Q5/Q6 handoff for the next chat
## September 15 - recovery diagnostics terminal; publication awaiting approval

**No acquisition or monitor from this task remains active.** All ACTIVE entries
below are historical. Both runs ended with the expected Yokogawas verified at
0 A/output off, current mode, 10 mA range, 30 V compliance. Guard/monitor sessions
87754/52719 and 54503/79643 exited with code 0. Do not relaunch failed plans.

Saved plan and terminal decisions: `docs/q5-q6-recovery-plan-20260915.md`.
A-B-A completed all six windows: both readouts lost the lower line at B=-5.652 mA
and recovered at A=-5.6566 mA, but M6 return and initial cross-readout frequency
checks failed. The 64-block counterbalanced run completed in 961.734 seconds:
M6 t=9.684 passed; M2 t=3.945 failed. Time-half checks passed, but the paired
readout comparison could not pass. Neither diagnostic establishes a usable
protocol for a full map. No minimum splitting, resonance bias, J, physical Q5
identity, or microscopic mechanism is accepted. No configuration/QICK changes.

Both runs' 148 HDF5 files and 78 acquisition plots passed audit. Exact evidence
is in IDE workspace folders `q5q6_recovery_0915` and `q5q6_interleave_0915`;
canonical raw runs are `FIRST_LIGHT/ac_roaba/2026-09-15_15-44-33` and
`FIRST_LIGHT/ac_roint/2026-09-15_16-29-21`.

The completed local report is `IDE/q5q6_interleave_0915/recovery_final.pptx`,
195 slides, SHA256
`D55F2C5180EE970DE78A49D965B3C86101684377C5DAFC94BB3C0D859E27094B`.
All 195 native PowerPoint renders passed layout checks; all 19 added slides were
visually reviewed. All 176 historical slide/notes contents and 141 media hashes
were preserved. **It is not published.** The existing M: report is still the
176-slide version with SHA256 `50065883A3A09A32DED465B46EAA4D80C2A3347ECAA0C1BFB9D773C7E57D0F36`.
Automatic approval review rejected M: archival/report replacement because it
requires explicit publication authorization. A question is pending; do not retry
or work around the rejection without new authorization. Reviewed publication
script: `IDE/q5q6_publish_0915.ps1`; requires fresh archive, old/new hashes and
native checks and preserves a verified backup. No publication script ran.

Next scientific prerequisites: independently verified Q5 routing/drive evidence
and a justified intervention that yields fresh paired lines with passing repeat
controls on both sides of the visibility gap. More averaging or unchanged map
retries are unsupported. The upper-ON branch at failed bias B remains a specific
coverage gap, not evidence of an absent mode. See the saved plan for limits.

## September 15, 16:29 - short-block timing diagnostic ACTIVE

ac_roaba is terminal: guard/child/monitor exited, bothsources0A/off verified.
Allfour A windows pass detection; bothB windows fail (M2t2.175/M6t2.231).
M6Areturn shifts67.725kHz and firstcross-readout difference54.450kHz exceed
50kHzlimit. No map/J accepted. Exact assessment in IDE q5q6_recovery_0915.

NEW ac_roint/2026-09-15_16-29-21 active under guard21004/child45620,
acquisition session54503, monitor79643. PID45620 is a NEW child, not the old
discovery-map guard. Do not relaunch. IDE q5q6_interleave_0915 contains exact
plan/spec/preflight/monitor. OneheldA=-5.6566mA;32cyclesofM6/M2 withalternating
order,500reps/state/block,16000totalreps/state/readout,101points5430.7..5432.7MHz,
gain.03,20us,100us. Same QICK/controller/configs,197charpath,3600s+180guard.
Both expectedserial/CURR/.01A/0A/off/30V preflightpassed;147-fileharness and6
aggregation/controlchecks passed. This tests timing; it doesnotpromote the failed
A-B-A protocol or solveBvisibility. No automaticmap. Publishedreportstill176slides.

## September 15, 15:44 - new dual-readout A-B-A diagnostic ACTIVE

User requested a saved recovery plan and execution. Read
`docs/q5-q6-recovery-plan-20260915.md`. New ac_roaba/2026-09-15_15-44-33
is active under guard43468/child54148, acquisition session87754 and monitor52719.
Plans/spec/hashes/preflight/monitor are in IDE workspace q5q6_recovery_0915.
Do not relaunch. Earlier no-active statements describe previous closed work.
Six lower-line windows: A=-5.6566mA (M6,M2), B=-5.652mA (M2,M6), A return
(M6,M2),16000reps/state,101points,gain.03,20us,100us,one round. Exact bounds
and frozen controls in the recovery plan. No shared config/QICK/attenuator change.
Both sources passed fresh expectedserial/CURR/.01A/0A/off/30V preflight;
146-file offline check and seven diagnostic decision tests passed.
Purpose: distinguish readout-specific visibility/current dependence from simple
time variation; no assumed dark-state mechanism or physical Q5 confirmation.
No new map or J accepted. Published report still176slides.

## September 15 — report published; scientific result unresolved

**No acquisition or monitor is active.** Both Yokogawas were verified in current
mode, 10 mA range, at 0 A with outputs off and 30 V compliance after cleanup.
All earlier ACTIVE headings are historical. Never relaunch the failed plans.

The requested debugging plan was executed, but full Q5/Q6 scientific closure was
not achieved. The published report now contains 176 slides. All slides rendered
in PowerPoint with zero layout issues; the 19 new slides were visually reviewed.
The 157 historical slides, their notes, and 133 media items were preserved.

Report: `pucq4_first_light/PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx`.
SHA256: `50065883A3A09A32DED465B46EAA4D80C2A3347ECAA0C1BFB9D773C7E57D0F36`.
Evidence: `pucq4_first_light/ac_review/profile_debug_20260915/`, including
`publication.json`, the 876-file `archive_index.json`, and the prior deck at
`prior_157_slide_report.pptx`. Final documentation remains in this repository. Source references are in the
archived design evidence; the optional source-document export was not performed.

### Results and limits

- The fresh `ac_prof2` pilot passed all four windows and both center repeats.
- `ac_pmap` failed its first lower window: frozen-profile t = 2.764.
- `ac_dmap` failed its first lower window: held-out t = 3.963, below 5.
- Both maps stopped at the next acquisition boundary, preserving the upper OFF
  references already underway. Neither completed a paired bias or repeat test.
- No minimum splitting, resonance bias, or coupling J is accepted. Shared
  configuration, attenuators, and QICK programs remain unchanged.
- The audit reverified 228 prior raw files and verified eight new HDF5 files
  and four new plots. All 78 previous legacy metric records reproduce exactly
  under the original numerical-library thread settings.

The recovered PUCQ4 `Device04V2.py` and DXF establish the intended Q5/M2 design
assignment. The original author's `08042026_WeeklyUpdate.pptx` labels M2 as
`Q5?` and calls the assignment a spatial guess. Independent experimental identity
and a separately wired Q5 microwave drive remain unverified. Saved readout
tracking did not change and does not establish the cause of weaker spectroscopy.

### Next evidence required

A physically supported way to obtain repeatable paired lines across bias, plus
independent device/index/wiring confirmation. Do not increase averaging, lower
thresholds, or repeat failed plans merely to obtain a pass. The physical cause
of the weak held-out response is not established. The user requests autonomous
progress without routine permission prompts; missing device parameters and
undocumented ports must still never be guessed.

Exact plans, partial results, safe terminal states, and method details are in
[the debugging plan](../docs/q5-q6-profile-debug-plan.md) and the archive.
## September15,13:01 - both maps terminal; hardware parked

ac_dmap guard45620/child56568, acquisition session5481 and monitor8966 exited.
Guardfailed/exit1 after1282.156s because firstlower held-outt3.963223<5.
Both expected Y3/Y4serials verified CURR/.01A/0A/outputoff/30V. The already-started
upperOFF reference saved; no upperON followed. ac_pmap is also terminal.
All previous ACTIVE/STOP-pending headings are historical. Do not relaunch either
failed plan. No paired map, repeat-grid test, resonance minimum or J accepted.
Final assessment: IDE q5q6_discovery_map_0915/final_analysis/map_assessment.json.
Artifact audit and final PowerPoint publication are in progress; nohardwareactive.
## September15,12:54 - discovery map STOP requested after failed first window

ac_dmap/2026-09-15_12-39-29 first lower window failed held-out t3.963223<5.
OFFp0.428918, no dependence flag, conditional fit quality passed; fittedcenter
5432.509641+/-0.021673MHz,FWHM0.245632MHz. Frozen monitor8966 wrote STOP.
Guard45620/child56568, session5481 still active while the already-started upper
OFF reference saves. Do not relaunch or claim parked until final guard readback.
No map, shifted-grid repeat, temporal repeat, minimum splitting or J accepted.

Saved M2 tracking across both pilot visits and both maps has identicalselected
8951.210MHz tone,gain0.56,length5us,coarseminimum8951.610MHz. Curves overlap;
these ground-state scans do not measure excited-state discrimination or establish
a physical cause of weaker spectroscopy. No evidence supports changing readout
settings. Preserve both failed plans, finish artifact/report closeout, no further
automatic averaging or sweep retries. Earlier ACTIVE-plan entries are historical
except for the current boundary-stop cleanup described here.
## September 15,12:39 - ACTIVE discovery/validation map

New ac_dmap/2026-09-15_12-39-29 active under guard45620/child56568,
acquisition tool session5481, monitor session8966. Do not relaunch. Exact plan,
spec, frozen analyzers, monitor and guard/runtime records: IDE workspace
q5q6_discovery_map_0915. Read docs/q5-q6-profile-debug-plan.md for envelope.
Fresh preflight passed all8acquisition hashes and both expected sources
CURR/.01A/0A/off/30V.143-file offline verifier and canonicalIQ roundtrip passed.
This supersedes the no-active-run statement below. The old fixed-center map
remains failed and preserved; no J accepted. Published report remains157slides.
## September 15 - fixed-center map stopped; discovery method validated offline

ac_pmap/2026-09-15_12-09-11 failed its first lower-window frozen-profile
control (t=2.764). Guard13836/child60040 and monitor exited. Final guard confirms
both expected sources CURR, 0.01 A range, 0 A, output off, 30 V compliance;
elapsed876.515s. All ACTIVE entries below are historical. Never relaunch map.json.
The lower fitted center shifted +0.131 MHz from its frozen prediction. Retrospective
profile discovery on half the data gives held-out t=4.746, still below5; this
failed record remains rejected. No J accepted.

Fresh ac_prof2 pilot still passes all four windows and both repeats. Audit:
228 HDF5 files, 146 plots, no failures; all78 previous legacy metric records
reproduce exactly under the original numerical-library thread settings.
A167-slide pilot/design report is staged and native-rendered without errors,
but the published157-slide report has not yet been replaced.

A revised unknown-center method passed400 synthetic trials: 0/80 white null,
0/80 correlated null, 0/80 discovery-only spur, 0/80 drift accepted; 78/80 shifted
lines accepted. Discovery and validation use disjoint64-block halves. Planned
new scan uses32000 repetitions per ON/OFF state, restoring16000 held-out reps,
plus early shifted-grid and final temporal repeats. It has not yet started.

## September 15, 12:09 — ACTIVE seven-bias profile map

Fresh prospective pilot PASSED all4windows and both center repeats. Pilot
guard53552/child59464 and monitor session61557 exited; both sources0A/off at
completion1943.218s. Full pilot assessment is in IDE workspace
q5q6_profile_pilot_0915/pilot_assessment.json. Frozen upper t=9.438/8.237 on fresh
visits; lower t=7.931/9.561. Old peak-height rejection labels are retained.

New map ac_pmap/2026-09-15_12-09-11 is ACTIVE: guard13836,child60040,
tool session36347; monitor session65902. Never relaunch. Plan/spec/guard/runtime/
monitor results live in IDE workspace q5q6_profile_map_0915. Exact16-window
envelope and frequency table are in docs/q5-q6-profile-debug-plan.md.
Seven unique M2 paired biases (-5.700..-5.625mA) plus repeated first bias,
alternating sides. Lower16000/upper32000reps,state;101points/window;.03gain,
20us/100us/one round;14400s+180guard;196char path. Independent monitor applies
frozen controls to every completed window and requests boundarySTOP onfailure.
Final map must also pass repeated-center stability and unchanged coupled-fit
gates. No coupling result exists yet. Existing157-slide report is still published;
new report builder is prepared but has not yet been run/native-validated/published.

Source preflight passed expectedserials/CURR/.01A/0A/off/30V and all8unchanged
acquisition source hashes. Safe142-file verifier and canonicalIQ check passed.
Pilot and map acquisition/QICK/shared configs are unchanged.

New original-author evidence: Dropbox/People/Shishir/Meeting Slides/
08042026_WeeklyUpdate.pptx labels M2 asQ5? and says the closer coupled qubit was
guessed and needed confirmation. The recovered intended design mapping below
is useful source evidence, but physical experimental Q5 identity remains open.

## September 15, 11:31 — ACTIVE prospective profile pilot

NEW SOURCE LEAD: local Dropbox (not the smaller M:/malab mirror) contains the
actual PUCQ4 design, `SQUILL/FUN15-12052025/design codes/Device04V2.py` and
`output_dxf/PUCQ4.dxf`. Geometry matches the characterization layout. Intended
design mapping is Q1->M3,Q2->M4,Q3->M1,Q4->M5,Q5->M2,Q6->M6 from exact nominal
resonator assignments. The characterization explicitly flags measured indexing
uncertainty, so do not conflate design assignment with independent experimental
confirmation. See the new plan's provenance section and IDE pilot/design_review.
Earlier statements that no chip-specific design source was found are historical.

User requested a new debugging plan and execution. Read
`docs/q5-q6-profile-debug-plan.md` before continuing. New offline profile analysis
passes all eight retrospective windows, including upper features; original
rejections remain unchanged. This motivated a fresh prospective fixed-profile
test, not an averaging escalation. Guard53552/child59464, tool session54867;
run FIRST_LIGHT/ac_prof2/2026-09-15_11-31-36. Plan/guard/runtime/log/spec and frozen
analysis sources are in the IDE workspace `q5q6_profile_pilot_0915/` folder.
Recheck the live handles and files; do not relaunch the plan.

M2/index1 readout, physical identity unassigned; Q6/M6/index5/Yoko4 flux drive at
-5.6566mA. Two lower/upper visits,16000reps/state/window; lower5430.7-5432.7MHz/
101points, upper5443.2-5445.8MHz/131points;gain.03,20us,100us,one round.
3600s+180guard;197char maximum path; .5mA/s ramps,Y3zero/off,Y4zero/offcleanup.
Fresh serial/CURR/.01A/0A/off/30V preflight passed; all8 acquisition source hashes
match ac_r6new. Offline140-file verifier, raw roundtrip and synthetic checks pass.
All four windows must pass the frozen profile, reference, stationarity, parameter
and repeated-center gates before considering a map. No automatic map is launched.
The published157-slide report predates this active test; do not claim closure.

## September 15 renewal: read this before the historical handoff below

The renewed bounded investigation and report update are complete. No acquisition
is active. New run ac_r6new/2026-09-15_08-43-58 ended exit0/2879.5s with both
sources verified CURR/.01A/0A/outputoff/30V. Guard29520, child61344 and offline
helper7924 have exited. Never relaunch right_m6_renewal_0915.json.

The published report is now157slides, SHA256
DBA2E7CA4AC8E4509523C9DD0B9831A87BC9AAFD43D9B0A881CD9E872634E7D6.
New authoritative evidence is FIRST_LIGHT/ac_review/renewal_20260915/:
full_analysis/analysis.json; raw_audit.json; sensitivity_v2/sensitivity.json;
combined_block_analysis.json; metric_preservation.json; publication_receipt.json;
native_final/native_validation.json. Prior143slides are preserved there.

New M6/-5.59mA/.03 result: lower5436.352749361+/-.001591724MHz,
FWHM.183947718,SNR37.314898 accepted; upper5455.329437014+/-.023193449MHz,
FWHM.612135478,SNR4.262093 REJECTED for SNR at32000 averages, clean references.
The previously missing M6 high-averaging window was tested. Its failed-upper
stop rule was honored; no blind map or extra averaging followed. Seven paired
biases/minimum/J remain unavailable. Only two distinct detuned paired biases
are accepted, as below. This new candidate is not a third accepted pair.

Central review: no accepted upper quarter among12 M2/M6 quarters; M2 saved
readout settings match between ctrhi/cmax. All raw ON/OFF means and quarter
recombination reproduce saved averages. Signed amplitudes retain quarterwise
signs but change magnitude/noise between visits. This assigns no physical cause.
Additional Documents/Desktop and visible M:/malab source searches supplied no
established PUCQ4 mapping or independent drive calibration. Physical Q5 identity
and an independent spatial drive still require external device/wiring evidence.

23guardedruns/78spectra/218canonicalHDF5/140plots audited; all76 earlier spectral
metrics identical. Native157-slide validation and visual review passed. These
are validated bounded results with unresolved scientific prerequisites, not a
claim that the scientific targets are solved. See the latest resolution log/state
for exact details; all older counts, active labels and conditional plans below
are historical. Do not repeat the exhausted scans without new evidence and a
specific hypothesis that respects their recorded stop rules.

## User objective and working preferences

Offline follow-up: renewal_20260915/right_m6_full_blocks/block_analysis.json
checks all four chronological quarters of the new M6 upper trace (8000 ON
repetitions each). None is accepted: quarters 1/2 have no detector candidates;
quarters 3/4 have rejected candidates at 5455.314854/5455.356183 MHz with
SNR 4.093185/3.136993. All four lower quarters remain accepted and their metrics
exactly reproduce the earlier lower-only analysis. The shared full OFF reference
makes quarter errors correlated; reduced averaging limits this diagnostic.
No drift mechanism or additional accepted pair follows. This supplementary null
check leaves the validated 157-slide report and its conclusions unchanged.

Continue resolving the remaining Q5/Q6 tasks, then update the existing PowerPoint.
The user explicitly requested this handoff so another chat can continue. Their
earlier instruction was: "make a plan to resolve the unresolved things, and then
execute resolving them. update the powerpoint when you complete these tasks."
They strongly prefer autonomous work without routine questions or permission
prompts. They previously authorized the bounded lab investigation described below.
Carry that authorization and preference forward within its documented scope;
do not interpret it as permission to invent wiring, exceed device limits, or
promote unsupported scientific conclusions.

This handoff was created as a documentation-only task. No new acquisition was
started to create it. The last investigation is closed, not running in the
background. Scientific targets remain unresolved despite completed measurements
and reporting. The user's new request is to continue resolving those targets,
not merely to repeat the old closeout.

## First actions

1. Work in `C:\_Lib\MaLab_fridge_QZE_cooldown`. Inspect `git status` and preserve
   all existing user changes. The tree is extensively dirty; many experiment
   and analysis files are untracked. Do not reset, clean, or broadly refactor it.
2. Read `AGENTS.md`, `harness_engineering/README.md` and its mandatory safety
   sections, then `harness_engineering/PUCQ4_STATE.md`, `SOURCES.md`,
   `VERIFICATION.md`, and `FAILURE_MODES.md`. Follow task routing and define
   acceptance criteria with `TASK_TEMPLATE.md`.
3. Read the latest sections of `docs/pucq4-crossing-resolution-plan.md` and the
   machine-readable final evidence below. Older ACTIVE/PREPARED entries in the
   chronological documents are historical, not instructions to resume a process.
4. Make a new evidence-based plan for the remaining questions. Start with useful
   offline analysis and missing-source review. A new lab test must distinguish
   a specific hypothesis or fill a defined coverage gap; do not simply repeat
   the failed central scan with more averaging.

## Authoritative artifacts

Define `FIRST_LIGHT` as this exact directory:

```text
M:\_Data\20250822 - Olivia\pucq4_run_started_Aug_3\PUCQ4\pucq4_first_light
```

Define `EVIDENCE` as `FIRST_LIGHT\ac_review\resolution_20260914`.

| Artifact | Location / purpose |
| --- | --- |
| Published PowerPoint | `FIRST_LIGHT\PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx`; 143 slides |
| Final closeout | `EVIDENCE\task_closeout.json`; explicitly says scientific targets are not all resolved |
| Final analysis | `EVIDENCE\final_analysis\analysis.json`; exact list of all 22 source manifests, metrics, rejected candidates, pair reviews and plots |
| Raw-data audit | `EVIDENCE\evidence_audit_final.json`; raw-file hashes, expected plots, source provenance and terminal guards |
| Report findings | `EVIDENCE\findings_final.txt` |
| Report validation | `EVIDENCE\native_final\native_validation.json` and `publication_receipt.json` |
| Temporal analysis | `EVIDENCE\combined_block_analysis.json`; links independent M4/M2 block analyses without pooling data |
| Final code/document snapshots | `EVIDENCE\final_sources\hashes.json` and copied files |
| Full chronological plan | Repository `docs/pucq4-crossing-resolution-plan.md` |
| Previous report backup | `FIRST_LIGHT\ac_review\final_20260914\PUCQ4_Q5_Q6_94slides.pptx` |
| Original characterization | `C:\Users\Ma Quantum Lab\Downloads\PUCQ4 Initial Characterization (1).pptx`; unchanged |

Published report SHA256:
`0B3AB5B08AC4C5AA496FA3ACCB84BA08884397FD4E3E86894A6517A09678DA0F`.
Original Downloads deck SHA256:
`626B42201DD6C2E02417806AAE34EF94122196F2D4E88D2DAC59B9492B44EC71`.
Final analyzer SHA256:
`DB3C959630627FDAD9F5726F74A3BDE955A9928A05DA3DB4D8E4DD1F524944D7`.

The final analysis was generated at `2026-09-15T00:36:05Z`, which is September
14 evening in local time. Audit: 22 successful guarded plans, 76 drive-on/off
spectra, 213 canonical HDF5 files (4.295914 GiB), 137 expected acquisition plots.
49 individual windows have one accepted line; 27 have none. Three pair
observations span only TWO distinct biases. All 143 report slides rendered in
native PowerPoint without detected overflow; key slides were visually reviewed.

## What is established

Physical Q6 is M6 / Python index 5 / Yoko4 / DC A5. Physical Q4 is M5 /
Python index 4 / Yoko3 / DC D4. **M5 is not physical Q5.** M1-M4 physical
identities remain unassigned in the original device table.

M2 transitions were recovered at Y4 -5.8, -5.4 and -4.8 mA, at respective
frequencies 5439.864905, 5438.068054 and 5438.540998 MHz with drive gain .007.
Differences between those different-current single lines are not splittings.

Accepted pairs from matched drive settings and explicit disjoint windows:

| Y4 setting (mA) | Drive gain | Lower (MHz) | Upper (MHz) | Separation (MHz) | Run |
| --- | --- | --- | --- | --- | --- |
| -5.8 | .007 | 5402.791626 +/- .007499 | 5439.862749 +/- .003573 | 37.071123 +/- .008306 | `ac_pair1/2026-09-14_15-29-43` |
| -5.8 | .015 | 5402.776076 +/- .005000 | 5439.870762 +/- .003572 | 37.094685 +/- .006145 | `ac_bridge/2026-09-14_16-35-42` |
| -5.74 | .015 | 5416.233093 +/- .002926 | 5440.546944 +/- .004619 | 24.313851 +/- .005468 | `ac_bcf/2026-09-14_17-02-42` |

These pairs are resolved through M2. M6 corroborates the lower line in ac_pair1.
Each run path in this file is relative to FIRST_LIGHT; its source manifest is
`<run>/documentation/manifest.json`. Windows were sequential, not simultaneous;
intervening gaps were unmeasured. Errors are conditional statistical fit errors.
These separations are NOT minimum splittings and do not establish J or Q5 identity.

## Remaining tasks and concrete next work

### 1. Establish central and opposite-side branch visibility

The unresolved central upper feature is near 5444.5 MHz at recorded Y4
-5.656600 mA. Relevant completed tests:

| Run | Measurement | Result |
| --- | --- | --- |
| `ac_ctrlo/2026-09-14_15-55-29` | M2 and M6, gain .007, 8000 averages; lower and upper windows | Neither line recovered. Lower positive control was lost, so this null cannot exclude an upper mode. |
| `ac_ctrhi/2026-09-14_17-32-32` | M2, gain .03, 32000 averages | Lower 5431.727525 MHz accepted, SNR 6.7414. Upper 5444.489811 +/- .019892 MHz, FWHM .428827 MHz, SNR 4.2007: rejected for SNR. |
| `ac_m6hi/2026-09-14_18-29-53` | M6, gain .03; lower 16000, upper 32000 | Lower 5431.697853 MHz accepted, SNR 8.7590. Upper 5443.2-5445.8 MHz: no candidate, vector SNR 2.7084. |
| `ac_cmax/2026-09-14_19-44-57` | M2, gain .03; lower 32000, upper 64000 | Lower 5431.735514 +/- .006458 MHz accepted, FWHM .170163 MHz, SNR 7.9745. Upper 5444.498749 +/- .019163 MHz, FWHM .359666 MHz, SNR 3.7118: rejected for SNR. |
| `ac_right/2026-09-14_18-58-07` | M2 at -5.59 mA, gain .03; lower 8000, upper 32000 | Lower 5436.331130 +/- .002384 MHz accepted, FWHM .197491 MHz, SNR 30.1265. Upper 5452.5-5458.5 MHz: no candidate, vector SNR 2.4944. |

All high-power tests above have clean references. ac_cmax upper has 18 samples
per FWHM, residual .9052 noise units, full-rank fit and no active bounds; SNR
alone fails. A repeated fitted center does not turn it into an accepted line.
The anticipated square-root averaging improvement was not observed. Keep both
rejected traces and do not assign a cause without evidence.

Useful next offline work: compare saved raw repetition blocks, signed I/Q
amplitudes, reference behavior, readout updates and acquisition times across
ac_ctrhi/ac_m6hi/ac_cmax. Test whether temporal or readout changes can explain
the sensitivity discrepancy. Preserve the original metrics and gates. If a new
analysis method is justified, validate it independently and retain the old result;
do not lower thresholds or selectively pool visits to manufacture acceptance.

A remaining coverage gap is high-averaging M6 readout at -5.59 mA. Earlier
`ac_up1/2026-09-14_11-06-02` measured M6 and M2 upper 5450-5480 MHz with only
4000 averages / 50 kHz steps and accepted no candidate. M6's lower control in
`ac_fm2/2026-09-14_11-28-18` at this bias was strong: 5436.328830 MHz,
SNR 24.9625 at 4000 averages. A M2 null is not a global M6 null.

There is a PREPARED, NEVER LAUNCHED `right_m6_0914.json` in the TEMP directory
below: lower 5435.2-5437.6 MHz/121 points/8000; upper 5452.5-5458.5 MHz/
301 points/32000; gain .03, 20 us pulse, 100 us relaxation, equal off/on per
window, deadline 4500 s plus 180 s guard. Its explicit prerequisite was an
accepted central pair, which FAILED; it was correctly skipped. Do not launch
that old plan as though its prerequisite passed. In this renewed investigation,
evaluate whether a newly framed readout-specific question warrants a fresh
bounded plan; document the scientific justification and changed decision.
This gap is not evidence that such a test will succeed.

### 2. Determine minimum splitting, resonance current and J

These remain unaccepted. First obtain a usable pair-visibility protocol.
Then measure at least seven UNIQUE paired biases with the minimum bracketed by
at least two points on either side. Use matched drive settings and explicit
stability controls. Do not count the two powers at -5.8 mA as two biases.

`tools/analyze_pucq4_crossing.py` currently groups fits by source manifest,
gain and readout, with supported matched M6/fixed-row corroboration. It does
not automatically justify mixing guards, gains or readouts into a global map.
Tagged disjoint windows are supported; repeated/overlapping visits are not
silently merged. Any needed aggregation requires explicit provenance and tests.

Current coupled model: linear bare tunable frequency, constant bare fixed
frequency, and constant J. Acceptance includes full rank/no active bounds,
J > 3 SE, reduced chi-squared < 10, interior minimum, and a fixed-frequency
slope alternative test (delta AIC < 6). Inspect current code for exact gates.
If real data reject this model, retain the rejection. A different model needs
physical justification and validation, not a gate change to force a number.

### 3. Resolve physical Q5 mapping and independent drive routing

The original two-slide characterization explicitly labels M1-M4 physical
identities with question marks. Its layout places Q5 near Q6 but does not
identify M2 as Q5. A scoped search of Downloads and C:\_Lib found no additional
PUCQ4 routing document. Other wiring/measurement decks concerned unrelated
devices or generic setups. Do not repeat those sources as new mapping evidence.

Look for genuinely new design, GDS/layout-to-resonator, packaging or wiring
records, or a uniquely identifying measurement. Do not infer identity solely
from a shared peak or apparent coupling. Current qubit and ef software channels
both use shared DAC1; DAC0 is the resonator drive, ADC0 the readout. An independent
spatial drive requires a documented wired path and calibration. Do not invent
ports, repurpose DAC0 or claim physical rewiring was performed remotely.
If external records or physical access are essential, name the exact missing
prerequisite rather than declaring this task solved.

### 4. Explain temporal variation / weak-feature mechanism

Observed, cause unassigned: the M2 lower line at -5.74 mA shifted +25.377 kHz
between same-gain visits. Its four chronological quarters span 27.215 kHz;
three accepted upper quarters span 2.738 kHz. The right-side lower control's
four quarters span 9.022 kHz. Earlier M4 held-current blocks span about .166 MHz;
its A-B-A control did not resolve the intended bias response.

Use `tools/analyze_pucq4_crossing_drift.py` and the combined block report.
Raw loop order and full-average normalization were checked. Blocks share a full
off reference; their errors are correlated, so descriptive spans are not an
independent drift significance. Do not assert a dark-state, two-photon, drift
source or entanglement mechanism from these observations alone.

## Hardware state, envelope and execution

Last VERIFIED terminal state, September14 about20:35 local: Yoko3 and Yoko4
both CURR, .01 A range, 0 A setting, output off, 30 V compliance. ac_cmax guard
40956 / child59996 completed exit0 in2989.984 s; neither remained running.
This is historical evidence, not a live query. Recheck process ownership and
fresh parked-source preflight before any new acquisition. Preserve unrelated
legacy processes; do not kill Python processes indiscriminately.

- Yoko3: `192.168.1.73`, Q4/M5/D4, serial90Z114063; hold zero/off.
- Yoko4: `192.168.1.77`, Q6/M6/A5, serial90Z114062.
- Source readback is `:SOURCE:LEVEL?`, not an independent ammeter.
- Device reference hard boundary: +/-10 mA. Existing crossing controller is
  narrower: Y4 -5.8 through -4.8 mA; ramp .5 mA/s; no extra DC hold.
- Controller: 5300-5660 MHz, span <=250 MHz, 101-12501 points, 500-64000 reps,
  0 < drive gain <=.03, 20 us pulse, 100 us relaxation, one round.
- Keep unchanged section004 spectroscopy and section002 resonator programs.
  Use saved/calibrated row-specific readout settings and the existing tracker.
- Nominal attenuation arguments: readout10/15, qubit5/4, ADC17 dB. Setters are
  disabled; actual attenuations were not independently read back. Do not change them.
- Orchestrator: `pucq4_13_avoided_crossing.py`; guard:
  `tools/pucq4_crossing_guard.py`. Read both before using a new plan.
- Use a fresh plan filename/label; never overwrite a launched plan, guard record
  or raw artifact. Guard checks a deadline plus180 s and records final parking.
  `<run>/STOP` is checked at acquisition boundaries after returned data are saved.
- Canonical raw data: `<run>/study_data/Data_h5/{res_ge,qspec_ge}/`.
  Validate longest generated Windows path <260 characters and verify plots AND
  HDF5 after acquisition. Abort source/acquisition/save/path/deadline failures.
- Never import `round_robin_benchmark.py` as a test: it starts hardware work.
  No concurrent controllers, no diagnostic calibration promotion, no raw rewrites.

Temporary plans/logs (may disappear; archived run plans are authoritative):
`C:\Users\MAQUAN~1\AppData\Local\Temp\pucq4_cross_20260909`.
Runtime and guard files are adjacent to each plan. Guard logs are archived under
the run's documentation. Use `Start-Process -WindowStyle Hidden` for background
helpers. Do not auto-launch the old conditional plan.

## Environment, verification and report completion

Follow the python-tools skill; before EVERY Python command, query PyCharm
`get_python_environment` for the relevant module. Last observed SDK was Python
3.9.13 at `C:\Users\Ma Quantum Lab\anaconda3\python.exe`. Activate its Conda
runtime or add its root, Library\bin and Scripts to PATH to avoid NumPy DLL errors.
Target Python3.9 syntax. No subagents unless current user/repository instructions
explicitly authorize delegation. Follow the actual next-chat tool permissions.

Safe tools (query the SDK first):

```text
tools/harness_check.py
tools/check_pucq4_crossing_analysis.py
tools/check_pucq4_crossing.py NEW_PLAN.json
tools/analyze_pucq4_crossing.py MANIFEST... --output NEW_ANALYSIS_FOLDER
tools/build_pucq4_crossing_ppt.py MANIFEST... --analysis ANALYSIS_JSON --output REPORT --findings FINDINGS_TXT --conclusion TEXT --drift-analysis BLOCK_JSON
```

The last 138-file offline verifier, scientific regression suite and whitespace
check passed. The original detector/fitter was unchanged in the last continuation.
New analysis work preserves actual saved current, requested current separately,
actual QICK axes, disjoint-window provenance and per-window averaging. A requested
-5.656667 mA was recorded as -5.656600 mA; do not lose this distinction. Some
other numeric differences are merely floating-point epsilon, not physical offsets.

Final-report workflow: preserve the currently published143-slide report before
updating; build a staged report from validated results, including rejected/null
tests and remaining limits. Use `EVIDENCE\validate_report.ps1` for native export
and approximate overflow checks, plus visual review. It preserves user-open
presentations and closes its own. In the prior session it was run using a
process-local PowerShell execution-policy override; no global policy was changed.
Do not leave draft/pending text visible or stop with an acquisition active.

Raw audit helper: original TEMP `audit_crossing_closeout.py`, also copied into
EVIDENCE. CLI: `ANALYSIS_JSON --repo REPO --output NEW_AUDIT_JSON`. It refuses an
existing output. Run the TEMP helper when output is in EVIDENCE; running the copy
inside its output folder caused a SameFileError during helper self-copy after
the audit result was written. That was corrected by a successful fresh invocation.
PowerShell UTF-8 output can add a BOM; write plan JSON with
`IO.File.WriteAllText(..., new Text.UTF8Encoding(false))` and validate it.

Done means measured/validated resolutions where possible, exact unresolved
prerequisites where evidence is still missing, saved raw/plot provenance, verified
terminal hardware state, updated PowerPoint, and current repository state/log.
Do not call all scientific tasks resolved merely because a report is complete.
