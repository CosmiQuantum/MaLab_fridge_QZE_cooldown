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


## September 15 - recovery diagnostics terminal

No acquisition or monitor from this task is active; ACTIVE headings below are
historical. Both guarded runs completed with both expected sources verified
0 A/output off, CURR, 10 mA range and 30 V compliance. No QICK/shared config or
attenuator changes were made by this recovery task.

The A-B-A lower-line diagnostic recovered both M6/M2 at -5.6566 mA, lost both
at -5.652 mA, and recovered both on return. M6 return shifted 67.725 kHz and
initial cross-readout centers differed 54.450 kHz: both failed 50 kHz limits.
The subsequent 32-cycle alternating M6/M2 experiment at -5.6566 mA completed
all 64 blocks, retaining 16000 repetitions/state/readout. M6 t=9.684 passed;
M2 t=3.945 failed. Both time-half checks passed; paired comparison did not.
No usable full-map protocol, J, minimum, Q5 identity or mechanism was established.

Audits passed: 148 new HDF5 files and 78 acquisition plots. Details and next
prerequisites: `docs/q5-q6-recovery-plan-20260915.md` and the top handoff entry.
The 195-slide local report passed native rendering/layout and 19-slide visual
review, preserving all previous 176 slides/notes and 141 media hashes.
Publication/archival on M: was rejected by automatic approval review pending
explicit publication permission; existing published report remains 176 slides.
Do not bypass that rejection. Local report: IDE/q5q6_interleave_0915/recovery_final.pptx.

## September 15, 16:29 - interleaved timing diagnostic ACTIVE

ac_roaba completed2619.157s, bothsourcesverified0A/off; guard43468/child54148
andmonitor exited. AllfourA detectionspass;bothB fail. M6return67.725kHz and
initialcross-readout54.450kHz exceed50kHzlimit; no map/J accepted.
NEW ac_roint/2026-09-15_16-29-21,guard21004/child45620,acquisition54503,
monitor79643. Do notrelaunch; PID45620 isreused for thisNEWchild. Exactplans
inIDE q5q6_interleave_0915 and docs/q5-q6-recovery-plan-20260915.md.
Same-total-shot counterbalanced32x500block M6/M2 comparison atheld-5.6566mA.
Freshsourcepreflight/147-fileverifier/6offlinechecks pass; no config/QICKchange.
All earlier ACTIVE labels describe historical runs unless explicitly noted here.

## September 15, 15:44 - new recovery diagnostic ACTIVE

New ac_roaba/2026-09-15_15-44-33 under guard43468/child54148;
acquisition session87754, monitor52719. Do not relaunch. Saved exact plan:
docs/q5-q6-recovery-plan-20260915.md; machine plans/spec/preflight in IDE
q5q6_recovery_0915. A-B-A dual-readout lower-line diagnostic, not a full map.
Fresh bothsource expectedserial/CURR/.01A/0A/off/30V preflight passed.
All8 acquisition hashes unchanged;146-file verifier and7 decision checks pass.
Older no-active statements below are historical. No new J or Q5 identity claim.

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

# PUCQ4 persistent state
## September 15, 12:09 — ACTIVE seven-bias fixed-profile map

ac_prof2 pilot complete and PASSED: all4fresh windows, bothcenter repeats;
1943.218s,guardexit0,bothsources0A/off. New ac_pmap/2026-09-15_12-09-11
ACTIVE under guard13836/child60040,tool session36347,monitor session65902.
Plan/spec/runtime/guard/checkpoints: IDE workspace q5q6_profile_map_0915.
Check live handles and do not relaunch. Exact table/envelope/stop rules:
docs/q5-q6-profile-debug-plan.md. Sevenunique M2 paired biases plus start-bias
repeat,16windows;16000/32000reps,.03gain,20us/100us,101points;14400s+180guard.
Frozen monitor stops on failedwindow; finalstability and unchangedcoupledmodel
gates required. No J accepted. Pilot/history ACTIVE entries below are superseded.

New Dropbox PUCQ4 design source establishes intended Q5/M2 mapping, but original
author's August4 deck explicitly labelsQ5? and calls the assignment unconfirmed.
Do not silently promote measured physical identities. Report157slides remains
published until fresh evidence is audited, rendered, reviewed and republished.

## September 15, 11:31 — ACTIVE fixed-profile pilot

New authorized debugging pilot ac_prof2/2026-09-15_11-31-36 is active under
guard53552/child59464 (tool session54867). Check live handles before acting.
Plan and frozen spec: IDE workspace q5q6_profile_pilot_0915/pilot.json and
frozen_spec.json. Never relaunch. Exact envelope and new plan are at the top of
docs/Q5_Q6_NEXT_CHAT_HANDOFF.md and docs/q5-q6-profile-debug-plan.md.
Two central M2 lower/upper visits atY4-5.6566mA,16000reps/state/window,unchanged
.03gain/20us/100us;3600s+180guard,zero/offcleanup. The earlier stopped scans
remain historical. New independent ON/OFF block analysis motivates a prospective
profile test; it does not replace old rejection labels or establish Q5/J.

## September 15, 09:38 local - renewal complete, report published

No crossing acquisition or guard is active. The M6 coverage run ac_r6new/
2026-09-15_08-43-58 completed exit0 in2879.5s; both sources verified expected
serials/CURR/.01A/0A/outputoff/30V. Guard29520/child61344 and offline helper7924
have exited. All ACTIVE entries below are historical; do not relaunch old plans.

M6 at Y4-5.59mA/gain.03: lower5436.352749361+/-.001591724MHz,
FWHM.183947718,SNR37.314898 accepted at8000 averages. Upper at32000 gives
NEW REJECTED candidate5455.329437014+/-.023193449MHz,FWHM.612135478,
SNR4.262093. SNR alone fails; both references pass. Preserve the predeclared
failed-upper stop rule. No new pair, minimum, resonance-current or J acceptance.
Two unique detuned paired biases remain; a usable seven-bias bracket is missing.
Physical Q5 mapping and independent spatial drive still need PUCQ4 design/wiring
evidence. Central raw-quarter/readout review constrains recorded-setting and
coarse phase-reversal explanations but assigns no physical mechanism.

Published157-slide PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx SHA256:
DBA2E7CA4AC8E4509523C9DD0B9831A87BC9AAFD43D9B0A881CD9E872634E7D6.
Native157/157renders/nooverflow and key-slide visual review passed. The prior143
slides are backed up. All76 earlier metrics reproduced unchanged. Final audit:
23guardedruns,78spectra,218HDF5(4.611908GiB),140expectedplots; all source hashes,
raw persistence and terminal guards pass. Exact evidence and publication receipt:
ac_review/renewal_20260915/. Shared calibration and QICK programs unchanged.
Scientific targets remain unresolved; report publication is not scientific closure.

## September 15 renewal - ACTIVE M6 coverage test

Fresh right_m6_renewal_0915.json launched 08:43:57 local, guard29520/child61344,
run ac_r6new/2026-09-15_08-43-58. Query these processes and the adjacent TEMP
guard/runtime files before acting. Do not relaunch from this checkpoint.
Y4-5.59mA; M6 lower5435.2-5437.6MHz/121points/8000, upper5452.5-5458.5MHz/
301points/32000;gain.03/20us/100us/one round/equal off-on perwindow.
4500s+180guard,.5mA/s,Y3zero/off,Y4zero/offafter;197charHDF5path.
Fresh preflight both expectedserials/CURR/.01A/0/off/30V. All source hashes
match ac_cmax and offline safety/scientific checks pass. Full envelope and
changed decision from the old skipped conditional plan are in the resolution log.
Lower is accepted5436.352749361+/-.001591724MHz,width.183947718,SNR37.3149,
cleanreference; upper result is not yet available. This is not terminal cleanup.

New offline central quarter review: no accepted upper quarter among all12;
M6 lower passes4/4. M2 recorded readout settings match between ctrhi/cmax.
Signed amplitude/noise details, raw-average validation and six diagnostic plots
are in ac_review/renewal_20260915/sensitivity_v2; two block reports are adjacent.
Physical mechanism, Q5 identity, independent spatial drive, and minimum/J remain
unresolved. The published143-slide deck remains the prior report until new result
review and native validation. All September14 ACTIVE entries below are historical.
## September 14, 20:41 local — bounded investigation closed, report published

No acquisition is active. All 22 guarded plans completed with exit 0 and both
sources in CURR mode, 0.01 A range, 0 A setting, output off. The latest ac_cmax
upper candidate remains rejected at SNR 3.7118 despite 64000 averages; its
lower control passes. Conditional right_m6 was not launched. Two distinct
detuned biases have accepted pairs; minimum splitting/J, physical Q5 mapping
and an independently routed spatial drive remain unresolved.

The updated **143-slide** `pucq4_first_light/PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx`
is published and matches the natively rendered/reviewed copy, SHA256
`0B3AB5B08AC4C5AA496FA3ACCB84BA08884397FD4E3E86894A6517A09678DA0F`.
Final audit passes 76 spectra, 213 HDF5 files, 137 expected plots and source
provenance. Exact evidence: `ac_review/resolution_20260914/`; disposition and
remaining prerequisites: `docs/pucq4-crossing-resolution-plan.md`.
All ACTIVE/PREPARED entries below are historical; do not resume their plans
as though an acquisition were still running.

Latest September14 20:35: ALL ACQUISITIONS COMPLETE. ac_cmax exit0/2989.984s,
bothsourcesCURR/.01A/0A/output-off; no recent Python acquisition controller.
LowerM2/-5.656600mA accepted5431.735513621MHz/SNR7.9745. Upper at64000/.03
repeats weak candidate5444.498748865+/-.019163206MHz,width.359666479,
SNR3.711767: REJECTED solely for SNR,cleanreference,18samples/FWHM,
residual.905212,rank2/no active bounds. No central pair. Raw plot reviewed.
Conditional right_m6_0914/ac_r6hi was NOT launched because its central-pair
prerequisite failed. No further blind maps. Two distinct detuned paired biases
(-5.8,-5.74mA) remain accepted; seven-bias/minimum/J and physicalQ5 assignment
remain unresolved. Final audit and PowerPoint closeout now in progress.
Earlier ACTIVE/PREPARED entries below are historical, not current instructions.

Latest20:00: ac_cmax lower ACCEPTED5431.735513621+/-.006457948MHz,
width.170162996,SNR7.9745,cleanreference. Upper64000/.03 nowacquiring;
guard40956/child59996,run2026-09-14_19-44-57 remainsACTIVE.
Prepared/validated but NOT launched right_m6_0914/ac_r6hi is conditional on
an accepted central pair: opposite-sideM6/-5.59/.03,lower121pts8000,
upper5452.5-5458.5MHz301pts32000,20us100us/equalrefs,4500s+180guard,
zero/off. M6 atthisbias has only the earlier4000-average upper survey;
M2's32000 null does not exclude a readout-specific M6 response. Skip extension
if central_max upper is unaccepted; then publish the bounded result with limits.
Full prerequisites/envelope/rationale in docs/pucq4-crossing-resolution-plan.md.

Latest September14 19:44: ac_right completed exit0/2729.562s, bothsources
CURR/.01A/zero/off. LowerM2/-5.59 accepted5436.331129599MHz,SNR30.1265;
upper5452.5-5458.5MHz/32000/.03 has no candidate,vectorSNR2.49436,
cleanreference. No opposite-side pair. Both raw plot and terminalguard reviewed.
Now ACTIVE central_max_0914/ac_cmax: final targeted M2/-5.656600mA candidate
confirmation, upper5443.2-5445.8MHz/131pts/64000 reps (existing maximum),
lower5430.7-5432.7/101pts/32000,.03gain20us100us/equalrefs,4200s+180guard,
zero/off cleanup. No blind map if upper remains unaccepted. See guard/runtime
for live PID/run. Prior active entries below are historical. Final PPT pending.
Offline review also found descriptive within-trace M2 center spans27.215kHz
in ac_bcf lower and9.022kHz in ac_right lower; common-reference block errors
are correlated. combined_block_analysis.json collects independent M4/M2 plots.

Latest September14 19:00: ac_m6hi completed exit0/1607.734s, both sources
CURR/.01A/0A/output-off. Upper5443.2-5445.8 MHz has no candidate, vectorSNR
2.70837, clean reference; accepted lower5431.697853 MHz retained. M2 central
candidate5444.489811 MHz remains rejected at SNR4.200675. No central pair.
Now ACTIVE right_visibility_0914/ac_right, guard13696/child55692, M2/-5.59mA,
gain.03: lower5435.2-5437.6/121pts/8000; upper5452.5-5458.5/301pts/32000,
20us/100us/one round/equal off-on per window,4500s+180guard,zero-off cleanup.
Original hardware/program/tracker controls retained. Full prior review and
parked-state verification completed before launch. No report publication yet.

Latest September14 ~18:38: ac_ctrhi completed exit0/3300.375s, bothsources
zero/off. Upper M2 candidate5444.489811332 MHz/.03/32000 haswidth.428826934,
SNR4.200675, cleanreference and otherwise passing checks; REJECTED forSNR.
NowACTIVE ac_m6hi/2026-09-14_18-29-53, central_m6_0914.json,
guard58580/child18260. Unlaunched plan was narrowed before launch to the
measured candidate: upper5443.2-5445.8 MHz/131points/32000; lower unchanged
5430.7-5432.7 MHz/101points/16000, .03gain20us100us/equalrefs perwindow.
Deadline3600s+180guard; original controls/zero-offcleanup. LowerM6 ACCEPTED
5431.697853016 +/- .005152324 MHz,width.162832365,SNR8.7590,cleanreference.
Upper nowacquiring. The ~30kHz difference from earlier M2 combines visit/RO
changes; retain systematic uncertainty. right_visibility remains prepared.
Latest completed19-run/70trace analysis: resolution_20260914/central_m2_checkpoint/.
Per-window averaging/source-request metadata nowexplicit; scientific checkspass.

September14 ~18:16: ac_ctrhi still ACTIVE, guard13892/child60576, final M2
upper5442-5448 MHz driven trace about halfway complete. Lower control passed.
New PREPARED/UNLAUNCHED central_m6_0914/ac_m6hi is conditional on an
unaccepted M2 upper result: same bias/gain/frequency windows via M6, lower
16000 reps/101points, upper32000 reps/301points,5400s+180guard,zero/off.
M6's prior central lower control was stronger at the same averaging; this
checks a distinct readout limitation rather than inferring a global null from
M2 alone. Validated offline. Review M2 result and parked state before launch;
skip M6 cross-check if M2 upper is accepted. right_visibility_0914 remains
prepared for the opposite side, adaptable to the accepted readout evidence.
All details and conditional criteria are in the resolution plan. Completed
18-run audit passed193HDF5/125plots; unpublished draft131slides is readable.

September 14, 17:47: ac_ctrhi remains ACTIVE, guard13892/child60576.
Central M2 lower positive control ACCEPTED at -5.656600 mA/.03/32000:
5431.727524559 +/- .005139544 MHz, width .141231591 MHz, SNR6.7414,
7.068 samples/FWHM, clean reference. Upper5442-5448 MHz/301points/32000
is now acquiring, about40minutes off/on. Latest partial analysis is
ac_ctrhi/2026-09-14_17-32-32/documentation/analysis_p1.
Prepared, NOT launched: right_visibility_0914/ac_right at -5.59 mA through M2,
gain .03, lower5435.2-5437.6 MHz/121points/8000 thenupper5452.5-5458.5 MHz/
301points/32000,20us100us/equalrefs perwindow,4500s+180guard,zero/off.
It improves ac_up1 upper sensitivity8x and retains a measured strong lower
control; conditional upper locator5455.606891MHz is not a calibration/J.
Must review central result and parking before launch. If both central/opposite
upper tests fail with accepted lower controls, do not launch blind repeated maps.

Latest September 14, 17:33: ACTIVE central_highpower_0914/ac_ctrhi,
guard 13892 / child 60576, run 2026-09-14_17-32-32. M2 at recorded/requested
-5.656600 mA, gain 0.03, 32000 reps, lower 5430.7-5432.7 MHz/101 points
then upper 5442-5448 MHz/301 points, 20 kHz grid, 20 us/100 us/equal refs.
5400 s deadline + 180 s guard; original source/program/tracker controls and
zero/off cleanup. No other controller. Do not stop while this run is active.
ac_bcf completed exit0/1687.813s, bothsources zero/off. Accepted matched M2
pair at -5.74 mA/.015: lower 5416.233092832 +/- .002925923 MHz and upper
5440.546943643 +/- .004619457 MHz; separation24.313850811 +/- .005468127 MHz.
This gives TWO distinct paired biases including -5.8 mA, not seven or a J fit.
Lower shifted +.025377113 MHz between same-power visits, beyond conditional
errors; cause unknown. Preserve this systematic and require stability checks
for a later map/model. Full details in docs/pucq4-crossing-resolution-plan.md.
All prior ACTIVE entries below are historical; consult current guard/runtime.

September14 17:17: ac_bridge completeexit0/1577.359s,bothsources0/off.
NowACTIVE bridge_confirm_0914/ac_bcf,guard50752/child48824,
run2026-09-14_17-02-42. UpperM2/-5.74mA/.015gain32000reps CONFIRMED:
5440.546943643+/-.004619457MHz,width.154880317MHz,SNR11.7934,
7.751samples/FWHM,cleanreference. Lowerrepeat stillacquiring; matchedpair
notyetaccepted. Priorbridge loweraccepted5416.207715719MHz; upperweak
candidate remainsrejected. Central_highpower_0914 stillprepared/unlaunched.
Latest completed17-run/66trace analysis: ac_review/resolution_20260914/
bridge_checkpoint/analysis.json. Draft126slides inTEMP/bridge_progress.pptx
renderednatively, no text/celloverflow, pairtable/map reviewed; unpublished.
Source snapshots/diffs: resolution_20260914/code_checkpoint_1706/.

September14 16:59checkpoint: ac_bridge stillACTIVE (guard38352/child42576,
run2026-09-14_16-35-42), finalM2/-5.74mA lowertrace acquiring. Anchor-5.8
hasaccepted .015gain pair:5402.776076496/5439.870761686MHz,
separation37.094685190+/-.006144774MHz. Inwarduppercandidate5440.544543328
MHz,FWHM.181860461,SNR4.6535 FAILS SNRonly; retainrejection.
Prepared/notlaunched bridge_confirm_0914/ac_bcf repeatsboth-5.74windows
at.015gain32000reps/101points20kHz,3600s+180guard, afterlowerreview/parked.
Prepared/notlaunched central_highpower_0914/ac_ctrhi: M2/-5.6566,.03gain
32000reps,lower5430.7-5432.7MHz101points thenupper5442-5448MHz301points,
5400s+180guard; conditional on bridgeevidence andparkedverification. Full
envelopes andacceptance conditions in resolutionplan. Bothplansvalidated.
Completed16-run auditpassed178HDF5/116plots/2.706736445GiB; activebridgeexcluded.
Latest scientific regression/harness138files/whitespace checksPASS.
Native pair-IQ/current-axis slides readable. Final reportstillpending.

Latest September14 continuation: central_lowpower_0914 completed exit0,
allfour upper/lower M6/M2 traces have no candidates atgain.007; cleanreferences.
Bothsources verified0/off. This protocol lacks a lower-line positive control
at the center and cannot exclude the upper mode. Now ACTIVE bridge_pair_0914,
guard38352/child42576: M2 at-5.8 then-5.74mA, both tagged windows eachbias,
gain.015/8000reps20us100us/equalrefs,3600s+180guard, original controls.
Use liveguard/runtime forlateststatus. Analysis now uses recorded source settings
instead of requested current, preserving both; maximum correction67nA.
Original line fits/raw data unchanged; recorded-axis checkpoint includes16runs,
62traces. Final report/audit remain pending, earlier ACTIVE entries historical.

September14 16:19checkpoint: central_lowpower_0914/ac_ctrlo stillACTIVE,
guard16720/child9888. Both upper-window traces5442-5448MHz/.007gain have
NO candidates: M6vector2.7428,M2vector2.5329,cleanreferences. Bothlower
controls stillacquiring. Partialanalysis_p1/p2 preserved inrun documentation.
PREPARED ONLY bridge_pair_0914/ac_bridge forafterfullcentralreview/parked
verification: M2/-5.74mA, upper5439.5-5441.5MHz andlower5415.2-5417.2MHz,
201points8000reps/.015gain20us100us/equalrefs,1800s+180guard,zero/offcleanup.
Validated longestpath198chars; measured ac_m2a supports.015drive envelope.
It follows the measured detuned pair inward; no multi-bias map exists yet.
Continue toward >=7paired biases if visibility permits; do not claimJ fromone.
Analysis now supports single-readout tagged windows, retains cross-readout
conflict checks when both are present, and reports ac_pair1 as one M2 pair.
Synthetic/real-integration checks pass. Native newtable is readable and
deduplicates self/cross reviews. Publishedreportstill94slides;draft110inTEMP.

Latest September14 15:55local: ac_pair1 completed exit0/1492.5s,bothsources0/off.
Full common-power matched-window review ACCEPTS one line pair at-5.8mA/.007:
both lines inM2,lower5402.791625628+/-.007498609MHz andupper5439.862748670
+/-.003573086MHz,separation37.071123MHz (conditionalSE.00831MHz).
M6lower5402.788027MHz agrees; finalM6upper hasnoacceptedline/vector1.8389;
allrefs clean. This is one paired bias, NOT a minimum/J or physicalQ5 identity.
ACTIVE central_lowpower_0914/ac_ctrlo,guard16720,launched15:55:27;
read liveguard/runtime forchild/run. BothM6/M2 at-5.656667,.007gain8000reps,
upper5442-5448MHz301points/lower5430.7-5432.7MHz201points,
20us100us1round/equalrefs,paired_windows_id=central1,orderupperboththenlowerboth.
4000s+180guard,originaltracker/nominalattenuations/zero-offcleanup.
Upperwindow is model-guided search only; no acceptedJ. Full resolution plan
has derivation/provenance. Finalreport/audit pending. Previous94slide report
remains published and preserved. Do not stop with the current acquisition active.

Latest September14 15:29local: ac_dq6 completed exit0/796.297s, bothsources0/off.
M6/-5.8mA accepted5402.784229173+/-.008315814MHz,width.514242907MHz,
SNR17.4509/.03gain,cleanreference. M2samebiasupper5439.864905MHz is at.007,
so different-power locators are NOT a paired-mode acceptance.
ACTIVE ac_pair1 guard20120/child26208/run2026-09-14_15-29-43,
matched_pair_0914.json: bothM6/M2 atY4-5.8,.007gain/8000reps/20us100us,
bothwindows5401.8-5403.8MHz201points and5439.2-5440.6MHz141points,
10kHzgrid/equalrefs, taggedpaired_windows_id=detuned1. OrderM6lower,M2upper,
M2lower,M6upper.3000s+180guard,zero/offcleanup. Only offline analysis changes
to support explicitly matched disjoint windows, with gaps unmeasured; same
scientific gates and seven-bias requirement. See resolution plan for details.

Latest September14 15:13local: ac_dcf1 completed exit0/686.204s, bothsources
0/off. M2/-5.8mA accepted5439.864904956+/-.002935329MHz,width.130707246MHz,
SNR11.0063; M2/-4.8mA accepted5438.540998283+/-.001467710MHz,
width.077923430MHz,SNR14.7554. Bothreferencesclean. AcceptedM2endpoint
shift(-5.8 minus-5.4)1.796850771MHz is not a simultaneous splitting.
ACTIVE ac_dq6 guard21480/child59888/run2026-09-14_15-13-49,
plan detuned_q6_0914.json: M6/-5.8mA5350-5442MHz1841/1500/.03,
20us100us1round/equalrefs/tracker,1800s+180guard,zero/offcleanup.
This is a50kHz-grid second-mode locator; no new frequency calibration or J.

Latest September14 15:01local: ac_det1 completed exit0/1081.89s, bothsources
verified0/off. M2/-5.4mA accepted5438.068054185+/-.001443033MHz,
FWHM.071050491MHz,SNR15.1683,cleanreference. M2/-5.8mA candidate
5439.870923242MHz/SNR4.859624 remains rejected. ACTIVE ac_dcf1,
guard53980/child32168, run2026-09-14_15-01-34, plan detuned_confirm_0914.json:
-5.8 M2 confirmation5439.3-5440.3MHz101/16000/.007, then -4.8 M2 control
5437.8-5439.8MHz201/4000/.007.20us100us1round/equalrefs/tracker,
1800s+180guard,parkzero/off. No resolvedpair/J/physicalQ5assignment.

ACTIVE CONTINUATION, September 14 14:42 local: user reopened the unresolved
Q5/Q6 work and requested a plan, execution, and PowerPoint update without routine
questions. See docs/pucq4-crossing-resolution-plan.md. Guard32264/child34156
running TEMP/pucq4_cross_20260909/detuned_m2_0914.json, labelac_det1,
run2026-09-14_14-42-36. M2 readout at Y4 -5.8/-5.4mA,5436-5441MHz,
501points4000reps/.007gain/20us/100us/equal off-on, unchanged tracker.
Both sources passed fresh parked checks; current run must finish/park before
another hardware action. Deadline1800s+180guard. Previous94slide report archived
with matching hash; it remains the published report until this continuation's
results are reviewed. Below is the previous closeout, now historical.

CLOSED WITH UNRESOLVED COUPLING — September 14, 2026. The authorized physical
Q5-Q6 spectroscopy task is finished. No acquisition or guard remains active;
no follow-up is queued. Final confirm_m2_0914 ended exit0 in911.594s, with
both Yokogawas verified CURR/10mA range/0A/outputoff. Current readback is the
source setting, not an independent ammeter measurement.

Final evidence: 11 guarded runs,49paired drive-off/on traces,143canonical
HDF5 files (2.029700GiB) and94expected acquisition plots.33traces have one
accepted line,16none,0doublets. Coarse/fine maps, both available readouts,
power checks and wider searches completed. The final localized confirmation
at-5.656667mA resolves M6 5431.742654+/-.005062MHz/FWHM.193551+/-.022556MHz
and M2 5431.757418+/-.007018MHz/FWHM.220711+/-.033103MHz. These overlapping
single lines do not establish a splitting. Resonance bias,2|J|,J and physical
Q5's readout row remain unresolved. M2 is a candidate row, not an accepted
physical assignment. No entangled-state preparation is inferred.

Final PowerPoint: pucq4_first_light/PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx,
94slides, all94native-rendered; findings, tables and fine maps visually checked.
Evidence/analysis/checksums: pucq4_first_light/ac_review/final_20260914/.
Audit passes all expected raw/plot files, archived cleanup records and source
hashes. Seven protected configuration/acquisition files and the original
Downloads deck match task-start snapshots. No shared calibration was promoted.
Offline harness138files, scientific regression, canonical IQ round-trip and
final diff checks pass. The following entries are historical checkpoints,
not instructions to restart their prepared plans or old ACTIVE processes.

September14 12:24local: gap_m2_0914 completed exit0 in1317.219s, both sources
0A/off verified and controller processes gone. M6 only sampling-limited
candidate5431.706327MHz; M2 no candidate (vector score2.348). Both references
clean, wider5428-5458MHz search found no additional branch. Complete analysis
is ac_gap/2026-09-14_12-02-06/documentation/analysis_complete.
Reviewed and launched final localized confirm_m2_0914/ac_cf2 at12:24:30,
guard58892/child54820, run2026-09-14_12-24-31. Bothreadouts -5.656667mA,
5431.2-5432.2MHz101points16000reps/.03gain/equalrefs,10kHzgrid,
20us100us1round/originalROtracking/ramp/zero-offcleanup,2500s+180guard.
Exact live state TEMP/pucq4_cross_20260909/confirm_m2_0914.guard.json.
After completion: final analysis/audit/PowerPoint native render and closeout.
No resolved pair, J or physical Q5 assignment exists. No parallel controller.

Gap-search partial: ac_gap M6 saved, candidate5431.706327MHz,
FWHM.082930MHz,SNR6.754 but1.659samples/FWHM: sampling rejected.
No additional candidate in5428-5458MHz. M2 still acquiring. Prepared
confirm_m2_0914/ac_cf2 is NOT LAUNCHED: after full gap review/parking,
one localized confirmation at -5.656667mA, M6/M2 both5431.2-5432.2MHz,
101points/16000reps/.03gain/equalrefs,10kHzgrid,20us100us1round,
originalROtracking/currentprotocol,2500s+180guard,15minutesexpected,
maxpath195chars; validation-onlypassed. This is a final bounded line
confirmation; no doublet/J/physical assignment inferred. Currentgap
guard54572/child56092 only. Combined progress_1205 analysis45traces and
evidence_audit pass:9runs,131HDF5,86plots,1.790GiB; rawSHA256saved,
protected7sources and originaldeck unchanged. Latest87slidedraft passed
native export, but findings page overflow was found and builder now
paginates three paragraphs/page. Final rebuild/render remains required.

September14 12:02 local: fine_m2_0914/ac_fm2 completed exit0 in1973.203s,
both sources0A/off verified, processes gone;14paired traces/42HDF5 present.
Eleven traces have one accepted line, three unresolved; no doublet. Final
endpoint -5.59mA M6/M2 centers5436.328830/5436.328536MHz. Fine maps preserve
actual shifted windows (white areas unmeasured), source-setting axes and no
interpolation. Complete fine analysis is documentation/analysis_complete.
Reviewed then launched gap_m2_0914 at12:02:03, guard54572/child56092,
run ac_gap/2026-09-14_12-02-06. Exact live state: TEMP/pucq4_cross_20260909/
gap_m2_0914.guard.json. Both readouts5428-5458MHz at -5.656667mA,
601points/4000reps/.03gain/equalrefs/original guarded protocol,4000s+180.
No concurrent controller. Final report and closeout audit remain pending.

Fine-map update September14: analysis_p10 has ten saved paired traces.
At -5.67mA both lines are accepted near5429.81/5429.77MHz (M6/M2).
At -5.656667mA M6 candidate5431.676940MHz fails SNR4.052; M2 has no line.
At -5.643333mA M6 accepted5433.215602MHz; M2 candidate5433.234355MHz
fails SNR4.280. Both readouts recover accepted lines5435.241324/5435.251044
MHz at -5.616667mA. References clean. No pair/J/physical assignment accepted.
Prepared gap_m2_0914/ac_gap is NOT LAUNCHED: after this fine map ends and
parks, review then search5428-5458MHz at -5.656667mA on M6/M2,601points/
4000reps/.03gain/equalrefs, same20us100us1round/originalROtracking/ramp,
4000sdeadline+180guard,22minutes expected,maxpath195chars. Validation-only
passed. It checks outside the narrow fine window without assigning a mechanism.
Current fine guard31788/child13704 remains the only controller.

Fine-map central pair saved September14 around11:33: M6 5434.393632MHz,
FWHM.165115MHz,SNR14.377; M2 5434.411515MHz,FWHM.135312MHz,SNR11.125,
both accepted, clean references. The .017883MHz center difference is below
the linewidths and is not a resolved pair. At the same .03gain/-5.63mA,
September10 M6/M2 centers were5434.608891/5434.619936MHz: about .21MHz
inter-day shift, larger than conditional errors. Keep days separate when
interpreting dispersion. Current windows still contain the line. Remaining
fine biases continue under guard31788/child13704.

September 14, 11:28 local: upper1 completed exit0 in1317.188s, both sources
verified0A/off and both processes gone. Neither M6 nor M2 has an accepted line
over5450-5480MHz; vector scores2.757/2.795, references clean. Full analysis:
ac_up1/2026-09-14_11-06-02/documentation/analysis_complete/analysis.json.
Fine map fine_m2_0914 launched11:28:17, guard31788/child13704, run
ac_fm2/2026-09-14_11-28-18. Seven unique biases -5.67 to -5.59mA,
central -5.63 first; M6/M2 each 2.4MHz locator-based windows/121points/
4000reps/.03gain,20us/100us/1round/equalreferences. Guarded original protocol,
5000sdeadline+180,expected37minutes. Exact live state is TEMP/
pucq4_cross_20260909/fine_m2_0914.guard.json. No overlapping controller.
This fine scan follows the observed line only; no second branch or J is
accepted, and no physical Q5 row assignment is established. Final report pending.

September 14, 11:06 local resume: ac_m2a completed September 10 at 02:14,
exit 0, both sources recorded CURR/10 mA range/0 A/output off. Complete
eight-trace analysis is in its documentation/analysis_complete_0914. Lower
gain .015 at -5.63 mA gives M6 5434.597693 MHz/FWHM .107013 MHz and
M2 5434.595834 MHz/FWHM .113182 MHz, both accepted with clean references.
These are coincident single lines; no J or physical Q5 assignment is accepted.
No post-September-1 Python controller was active before resumption. Prepared
upper1/ac_up1 launched at 11:06:01, guard 61368 / child 47680, after fresh
parked-source preflight. Exact live state is TEMP/pucq4_cross_20260909/
upper1.guard.json. Run ac_up1/2026-09-14_11-06-02; -5.59 mA, M6/M2,
5450-5480 MHz, 601 points/4000 reps/.03 gain, original guarded protocol.
Do not overlap a controller. Fine bias mapping and final report remain pending.

Separate physical Q5-Q6 crossing task ACTIVE (September 9, evening). The user
explicitly authorized autonomous measurements and the final PowerPoint. The
Downloads deck's physical Q5 readout assignment is still unknown; do not
substitute displayed Q5/M5 (physical Q4). Q6 is M6/index5/Yoko4. See
[crossing task and hardware envelope](../docs/pucq4-avoided-crossing.md).
Locator ac_loc1 completed with Q6 5587.968910MHz at -5mA, FWHM.419453MHz,
canonical complex-IQ drive-off/on data and verified both0A/off. Coarse candidate
M4 check ac_c1 completed with one accepted line per trace and both0A/off.
Visibility pilot ac_v1 completed with both0A/off; no doublet. The weak M4
feature fails the preset SNR gate. Its accepted high-power center shifted
0.113MHz between visits, prompting a bias-reversal control before surveying
other candidates. ac_b1 completed with both0A/off; its B-A contrast
-.004996+/-.010940MHz did not reproduce the earlier shift as a bias response.
ac_cs1 completed September10 00:43local, exit0, both sources0A/off. At-5.135mA,
M6=5557.872146MHz and M3=5557.323580MHz are visible single lines. M1 remains
visible at5517.685971MHz (-5.305mA), while M6 is unresolved in that window.
At-5.63mA neither M6 nor M2 resolves in5423.7-5453.7MHz despite healthy
resonator tracking. This prompts a wider search, not a coupling inference.
ac_v2 completed September 10, 01:13 local, exit0, both sources0A/off. M6 and
M2 both show a strong narrow candidate near5434.65MHz at-5.63mA/.03gain;
both fail sampling (2.76/2.99 samples per FWHM on a50kHzgrid). A global I/Q
PCA detector hid these sparse lines; it is now replaced by detection using
both components, while fits still use signed IQ. Regression checks pass;
see FAILURE_MODES.md and ac_review/progress_0102 for the corrected evidence.
Current ac_m2a launched September10 01:15:12local, guard20300; exact live
state is TEMP/pucq4_cross_20260909/map_m2_1.guard.json plus actual processes.
Plan: Y4-5.67,-5.59,-5.63mA, M6 and M2 each5420-5455MHz1401points1000reps
at common.03gain (25kHzgrid). Last bias adds.015gain5433.6-5435.6MHz201points:
M6 8000reps, M2 16000reps. All20us/100us/1round, matched references and
original res tracking. Y3zero/off,.5mA/s ramps,noextraDCsettle,zero/offcleanup,
7000sdeadline+180guard;expected57minutes. No physical Q5 assignment or exchange
estimate is accepted. M2 is the priority candidate for bias-dependent evidence.
Fine M3 plan fine_m3_1/ac_f3a is PREPARED ONLY and deferred.
At -5.59mA, ac_m2a has accepted single lines on M6 (5436.399577MHz) and M2
(5436.409616MHz); these are coincident within errors, not a resolved pair.
The -5.67mA candidates near5430.2MHz fail the SNR gate. Central-bias and
lower-drive scans remain active. Prepared upper1/ac_up1 is NOT LAUNCHED:
after reviewing this map, extend -5.59mA coverage to5450-5480MHz on M6/M2,
601points4000reps.03gain/equalrefs. See the task log and exact JSON plan.
Never overlap a second hardware controller. The earlier round-robin campaign
below remains closed with gaps.

CLOSED WITH GAPS (September 9, 21:51 local): the time-bounded round-robin
validation campaign has ended. No task acquisition or guard remains active and
no follow-up is scheduled. Final readbacks at21:50local: both Yokogawas in
current mode,10mA range,0A,outputs off. Legacy pre-task processes were preserved.

Final all-six default run: M4--M6 pass; M1 coherence/model, M2 SSF comparison,
and M3 T1 comparison/repeatability findings remain. Previously selected complete
zero-current evidence covers5/6rows. Each flux qubit has5/8accepted nonzero
currents (-2.5,+2.5,+5,+7.5,+10mA). Q4/M5 -5mA was measured but remains unresolved:
its two qspec centers differ.126506MHz, above the.015666MHz repeat gate.
Q4 -7.5/-10mA and Q6 -5/-7.5/-10mA remain unmeasured in this campaign.
M1/Y3+10mA control is unresolved; M2 control accepted; M3/M4 atY4-10mA unmeasured.
Separate linewidth and peak-population tuning targets remain explicit in the deck.

Final M4--M6 zero-current values were propagated to shared configs; no closeout
M1--M3 or Q4/-5mA promotion. All15selected complete profiles match current settings.
Offline harness passed131Pythonfiles,81x6profile routing passed,33protected source
hashes and both reference PowerPoints are unchanged. Focused diff passed.
The single new149-slide validation deck passed fresh native PowerPoint rendering.
See [closeout and exact changes](../docs/pucq4-round-robin-closeout.md).
Final manifest, source readbacks, report checks, and execution archive `exec_0909`
are under `PUCQ4_round_robin_validation_20260908_artifacts` in the first-light run.
Do not execute archived plans in place or resume the historical queue blindly.

Previous final confirmation (September 9, 21:41 local): Q4 / M5 / Yoko3 -5mA
full pass completed with only a qspec reference-frequency finding. All raw/model,
calibration, Rabi and coherence gates pass. One exact qspec repeat is running:
`m5_m050_qspecrepeat.json`, controller26860 / RR54288, launched21:41:05local,
360sdeadline. This short confirmation is the only addition to the last-run plan;
no more current groups or retries, closeout remains22:00local. Both sources were
0A/off before launch; guarded0A/off afterward. After audit, accept only the preset
.1linewidth/3combinedSE repeat gate. If it disagrees, retain the full-pass finding.
Then promote eligible final zero rows M4--M6 (preview already checked) and -5mA
only if accepted, capture changes, record last bias evidence, finalize manifest,
build/render the new deck, collect checks, write notes and archive helpers.

Previous closeout checkpoint (September 9, 21:23 local): last hardware run is
physical Q4 / M5 / index 4 at Yoko 3 -5 mA, Yoko 4 zero.
`m5_m050_closeout.json`, controller19836 / RR27696, launched21:22:44local;
normal RR all seven stages, one pass,1500sdeadline. Both sources zero/off before
launch and guarded zero/off cleanup afterward. Expected finish about21:40.
No additional negative groups should be launched under the22:00closeout target.

The final all-six default run completed21:16:37local (113.17min), all canonical
datasets saved, sources0A/off. Watcher finished21:17:03 and is no longer active.
Final M4--M6 pass. M1 has Ramsey model/repeatability and echo comparison findings;
M2 SSF contrast .2678 versus historical .34317; M3 T1 31.4602us versus45.7928us.
Exact M3 repeat37.9150us disagrees (6.4548us difference >5.2031us repeat gate).
Exact M2 SSF repeat.2853 agrees within.03, but the original comparison and readout
optimization gap remain. Both confirmations saved, both sources0A/off afterward.
No final M1--M3 configuration promotion. See validation artifacts
`final_default_confirmation_review.json`; every final outcome is in the new deck.
After this last Q4 run, audit, promote only eligible accepted rows, rebuild/render
the report, collect closeout checks and archive helpers. Final notes are prepared
but not yet written; use `collect_closeout_checks.py` then `write_closeout_notes.py`.

Previous checkpoint (September 9, 19:24 local): final all-six default zero-current
RR is running, launched19:23:26local by `final_all6_closeout.json`.
Controller24772 / guard36224 / RR58200; no PUCQ4 environment overrides.
All six rows, all seven normal experiments, one pass,8100sdeadline.
Both sources verified current mode/.01A range/0A/output off before launch.
Full envelope and immutable launch hashes: validation artifacts
`final_all6_closeout_envelope.json`; longest projected artifact193characters.
Expected finish around21:16local, then audit/report and safe cleanup by22:00.

Offline final-audit watcher launched19:53:42local: controller58720 / Python60764,
`audit_after_final.py`. It waits for final guard cleanup AND matching copied
source hashes, then writes six fresh audits, records all outcomes, populates
manifest.final_default_validation and refreshes the single PowerPoint. It does
not operate hardware or promote shared config. DO NOT audit the final run in
parallel with this watcher. Check `audit_after_final.result.json` and its
stdout/stderr before manual final review. `promote_final_zero.py` is prepared,
not applied; it previews accepted final rows M2--M6 and deliberately excludes M1.

M1's predetermined180us pair failed repeatability:31.038210+/-0.774484us
and19.473528+/-1.153028us; difference11.564683us >4.166975us gate.
Both individual raw/model audits pass; both records and automatic fits retained.
No M1 profile or phase-reference feature was adopted. M1 stays unresolved.
See `M1_frame180_repeat_review.json` in validation artifacts. Both sources were
verified zero/off after each diagnostic. Q6 -2.5mA remains accepted/promoted.
Six negative groups (-5,-7.5,-10mA on each flux qubit), unresolved M1 endpoint
control and unmeasured M3/M4 endpoint controls remain explicit closeout gaps.

Previous closeout checkpoint (September 9, 19:07 local): Q6/M6 -2.5mA full E and exact
50k-shot SSF confirmation passed the unchanged review: .32474/.30334,
difference .02140 < .03, bootstrap SE .00313934/.00289345. Both remain below
historical .427; no physical cause assigned. `validation_M6_ssf_reviewed.json`
is selected and its tested pulse/readout and averaging values are now persistent.
Full-E actual carrier6103.73104802091MHz, pi.651681621193,
sigma.056022425us, RO9057.86/.8/10us/offset-.04MHz. All15selected complete
points match their live reusable profiles; protected classes and81x6routing pass.
Both sources were verified zero/off after confirmation.

Previous active run `m1_frame180_a.json`, guard25596/child55560, launched19:04:27local.
After safe completion/audit, run the predetermined identical b diagnostic.
No M1 calibration promoted. A complete M1 validation remains necessary.
The single validation deck now has131slides and passed native PowerPoint
preview at approximately18:54local (22:54UTC).

September 9 closeout constraint (18:49 local): the user requested finishing
within the next few hours. Target approximately 22:00 local, including final
report and verified zero/off source cleanup. The active Q6 full-E run is still
healthy. The previous all-six pass took 111.7 minutes; the entire remaining
matrix exceeds this time allowance. With no optional preference reply, reserve
time for the final all-six zero-current benchmark and report. A later user reply
may change this priority. Continue independent audit/report preparation; retain
all quality gates and flag any unfinished or rejected measurement explicitly.

Previous active checkpoint (September 9, 18:35 local): Q6/M6/index5 is running
complete normal RR at Yoko4 -2.5 mA, Yoko3 zero (`m6_m025_fulle.json`,
guard7740/child14000). Three predetermined 50000-shot SSF records passed every
pair of the preset repeat gate (0.30804/0.28324/0.30724; limit 0.03). Full E
uses 50000 state-calibration/SSF shots and the previous full-d spectroscopy and
coherence averaging. Full d's readout repeat failed and was not promoted. After
E completes, audit all seven saved traces and obtain a NEW exact SSF repeat if
the lower historical contrast is its sole finding. Q6 -2.5 remains unpromoted.

M1's full 120 us pass and exact repeat pass quality but disagree in T2R:
31.2046 versus25.8081 us, difference5.3965 >4.2760 us repeat limit. Neither is
promoted. Two high-phase scans infer an absolute midpoint within0.000223 MHz;
the earlier constant phase was reused after the carrier changed6.37 kHz.
Prepared `m1_frame180_a.json` and `m1_frame180_b.json` test a fresh measured
phase +0.001330551162455 MHz over180 us/901points at the SAME carrier/pi/RO.
They are diagnostic only, require both records and unchanged quality/repeat
gates, and must wait until active Q6 completes. No adaptive frame feature or
M1 shared-profile change has been implemented. A fresh complete M1 pass, exact
confirmation and Yoko3 +10 mA control remain necessary. The prepared endpoint
control plan must be reconciled with the eventual validated M1 profile.

Q4/M5 -2.5 mA and all positive flux rows are accepted. Six other negative groups
and final all-six default zero remain. All14 accepted complete points matched
shared profiles at17:34; no core config changed since then. The one new report
has125 slides (native QA18:08); newest diagnostics need the next build. Exact
envelopes and chronological evidence are in `docs/pucq4-round-robin-validation.md`
and the temporary checkpoint. Continue the complete authorized task.

Task-start snapshot (September 8, 21:14 local):
`rr6_0mA/2026-09-08_21-14-32`, guard PID59668 / RR PID60780. All six rows are
running with persistent averaging/timing profiles and calibrated population
plots. The new `set_yoko_bias` function now COMMANDS and verifies both sources;
each RR invocation returns them to zero/off. Older config-only-selector notes
below describe historical behavior. Exact plan/progress is in
`docs/pucq4-round-robin-validation.md`; completed source calibrations below
remain the reference until new raw-data audits and follow-ups pass. The user
explicitly requested ONE new validation PowerPoint; preserve the two existing
decks and add only `PUCQ4_round_robin_validation_20260908.pptx`.

September 8 linewidth refresh: full normal RR runs line_4b/2026-09-08_19-53-42
(M4/index3) and line_6d/2026-09-08_18-51-46 (M6/index5) are accepted and their
shared configurations promoted. Both Yokos were CURR/0 A/output off before and
after. That earlier acquisition task finished. FWHM .682091 +/- .046102 MHz for M4 and
.211477 +/- .010673 MHz for M6: the requested .10 MHz target was not reached.
Gain .005 line_4a remains diagnostic (qspec R2 .6515). Both active decks are
updated and audited:42-slide chip summary and114-slide characterization history.
Yoko FWHM tables are on summary slides16/20/28/32, with actual current, gain,
pulse length and averaging; weak/coarse fits are marked. Exact run envelopes and
completion verification are recorded in docs/harness-log.md and summary artifacts.

Baseline consolidated: 2026-08-28; M4/M6 updated 2026-09-08. This document separates executable configuration,
validated measurement, deck/search seeds, rejected candidates, and desired
acceptance targets. Check live source before every run.

## Identity map

| Resonator row | Python index | Usual displayed label | Physical assignment | Flux line |
| --- | ---: | --- | --- | --- |
| M1 | 0 | Q1 | physical number unknown | fixed-frequency for this work |
| M2 | 1 | Q2 | physical number unknown | fixed-frequency for this work |
| M3 | 2 | Q3 | physical number unknown | fixed-frequency for this work |
| M4 | 3 | Q4 | unassigned; not physical Q4 | no documented Yoko |
| M5 | 4 | Q5 | physical Q4 / DC D4 | Yoko 3, `192.168.1.73` |
| M6 | 5 | Q6 | physical Q6 / DC A5 | Yoko 4, `192.168.1.77` |

Some older config comments mention an avoided-crossing mapping hypothesis. The
accepted deck/run mapping for this campaign is physical Q4 -> M5 and physical Q6
-> M6. Always use all four identifiers in a hardware task.

## Shared zero-current starting values after the September 8 refresh

These are shared runtime values from `system_config.py`, including selected
zero-current flux rows and the readout offsets used by round robin. They are not
proof that every row meets the linewidth target. M4/M6 use final line_4b/line_6d;
M5 retains its September 7 calibration. Check live source before a new run.

| Row | res center (MHz) | RR offset (MHz) | qfreq (MHz) | qspec gain | sigma (us) | pi amp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 8919.850 | -0.52 | 5517.69373 | 0.001 | 0.050 | 0.6786303 |
| M2 | 8951.650 | -0.40 | 5438.71351 | 0.007 | 0.101 | 0.6486907 |
| M3 | 8975.000 | -0.08 | 5557.33447 | 0.002 | 0.124 | 0.6586706 |
| M4 | 8999.940 | -0.21 | 5594.109362350 | 0.0075 | 0.079003864231 | 0.651681621193 |
| M5 | 9014.740 | +0.20 | 7084.280775100 | 0.003 | 0.056940825 | 0.646706952 |
| M6 | 9058.880 | -0.40 | 6522.511432598 | 0.00075 | 0.062612796167 | 0.646706952329 |

Important override/state notes:

- At the baseline, readout lengths were `[10, 5, 12, 10 placeholder, 5, 3]` us.
  M4 now uses 5 us, gain 0.95 and offset -0.21 MHz after alternating SSF
  optimization. Current M5/M6 values come from the later flux calibration.
- `pucq4_config.QUBIT_FREQS_MEASURED_0MA` and shared runtime configs now carry
  measured M4 frequency 5594.109362350 MHz; `QUBIT_FREQS_VNA` retains the historical
  5575 MHz deck value. M4 physical number remains unknown; its coherence is now
  measured and accepted below.
- `round_robin_benchmark.py` is restored to one pass over validated rows
  `[0, 1, 2, 3, 4, 5]` with res spec, qspec, Rabi, SSF, T1, T2R, and T2E enabled.
  The September 8 reproduction task includes M4 by default. Weak-drive settings use
  M4 qspec128000 x 1/97 points and M6 qspec64000 x 1/101 points. Exact reusable
  historical plans in `PUCQ4_chip_summary_20260908_artifacts/linewidth_validation/`
  (`4b_plan.json`, `6d_plan.json`) and their guard/HDF5. These averaging and short
  coherence timing settings are now persisted in `expt_config.pucq4_round_robin_profile`.
  The older M4_coherence_20260908_settings.json is historical.

## Accepted zero-current measurements

Both known Yokogawas were verified in current mode at exactly 0 A on the 10 mA
range, with outputs off, for the recorded baseline campaign.

The coherence values below are accepted measurements, not simultaneous all-row
results: M1-M3 August 28, M4/M6 September 8, M5 September 7. The September 8
summary rechecks archived fits/configurations and retains each acquisition date.

| Row / physical identity | accepted res (MHz) | accepted qfreq (MHz) | Qspec FWHM (MHz) | T1 (us) | T2R (us) | T2E (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 8919.830 | 5517.69373 | 0.050093 | 31.45 | 16.53 | 37.13 |
| M2 | 8951.650 | 5438.71351 | 0.093211 | 8.77 | 11.71 | 15.88 |
| M3 | 8974.970 | 5557.33447 | 0.036154 | 45.79 | 42.85 | 77.22 |
| M4 / unassigned | 8999.940 | 5594.109362350 | 0.682091 +/- 0.046102 | 0.816979 +/- 0.007847 | 0.471189 +/- 0.016288 | 1.757433 +/- 0.053692 |
| M5 / physical Q4 | 9014.890 | 7084.280775100 | 0.110477 | 12.440386 +/- 0.110609 | 6.123946 +/- 0.123923 | 13.850536 +/- 0.253370 |
| M6 / physical Q6 | 9058.880 | 6522.511432598 | 0.211477 +/- 0.010673 | 6.435512 +/- 0.077920 | 1.803286 +/- 0.046256 | 5.466466 +/- 0.107097 |

M5's resonator is its actual September 1 gallery trace; its selected calibration
base is9014.74 MHz. M5 corrected frequency/coherence come from
`ramsey_sigma_coherence_manifest.json` at0 mA. M4/M6 values come from the final
September 8 one-pass runs. Other FWHMs are joint-IQ refits of archived qspec.
The tables retain statistical fit errors and acquisition dates; sequential
coherence fits need not satisfy a simultaneous stationary model exactly.

Additional accepted evidence:

- Clean low-gain qspec operation was demonstrated with gains M1 0.001, M2 0.010,
  M3 0.002, M5 0.001, and M6 0.002. Later executable gains differ for M2/M6
  and require plot-level revalidation.
- Physical Q4/M5 was measured with 0.165 MHz qspec FWHM in its diagnostic run.
- Q6 was confirmed at 6522.4767 MHz with about 0.3 MHz FWHM in a wider search.
- Narrow qspec was revalidated for the five observable modes. M5 reaches about
  0.30 population at gain 0.004 with ~0.1 MHz FWHM; M6 reaches about 0.30
  absolute population at gain 0.006 with ~0.1 MHz FWHM, but its elevated
  baseline reflects weak readout separation.
- August 28 round-robin SSF values were 58.2%, 34.3%, 37.2%, 17.4%, and 21.5% for
  M1, M2, M3, M5, and M6. Three alternating optimization rounds produced no
  significant further improvement. These are measured/optimized but not high.
- The reusable deck is `pucq4_first_light/PUCQ4_first_light_characterization.pptx`.
  Its September 8 M4 evidence was appended with `tools/append_pucq4_search_slides.py`;
  the earlier builder does not include these appended measurements.

## Qspec and Rabi lessons

- The strange oscillation/ringing was coherent sidelobe structure from a 20 us
  rectangular spectroscopy pulse driven at the former 0.03-0.08 gains. Lowering
  config gain removed it; the QICK program was not changed.
- The scaled qspec fitter once used `argmax`'s array index as a MHz initial guess
  and allowed negative linewidth. The wrapper was repaired to seed from acquired
  frequency and bound center/width. A fit outside sweep bounds is invalid.
- Current Rabi acceptance is just over half an oscillation with pi gain 0.6-0.7.
  The earlier M2 seed 0.7385 exceeded that target; the accepted August 28 Rabi
  fit is 0.648691. Placeholders must never be presented as measurements.

## M4 bounded search result

September 8 renewed-search checkpoint: the user explicitly requested another
hardware attempt using the supplied VNA deck and successful first-light settings.
`pucq4_12_m4_search.py` is the guarded, plan-driven diagnostic runner; it uses the
unchanged repository spectroscopy programs and canonical HDF5 labels, asserts
effective readout length/gain, and verifies saved vectors before continuing.
The new `m4_s01` readout-power survey resolves the M4 resonator near 9000 MHz;
the 0.03-gain slope estimate is too noisy to use. A weak M3 control was rejected
in `m4_s02`; the `m4_s03` M1 control resolves 5517.689621 MHz with 0.174420 MHz
FWHM and joint-IQ amplitude/noise 56.94. Both sources were independently checked
at 0 A/output off. `m4_s04` completed 5525-5675 MHz at 0.05 MHz requested spacing
using both measured M4 readout slopes, without a resolved line. A weak
5575.421 MHz fluctuation was rejected by the independent 8000-average repeat
in `m4_s06`: the width collapsed to one frequency bin. A matched 1 us M1
control in `m4_s07` resolved its known line. With 1 us readout at gain 0.95
and qubit drive 0.10/20 us, `m4_s08` now resolves a new candidate at
5594.45378 and 5594.46882 MHz using physical readout tones 9000.20 and
8999.65 MHz, respectively (FWHM 1.355 and 1.469 MHz). Independent repeat,
zero-drive, and lower-power tests in `m4_s09` passed. Clear Rabi then confirmed
coherent M4 control near 5594.49 MHz: sigma 0.0788 us gives pi gain 0.646707.
Alternating SSF optimization converged at physical readout 8999.80 MHz, gain
0.95, length 5 us (offset -0.15 MHz from base 8999.95 MHz).
The first calibrated wider round-robin scan in `m4_final` resolved 5594.221 MHz
but also an extra high-edge response, so its poor single-line fit was rejected
as final spectroscopy evidence. Same-pass Rabi R2 was 0.9973 and SSF 0.3212.
A focused independent `m4_clean01/2026-09-08_14-17-43` round-robin repeat passed:
frequency 5594.388460 MHz, FWHM 0.723922 MHz, fitted calibrated peak population
0.37266, spectroscopy R2 0.97065, independent raw-IQ SNR 23.19. Same-pass
Rabi R2 is 0.99674 with pi gain 0.646707, and SSF contrast is 0.32834.
All qspec/Rabi/SSF HDF5, raw arrays and normal plots were verified; both sources
finished at 0 A/output off. Shared M4 frequency, qspec gain 0.05, sigma/pi,
and optimized readout values are promoted from these measurements.
The preferred 0.1-0.2 MHz linewidth target was not reached. Centers varied from
5594.22 to 5594.49 MHz during the session; rerun spectroscopy before subsequent
pulse work rather than treating the stored fit precision as long-term stability.
The deck has 81 slides: original slides 1-68 are preserved, search progress is
on 69-79, final Rabi/SSF is on 80, and final clean spectroscopy is on 81.
The final slide/media/notes audit checked 301 unchanged original parts.
`m4_clean01/2026-09-08_14-17-43/documentation/completion_audit.json` records
the preservation/config checks; `M4_handoff.md` gives the reusable settings.

The historical null-result conclusions below are superseded by this confirmed
M4 discovery; they must not be used to label the qubit unobservable now.
See the September 8 entry in `docs/harness-log.md` for exact paths and settings.

M4 means resonator row M4/index 3/displayed Q4, not physical Q4. The deck gives a
9.000 GHz resonator, 5.575 GHz `f_ge` seed, 5.466 GHz `f_gf/2`, 218 MHz
anharmonicity, and no physical-qubit number.

Accepted diagnostic conclusions:

- Low-power res spec found a real dip at 8999.980 MHz and maximum measured slope
  near 9000.260 MHz.
- Searches covered narrow and wide qspec around the deck seed, zero/slope and
  multiple readout parking points, a 2D readout-frequency grid, all six readout
  rows, pulse lengths 20-80 us, gains through high power, and an earlier
  4.075-7.075 GHz scan.
- A same-chain M1 positive control recovered 5518 MHz at about eight local sigma,
  while M4 stayed flat. This rules out a simple broken qspec program/frequency
  axis explanation.
- The final 5525-5625 MHz scan used 0.05 MHz spacing, gain 0.30, 40 us qubit
  pulse, M4 readout 9000.260 MHz at gain 0.03/25 us, 2048 reps, and was flat.
- The 5584.20 and 5702 MHz candidates did not reproduce and are rejected.
- The last final-scan plot exists, but its HDF5 save failed at a 268-character
  Windows path. It cannot serve as complete persisted raw evidence.

The historical stop rule required new external evidence before another blind
search. The September 8 user-authorized search and confirmed Rabi response
supersede that null result. `_tmp_m4_readout_debug.py` remains a historical
diagnostic, not a reusable calibration script; use the accepted settings above.

## Flux-tunable scope and deck range

Only physical Q4/M5 and physical Q6/M6 are documented as flux tunable. The deck
plots cover -10 to +10 mA. At zero current, accepted frequencies are approximately
7084.72 MHz (physical Q4) and 6522.477 MHz (physical Q6). The deck curves are
search guidance at other currents, not a replacement for measuring each point.

Future current-indexed config must key sigma/calibration by physical line and
current with units, and must not overwrite the zero-current list with an
unlabeled value.

The authorized campaign grid is `[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]`
mA. `pucq4_07_flux_campaign.py` is the resumable controller. It must leave the
inactive source at 0 A/output off, ramp the active source at 0.5 mA/s, abort on
an out-of-window locator fit, and always ramp back to 0 A before disabling its
output. The controller reuses `round_robin_benchmark.py`; it does not replace
the section-class QICK programs.

## Pending user objectives

- M4 discovery is complete as of September 8. Its physical-number assignment,
  narrower-line calibration and T1/T2 characterization remain future work.
- Revalidate narrow qspec and Rabi acceptance for all validated rows in the normal
  round-robin plot/save path.
- Complete alternating SSF length and gain/frequency optimization; propagate and
  revalidate stable values.
- Run a clean one-pass res spec -> qspec -> Rabi -> SSF -> T1 -> T2R -> T2E chain
  with reviewed flags and no diagnostic override.
- Build the reusable `pucq4_first_light` PowerPoint: first-slide zero-current
  identity/res/qfreq/coherence table; per-experiment six-panel slides; last
  length and 2D SSF optimization slides; configs and provenance on every slide.
  Unknown/unaccepted fields must stay visibly unknown.
- Within an explicitly authorized -10 to +10 mA plan, map qspec, res spec, Rabi,
  sigma/pi, T1, T2R, and T2E for Yoko 3/physical Q4 and Yoko 4/physical Q6; add
  current-sweep summary tables/plots to the PowerPoint.

## 2026-08-28 flux-campaign checkpoint

- Both sources were repeatedly verified at `0.0 A`, output off, between points.
- Q4/M5 at `0 mA`: `fq=7084.7785 MHz`, FWHM `0.1 MHz`, sigma `0.05996 us`,
  pi `0.63871`, T1 `5.96 us`, T2R `4.33 us`, T2E `11.35 us`.
- Q4/M5 at `-2.5 mA`: `fq=6945.5141 MHz`, FWHM `0.1 MHz`, sigma
  `0.05996 us`, pi `0.63871`, readout `8 us / 0.57 / +0.08 MHz`, SSF
  `0.552`, T1 `11.22 us`, T2R `3.48 us`, T2E `9.75 us`.
- Q4/M5 at `-5 mA`: wide qspec rejected a false `6708 MHz` fit and found
  `6725.447 MHz`. Accepted result: `fq=6725.3975 MHz`, FWHM `0.2 MHz`, peak
  population about `0.40`, sigma `0.08530 us`, pi `0.64869`, readout
  `7 us / 0.89 / +0.20 MHz`, SSF `0.572`, T1 `15.53 us`, T2R `2.40 us`,
  T2E `7.74 us`.
- Q4/M5 at `-7.5 mA`: the circular Rabi/SSF dependency was resolved by running
  a 1200-average unscaled raw-I/Q Rabi before readout optimization. Accepted
  result: `fq=6423.2222 MHz`, sigma `0.05911 us`, pi `0.64869`, readout
  `7 us / 0.9267 / -0.16 MHz`, SSF `0.666`, T1 `19.09 us`, T2R `1.86 us`,
  and T2E `6.01 us`.
- Q4/M5 at `-10 mA`: a wide diagnostic found a weak feature at `5984.104 MHz`
  (about `0.7 MHz` at gain `0.03`), but the low-power narrow trace was noise
  and Rabi was flat. Q4/M5 at `+2.5 mA` likewise produced a failed Lorentzian
  and a visibly noisy narrow trace. Neither point is accepted.
- Remaining Q4 currents, all Q6 currents, and aggregate 2D plots remain
  incomplete. Accepted Q4 current-indexed settings are now in
  `system_config.PUCQ4_FLUX_CALIBRATION`, and the PowerPoint includes the
  completed -7.5 mA evidence plus explicit incomplete-point labels.

## 2026-08-29 flux-campaign checkpoint

- Q4/M5 is now accepted at every grid point except `-10 mA`. Newly accepted:
  `+2.5 mA fq=7143.8491 MHz`, `+5 mA fq=7121.7063 MHz`,
  `+7.5 mA fq=7022.3193 MHz`, and `+10 mA fq=6841.5394 MHz`. Exact sigma,
  pi, readout, and provenance live in the campaign manifest.
- Q4 `-10 mA` remains unresolved: repeated readout optimization/rescue stayed
  below 9% fidelity, so coherence was not run or claimed.
- Q6/M6 is accepted at `0 mA` and `+2.5 mA`. The `+2.5 mA` result is
  `fq=6842.1315 MHz`, `sigma=0.05800 us`, `pi=0.60877`, resonator base
  `9060.46 MHz`, readout `12 us / 0.7967 / -0.52 MHz`, SSF `0.5753`, with
  T1/T2R/T2E completed in the normal round-robin path.
- Q6 nonzero-current locator/qspec/Rabi must inherit the accepted readout seed;
  historical defaults produced noise-only Rabi aliases. A fitted pi in the
  target window is insufficient unless the saved curve visibly resolves an
  oscillation above the scatter.
- Q6 unresolved: `-2.5 mA` (qspec near 6122.8 MHz but noise-only Rabi/<9% SSF),
  `+5 mA` (unphysical narrow candidates near 7025/7034 MHz), and `+7.5 mA`
  (unstable fits near 7156-7157 MHz). Q6 `-10`, `-7.5`, `-5`, and `+10 mA`
  remain unmeasured. The aggregate flux maps and final report are therefore
  still incomplete and must not be represented as finished.

- Later bounded recovery attempts also rejected Q6 `-5 mA`, `-7.5 mA`, and
  `+10 mA`. At `-5 mA`, a wide diagnostic candidate at `5587.7655 MHz`
  (`0.4814 MHz` width) collapsed to `0.00175 MHz` at calibration power. At
  `-7.5 mA`, `5061.7466 MHz` collapsed to `0.00787 MHz`, and two raw Rabi
  traces returned the identical `0.8482879` alias despite a sigma change. At
  `+10 mA`, the low-power width was `0.00788 MHz` and raw Rabi was noise-only.
  These are rejected diagnostic candidates, not current-indexed calibrations.
- Q6 `-10 mA` has a reproducible spectroscopy candidate at `4391.6689 MHz`.
  The wide diagnostic width was `0.9435 MHz`; a recentered narrow acquisition
  proceeded to raw Rabi, but I/Q was monotonic/non-oscillatory and the fitter
  returned the same `0.8482879` boundary alias. Record the frequency as
  spectroscopy-only evidence; do not propagate sigma, pi amplitude, or
  coherence for this point.
- Failed qspec HDF5 files may contain `"None"` placeholder datasets rather than
  numeric sweep arrays. Inspect dataset type/content before aggregation; a file
  existing on disk does not satisfy raw-data persistence for a failed point.

- Q6/Yoko4 at `-10 mA` now has a current-specific raw-I/Q pulse seed:
  `sigma=0.0986291 us`, measured `pi_amp=0.6886101`, qspec drive frequency
  `4375.3926 MHz`. This lives in
  `system_config.PUCQ4_FLUX_PULSE_CALIBRATION`, separate from the zero-current
  `sigma=0.062 us`. Treat it as `revalidation_required`: final post-readout
  qspec was `4377.9192 MHz / 0.81095 MHz FWHM` and post-readout Rabi returned
  the `0.8482879` boundary fit, so coherence was not acquired or claimed.

- Q6/Yoko4 at `-7.5 mA` remains unresolved after a bounded current-specific
  sigma sweep at the established `5061.7466 MHz` candidate. Four sigma values
  from `0.06923-0.09863 us` produced non-monotonic pi fits from `0.279-0.778`;
  all saved I/Q traces were noise-dominated on visual inspection. Do not save a
  `-7.5 mA` sigma or use these fits for coherence. The raw numeric rows may be
  displayed only as rejected evidence.

- Q6/Yoko4 at `-5 mA` also remains unresolved. At the established
  `5587.7655 MHz` candidate, bounded sigma trials `0.0986291`, `0.1126730`,
  `0.0825961`, and `0.0779433 us` returned pi fits `0.8482879`, `0.3492950`,
  `0.5788317`, and `0.7285296`. The last trace was visibly noise-dominated and
  no trial met the `0.6-0.7` acceptance window. Do not retain a `-5 mA` sigma
  seed or promote these fits to shared configuration.

- Q6/Yoko4 at `+5 mA` remains unresolved after a 1200-average raw-I/Q check at
  `7027.0 MHz`. `sigma=0.058001 us` produced a nominal `pi_amp=0.6886101`, but
  the saved I/Q curve was scatter-dominated with no resolved oscillation. Treat
  this repeated value as a fit alias, not a pulse calibration; do not retain the
  `+5 mA` sigma seed or add it to shared configuration.

- A cross-resonator qspec diagnostic was validated on Q6 at `+2.5 mA`: the
  established `6842.1315 MHz` line appeared only through M6 at `7.67` local
  sigma. Repeating all six readout rows at `+5 mA` over `7025-7029 MHz` found
  no row above `2.16` local sigma. This rejects a simple current-dependent
  readout-row swap in that window; the next bounded test is the earlier
  high-power `7025.2 MHz` candidate at its original qspec gain.

- That targeted `7025.2 MHz`/gain-`0.03` M6 repeat returned only `0.77` local
  sigma, so the earlier candidate is rejected as nonreproducible. A bounded
  6987-7067 MHz M6 search is the remaining frequency-error test at `+5 mA`.

- The bounded +5 mA search found the actual Q6 line at `7062.76592 MHz`.
  At qspec gain `0.006`, 3000 reps, and 5 kHz spacing, the fitted FWHM is
  `0.17924 MHz` with an `8.38`-local-sigma amplitude and `0.43`-sigma residual
  RMS. This supersedes rejected 7025/7027/7034 MHz seeds and is approved as the
  +5 mA frequency input for a new Rabi/sigma calibration.

- Q6/Yoko4 at `+5 mA` now has a resolved current-specific pulse calibration:
  `qfreq=7062.76592 MHz`, `sigma=0.0556693 us`, and measured
  `pi_amp=0.6287310`. It is stored separately in
  `PUCQ4_FLUX_PULSE_CALIBRATION`; the 0 mA sigma remains `0.062 us`. Treat the
  new entry as `revalidation_required` until readout optimization and the
  normal qspec/Rabi/coherence chain complete at +5 mA.

## 2026-08-29 renewed zero-current M4 search

- At the user's explicit request, the prior M4 stop rule was reopened without
  new external device evidence. Both Yokogawas were verified in current mode,
  10 mA range, exactly 0 A, with outputs off before acquisition.
- A previously unrecognized validity issue was corrected: the old 4.075-7.075
  GHz sweep crossed the 4.9152 GHz qubit-DAC Nyquist boundary while using zone
  2. The renewed search explicitly used zone 1 below the boundary and zone 2
  above it.
- Persisted short-pulse surveys covered 1.000-4.910 GHz in zone 1 and
  4.916-9.700 GHz in zone 2 through the M4 readout. All candidate extrema were
  at or below 1.22 local sigma. A gain-1.0, 1 us survey over 4.916-7.100 GHz was
  also flat (maximum 1.07 local sigma).
- A 4.916-7.100 GHz cross-resonator survey used every established M1-M6 readout
  point. It recovered clear positive controls near 5518 MHz on M1 and 7085 MHz
  on M5, but M4 and the other rows showed no additional transition. This rules
  out a simple M4-to-other-readout mapping swap in the tested band.
- Raw HDF5 and plots are under `tmp_m4_hr_qspec_0mA/` with timestamps
  `2026-08-29_21-12-58` through `2026-08-29_21-42-29`. No M4 candidate was
  promoted at that time. The then-current unobservable/uncoupled/nonfunctional
  hypothesis was superseded by the September 8 spectroscopy and coherent Rabi.
### Yoko4 / physical Q6 / M6 at +5 mA (validated 2026-08-30)

- Qubit frequency: 7062.771244 MHz (low-power FWHM 0.175556 MHz, qspec gain 0.006).
- Current-indexed sigma: 0.0556693 us; pi amplitude: 0.618751.
- Readout: base 9060.98 MHz, length 12.0 us, gain 0.9167, offset -0.76 MHz, SSF 0.4910.
- Coherence: T1 5.48 us, T2R 2.77 us, T2E 8.00 us.
- Full guarded chain passed and persisted canonical plots/HDF5. Both sources were independently verified at 0 A/output off after completion.
### Yoko4 / physical Q6 / M6 at +7.5 mA (qspec accepted 2026-08-30)

- M6 resonator shifted to approximately 9061.58 MHz; diagnostic readout 9061.06 MHz.
- Accepted low-power qubit line: 7185.658453 MHz, 0.137623 MHz FWHM at gain 0.006, 3,000 reps, 5 kHz spacing.
- The former 7157.242 MHz candidate was noise and is rejected.
- Full-chain final values: 7185.659478 MHz, sigma 0.058001 us, pi 0.638711; readout base 9061.53 MHz, 11.0 us, gain 0.8334, offset -0.04 MHz, SSF 0.4740; T1 5.48 us, T2R 4.13 us, T2E 9.35 us.
- Full guarded validation and canonical persistence passed; both sources independently verified at 0 A/output off.
### Yoko4 / physical Q6 / M6 at +10 mA (qspec accepted 2026-08-30)

- Current-shifted M6 resonator approximately 9061.68 MHz; diagnostic readout 9061.16 MHz.
- Accepted low-power qubit line: 7211.438328 MHz, 0.149861 MHz FWHM at gain 0.006, 3,000 reps, 5 kHz spacing.
- Prior 7153/7162 MHz candidates are rejected.
- Full-chain final values: 7211.443554 MHz, sigma 0.0511568 us, pi 0.628731; readout base 9061.68 MHz, 12.0 us, gain 0.8722, offset -0.16 MHz, SSF 0.4710; T1 5.35 us, T2R 6.35 us, T2E 9.56 us.
- Full guarded validation and canonical persistence passed; both sources independently verified at 0 A/output off.
### Yoko4 / physical Q6 / M6 at -2.5 mA (validated 2026-08-31)

- M6 resonator approximately 9058.16 MHz; diagnostic readout 9058.11 MHz.
- Corrected locator line: 6103.846687 MHz at qspec gain 0.002, 4,000 reps; FWHM 0.23265 MHz and 4.29 local sigma. Prior 6121.9-6124.3 MHz candidates are rejected.
- Full-chain final values: 6103.796753 MHz, post-readout FWHM 0.24612 MHz, sigma 0.0605405 us, pi 0.608771; readout base 9057.78 MHz, 12.0 us, gain 0.7967, offset +0.08 MHz, SSF 0.4270; T1 5.03 us, T2R 1.08 us, T2E 4.46 us.
- The Rabi half-oscillation and coherence fits are visibly resolved. Canonical plots/HDF5 passed, and both sources were independently verified at 0 A/output off.

### Yoko4 / physical Q6 / M6 at -5 mA (pulse calibrated; readout blocked 2026-08-31)

- Corrected-readout qspec diagnostics place the transition near 5587.9 MHz. A gain 0.006, 3,000-repetition narrow scan visibly resolves the response, although the raw normal-chain linewidth was 0.82564 MHz, just above its 0.8 MHz pre-readout bound.
- Bounded raw-I/Q Rabi tuning accepted sigma 0.0671429 us and pi amplitude 0.658671. The I trace visibly resolves just over half an oscillation; the value is stored only as current-specific provisional calibration.
- Two complete alternating readout passes plus the wide rescue sweep failed both before and after the corrected Rabi pulse: best SSF was 0.085 before recalibration and 0.080 after it, below the 0.20 acceptance floor. No T1/T2R/T2E values are accepted for this row.
- The controller aborted and cleaned up; both sources were independently verified in current mode at 0 A/output off.

### Remaining negative-current blockers (2026-08-31)

- Q6 -7.5 mA: corrected qfreq approximately 4840 MHz and provisional
  sigma/pi 0.1222528 us / 0.6985900; post-readout qspec linewidth is unstable
  and has not passed the full chain.
- Q6 -10 mA: raw qspec near 4376.4 MHz and provisional sigma/pi
  0.0986291 us / 0.6886101; four post-readout qspec retries selected broad,
  failed, or noise-spike fits. No coherence is accepted.
- Q4 -10 mA: M5 is a peak near 9012.87 MHz, not a dip. The current-specific
  peak selector recovers qspec 5984.071123 MHz / 0.173861 MHz FWHM. Provisional
  sigma/pi is 0.0451969 us / 0.6985900, but readout rescue fidelity remains
  0.077 after pulse calibration, so the row is not validated.

### Final bounded negative-current reruns (2026-08-31)

- Q4/M5 at -10 mA was repeated with a current-specific readout seed and the M5
  peak selector. The narrow fit collapsed to a 0.009518 MHz noise spike, the
  wide readout rescue reached only 0.086 fidelity, and bounded raw-I/Q Rabi
  retries returned pi amplitudes 0.8483, 0.8483, 0.5589, and 0.4192. The point
  remains failed and no coherence is accepted.
- Q6/M6 at -7.5 mA was repeated from a measured readout seed. The locator found
  4838.9471 MHz; post-readout qspec found 4838.7552 MHz with 0.45768 MHz FWHM,
  and readout fidelity reached 0.3552. Bounded Rabi retries ended at pi 0.77843,
  above the 0.6-0.7 gate, so no coherence is accepted.
- A follow-up forced raw-I/Q Rabi at the newly measured center instead of
  reusing the provisional pi seed. One acquisition produced qspec
  4839.162419 MHz / 0.158162 MHz FWHM and raw pi 0.678630 at sigma
  0.1222528 us, confirming that pulse preparation can work at this bias.
  However, final calibrated-population qspec broadened to 1.367 MHz. Lowering
  only the -7.5 mA qspec gain from 0.006 to 0.004 and raising averages from
  5000 to 8000 produced nonreproducible widths of 1.357 and 2.177 MHz after an
  intermediate bootstrap. That gain/repetition change was reverted. The point
  remains incomplete and no coherence is accepted.
- Both Yokogawas were independently verified in current mode at exactly 0 A
  with outputs off after each guarded run. Further blind retries at these bias
  points are not justified without a new pulse/readout hypothesis.

- Q6/M6 at -10 mA was then repeated with a fresh raw-I/Q pulse calibration at
  4376.306486 MHz and a readout search seeded at -0.40 MHz offset with gain
  allowed through 0.95. Raw Rabi passed the pi window twice (latest pi
  0.688610 at sigma 0.0986291 us), but two optimizer/rescue sequences reached
  only 0.0820 and 0.0802 fidelity. The previous isolated 0.60 optimizer maximum
  is therefore nonreproducible. Qspec also failed its initial Lorentzian gate.
  No coherence is accepted.
- Q6/M6 at -5 mA has the same bounded separation: a visibly resolved raw-I/Q
  pulse calibration at sigma 0.0671429 us / pi 0.658671, followed by repeated
  readout optimization/rescue near 0.08 fidelity. No further resonator-offset
  shift is supported by the completed broad rescue grid, and no coherence is
  accepted.

### Q4/M5 at -10 mA settled recheck (2026-09-02)

- A verified -10.0 mA run with 600 s settling and a +/-100 MHz diagnostic found
  5997.922411 MHz / 0.099989 MHz FWHM. In the continuous validation chain the
  fitted center moved to 6004.213528 MHz / 0.110991 MHz FWHM.
- Bounded 171-point, 300-average raw-I/Q Rabi tuning did not reproduce a usable
  half oscillation. Five fitted pi amplitudes remained outside 0.6-0.7; the last
  was 0.248733. No sigma, pi amplitude, SSF, T1, T2R, or T2E is accepted.
- Q4 is therefore complete at 8 of 9 authorized current-grid points. The -10 mA
  point remains explicitly failed because the spectroscopy center drifts and
  the response does not survive Rabi validation. Yoko3 was returned to 0 A and
  output off by guarded cleanup.
- The reusable 35-slide PowerPoint and aggregate maps were rebuilt. Resonator
  heatmaps now use absolute swept frequency (`freq_pts + freq_center`), and
  calibrated-population/coherence panels leave the failed -10 mA column blank.

### Q4/M5 corrected final rows (2026-09-02; supersedes the failed -10 mA entry above)

- A 5747.922-6247.922 MHz persisted locator scan showed that the earlier
  5984/5998/6004 MHz fits were aliases. The isolated Q4 transition is near
  6037.97 MHz, and the final local population-calibrated qspec fit is
  6037.585408 MHz with 0.203776 MHz FWHM at gain 0.00164927 and 40 us drive.
- At -10 mA the final resonator base is 9010.49 MHz. The selected averaged
  readout is 5 us / gain 0.95 / +0.95 MHz and has low SSF fidelity 0.124. Rabi
  is intentionally provisional under the user's exception: sigma
  0.0165397052 us and pi amplitude 0.994934. Coherence was nevertheless
  acquired as averaged data: T1 17.60 us, T2R 1.61 us, T2E 3.63 us.
- At +5 mA, low-gain raw-I fits could appear narrow but excited only about 0.04
  population. The final population-targeted scan uses gain 0.03 and 40 us,
  finding 7121.580622 MHz, 0.816923 MHz power-broadened FWHM, and peak
  population 0.3051. Final readout is base 9014.59 MHz, 6 us / 0.92 / -0.48
  MHz, SSF 0.512; sigma 0.0808489431 us, pi 0.678630; T1 2.31 us, T2R 3.99 us,
  and T2E 1.90 us.
- At -2.5 mA, the strict population check selected gain 0.0025 and 20 us:
  6945.111992 MHz, 0.120213 MHz FWHM, peak population 0.3036, and SNR 128.4.
  Fresh Rabi at that exact frequency gave sigma 0.0591114617 us and pi
  0.648691. Final readout is base 9013.74 MHz, 6 us / 0.73 / +0.32 MHz, SSF
  0.6183; T1 11.95 us, T2R 3.63 us, and T2E 9.07 us.
- All values above are current-indexed and do not replace the six-row
  zero-current arrays. Both sources were independently read back in current
  mode at 0.0 A with outputs off after every guarded run.

### Q4/M5 identical-grid qspec closure (2026-09-03)

- The final Q4 map contains canonical HDF5 rows at all nine authorized Yoko 3
  currents: -10, -7.5, -5, -2.5, 0, +2.5, +5, +7.5, and +10 mA. Every row
  requested the same 5950-7200 MHz sweep at 0.02 MHz spacing and 300 reps. The
  QICK-realized axis is identical for all rows: 62,501 points from
  5950.00000305 to 7198.83652039 MHz with 0.01998138 MHz median spacing.
- The final widened/local qspec centers and physical readout tones used for the
  common-grid acquisition are:

  | Yoko 3 (mA) | qspec center (MHz) | readout tone (MHz) | map gain | map length (us) |
  |---:|---:|---:|---:|---:|
  | -10.0 | 6037.585408 | 9011.44 | 0.030000 | 40 |
  | -7.5 | 6422.620535 | 9012.22 | 0.000839 | 20 |
  | -5.0 | 6724.886451 | 9012.87 | 0.002000 | 20 |
  | -2.5 | 6945.111992 | 9014.06 | 0.002500 | 20 |
  | 0.0 | 7084.565678 | 9014.94 | 0.003000 | 20 |
  | +2.5 | 7143.859571 | 9014.56 | 0.004500 | 20 |
  | +5.0 | 7121.580622 | 9014.11 | 0.030000 | 40 |
  | +7.5 | 7022.121082 | 9013.80 | 0.002000 | 20 |
  | +10.0 | 6841.201740 | 9012.97 | 0.002000 | 20 |

- Each current row acquired 1,000 ground and 1,000 excited SSF shots and uses
  one fixed centroid pair for its entire 62,501-frequency trace. All rows use
  the same signed affine ground=0/excited=1 population definition, fixed 0-1
  color scale, and no per-row normalization. The raw centroids intentionally
  differ between currents because the validated physical readout tone, gain,
  and length differ. Full-grid peak offsets from the final local centers are
  -0.281 to +0.148 MHz and local population SNR spans 6.99-26.23.
- The closest common-grid boundary is still 54.98 MHz from its selected center;
  the -10 mA center is 87.59 MHz above the lower edge. Thus every bias has a
  resolved transition inside the shared range rather than an edge-selected fit.
  Every saved Rabi row was also audited to use its corresponding final widened-
  scan qspec center exactly.
- Final report products include raw-signal and calibrated-population qspec
  heatmaps with current on x and the complete realized frequency axis on y,
  full-range curve galleries, a frequency/Rabi consistency check, widened
  resonator map, upward-sorted Rabi map, and all-nine-bias T1/T2R/T2E maps.
  They are under `yoko_flux_campaign/aggregate_2d/`; the reusable
  `PUCQ4_first_light_characterization.pptx` was rebuilt as 32 slides.
- The -10 mA pulse remains explicitly provisional under the user's exception
  (sigma 0.0165397052 us, pi amplitude 0.994934) and its SSF fidelity is only
  0.124. The +5 mA row is the population-valid but power-broadened solution
  (0.816923 MHz FWHM). Neither exception is hidden by map normalization.
- After acquisition, Yoko 3 and Yoko 4 were independently queried in current
  mode at exactly 0.0 A with outputs off.

### Two-Yoko lower-frequency qspec-map closure (2026-09-03)

- The follow-up map extends low enough to contain every current-grid line. Q4/M5
  requested 5000-7300 MHz in overlapping 5000-5802, 5800-6552, and
  6550-7300 MHz bands. Its common stitched QICK axis is
  5000.000002-7299.688149 MHz with 0.03998337 MHz median spacing. Q6/M6
  requested 4000-7300 MHz in 4000-5102, 5100-6202, and 6200-7300 MHz bands;
  its common stitched axis is 4000.000005-7299.542615 MHz with the same median
  spacing. Every one of the nine current columns is present for both devices.
- Production used 100 qspec averages per frequency. Every current and band
  acquired its own built-in 1,000-shot ground plus 1,000-shot excited SSF
  calibration with 250 us SSF relaxation. The saved population is the signed
  affine projection with ground=0 and excited=1. No current-row normalization
  or global SSF replacement is used. The three bands per row make 54 canonical
  qspec HDF5 files in total.
- Resonator DAC attenuations were [10, 15] dB, qubit DAC attenuations [5, 4] dB,
  and ADC attenuation 17 dB. Each row used the physical readout tone
  `resonator base + current-specific offset`, verified from the round-robin log.
  The exact inputs and bounded local checks are:

  | I (mA) | Q4 f01 used | wide local center | qspec gain/length | fres -> readout tone | readout length/gain/offset |
  |---:|---:|---:|---:|---:|---:|
  | -10.0 | 6037.585408 | 6037.900126 | 0.03 / 40 us | 9010.49 -> 9011.44 | 5 us / 0.95 / +0.95 MHz |
  | -7.5 | 6422.620535 | 6422.474029 | 0.000839 / 20 us | 9011.54 -> 9012.22 | 7 us / 0.95 / +0.68 MHz |
  | -5.0 | 6724.886451 | 6724.621847 | 0.002 / 20 us | 9012.79 -> 9012.87 | 6 us / 0.95 / +0.08 MHz |
  | -2.5 | 6945.111992 | 6944.870440 | 0.0025 / 20 us | 9013.74 -> 9014.06 | 6 us / 0.73 / +0.32 MHz |
  | 0.0 | 7084.565678 | 7084.145347 | 0.003 / 20 us | 9014.74 -> 9014.94 | 6 us / 0.65 / +0.20 MHz |
  | +2.5 | 7143.859571 | 7143.807797 | 0.0045 / 20 us | 9015.04 -> 9014.56 | 5 us / 0.93 / -0.48 MHz |
  | +5.0 | 7121.580622 | 7121.467029 | 0.03 / 40 us | 9014.59 -> 9014.11 | 6 us / 0.92 / -0.48 MHz |
  | +7.5 | 7022.121082 | 7022.465551 | 0.002 / 20 us | 9014.04 -> 9013.80 | 10 us / 0.95 / -0.24 MHz |
  | +10.0 | 6841.201740 | 6841.627731 | 0.002 / 20 us | 9013.09 -> 9012.97 | 6 us / 0.95 / -0.12 MHz |

  | I (mA) | Q6 f01 used | wide local center | qspec gain/length | fres -> readout tone | readout length/gain/offset |
  |---:|---:|---:|---:|---:|---:|
  | -10.0 | 4376.306486 | 4375.479872 | 0.03 / 20 us | 9055.31 -> 9054.91 | 12 us / 0.88 / -0.40 MHz |
  | -7.5 | 4839.162419 | 4839.645692 | 0.03 / 20 us | 9056.01 -> 9055.25 | 12 us / 0.7945 / -0.76 MHz |
  | -5.0 | 5587.900000 | 5588.112003 | 0.03 / 20 us | 9056.91 -> 9056.63 | 12 us / 0.8767 / -0.28 MHz |
  | -2.5 | 6103.796753 | 6103.955256 | 0.002 / 20 us | 9057.78 -> 9057.86 | 12 us / 0.7967 / +0.08 MHz |
  | 0.0 | 6522.568402 | 6522.631871 | 0.006 / 20 us | 9059.26 -> 9058.86 | 10 us / 0.95 / -0.40 MHz |
  | +2.5 | 6842.131507 | 6842.044055 | 0.006 / 20 us | 9060.46 -> 9059.94 | 12 us / 0.7967 / -0.52 MHz |
  | +5.0 | 7062.771244 | 7062.717440 | 0.006 / 20 us | 9060.98 -> 9060.22 | 12 us / 0.9167 / -0.76 MHz |
  | +7.5 | 7185.659478 | 7185.606785 | 0.006 / 20 us | 9061.53 -> 9061.49 | 11 us / 0.8334 / -0.04 MHz |
  | +10.0 | 7211.443554 | 7211.410489 | 0.006 / 20 us | 9061.68 -> 9061.52 | 12 us / 0.8722 / -0.16 MHz |

- The `wide local center` column is a bounded positive-Lorentzian check of the
  wide, map-power data, not a replacement pulse calibration. In particular,
  Q6 at -10 mA is broad/noisy and reached the 2.0 MHz fit-width bound; Q6
  -10/-7.5/-5 mA therefore remain spectroscopy-map evidence rather than newly
  accepted full-chain Rabi/coherence points. The local-curve slides expose this
  directly instead of hiding it with normalization.
- The full-range heatmaps put current on x, every actually swept frequency on
  y, and saved calibrated population on the color scale. For slide
  rasterization, ordinary bins are averaged and only bins within +/-1 MHz of
  the validated f01 preserve their local five-sample maximum. The raw arrays
  remain unchanged in canonical HDF5, and separate nine-panel plots show the
  actual samples within +/-5 MHz of every line.
- `PUCQ4_first_light_characterization.pptx` now has 36 slides. Slides 33-36 are
  the two full-range maps, both per-current readout tables, and both local-curve
  verification galleries. A pre-update 32-slide backup is retained beside the
  deck. Independent final readback found both Yokogawas in current mode at
  exactly 0.0 A with output off.

### Continuous one-sweep qspec-map closure (2026-09-05; supersedes the segmented map for display)

- `pucq4_08_single_sweep_qspec_maps.py` acquired one uninterrupted full-range
  qspec program per current, rather than joining frequency bands. Both devices
  use the authorized nine-point current grid from -10 to +10 mA. Q4/M5 has
  57,501 points/current on an identical realized axis of
  5000.00000153-7299.04365692 MHz with 0.0399833700 MHz median spacing. Q6/M6
  has 82,501 points/current on an identical axis of
  4000.00000534-7298.62785873 MHz at the same spacing.
- Every frequency used 500 repetitions. Each complete current row used exactly
  one built-in 2,000-ground plus 2,000-excited-shot SSF calibration with 250 us
  relaxation, held fixed across that row. Each sweep used its current-specific
  physical resonator readout tone from the final readout table; no resonator
  base frequency was substituted. Attenuations were resonator DAC [10, 15] dB,
  qubit DAC [5, 4] dB, and ADC 17 dB.
- The positive population maxima measured from the continuous map-power data
  are below. These are map observations, not replacements for the validated
  local qspec/Rabi drive frequencies stored in shared pulse calibration.

  | Current (mA) | Q4/M5 peak (MHz) | Q4 local SNR | Q6/M6 peak (MHz) | Q6 local SNR |
  |---:|---:|---:|---:|---:|
  | -10.0 | 6038.168150 | 21.95 | 4375.643747 | 3.76 |
  | -7.5 | 6422.888116 | 7.42 | 4839.850648 | 4.77 |
  | -5.0 | 6725.202361 | 12.22 | 5587.979446 | 11.92 |
  | -2.5 | 6945.390768 | 16.33 | 6103.644942 | 3.78 |
  | 0.0 | 7084.532888 | 13.01 | 6522.550687 | 17.96 |
  | +2.5 | 7143.788239 | 30.23 | 6842.137747 | 19.34 |
  | +5.0 | 7121.477520 | 16.44 | 7062.765971 | 15.78 |
  | +7.5 | 7022.398734 | 20.55 | 7185.594878 | 10.87 |
  | +10.0 | 6841.593945 | 13.24 | 7211.424133 | 17.14 |

- All 18 rows completed and saved to the canonical round-robin qspec HDF5
  hierarchy. Q6 at -10 and -2.5 mA fall slightly below the map controller's
  conservative local-SNR threshold of 4. An independent review still locates
  the same responses: -10 mA has a 4375.747297 MHz joint center and about
  0.5245 MHz FWHM; -2.5 mA has a 6103.609930 MHz joint center and about
  0.5037 MHz FWHM. Their low state separation remains visible in the local
  panels and must not be described as optimized SSF fidelity.
- Final heatmaps put actual swept frequency on y, current on x, and either
  calibrated population contrast or SSF-axis signal contrast on color. A slow
  interpolated 20-MHz block-median baseline is subtracted from each fixed-
  smoothing row, but row amplitude and peak height are never normalized. The
  display has 1,500 frequency bins: ordinary bins average the smoothed samples,
  while the +/-1 MHz measured-line bin preserves its local maximum and is
  repeated over three display pixels so a sub-MHz peak remains visible over a
  multi-GHz axis. A fixed gamma=0.55 color transform improves low-level visual
  contrast; population uses the same 0.00-0.35 Delta-Pe scale for both devices.
  Raw HDF5 arrays are unchanged. Nine-panel companions show the saved physical
  population samples and local scales without normalization.
- The reusable deck was rebuilt from the clean 32-slide checkpoint and now has
  38 slides. Slides 33-35 contain the Q4 population map/readout table, signal
  map, and local curves; slides 36-38 contain the same for Q6. The table records
  current, validated seed, measured map peak, resonator base, physical readout
  tone, readout length, gain, and offset. PowerPoint COM reopening and full
  1600x900 slide exports verified all six additions. The earlier 36-slide
  segmented-map deck remains as
  `PUCQ4_first_light_characterization_superseded_segmented_qspec_20260904.pptx`.
- This focused run did not reacquire Rabi or coherence. The clean checkpoint
  already contains the final res-spec, Rabi, SSF, T1, T2R, and T2E evidence and
  tables; only the superseded qspec-map appendix was replaced.
- During the first attempt, loss of the attached terminal killed Python while
  Q4 +7.5 mA was running, before HDF5 creation and before `finally` cleanup.
  Yoko3 was found at +7.5 mA with output enabled and was immediately ramped at
  0.5 mA/s to 0 A and disabled; Yoko4 was already safe. The incomplete row was
  reacquired under a hidden detached process with inherited Conda DLL paths.
  After final completion, independent readback found Yoko3 and Yoko4 in current
  mode at exactly 0.0 A with outputs off.

### Peak-visible continuous-map display revision (2026-09-06)

- The original continuous-map PNGs compressed sub-MHz peaks into less than one
  raster row and allowed current-dependent additive SSF baselines to dominate
  the color scale, even though the nine local panels clearly resolved the same
  transitions. No new acquisition was necessary: the existing 18 HDF5 rows
  already use the complete identical frequency grid at every current, the same
  qspec gain/length as the local panels, 0.0399833700 MHz median spacing, and
  500 repetitions per frequency.
- `tools/build_pucq4_single_sweep_qspec_maps.py` now produces explicit
  `population_contrast_2d` and `signal_contrast_2d` products. Each row receives
  only a slow interpolated 20-MHz block-median subtraction, then fixed 5-point
  population or 11-point signal smoothing. Ordinary 1,500-bin display cells are
  averaged; the measured +/-1 MHz line cell retains its local maximum and is
  repeated across three raster rows. A fixed gamma=0.55 color transform is used,
  with the same 0.00-0.35 Delta-Pe population scale for Q4 and Q6. There is no
  row amplitude, peak-height, range, or noise normalization, and canonical HDF5
  values are unchanged.
- PowerPoint slides 33-38 were replaced with the contrast maps and verified by
  a fresh 1600x900 PowerPoint COM export. The colored ridge is visible at all
  nine currents for both devices, including Q4 and Q6 at -10 mA. The population
  slides now tabulate qspec drive gain and length in addition to seed/observed
  frequency, resonator base, shifted readout tone, readout length/gain, and
  offset. The local physical-population panels remain beside each map as the
  quantitative visual check.

### Fixed-sigma current-by-gain Rabi maps (2026-09-07)

- `pucq4_09_single_sweep_rabi_maps.py` completed one amplitude-Rabi sweep at
  each of the nine authorized currents for both tunable devices. Every row used
  the observed population maximum from the continuous wide-qspec map as its
  qubit drive frequency and the exact shifted physical readout tone used for
  that qspec row. The controller verified both values before acquisition.
- Q4/M5 used one fixed, map-local sigma of 0.060 us at all currents; Q6/M6 used
  0.090 us. These values were not written into the shared/current-indexed
  system configuration. The requested gain axis was 0.000-1.000 with 201
  points; the common QICK-realized axis for all 18 rows is
  0.00000000-0.99493377 with 0.00497467 median spacing. Acquisition used 1,200
  repetitions x 1 round and 250 us Rabi relaxation.
- Every row acquired and saved its own 2,000-ground plus 2,000-excited-shot SSF
  calibration with 250 us relaxation. Population is the signed affine
  projection with ground=0 and excited=1. No current-row offset, division,
  range scaling, peak normalization, or other amplitude normalization is used.
  Resonator DAC attenuations were [10, 15] dB, qubit DAC attenuations [5, 4]
  dB, and ADC attenuation was 17 dB.
- Final current-specific inputs and measured outputs are:

  | I (mA) | Q4 qfreq | Q4 fres -> RO | RO us / gain / off | Q4 pi | Pe span | SSF sep |
  |---:|---:|---:|---:|---:|---:|---:|
  | -10.0 | 6038.168150 | 9010.49 -> 9011.44 | 5 / 0.9500 / +0.95 | 0.994934 | 0.9615 | 2.8400 |
  | -7.5 | 6422.888116 | 9011.54 -> 9012.22 | 7 / 0.9500 / +0.68 | 0.641732 | 0.9999 | 3.5868 |
  | -5.0 | 6725.202361 | 9012.79 -> 9012.87 | 6 / 0.9500 / +0.08 | 0.915339 | 0.9791 | 4.2084 |
  | -2.5 | 6945.390768 | 9013.74 -> 9014.06 | 6 / 0.7300 / +0.32 | 0.636758 | 1.0064 | 3.2652 |
  | +0.0 | 7084.532888 | 9014.74 -> 9014.94 | 6 / 0.6500 / +0.20 | 0.611884 | 0.9390 | 2.9526 |
  | +2.5 | 7143.788239 | 9015.04 -> 9014.56 | 5 / 0.9300 / -0.48 | 0.577062 | 0.9681 | 3.9891 |
  | +5.0 | 7121.477520 | 9014.59 -> 9014.11 | 6 / 0.9200 / -0.48 | 0.631783 | 0.9547 | 3.8216 |
  | +7.5 | 7022.398734 | 9014.04 -> 9013.80 | 10 / 0.9500 / -0.24 | 0.596960 | 0.9718 | 3.5960 |
  | +10.0 | 6841.593945 | 9013.09 -> 9012.97 | 6 / 0.9500 / -0.12 | 0.611884 | 1.0026 | 4.0314 |

  | I (mA) | Q6 qfreq | Q6 fres -> RO | RO us / gain / off | Q6 pi | Pe span | SSF sep |
  |---:|---:|---:|---:|---:|---:|---:|
  | -10.0 | 4375.643747 | 9055.31 -> 9054.91 | 12 / 0.8800 / -0.40 | 0.751175 | 0.9296 | 0.4776 |
  | -7.5 | 4839.850649 | 9056.01 -> 9055.25 | 12 / 0.7945 / -0.76 | 0.925288 | 1.3138 | 0.5371 |
  | -5.0 | 5587.979446 | 9056.91 -> 9056.63 | 12 / 0.8767 / -0.28 | 0.512391 | 0.9175 | 1.0932 |
  | -2.5 | 6103.644942 | 9057.78 -> 9057.86 | 12 / 0.7967 / +0.08 | 0.402948 | 0.8718 | 1.3952 |
  | +0.0 | 6522.550687 | 9059.26 -> 9058.86 | 10 / 0.9500 / -0.40 | 0.452695 | 0.9259 | 2.3381 |
  | +2.5 | 6842.137747 | 9060.46 -> 9059.94 | 12 / 0.7967 / -0.52 | 0.388024 | 0.9781 | 2.0019 |
  | +5.0 | 7062.765971 | 9060.98 -> 9060.22 | 12 / 0.9167 / -0.76 | 0.383050 | 0.9250 | 2.0193 |
  | +7.5 | 7185.594878 | 9061.53 -> 9061.49 | 11 / 0.8334 / -0.04 | 0.417872 | 0.9594 | 1.7683 |
  | +10.0 | 7211.424133 | 9061.68 -> 9061.52 | 12 / 0.8722 / -0.16 | 0.363151 | 0.8776 | 1.6457 |

- All 18 canonical Rabi HDF5 files, plots, logs, and 2,000-sample copies of all
  four SSF arrays passed the post-run audit. The Q4 pi range is
  0.577062-0.994934 and the Q6 range is 0.363151-0.925288. The fixed-sigma map
  intentionally retains endpoints outside the separate 0.6-0.7 sigma-tuning
  target. Q6 -7.5 mA has low SSF centroid separation and a measured affine
  population overshoot (5-95% span 1.3138); it remains a clear Rabi response and
  is shown without normalization.
- Final heatmaps place current on x, actual realized gain on y, and calibrated
  population on color. Both devices share a physical -0.30 to 1.50 display
  scale and a fixed three-sample display mean; canonical HDF5 is unchanged.
  Companion nine-panel plots show saved samples, the same mean, saved fit, and
  first pi extremum for every current.
- `PUCQ4_first_light_characterization.pptx` now has 42 slides. Slides 39-42 are
  the Q4 heatmap/settings table, Q4 curves, Q6 heatmap/settings table, and Q6
  curves. A 38-slide pre-Rabi backup is retained as
  `PUCQ4_first_light_characterization_pre_fixed_rabi_20260907.pptx`. PowerPoint
  COM reopened the final deck and exported all four additions at 1600x900 for
  visual verification. Final instrument readback found both Yokogawas in
  current mode at exactly 0 A with output disabled.

### Ramsey-corrected common-60-ns Rabi maps (2026-09-07)

- `pucq4_10_ramsey_corrected_rabi_maps.py` completed the requested two-stage
  acquisition at all nine authorized currents for both tunable devices. It first
  called the same `T2RMeasurement.adjust_qspec` path used by
  `round_robin_benchmark.py`, starting from the prior wide-qspec frequency and
  prior fitted Rabi pi amplitude. The resulting corrected qubit frequency was
  then used immediately for a new current-by-gain Rabi row.
- Ramsey used 0-1 us, 50 delay points, 4 MHz intentional detuning, 1,000
  repetitions x 3 rounds, and 4,000 us relaxation. The correction pulse used
  the prior calibrated pi amplitude and its matching calibration sigma (Q4
  0.060 us; Q6 0.090 us), because changing that pulse would invalidate the
  frequency-correction measurement. The final Rabi maps use the requested one
  common sigma of 0.060 us for both Yokos, gain 0-1 at 201 points, 1,200
  repetitions x 2 rounds, 250 us relaxation, and 4,000 SSF shots per state.
  The shifted physical resonator readout tone and readout settings from each
  accepted qspec row were preserved exactly.
- Final corrected frequencies and Rabi-map inputs/outputs are:

  | I (mA) | Q4 seed -> corrected qfreq (MHz) | df (MHz) | fres -> RO (MHz) | RO us / gain / off (MHz) | pi gain | Pe 5-95% span |
  |---:|---:|---:|---:|---:|---:|---:|
  | -10.0 | 6038.168150 -> 6038.096816 | -0.071333 | 9010.49 -> 9011.44 | 5 / 0.9500 / +0.95 | 0.994934 | 0.9776 |
  | -7.5 | 6422.888116 -> 6422.855710 | -0.032406 | 9011.54 -> 9012.22 | 7 / 0.9500 / +0.68 | 0.641732 | 1.0141 |
  | -5.0 | 6725.202361 -> 6725.005709 | -0.196651 | 9012.79 -> 9012.87 | 6 / 0.9500 / +0.08 | 0.910364 | 0.9605 |
  | -2.5 | 6945.390768 -> 6945.200012 | -0.190755 | 9013.74 -> 9014.06 | 6 / 0.7300 / +0.32 | 0.636758 | 0.9961 |
  | +0.0 | 7084.532888 -> 7084.372835 | -0.160053 | 9014.74 -> 9014.94 | 6 / 0.6500 / +0.20 | 0.616859 | 0.9821 |
  | +2.5 | 7143.788239 -> 7143.693599 | -0.094641 | 9015.04 -> 9014.56 | 5 / 0.9300 / -0.48 | 0.577062 | 0.9807 |
  | +5.0 | 7121.477520 -> 7122.150310 | +0.672790 | 9014.59 -> 9014.11 | 6 / 0.9200 / -0.48 | 0.636758 | 0.9617 |
  | +7.5 | 7022.398734 -> 7022.140004 | -0.258730 | 9014.04 -> 9013.80 | 10 / 0.9500 / -0.24 | 0.596960 | 0.9573 |
  | +10.0 | 6841.593945 -> 6841.237437 | -0.356507 | 9013.09 -> 9012.97 | 6 / 0.9500 / -0.12 | 0.611884 | 0.9807 |

  | I (mA) | Q6 seed -> corrected qfreq (MHz) | df (MHz) | fres -> RO (MHz) | RO us / gain / off (MHz) | pi gain | Pe 5-95% span |
  |---:|---:|---:|---:|---:|---:|---:|
  | -10.0 | 4375.643747 -> 4375.815431 | +0.171684 | 9055.31 -> 9054.91 | 12 / 0.8800 / -0.40 | 0.994934 | 0.5888 |
  | -7.5 | 4839.850648 -> 4839.119349 | -0.731300 | 9056.01 -> 9055.25 | 12 / 0.7945 / -0.76 | 0.994934 | 0.8995 |
  | -5.0 | 5587.979446 -> 5588.171561 | +0.192115 | 9056.91 -> 9056.63 | 12 / 0.8767 / -0.28 | 0.766099 | 0.9472 |
  | -2.5 | 6103.644942 -> 6103.840361 | +0.195419 | 9057.78 -> 9057.86 | 12 / 0.7967 / +0.08 | 0.606910 | 0.7220 |
  | +0.0 | 6522.550687 -> 6522.361093 | -0.189594 | 9059.26 -> 9058.86 | 10 / 0.9500 / -0.40 | 0.676555 | 0.9880 |
  | +2.5 | 6842.137747 -> 6841.856793 | -0.280954 | 9060.46 -> 9059.94 | 12 / 0.7967 / -0.52 | 0.582036 | 0.9842 |
  | +5.0 | 7062.765971 -> 7062.539050 | -0.226922 | 9060.98 -> 9060.22 | 12 / 0.9167 / -0.76 | 0.577062 | 1.0053 |
  | +7.5 | 7185.594878 -> 7185.383883 | -0.210995 | 9061.53 -> 9061.49 | 11 / 0.8334 / -0.04 | 0.626808 | 0.9621 |
  | +10.0 | 7211.424133 -> 7211.196119 | -0.228015 | 9061.68 -> 9061.52 | 12 / 0.8722 / -0.16 | 0.537264 | 0.9171 |

- All 18 accepted Ramsey records and all 18 final Rabi records have canonical
  HDF5 and plot artifacts. The common realized Rabi gain axis is
  0.00000000-0.99493377 with 0.00497467 median spacing. Maximum independent
  Ramsey-frequency standard error was 0.00601 MHz for Q4 and 0.02699 MHz for
  Q6. Minimum Rabi 5-95% population span was 0.9573 for Q4 and 0.5888 for Q6.
- The first Q6 -10 mA Ramsey attempt was rejected before Rabi because the
  canonical endpoint heuristic selected a weak Q quadrature. Independent fits
  showed a strong I fringe, but the row was reacquired rather than promoted.
  The accepted repeat selected I and passed both the saved and independent fit
  checks. The validation now checks both quadratures and permits a weak saved
  quadrature only when an independent strong fit agrees within the stated
  uncertainty and frequency limits.
- Final plots and machine-readable provenance are under
  `yoko_flux_campaign/ramsey_corrected_rabi_2d/`. The reusable PowerPoint now has
  48 slides; slides 43-48 contain the Q4 and Q6 Ramsey galleries, Rabi heatmaps
  with complete frequency/readout tables, and companion Rabi curves. A
  pre-update backup is retained as
  `PUCQ4_first_light_characterization_pre_ramsey_rabi_20260907.pptx`. PowerPoint
  COM reopened the deck and a fresh 1600x900 export of all six new slides passed
  visual inspection. Final independent readback found both Yokogawas in current
  mode at exactly 0 A with output disabled.

### Ramsey-corrected current-indexed sigma and coherence maps (2026-09-07/08)

- `pucq4_11_sigma_coherence_maps.py` completed the requested fresh chain at all
  nine authorized currents for both tunable devices: Ramsey frequency
  correction, bounded sigma tuning to a 0.650 target pi gain, then T1, T2R, and
  T2E using the accepted corrected frequency and selected sigma. Acquisition
  order was 0, -2.5, -5, -7.5, -10, +2.5, +5, +7.5, +10 mA. The sources used
  current mode, 0.010 A range, and a 0.0005 A/s ramp; Q4 -10 mA retained its
  required 600 s settling interval.
- Standard Ramsey settings were actual delay 0-1 us requested, 50 points,
  4 MHz intentional detuning, 1,000 reps x 3 rounds, and 4,000 us relaxation.
  Q6 -7.5 and -10 mA used the planned high-statistics 2,000 reps x 4 rounds.
  Sigma tuning requested gain 0-1 at 201 points, used 1,200 reps x 2 rounds,
  250 us relaxation, and 4,000 SSF shots/state per attempt. The bounded sigma
  interval was 0.025-0.160 us, with target 0.650 and acceptance 0.625-0.675.
  Population is the saved signed current-specific SSF affine projection with
  ground=0 and excited=1; no row normalization was used.
- Final Q4/Yoko3/M5 values are:

  | I (mA) | corrected f01 (MHz) | sigma (ns) | pi gain | T1 (us) | T2R (us) | T2E (us) |
  |---:|---:|---:|---:|---:|---:|---:|
  | -10.0 | 6038.019853 | 96.998 | 0.651682 | 23.728 | 1.058 | 4.660 |
  | -7.5 | 6422.718837 | 59.237 | 0.651682 | 16.639 | 1.492 | 5.920 |
  | -5.0 | 6724.844814 | 84.034 | 0.651682 | 14.807 | 2.111 | 7.356 |
  | -2.5 | 6945.029248 | 58.778 | 0.646707 | 12.663 | 3.492 | 9.684 |
  | +0.0 | 7084.280775 | 56.941 | 0.646707 | 12.440 | 6.124 | 13.851 |
  | +2.5 | 7143.681614 | 53.267 | 0.651682 | 8.626 | 6.470 | 13.621 |
  | +5.0 | 7122.148333 | 58.778 | 0.651682 | 5.681 | 1.304 | 8.890 |
  | +7.5 | 7022.138349 | 55.104 | 0.651682 | 7.214 | 3.767 | 8.624 |
  | +10.0 | 6841.221063 | 56.482 | 0.646707 | 15.897 | 2.140 | 7.979 |

- Final Q6/Yoko4/M6 values are:

  | I (mA) | corrected f01 (MHz) | sigma (ns) | pi gain | T1 (us) | T2R (us) | T2E (us) |
  |---:|---:|---:|---:|---:|---:|---:|
  | -10.0 | 4375.850090 | 105.432 | 0.646707 | 18.946 | 0.665 | 4.035 |
  | -7.5 | 4839.206595 | 132.845 | 0.646707 | 10.898 | 0.885 | 3.780 |
  | -5.0 | 5588.103711 | 70.717 | 0.656656 | 6.463 | 0.723 | 3.462 |
  | -2.5 | 6103.736073 | 56.022 | 0.651682 | 5.917 | 1.174 | 4.151 |
  | +0.0 | 6522.416914 | 62.451 | 0.646707 | 5.535 | 1.535 | 4.998 |
  | +2.5 | 6841.879743 | 53.726 | 0.651682 | 7.219 | 2.288 | 6.164 |
  | +5.0 | 7062.539019 | 53.267 | 0.646707 | 6.308 | 3.342 | 7.630 |
  | +7.5 | 7185.390637 | 57.859 | 0.651682 | 5.602 | 4.927 | 8.033 |
  | +10.0 | 7211.220857 | 49.594 | 0.641732 | 6.304 | 4.449 | 9.649 |

- Coherence requested 0.01-60 us/241 points for T1, 0-30 us/301 points and
  0.5 MHz detuning for T2R, and 0.01-60 us/301 points and 0.5 MHz detuning for
  T2E. Standard acquisition used 1,200 reps x 2 rounds, 250 us relaxation, and
  a separate 4,000-shot/state SSF calibration for every curve. A targeted
  2,000-rep x 4-round retry was made for all three Q6 -7.5 and -10 mA curves.
  Every -7.5 mA candidate had lower R2, so the prior curves were retained; all
  three -10 mA candidates improved and were selected. Old and new HDF5/plot
  artifacts remain intact. The selected Q6 -10 mA fits are T1 18.946 us
  (R2 0.979), T2R 0.665 us (R2 0.557), and T2E 4.035 us (R2 0.742).
- The final audit found 18/18 complete rows, 180/180 expected canonical
  HDF5/plot paths, and 4,000 saved samples in every SSF I/Q ground/excited
  array. Q4 pi gains span 0.646707-0.651682 and Q6 spans
  0.641732-0.656656. Minimum saved Ramsey/Rabi R2 is 0.9965/0.9975 for Q4 and
  0.9430/0.9547 for Q6. Selected coherence minimum R2 values are
  0.9940/0.9086/0.9396 for Q4 T1/T2R/T2E and
  0.9672/0.5531/0.6557 for Q6.
- `tools/build_pucq4_sigma_coherence_maps.py` generated fresh Ramsey galleries,
  sigma-versus-current plots, selected-sigma Rabi galleries, and current-x /
  actual-delay-us-y / saved-population-color T1/T2R/T2E maps plus companion
  curves. Pulse-dependent QICK timing shifts the native delay centers slightly
  between currents, so each heatmap column uses its own validated midpoint
  edges; no interpolation or substituted common axis is used. Display uses one
  fixed -0.30 to 1.50 population scale and a disclosed three-point mean only.
- Final values were promoted to
  `system_config.PUCQ4_FLUX_PULSE_CALIBRATION` for all 18 current/device
  combinations. This replaces the old provisional pulse-only entries but does
  not modify the six-row zero-current `sigma` or `pi_amp` arrays.
- The source manifest is
  `yoko_flux_campaign/ramsey_sigma_coherence_manifest.json`; final plots and
  `ramsey_sigma_coherence_provenance.json` are under
  `yoko_flux_campaign/ramsey_sigma_coherence_2d/`. The reusable PowerPoint now
  has 60 slides. Slides 49-60 contain each device's fresh Ramsey evidence,
  sigma/current plot and full frequency table, accepted Rabi curves, and
  T1/T2R/T2E heatmaps with companion curves. A 48-slide baseline is retained
  as `PUCQ4_first_light_characterization_pre_sigma_coherence_20260907.pptx`;
  all 12 additions passed fresh 1600x900 PowerPoint COM export and visual
  inspection. Final independent source readback found both Yokogawas in current
  mode, 0.010 A range, exactly 0 A, with output disabled.

### Runtime promotion of the final current calibration (2026-09-08)

- The final 18-row calibration is now the executable source of truth, exposed
  as `system_config.PUCQ4_FLUX_CALIBRATION`. Each row contains corrected f01,
  qspec gain and rectangular-drive length, Gaussian sigma (with QICK pulse
  length derived as 4*sigma), pi gain, resonator base, physical readout tone,
  readout offset/gain/length, and the selected coherence results. The earlier
  fit candidates remain private history and are not used at runtime.
- `round_robin_benchmark.py` has two explicit selectors at the top:
  `YOKO3_CURRENT_MA` for physical Q4/M5/Python index 4 and
  `YOKO4_CURRENT_MA` for physical Q6/M6/Python index 5. The subprocess forms
  are `PUCQ4_YOKO3_CURRENT_MA` and `PUCQ4_YOKO4_CURRENT_MA`. Only
  -10, -7.5, -5, -2.5, 0, +2.5, +5, +7.5, and +10 mA are accepted; invalid
  values fail before folder creation or QICK connection.
  At the initial table promotion these selectors chose configuration only.
  Superseded September 8: RR's `set_yoko_bias` now sets/verifies both physical
  currents, selects the matching profiles, and returns both sources to zero/off.
- `resolve_pucq4_flux_config` combines the selected two rows with the four
  fixed rows and returns ordinary six-value lists. `QICK_experiment` uses those
  lists for f01, qspec gain/length, sigma, pi gain, resonator base, readout
  offset/gain/length. `build_state` then selects the requested qubit and copies
  `res_length_ge` into the scalar `res_length` consumed by QICK, so no nested
  current table reaches an experiment program.
- `apply_pucq4_flux_experiment_config` updates the M5/M6 `VNA_qubit` and
  `VNA_res` entries and recenters all f01-dependent spectroscopy/chevron
  frequency bounds. The static zero-current defaults were also promoted. For
  Q4: f01 7084.77785 -> 7084.280775100 MHz, resonator base
  9014.95 -> 9014.74 MHz, physical readout 9015.39 -> 9014.94 MHz,
  readout length 5 -> 6 us, gain 0.33 -> 0.65, offset +0.44 -> +0.20 MHz,
  sigma 0.055 -> 0.056940825 us, and pi gain 0.6486907 -> 0.646706952.
  For Q6: f01 6522.54018 -> 6522.416914100 MHz, resonator base
  9059.45 -> 9059.26 MHz, physical readout 9059.41 -> 9058.86 MHz,
  readout length 3 -> 10 us, gain 0.60 -> 0.95, offset -0.04 -> -0.40 MHz,
  sigma 0.062 -> 0.062451228 us, and pi gain 0.6486907 -> 0.646706952.
  The standalone `pucq4_config.py` bootstrap arrays now carry the matching
  final 0 mA M5/M6 frequencies and readout values; nonzero rows remain
  centralized in `system_config.py` to prevent a second drifting table.
- The shared flux launcher now supplies both selector variables on every
  round-robin or readout-optimizer subprocess, selecting the swept line and
  holding the other calibration row at 0 mA. The qspec/Rabi/Ramsey/coherence
  map scripts inherit this path. The cross-readout Q6 diagnostic passes its
  explicit Yoko4 current directly. No hardware acquisition was run for this
  configuration-only promotion; values come from the validated manifests and
  canonical HDF5 campaign described above.

### Corrected-frequency lookup summaries (2026-09-08)

- The final sigma, T1, T2R, and T2E values for Q4/Yoko3/M5 and Q6/Yoko4/M6
  were paired row-by-row with the fresh Ramsey-corrected `f01` in
  `yoko_flux_campaign/ramsey_sigma_coherence_manifest.json`. The analysis
  validates that the selected-Rabi and coherence drive frequencies equal the
  corrected frequency before plotting.
- Eight summary plots are stored under
  `yoko_flux_campaign/ramsey_sigma_coherence_2d/`. Their bottom x axis is the
  authorized Yokogawa current grid, while the top tick at each bias is the
  exact corrected `f01` for that same row. The top labels are an explicit
  nonlinear, potentially non-monotonic lookup and are not presented as a
  linear coordinate transform. Point labels show sigma in ns and coherence
  fits in us. `frequency_axis_summary_provenance.json` records all 18 paired
  rows and the axis semantics; no raw HDF5 data were changed.
- The reusable PowerPoint now has 64 slides. Slides 61-62 show Q4 sigma/T1
  and T2R/T2E, and slides 63-64 show the corresponding Q6 summaries. The
  prior 60-slide deck is retained as
  `PUCQ4_first_light_characterization_pre_frequency_summary_20260908.pptx`.
  All four additions passed a fresh 1600x900 native PowerPoint export and
  visual inspection.

### Frequency-only sigma and coherence summaries (2026-09-08)

- A second set of eight summary plots uses the Ramsey-corrected `f01` in MHz
  as the sole x coordinate. Within each device, the nine validated rows are
  sorted strictly from lowest to highest corrected frequency before their
  sigma, T1, T2R, and T2E values are plotted. There is no current axis or
  secondary current label; the metric and frequency remain paired from the
  same original manifest row.
- The plots and `frequency_only_summary_provenance.json` are stored under
  `yoko_flux_campaign/ramsey_sigma_coherence_2d/`. Provenance records all 18
  frequency-sorted rows, confirms that each device's frequency axis is
  strictly increasing, and states that no raw data were modified.
- The reusable PowerPoint now has 68 slides. Slides 65-66 show the Q4
  frequency-only sigma/T1 and T2R/T2E plots; slides 67-68 show the matching Q6
  plots. The prior 64-slide deck is retained as
  `PUCQ4_first_light_characterization_pre_frequency_only_summary_20260908.pptx`.
  All four slides passed fresh 1600x900 native PowerPoint export and visual
  inspection, and the updated deck was reopened in PowerPoint.

## 2026-09-08 M4 coherence follow-up checkpoints (completed below)

The user authorized narrowed spectroscopy, renewed alternating SSF optimization,
Rabi sigma calibration and T1/T2R/T2E, plus existing-table updates and a new
six-qubit/Yoko-map summary deck. The following entries preserve intermediate
state; the final accepted result and promoted configuration are recorded below.

- `m4_n01` (15:49:05): gain0.01/20us,16000reps; center5594.806873MHz,
  FWHM0.541814MHz, peak population0.09331, R2.91962. Both sources0A/off.
- `m4_n02` (15:57:30): gain0.005/20us,32000reps; center5594.796624MHz,
  FWHM0.586003MHz, peak0.02389,R2.75177. Lower drive did not narrow further;
  this low-SNR diagnostic is not final narrow-line evidence. Coherence is pending.
- `m4_rc01` (16:11:21): normal RR res/Rabi/SSF;resminimum8999.91MHz,
  RabiR2.99632/pi.646707atsigma.0788us;SSF.26542at8999.76MHz/.95/5us.
- `m4_clen01..07`:2/3/4/5/6/8/10us at8999.8MHz/.95,30kshots/state;
  best initial6us,contrast.28580. `m4_cg01..15` is the ensuing6us2D readout
  grid:8999.6..9000.0MHz/.1step,gains.70/.85/.95. Subsequent shorter-length
  revalidation, final Rabi and coherence are still required.
- Existing deck now82slides;slide82documents the two lower-drive traces.
  New reporting utilities are `tools/audit_pucq4_m4.py` and
  `tools/build_pucq4_chip_summary.py`. Temp continuation files are under
  `%TEMP%/pucq4_m4_coh_20260908`, including plans, stdout/stderr, guarded
  round-robin launchers, chip_trace_sources.json and progress_slides.json.
  The original81-slide deck and pre-task shared-config copies are backed up.

### 16:45 M4 coherence checkpoint

- Readout alternation is complete: selected5us/.95 at8999.70MHz,offset-.21MHz
  frombase8999.91. Repeated5us contrasts.27256/.28280 and localgrid.29478.
  Initialgrid.31450at6us did not repeat. Existingdeck84slides,83-84documentit.
- Pilot m4_cp01/2026-09-08_16-32-33:pi.646707at78.8ns,RabiR2.99483,
  T1.879921+-.038640us(R2.93795),T2R.457470+-.036834(R2.75284),
  T2E1.205298+-.100306(R2.59520). T2results are diagnostic, below finalR2
  acceptance; exactplots/raw/sourcechecksinm4_audit.json. Actual50kstatecal
  at100us (Rabi/SSF override order), populationreconstructionpassed.
- Active finalcandidate: m4_f01,PID60296,start16:45:23,guardedonepassRR
  res_spec,q_spec,rabi,ss,t1,t2r,t2e. Resminimum8999.94MHz -> physicalRO
 8999.73MHz withselectedoffset-.21/gain.95/5us. Qspec5593-5596MHz/201points,
  gain.01/20us/32000reps. Rabi5000reps,78.8ns/pi~.65. T1.01-5us/201/6000reps;
  T2R0-3us/201/8000reps;T2E.01-5us/201/8000reps;bothT2phase2MHz.
  Allcoherence100usrelaxation,fresh50kshots/state/100us; exactplan/logs in
  %TEMP%/pucq4_m4_coh_20260908/f01_plan.json. Expected25minutes,limit32.
- RR now savesqspecSSFrawarrays and appliescoherenceSSFoverridesafterRabi/SSF.
  QICK pulseprograms untouched. Safeharness127filespassedbeforef01.
- Summarybuilderhasallotherfivezero-currenttraces and22Yoko map/configslides.
  Q6wide-map-10/-2.5mA flagsbelow4SNRareexplicitlymarked withtheirprior
  independentreview. Nativefluxpreviewpassed. Finalexistingtable/newdeck,
  acceptedM4promotionandfullartifactchecks remain pendingf01data.

### Final accepted M4 coherence and chip summary (2026-09-08)

- Final evidence is one normal `round_robin_benchmark.py` pass in
  `pucq4_first_light/m4_f01/2026-09-08_16-45-24`, index 3 only, with
  res spec, qspec, Rabi, SSF, T1, T2R and T2E enabled. All seven canonical
  HDF5 datasets and nine normal plot files persisted. Independent fits and
  raw-IQ population reconstruction passed for all five population curves;
  each has exactly 50,000 calibration shots per state.
- Measured f01 is 5594.046778406 MHz; qspec gain 0.01, 20 us pulse,
  32,000 x 1 averages and 201 points. FWHM is 0.641590 MHz, peak population
  0.06947, R2 0.90742 and independent joint-IQ SNR 11.83. Lowering gain to
  0.005 in the earlier comparison reduced signal without further narrowing.
  The preferred 0.1-0.2 MHz width and 0.3-0.5 peak-population targets were
  not reached. The final curve is resolved; do not report those targets as met.
- Rabi at sigma 78.8 ns gives pi gain 0.651681621193 and R2 0.99678.
  Alternating readout length and gain/frequency grids, including repeated
  lengths, selected the shortest measured plateau at 5 us and gain 0.95.
  Final res minimum 8999.94 MHz plus offset -0.21 MHz gives physical readout
  8999.73 MHz; final same-pass SSF contrast is 0.30808. This is repository
  histogram contrast, not assignment accuracy or a proved global optimum.
- T1 = 0.826635 +/- 0.009267 us (R2 0.99521),
  T2R = 0.479394 +/- 0.015215 us (R2 0.95616), and
  T2E = 1.702556 +/- 0.058259 us (R2 0.95306). Errors are statistical fit
  uncertainties. T1 used 6,000 averages; each T2 used 8,000, all one round,
  201 points and 100 us relaxation. T2 phase detuning was 2 MHz; actual
  Ramsey/echo sampling gives 33.90/20.07 points per independently fitted fringe.
- Shared M4-only promotion: f01 5594.38846 -> 5594.046778406 MHz;
  qspec gain 0.05 -> 0.01; pi gain 0.646706952 -> 0.651681621193;
  res base 8999.95 -> 8999.94 MHz; offset -0.15 -> -0.21 MHz;
  physical readout 8999.80 -> 8999.73 MHz. Sigma 78.8 ns, readout gain 0.95
  and length 5 us were revalidated. Other rows and RR selection were preserved.
- Exact acquired overrides, final selected settings and source readbacks are
  in `M4_coherence_20260908_settings.json`; the final run's documentation has
  `m4_guard.json`, `m4_audit.json`, `config_promotion_audit.json` and hashed
  `acquisition_source/`. `M4_coherence_handoff_20260908.md` records reuse details.
  Centers drifted across this session, so reacquire qspec before later pulse
  calibration; the small fit error is not long-term frequency stability.
- The original characterization now has 93 slides, with an updated opening
  zero-current table, progress on 82-86, final M4 results on 87 and six actual
  traces on 88-93. The new `PUCQ4_chip_summary_20260908.pptx` has 39 slides:
  six-row tables/settings, 36 actual zero-current traces, final M4 readout
  optimization, and all six requested maps for each Yoko plus sigma maps.
  Its provenance records acquisition dates, complete saved configurations,
  current rows and display processing. The Q6 -10/-2.5 mA wide-map visibility
  flags remain explicitly marked; their prior independent review is retained.
- Native PowerPoint renders were checked for table and trace readability.
  Only the original opening table/notes and newly added slides were edited;
  398 pre-existing slide/media/notes parts were checked for preservation.
  Resonator viewports crop unused panels while retaining original images,
  all data and axis labels. Saved echo images retain their legacy T2R title
  typo, explicitly explained in slide captions; future plot labels are fixed.
- Both Yokogawas finished in CURR mode at 0 A, on the 0.01 A range, outputs
  off. No current or attenuation changes were made in this follow-up.

### PowerPoint folder cleanup (2026-09-08)

The user requested removal of the duplicate-deck clutter. The run folder now
contains only `PUCQ4_chip_summary_20260908.pptx` (main report) and
`PUCQ4_first_light_characterization.pptx` (full history) at its top level.
All 29 backup/intermediate PowerPoints were moved unchanged into
`_PowerPoint_archive/`; the archive manifest maps historical paths to the new
locations and records SHA256 verification. Current summary provenance/audit
backup references were relocated accordingly. Keep future backup copies in this
archive folder, not alongside the two active decks. Raw measurement paths and
both active deck contents were preserved.

### 18:40 zero-current linewidth refresh in progress

- User means the broad displayed Q4/M4/index3 and Q6/M6/index5 traces; physical
  Q4 is M5/index4, whose archived line is already about0.11MHz. Optional
  clarification had no answer; stated the trace-based interpretation explicitly.
- M6 line_6a gain.003/20us: FWHM.202220+/-.008502MHz. line_6b gain.0015:
  .195655+/-.009207MHz; f016522.527958MHz, Rabi pi.651681621 at sigma.062451228us,
  SSFcontrast.3638. Both sources verified current0A/off. Raw/calibration audits pass.
- Active line_6c/2026-09-08_18-33-42, PID38600, full normal one-pass RR index5,
  gain.00075/20us/64000x1/101points, sigma.062612796us,50kstatecal; fresh Rabi,
  SSF and T1/T2R/T2E pending.32minute monitored limit; both sources0A/off.
  Exact plan, guard and logs in %TEMP%/pucq4_line_20260908; acquisition/config
  source snapshot in run/documentation/acquisition_source. No config promoted.
- Next acquire reduced-drive M4, refresh Rabi/sigma/SSF/coherence, then promote
  accepted zero-current rows and replace summary traces/tables. Nonzero-current
  raw maps remain archived campaign measurements. All new PPT backups in archive.
- Added request: FWHM(MHz) paired with actual current/qspec gain/pulse length in
  both Yoko tables. tools/audit_pucq4_linewidth_tables.py fits raw local and wide
  spectra separately, flags weak/coarse fits, and leaves unmeasured gain pairs
  unavailable. New tools/refresh_pucq4_summary.py is offline and has preview mode;
  validate preview and native rendering before replacing either active PPT.

### 19:25 linewidth refresh checkpoint (M6 accepted; M4 acquiring)

- line_6c is incomplete: scaled-fit default half-width1MHz exceeded actual
  .999069MHz span, produced x0 infeasible, and legacy RR wrote placeholders.
  Raw PNG only; exclude it from final tables. Fit initializer now stays inside
  existing bounds; all QICK pulse-program ASTs match pre-task versions. Early
  RR collection now preserves raw res/qspec on later fit rejection; offline
  exact-branch/temporary-HDF5 replay passed. The running M4 uses both fixes.
- M6 final line_6d/2026-09-08_18-51-46 is complete and independently accepted.
  Both sources verifiedCURR/0A/.01A/off. Res9058.88MHz;RO9058.48/.95/10us.
  f016522.511432598MHz (actual pulse setting); gain.00075/20us/64000x1/101pts;
  FWHM.211476902+/-.010673046MHz,R2.94611,peakPe.05836,IQSNR14.17.
  Sigma.062612796167us,pi.646706952329,RabiR2.99848;SSFcontrast.37234.
  T1=6.435511765+/-.077919841us;T2R=1.803285514+/-.046256470us;
  T2E=5.466466197+/-.107096618us. R2.99518/.96934/.97257, all50kcalshots
  reconstruct populations; T2fringes19.95/10.01points. All seven PNG/HDF5 present.
  M6 constants/zero-currentaliases now promoted in system/expt/pucq4_config;
  all other rows and every nonzero calibration dictionary entry unchanged.
- Active M4 line_4a/2026-09-08_19-17-11,PID60000, began19:17:09, limit45min
  (20:02:09). Full one-pass flags;bothYokos0/off, resminimum8999.90MHz gives
  RO8999.69/.95/5us. Qspec5592.85-5595.25/121,gain.005/20us/96000x1;
  sigma.079003864us,pi seed.651681621,Rabi5000x1/171; T1.01-5/201/6000x1,
  T2R0-3/201/8000x1/2MHz,T2E.01-5/201/8000x1/2MHz,all100usrelax/50kcal.
  Snapshot of actual acquisition/config sources is in each final run's
  documentation/acquisition_source, before config promotions. Do not run
  another hardware task concurrently or promote M4 before independent audit.
- TEMP/pucq4_line_20260908 holds plans/guard/stdout/stderr and report previews.
  tools/refresh_pucq4_summary.py accepts --plan and --preview-dir. Preview6
  (only M6 refreshed; M4 still old) passed native render and record checks,
  42summary/106historyslides. Newest change adds width-scan averages/N to both
  Yoko pulse tables; final preview must exercise it before saving active decks.
- Both active PowerPoints remain unchanged by this refresh until final save.
  Final plan should select M4 line_4a (if accepted) and M6 line_6d, compare
  old M4n01/n02/f01 reaudit reports plus6a/6b/6d/4a, and include6c_failure.json.
  Current-width analysis handles local and wide raw files separately and shows
  unresolved for SNR<4 rather than promoting diagnostic widths. All18current
  pairs/36source slots reviewed visually; missinglocalM6negative3 noted.
  Raw sources, exact fit seeds/windows and acquisition settings are in JSON.
- Main summary: refresh first14slides, current tables0mA, add FWHM+gain/current
  and width-scan averages/N; retain original heatmap images with date caveats.
  Append freshM6detail/SSF/drivecomparison; fullhistory openingtable and fresh
  details/traces+fourcurrent-linewidth tables. Backups/pendingPPT only in archive.
  Copy exact final plans and checks into summary artifacts/linewidth_validation;
  do not add extra active PowerPoints. Final independent deck/source/hardware
  audit, native render and harness/diff checks remain required.

### September 9 round-robin reproduction is in progress

The earlier September8 linewidth-refresh checkpoints above are historical;
those acquisitions have finished. Their final reference rows are M4 line_4b
and M6 line_6d, in the42-slide chip summary and114-slide full history. The
current user requested a new RR validation deck and verification of all six
rows and the documented bias grid, with configuration/orchestration changes
only. See docs/pucq4-round-robin-validation.md for scope, acceptance and envelopes.

A complete all-six zero-current RR saved every required plot/HDF5 in
rr6_0mA/2026-09-08_21-14-32. The independent audit accepted M2/M4/M5/M6;
M1's Ramsey model and M3's baseline readout required follow-up. Both reference
PPT SHA256 values remain unchanged. The new report is the single additional
PUCQ4_round_robin_validation_20260908.pptx, still in progress.

Current full M1/M3 repeat: rrv13_0b/2026-09-08_23-57-09, controller26812 /
RR59244, launched23:57 September8,6000 s upper limit; both sources0 A/off.
M3 candidate readout9 us/.95 gain/8974.63 MHz uses measured dip8974.84 and
-.21 MHz offset. Its fresh qspec is about5557.336 MHz/.037 MHz FWHM, Rabi
pi.666606 at sigma.124 us, SSF.5634; coherence/raw final audit remains pending.
The settings are now persistent, including local qspec±.25 MHz/201/10000x1
and20k state calibration. M1's500-rep Ramsey repeat still shows a large revival;
its9.44 us single-decay result is not accepted. Phase-detuning diagnostics are
planned next, then the16 nonzero flux-row points plus fixed-row endpoint
controls. Do not interpret a running-task candidate as completed validation.

### September 9 00:49 local validation checkpoint

The M1/M3 full chain and three follow-up Ramsey acquisitions are complete.
M3 now passes the raw/calibration/model checks with readout9 us/.95 gain,
offset-.21 MHz and actual full-chain tone8974.59 MHz. T1/T2R/T2E are
52.8797/59.8376/71.5319 us. Independent same-setting Ramsey58.3425 us
confirms the increase from prior42.8505 us; validation_M3_reviewed.json
retains the original comparison flag and confirmation evidence. M1 T1/echo
and pulse calibration pass, but Ramsey revivals persist at .25/.5/.75 MHz
phase detuning and500/1200 averages. No single-decay T2R is accepted. Restore
its normal1200 reps/.5 MHz/full60 us window; classes/programs remain unchanged.

Fresh zero-current resonance/frequency/pi values are in shared configs.
M5/M6 zero metadata use the complete all-six baseline. The16-point nonzero
bias queue is active (controller35580), starting with M5/Yoko3+2.5 mA at
rrv5_p025a/2026-09-09_00-36-49; Yoko4zero. Its line is7143.8884 MHz,
FWHM.119 MHz, pi.646707 at sigma.053267224 us. Coherence audit is pending.
The serial guard stops on findings and returns both sources0/off per point.
All nonzero config entries remain the prior measured values until audited.
The single new validation PPT is still in progress; native rendering passes.

### September 9 01:25 local checkpoint

The first nonzero M5/Y3+2.5 full chain passed line/Rabi/coherence checks but
lost readout contrast (.4706 versus prior.615). The queue stopped, both sources
returned0/off, and four alternating readout grids completed. Best tested setting:
4 us/.95 gain/9014.76 MHz, SSF.58675. Length4/5/7 us atnearby.93 gain formed
a plateau; tone9014.96 lowers contrast. A fresh res-only RR found9014.62 MHz
at.95/4 us, so the persistent offset is now+.14 MHz (previous-.48), length
5->4 us,gain.93->.95. Fresh independently checked f/pi were also propagated:
7143.888395658809 MHz/.646706952329, sigma.053267224 us unchanged.
The row is explicitly readout_candidate_pending_rr until a full-chain pass.

ACTIVE bias_queue_b controller35580 (NEW start01:21:40, recycled PID), first
rrv5_p025b/<01:21 timestamp>, Y3+2.5/Y4zero, all7normalflags and persistent
profiles,5400s upper limit. Resfound9014.68 -> actualRO9014.82 MHz from+.14;
qspec is in progress. Queueb has the original remaining15bias/control points
and stops on audit findings. No other hardware job is running. Offline131file,
81x6profiles,protectedhash/class andfocused diff checks passed before launch.

### September 9 02:44 local checkpoint

M5 / physical Q4 / Yoko3 +2.5mA repeat rrv5_p025b/2026-09-09_01-21-43
passed every normal RR check. Shared row is now validated: f7143.883429322838
MHz, FWHM.119323606640MHz, pi.646706952329, sigma.053267224us; res9014.68,
RO9014.82MHz/.95/4us/offset+.14MHz, SSF.5841; T1/R/E10.1960/7.60555/
15.5660us. Original offset lost readout contrast; four alternating SS-only
length and gain/frequency grids plus res anchor produced the accepted setting.

M6 / physical Q6 / Yoko4 +2.5mA first complete chain rrv6_p025a at01:38
finds f6841.813506590MHz/FWHM.207345MHz, pi.656656290057/sigma.053726424us;
T1/R/E7.29845/2.29244/6.34421us pass all model/comparison checks. Only the
qspec frequency shift from old6842.101132MHz and SSF.2693 vs.5753 need review.
Four completed readout grids raise SSF to.67225/.683 at22/24us with.80gain/
9059.98MHz; fine gain/frequency grid at22us ACTIVE controller11404, launched
02:42:10, rrv6p25_gb. Each point runs normal SS-only RR and returns bothsources
0/off. No other hardware job. M6 nonzero persistent row is not yet promoted.
Next: finish gain/frequency, length repeat, res anchor, complete RR repeat;
independently confirm qspec shift if it remains the only finding. Fourteen
other nonzero points and final all-six verification remain. M1 single-decay
Ramsey remains unresolved (repeatable revivals); M3 zero T2R increase was
independently confirmed and documented. One new validation PPT, references
preserved. Classes/programs remain protected and unchanged.

### September9 03:05 local checkpoint

M6/Y4+2.5 readout scans completed: selected22us/.80/9059.98MHz, repeated
contrast.65065--.67225 (one.5859 repeat retained),24us.6783--.683,shorter20us
.63187--.64115. Resanchor rrv6p25_rb found9060.08MHz. M6+2.5 persistentrow
is readout_candidate_pending_rr with this tone/length/gain/offset-.10MHz,
f6841.813506589753/pi.656656290057/sigma.053726424us.

Normal RR now preserves the calibrated absolute readout tone after eachfresh
resonator scan and records the offset from that newdip. This corrects the
observed newdip+oldoffset shifts behind M5/M6+2.5 readoutloss. Explicitbase/
offset diagnostics and pre_optimize retain their meanings; absoluteoverridewins.
No section class/QICK program changes;90 offlinecases,131fileharness,81x6profiles
andprotectedhash/ASTchecks pass. Fullhardware validation is active.

ACTIVE bias_queue_c controller57332 start02:56:51; first rrv6_p025b/2026-09-09_
02-56-53,all7normalflags,index5/Y3zero/Y4+2.5,onlyselector/substudyenv.
Alreadyfoundres9060.08,retainedRO9059.98MHz; cleanqspec6841.775497164MHz/
FWHM.205257054MHz/peakabout.4; measuredRabi pi.651681621193,SSF.657;
T1 is acquiring03:04. Normalrawresults awaitbatchsaving/audit. Queuecstops
onfindings before14remainingpoints. Do not overlap anotherhardwarejob.
Prepared m6_p025_qrep.json is NOT RUN: exactPRE-qspecseed/pi/readout repeat
if onlyold-frequencycomparison remains; do not substitute laterRabi pi.
ReferencePowerPointSHAunchanged02:59. One newreport remainsinprogress.

### September9 03:30 local checkpoint

M6/Y4+2.5 is accepted and promoted from rrv6_p025b/2026-09-09_02-56-53:
f6841.775497163998MHz,pi.651681621193,sigma.053726424us,RO9059.98/.80/22us,
res9060.08/offset-.10;SSF.6575;T1/R/E6.796211059/2.320281858/6.533448775us;
qspecFWHM.205259985+/- .002032026MHz. Independent same-settingqspec
rrv6p25_qr/2026-09-09_03-15-36 confirms6841.779011697MHz (3.514kHzagreement).
The reviewedprimaryaudit retains the old6842.101132MHz comparison and accepts
replicatedfrequencychange with no assignedcause. Bothsources0/off afterward.

ACTIVE bias_queue_d controller38236 start03:22:33,first rrv5_p050a/2026-09-09_
03-22-35 (M5/physicalQ4/Y3+5/Y4zero),all7normalflags,persistentprofiles;
14nonzeropoints+endpointcontrols remain inqueue. Foundres9014.59,retained
RO9014.11MHz;cleanqspec7122.257671006MHz/FWHM.624342MHz,peak~.46;Rabi
finishing03:30. Queue stops onfindings and returns0/off perpoint. Nooverlap.
One newPPT87slides, nativelyverified;referencePPTSHAunchanged02:59.

### September 9 09:04 local checkpoint

M5/Y3 +5 (rrv5_p050a/03:22:35) and M6/Y4 +5 (rrv6_p050a/03:40:19)
passed complete normal RR and are promoted, alongside accepted +2.5 rows.
M5 +5: f7122.257671006202 MHz, FWHM.624343669 MHz, pi.651681621193,
sigma.058777626 us, RO9014.11/.92/6us/offset-.48, SSF.5782,
T1/R/E5.174480411/1.131040093/8.360387540 us.
M6 +5: f7062.612211354989 MHz, FWHM.171174610 MHz, pi.651681621193,
sigma.053267224 us, RO9060.22/.9167/12us/offset-.56, SSF.4262,
T1/R/E7.120905648/2.452151628/7.949307822 us.

M5/Y3 +7.5 primary rrv5_p075a/03:57:51 has two comparison flags only:
f7022.356825833 MHz versus old7022.120241338 MHz, and T1 12.126387us
versus7.214108us. Independent exact-setting T1 repeat rrv5p75_t1/04:16:54
is11.044569us and passes the repeat/model gates. Independent qspec repeat
rrv5p75_qr/04:21:41 gives7022.336430607MHz/FWHM.114938592 MHz, clean
but 20.395kHz from the primary, exceeding the strict0.1-linewidth repeat gate.
Do not promote this row or execute prepared review_repeated_metrics.py yet.
Both sources returned0/off after the repeats, and no hardware ran during the
04:27--09:01 gap. A new user continue has resumed acquisition.

ACTIVE controller51424/child49148, start09:01:27: normal full-chain repeat
rrv5_p075b, M5/index4/Y3+7.5/Y4zero, unchanged shared calibration and profiles,
5400s deadline, only selector/substudy overrides. Source snapshots and guard
under run documentation; logs/plan in TEMP/pucq4_rr_verify_20260908/
m5_p075_full_b.*. No other active QICK connection (old2025/Feb2026 plot
processes remain untouched). Prepared bias_queue_e has11remaining groups,
not started. M6 -7.5 qspec +/-2MHz and -10 20000reps profiles are still
unmeasured. Final all-six default RR remains required. M1 Ramsey single-decay
model issue remains explicit; do not crop/force fits or modify classes/programs.
One new validation PPT only. Report builder now withholds unresolved metrics
and dependent calibrations from summary tables/maps; pending native rebuild.

### September 9 09:27 local checkpoint

M5/physical Q4/Y3+7.5 is accepted and promoted from complete normal RR
rrv5_p075b/2026-09-09_09-01-35: f7022.162967054066MHz,
FWHM.122318558483+/- .001824835716MHz,pi.651681621193,sigma.055104024us,
res9013.96MHz,RO9013.80MHz/.95/10us/offset-.16MHz,SSF.7133;
T1/R/E12.757370733/4.038153856/10.907325683us. Only T1 differed from prior
7.214108us; exact-setting independent repeat rrv5p75_tb/09:20:33 measured
12.337499645+/-.094050848us. Strict review_repeated_metrics.py accepted
this replicated increase with every raw/model/calibration gate retained.
Reviewed primary validation_M5_reviewed.json has plural review_decisions.
The earlier primary03:57 and qspec-only04:21 show frequency variation and
remain unpromoted; diagnostic M5_p075_frequency_variation.png/json records
all three actual scans with no assigned cause. Both sources0/off after repeats.

ACTIVE bias_queue_e controller57136, start09:26:37, first rrv6_p075a
(M6/Y4+7.5/Y3zero), all7normalflags, persistent profiles,5400s deadline.
Queue11remaining groups includes documented endpoints/fixed controls and
all negative biases. It pauses on any audit finding; returns0/off aftereach.
No other active hardware acquisition. Promoted M5+7.5 plus prioraccepted
M5/M6+2.5,+5 gives7of18 flux operating points includingzero. Finalall6
normaldefault zero run remains PREPARED NOTRUN in final_all6_default.json
(environment{},10800s). M1 Ramsey model limitation remains explicit.
81x6/protectedsource/whitespace checks passed afterlatestpromotion09:25.
Report rebuilding same sole newPPT;108configchanges tracked; originalPPTs
preserved. Do not stop with a partial handoff; continue queue/review/finalrun.

### September 9 09:47 local checkpoint

Queue_e57136 exited normally after M6/Y4+7.5 rrv6_p075a/09:26:40.
Current versioned audit validation_M6_iq.json preserves original and passes
all raw-IQ/model/calibration/coherence gates. f7185.488650128438MHz,
FWHM.129294582766MHz,R2.956205;rawSNR30.4676,residual1.1553sigma,
raw/popcenter difference.644kHz. Only flag is oldqspecfrequency7185.659470150
versusnew-.170820MHz (tolerance.146969MHz). T1/R/E6.524914838/
5.515944113/10.074508006us,pi.651681621193,sigma.057859226us,
res9061.47/RO9061.49MHz/.8334/11us/offset+.02,SSF.4289. Not promoted.

ACTIVE independent qspec-only repeat m6_p075_qrep.json controller16148,
start09:46:15, rrv6p75_qr, exact full PRE-qspecseed7185.390636570MHz,
pi.651681621,sigma.057859226us,RO9061.49/.8334/11us,+.02offset,
qgain.006/20us/401/5000x1/100us,cal10k250us,Y3zero/Y4+7.5,
1200sdeadline,both0/offafter. Primary both0/offbeforethislaunch.
Prepared queue_f has10remaininggroups, NOTSTARTED, will load strengthened
qspec audit. After repeat: currentaudit, strictreview_repeated_metrics.py
primary validation_M6_iq.json + qspec_ge=confirmation if repeat agrees;
then promote/check/queue_f. If not, preserve findings and investigate.
Audit CLI now supports --audit-suffix to retain previous audits and removes
same-run approval only for new unresolved findings. JSON-fixturecheck and
131-fileharness/whitespace pass. Original resolved-comparison reviews retained.
M5+7.5 is already accepted/promoted; finalall6default stillNOTRUN. PPT91
lastbuilt09:27,taskongoing. No class/programchanges oroverlappinghardware.

### September9 10:16 local checkpoint

M6/Y4+7.5 accepted and promoted from rrv6_p075b/2026-09-09_09-54-43:
f7185.513939668361MHz,FWHM.136510718884+/-.002942536122MHz,
pi.646706952329,sigma.057859226us,res9061.47/RO9061.49/.8334/11us/
offset+.02,SSF.433,T1/R/E5.451795273/5.723199056/9.238789593us.
All raw/cal/model/reference gates PASS; preceding two scans preserved in
M6_p075_frequency_variation.png/json and the same new PowerPoint. No fixed
shift waiver was made; fresh full frequency is within old-reference linewidth.
Allprofile/protectedsource/whitespacechecks passafterpromotion10:13:28.
Bothsources0/offbetweenpoints. Eightof18fluxpointsincludingzero accepted.

ACTIVEqueue_fcontroller30964 (start09:54:38), now rrv5_p100a/
2026-09-09_10-12-32, indices0,1,4/M1,M2,M5, Y3+10/Y4zero,all7normalflags,
persistentprofiles,7200sdeadline. M1qspecactive10:15. Newauditorwithraw-IQ
gatesisloaded;queuepausesonfindings. M1Ramseymodelmayagainfail atthiscontrol
point; preserveitsflag whilecheckingvalidM5/M2results andcontinuingotherwork.
Finalall6defaultzero run remainsNOTRUN (planfinal_all6_default.json,env{},
10800s). NewPPT93slides/121configchanges,3knownM1zeroissues,native8slides
exported10:15 andusageviewed; referencesunchanged. Onlycomments/PPTusage
nowexplicitlystatejointnonzeroYokobiaspairsunverified(oneYokoatatimecalibration).
Noexperimentclass/QICKprogramedits. Continue remaining10bias/controlgroups.

### September 9 10:52 local - +10 mA review and exact-setting confirmations

The all-seven-experiment M1/M2/M5 control group rrv5_p100a/2026-09-09_10-12-32 completed normally in 2182.95 s; all canonical raw data and plots persisted and both Yokogawas returned to 0 A/output off. M1 repeats the known structured Ramsey revivals (single-model 7.9329 us is withheld), while T1 31.4180 us and echo 32.0112 us match the original comparison. M2 has clean longer T1 12.05570 +/-0.42916 us and Ramsey 15.60720 +/-0.66088 us, versus 8.76716/11.70821 us in the reference. Echo20.45527us passes its comparison. These are biased controls, not zero-current calibration replacements.

M5/physical Q4 at Yoko3 +10 mA has center6841.583959579MHz, FWHM0.160779538MHz, sigma0.056481625us, measured pi0.651681621193, res9013.07MHz, RO9012.97MHz/.95/6us/offset-.10MHz, SSF.6415. T1/R/E14.718917807/2.435159840/8.426011469us all pass. Only qspec old-reference shift+0.383215MHz is flagged; this row is not promoted yet.

Queue_f exited and paused correctly. ACTIVE m2_p100_trrep.json controller57472 (10:50:50), normal RR flags t1,t2r, M2/index1 at Y3+10/Y4zero, f5438.719629589458MHz/pi.661630958921/sigma.101us, RO8951.21MHz/.56/5us/offset-.46MHz. T1 .01-60us/121; Ramsey0-40us/201/.5MHz; both1200x1/250us, statecal10k/250us, 900s deadline, .5mA/s ramps, normal raw hierarchy, both0/offafter. Exact repeat plan m5_p100_qrep.json is prepared, NOT STARTED: M5 qspec at same bias with PRE-qspec seed6841.22106308MHz/pi.646706952/sigma.056481625us, same RO, qgain.002/20us/401/5000x1/100us,1200s deadline. Nine remaining groups are prepared in bias_queue_g.json, NOT STARTED. Do not overlap hardware.

Offline review helper now supports multiple comparison-only changes confirmed by a single independent run and fixed controls at their actual two-source biases. It never waives raw/model/quality findings. Isolated JSON fixture passes (10:51:56), verifies two decisions, preserved originals/history, no zero/flux promotion and model rejection. Report builder includes unresolved control traces and actual-bias confirmed-change rows. Final all-six normal default zero-current run remains required and not started.

### September9 10:59 local continuation pointer
M2 endpoint control was reviewed using independent rrv2p10_tr/10:50:52: T1 repeat10.620644us returns within reference band and agrees with primary12.05570us (recorded as variation, no persistent increase); Ramsey15.705896us confirms primary15.60720us. Selected validation_M2_reviewed.json is only a Y3+10/Y4zero control, no zero/shared calibration promotion. M1 remains model-limited. Sole active hardware m5_p100_qrep controller56736, qspec-only exact-setting M5+10mA repeat; ~75% at10:58:36. Queue_g has remaining9groups and is NOT STARTED. Same newPPT97slides built10:57, native10slides exported10:58; comparison table58 and controlT1trace59 visually checked. execution/checkpoint_1058 archives current helper/report/audit versions and plans. Final all6default zero run still required, not run.

### September9 11:03 local - Q4 +10 mA frequency variation and settling check

Independent qspec rrv5p10_qr/2026-09-09_10-54-46 completed with both sources0/off, canonical raw/plot and raw/cal/model gates passing. Center6841.457614725295MHz,FWHM.148709094912551MHz,R2.973385435. It differs0.126344854MHz from primary6841.583959579MHz, so it does NOT confirm a fixed shift under the unchanged0.1-linewidth frequency-repeat criterion. Neither +10mA acquisition is promoted. The primary scan followed ~20min of fixed-row controls at bias, whereas this scan began soon after the ramp. This timing difference is observed; it does not establish a physical cause.

Configuration-only next check: round_robin_benchmark.py now applies a fixed600s post-ramp settling interval for Yoko3 +10mA (old0s -> new600s) as already used for Yoko3 -10mA; all other bias waits unchanged. It records yoko_settle_seconds in acquisition_profile.json and emits30s progress. No experiment/pulse/readout/gain/driver classes changed. Offline131-file harness and81x6profile/protected-source checks PASS11:02; focusedwhitespacePASS.

ACTIVE m5_p100_fullb.json controller21624 start11:02:52, all7normalRRflags M5/index4/physicalQ4 atY3+10/Y4zero,600ssettle, persistent oldseed6841.22106308MHz and +/-1MHz qspec window (both observed peaks inside); .002gain/20us/401/5000x1/100us. Sigma.056481625us, RO9012.97MHz/.95/6us, normal Rabi0-.85/171/1200x2 and nonzero coherence1200x2/250us, statecal10k250us.5400sdeadline,.5mA/sramps,both0/offafter. Expected~28minincludingwait. Review raw/coherence and frequency variation before any promotion. Remaining9groups inqueue_g are NOTSTARTED; finalall6default zero run remains required.

### September9 11:22 local - settled M5 +10 mA run in progress

Active m5_p100_fullb controller21624/child2968, ROOT/rrv5_p100b/2026-09-09_11-02-54. Recorded600s settling finished; acquisition_profile confirms all7flags/index4/Y3+10/Y4zero/maxpath196. Res plot9013.11MHz, preserved RO9012.97MHz(offset-.14); qspec normal plot6841.484578285427MHz/FWHM.150449104305763MHz, peak~.27 and visibly resolved. Fresh Rabi pi.6516816211926998 at sigma.056481625us; SSF.6431. Res/qspec/Rabi/SSF plots visually inspected. T1/R/E in progress; no full audit or promotion yet.

Prepared exact-settling qspec confirmation m5_p100_qrepb.json (substudyrrv5p10_qs), NOT STARTED.600ssettling/oldPREseed6841.22106308MHz/pi.646706952/sameRO/offset-.14/sameqspecsettings,1800sdeadline. Only launch after full run/audit if frequency is the sole comparison finding. Current9-groupqueue_g remains NOT STARTED; finalall6default stillNOTRUN. Frequency-history helper now accepts M5p100 and references first full10:12, prompt repeat10:54, settled full11:02; do not run it before the settled full audit exists. Review helper compares explicit settling settings (fallback oldY3-10=600/others0) and refuses mismatches. Auditor now carries settling metadata; source/profile/harness checks pass, latest configdiff122 entries. Same newPPT97slides lastbuilt10:57; future build includes newwait instructions. No new PowerPoint file or experiment-class edits.

### September9 11:32 local - settled Q4 +10 mA full pass completed

rrv5_p100b/2026-09-09_11-02-54 completed normally (1616.90s including600s settling); both Yokogawas0A/off verified after. All7 normal plots, canonicalHDF5 and statecal raw persisted. Audit all raw/cal/model/Rabi/SSF/coherence gates PASS; sole issue remains oldqspecfrequency comparison. Actual pulse6841.484578285427MHz; independent center6841.484575238515MHz; FWHM.150452718577631+/- .002470953882490MHz; R2.9740840938; peakpopulation.265409. Pi.651681621193/sigma.056481625us, res9013.11MHz/RO9012.97MHz/.95/6us/offset-.140000000001MHz, SSF.6431. T1/R/E15.426027649/2.438365184/8.333998671us all match previous bands and the earlier full pass. Every normal plot visually inspected. This row is NOT PROMOTED pending frequency repeat.

ACTIVE exact-setting qspec confirmation m5_p100_qrepb.json controller41236 start11:31:57, sub_studyrrv5p10_qs, M5/index4/Y3+10/Y4zero,600s settling, PREseed6841.22106308MHz/pi.646706952/sigma.056481625us,RO9012.97MHz/.95/6us/exactoffset-.1400000000012369,qgain.002/20us/401/5000x1/100us,statecal10k250us,1800sdeadline,.5mA/s ramps, both0/offafter. Expected~15min. After completion: normal audit; strict review_repeated_metrics.py primary settled validation_M5.json + qspec_ge=confirmation (requires matching settling and center agreement0.1linewidth/3SE); if accepted promote_bias_config preview/apply then profile/protected/whitespace checks and start prepared queue_g nine remaininggroups. If repeat fails, retain diagnostic history and investigate; do not waive fit quality or force reference center. Finalall6defaultzero remainsNOTRUN.

### September9 11:57 local - Q4 +10mA tracking validated and promoted; queue resumed

Matching600s qspec repeat rrv5p10_qs/2026-09-09_11-32-03 completed normally with both0/off and all raw/cal/fit gates passing. Center6841.512308139529MHz/FWHM.155251292259240MHz/R2.9724637055. Its27.732901kHz difference from the complete settled pass exceeds the unchanged15.045272kHz fixed-shift limit. Static shift is NOT confirmed. The predeclared11:44 separate tracking criteria passed in review_m5_p100_tracking.py preview and apply: four resolved scans span6841.457614725295-6841.583959579264MHz (126.344854kHz, less than the smallest measured linewidth); matched-settling widths agree; two complete RR sequences have matching sigma/readout/averaging/axes/statecal and reproduce pi/T1/T2R/T2E under the unchanged repeat tolerances, with each newly measured qspec center propagated to all pulse experiments. No general audit or quality gate was relaxed. All earlier audits/raw retained.

Selected settled full validation_M5_tracking_reviewed.json has statusvalidated_tracking, explicit static_shift_confirmed=false and requires_qspec_each_pass=true. Promoted11:52:56 only that complete run's measured values: f6841.22106308 ->6841.484578285427MHz; pi.646706952 ->.651681621193; res9013.09 ->9013.11MHz; offset-.12 ->-.14MHz; T1/R/E15.896839/2.139733/7.979470 ->15.426027649/2.438365184/8.333998671us. Sigma.056481625us/qspecgain.002/20us/RO9012.97MHz/.95/6us unchanged. FWHM.150452718578+/- .002470953882MHz,SSF.6431,counts and provenance saved, plus observedfrequencymin/max andtrackingrequirement. The config stores the full-pass carrier, never the later qspec-only center.

Normal RR now rejects pulse-only acquisition at a tracking-required selected row unless qspec is enabled or a freshly measured carrier is explicitly supplied for diagnostics. This guard is before hardware/folder actions. Parsed guard cases,81x6profiles/timing/protectedhashes and131-fileharness PASS11:53; focusedwhitespacePASS. No underlyingclass/QICKprogramchanges. Nine of18fluxoperatingpoints have reviewed complete RR evidence, including this explicitly variable-frequency case.

ACTIVE queue_g controller60776 start11:54:35, first rrv6_p100a/2026-09-09_11-54-43, M6/Q6/index5,Y3zero/Y4+10, all7normalflags,5400sdeadline; qspec~23% at11:56:40. Remaining8negative-currentgroups follow if audits pass. First point exact envelope above. Queue pauses on any finding and returns0/offaftereach. M1 zero/control Ramsey remains model-limited. Finalall6normaldefaultzero run is still PREPARED NOTRUN and required. Same single newPPT is rebuilding with four Q4+10 spectra and dedicated tracking explanation;138configchanges recorded. No extra rootPowerPoint created.

### September9 12:03 local checkpoint

Queue_g60776 remains active on rrv6_p100a/2026-09-09_11-54-43, child4476, M6/Q6 atY4+10/Y3zero. Normal plot f7211.421318376494MHz/FWHM.170435763376556MHz, peak~.4; res9061.52MHz/RO9061.52MHz(expectedoffset0); Rabi pi.6417322834645669 matches storedseed; sigma.049593622us,SSF~.424. Qspec and Rabi plots visually inspected; T1/R/E underway, not yet audited/promoted. Prior T1/R/E6.303632/4.448974/9.648699us,SSF.471. Expect finish~12:12; then queue automatically proceeds M5-2.5 ifallgatespass. Before promoting a completed point during an advancing queue, verify the next child has written acquisition_profile.json (imports complete), or that the queue has stopped; avoid changing configs during source snapshot/import.

Q4+10tracking promotion complete11:52:56 (see above). Same sole newPPT100slides rebuilt12:01,138configurationchanges,5remainingM1zero/controlmodel/comparisonissues; native12slidesexported12:02. Trackingexplanation59 and four-scanhistory68 visuallychecked; settingslabel now (tracking), not (review). Finalall6default stillNOTRUN. Helpers/source/plans archived execution/checkpoint_1159; taskbriefupdatedthere, future finalarchive stillneeded. No hardware overlap, no class/QICKprogramedits. Continue9remainingfluxpointsincludingactive, remainingendpointcontrols, finalnormalzero, report/verification.

### September9 12:16 local - Q6 +10mA Ramsey model review
Queue_g60776 paused/exited after rrv6_p100a/11:54:43; all raw saved and bothsources0/off. All numeric reference comparisons passed, including qspec7211.421316779MHz/FWHM.170419458622MHz/R2.977226, pi.641732283465, SSF.4235211761, T1/R/E6.898007767/4.687802144/10.148035373us. Only issue is t2_ge structured residuals: R2.879455712, residual peakpowerfraction.282693610 (> .25), residual RMS/fitspan.042696249 (> .03). Ramseyfringe.521079816MHz,19.329points/fringe, independent fit4.687778us. This is a quality finding and cannot use the comparison/tracking reviewers. Q6+10 is NOTPROMOTED.
ACTIVE m6_p100_r75.json controller60748 start12:15:16, normal RR Ramsey-onlyM6/index5/Y4+10/Y3zero; changes onlyphase detuning .5->.75MHz, retaining1200x2/0-30us301points/250usrelax/statecal10k250us; exactcarrier7211.421318376494MHz/pi.6417322834645669/sigma.049593622us/RO9061.52MHz/.8722/12us/offset0.900sdeadline,.5mA/sramps,both0/offafter. QICKclasses unchanged. Review curve/residuals before any persistent phase change; if clean, require a fresh complete RR with that profile. Remaining8negativegroups prepared in bias_queue_h.json, NOTSTARTED. Finalall6default stillNOTRUN. M1model limit and Q4+10validated_tracking remain explicit. Continue task, no partial handoff.

### September 9, 12:30 local — Q6 +10 mA phase diagnostic and full repeat

The normal Ramsey-only diagnostic rrv6p10_r75/2026-09-09_12-15-18 completed with canonical raw data, its population plot, and both sources verified at 0 A/output off. Phase detuning 0.5 -> 0.75 MHz was the only acquisition change. T2R = 5.939635633 +/- 0.217927309 us, R2 = 0.910275, residual peak fraction = 0.189874, residual RMS/fit span = 0.0381381, 13.115 actual samples/fringe. All raw/calibration/model gates pass. Its sole finding is a 33.51% change from historical T2R; a diagnostic-only result is not promoted.

Persisted Ramsey phase 0.75 MHz only for M6/index 5/Yoko 4 +10 mA in expt_config.py; echo remains 0.5 MHz. Offline 131-file harness, 81 x 6 profile/timing/protected-source checks and focused whitespace check passed at 12:29. Configuration change recorded explicitly in MHz.

ACTIVE m6_p100_fullb.json, controller 20808 / child 27688, started 12:29:41. All seven normal RR flags at Yoko 3 zero / Yoko 4 +10 mA, persistent settings, sub-study rrv6_p100b. Qspec seed 7211.22085715 MHz +/-1 MHz, 401 points, gain 0.006, length 20 us, 5000 x 1 / 100 us; Rabi 0-0.85/171/1200 x 2, sigma 0.049593622 us; RO 9061.52 MHz / 0.8722 / 12 us; T1 0.01-60 us/241, Ramsey 0-30 us/301/0.75 MHz, echo 0.01-60 us/301/0.5 MHz, 1200 x 2 / 250 us; state calibration 10k/state / 250 us. Nominal attenuation arguments unchanged, 0.5 mA/s ramps, no extra Yoko 4 settling, paths below 210 characters, 5400 s deadline, abort on acquisition/save/readback failure, both sources zero/off afterward.

Review the complete raw/calibration/fit/reference evidence before promotion. If only a historical T2R comparison remains, obtain a fresh exact-carrier independent repeat and use the unchanged repeat criteria. Remaining eight negative-current groups in bias_queue_h.json and the final all-six default zero-current pass remain unstarted. No underlying classes or QICK programs were changed.

### September 9, 12:52 local — Q6 full-repeat outcome and midpoint-frame diagnostic

rrv6_p100b/2026-09-09_12-29-43 completed with all canonical raw/plots and both sources 0 A/output off. Qspec center 7211.432304469609 MHz, actual pulse 7211.432304725785 MHz, FWHM 0.176175419422350 MHz, R2 0.977050; pi 0.641732283465, sigma 0.049593622 us; res 9061.54 MHz, RO 9061.52 MHz / 0.8722 / 12 us, measured offset -0.02 MHz; SSF about 0.4557. T1/T2R/T2E = 6.996602547 / 5.413374422 / 9.870541226 us. All historical comparisons pass, but Ramsey still has structured residuals (R2 0.898434, peak fraction 0.279817, residual RMS/fit span 0.0395261). The 0.75 MHz phase change did not reliably solve this quality finding. No Q6 +10 calibration was promoted, and no audit gate was relaxed.

Independent exploratory full-window analysis, M6_p100b_phase075_residual.json/png, resolves two fitted components 0.700545544 and 0.779255431 MHz (separation 0.078709887 MHz). Shared-envelope exploratory tau 7.514356 us, R2 0.960589, no parameter bound hit; the saved canonical single fit remains unchanged. These components do not establish a physical cause.

ACTIVE m6_p100_mid.json, controller 58892, started 12:51:56: normal RR Ramsey-only at Y3 zero/Y4 +10 mA, same fresh carrier/pi/sigma/readout and full 0-30 us/301/1200 x 2/250 us/statecal10k250us as the failed complete pass. Only programmed phase detuning changes 0.75 -> 0.015398310214337 MHz. This approximate midpoint was derived from fitted mean 0.739900487 MHz and recorded delay step 0.09928385 us vs requested 0.1 us: 0.75 - 0.739900487 * 0.09928385 / 0.1. It is a diagnostic phase choice, not a qubit-frequency correction or calibration promotion. Deadline 900 s, nominal attenuation arguments unchanged, paths <210 characters, 0.5 mA/s ramps, both sources zero/off afterward. Inspect the complete trace, identifiability and raw/model gates before considering any persistent change. Eight negative groups and final all-six zero remain unstarted.

### September 9, 12:59 local — midpoint diagnostic passes; normal full sequence launched

rrv6p10_mid/2026-09-09_12-52-03 completed with canonical Ramsey HDF5 and plot and both sources 0 A/output off. All normal raw/calibration/model/reference gates PASS. T2R = 5.707115956 +/- 0.616388545 us (28.28% above reference, within the predefined comparison band), R2 = 0.971122906, residual peak power fraction = 0.0346299, residual RMS/fit span = 0.0377492. Independent fit tau = 5.707155 us; fringe = 0.040154162 MHz, 250.84 actual points/fringe. The unchanged fitter describes the full acquired 29.785 us window; nothing was cropped or replaced. Diagnostic-only results are not promoted.

Persisted M6/index 5/Yoko 4 +10 mA Ramsey phase 0.75 -> 0.015398310214337 MHz in expt_config.py. Task-start value was 0.5 MHz; echo remains 0.5 MHz and all other current/row profiles are unchanged. Offline 131-file harness, 81 x 6 profiles, timing, protected source/class and isolated bias/guard checks PASS at 12:58; focused whitespace PASS.

ACTIVE bias_queue_i.json controller 18724, launched 12:58:28. First point rrv6_p100c is a fresh all-seven normal RR at Y3 zero/Y4 +10 mA, persistent profile and stored old qspec seed, with no diagnostic pulse/readout/phase overrides. Envelope matches m6_p100_fullb except the newly persisted Ramsey phase; 5400 s deadline, 0.5 mA/s ramps, both sources zero/off after. The queue then contains all eight negative-current groups from queue_h, including M3/M4 controls at Y4 -10 mA; their exact configuration/axis/count/settling envelope is archived in negative_bias_preflight.json. Any audit issue stops the queue for review. Before promoting a row while a later point starts, verify its acquisition_profile.json has been written to avoid editing configs during source snapshot/import.

M1 offline midpoint estimates from three same-carrier records (5517.691628432926 MHz) are phase 0.00704713, 0.00630361 and 0.00579702 MHz after actual/requested delay-step correction; exploratory splitting 0.01715-0.01830 MHz. These are untested diagnostic estimates, not calibration. Consider a fresh-carrier M1 midpoint diagnostic after the current queue if Q6 full validation supports this approach. Final all-six default zero pass remains required and unstarted. Same single new validation PowerPoint; no underlying experiment/class/QICK changes.

### September 9, 13:23 local — existing fitter initialization connected in RR

The midpoint full pass rrv6_p100c/2026-09-09_12-58-36 completed with all raw/plots and both sources zero/off; queue_i paused. Qspec center 7211.424143213345 MHz (actual pulse 7211.424142063969 MHz), FWHM 0.186796710799236 MHz; pi 0.641732283465/sigma 0.049593622 us, res 9061.50 MHz/RO 9061.52 MHz/.8722/12 us/offset +.02; SSF about .4459. T1 7.069538316 us and echo 9.252036836 us pass comparison/quality. The normal automatic Ramsey fit missed the visible slow fringe: 2.655297172 +/- .121736143 us, R2 .881535876, while the independent same-model fit gave 5.768092278 +/- .589012574 us, R2 .979532014 and .037137599 MHz. It therefore failed comparison, residual and fit-agreement checks. This run is not promoted.

The unchanged public T2RMeasurement.t2_fit already supports guess={'f': ...}. Offline AST-extracted calls on frozen HDF5 with only f=.04 MHz recover the independent solution: tau 5.768146363 +/- .588979845 us, R2 .979532014, fitted signed frequency -.037137638 MHz, residual peak fraction .02972326. On the prior clean midpoint diagnostic the same input reproduces 5.707386495 +/- .616515276 us and R2 .971122906. Frequency and tau remain free; no decay-time guess, constraint, model change, cropping or raw rewrite was applied. Both original raw/automatic fits remain preserved. See M6_p100c_fit_frequency_seed.json and M6_midpoint_fit_frequency_seed.json.

Localized configuration/orchestration change: Ramsey profiles default fit_frequency_guess_MHz=None, with .04 only for M6/Y4+10. RR calls the existing fit and plot methods with that frequency input after acquisition, retains the original automatic curve as additional HDF5 field Fit Unseeded, and records both original tau/error and configured/free fitted frequencies in Syst Config. It suppresses only the intermediate automatic plot for configured fits so the normal class saves the final configured curve once. Underlying classes/QICK programs and Data_H5 implementation are unchanged. Auditor verifies the original fit is preserved and finite, records the initialization metadata, and keeps every prior raw/model/comparison gate. PPT tables show phase to six decimals and add configured-fit settings where actually measured. Repeat reviews now require matching fit initialization as well as phase/carrier/axes/counts.

Meaningful offline integration check check_rr_ramsey_guess.py executes the actual RR block and unchanged fitter on both measured fixtures. Eight cases cover configured/default/no-fit/save-figures-off behavior, preservation of raw/automatic arrays, free fitted frequency, normal plotting arguments and agreement with independent fits. PASS13:21:44, alongside131-file harness,81x6 routing/reset/timing, protected source/class hashes and isolated bias/guard checks; focused whitespace passes.

ACTIVE bias_queue_j.json controller28644, start13:22:25, first rrv6_p100d atY3zero/Y4+10. All7normalflags, persistent midpoint phase and .04 MHz fit initialization, other acquisition/pulse/readout settings unchanged,5400sdeadline,.5mA/sramps,both0/offafter. Then eight unstarted negative groups, stop on any audit finding. Do not promote offline-refitted prior traces; require this fresh complete pass, normal plot, canonical HDF5 including original fit and current readbacks. If queue advances before promotion, wait until next acquisition_profile.json is present before editing shared config. m1_midpoint_prepare.json remains prepared/unstarted; final all-six default zero also unstarted. Continue the complete authorized task.

### September 9, 13:44 local — frequency-only fit initialization was insufficient

rrv6_p100d/2026-09-09_13-22-34 completed; both sources 0 A/output off, queue_j stopped for review. Canonical HDF5 successfully preserved Fit Unseeded and all new fit-initialization metadata. Qspec center 7211.423394881848 MHz / FWHM .181330580368608 MHz; actual pulse 7211.4233942938945 MHz, pi .641732283465, sigma .049593622 us, res9061.54/RO9061.52MHz/.8722/12us/offset-.02, SSF about.4436. T1 7.008677137 us and echo9.075294012 us pass reference/quality. Frequency-start-only Ramsey converged to the wrong solution again:1.486253454+/-.722027088us,R2.900749642; original automatic1.486744889us. Independent same-model tau6.474332939+/-.653304264us,R2.977743887,f.037583488MHz. Four Ramsey findings remain; no Q6+10 result is promoted.

The unchanged public fitter with calibrated-population starting values f=.04MHz, phase=0rad, amplitude=1, initial_offset=.5, final_offset=.5 converges freely on this raw trace to6.474494242+/-.653190264us,R2.977743887,residualpeak.04879798,matching the independent fit. It also reproduces5.768131us on the prior full c trace and5.707438us on the clean midpoint diagnostic. No T2 guess or constraint is supplied; class/model/raw data remain unchanged. These are offline diagnoses, not accepted measurement substitutions.

Profile now has fit_population_guess=None everywhere except M6/Y4+10, where it stores the four population starting values above. RR merges them into the existing API guess only for calibrated-population fitting, records fit_starting_values in Syst Config, and preserves Fit Unseeded. Auditor reads/validates the full starting-value dictionary (explicitly no T2 key), and repeat review requires all starting values to match. Report tables include phase/amplitude/offset starts where actually measured. Underlying classes/QICK programs remain untouched. The actual RR-block integration test now covers THREE measured fixtures /12 configured/default/no-fit/no-figure cases, including preserving the true automatic Fit Unseeded for the d fixture. PASS13:43:25, together with131-file harness,81x6 routing/fit-option reset/timing/protected-hash/class checks and whitespace.

ACTIVE m6_p100_fitpop.json, controller32408, launched13:44:09. Ramsey-only normalRR atY3zero/Y4+10, exact carrier7211.4233942938945MHz/pi.6417322834645669/sigma.049593622us/RO9061.52MHz/.8722/12us/offset-.02 from full d. Full0-30us/301/1200x2/250us, statecal10k250us, persistent phase.015398310214337MHz and the new full population starting values. No acquisition-window change.900sdeadline,.5mA/sramps,unchangednominalattenuationargs,canonicalpaths<210,both0/offafter. Audit after completion. If raw/model fits are now reliable, require another fresh complete M6+10 RR before promotion; a diagnostic alone cannot replace complete-chain evidence. Remaining8negativegroups are still unstarted in bias_queue_h.json. M1 fresh-carrier preparation m1_midpoint_prepare.json and final_all6_default.json remain prepared/unstarted. Continue all remaining hardware/report work; do not hand off a partial completion.

### September 9, 14:03 local — Q6 fit stability separated from measurement variation

The 13:44 Ramsey-only run rrv6p10_fit/2026-09-09_13-44-24 completed with HDF5/normal plot and both sources zero/off. Its configured population-start fit gives T2R 2.804590154 +/- .260938248 us, R2 .968238283; independent same-model fit 2.917855229 +/- .306061148 us, R2 .968218866. Raw, calibration, fit agreement and residual gates pass; only the historical coherence comparison fails (-36.96%). The original automatic fit returned -9215.596943 us and is retained as Fit Unseeded. This is a shorter measured repeat, not another failure of the configured fit. No diagnostic-only promotion.

Offline check_ramsey_multistart.py applied the UNCHANGED public fitter to four full recorded traces, with five frequency starts .01/.02/.04/.06/.10 MHz and automatic or physical 1/3/5/10/20 us decay starts. Each trace's near-minimum solutions agree to <.001 us: midpoint 5.707 us, full c 5.768 us, full d 6.474 us, latest diagnostic 2.805 us. Thus changing decay-time initialization does not restore the historical value. Existing T2 API normalization requires physical_tau_start_us / dot(recorded_delays, recorded_delays); this was used only in offline sensitivity analysis. NO T2 guess was added to any persistent configuration, no class/model/window was changed, and no raw or canonical result was rewritten. Evidence: artifacts/M6_low_phase_multistart.json and M6_p100_fitpop_tau5_start.json/png.

ACTIVE bias_queue_k.json controller59096, launched13:59:47, child26216. First rrv6_p100e uses all seven normal experiments and persistent population-start configuration at Y3zero/Y4+10, with fresh qspec and Rabi before coherences. Exact envelope is in the plan; 5400s deadline, .5mA/s ramps, zero/off afterward. It then queues eight unstarted negative groups, stopping on any audit issue. M1 fresh-carrier preparation and final all-six default zero remain unstarted. No Q6+10 full pass has yet been promoted. Continue the complete hardware/report task.

### September 9, 14:20 local — complete Q6 +10 pass has only a coherence comparison finding

rrv6_p100e/2026-09-09_13-59-51 completed in1045s with all7canonical HDF5 files (longest196characters), normal calibrated plots, source snapshots, and both Yokos0A/outputoff. All raw/calibration/model/fit-agreement gates pass. Qspec center7211.413079784586MHz, actualpulse7211.413079440341MHz, FWHM.178059357156368+/-.002853806284889MHz,R2.975415624,rawIQSNR32.258; qgain.006/20us. Rabi pi.641732283464567 at sigma.049593622us; RO9061.52MHz/.8722/12us, measuredres9061.54/offset-.02;SSFabout.436. T1 6.679829637+/-.084667277us and echo9.700775749+/-.264197044us match the prior values. Configured Ramsey6.114422315+/-.492541864us,R2.975372131, independent6.114422622us, fringe.042598020MHz/236.45points, residualpeak.03518385. Automatic1.393429993us retained in Fit Unseeded. Sole finding: T2R+37.43%fromold4.448973504us, just outside the original comparison band. No promotion yet.

Queue_k stopped for this comparison finding. ACTIVE exact-setting Ramsey confirmation m6_p100_erepeat.json, controller11156, child22796, launched14:19:09; labelrrv6p10_er. Uses the complete e run's exact carrier/pi/sigma/readout, persistent phase.015398310214337MHz,301points/0-30us/1200x2/250us,statecal10k250, same f.04/phase0/amp1/offset.5/.5 free-fit starts, no T2 guess.900sdeadline,.5mA/sramp,0/offafter,normalHDF5/plot. Predeclared review: all quality gates and exact settings must match, repeat within15%mean or3combinedstandarderrors; preserve originalcomparison. If repeat fails, retain Q6+10 under review and proceed with the independent negative-current coverage before revisiting it. Eightnegativegroups, M1 fresh-carrier diagnostic and finalall6zero remainunstarted. Do not repeat full Q6 passes indefinitely at the expense of unmeasured bias coverage.

### September 9, 14:26 local — Q6 +10 validated; negative-current acquisition started

Exact repeat rrv6p10_er/2026-09-09_14-19-13 completed, all raw/calibration/model/reference gates PASS and both sources0A/off. T2R5.700455215+/-.468902458us versus complete e pass6.114422315+/-.492541864us. The .414us difference is inside predeclared2.040147795us repeat tolerance; all carrier/pi/sigma/readout/phase/axes/counts/statecal/fit-start settings match. Reviewed full audit validation_M6_reviewed.json is selected as 'consistent repeat; reference band overlap': repeat itself is inside the historical band, so NO persistent physical increase is claimed. Both actual normal RR plots are added together to the one validation PowerPoint; prior short2.805us diagnostic and failed automatic fits remain visible.

Promoted system_config.py Yoko4_Q6_M6/+10 row at14:25:11: frequency7211.220857150->7211.413079440341MHz; resbase9061.68->9061.54MHz, offset-.16->-.02000000000044MHz (absoluteRO9061.52MHz unchanged); pi stores exact measured.641732283464567; sigma.049593622us, qspecgain.006/20us,RO.8722/12us retained. Complete-pass T1/T2R/T2E now6.679829637/6.114422315/9.700775749us; FWHM.178059357+/-.002853806MHz,SSF.4358,counts/provenance persisted. Ramsey configuredphase.015398310214337MHz andf.04/populationstarts stay in expt_config.py. No classes/QICK modified.131-fileharness,81x6profiles/timing/protectedclasshashes and whitespacePASS14:25:15.

ACTIVE bias_queue_h.json controller24124, launched14:25:33. Eight documented negative groups now start with M5/index4/physicalQ4 atY3-2.5mA,Y4zero,labelrrv5_m025a. All7normalflags,persistentconfiguration, exactnegative_bias_preflight.json envelope,.5mA/sramps,zero/offafter,5400sdeadline exceptM3/M4/M6endpoint10800. Each raw audit stops the queue on any issue. AtY3-10 wait600s; no joint nonzero biases. Before changing source configs on a passed point, ensure the next child's acquisition_profile.json exists. Remaining after negativegrid: M1 fresh-carrier/midpoint investigation and finalall6defaultzero pass, finalPPT/checks. All positive flux rows are now selected and promoted.

### September 9, 14:44 local — first negative-current full pass; exact qspec confirmation active

rrv5_m025a/2026-09-09_14-25-37 completed in989.34s, all7canonicalHDF5/normalplots saved, sourcesboth0A/off. All raw/calibration/model gates PASS. Sole finding: qspecfrequency shifted from documented6945.111955770774MHz (storedpulse6945.029248480) to center6944.563244982731+/-.000704816MHz; actualpulse6944.563244921842MHz. FWHM.135409976398216+/-.002212850MHz,R2.974111749,rawIQSNR32.953/residual1.0178sigma; gain.0025/20us/401/5000x1/100us. Rabi pi.641732283464567/sigma.058777626us; res9013.68/RO9014.06MHz/.73/6us/offset+.38;SSFabout.600. T1/T2R/T2E11.026177267674/3.310124660468/8.740105776448us all pass prior comparison; echoerror.185991743us,R2.967745040. No calibration row promoted yet.

ACTIVE m5_m025_qrepeat.json controller54752, launched14:43:41. Normal RR qspec-only, M5/index4/physicalQ4/Y3-2.5mA,Y4zero,labelrrv5m25_qr. Exactly matches full pass PRE-Rabi qspecseed6945.029248480MHz,pi.646706952,sigma.058777626us,RO9014.06/.73/6/+.38,requested6944.029248480-6946.029248480MHz401/5000x1/100us,qgain.0025/20us,statecal10k250.1800sdeadline,.5mA/sramp,0/offafter,canonicalHDF5max196chars. Require unchanged raw/model gates and exact-setting staticfrequencyrepeat (0.1smallerFWHM or3centerSE). Do not change storedfreq until review. Queue_h stopped after first group. bias_queue_l.json contains the remaining7groups, PREPARED/UNSTARTED, firstM6-2.5 thenM5/M6-5,-7.5,-10 plusendpointM3/M4controls. M1/finalall6zero remainunstarted.

New TEMP/prepare_exact_repeat.py is offline only: constructs explicit guard JSON from a completed audit and recorded runtime seed, verifies implicitprofile rounds/relax and Ramseyfitstarts, preservesrequestedaxis/statecal/RO/calibration,checksHDF5path. It does not contact hardware. Generated M6+10Ramsey andM6+2.5qspec fixtures reproduce the previously executed confirmations (checked14:34/14:39). Use it to avoid confusing a qspec PRE-Rabi seed/pi with the later coherencecarrier/pi. The sole validationPowerPoint has109slides asof14:28 with154configurationchanges; lastnativeQAincludedconfiguredfit tables andsidebysideQ6repeat. NewM5negativepass stillneedsreportrefresh afterreview.

### September 9, 15:03 local — M5 frequency repeat disagrees; bounded held-bias diagnostic prepared

M5-2.5 qspec repeat rrv5m25_qr/2026-09-09_14-43-53 completed with canonicalraw/plots and both0A/off. Center6944.798479013769MHz,FWHM.133291206221594MHz,R2.972727147; all raw/calibration/model gates pass, onlyhistoricalfrequencycomparisonfails. This is .235234031038MHz from the full pass center6944.563244982731MHz, beyond both staticrepeat tolerance .013329MHz and one linewidth. NO repeat-review waiver, no frequency/sigma/pi/readout/calibration promotion for this row. The two actual spectra are added side by side to the same report; stablefrequencyisnotestablished. Testing post-step time dependence does not yet identify a cause.

ACTIVE independent M6-2.5 complete pass m6_m025_full.json/controller23660/child in resultfile, started14:50:24,labelrrv6_m025a/2026-09-09_14-50-27. This is ONLY one group, not queue_l. All7normalflags,persistentnegativepreflightenvelope,5400sdeadline,.5mA/sramps,0/offafter. Preliminary res9057.88/RO9057.86MHz/.7967/12us/offset-.02; qfreq6103.945208979526MHz,FWHM.257278479MHz,Rabi pi.651681621193/sigma.056022425us. SSF.290 versus prior.427 will need readout-length and gain/offset optimization after the full audit. Ramsey currently acquiring. Six other negative groups (-5,-7.5,-10 bothYokos withendpointM3/M4controls), M5-2.5 review, M1 Ramsey andfinalall6zero stillremain. Do not start queue_l as-is: its first M6-2.5 label is now consumed by this single guard.

PREPARED ONLY m5_m025_held_qspec.json, labelrrv5m25_hold. Three qspec-only passes at one held Y3-2.5mA,Y4zero, same exactseed6945.02924848/pi.646706952/sigma.058777626/RO9014.06/.73/6us/+.38,6944.02924848-6946.02924848MHz401/.0025gain/20us/5000x1/100us,statecal10k250. No added initial settling; measure successive intervals across~14minwithoutzero/rampbetweenrounds.1800sdeadline,normalcanonicalbatch/plotperround,0/offafter. Launch onlyafterM6finishesandisreviewed; do not overlap. Then TEMP/analyze_held_qspec.py <run> analyzes ALL batches, verifiescalibration/settings/axes/bias/metadata/rawfits andrecordsconsecutivecenteragreement/timings; diagnostic-only and no promotion. It expects log round-start timestamps andall3HDF5/plots; not yethardwaretested.

Localized RR option PUCQ4_QSPEC_PASSES enables only1-3qspec-onlypasses onone row, validatedbeforehardware. Defaultn=1 remainsliteralunchanged. Existing whileloop holdsbias outside rounds andsaveswithsave_r=1, soactualcanonicalcollector yieldsdistinctbatch1/2/3 withRoundNum1/2/3 andBatchNum0/1/2. run_guard validates the sameboundedoption; no changes toclasses/QICK/Data_H5. Normal auditor explicitlyrejects runtime rounds!=1 asfinalcalibrationevidence. check_held_qspec.py executesactualselector/collector/save/reset withmocksave,checksdefaults/bounds/flags/one-row,thatnocurrentcallisinmeasurementloop,andparsesnewanalysishelper. PASS14:56and15:02.131fileharness,81x6profile/protectedhash/timingchecks andwhitespacePASS14:57. Initial testharnessASTselection errors were fixed toselect the actual immediateqspeccollector (rawiscollected immediately afteracquisition); no hardwarefailure orrawloss occurred. Newsourcechanges were made onlyafter activeM6acquisition_profile existed; its launchsources remainarchivedunmodified.

### September 9, 15:29 local — held-bias diagnosis complete; negative-bias readout optimization

M5/physical Q4 at Yoko3 -2.5mA held across three normal qspec-only rounds completed: rrv5m25_hold/2026-09-09_15-09-45, guard complete in804.56s, three distinct canonical HDF5 batches and three normal population plots; both sources zero/off afterward. Centers6944.968211377013,6944.953488150237,6944.958240321407MHz; FWHM.168506356,.171401802,.161682296MHz; R2.965185,.958424,.971990; rawIQ SNR30.98,27.86,32.48. Identical requested axis/pulse/readout/averaging and reconstructed calibrated populations verified for every round. Consecutive differences.014723227/.004752171MHz pass original static limits.016850636/.016168230MHz. This diagnostic does not establish a need for extra initial settling or a fixed shift across separate visits. No M5-2.5 calibration promoted. Existing full/repeat separated by.235234MHz remains documented. Prepared m5_m025_fullb.json: fresh all7 one-pass, last diagnostic center6944.958240321407MHz as acquisition seed only, unchanged pulse/readout/count/timing profile, no extra settling,1800sdeadline; run after independent Q6 readout work.

TEMP/analyze_held_qspec.py successfully checks ALL3 batches, actual RoundNum/BatchNum, raw/calibration/settings/axes/plots and fit gates. Analysis v2 uses the first acquisition-profile PC log timestamp as elapsed-time origin; M-drive file mtime differs by1.94s from the PC clock and is unsuitable for precise bias elapsed times. Both original diagnostic analysis and corrected v2 are preserved; v2 is selected for the PPT. No raw rewrite. Default RR remains one pass; bounded PUCQ4_QSPEC_PASSES3 is diagnostic only and normal auditor rejects it for final evidence.

M6/physical Q6 at Yoko4 -2.5mA full rrv6_m025a/2026-09-09_14-50-27 completed in1017.10s, all7HDF5/normalplots and both0/off. Qspeccenter6103.945208985MHz/FWHM.257277818MHz/R2.787383940 FAIL; SSF.2903 versus prior.427 FAIL; Ramsey1.221549362+/-.074619919us/R2.836662779 FAIL quality. T1 6.856581884+/-.138900548us and echo3.730736727+/-.169096784us pass model/reference. Actualpulse6103.945208979526MHz,pi.651681621193,sigma.056022425us; res9057.88/RO9057.86MHz/.7967/12us/offset-.02. No promotion. Readout improvement precedes any qspec gain/count or coherence-window change.

Readout length grid m6_m025_length1 completed with four SS-only normalRR points at fixed9057.86MHz/.7967 and20kshots/state250us:6usSSF.3014575,10us.3309,14us.3023076,18us.2921. All raw/plots persisted, both0/off after each. Selected10us for next gain/frequency step; this is not final readout calibration. Grid archived in execution/m6_m025_length1.grid.json and included in the same validation report.

ACTIVE m6_m025_gainfreq1.json controller50948, launched15:28:15. M6/index5,Y3zero/Y4-2.5, same diagnosticcarrier6103.945208979526/pi.651681621193/sigma.056022425;10us/20kshots/state250us; tone9057.46/9057.86/9058.26MHz (offset-.42/-.02/+.38 from measuredres9057.88), gain.65/.8/.95, nine points. NormalSS-only onepass,canonicalpaths<210,.5mA/sramps,0/offaftereach,240sdeadlineperpoint,abortacquisition/save/readbackerrors,nominalattenuatorargsunchanged. Reviewbestpointthenrepeatlength/finer2D until plateau; newfullchainrequired beforepromotion. Six remaining negative groups (-5,-7.5,-10 each source plusendpointM3/M4controls), M5-2.5full/review,M1Ramsey andfinalall6defaultzero remain. Do not run queue_l as-is: firstlabelconsumed. Continue complete task.

Environment lesson: initial15:24:09 Start-Process launch failed before Python/script/hardware because PowerShell array concatenation built malformed spaced paths. Original stderr retained; no measurement created. Retry15:24:51 used one formatted quoted argument string and completed the length grid. Use that quoting pattern for background launchers. No experiment classes/QICK/Data_H5 changed. Readout report titles now derive concise row/bias labels from recorded settings instead of placing long authorization text inside scientific plots; full plan remains in slide provenance.

### September 9, 15:47 local — Q6 readout plateau and final fine scan

M6-2.5 coarse2D m6_m025_gainfreq1 completed nine points at10us/gains.65,.8,.95/tones9057.46,.86,9058.26; best.3198 at.8/9057.86. Highergain.95 atcenter reducedcontrast to.11245. Interleaved length2 at.8/9057.86 gave10us.30455,8us.31805,12us.3068,10us.32645:8us is the shorter point at the observed8-12us plateau, with ~.02 repeat variation. Allcanonicalraw/plots and0/offaftereach. Fine2D gainfreq2 at8us/gains.74,.8,.86/tones9057.66,.86,9058.06 gave no significant improvement; best.30285 at.8/9057.86. Completed grids are archived in execution/ and included in the same report manifest. Final exact settings remain unpromoted.

ACTIVE m6_m025_gainfreq3.json controller10884, launched15:45:22. Same M6/Y4-2.5/Y3zero,diagnosticcarrier6103.945208979526MHz/pi.651681621193/sigma.056022425us,8us/20kshots/state250us. Finer tones9057.76/9057.81/9057.86MHz, gains.77/.8/.83, nine normalSS-onlyonepasses,.5mA/sramps,0/offaftereach,240sdeadlineperpoint,canonicalpaths<210. Review after all9; then finalinterleavedlength/repeat at selectedpoint before newfullchain. No additional currentsettling or class/QICK changes. No demonstrated cause is assigned to the contrast decrease from historical.427.

Next M6 full pass should increase qspec5000->10000reps (same401points/.002gain/20us/100us) and Ramsey1200x2->2400x2 (samefull0-30us/301/.5MHz/250us). Both first-pass traces failedR2 on noise; leave drivegain and delayrange unchanged while improving averaging. Use20kstatecalshots250us consistently. Select RO from completed optimization; measured diagnosticfrequency is only a fresh acquisition seed, not promotion. Existing env overrides support these counts, so no shared-code mutation is needed during readout-grid child launches. Persist exact counts/RO/pulse only after fullpassquality/repeatreview. If readoutcontrast remains belowpreviouscomparison despite replicatedoptimization, preserve that finding and evaluate repeatedcomplete-sequence evidence; never force.427 by weakening raw/model gates.

Prepared/unstarted m5_m025_fullb.json afterhelddiagnostic, negative_remaining_6.json (fresh six remaining negativegroups), m1_midpoint_prepare.json and final_all6_default.json remain necessary. Last PPT116slides/9issues at15:30; native116slidereview15:31 confirmedheld-biasplot andreadoutplotlegibility. Laterreadoutgrids stillneednextbuild. Offline131fileharness passed15:31, focusedwhitespacepassed15:30. Sourcecheckpoint_1543 archivescurrentcores/helpers/remainingplans. Task remains running; do not end with a partial handoff.

### September 9, 16:10 local — Q6 complete sequence active; selected readout and exact repeat review

All six M6/Y4-2.5 readout grids completed with normalSS-only raw/plots andbothsources0/off aftereachpoint: length1,gainfreq1,length2,gainfreq2,gainfreq3,length3. Close gainfreq3 at8us/gains.77/.8/.83/tones9057.76/.81/.86 showed no significant improvement (best.29995 at.8/9057.81). Finalinterleavedlength3 at.8/9057.86 yielded4us.28845,8us.28925723,6us.3123,10us.32285,8us.3117. Across repeated measurements10us remains the consistent higher-contrast choice: .3309(initialgain.7967),.3198,.30455,.32645,.32285 atgain.8. Selected10us/.8/absoluteRO9057.86MHz; longer12-18usnoimprovement,shorter4-8generallylower. No static readout/pulse calibration promotion yet. Allsixgrids are archived in execution and referenced in the same validationmanifest.

ACTIVE m6_m025_fullb.json guard51368/child39164, launched15:56:34, runrrv6_m025b/2026-09-09_15-56-38. All7normalflagsonepass atY3zero/Y4-2.5; selectedRO10us/.8/9057.86/offset-.02, same sigma.056022425us and qspecgain.002/20us. Qspec10000x1/401/100us (old5000), statecal/SSF20kshots/state250us, Rabi0-.85/171/1200x2/250us. T1.01-60/241/1200x2, Ramsey0-30/301/2400x2 (old1200reps), echo.01-60/301/1200x2;250usrelax andphase.5MHz. Full windows and drivegain retained. .5mA/s ramps, no extrasettling,2700sdeadline,both0/offafter,nominalattenuationsunchanged10/15,5/4,17dB. Longestprojectedplot/HDF5path196 verifiedfromsame-lengthpriorfullpath; runtimealsoverifiesbounds. Both sourcesverified0/offbeforelaunch. CurrentT1acquisition; all finalnumbersawaitcompleteHDF5audit.

Provisional current-run console/normalplots: measuredres9057.88MHz,actualRO9057.86/offset-.02; qspeccenter6104.052944660058MHz/FWHM.279261949MHz; newRabi pi.6516816211926998 at unchanged sigma.056022425us; SSFabout.315. Qspec now10000reps, but actual requested window was6102.736072740-6104.736072740MHz fromoldsharedseed6103.736072740, despite diagnosticPUCQ4_QFREQ_MHZ6103.945208979526. The line lies inside this actualwindowwithbaseline; preserveexactactualbounds. The priorcomparison tolerance ismax(.1,oldFWHM,newFWHM), so frequency comparisonmustusefullauditor, notoldwidthalone.

Diagnosed orchestration inconsistency: per-row qspecbounds were applied before the optionalcarrier override. RR now recenters the existing profile span whenPUCQ4_QFREQ_MHZisprovided,unless explicitqspecstart/stop orhalfspan override wins. Ordinary defaults unchanged. This localized sourcechange wasmade16:08 after activechild's15:56 acquisition_profileexisted; activefullb retainsits OLD launchsources capturedbyguard. NextM5fullb willusefixedselector; exactqspecrepeathelperalreadycopiesexplicitrecordedbounds andremainsunchanged. check_qspec_seed_bounds.py executesactualASTselector7cases(default,recenter,explicitspan/bounds,missingbound,rowisolation),PASS16:08:39.131fileharness,81x6profiles/timing/protectedclasses/guardchecks,held3batchintegration,focusedwhitespacePASS16:09. No underlyingclass/QICK/Data_H5change.

TEMP/analyze_ssf_repeats.py AST-extracts unchanged SingleShot.hist_ssf without importing hardware; plot=False on actual rawI/Q exactly reproduces all4matching10us/.8/9057.86 SSFvalues.300pairedI/Qbootstrapresamples/state/record,seed1729,SE.004281/.004562/.004756/.004682. Range.0219 is somewhatlarger than samplingerror; no physicalcause assigned. JSON/plotM6_m025_SSF_matching_repeats inARTIFACTS,diagnosticonly/rawunchanged. This does notpromote oldSS-onlyruns orforcehistorical.427.

PREPARED offline review_repeated_ssf.py onlyfor a fullpass whose SOLE finding isSSFhistoricalcomparison. Requiresall7raw/model/Rabi/cohchecks, matchingindependentSS-onlyrepeatcarrier/pi/sigma/RO/trigger/phase/shots/relax andsameconfiguredsettling,verified0/off/normalplots,exacthistogramreproduction/bootstrap,>=2lengthand>=2gainfrequencyscanswithselectedpointrepeats. Predeclared repeatagreement<=max(.03contrast,3combinedbootstrapSE). KeepsoriginalSSFcomparison andbothaudits; no change toglobalquality/referencegates. Ifconfirmationreturnsinsideoldband,labelband-overlapinstead ofpersistentloss. Pure16syntheticpolicycases PASS16:05, including rejectingmismatchedcarrier/counts/bias/duplicateHDF5/weakqspec/Rabi/badcoherence/missingoptimization/incorrectcomparisonflag. NEVERapplywhileotherfullpassqualityfindingsremain. PPTbuilder supportsSSFreviewrowswithunitfraction. Helpernotyetappliedtorealnewfullpassorusedforpromotion.

Remaining: auditcurrentM6fullb,exactSSFconfirmation/reviewifneededandpossible,thenpromoteexactselectedsettingsandpersistqspecreps10000/Ramsey2400x2/statecal20k inthatcurrent'sprofile ONLY ifvalidated. NewM5-2.5fullb planprepared; sixunstartednegativegroupsinnegative_remaining_6.json; M1fresh-carrier/midpoint investigation; finalall6defaultzero; finalsource/reference/PPTQA. LastPPT119slides/9issues16:08; newtwofinalgrids/bootstrapneednextbuild. Do not stop atpartialhandoff.

### September 9, 16:32 local — Q6 echo follow-up required; Q4 full pass active

M6 fullb rrv6_m025b/2026-09-09_15-56-38 completed in1518.507s with all7normalHDF5/plots andbothsources0A/off. Actualpulse6104.052944660058MHz; qspecpopulationcenter6104.052943671447+/-.003090372MHz,FWHM.279252497252343+/-.010978605MHz,R2.868332636;rawIQcenter6104.054199426,width.28989146,SNR10.39939,residual1.05321sigma. Requestedaxis6102.73607274-6104.73607274MHz401points/10000x1/.002gain/20us/100us. Allqspecraw/model/referencegatesPASS (difference.2547915MHz versusoldqspec iswithinmaxold/newFWHM.2792525). Rabi pi.6516816211926998 at sigma.056022425us,R2.993085937. Res9057.88/RO9057.86MHz/.8/10us/offset-.02;SSF.31525 versusold.427 stillfailscomparison. T1 6.013042039+/-.114702813us,R2.980198219 PASS(+1.62%old);Ramsey1.219826536+/-.055882392us,R2.891216300 PASS(+3.94%). Echo4.268605296+/-.222166973us roughlymatchesold4.150655(+2.84%),butR2.835244620 FAILquality;independenttau4.268577676 agrees,fringef.5094064MHz/9.886points,residualpeak.0365163. Do notpromoteoruseSSF-onlyreviewwhileechoqualityfails.

Original validation_M6.json falselyflaggedRabi calibrationcount mismatch: actual20000,oldauditorassumedprofile10000 despiteqspecoverride20000remainingactivebeforeRabi. Rawpopulationreconstruction verified. Newpure requested_state_calibration helper in tools/audit_pucq4_m4 follows enabled qspec->Rabi->SSF->coherence override order forsharedReadout_Optimization, separatelytrackingcountsandrelaxationsources. Exactcount/reconstructiongatesretained. check_calibration_request.py passes11precedence/default/disabledstagecases,all5actualfullbcalibrationrecords,and65calibrationrecordsfrom13selectedfullpasses. Reauditpreservedoriginalasvalidation_M6_counts.json:onlySSFcomparisonandechoqualityremain. No raw/class/QICKchange. This corrected audit is the fullb evidence for furtherwork.

ACTIVE M5fullb m5_m025_fullb.json guard59412/child16912, launched16:24:57, runrrv5_m025b/2026-09-09_16-25-00. All7normalflagsonepass,Y3-2.5mA/Y4zero,.5mA/sramps,noextra settling,1800sdeadline,0/offafter. Lasthelddiag6944.958240321407MHz is explicitseed; newRRselectorcorrectlyrequests6943.958240321407-6945.958240321407MHz401points/.0025gain/20us/5000x1/100us. OtherpersistentM5profilesunchanged,sigma.058777626,RO9014.06/.73/6,cal10k250. Currentnewqspecfrequency6945.1129971731425MHz/FWHM.119486858141MHz matchesold6945.111955771/FWHM.1205261565. Rabi/coherencecontinuing; do notpromotefromconsolealone. SourcefixhardwarewindowverifiedinactualRRlog; no classmodification.

PREPARED ONLY m6_m025_fullc.json: freshcompleteM6atY4-2.5 afterM5fullb. Onlysciencechangefromfullb isecho1200x2->2400x2 (samefull.01-60us/301/.5MHz/250us). Keepqspec10000x1 andRamsey2400x2,allcal20k250. UseEXPLICITfullbqspecbounds6102.73607274-6104.73607274MHz so recentrebugfixdoesnotchangesweep;samepreRabi seed6103.945208979526/pi.651681621193/sigma.056022425,RO9057.86/.8/10/-.02. Rabi20kstatecalnowexplicitPUCQ4_RABI_SSF_SHOTS instead of inherited; actualscienceunchanged. All7normalflags,2700sdeadline,196charcanonicalpaths,.5mA/sramps,0/offafter. Do notlaunchinparallel. Previousm6_m025_ssrepeat.json is PREPARED/UNUSED basedonfullbcarrier; do notuseitfornewfullcwithoutcheckingnewactualcarrier/config. AnewexactSSFrepeat/fullcbootstrapreviewmayfollowONLYifallotherqualitygatespass. No M6-2.5 parametershavebeenpromoted.

Remainingafterthesefollowups: sixnegativegroupsinnegative_remaining_6.json(-5,-7.5,-10 eachsource,endpointM3/M4controls),M1fresh-carrier/midpointinvestigation,andfinalall6defaultzero. SameonePPT122slides/9issuesat16:16; native122QA16:16-17 checkedreadoutsettingsfooter. Duplicate-lengthreadoutpointsnowplottedasunconnectedpoints toavoidimplyingacquisition-orderlinesbetweenrepeats;nextbuildwillapply. No newPowerPointfile. Finalsource/reference/presentationQAandconfigurationpromotionstillrequired;continuecompleteauthorizedtask.
