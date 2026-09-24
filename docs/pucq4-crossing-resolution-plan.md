# Q5/Q6 unresolved-evidence continuation

## Published closeout — September 14, 20:41 local

Executed eleven additional guarded runs and published the updated **143-slide**
`pucq4_first_light/PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx`.
The bounded investigation and reporting are complete; the scientific limits
below remain explicit. No acquisition is active, and both sources ended at
0 A with outputs off. Earlier checkpoint instructions below are historical.

| Planned item | Final disposition |
| --- | --- |
| Physical identity and routing review | Complete review; physical Q5 remains unassigned. A device-to-resonator mapping or uniquely identifying measurement is still required. |
| Separate spatial Q5 drive | No separately wired/calibrated path documented; current software uses shared DAC1. Hardware routing/calibration evidence is still required. |
| Detuned transition recovery | Accepted M2 transitions at -5.8, -5.4 and -4.8 mA. |
| Matched transition pairs | Three pair observations at two distinct biases, -5.8 and -5.74 mA. |
| Central/opposite-side sensitivity | Completed bounded power, averaging and readout tests. Central upper candidate remains rejected at 64000 averages; opposite-side M2 upper remains unaccepted. |
| Seven paired biases, minimum splitting and J | Unresolved because the tested central/right branch visibility does not supply accepted pairs; no coupling value promoted. |
| Evidence and report | All 22 plans audited: 76 spectra, 213 HDF5 files, 137 expected plots, 4.295914 GiB raw data; all guards exit 0 and sources parked. All 143 slides rendered natively without detected overflow; key slides visually reviewed. |

Final evidence is under `ac_review/resolution_20260914/`: `final_analysis/`,
`evidence_audit_final.json`, `combined_block_analysis.json`, `findings_final.txt`,
`native_final/`, and `publication_receipt.json`. Published SHA256:
`0B3AB5B08AC4C5AA496FA3ACCB84BA08884397FD4E3E86894A6517A09678DA0F`.
The original Downloads deck and earlier 94-slide report remain preserved.

## Final acquisition disposition (September14 20:35)

All22 guarded plans are complete; no acquisition controller remains. Latest
central_max_0914/ac_cmax completed exit0/2989.984s; both sources verified
CURR/.01A/0A/output-off. Its lower control passes at5431.735513621MHz.
Upper candidate5444.498748865 +/- .019163206MHz, width.359666479 +/-
.086337762MHz, SNR3.711767,18.000samples/FWHM,residual.905212,rank2/no
active bounds and clean reference still FAILS SNR at64000/.03. It is not an
accepted line or pair. Raw signed-IQ plot was reviewed; the weak feature
reappears near the earlier candidate, but increased averaging did not produce
the assumed stationary-noise sensitivity improvement. No cause is assigned.

The explicit central-pair prerequisite for right_m6_0914 was not met; that
conditional plan was NOT launched. Stop repeated maps at this demonstrated
ceiling. Two distinct detuned biases retain accepted pairs, but the required
seven-bias bracketed series, minimum splitting and J remain unresolved by
branch visibility. PhysicalQ5 mapping and a separate calibrated spatial drive
also remain unsupported by available device documentation. These are precise
remaining scientific/documentation limits, not claims of global mode absence,
zero coupling or a demonstrated dark-state mechanism. Final audit/report follow.
All ACTIVE/PREPARED entries below are historical unless explicitly stated here.

September14 20:00: ac_cmax lower positive control ACCEPTED at5431.735513621
+/- .006457948 MHz,width.170162996+/-.022853116MHz,SNR7.97450,
8.516samples/FWHM,residual1.05308,cleanreference. It agrees with earlier
M2 central5431.727524559 within conditional errors. Upper64000 nowacquiring.

Conditional extension prepared/validated, NOT launched: right_m6_0914/ac_r6hi.
ONLY if central_max establishes an accepted central pair, test -5.59mA through
M6 with lower5435.2-5437.6/121pts/8000 andupper5452.5-5458.5/301pts/32000,
.03gain20us100us/equalrefs,tagright_m6,4500s+180guard,about48min,zero/off.
M6 has not received the higher-averaging right-side test: priorac_up1 used
4000reps50kHz; new32000/20kHz stays inside its measured upper coverage.
M6's prior lower atthisbias5436.328829562MHz/SNR24.9625 is the control seed.
The M2 right null cannot exclude M6 visibility, and a different bias can change
relative readout response; this is a testable hypothesis, not a mechanism claim.
Original current/program/tracker/nominalattenuation/storage controls retained.
If central_max upper remains unaccepted, skip this extension and close the
bounded visibility-limited investigation. If both central/right pairs pass,
continue toward the required seven-bias bracketed series with stability controls.
Do not claim J or pool incompatible powers/visits to manufacture seven biases.

September14 19:44: ac_right completed exit0/2729.562s with bothsources
CURR/.01A/0A/output-off; no recent Python controller remained. M2 lower
control5436.331129599MHz/SNR30.1265 accepted; upper5452.5-5458.5 MHz at
32000/.03 has no candidate,vectorSNR2.49436,cleanreference. Raw signed-IQ
reviewed. This is a qualified M2/window/power sensitivity limit, not global
absence or zero coupling. No opposite-side pair is accepted.
The previously prepared central_max_0914 plan was updated with this exact
review and parking evidence, revalidated offline, and launched as the final
targeted central M2 candidate confirmation. Envelope unchanged: -5.656600mA,
.03gain,lower5430.7-5432.7/101pts/32000,upper5443.2-5445.8/131pts/64000,
20us100us/equalrefs,4200s+180guard,canonicalac_cmax storage,zero/offcleanup.
If the upper candidate remains unaccepted at this ceiling, stop repeated maps
and publish the complete visibility-limited result. If accepted, review the
remaining readout/branch/stability evidence before any further bounded test.

Additional offline stability evidence19:10: unchanged chronological-block
analyzer validates raw reps/frequency/readout/IQ shape and full-average
normalization point by point. ac_bcf M2 lower has four accepted quarters with
centers5416.231567,5416.247497,5416.220282,5416.232267 MHz: span27.215 kHz.
Its upper has three accepted quarters, span2.738 kHz (first not accepted).
ac_right lower has four accepted quarters, span9.022 kHz. These are descriptive
within-trace changes; each block shares the complete saved off reference,
so no independent drift significance or cause is assigned. They reinforce
retaining temporal uncertainty beyond conditional whole-trace errors.
Saved in M2_bridge_blocks/ and M2_right_lower_blocks/; combined_block_analysis.json
collects these and the earlier M4 analysis with independent source hashes and
raw paths, without pooling data. Use this collection for the final report.

September14 19:04 checkpoint: ac_right lower M2 control is accepted at
-5.59mA/.03/8000:5436.331129599 +/- .002384236 MHz, FWHM.197490834
+/- .008795630 MHz,SNR30.12649,9.884samples/FWHM,residual1.34966,
clean reference. It agrees with the earlier ac_fm2 center5436.328535938
within conditional errors. Upper5452.5-5458.5 MHz off/on is acquiring.
20 completed plans passed the evidence audit:203HDF5/131plots/3.633654982GiB,
all expected artifacts/source hashes and zero/off terminal guard states.
Use evidence_checkpoint_20runs_verified.json (exit0); the first audit wrote
its result then hit a SameFileError copying its own helper. Re-running from
the original TEMP helper resolved this output-bookkeeping failure. Raw data
were unchanged. Native135slide checkpoint renders without overflow; findings,
paired table and updated per-window averaging figure were visually reviewed.
Final report still requires active/subsequent results and final audit.

Prepared central_max_0914/ac_cmax, NOT launched while ac_right runs. One targeted
confirmation at existing maximum64000 averages and gain.03 will test the M2
central upper candidate that failed only SNR4.200675 at32000. Repeat measured
5443.2-5445.8 MHz/131points, source-5.656600mA; lower control5430.7-5432.7/
101points/32000;20us/100us/one round/equal off-on within each window,
tagcentral_max. Upper averages32000->64000, window5442-5448->5443.2-5445.8,
frequency step stays20kHz. Expected~46min,4200s+180guard, unchanged source,
program, tracker, nominal attenuation and zero/off controls. Offline validation
passed. Requires full ac_right result/parked review before launch. Stationary
noise would give SNR~5.94 after doubling averages, but repeatability is unknown;
this calculation is not an accepted line. No increase beyond the demonstrated
gain/averaging ceiling or blind multi-bias map if usable pairs remain absent.

Latest September14 19:00: central M6 completed exit0/1607.734s, both sources
CURR/.01A/0A/output-off and no remaining controller. Upper5443.2-5445.8 MHz
has no candidate (vector SNR2.70837), clean reference; raw signed IQ reviewed.
Lower remains accepted. The M2 central upper candidate remains rejected for
SNR4.200675; M6 does not supply a confirmed second line.
After full review/parking, right_visibility_0914/ac_right launched with the
previously recorded M2/-5.59 envelope unchanged: guard13696/child55692.
M2 was retained because its central upper candidate was stronger than M6's;
this does not imply globally superior readout. See runtime/guard for live state.
The unlaunched plan's authorization/provenance was updated, validated and saved
as UTF-8 without BOM. PowerShell Set-Content -Encoding UTF8 initially introduced
a BOM rejected by the offline checker; rewritten before successful validation
and launch. No hardware was started by the failed check.

User request: make a plan, execute it without routine permission prompts, and
update the existing PowerPoint. This reopens the completed spectroscopy campaign;
it does not make its unresolved scientific claims accepted results.

## Ordered work and acceptance criteria

1. Check physical identity and drive routing against the original Downloads
   characterization and available device documentation. Accept a physical Q5
   assignment only from documented routing or a uniquely discriminating measured
   signature. The original table explicitly leaves M1-M4 unassigned. A shared
   response near M2 alone cannot assign Q5.
2. Recover the fixed-row M2 transition on both sides of the previously scanned
   interaction region. First measure M2 at Y4 -5.8 and -5.4 mA, 5436-5441 MHz,
   501 points, 4000 repetitions, gain .007, 20 us pulse, 100 us relaxation,
   one round, equal drive-off references. This uses M2's measured zero-current
   drive gain and frequency as a search seed, not a current calibration.
   Retain the unchanged resonator tracker. Accept peaks only through the existing
   raw-IQ noise, linewidth, sampling, residual, and reference gates.
3. Adapt subsequent bounded scans to measured endpoint evidence. If both modes
   become visible, acquire at least seven unique currents resolving both modes,
   with two points on each side of the minimum, then apply the existing coupled
   fit and fixed-frequency alternative-model tests. No single-branch estimate of
   J, minimum splitting, or resonance current is accepted. If weak visibility
   persists, test averaging/power using existing demonstrated limits, preserving
   negative results and stopping repeated scans that add no discriminating data.
4. Establish whether separately routed Q5 drive exists in documented hardware.
   Software frequency selection on DAC1 is not independent spatial addressing.
   Do not invent a port, repurpose readout DAC0, or assert physical rewiring was
   performed remotely. Record any exact external prerequisite that remains.
5. Preserve the previous 94-slide report and raw evidence. Update the same
   report with the new plan, measurements, accepted conclusions, and remaining
   limitations. Verify canonical HDF5, plots, terminal source states, report
   readability, and focused source changes. Record the outcome in the task log.

## First-run hardware envelope

Physical Q6=M6/index5/Yoko4 192.168.1.77/A5. Yoko3 physical Q4 stays zero/off.
Both sources must start in current mode, .01 A range, zero/off. Y4 ramps at
.5 mA/s; zero extra DC settling; current settings -5.8 and -5.4 mA lie inside
both the existing controller's -5.8 to -4.8 mA envelope and the reference deck's
+/-10 mA limit. Park both zero/off after completion or error. Deadline 1800 s
plus guard allowance. Shared DAC1; DAC0/ADC0 readout. Nominal attenuation
arguments remain 10/15, 5/4, 17 dB, with no fresh attenuation-readback claim.
Use canonical `pucq4_first_light/ac_det1/<timestamp>/study_data/Data_h5` storage;
validate artifact path below 260 characters. Abort source-state, acquisition,
save, path, or deadline failure. No concurrent hardware controller and no
persistent calibration promotion from diagnostic spectra.

## Documentation and source review

The original characterization slide explicitly states uncertainty about indexing
and labels M1-M4 with question marks. Its layout places physical Q5 beside Q6,
but does not establish which measured fixed resonator belongs to Q5. Searches
of Downloads and C:/_Lib for PUCQ4/layout/wiring/GDS files found no additional
PUCQ4 routing source. This is a scoped search, not proof that no document exists.

Current system_config.py assigns FSGEN_CH=1 and MIXMUXGEN_CH=0; both qubit_ch
and qubit_ch_ef use FSGEN_CH. The unchanged PulseProbeSpectroscopyProgram uses
the requested qubit_freq_ge sweep on qubit_ch and then the selected readout.
The crossing controller explicitly copies the selected resonator's integration
length, gain, phase and offset. No sweep/channel-selection defect was found.

A separate spatial drive requires a documented wired control path and its
calibration. A physical Q5 identity requires a design-to-measured-resonator map
or another uniquely identifying device measurement. Neither can be established
by renaming M2 or declaring a second software channel on the shared DAC.

The new guarded run ac_det1 started 2026-09-14 14:42:36 local, guard32264,
child34156, after fresh parked-source checks. Previous report was preserved as
ac_review/final_20260914/PUCQ4_Q5_Q6_94slides.pptx with matching SHA256
1885B880D5B785DBCA4DB706D627BD0F9EFCEB764305786770084C26FF676496.

First endpoint partial review: M2/-5.8mA candidate5439.870923242 +/-
.006289201MHz, FWHM.137324733 +/-.021419827MHz, SNR4.859624,
13.745samples/FWHM, residual1.0083noise units. Drive-off reference clean.
Only SNR fails; it is not an accepted transition. Prepared/unlaunched
detuned_confirm_0914.json/ac_dcf1 repeats5439.3-5440.3MHz101points16000reps,
same .007gain/20us/100us/equal references, at-5.8mA. This increases averaging
fourfold while preserving drive power and10kHz sampling. Deadline1400s+180guard;
all other first-run controls and canonical hierarchy retained. Requires current
guard completion and parked verification before launch. Partial evidence lives
in ac_det1/2026-09-14_14-42-36/documentation/analysis_p1/.

ac_det1 completed exit0/1081.89s, both sources independently verified0/off.
M2/-5.4mA accepted5438.068054185 +/-.001443033MHz,
FWHM.071050491 +/-.004489269MHz,SNR15.1683,7.112samples/FWHM,
clean reference. Complete analysis is documentation/analysis_complete.
The unlaunched ac_dcf1 plan is expanded to add a fresh more-detuned M2/-4.8mA
control5437.8-5439.8MHz201points4000reps/.007, using the validated zero-current
5438.708MHz line as its search seed. Deadline becomes1800s+180guard; all
other controls unchanged. This separates recovery of a shifted endpoint line
from reliance on a historical fixed-frequency value. No bare-frequency
constancy or physical identity is inferred from the accepted -5.4mA line alone.

Prepared/unlaunched detuned_q6_0914.json/ac_dq6, conditional on the confirmation
review: atY4-5.8mA, M6 readout5350-5442MHz1841points1500reps/.03,
20us100us1round/equalreferences,1800s+180guard. Search extends below the M2
candidate, following the measured lower-frequency trend of Q6 with decreasing
current. Bounds are search coverage, not an assumed tuning calibration.
50kHz grid is a locator; narrow candidates need refinement. Existing tracker,
nominal attenuations, source guards and zero/off cleanup remain unchanged.
No simultaneous controller. This tests second-mode visibility at greater
detuning rather than repeating the previously searched central gap.

ac_dcf1 first trace accepted: M2/-5.8mA5439.864904956 +/-.002935329MHz,
FWHM.130707246 +/-.011378457MHz,SNR11.0063,13.083samples/FWHM,
residual1.0415,cleanreference. It reproduces the earlier rejected candidate
within conditional errors. Accepted M2 endpoint difference(-5.8 minus-5.4)
is1.796850771MHz; these are different bias settings, not a simultaneous
doublet or a measured minimum splitting. More-detuned control is acquiring.
This passes the predeclared condition for the prepared Q6 second-mode locator,
which remains unlaunched until guard completion/parked verification.

ac_dcf1 completed exit0/686.204s, bothsources0/off. More-detuned M2/-4.8mA
control accepted5438.540998283 +/-.001467710MHz,FWHM.077923430
+/-.004610503MHz,SNR14.7554,7.800samples/FWHM,cleanreference.
These three different-bias centers do not establish a constant bare frequency.
The Q6 locator launch condition is met; ac_dq6 launched after both previous
controller processes exited. Shared calibration and programs remain unchanged.

ac_dq6 completed exit0/796.297s, bothsources0/off. M6/-5.8mA accepted
5402.784229173 +/-.008315814MHz,FWHM.514242907 +/-.032236174MHz,
SNR17.4509,10.286samples/FWHM,cleanreference. Only one accepted line in
5350-5442MHz. Together with the M2 line this provides two separated locator
frequencies at the same bias, but at different drive powers; no paired-mode
acceptance or J is claimed from those locators.

Next matched_pair_0914.json/ac_pair1 verifies BOTH readouts in BOTH measured
windows at common .007gain/8000reps/20us100us1round/equalreferences:
5401.8-5403.8MHz201points and5439.2-5440.6MHz141points, Y4-5.8mA.
Order M6lower,M2upper,M2lower,M6upper. Each trace explicitly identifies
paired_windows_id=detuned1. The intervening frequency gap is unmeasured in
this run; do not interpolate it or combine unrelated/repeated visits.
Offline analysis must require identical disjoint windows on both readouts,
matching current/power/pulse, clean references, valid individual peaks,
and no conflicting additional accepted line before forming a branch pair.
This retains the existing branch-quality and seven-bias coupled-fit criteria.
No hardware program changes. Original tracker/attenuation/source guards and
parking retained; deadline3000s+180guard. All raw data remain per-window
canonical HDF5. Segmented acquisition avoids repeatedly sampling the empty
gap while preserving each trace's real measured coverage.

ac_pair1 first primary trace passes at common .007gain: M6/-5.8mA
5402.788027079 +/-.002497761MHz,FWHM.321908374 +/-.010400075MHz,
SNR20.1525,32.221samples/FWHM,cleanreference. Center agrees with the
higher-power locator. The smaller fitted width is a sequential comparison,
not an isolated causal power test. Remaining upper-primary and cross-window
controls are acquiring; no complete matched-window pair is accepted yet.

Offline analysis now combines ONLY explicitly tagged identical disjoint windows
on both readouts, preserving each original trace, window, timestamp and raw path.
It rejects mixed manifests/settings/visit IDs, missing or overlapping windows,
reference findings, out-of-window peaks and unmatched additional accepted lines.
Ordinary untagged repeated visits remain rejected. Maps paint actual bins only.
Synthetic known-J recovery matches continuous-window results; negative tests
cover each of these failure cases. Existing signed-IQ, close-doublet, zero-J,
drift and duplicate checks also pass. Offline harness138files and whitespace
checks passed after these changes. Acquisition code and calibration untouched.

Prepared/unlaunched central_lowpower_0914.json/ac_ctrlo, conditional on a fully
accepted common-power endpoint pair and parked sources: at-5.656667mA both
readouts acquire upper5442-5448MHz301points(20kHz) and lower5430.7-5432.7MHz
201points(10kHz), everytrace8000reps/.007gain/20us100us1round/equalrefs,
paired_windows_id=central1, orderM6upper/M2upper/M6lower/M2lower.
Deadline4000s+180guard, original tracker/nominalattenuation/source controls
andzero/offcleanup. Previous central gap used.03gain,50kHzgrid,4000reps;
this explicitly tests lower-power visibility with finer sampling.

Upper-window selection is a conditional locator, not a reported coupling fit:
using fixed-row seed f0=5438.710887054MHz, detuned frequencies
l=5402.788027079,u=5439.864904956MHz and prior central lower c=5431.742653763,
the two-mode equation gives upper=f0+(u-f0)*(f0-l)/(f0-c)=5444.6601MHz.
The5442-5448MHz window includes margin. These assumptions have not established
J, a bare fixed frequency, or physical Q5 identity. Reject rather than promote
the model if the missing branch is not measured. No launch before current
plan review and source cleanup; do not run prepared plans blindly.

ac_pair1 second primary trace accepted M2upper5439.862748670
+/-.003573086MHz,FWHM.102964572 +/-.012149546MHz,SNR7.0742,
10.306samples/FWHM,cleanreference. Primary lower/upper are now measured at
common.007gain, but both cross-window controls still need completion before
pair acceptance. analysis_p2 correctly reports missing paired coverage.
Window grouping additionally requires contiguous source-record indices, so
tag reuse across a return visit cannot silently stitch different bias visits.
Regression tests for this and actual-versus-requested coverage pass.

Third trace M2lower also accepted5402.791625628 +/-.007498609MHz,
FWHM.336541067 +/-.031735416MHz,SNR8.0122,33.685samples/FWHM,
cleanreference. It agrees with M6lower within conditional errors. Thus M2
has two individually resolved lines in the two disjoint windows atcommonpower;
full matched-coverage acceptance still awaits finalM6upper control.
The report builder now includes an explicit table of accepted matched-power
line pairs, with separation uncertainties (including joint covariance when
available), and distinguishes each bias's separation from a minimum or J.

ac_pair1 completed exit0/1492.5s, bothsources0/off. FinalM6upper has no
candidate (vectorSNR1.8389), cleanreference. Full actual common-window and
contiguous-visit checks PASS. Selected pair: bothlines inM2,
5402.791625628 +/-.007498609MHz and5439.862748670 +/-.003573086MHz,
separation37.071123043 +/-.008306388MHz at-5.8mA/.007gain. This is one
resolved paired bias, not a measured minimum or J. M6lower agrees.

central_lowpower_0914 launched15:55:27, guard16720/child9888,
ac_ctrlo/2026-09-14_15-55-29, after full accepted pair and fresh parked checks.
Run remains active. Completed15-run/58-trace analysis checkpoint is
ac_review/resolution_20260914/progress_pair/analysis.json: all49earlier trace
classifications, centers and widths exactly unchanged by the analysis extension.
110-slide draft in TEMP/pucq4_cross_20260909/pair_progress.pptx remains UNPUBLISHED;
native preview of title/findings/line-pair table is readable. Correct the manually
rounded separation SE in final findings to.008306MHz (table already correct).
Final report/audit still pending completion and review of the central test.

Both original PowerPoint slide XML parts were also inspected; no speaker-note
parts are present. The table explicitly leaves M1-M4 physical identities as
question marks, and slide2 adds tuning-range discussion, not a Q5 mapping.
No hidden text in this supplied reference resolves physical Q5 identity.

Central first upper-window trace(M6) has no candidate, vectorSNR2.7428,
cleanreference; evidence ac_ctrlo/.../documentation/analysis_p1. M2upper
and bothlower windows still acquiring. This does not exclude an upper mode.

Prepared/unlaunched bridge_pair_0914.json/ac_bridge follows the measured M2
pair inward to-5.74mA after full central-run review and parked verification.
M2 upper5439.5-5441.5MHz thenlower5415.2-5417.2MHz, each201points8000reps,
.015gain/20us100us1round/equalrefs,paired_windows_id=bridge1,1800s+180guard,
originaltracker/nominalattenuations/sourcecontrols/zero-offcleanup.
Gain.015 was already demonstrated onM2 in ac_m2a with narrow accepted lines.
The same conditional locator equation and an interpolated bare-mode search
curve place candidate centers at5416.2258/5440.5510MHz; these are search
centers only, not accepted branch frequencies, calibration, J or identity.

The scientific objective remains to obtain at least seven distinct paired
biases with an interior bracketed minimum before accepting the existing
coupled fit. A weak central window is not a reason to abandon tracking the
newly measured branches from the detuned side. The bridge pilot tests whether
a multi-bias M2 map is viable; no such map has been launched or prepared yet.

Analysis now also permits a single readout to supply both lines from explicitly
tagged disjoint windows, preserving the same source/visit/coverage/reference/
linewidth/error checks. If two readouts are present, cross-readout disagreement
still blocks their segmented self-fits. Known-J single-readout tests pass,
including missing/repeated windows, unresolved wide lines and conflicting
readouts. Original acquisition code and raw data remain unchanged. The report
deduplicates identical within-readout/cross-readout pairs and labels row/gain.

M2 central upper-window trace also has no candidate(vectorSNR2.5329), clean
reference; analysis_p2 preserves both negative upper windows. Lower controls
remain active at16:19. The bridge pilot is validated, longestpath198chars,
but remains unlaunched until the central guard finishes and sources are parked.
Real ac_pair1 integration with segmented self-fits retains the same accepted
M2 pair and the same one-bias fit rejection. Native table QA confirms one
deduplicated row with correct M2 label and separationSE.008306MHz.

Bridge plan revised BEFORE launch: prepend a common .015gain anchor at-5.8mA,
M2upper5439.2-5440.6MHz141points thenlower5401.8-5403.8MHz201points,
paired_windows_id=bridge_anchor. Existing-5.74mA point follows. Allfourtraces
8000reps/.015/20us100us/equalreferences. Deadline becomes3600s+180guard;
expected26minutes, otherwise unchanged controls. The prior single-point plan
was never launched. This measures anchor/inward pairs at the same power before
using them to guide a larger map; it avoids defining that map from mixed-power
locator differences. Two pairs alone still do not validate a coupling model.

Central M6 lower-window control atgain.007 also has no candidate,
vectorSNR1.6807,cleanreference (analysis_p3). Thus this low-power central
protocol lacks a positive lower-line control onM6 and cannot decisively test
the upper-branch model through that readout. FinalM2lower still acquiring.
Retain this sensitivity limitation; do not label the absent upper line proof
of zero coupling, a dark state, or an incorrect two-mode model.

Central run completed exit0 in2158.75s; both sources verified CURR/.01A/0/off.
FinalM2lower also no candidate(vectorSNR1.3329), cleanreference. Allfour
traces reviewed in documentation/analysis_complete. Bridge two-point plan
passed offline safety/raw roundtrip checks after removing a PowerShell UTF8
BOM before launch. No acquisition occurred with the invalid JSON.
Bridge launched guard38352/child42576; currently active.

Analysis current-axis correction: use saved actual_yoko4_current_mA for
maps, grouping and model fits; preserve requested_current_mA separately.
Requested-5.656667mA was recorded-5.656600mA, a67nA difference corresponding
to about15kHz at235MHz/mA. This is a SOURCE:LEVEL? setting, not an independent
ammeter. Require matching finite off/on settings; original request tolerance
and raw data remain unchanged. Original acquisition plot titles may retain
requested settings. Synthetic mismatch/nonfinite tests and all prior scientific
checks pass. Individual frequency-fit invariance will be rechecked at closeout.

Recorded-axis regression compared all58previously completed spectra: zero
changes in complete metrics/candidate fits;22current coordinates corrected.
All138tracked/new Python files parse and focused whitespace checks pass.
Additional Downloads slide text review: the Soyoung wiring deck concerns
a2024mechanical-switch sample, Measurement Update_2025-04-02 concerns TI
devices, and PNAX Pulsed Measurements is a general instrumentation setup.
None supplies PUCQ4 physical-Q5-to-resonator routing.

Bridge first trace accepted M2/-5.8mA/.015gain:
5439.870761686+/-.003571718MHz,FWHM.160788421+/-.013398510MHz,
SNR8.5371,16.094samples/FWHM,cleanreference. analysis_p1 preserves this
partial result; lower anchor and both inward windows still acquiring.

Prepared, NOT launched: central_highpower_0914/ac_ctrhi, conditional on full
bridge review and parked verification. M2 at exactly-5.656600mA source request,
matching the recorded ac_cf2 setting. Lower5430.7-5432.7MHz101points first,
thenupper5442-5448MHz301points, both32000reps/.03gain20us100us/equalrefs,
paired_windows_id=central_high,5400s+180guard. Expected54minutes plusoverhead.
Priorac_cf2 at.03/16000 recovered the lower positivecontrol, while ac_gap
upper search usedonly4000averages/50kHzbins. This tests8xaveraging/20kHzbins
at demonstrated power after lowpower lacked bothcontrols. Conditionalupper
window remains uncalibrated; bridge results must support it before launch.
If no upperline is recovered despite a clean accepted lowercontrol, retain
a visibility-limited result rather than launch a blind multi-bias map.
If a central pair is recovered, follow it with measured-window tracking to
obtain >=7paired biases and bracket a minimum. Same original hardware
programs/tracker/source limits/nominalattenuations/zero-offcleanup.

Report now includes signed-IQ panels for accepted segmented pairs, preserving
separate actual axes and unmeasured gaps. NativePowerPoint pair/current-axis
slides checked readable after limiting crowded x-axis tick labels. No data
normalization or line-fit changes were introduced for this visualization.

Bridge anchor complete: bothM2 windows pass, .015gain at-5.8mA.
Lower5402.776076496+/-.005000108MHz,FWHM.430022323+/-.023329835MHz,
SNR13.0436,43.042samples/FWHM; upper5439.870761686+/-.003571718MHz,
FWHM.160788421MHz,SNR8.5371. Bothreferencesclean. The repeated samebias
pair is accepted with identicalactualwindows/contiguousvisit checks, but
does not add a new distinct bias. Inward-5.74mA windows nowacquiring.

Completed16-run evidence checkpoint passed:178canonicalHDF5 files,
2.706736445GiB,116expected acquisitionplots, allsaved launch-source hashes
and protected executable sources match; eachguard exit0 and bothsources
parked. Artifact: ac_review/resolution_20260914/evidence_checkpoint_16runs.json.
This excludes the stillactive bridge run and does not replace its finalaudit.

ac_bridge completed exit0/1577.359s, bothsources verifiedCURR/.01A/0/off,
no recentPythoncontrollers remained before nextlaunch. InwardM2/-5.74mA
lower ACCEPTED5416.207715719+/-.003495638MHz,FWHM.365257099+/-.015255157MHz,
SNR15.5462,36.560samples/FWHM,cleanreference. Uppercandidate5440.544543328
MHz failsSNR4.6535only. Therefore-5.74 is notyet anacceptedpair.

bridge_confirm_0914/ac_bcf launched September14~17:02:46,guard50752;
runtime supplieschild/run. RepeatsM2/-5.74mA upper5439.5-5441.5MHz then
lower5415.2-5417.2MHz, both101points/32000reps/.015gain20us100us/equalrefs,
paired_windows_id=bridge_confirm,3600s+180guard. Fourfoldaveraging while
retaininggain/bounds;20kHzbins still resolve upperwidth with~9samples.
Fullpilotreview andparkedchecks completed; fresh plan safety/rawroundtrip pass.
Central_highpower_0914 remains PREPARED, unlaunched, pending thisconfirmation.

ac_bcf upper confirmation ACCEPTED at-5.74mA/.015gain32000reps:
5440.546943643+/-.004619457MHz,FWHM.154880317+/-.016123992MHz,
SNR11.7934,7.751samples/FWHM,residual1.1738,cleanreference.
The earlier8000rep candidate remains rejected. Fullmatched pair awaits
lowerrepeat, currentlyactive. Evidence ac_bcf/2026-09-14_17-02-42/
documentation/analysis_p1. Guard50752/child48824. This confirmation directly
addresses the earlierSNR-only failure without changingpower orbounds.

The confirmation is complete (guard exit 0, 1687.813 s). Both sources were
verified in CURR mode, 0.01 A range, at zero with outputs off; no recent Python
controller remained before the next launch. M2 at -5.74 mA now has an accepted
matched-window pair at gain 0.015: lower 5416.233092832 +/- 0.002925923 MHz,
FWHM 0.370590245 +/- 0.012796654 MHz, SNR 27.9711; upper
5440.546943643 +/- 0.004619457 MHz, FWHM 0.154880317 MHz, SNR 11.7934.
Both references are clean. Separation is 24.313850811 +/- 0.005468127 MHz,
compared with 37.094685190 +/- 0.006144774 MHz at -5.8 mA at the same gain.
There are now TWO distinct paired biases, still insufficient for J/minimum.
An extra broad lower-window candidate hit its width bound and remains rejected;
the signed-IQ pair figure was inspected and shows the two accepted lines clearly.

The lower center moved +0.025377113 MHz from the earlier -5.74 mA visit,
exceeding the combined conditional fit errors. Do not attribute this to power
(gain was unchanged), source-current accuracy, or a specific mechanism without
evidence. Any later multi-bias model needs an explicit temporal-stability check
and must retain this systematic limitation, not merely report fit errors.

Central_highpower_0914/ac_ctrhi has now launched, guard 13892; read runtime
for child/run. Fresh full confirmation review, parked-source checks and offline
plan/raw-roundtrip validation passed. The two same-power detuned pairs give
a CONDITIONAL upper locator of 5444.564721 MHz at the earlier central lower
line, supporting the unchanged 5442-5448 MHz search window. This assumes a
constant fixed mode and is not an accepted model, bare frequency or J.
Evidence: resolution_20260914/central_locator_after_confirmation.json.
M2 at -5.656600 mA, gain 0.03, 32000 averages: lower 5430.7-5432.7 MHz
(101 points) then upper 5442-5448 MHz (301 points), 20 kHz spacing,
20 us pulse/100 us relaxation/equal references. Deadline 5400 s + 180 s guard,
all original source/program/tracker/attenuation controls and zero/off cleanup.

Prepared, not launched: right_visibility_0914/ac_right. After full central-run
review and verified parking, test the opposite side at -5.59 mA through M2.
The measured lower control was strong in ac_fm2 (5436.328535938 MHz,
FWHM 0.176345738 MHz, SNR 16.2878 at 4000 reps/gain 0.03). The earlier upper
scan ac_up1 used only 4000 reps/50 kHz bins and found no candidate. Repeating
upper 5452.5-5458.5 MHz with 301 points/32000 reps tests eightfold averaging
and 20 kHz sampling inside that previously measured coverage. Lower control
5435.2-5437.6 MHz uses 121 points/8000 reps; both gain 0.03/20 us/100 us,
equal off/on reps within each window, tag right_visibility. Different averaging
between windows is explicit; their power/timing/source setting are matched.
Expected about 44 minutes; deadline 4500 s + 180 s guard. All original hardware
and zero/off controls retained. The two detuned pairs and measured right-side
lower frequency give a conditional upper locator of 5455.606891 MHz, not an
accepted model/J/calibration. This is a specific sensitivity test, not a blind
map. If upper visibility fails both centrally and here despite accepted lower
controls, retain a visibility-limited outcome rather than launch repeated maps.
If both sides supply accepted pairs, use measured windows and explicit stability
controls to guide the required seven-bias bracketed test. Plan validated offline.

Central high-power positive control ACCEPTED: M2 at -5.656600 mA, gain 0.03,
32000 averages, lower 5431.727524559 +/- 0.005139544 MHz; FWHM
0.141231591 +/- 0.017672403 MHz, SNR 6.7414, 7.068 samples/FWHM,
residual 0.7869 noise units, clean reference. Evidence ac_ctrhi/
2026-09-14_17-32-32/documentation/analysis_p1. The upper 5442-5448 MHz
window is now acquiring (about 40 minutes for its off/on pair). Unlike the
gain-0.007 test, this protocol has recovered the known lower control.
Current guard13892/child60576 remain active; right_visibility remains prepared.

The completed18-run audit passed: 193 canonical HDF5 files (3.077016354 GiB),
125 expected acquisition plots, matching saved/protected source hashes,
and all archived guards exit0 with both sources zero/off. Evidence:
resolution_20260914/evidence_checkpoint_18runs.json. The active central run
is excluded and still needs its terminal review/audit; do not treat this
checkpoint as current hardware cleanup.

Report layout review: the expanded temporal-limits paragraph brought one
three-paragraph findings page within 1.6 pt of the footer. Findings now use
two paragraphs per page; native text bottoms are at most 370 pt versus the
507.6 pt footer. The densest page was visually checked and is readable.
Current unpublished draft: TEMP/pucq4_cross_20260909/confirmed_progress_v2.pptx
(131 slides from the 18 completed runs). Final findings/results/audit still
must replace pending text after the active central and conditional right-side
tests. The published 94-slide deck remains unchanged until final review.

Readout-specific limitation and conditional cross-check: a central M2 null
does not establish absence through M6. In ac_cf2, the M6 central lower control
had SNR 8.2757 at 16000 averages versus M2 SNR 5.568. Prepared/unlaunched
central_m6_0914/ac_m6hi therefore tests the same -5.656600 mA/gain0.03
windows through M6 if the current M2 upper line remains unaccepted despite
the accepted lower control. Lower5430.7-5432.7 MHz/101points/16000 reps,
upper5442-5448 MHz/301points/32000 reps, tagcentral_m6,20us100us/equal
off/on within each window. Expected about50minutes,5400s+180guard,
original source/program/tracker/attenuation controls and zero/off cleanup.
This is an alternative readout on shared DAC1, not a new spatial drive.
Plan safety/raw-roundtrip validation passed; full M2 review and parked checks
are mandatory before launch. If M2 upper is accepted, skip this conditional
cross-check and use the measured result to guide the opposite-side test.

Central M2 high-power run completed exit0/3300.375s; both sources verified
CURR/.01A/zero/off, and no recent Python controller remained. Upper candidate
5444.489811332 +/- .019892234 MHz has FWHM .428826934 +/- .092132496 MHz,
SNR4.200675,21.461samples/FWHM,residual.954469,rank2/noactivebounds,
cleanreference. It fails SNR only and remains rejected. The raw signed-IQ plot
was inspected; the weak feature supplies a locator, not an accepted transition.
Lower control remains accepted; current pair/minimum/J remains unresolved.

M6 plan revised BEFORE launch using this measured locator: upper window
5442-5448 MHz/301points ->5443.2-5445.8 MHz/131points, retaining20kHz bins,
.03gain/32000reps/20us100us and about3FWHM baseline margin eachside.
Lower remains5430.7-5432.7 MHz/101points/16000reps. Deadline5400->3600s
(+180s guard), expected~26minutes. No launched plan/raw data changed.
Revised plan passed offline safety/raw-roundtrip validation and launched
after full M2 review/parking. Read central_m6_0914 guard/runtime for current
process and run state. right_visibility_0914 remains prepared/unlaunched.

M6 run is ac_m6hi/2026-09-14_18-29-53, guard58580/child18260. Its lower
control is ACCEPTED at -5.656600 mA/.03/16000: 5431.697853016 +/-
.005152324 MHz, FWHM .162832365 +/- .017945057 MHz, SNR8.7590,
8.149samples/FWHM,residual.943892,cleanreference. This is stronger than
the M2 central control despite fewer averages. The approximately -29.672 kHz
center difference from the earlier M2 visit combines readout and temporal
differences; no cause is assigned. Upper5443.2-5445.8 MHz is now acquiring.

Analysis now preserves window_requests and window_source_record_indices
for segmented spectra; each panel also labels its own averages and frequency
sample count. This prevents the inherited first-window request from hiding
different averaging in the second window. Mixed-averaging M6 provenance,
copy-independence and all scientific regression checks pass. Latest completed
19-run/70-spectrum analysis is resolution_20260914/central_m2_checkpoint/.

## September 15 renewed investigation: offline evidence and M6 coverage decision

Central raw repetition quarters reanalyzed with unchanged detector/fitter in
ac_review/renewal_20260915/{central_m2_blocks,central_m6_blocks}/block_analysis.json.
All 12 upper quarters are unaccepted. M2 lower accepted 2/4 (ctrhi), 1/4 (cmax);
M6 lower accepted 4/4. Full-average normalization reproduced at every frequency.
M2 saved tone8951.210MHz/gain.56/5us and tracked minimum8951.66MHz match
ac_ctrhi and ac_cmax; a recorded readout-setting change does not explain their
sensitivity discrepancy. M6 lower quarter centers span24.946kHz; shared full
references correlate errors, so no independent drift significance/cause assigned.

Fresh right_m6_renewal_0915.json (ac_r6new) replaces neither the old conditional
plan nor its failed prerequisite. It asks an independent readout-specific coverage
question at Y4-5.59mA, justified by earlier M6 lower SNR24.96 and only4000-average
upper coverage. The user explicitly renewed bounded execution. Same predeclared
lower121points/8000 and upper301points/32000 at gain.03,20us,100us; bounds
5435.2-5437.6 and5452.5-5458.5MHz. One round/equal off-on per window;
4500s+180guard, Y3zero/off,Y4.5mA/s/noextra hold/zero-off cleanup, original
programs/tracker/nominal attenuations. Fresh source preflight and offline check
required before launch. A failed upper with accepted lower ends this coverage
probe; an accepted pair needs independent confirmation, not immediate J promotion.
Scientific minimum/J, physical Q5 mapping, and independent drive remain open.

September15 08:43:57 local: fresh M6 test launched as guard29520/child61344,
ac_r6new/2026-09-15_08-43-58. Current process ownership rechecked; no recent
controller existed before launch, unrelated legacy processes preserved. Both
sources freshly verified expected serials/CURR/.01A/zero/off/30V. Longest
canonical HDF5 path197characters. All eight acquisition launch-source hashes
match ac_cmax; unchanged QICK programs/shared config. Plan safety/raw roundtrip,
138-file offline harness, scientific regression and whitespace checks passed.
Guard/runtime filenames are right_m6_renewal_0915.* in the documented TEMP folder.
This is an active-run checkpoint; do not infer terminal parking from preflight.

Additional offline review: sensitivity_v2/sensitivity.json under renewal_20260915
preserves six signed-IQ quarter/reference panels, exact configurations, times,
original metrics and descriptive fixed-shape amplitudes. Individually verifies
raw ON and OFF means against saved averages and quarter recombination. M2 upper
fitted delta amplitudes(I,Q), in ADC units, are(-.036955,-.037289) at32000 and
(-.028298,-.019660) at64000; noise(I,Q)=(.013049,.012019) and(.008764,.010740).
Both candidates remain rejected. Distinct fit windows/widths and correlated
full-reference quarters prevent interpreting these as an independent test of
a physical cause. Recorded M2 readout configuration is identical. Existing full
reference gates stay clean; no selective pooling or threshold modification.
Native diagnostic review found crowded lower-control ticks; regenerated into
fresh sensitivity_v2 with five major ticks, preserving first output. Direct M:
image viewing was denied; copied derived plots into IDE review workspace for
visual inspection. No raw files changed.

Additional mapping-source search: Documents/Desktop filenames for PUCQ, GDS,
OAS, DXF, routing, wirebond and packaging returned CandleQ_PUR1C1_CHIP_1 and
PUR1C2_CHIP_2 variants only. These names do not establish PUCQ4 provenance;
physical Q5 assignment still requires a PUCQ4 layout-to-resonator record or
uniquely identifying measurement. No independent spatial drive path documented.

September15 continuation after user interruption: existing guard29520 and
child61344 verified live with normal progress; no restart. Native supplement
layout review passed152/152renders/nooverflow; visually reviewed144,145,148,152.
The published143-slide report is unchanged and backed up as renewal_20260915/
prior_143slides.pptx. New nine-slide supplement is staged only; completed M6
result and final cleanup/audit are required before publication.
Today M6 lower at-5.59mA passes all four chronological quarters (8000totalreps);
right_m6_lower_blocks/block_analysis.json retains exact fits/raw paths.
Additional shared archive metadata search under M:/malab/People returned PUR
variants and unrelated 4Q design/packaging names, no PUCQ4 file match; no identity
promoted. M:/malab/_Data/PURCQ contains a cooldown7/Icarus characterization
folder with no visible contents. M:/malab_dropbox has no visible entries.
An initially ambiguous unlabeled directory listing placed People under the
wrong parent; corrected by inspecting FullName (People is under M:/malab).
These limited searches do not prove that no external PUCQ4 records exist.

September15 ~09:12 local: offline postprocessor PID7924 waits on kernel process handle for guard29520 exit, then requires complete/exit0/parked terminal metadata before running unchanged full23-manifest analyzer and raw audit. Helper/logs: TEMP/renewal_postprocess_0915.*; outputs renewal_20260915/full_analysis/analysis.json and raw_audit.json. Do not start a duplicate while this handle is live. Publication remains manual after scientific/native review. Upper off reference saved09:10:08, no detected candidates/vectorSNR2.1230; on acquisition nowactive. New lower quarter centers span8.136kHz, descriptive/shared-reference errors only.

## September 15, 09:32 local - M6 coverage probe complete

ac_r6new/2026-09-15_08-43-58 completed guardexit0 in2879.5s. Guard29520 and
child61344 are terminal and absent. Both guard/source records verify expected
serials,CURR,.01A,0A,outputoff,30V. No crossing controller remains active.
Lower5436.352749361+/-.001591724MHz,FWHM.183947718,SNR37.314898 accepted;
all4lowerquarters accepted, center span8.136253kHz (correlated reference errors).
Upper NEW REJECTED candidate5455.329437014+/-.023193449MHz,FWHM.612135478
+/-.106883441MHz,SNR4.262093,30.635samples/FWHM,residual.969192,fullrank2,
noactivebounds. SNR alone fails; both references clean. Visual raw review agrees
with a weak feature, not an accepted branch. The predeclared failed-upper stop
rule ends this coverage probe; no extra averaging or blind map launched.

Offline postprocessor7924 completed. Full23-run analysis has78spectra; all76old
metric objects reproduce exactly. Raw audit PASS218canonicalHDF5(4.611908GiB),
140expectedplots, all source hashes and guarded terminal states valid.
Evidence: ac_review/renewal_20260915/{full_analysis/analysis.json,raw_audit.json,
metric_preservation.json}. Shared configs and QICK programs unchanged.
Report built157slides in renewal_20260915/PUCQ4_Q5_Q6_renewal_20260915.pptx;
native validation and publication remain pending at this checkpoint. Prior143
published deck backed up byte-for-byte before staging. Minimum/J still lacks
seven unique paired biases and bracket; physical Q5 and independent spatial drive
need PUCQ4 mapping/wiring evidence. Temporal/weak-feature cause is not assigned.

## September 15 publication and environment lessons

Published157slides at09:38local; SHA256
DBA2E7CA4AC8E4509523C9DD0B9831A87BC9AAFD43D9B0A881CD9E872634E7D6.
Native157renders/noissues; visually reviewed1,6,15,55,146,147,148,150,157.
Publication receipt verifies reviewed/published bytes and both prior143 backups.
Scientific stop disposition and remaining prerequisites are in the refreshed
handoff and PUCQ4 state. No acquisition remains active.

Environment: PowerShell5 File.Replace with null backup produced an invalid-path
exception after staging; original published hash was unchanged. Revalidated
source/pending/prior hashes and repeated with an explicit backup path; atomic
replacement and both hashes passed. Use ErrorActionPreference=Stop for publication.

Encoding: PowerShell5 Get-Content defaults to Windows-1252. A whole-file state
rewrite introduced mojibake. Verified the entire historical tail exactly matched
the old source snapshot decoded as Windows-1252, then restored only that proven
encoding transform from the snapshot's UTF-8 text. No historical user content was
lost. Future whole-file edits use IO.File.ReadAllText and explicit UTF-8 writes.