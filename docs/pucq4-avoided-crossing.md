# Physical Q5-Q6 avoided crossing

The September14 follow-up is closed and the **143-slide** report is published
at `pucq4_first_light/PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx`.
Final audit passes213HDF5/137expectedplots across22 completed guarded plans;
native143/143slides render with no detected overflow. Both sources ended0A/off.
Two distinct detuned paired biases are accepted; minimum splitting/J and
physicalQ5 identity remain unresolved. See the resolution plan for exact
measurements, remaining prerequisites, report hash and evidence locations.

## September14 continuation outcome (20:35 local)

The bounded follow-up measurements are complete:22 total guarded plans,
76 drive-off/on spectra,49 accepted single-line windows and27 without an
accepted line. Three matched pair observations cover two distinct biases:
-5.8mA atgain.007 and.015, and-5.74mA atgain.015. Same-.015gain separations
are37.094685+/-.006145MHz and24.313851+/-.005468MHz respectively.
These are measured separations at the stated biases, not minimum splitting/J.

Central M2's64000-average/.03gain confirmation still fails SNR3.7118;
lower control passes. Central M6 and higher-averaging opposite-side M2 tests
also provide no accepted upper line. References are clean. Conditional
right_m6 was not launched because its required central pair was not established.
The seven-bias bracketed-pair task, coupling/minimum/resonance and physicalQ5
mapping remain unresolved. No independent spatial drive is documented here.
Both current sources were verified CURR/.01A/0A/output-off at final cleanup;
no acquisition controller remains. Full evidence and final report completion
are recorded in [the resolution plan](pucq4-crossing-resolution-plan.md).
Earlier ACTIVE/PREPARED/report counts below are historical.

## Reopened continuation — September 14, 15:55 local

User requested a plan, autonomous resolution work, and an updated PowerPoint.
Current plan and checkpoints: [pucq4-crossing-resolution-plan.md](pucq4-crossing-resolution-plan.md).
New common-power ac_pair1 measurement accepts a line pair atY4-5.8mA:
M2 lower5402.791626+/-.007499MHz,upper5439.862749+/-.003573MHz,
separation37.071123+/-.008306MHz. Both readouts measured identical disjoint
windows; M6lower agrees, M6upper has no accepted line, allreferencespass.
One paired bias does not determine J/minimum/resonance or physicalQ5identity.
ACTIVE central_lowpower_0914/ac_ctrlo guard16720/child9888,run15-55-29:
bothreadouts/.007gain/8000reps at-5.656667mA, upper5442-5448MHz andlower
5430.7-5432.7MHz, originalprograms/guards/zero-offcleanup. Finalreportpending.
The published94slide report is preserved;110slide draft is in TEMP only.
The next section is the previous closeout, retained as historical evidence.

## Final result — September 14, 2026

Measurement and reporting are complete, with unresolved coupling. Eleven
guarded runs saved49paired drive-off/on traces:33have one accepted line,
16have no accepted line, and none resolves a doublet. The seven-bias fine
map spans-5.67 to-5.59mA and shows a line rising from about5429.8 to5436.33MHz
through M6 and M2. Both wider searches found no additional branch in their
acquired windows. Final localized confirmation at-5.656667mA,10kHzsteps/
16000averages, accepts M6 5431.742654+/-.005062MHz (FWHM.193551+/-.022556MHz)
and M2 5431.757418+/-.007018MHz (FWHM.220711+/-.033103MHz). Both references
are clean. The lines overlap and cannot be combined into two modes.

No resonance bias, minimum splitting, J, or physical Q5 readout assignment is
accepted. Constant bare fixed-qubit frequency is not independently identifiable
from these single-line spectra. The same-bias/.03gain central line moved about
.21MHz between September10and14; conditional fit errors exclude temporal
variation and source-current accuracy. No zero-coupling or entangled-state
claim is made. Power-check widths and exact settings are in the report.

Final94-slide report is
`M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/pucq4_first_light/PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx`.
All94slides reopened and exported in native PowerPoint; findings, tables,
fine maps, branch-resolution summary and cleanup slides were visually checked.
Report SHA256:1885B880D5B785DBCA4DB706D627BD0F9EFCEB764305786770084C26FF676496.
Final analysis, raw checksums, report code and evidence audit are under
`pucq4_first_light/ac_review/final_20260914/`. Audit passed143HDF5 files,
94expected acquisition plots,2.029700GiB raw, all11archived guards/cleanups,
seven protected source files and the unchanged original Downloads deck.

Final run confirm_m2_0914 ended exit0 in911.594s; both sources verified
CURR/.01Arange/0A/outputoff. No recent Python acquisition/guard process remains;
legacy processes were preserved. No follow-up is queued. No shared calibration
promotion or acquisition-program change was made during this resumption.
Offline harness138files, scientific regression, canonical IQ persistence and
diff checks pass. Historical plans/checkpoints below are retained as provenance.

September14 12:24local: fine map completed exit0 in1973.203s,14pairedtraces/
42HDF5, both sources0A/off verified. Eleven accepted single-line traces,
three unresolved; no doublet. The observed line spans about5429.8-5436.33MHz
over-5.67 to-5.59mA. At-5.656667mA the fine M6 candidate fails SNR4.052;
M2 has none. At-5.643333mA M6 accepted5433.215602MHz while M2 candidate
fails SNR4.280. Both readouts recover strong lines near5435.25MHz at
-5.616667mA and5436.33MHz at-5.59mA. Inter-day central-bias shift~.21MHz
precludes treating conditional fit errors as total stability errors.
gap_m2_0914 then searched5428-5458MHz at-5.656667mA,601points4000reps/.03
onM6/M2. Completed exit0 in1317.219s, both sources0A/off. M6 only candidate
5431.706327MHz/.082930MHzFWHM/SNR6.754 fails sampling(1.659samples/FWHM).
M2 no candidate, vector score2.348, clean references. Both raw plots inspected.
Final localized confirm_m2_0914/ac_cf2 launched12:24:30 guard58892/child54820,
run2026-09-14_12-24-31, after terminal cleanup and fresh preflight. Bothreadouts
5431.2-5432.2MHz101points16000reps/.03gain/equalrefs at-5.656667mA;
10kHzgrid,20us100us1round/originalROtracking/nominal10/15,5/4,17dB,
Y3zero/off,Y4.5mA/s/noextraDCsettle/zero-offcleanup,2500s+180guard,
expected15min/maxpath195chars/abortsource-acquisition-saveerrors.
This confirms the weak observed line; it does not assign a missing branch.
Afterward finalize analysis, audit, report/native render and parked closeout.

September14 11:28local: upper1/ac_up1 completed exit0 in1317.188s, both sources
CURR/10mArange/0A/off verified, guard and child gone. Both 5450-5480MHz traces
are unresolved, vector scoresM6 2.756795/M2 2.795433, no candidates and clean
references. Both raw plots inspected. No absent-branch or zero-coupling claim.
After review and fresh source preflight, fine_m2_0914/ac_fm2 launched11:28:17,
guard31788/child13704, run2026-09-14_11-28-18. Seven biases -5.67,-5.656667,
-5.643333,-5.63,-5.616667,-5.603333,-5.59mA (central bias first), M6 and M2
readouts each. Windows are 2.4MHz wide, around observed/interpolated locator
positions 5430.2,5431.9,5433.4,5434.6,5435.4,5436.0,5436.4MHz; these are
search bounds, not calibrated intermediate frequencies.121points/4000reps/
.03gain/equaloff-on,20us100us1round,originalROtracking and nominal attenuation
10/15,5/4,17dB unchanged;Y3zero/off,Y4.5mA/s/noextraDCsettle/zero-offcleanup.
Deadline5000s+180guard,expected37minutes,maxpath195chars,abortsource/
acquisition/path/savefailure. The narrower windows resolve the observed line
versus bias but cannot exclude a mode outside coverage or establish J from
one branch. The first central pair tests return of the feature after four days.

September 14 resume: verified map_m2_1 terminal exit 0, elapsed 3530.718 s,
both sources recorded 0 A/off at September 10 02:14. Full eight-point analysis
saved as ac_m2a/2026-09-10_01-15-13/documentation/analysis_complete_0914.
At -5.63 mA/.015 gain M6 center 5434.597693455 +/- .001332508 MHz,
FWHM .107013465 +/- .004355700 MHz, SNR17.248; M2 center
5434.595833517 +/- .001648692 MHz, FWHM .113181607 +/- .005434436 MHz,
SNR16.981. Both references clean and both single lines accepted. Coincident
centers do not establish two modes; conditional errors exclude temporal drift.
Scientific regression checks pass. User requested continued autonomous work.
No Python process created since September 1 was active before launch.
Prepared upper1 plan launched September 14 11:06:01 local, guard61368,
child47680, run ac_up1/2026-09-14_11-06-02 after fresh source preflight.
Exact bounds and settings remain those recorded below for upper1. No shared
config or acquisition-program edits. Fine bias map and final report pending.

User authorized autonomous acquisition and a final PowerPoint on September 9,
2026. Physical Q5/Q6 mean the labels in Downloads/PUCQ4 Initial
Characterization (1).pptx. Physical Q6 is M6/index5/Yoko4 (192.168.1.77/DC A5).
The fixed physical Q5 readout row remains unassigned in that deck. M1-M4 are
the fixed candidates; displayed Q5/M5 is physical Q4 and must not be substituted.
Identify the coupled row experimentally, retaining uncertainty in physical
assignment unless evidence uniquely establishes it.

Scope: localize Q6, verify readout, acquire coarse and fine bias/frequency maps
with drive-off references, repeat at lower power, fit resolved coupled branches
with uncertainty, inspect constant fixed-frequency assumption, save all raw
complex data/configuration and put final plots in a PowerPoint. An avoided
crossing does not establish entangled-state preparation. No shared calibration
promotion or unrelated experiment changes are intended.

Hardware and protocol: shared qubit DAC1, readout DAC0/ADC0; sequential single
readout tones. There is no separate spatial Q5 drive port in this configuration.
Reuse PulseProbeSpectroscopyProgram and SingleToneSpectroscopyProgram without
modification. Yoko4 hard boundary +/-10mA; current range10mA; normal ramp0.5mA/s;
parking0A/output off. Yoko3 stays0A/output off. The established Yoko4 protocol
adds no extra DC settling delay near -5mA; resonator tracking precedes qspec,
which retains100us relaxation. Initial current readbacks: both CURR/.01A/0A/off,
30V compliance, serials Y3 90Z114063 and Y4 90Z114062.

Calibration: system_config.py PUCQ4_FLUX_PULSE_CALIBRATION stores Q6
4839.20659511MHz at -7.5mA,5588.10371144MHz at -5mA,6103.73104802091MHz
at -2.5mA. First two retain September7/8 evidence; -2.5 was revalidated
September9. Fixed M1-M4 executable frequencies are5517.69162843,
5438.71088705,5557.33568830,5594.54190290MHz. These locate a diagnostic
survey, not an assigned physical Q5 or a measured resonance bias.

The new controller pucq4_13_avoided_crossing.py accepts explicitly reviewed
JSON plans only within -5.8 to -4.8mA and5300-5660MHz. Actual bias is recorded
separately from its -5mA configuration anchor; intermediate points are not
represented as validated calibration. Plans live in TEMP/pucq4_cross_20260909
and are copied to canonical first-light/<label>/<timestamp>/documentation.
Raw qspec_ge/res_ge datasets use create_data_dict/save_h5, with Dates, Round
Num, Batch Num, Exp Config and Syst Config plus actual bias and raw IQ.
Each drive-off and drive-on record saves independently, before proceeding.

First pilot: locator1.json/ac_loc1, Q6 atY4-5mA. Sweep5576.10371144 to
5600.10371144MHz,601points,3000reps,1round,.006DAC gain,20us pulse,100us
relaxation; equal drive-off reference. Initial readout9056.91MHz base with
-.28MHz offset,.8767gain,12us length. Res spec600reps/1round/120points at
.05MHz spacing,100us relaxation. Nominal attenuator constructor arguments
10/15,5/4,17dB are unchanged; their setter calls are disabled, so no fresh
attenuator readback is claimed. Maximum predicted HDF5 path196characters.
Deadline1800s plus180s independent guard grace; source0/off on exit. Abort
readback, acquisition, path or save failures. Hidden guardPID24504 launched
September9 22:21:06local; authoritative state is locator1.guard.json plus live
process identity, not this historical PID alone.

Offline canonical raw-IQ/reference roundtrip and current-envelope rejection
passed. Static harness passed132files before the guard/check helpers were
added; final checker/diff review remains required. No crossing result exists
at this checkpoint. Required acceptance is resolved branch curvature and
validated fit, or an explicit measurement limitation with no forced coupling
estimate; final rendered report and verified parking are mandatory.

First locator completed in536.062s with three canonical HDF5 files (resonator,
driveoff,driveon), both expected plots and independent zero/off cleanup. Q6
5587.968909939 +/- .015142285MHz, FWHM .419452536 +/- .071924358MHz,
SNR7.94449, residualRMS.92852noise,10.49frequency samples/FWHM. M6 resonance
9056.91MHz/readout9056.63MHz unchanged. Raw fit is single-line evidence only.

Coarse1/ac_c1 launched22:32:13local, guard33944/child50808. Current list
-4.995,-4.975,-4.955mA; M6 qspec5580-5608MHz/401/3000x1/.006/20us/100us,
matched driveoff. Middle current adds M4 same frequency window with
5000x1/.03gain as a higher-power visibility test. Deadline3000s+180guardgrace.
First point at-4.995 has accepted single-line center5589.072874252 +/-
.017332096MHz, FWHM.278529892 +/- .061196883MHz, SNR5.8624. Actual frequency
step .06999664MHz gives3.98samples/FWHM, so finer frequency spacing is required
for final branch inference. Acquiring middle bias at last observation; always
verify actual guard/process state before further hardware.
September 9, 23:00 local: ac_c1 completed all four paired qspec traces and four res-spec files; guard exit0, 1557.328s, independently verified both sources CURR/.01A/0A/off. Third Q6 point at-4.955mA is5598.045399082+/- .019326400MHz, FWHM.277508371+/- .067123608MHz, SNR5.7720. Candidate M4 at-4.975mA is5594.507302995+/- .014368989MHz, FWHM.833709725+/- .072826818MHz, SNR14.2418 at.03gain. M4 resonance8999.92MHz/readout8999.69MHz/.95/5us remained stable; all M6 tracked tones9056.63MHz. Only one line in each coarse readout, no accepted splitting or physicalQ5 assignment. Next bounded visibility1.json/ac_v1 tests masked splitting with lower power at local interpolation seed-4.9709mA (not a resonance estimate): M6 and M4 each5592.5-5596.5MHz201points, paired off/on,20us/100us/1round. M6 gains.003/16000reps and.006/4000reps; M4 gains.0075/16000reps and.03/4000reps. Same readout tracking/nominalattenuation; Y3zero/off,Y4.5mA/s,zeroextraDCsettle,4000sdeadline+180guard,zero/offcleanup,abortcurrent/acquisition/savefailure. Longest artifact194characters. Coarse controller is terminal before next launch.
Precision note: malab_yokogawa_gs200.get_current queries :SOURCE:LEVEL?, so archived current readbacks are the instrument source-level setting, not an independent ammeter measurement. Conditional spectroscopy/branch-fit errors do not include current-source accuracy or temporal drift. Prior zero-current M4 calibration (rr6_0mA/2026-09-09_19-23-29/documentation/validation_M4.json) had FWHM .534722MHz at gain.0075 and128000averages; broad linewidth can therefore persist at weak drive. This prior control is not a biased crossing measurement.
September9 23:18local visibility1 M6 higher-power paired trace complete. Gain.006/4000reps at same-4.9709mA and19.981384kHz grid: center5594.593315964+/- .011495885MHz, FWHM.411295728+/- .049908499MHz, SNR6.1005, residualRMS.86381. Single line at both powers. Lower/higher linewidth difference.119464+/- .064095MHz (~1.86SE); center difference.044369+/- .015486MHz. Sequential traces do not separate drive shifts from temporal drift; no cause or coupling inferred. M4 weak-drive trace is acquiring. Prepared but NOT launched candidates1/ac_cs1 to survey M3/M1/M2 fixed-row candidates with measured drive gains and grids resolving their prior widths; validated max path195characters. Current seeds-5.135/-5.305/-5.63mA within authorized survey envelope; inspect exact JSON and previous guard before any launch.
September9 23:31local visibility1 M4 weak-drive paired trace saved, but NOT accepted: candidate5594.615149MHz, FWHM.551332MHz, vectorSNR4.555 below preset5 gate. Raw/plot/reference retained; no accepted fixed-mode frequency or width from this trace and no zero-coupling inference. Drive-off control has no accepted feature. Same-bias gain.03/4000reps comparison is acquiring; guard60216/child17112 remains authoritative. Candidate survey will prioritize identifying a resolved partner; any later M4 weak-power validation needs more averaging justified by this measured SNR.
September9 23:37local visibility1 completed exit0 in2143.188s; independent guard verified both sourcesCURR/.01A/0A/off. Final M4 .03gain trace accepted center5594.606620780+/- .008942088MHz, FWHM.732032016+/- .041970152MHz, SNR13.0603. Its same-power center differs.113161MHz from coarse-4.975mA, so candidate survey candidates1 remains PREPARED, NOT LAUNCHED. Next biascheck1/ac_b1 is an A-B-A control at-4.975,-4.9709,-4.975mA, M4only201points5592.5-5596.5MHz4000reps.03gain20us100us1round, pairedoff/on. Both sourcesparkedpreflight; sameROtracking/.5mA/s/noextraDCsettle;2000sdeadline+180guard,zero/offafter,abortreadback/acquisition/savefailures. A reproducible bias contrast relative to the bracketing A visits would justify a local fine map; a drift-like result redirects to the other fixed candidates. No physical assignment or coupling estimate is accepted.
September9 23:51local ac_b1 complete exit0 in686.079s, bothsourcesCURR/.01A/0A/off independently verified and logs archived underdocumentation. Accepted A1/B/A2 centers5594.510200336+/- .008984081,5594.513145134+/- .008942393,5594.526079236+/- .008838655MHz. B minus time-interpolated A =-.004995777+/- .010939611MHz; A repeat difference.015878900+/- .012602997MHz. Stable repeat, bias contrast below.05MHz/3SE follow-up gate: earlier.113MHz displacement was not reproduced as a bias response. This does not establish zero coupling. Proceed with prepared candidates1/ac_cs1 (no prior guard): Y4-5.135M6/M3 5542.3-5572.3MHz; -5.305M6/M1 5502.7-5532.7MHz; -5.63M6/M2 5423.7-5453.7MHz. EachM6:601/3000/.006. Fixed M3:3001/600/.002; M1:3001/600/.001; M2:1501/1500/.007 (points/reps/gain), calibrated drives and linewidth-resolving grids. All20us100us1round, pairedreferences, originalres tracking; Y3zero/off,Y4.5mA/s/noextraDCsettle,6000sdeadline+180guard,park0/off,aborterrors. Max path195characters; full reviewed envelope in TEMP/pucq4_cross_20260909/candidates1.json. Expected duration about50minutes.
Chronological raw-shot review: ac_review/M4_time_blocks/block_analysis.json contains five M4 high-power traces split into four on-repetition blocks using a common full off reference. All20blockfits pass; the raw-to-normalized-IQ conversion is independently verified point by point (normalization2765). Coarse M4 line centers span5594.40895 to5594.57949MHz within one held source setting, and later controls show roughly.1MHz jumps within individual traces. The A-B-A whole-trace contrast remains-.0050+/-.0109MHz, but its small mean contrast does not imply absence of fast fluctuations. No mechanism or coupling is inferred. Both analysis source files and per-trace plots are saved with the review.
September10 00:30local: ac_cs1 M1 pair complete atY4-5.305mA. Accepted5517.685971109+/- .001137229MHz, FWHM .055872355+/- .003461784MHz,SNR17.1227,5.592samples/FWHM, clean driveoff. M6 same bias has no accepted line at.006gain within5502.7-5532.7MHz; its res tracking remained healthy (9056.81MHz resonance, selected9056.53MHz tone). No coupling/null inference. M2-region Q6 acquisition continues. Prepared fine_m3_1.json/ac_f3a but NOT launched: first wider/higher-power M6 locator5480-5560MHz801points1500reps.03gain at-5.305mA, then seven M3-region biases-5.1454,-5.1294,-5.1414,-5.1334,-5.1384,-5.1364,-5.1374mA. Both M6/M3 use common.006drive,5554.3-5560.3MHz; M6:301points4000reps, M3:601points1000reps. Last center adds.003power M6:201/16000,M3:601/3000. This frequency-match seed uses measured Q6/M3 lines, not an accepted resonance bias. All20us100us1round/equalrefs/originalROtracking; nominal10/15,5/4,17dB unchanged, Y3zero/off,Y4.5mA/s/noextraDCsettle/zero-offcleanup,10000sdeadline+180guard,abortreadback/acquisition/savefailure. Validation-only passed, longest canonical artifact195characters, expected~85minutes. Inspect M2 results and confirm terminal parked guard before launch; no overlapping controller. Static harness138files and diffcheck pass after narrow-line detail plot addition. Source setting readback remains distinct from an independent measured current.
September10 00:35local: Q6 M6 at-5.63mA has no accepted line over5423.7-5453.7MHz at.006gain/3000reps; smoothed projectedSNRmax2.15, noise .0434/.0409ADC units, clean reference. Resonator9056.71MHz tracked, tone9056.43MHz. M2 direct scan is underway. Prepared fine_m3_1 revised before launch to add M6 visibility locator at-5.63mA5400-5480MHz801points1500reps.03gain/equalrefs, following corresponding M1-region locator. Same bounded protocol, total~90minutes and max path195chars; validation-only passes. Single-line analyzer now explicitly rejects failed optimizers, rank-deficient fits and active parameter bounds, retaining diagnostic fields; synthetic known-mode/broad-line/drift controls pass. No physical Q5 assignment, branch splitting or null coupling is accepted.
September10 00:45local: ac_cs1 completed exit0 in3064.656s; both sourcesCURR/.01A/0A/off independently verified, no previous process remains. Six paired traces persisted. Final M2 at-5.63mA also has no accepted line at.007gain/1500reps over5423.7-5453.7MHz (projectedSNRmax2.65, noise.0610/.0608), clean reference. Its resonator is healthy8951.61MHz, selected8951.21MHz/.56/5us unchanged. Simultaneous loss of the two nominal lines within the limited spectral window makes a wider M2 search a priority; this is a visibility clue, not coupling evidence. Defer fine_m3_1/ac_f3a (NOT launched). Next visibility2/ac_v2: -5.63mA M6 and M2 both5380-5500MHz2401points1000reps.03sharedgain; then-5.305mA M6 5480-5560MHz801points1500reps.03gain. Gain rises from.006(M6)/.007(M2) to.03 for localization within prior envelope, not shared calibration. All20us100us1round/equaloff/on, originalROtracking and nominal10/15,5/4,17dB;Y3zero/off,Y4.5mA/s/noextraDCsettle/zero-offcleanup,5000sdeadline+180guard;abortreadback/acquisition/path/saveerrors. Max canonical artifact194characters;expected26minutes. The exact plan is TEMP/pucq4_cross_20260909/visibility2.json and validation-only passed. Latest draft PowerPoint40slides has native-rendered readable drift/detail plots; it remains explicitly in progress.
September10 00:52local: visibility2/ac_v2 guard7192/child7408 active, run ac_v2/2026-09-10_00-45-39; no parallel controller. Offline analyzer now supports evidence from sequential M6/fixed-row readouts at a common drive gain/pulse. It requires unique visits, clean references, common frequency coverage and separation beyond linewidth/center error; coincident lines are never counted twice, conflicting doublets and additional unmatched lines are rejected. A resolved doublet in one readout is preferred over combining independent singles. Paired readout timestamps/selection/raw paths are retained. Existing seven-bias/interior-minimum/J-significance/fixed-frequency tests remain mandatory, and temporal shifts between readouts remain outside conditional fit errors. Synthetic known-J/duplicate/mixed-power/repeated-visit/conflicting-doublet checks pass. Accepted branch fits will include measured centers/error bars and fitted coupled/bare-mode curves in the report; none has resolved in measured data yet. No acquisition program or shared device calibration changed.
September 10, 01:03 local: corrected a verification-feedback failure in the offline peak detector. The ac_v2 M6 wide scan at -5.63 mA/.03 gain has a real repeatable IQ feature near sampled bin 5434.644157 MHz. Four chronological on quarters give peak I 0.8554, 0.5494, 0.6979, 0.5662 ADC units with conditional SE about 0.124-0.135; all use the full off reference. Raw normalization 6636 is verified against saved averaged IQ. A first-global-PC detector missed the feature because the sparse line did not dominate full-scan covariance. Detection now uses both smoothed/noise-scaled components and still fits signed IQ. Vector score 9.18 versus old projection 3.00; candidate center 5434.643611 +/- .006728 MHz, FWHM .138036 +/- .020473 MHz, SNR 10.214, only 2.761 frequency samples/FWHM: NOT ACCEPTED pending a finer grid. All earlier task traces were reanalyzed; no additional accepted lines, no splitting or physical assignment. Updated estimates/source hashes are in ac_review/progress_0102/analysis.json; previous snapshots remain intact. Regression and existing scientific checks pass. The detector failure and repair are recorded in FAILURE_MODES.md. This was an analysis correction; acquisition code and raw data are unchanged. M2 wide drive-on is running under guard7192/child7408. Fine M3 remains deferred; the newly localized feature requires an evidence-based finer scan.
September 10, 01:14 local: ac_v2 completed exit0 in 1647.672 s; both Yokogawas independently verified CURR/10 mA range/0 A/output off, previous processes gone. M6 and M2 show the same localized feature near 5434.65 MHz at -5.63 mA/.03 gain, but both candidates fail the frequency-sampling gate (2.76 and 2.99 samples/FWHM). The wider M1-region M6 scan at -5.305 mA/.03 gain,5480-5560 MHz with100 kHz steps, still has no resolved line (vector score3.30); its limited sampling/sensitivity does not exclude a narrow line or coupling. Next bounded map_m2_1/ac_m2a scans Y4 -5.67,-5.59,-5.63 mA, each with M6 and M2 readouts at shared gain.03 over5420-5455 MHz,1401points1000reps (25 kHz steps). The final bias adds lower gain.015 over5433.6-5435.6 MHz,201points: M6 8000reps, M2 16000reps. All20 us pulses/100 us relaxation/1round/equal off-on references, original programs and resonator tracking; nominal attenuations10/15,5/4,17 dB unchanged. Y3 stays0/off;Y4 ramps.5 mA/s with established zero extra settling and returns0/off. Deadline7000 s plus180 s guard; expected57minutes; abort readback/acquisition/path/save failure. Validation-only passed; longest canonical artifact195 characters. The plan uses observed candidates as scan seeds, not accepted calibration, splitting, or physical identity. Fine M3 remains unlaunched/deferred. Static harness138files and focused diffcheck pass; source-projection regression and scientific checks pass after the detector correction.
September 10, 01:30 local: ac_m2a first bias (-5.67 mA) is saved on both readouts. M6 candidate5430.250887MHz/FWHM.311885MHz/SNR4.547 and M2 candidate5430.128080MHz/FWHM.204398MHz/SNR3.891 both fail the SNR5 gate despite adequate grid sampling. Neither center/linewidth is accepted, and the provisional readout discrepancy is not combined into a mode pair. References are clean. Positive-bias (-5.59 mA) scans are underway. This map will be followed by its programmed lower-drive comparison. A literature check (Filipp et al., PRA83,063827; https://arxiv.org/abs/1011.3732, Secs.V-VI) confirms that shared-drive dark branches and additional two-photon transitions can complicate spectra; these remain hypotheses here, not identified PUCQ4 mechanisms. No coupling/physical assignment is accepted. At01:22 all61 HDF5 files and40 expected plots from the six completed plans existed (1.049GiB raw HDF5); all seven protected configuration/program files and the original Downloads deck hash were unchanged. The latest draft has50 native-rendered slides; a subsequent build will include the concise interpretation/reference slide. M4 reanalysis with the corrected detector retains all20 accepted time-block fits and up to.166MHz within-trace center span (ac_review/M4_time_blocks_v2).
September 10, 01:42 local: ac_m2a at -5.59 mA now has one accepted line on each readout. M6:5436.399576929 +/- .004762979 MHz, FWHM .212030102 +/- .017851823 MHz, SNR12.6627,8.486samples/FWHM. M2:5436.409615797 +/- .005204478 MHz, FWHM .156549213 +/- .018157786 MHz,SNR10.0451,6.265samples/FWHM. Both references are clean; centers agree within conditional errors, and the coincident lines are not counted as a doublet. Central-bias high/low-drive scans continue. Prepared upper1/ac_up1 (NOT LAUNCHED) to extend coverage at -5.59 mA above the present5455MHz ceiling: both M6/M2,5450-5480MHz601points4000reps.03gain,20us100us1round/equalrefs. The upper-frequency search is motivated by the observed lower-frequency feature and the fixed-row seed, not an accepted single-branch estimate of J. Fifty-kHz sampling is a locator; narrower candidates require refinement. Same res tracking, nominal attenuations, Y3zero/off,Y4.5mA/s/noextraDCsettle/zero-offcleanup;4000sdeadline+180guard,aborterrors;22minutes expected and195character maximum HDF5 path. Validation-only passed. Review the remaining map results and verify terminal parked guard before this next launch. No accepted exchange splitting or physical Q5 assignment exists.
