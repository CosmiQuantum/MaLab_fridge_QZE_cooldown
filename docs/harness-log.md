# Harness task and diagnostic log

### 2026-11-04 17:25 - Q2 24h series started and verified

Requested outcome achieved: actual ss-only round-robin tracking running in
October7ROOT/PUCQ4/find_resonators_sandbox/ssf_24h_q2/2026-11-04_17-24-58.
Start1793831099.510199/end1793917499.510199,86400seconds;2000shots/state,
100usrelax,save_r1, fixedpulse, normalHDF5hierarchy/schema andplots eachround.
Supervisor35884/waiter33076/tracker54804/RR63200. BothYokos CURR0A/OFF/
.01A/30V verified atlaunch. No otherownedhardwarequeue active. Graceful
documentation/STOP aftersavedround;600sno-save and24h+15min watchdogs.

Readout plateau passed after4cycles(3-budget extendedboundedto5), balanced
selected/lowergain repeats chose.192DAC/7us/8949.920077MHz. Two independent
20kexamplescontrast.08155/.09935 retained with timevariation qualifier.
Final16kRabi/20kSSFnormalRR17-19-33 passed R2.9939664269650141,
pi.6765549655130318 andSSF.11905595279763988. Freshcarrier
5438.499681577915MHz,sigma.101us. Qualifiedcap.24 belowstable.30/first
resolved.60; no bare-frequency/photon-number or microscopicTLS claim.

Q2-onlysharedconfigpromotion completed with backups and pure-selector/
unchanged-five-row/unchanged-flux-table checks. ActualnormalRRlow-powerres
reference8950.93439MHz promoted, not complex-notch diagnostic replacement;
absolute selectedRO retained via recomputedoffset. Old->new index1 values:
res8950.85->8950.93439MHz,gain.95->.192DAC,offset-1.12->-1.014313MHz,
abstone8949.73->8949.920077MHz,qubit5438.515064891349->5438.499681577915MHz,
pi.666605627784899->.6765549655130318DAC,capNone->.24DAC.
Length7us,sigma.101us,qspecgain.009/pulse100us unchanged. Exactmanifest
config_changes.json archived under trackingdocumentation.148fileharness and
gitdiff-check passed beforelaunch;otherrows/nonzerofluxhistory preserved.

verify_tracking.py inspected realround34: all4rawIQarrays finite/2000samples,
metadata andbatch/roundnumbering correct, normalPNG exists/decodes, fixed
settings match plan,scheduledduration86400s,promoted-source snapshot hashes
match. Archived first_tracking_verification.json,q2_ready,powerqualification,
optimization,configmanifest andsupplemental doubletcomparisonPNG underactual
run documentation. No generatedraw data committed. Series RUNNING,not24h
complete. Remainingall-sixbenchmark/decksdeferred; don't resumeoldqueue
with stale sourcehashes/0.10Q2policy duringtracking.

### 2026-11-04 16:47 - correct Q2 visibility-floor/readout-ceiling conflation

17:15: four fresh power points accepted as recorded above;0.24 readout cap
qualified with low-return stability. Spectrum confirmation at0.24 independently
repeats doublet5438.379611/5438.512979MHz, split133.368+/-6.2kHz, R2.817/noisy
qualification retained. Original Rabi quality PASSED R2.988149,pi.6765549655.
Raw/normal plots retained plus supplemental independent-spectra plot in work.
Optimization threecycles peak.0962,.12605,.1518; laststillimproving, so its
iteration budget stopped before promotion. No hardware fault or raw loss.
Extended cycles3->5 with same0.06-.24 gain, res+/-2MHz tone/2-45us length;
cached identical completed plans/results reused. New supervisor35884/worker
67908 acquiring only cycle3 newpoints. Final accepted pulse still requires
balanced repeats/independentexamples/finalRabi>=16000reps,R2>=.98 andSSF>=.10.
No source promotion or24h claim yet. Allotherqueuesdeferred. Newtracking
controller will archive pucq4_config snapshot aswell asRR/expt/system; extra
provenance metadata does not change raw HDF5 schema or canonical folder layout.

User wants24h tracking now. Earlier0.10-cap SSF grid measured only~.06-.10
contrast; applying that extra-conservative policy to all readout optimization
would yield weak2000-shot tracking. Reviewed actual Q2 survey: repeated0.30
spectra stable versus lower-power refs, first resolved movement0.60. Initial
0.10 limit was a policy margin, not onset. Corrected interpretation: resonance
floor0.06 can remain low while SSF power optimizes below verified movement.
No user frequency/power guessed. New controller63956 in IDE/ssf_q2_ro1104
qualifies0.24=80%of measured stable0.30 through normal low/cap/cap/low scans
[.10,.24,.24,.10],7us,151pts,+/-3/.04MHz,5000reps, unchanged attenuation,
both sources0A/OFF checkedbefore/after. Require SNR>=15,R2>=.95, allminimum
comparisons<=max(.03MHz,3combinedSE), includinglow return; abortonmovement.
Onlythen raiseQ2readoutcap, rerunoriginalhigh-R2qspec/Rabi andoptimization,
requirefinalSSF>=.10, promoteQ2-only withbackups/checks andlaunch24h.

Stopped only priorparents22876/24172 while their short o2g1_3_1 SSF child
completed/saved normally; no raw deletion. Newfolder/canonicalsubstudy
ssf_pre_q2_ro1104 prevents label reuse. New helperPython3.9 parsingpassed.
No sharedcalibration promoted. Trackingstillnotstarted. Earlierstatistical
reviews remain evidence of noisy repeatable structure/pi, not cleanhigh-R2
passes. All-six/deck work deferred while Q2 gets priority; not markedcomplete.

### 2026-11-04 16:14 - user reprioritized Q2 24h ahead of remaining benchmark

16:39: independent16k qspec repeat accepted under the predeclared statistical
review: components5438.394540/5438.516762MHz,122.222kHz apart; centers agree
with8k record. Original high-R2 criterion remains false(two-componentR2.734).
This establishes repeatable spectral structure under noisy low-power readout,
not a microscopic TLS or clean-curve claim. Normal raw/population plots remain.
Q2 Rabi8k/171 completed with normalpi.6715802966489653 versus old.6666056278;
independent actual-cosine pi.6691472476+/-.0072150937DAC, R2.95818,lag1.0464.
Before any Rabi acquisition, declared pi-precision/repeat gates: one interior
maximum, amplitude>=10SE,piSE<=.015,R2>=.90,|lag1|<.3, bounded affine endpoints,
piwithin.01normalRR/.06prior and .6-.7window. OriginalR2>=.98 finding retained
separately; final optimized-pulse independentRabi must repeat withinmax(.02,
3combinedSE). Synthetic resolved/noise/structured cases passed. A cp1252 read
of Unicode experiment source failed offline; explicitutf-8-sig fixed it before
Rabi. No hardware failure. Alternating readout search now active below0.10cap.

16:27: initial capped-power Q2 qspec saved with raw/normalplot; carrier
5438.50556930511MHz/FWHM.1295625MHz/population.3195, but normalR2.556 and
two-componentR2.593. Doublet centers5438.389389/5438.513393MHz with10.00/3.85kHz
conditional centerSE, amplitudes4.28/14.09SE, separation124.00+/-9.57kHz,
deltaAIC15.14 versus a separately fitted single Lorentzian; residuallag1-.181.
Do not call this a clean high-R2 trace. Existing fixedR2>.94 worker scheduled
16k repeat; before it finished, predeclared separate repeat/uncertainty review
in low_power_qspec_review.py/RUN_BRIEF.txt, retaining original R2 finding.
Requires both scans>=4amplitudeSE, resolved widths/separation, interior peaks,
deltaAIC>=10, |lag1|<.3; latest each>=5SE/deltaAIC>=20; both centers repeat
withinmax(.025MHz,3combinedSE). Independent fresh Rabi/pi/SSF still mandatory.
This tests statistical identifiability under contrast-limited readout rather
than claiming the noisy curve is clean. Pure policy fixtures resolved/single/
noise/correlated residual all passed. No raw changes or smoothing.

Stopped only parent42460/49868; active16k scan/stage34324/RR65696 continues.
New supervisor22876 waits for c2qs1 receipt before resuming; explicit recovery
requires identical plan/returncode0/sources0off/existingraw and normal audit.
No new hardware until current scan exits. Q2-only promotion helpers tested on
all eight arrays/cap tuple: types and five other row values preserved. All
current-indexed flux profiles must remain byte-equivalent as literal values.

All-six power worker was still active on M3; Q2's fresh accepted floor0.06/
cap0.10DAC available. Latest user asked start24h now. Stopped ONLY owned parent
controllers51648/71596 without /T, preserving their active stage/RR child.
That scan fresh3_floor2_b completed16:13:54, returncode0, both sourcesCURR0A/
OFF/.01A/30V. Its normal raw/plot saved; parent audit/result assembly did not
run, so recover that completed receipt before resuming old worker. Original
queue_status archived and changed to deferred. Never overlap old queue/tracking.

Prepared IDE/ssf_q2_priority_1104 from existing bounded RR helpers; no low-level
pulse or repository acquisition changes. Frozen six sources still match. Python
3.9 AST parsing passed. Priority controller42460 launched16:13:52 and waited for
the old scan receipt before new hardware. Initial SSF offset checks active16:14.
Next qspec doublet, Rabi/pi, alternating low-power readout, independent examples,
final repeated Rabi/SSF, Q2-only shared config promotion/offlinechecks, then
24h2000shots/state fixed-pulse canonical RR. Updated authority/order explicitly
recorded in RUN_BRIEF.txt; no all-six/report gate falsely asserted. Shared
config/deck changes for remaining rows deferred. No24h acquisition claimed yet.

### 2026-11-04 - resumed queued all-six low-power campaign

14:32 hardware evidence: fresh1_cap_a completed(returncode0,270.39s), run
readout_power_1104/2026-11-04_14-27-52. Expected normal res plot and canonical
res_ge HDF5 both exist; sampled dip8919.17814MHz at0.10DAC/8us. Both sources
verified CURR0A/outputOFF/.01A/30V afterward. fresh1_cap_b is acquiring the
independent repeat; the all-six queue is still waiting. No cap/floor or final
calibration acceptance claimed from this first spectrum alone.

User requested continuation. Reviewed October30 queue logs/receipts/source
snapshots: all-six power qualification and conservative gain0.10 ceilings
passed, but M1 full1_low hit3600s with Echo49% acquired. Sources afterward
CURR0A/OFF/.01A/30V; study_data has no HDF5 final batch. Original controllers
ended. No report/config promotion or new24h tracking occurred. Classification:
execution environment/verification feedback, not a demonstrated hardware fault.

Created separate IDE workspace readout_power_1104, preserving old plans/logs/raw
and user changes. Archived six current acquisition sources before editing.
Added opt-in per-phase persistence to existing round_robin_benchmark.py:
one row/one pass/save_r1/debug/standard GE only; timed SSF excluded. All phases
use existing Data_H5 with unchanged field sets, labels, filenames and hierarchy;
normal final writers skip already saved phases. Default batch behavior remains.
Actual seven checkpoint calls passed hardware-free real-HDF5 schema/numbering/
duplicate/no-op checks;148-file harness and focused diff-check passed.

Fresh helper queue extends seven-experiment child budget3600->10800s, with
600s no-log-progress guard and owned-tree termination/error receipt. No pulse
program changes. Power controller51648 and finisher71596 launched hidden14:27
local, fresh res_spec acquisition confirmed in log. First stage M1/index0,
8919.17814006438MHz +/-3/.04/151/16000reps, gain0.10/8us. Remaining rows use
saved low-power seeds, never gain>0.10. Full hardware/acceptance envelope is
workspace/RUN_BRIEF.txt. Both sources checked before/after each stage. Raw path:
October7ROOT/PUCQ4/find_resonators_sandbox/readout_power_1104/<timestamp>.

Fresh paired cap measurements must resolve SNR>=15/R2>=.95 and repeat on old
low-power branch. Descending0.08/0.06/0.04 paired candidates refine the reliable
floor without exceeding the ceiling. Historical onset spectra retain October30
dates; future deck distinguishes them from November4 rechecks. Then fresh
all-six calibration/readout/fullRR -> new20261104 deck plus backed-upOct28/29
updates -> shared config promotion/checks -> new24hQ2SSF2000shots/state. No
dependent stage claimed complete at launch; old80round series preserved.

### 2026-10-30 - all-six power qualification and executable dependent queue ACTIVE

User expanded scope: all-six resonator power sweeps, lowest reliable power,
no optimization above frequency-movement onset; revalidate full benchmark and
both decks; then promote configs and automatically start full24hQ2SSF2000/state.
All-six power controller40456 active, last seen onM3. Dependency controller1544
waits for it; stages/paths and exact current status are at top ofPUCQ4_STATE.md.
Both are hidden background helpers using the existing RR and normal rawschema.
Power characterization at unchanged attenuators uses.001-.95DAC maximum,
bounded repeats/averaging<=16000, zero-current source guards and canonical
res_ge saves. Optimization is gated on reviewed per-row limits, with a
conservative additional power step reserved below onset when possible.
Weak M2/M1 returns were rejected; initial worker had loaded10000-average
repeats before the source was corrected to16000. Follow-up is queued with
matched16000-average fine spectra, twoSNR>=15/fitR2>.95 repeats. Do not convert
an unresolved weak return into proof of physical drift or a power threshold.
Later stages stop on failed gates; no high-gain fallback and no completed24h
claim. Original80round series stopped cleanly and remains preserved.

Localized RR changes add bounded SSF duration/STOP/status, explicit and future
persisted gain ceilings; shared limit tuple isNone until accepted promotion.
Added last_saved_path/last_plot_path attributes after successful writes to
avoid per-round full-directory scans during24h tracking. No rawschema, folder,
filename, QICK pulse program or section return signature changed. Hardware-free
2000sample raw roundtrip and cap/duration/selection guards passed;148file safe
harness passed. Source snapshots/proposed-source/hash checks guard queued
promotion. Current source snapshots and earlier pre-path-fix versions retained
in IDE readout_power_1030. Future PowerPoint/config claims require actual
reports_published.json/promotion_complete.json; queue status alone is not proof.

### 2026-10-30 - Q2 reduced-shot SSF setup; paused for readout-power validation

Fresh canonical RR calibration under ssf_24h_q2/2026-10-30_17-10-05 accepted:
M2/index1 res8950.850MHz, qcarrier5438.515064891349MHz, sigma.101us,
pi.666605627784899, RabiR2.996613. Doublet130.388kHz is visible; TLS unconfirmed.
Raw res/qspec/Rabi/SSF and population plots saved. system/expt/pucq4 carrier
5438.519514496114->5438.515064891349MHz; system pi.6715802966489653->
.666605627784899DAC. M2 RR shots50000->2000per state/reset100us. RO remains
8949.730MHz/.95DAC/7us. Other rows/nonzero current profiles preserved.
Added opt-in bounded24h SSF mode to existing RR; unchanged section/Data_H5 schema.
Guard/elapsed-time/normal-mode checks and148file harness passed; diffcheck passed.
Tracking2026-10-30_17-14-34 saved80rounds, then stopped gracefully (return0,
bothsources0A/OFF) for user's readout punch-out question. No full24h claim.
Single-power res and high SSF do not establish low-power dressed-regime operation.
Next lower-power spectra/state-dependent readout checks before restarting24h.
Full brief/plans/receipts/helpers in IDE ssf_24h_q2; archived calibration and
tracking documentation contain their plans/config snapshots/status.

### 2026-10-30 - Q2 Ramsey frequency labels in both benchmark decks

Added the existing fresh M2/index1 two-frequency diagnostic fit values to slides
34, 57 and 60 of both October29 and updated October28 final PowerPoints:
0.986883 and 1.125290 MHz; separation 138.407 kHz. These are Ramsey oscillation
frequencies, not absolute transition frequencies. Diagnostic interpretation and
unconfirmed microscopic cause are explicit. Source: benchmark_1029/fresh_models_M2.json
and its recorded October29 t2_ge HDF5 provenance. No new fit or hardware run.
Only the three slide XML entries changed in each deck; every other package entry
was byte-preserved, ZIP CRC passed, and deployed SHA256 matched the staged files.
Pre-edit decks were backed up beside the originals with suffix
`.before_q2_ramsey_annotation_20261030.bak`; receipt is archived in
`benchmark_1029/q2_annotation_receipt.json` under the user run root.

### 2026-10-29 closeout - fresh all-six benchmark and readout examples COMPLETE

Completed all7normalRRexperiments forM1-M6 after the warmup, including confirmed
M6/Q6 at6404.639622MHz (old6522seed rejected). Final zero-current summaries and
exact runpaths are now at the top of harness_engineering/PUCQ4_STATE.md.
All rawHDF5s and canonical calibrated-population plots verified; all T2 sampling
exceeds4points/fringe. No acquisition controller from this benchmark remains.
Final direct Yoko readback00:45:37UTC:bothCURR0A/OFF/.01A/30V. Preexisting
2025/February2026 Python processes were observed unchanged and preserved.

Alternating readoutlength/gain/frequency optimization and independent50000shot
examples completed all6rows. M1/M2/M4 required balanced repeated comparisons
and explicit drift-limited working settings; preserve raw variations and do not
substitute earlier high maxima. M2 matched T1/Echo controls at identical settings
showEcho12.80->10.38->9.16us; later pairs satisfy2T1 but a stationary lifetime
set is not reproduced. Main benchmark retains originalfull2values with effective
fit/time-variation flags. M1 mildT1model dependence/M5T2Rresidual structure
remain flagged; obsolete M4 andM5T1 warnings cleared from fresh results. M6
qspeclinewidth.2646MHz remains an explicit target warning, not a nominal pass.

Promoted only reviewed zero-current measured values to system_config.py,
expt_config.py,pucq4_config.py after hardware ended. AST-only checks match all6
rawresults, verify integer loop counts, and preserve all nonzero profiles/tables
and historical candidates. ROOT/calibration_changes_20261029.json records101
old->new values with row/index/units/provenance. RRM source bytes unchanged.
148-file safe offline harness and focused diff/whitespace checks passed.

ROOT M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7 now contains new
PUCQ4_benchmark_20261029_final.pptx and updatedPUCQ4_benchmark_20261028_final.pptx:
73slides,70images,6complete fresh rows; ZIPCRC/image/bounds/textencoding/raw-
settings QA passed. Originalotherdeck and manifest archived and hash-verified
before update. CSVsummary, six independentSSFexample contact sheet, manifest,
individualPNGcopies, exact plans/receipts/helpers and source snapshots saved.
No nativeOffice rendering claim. All user code edits and rawdata preserved.

### 2026-10-29 20:25 - final-pass review and bounded follow-up controls

Only acquiring controller29192 is on full3_a, thenfull4_a; controller71234
waits for successfulfull4_a receipt before matched M2 pair2_a/pair2_b
SSF/T1/Echo controls. M5 andM1 finalall7datasets/normalplots reviewed. M5
T1/T2R/T2E13.5552/5.1093/13.4068us, qspec.1540MHz/pop.459; retainonly
T2R effective-fit warning, not oldT1curvature. M1 values23.5391/20.1607/
30.5139us, qspec.1102/pop.5066; mildT1model dependence and readoutvariation
retained. A diagnostic M1 dual-frequency fit had near-coincident frequencies
and cancelling amplitudes~730 for populationrange~1: rejected, not evidence
of a doublet. Modelhelper now guards this verification failure; never replace
normal RR20.1607us with its degenerate alternative envelope.

Fresh M2 resolves a qspecdoublet separation.1270MHz and Ramseybeating;
normal T1=5.35174us, T2R=6.59883us, Echo=12.79781us. Echo exceeds2T1
under stationarysingle-rate assumptions, so M2 remains unaccepted pending
two matched controlpairs. Source inspection confirms Echoaxis sums two
half-waits; no factor-two axis substitution justified. Exact zero-current
controlenvelope in IDE benchmark_1029/paired2_envelope.json. Preserve
originalfull2values and allcontrols rather than select larger favorableT1s.

### 2026-10-29 19:04 - completed M6, temporal readout qualification

19:22 update: balanced controller69260 ended successfully. M2 working pulse
8949.73MHz/.95/7us, independent50kSSF.44472/.45248; samepoint three-block
gridmean.505687 ranges.467912-.534750, so no stationary optimum claim.
M4 working8999.25/.95/5us, examples.432704/.42988, lengthmean5us.45795
vs3us.38298, no reproducible frequencyoffset advantage in three-block grid.
Both qualified reviews accepted with raw ranges and temporal warning; original
failed strict gates remain preserved. This is a separate drift-limited review,
not a relaxed stable-repeat threshold. Controller65053 now confirms M1/3/5;
both sources remain0A/OFF. Full2_a andfull4_a plans prepared, not acquired.

M6 full6_a finished all7 normal datasets/plots with independently checked
coherence fits. Qubit6404.639622MHz, pi.651682/sigma.05558683us,
SSF.78504 at9057.695/.95/45us, T1/T2R/T2E4.01062/1.65802/4.39739us.
Retain qspeclinewidth warning .2646MHz despite population.320/R2.991.
M2 freshcontrolpre2_b pulse passed but adjacent-reference readout gate failed:
identical7usSSF.38836-.48430 within2minutes. Controller26146 ended, allraw
saved; no maximum promotion. M4 independent recovery22366 running. Balanced
three-block comparisons prepared to quantify variability without claiming a
stationary optimum. Shared sourceconfigs and both decks still unmodified.

### 2026-09-24 - benchmark NumPy import failure

- User launch failed at line 6, before hardware or data-folder operations, with
  NumPy 1.21.5 `_multiarray_umath` DLL-load failure and an MKL warning.
- Layer: execution environment; matches the recorded unactivated Conda runtime.
- PyCharm reports base Conda Python 3.9.13. Activating that environment with
  `Scripts/activate.bat` before invoking its Python successfully imported NumPy
  1.21.5 and h5py 3.7.0; a NumPy dot product returned 11.0.
- Use an activated Anaconda Prompt for launch. No package installation or
  experiment configuration changes were needed. No hardware was contacted.
- The earlier syntax-only verifier passed 147 files but cannot detect DLL
  loading failures; dependency import verification is separate from parsing.

Use one short entry per agent-assisted task. This file is persistent memory for
facts that should survive chat sessions; do not put credentials or raw data here.

### 2026-09-14 — next-chat Q5/Q6 handoff

- Created `docs/Q5_Q6_NEXT_CHAT_HANDOFF.md` at the user's request for another
  chat to continue the unresolved scientific tasks autonomously.
- Includes authoritative evidence/report paths and hashes, accepted pairs,
  rejected central/opposite-side tests, mapping/routing gaps, temporal evidence,
  acceptance gates, the bounded hardware envelope, environment and report workflow.
- Distinguishes historical parked state from fresh preflight, and the skipped
  conditional M6 plan from a newly justified investigation. Existing authorization
  and the user's no-routine-permission preference are preserved without expanding
  device limits. This task only wrote documentation; no hardware was contacted.

### 2026-09-14 — Q5/Q6 bounded follow-up complete; coupling unresolved

- Final20:41: eleven additional guarded runs completed, bringing the campaign
  to22 runs/76 drive-off/on spectra. Audit passes213HDF5/137expectedplots/
  4.295914GiB, all guardsexit0, protected/saved source hashes and original-deck
  preservation. Both sources ended CURR/.01A/0A/output-off; no controller remains.
- Accepted pairs at two distinct detuned biases: -5.8mA (gains.007 and.015)
  and-5.74mA (gain.015). Same-.015gain separations37.094685+/-.006145 and
  24.313851+/-.005468MHz are measured-bias separations, not minimum/J.
  Central64000/.03 upper repeats5444.498749MHz but failsSNR3.7118; lower
  passesSNR7.9745 and referencespass. RightM2/32000 upper has no candidate.
  Conditional right_m6 was skipped because the required central pair failed.
  Seven-bias/minimum/J and physicalQ5/independent-drive mapping remain unresolved.
- Updated143slide PowerPoint published at the original run-folder report path.
  Native143/143rendered, no detected overflow; findings, pair table, stability
  and cleanup reviewed visually. Published SHA256
  0B3AB5B08AC4C5AA496FA3ACCB84BA08884397FD4E3E86894A6517A09678DA0F.
  OriginalDownloads deck and previous94slide report preserved. All final
  analysis/audit/findings/render/publication evidence is under
  ac_review/resolution_20260914. Earlier ACTIVE bullets below are history.
- Durable lessons: retain actual source-setting axes and per-window averaging
  provenance; a repeated weak center is not acceptance; additional averaging
  need not improve measured SNR. Positive controls and temporal-block evidence
  qualify null results. No QICK program or shared calibration was changed by
  this continuation. Focused code review and scientific/harness checks passed.

- Latest19:45: ac_right completeexit0/2729.562s,bothsourcesCURR/.01A/zero/off;
  upper5452.5-5458.5MHz/32000/.03 no candidate,vectorSNR2.49436,
  cleanreference. Lower remains accepted; rawupperplot reviewed. No new pair.
  NowACTIVE central_max_0914/ac_cmax,guard40956/child59996,
  run2026-09-14_19-44-57. Original prepared envelope retained and revalidated:
  M2/-5.656600mA/.03,lower101pts32000,upper131pts64000,20us100us/equalrefs,
  4200s+180guard,zero/offcleanup. Actual observed rate gives about49min total.
  Review targeted upper candidate at the existing ceiling before closure or
  any scientifically justified dependent scan. No blind maps if it remains
  unaccepted. Latest completed21run/74spectrum analysis:right_checkpoint/.
  Focused code changes reviewed; latest138file harness/whitespace checks pass.

- At19:10, ac_right lower control passes at5436.331129599+/-.002384236MHz,
  SNR30.1265; upper reference continues. Prepared central_max_0914 repeats
  the central M2 candidate at64000 averages/.03gain, with32000-average lower
  control and measured narrow windows; not launched concurrently. Full envelope
  and stop criteria are in the resolution plan; offline plan validation passed.
- Additional saved-raw stability review: ac_bcf M2 lower's four chronological
  quarters span27.215kHz; its three accepted upper quarters span2.738kHz.
  ac_right lower's four quarters span9.022kHz. Shared full off references mean
  block errors are correlated; spans are descriptive, not independent drift
  significance or a diagnosed cause. Raw shape/normalization checks passed.
  combined_block_analysis.json preserves the independent M4/M2 sources for slides.
- Completed20run audit passes203HDF5/131plots/3.633654982GiB and all archived
  guard/source/hash checks. The first invocation wrote results then failed on a
  helper same-file copy; original TEMP invocation completed exit0 under
  evidence_checkpoint_20runs_verified.json. Use that verified checkpoint.
  Native135slide draft passed rendering/overflow and selected visual checks;
  final report still excludes active and subsequent acquisitions until reviewed.

- Latest19:00: ac_ctrhi and ac_m6hi complete exit0 with bothsources zero/off.
  M2 central upper5444.489811 MHz fails SNR4.200675; M6 upper has no candidate
  (vectorSNR2.70837). Both lower controls pass. RawIQ and terminal guards checked.
  Now ACTIVE ac_right/right_visibility_0914, guard13696/child55692 atM2/-5.59mA,
  .03gain/20us/100us,lower121pts8000/upper301pts32000,4500s+180guard.
  Full envelope and adaptive rationale in resolution plan. Prior ACTIVE bullets
  below are historical. Latest harness138files and whitespace checks pass.
- Environment lesson: Windows PowerShell Set-Content -Encoding UTF8 adds a BOM;
  strict utf-8 JSON readers reject it. Re-save generated plans with
  IO.File.WriteAllText and Text.UTF8Encoding(false), then validate before launch.
  The failed offline plan check performed no acquisition; corrected check passed.

- User requested a plan, autonomous execution and an updated PowerPoint.
  Plan: docs/pucq4-crossing-resolution-plan.md. Previous94slide report preserved
  in ac_review/final_20260914/PUCQ4_Q5_Q6_94slides.pptx with matching hash.
- Completed guarded ac_det1, ac_dcf1 and ac_dq6, each exit0 with bothsources0/off.
  Accepted M2 centers atY4-5.8/-5.4/-4.8mA are5439.864905/5438.068054/
  5438.540998MHz atgain.007. A weak -5.8candidate was confirmed with4x averaging;
  the rejected initial candidate remains rejected. M6/-5.8 locator finds
  5402.784229MHz atgain.03. These different-power locators are not a mode pair.
- ac_pair1 completed with an accepted M2 pair at-5.8mA/.007gain:
  separation37.071123+/-.008306MHz; M6lower corroborates. ac_bridge repeated
  the pair at.015gain,37.094685+/-.006145MHz. Bothguards exit0/zero-off.
  ac_ctrlo at recorded-5.656600mA/.007 recovered neither upper nor lower
  control, so its null is sensitivity-limited. Allreferencesclean,guardexit0.
- ac_bcf completed exit 0 with both sources zero/off. The matched M2 pair
  at -5.74 mA/gain .015 has separation 24.313850811 +/- .005468127 MHz:
  lower 5416.233092832 MHz, upper 5440.546943643 MHz. Two distinct paired
  biases are accepted, still insufficient for J/minimum. The initial weak
  upper candidate remains rejected. The lower line shifted +.025377113 MHz
  between same-gain visits, beyond conditional fit errors; cause unassigned.
- ACTIVE central_highpower_0914/ac_ctrhi, guard13892/child60576,
  run2026-09-14_17-32-32: M2/-5.656600 mA/.03 gain/32000 reps, 20 kHz bins.
  Lower control ACCEPTED 5431.727524559 +/- .005139544 MHz, width .141231591
  MHz, SNR6.7414, clean reference. Final upper5442-5448 MHz driven acquisition
  started around18:07; guard deadline5400s+180s and zero/off cleanup remain.
  Prepared/unlaunched right_visibility_0914 tests M2/-5.59 mA with a strong
  measured lower control and eightfold upper-window averaging. Full envelope
  and stop/continuation criteria are in the resolution plan.
- Analysis-only extension supports explicit identical disjoint frequency windows
  using actual saved axes, without filling gaps or merging repeated visits.
  Known-J recovery and negative source/power/window/reference/extra-line tests
  pass; no peak/coupling acceptance threshold relaxed. Acquisition programs and
  shared calibration unchanged. Original deck leaves M1-M4 identities unknown;
  no additional PUCQ4 routing document found in scoped local searches. Physical
  Q5 assignment and independently routed Q5 microwave drive remain unsupported.
- Report and final evidence audit remain pending until this continuation is
  reviewed. Do not interpret old closeout entries as current run status.
- Corrected analysis current coordinates to saved source settings, retaining
  requested values: maximum67nA difference; all58earlier metric objects exactly
  unchanged. Require matchingfinite off/on settings. This is source-level
  readback, not an independent ammeter. Classified asverification/source-of-truth.
  Lowpower lost-positive-control lesson recorded in FAILURE_MODES.md.
- Completed16-run audit passed178HDF5/116plots/2.706736445GiB andprotected
  source hashes. Latest17-run/66trace checkpoint is resolution_20260914/
  bridge_checkpoint/analysis.json. Unpublished126-slide draft renderednatively
  with no text/celloverflow; signed-IQ pairpanels andtables checked readable.
  Synthetic scientific checks and138-fileofflineharness passed; finalrefreshpending.
- Latest completed18-run audit passed193HDF5/125plots/3.077016354GiB,
  matching source snapshots/protected configurations and all guard cleanups.
  Checkpoint: resolution_20260914/confirmed_bridge_checkpoint/analysis.json
  (68 spectra). Current unpublished draft has131slides and two paragraphs per
  findings page after native review found footer crowding. Latest native findings
  are readable; final publication awaits active/conditional-run outcomes.
  Offline harness138files and whitespace check passed again after layout fixes.

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
- Output: `M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/`
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
- Spectroscopy output: `M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/`
  `PUCQ4/pucq4_first_light/qspec_gain_repeatability_0mA/2026-08-27_19-24-42/`.
- Rabi calibration: sigma 0.10 us for all calibrated modes; pi amplitudes M1
  0.3492950, M2 0.6486907, M3 0.7983886, M5 0.3692547, M6 0.4091741 DAC
  units. The sweep was 0–1 gain, 101 points, 200 reps, one round.
- Primary coherence run: one round on M1, M2, M3, M5, M6; T1 0.01–500 us,
  200 points, 300 reps; Ramsey/echo 0–150 us, 100 points, 100 reps. Output:
  `M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/`
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
- Refinement output: `M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/`
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
  `M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/`
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

## 2026-08-31 final negative-current reruns and report maps

- Repeated Q4/M5 at -10 mA with the corrected resonator-peak selection and a
  current-specific readout seed. Rescue SSF improved only to 0.086; qspec and
  Rabi became unstable, ending at pi 0.419154 after bounded retries. Rejected.
- Repeated Q6/M6 at -7.5 mA. The 4838.755 MHz post-readout line and 0.355 SSF
  were recovered, but bounded Rabi retries ended at pi 0.778429. Rejected; no
  coherence run.
- Rebuilt qspec/res-spec maps with bias current on X, actual acquired frequency
  on Y, and row-normalized measured magnitude as color. Added separate accepted
  fitted-center-versus-current and qspec-FWHM-versus-current plots and inserted
  them into the reusable PowerPoint. Missing/failed values remain unfilled.
- Independent post-run checks found Yoko 3 and Yoko 4 in CURR mode at 0.0 A,
  output off.

### Q6 -7.5 mA pulse/qspec separation test

- Forced a fresh raw-I/Q Rabi at the latest measured qspec center instead of
  reusing the old provisional pi seed. This recovered a clean current-specific
  pulse result: qspec 4839.162419 MHz / 0.158162 MHz FWHM, sigma 0.1222528 us,
  pi 0.678630.
- The subsequent calibrated-population qspec was not stable: final widths were
  1.367 MHz at gain 0.006, then 1.357/2.177 MHz after a bounded gain-0.004,
  8000-repetition and optimized-readout bootstrap trial. The lower-power change
  was reverted because it did not reproduce the narrow line.
- Classification: verification/state. A single narrow trace is insufficient;
  repeated final-chain qspec instability blocks coherence at this bias.
- Both Yokogawas were independently verified at 0 A/output off after each run.

### Q6 -10 mA reproducibility test and negative-bias closure

- Forced fresh raw-I/Q Rabi at 4376.306486 MHz and expanded the current-specific
  readout optimizer cap from 0.88 to 0.95 because the earlier isolated optimum
  lay at the gain boundary. Raw Rabi passed twice; latest pi was 0.688610 at
  sigma 0.0986291 us.
- Two independent alternating readout searches and wide rescue sweeps reached
  only 0.0820 and 0.0802 fidelity. The former 0.60 point did not reproduce.
  Qspec also failed the initial Lorentzian gate. No coherence was run.
- Q6 -5 mA already exhibits the equivalent bounded result: accepted raw-I/Q
  pulse calibration but approximately 0.08 maximum rescue fidelity across the
  broad readout-frequency/gain grid. Further offset shifting is not supported.
- Current-specific provisional sigmas remain separate in
  `PUCQ4_FLUX_PULSE_CALIBRATION`; no zero-current sigma was changed. Both
  Yokogawas were independently verified at 0 A/output off.

## 2026-09-02: Q4 -10 mA settled spectroscopy/Rabi test and deck refresh

- After a verified ramp to -10.0 mA and a 600 s settle, a persisted +/-100 MHz
  diagnostic found a narrow feature at 5997.922411 MHz with 0.099989 MHz FWHM.
- The immediately following continuous full-chain run found the narrow qspec
  center at 6004.213528 MHz with 0.110991 MHz FWHM. The 6.291 MHz shift over the
  run interval shows that a scalar qspec fit alone is not a stable calibration
  at this bias.
- Five 171-point, 300-average raw-I/Q Rabi attempts across bounded sigma updates
  returned pi amplitudes outside the 0.6-0.7 gate; the final value was 0.248733.
  The controller aborted before SSF/coherence and returned Yoko3 to 0 A/output
  off. Q4 therefore remains validated at 8/9 grid currents, with -10 mA failed.
- Rebuilt the aggregate maps. Resonator maps now reconstruct absolute swept MHz
  from `freq_pts + freq_center`; failed-point calibrated-population columns are
  deliberately blank, while diagnostic magnitude evidence remains labeled with
  failed status.
- Rebuilt `PUCQ4_first_light_characterization.pptx` as 35 slides. Its Q4 status
  table and negative-current evidence slide include the September 2 results and
  do not claim a -10 mA sigma, pi amplitude, SSF, or coherence value.

## 2026-09-02: Q4 corrected-frequency closure and population-qspec revalidation

- Expanded the -10 mA locator to 5747.922-6247.922 MHz (5001 points, 0.1 MHz,
  2000 reps). It isolated the real line near 6037.97 MHz and rejected the prior
  5984/5998/6004 MHz aliases. Final qspec is 6037.585408 MHz / 0.203776 MHz;
  final resonator base is 9010.49 MHz. The user-authorized Rabi exception is
  retained explicitly (sigma 0.0165397052 us, pi 0.994934, provisional), while
  averaged coherence completed at T1/T2R/T2E = 17.60/1.61/3.63 us.
- At +5 mA, absolute-population gain scans rejected an underexcited narrow
  raw-I candidate. Gain 0.03 reached population 0.3051 at 7121.580622 MHz with
  a reproducible 0.816923 MHz power-broadened width. Fresh exact-frequency Rabi
  gave sigma 0.0808489431 us / pi 0.678630; readout base/length/gain/offset/SSF
  is 9014.59 MHz / 6 us / 0.92 / -0.48 MHz / 0.512, and coherence is
  2.31/3.99/1.90 us.
- At -2.5 mA, gain 0.0025 passed the strict population gate at 6945.111992 MHz,
  0.120213 MHz FWHM, population 0.3036, and SNR 128.4. A fresh Rabi/coherence
  chain at that exact frequency gave sigma/pi 0.0591114617 us / 0.648691,
  readout 9013.74 MHz / 6 us / 0.73 / +0.32 MHz / SSF 0.6183, and
  T1/T2R/T2E = 11.95/3.63/9.07 us.
- Each run used the normal round-robin HDF5 hierarchy and guarded source
  cleanup. Both Yokogawas were independently verified at 0 A/output off.

## 2026-09-03: Q4 identical-grid qspec map and final reusable deck

- Forced a fresh nine-current Q4/M5 map at -10 through +10 mA using one common
  requested qspec grid (5950-7200 MHz, 0.02 MHz, 300 reps). All nine canonical
  HDF5 rows contain the same 62,501-point QICK-realized axis,
  5950.00000305-7198.83652039 MHz with 0.01998138 MHz median spacing.
- Each row used its final current-specific physical readout tone and acquired
  1,000 ground plus 1,000 excited SSF shots. One centroid pair is fixed over
  the entire row; every row uses the same signed affine ground=0/excited=1
  convention, a common 0-1 population scale, and no row normalization.
- All selected population peaks lie inside the common range. The minimum edge
  margin is 54.98 MHz; full-grid population peaks are within -0.281 to +0.148
  MHz of the final local centers, with local SNR 6.99-26.23. An automated
  provenance audit also confirmed that all nine Rabi acquisitions used their
  corresponding final widened-scan qspec frequency.
- Rebuilt the aggregate raw-signal/population heatmaps, complete full-frequency
  curve galleries, selected-frequency check, resonator and Rabi maps, and
  all-bias T1/T2R/T2E maps. Rebuilt the reusable PowerPoint as 32 slides with
  both Yoko tables and acquisition settings. The clean outputs are in
  `yoko_flux_campaign/aggregate_2d/` and
  `pucq4_first_light/PUCQ4_first_light_characterization.pptx` on the canonical
  `M:` run hierarchy.
- The report preserves two material caveats: Q4 -10 mA has provisional
  sigma/pi 0.0165397052 us / 0.994934 and low SSF 0.124 under the user's Rabi
  exception; Q4 +5 mA is population-valid but power-broadened to 0.816923 MHz.
- Fixed the report decoder after the new 62,501-value serialized arrays exposed
  a Windows `ast.literal_eval` stack overflow. Numeric vectors now take the
  bounded `numpy.fromstring` path before the legacy literal parser.
- Guarded cleanup completed, followed by an independent readback of both
  Yokogawas in current mode at 0.0 A with outputs off.

## 2026-09-03: lower-frequency two-Yoko qspec maps and deck update

- Acquired a new segmented full-range qspec map at all nine authorized currents
  for both tunable devices. Q4/M5 covers requested 5000-7300 MHz and Q6/M6
  covers 4000-7300 MHz. Each row has three overlapping bands, 100 averages per
  frequency, approximately 0.04 MHz requested spacing, and an independent
  built-in 1,000-shot/state SSF calibration per band. This produced 54 new
  canonical qspec HDF5 files; no raw artifact was overwritten.
- All Q4 rows share the realized 5000.000002-7299.688149 MHz stitched axis. All
  Q6 rows share 4000.000005-7299.542615 MHz. Median realized spacing is
  0.03998337 MHz for both, and every cross-current axis equality/readout-tone
  check passed. The shifted physical readout tone used at each bias is stored in
  the manifest and shown in the PowerPoint table.
- The lower endpoint transitions are now inside the map: Q4 -10 mA is near
  6037.9 MHz and Q6 -10 mA has a broad/noisy map-power response near
  4375.5-4376.3 MHz. The latter reaches the bounded local fit-width limit, so it
  is retained as spectroscopy-map evidence and is not mislabeled as a newly
  accepted Rabi/coherence calibration.
- Rebuilt the full-range calibrated-population maps with current on x, actual
  swept frequency on y, and saved signed SSF population as color. No row
  normalization or shared calibration is applied. Added nine-panel local curve
  galleries so every narrow response is visible at its actual frequency and
  broad/noisy cases remain apparent.
- Updated the reusable canonical deck at
  `pucq4_first_light/PUCQ4_first_light_characterization.pptx` from 32 to 36
  slides. Slides 33/35 contain the Q4/Q6 heatmaps and complete per-current f01,
  resonator-base, readout-tone, length, gain, and offset tables; slides 34/36
  contain the local measured-curve checks. PowerPoint COM rendering verified all
  four final slides. A 32-slide pre-update backup remains beside the deck.
- The production choice followed an endpoint pilot at 0.02 MHz/300 reps. The
  pilot found Q4 -10 mA near 6037.559 MHz (local SNR about 23) and Q6 -10 mA
  near 4375.950 MHz (local SNR about 5.34); the final 0.04 MHz/100-rep grid still
  resolves each several-sample line while avoiding redundant multi-GHz
  averaging. Resonator DAC attenuations were [10, 15] dB, qubit DAC [5, 4] dB,
  and ADC 17 dB.
- Every guarded acquisition returned its source to zero. Independent final
  readback found Yoko3 and Yoko4 in current mode at exactly 0.0 A with outputs
  off.

## 2026-09-05: uninterrupted high-repetition qspec maps and final deck

- Added `pucq4_08_single_sweep_qspec_maps.py` to run one continuous qspec sweep
  per current using the final current-specific shifted resonator readout. This
  replaces the three-band map only as the report's qspec-map evidence; it does
  not overwrite any prior raw data or change shared pulse calibration.
- Completed all 18 canonical qspec HDF5 rows at the nine authorized currents.
  Q4/M5 used 57,501 points from 5000.00000153 to 7299.04365692 MHz; Q6/M6 used
  82,501 points from 4000.00000534 to 7298.62785873 MHz. Both realized grids
  have 0.0399833700 MHz median spacing. Acquisition used 500 reps/frequency,
  one 2,000-shot/state SSF calibration per whole row, 250 us SSF relaxation,
  resonator DAC attenuation [10, 15] dB, qubit DAC [5, 4] dB, and ADC 17 dB.
- Continuous-sweep population maxima for Q4 from -10 through +10 mA are
  6038.168150, 6422.888116, 6725.202361, 6945.390768, 7084.532888,
  7143.788239, 7121.477520, 7022.398734, and 6841.593945 MHz. Q6 maxima are
  4375.643747, 4839.850648, 5587.979446, 6103.644942, 6522.550687,
  6842.137747, 7062.765971, 7185.594878, and 7211.424133 MHz. These map-power
  maxima are labeled separately from the validated drive-frequency seeds.
- Q4 local positive-peak SNR is 7.42-30.23. Q6 is 3.76-19.34; its -10 and
  -2.5 mA rows are the only controller warnings because SNR is just below 4.
  Independent local checks reproduce responses near 4375.747297 and
  6103.609930 MHz, respectively, but low state separation remains a documented
  readout-calibration caveat rather than being hidden through row normalization.
- Built population and signal-amplitude heatmaps with current on x and the
  complete actual swept frequency on y, plus nine-panel local-population
  galleries. Population uses the built-in calibration saved with each row and
  is not normalized across current. Display-only 1,500-bin reduction averages
  ordinary bins and peak-preserves only each measured line's +/-1 MHz
  neighborhood; raw canonical HDF5 samples are unchanged. Provenance is stored
  in `yoko_flux_campaign/single_sweep_qspec_2d/`.
- Rebuilt `PUCQ4_first_light_characterization.pptx` from the clean 32-slide
  checkpoint as a 38-slide deck. Slides 33-38 contain Q4 then Q6 population
  maps/readout tables, signal-amplitude maps, and local peak galleries. The
  notes record the realized hardware axes, reps, SSF shots, display treatment,
  and units. PowerPoint COM reopened the deck and exported all six slides at
  1600x900 for visual verification. The superseded 36-slide segmented deck is
  retained beside the canonical presentation.
- Execution/lifecycle incident: the original attached terminal process was
  killed while the Q4 +7.5 mA row was running. No HDF5 existed, the manifest
  remained `running`, Python cleanup did not execute, and Yoko3 remained at
  +7.5 mA/output on. It was immediately queried, ramped to 0 A at 0.5 mA/s, and
  disabled; Yoko4 was already 0 A/output off. A first bare detached launch then
  failed before hardware access because Conda's HDF5 DLL path was missing. The
  successful hidden detached launch inherited the configured Conda runtime and
  reacquired only the incomplete row before completing the campaign. The
  durable prevention/recovery rule is recorded in `FAILURE_MODES.md`.
- Final independent instrument readback: Yoko3 and Yoko4 were each in current
  mode at exactly 0.000000000 A with output disabled. This qspec-only closure
  did not rerun Rabi/T1/T2R/T2E; the clean base deck retains the prior final
  experiment maps and accepted/provisional labels.

## 2026-09-06: peak-visible continuous qspec heatmaps

- Corrected the display failure in which every local qspec panel contained a
  visible transition but the same sub-MHz lines disappeared in the multi-GHz
  heatmap. The source acquisitions were already the requested one-sweep/common-
  grid data: Q4 uses 5000.00000153-7299.04365692 MHz at 57,501 points, Q6 uses
  4000.00000534-7298.62785873 MHz at 82,501 points, both at 0.0399833700 MHz
  median spacing and 500 reps/frequency. No hardware was operated or raw data
  reacquired for this revision.
- Added population-contrast and signal-contrast heatmaps. The display subtracts
  only a slow interpolated 20-MHz block-median baseline from each fixed-smoothed
  row; it does not normalize row amplitude, peak height, range, or noise.
  Ordinary 1,500-bin display cells are averaged, while the independently
  measured +/-1 MHz peak cell keeps its maximum and is repeated across three
  raster rows. A disclosed gamma=0.55 color transform improves low-level
  contrast, and both population maps share a fixed 0.00-0.35 Delta-Pe scale.
  Canonical HDF5 arrays and the local physical-population panels are unchanged.
- Rebuilt the 38-slide canonical
  `pucq4_first_light/PUCQ4_first_light_characterization.pptx`. Slides 33-38 now
  use the revised population/signal maps; their colored ridges are visible for
  all nine currents on both Yokos, including both -10 mA endpoints. The two
  parameter tables now include qspec drive gain and length as well as the seed,
  measured peak, resonator base, shifted readout tone, readout length/gain, and
  offset. A fresh 1600x900 PowerPoint COM export verified all six slides.

## 2026-09-07: fixed-sigma two-Yoko Rabi population maps

- Added `pucq4_09_single_sweep_rabi_maps.py` and completed 18 one-pass canonical
  Rabi acquisitions: nine authorized currents each for physical Q4/M5 on Yoko 3
  and physical Q6/M6 on Yoko 4. Every row used the observed continuous-wide-
  qspec peak and the exact shifted resonator readout tone from its corresponding
  qspec row; the controller rejected any frequency mismatch before acquisition.
- Used one fixed sigma per device without changing shared configuration: 0.060
  us for Q4 and 0.090 us for Q6. All rows requested gain 0-1 at 201 points and
  realized one identical 0.00000000-0.99493377 QICK axis with 0.00497467 median
  spacing. Settings were 1,200 reps x 1 round, 250 us Rabi relaxation, and one
  saved 2,000-shot/state SSF calibration with 250 us relaxation per current.
  Attenuation was resonator DAC [10,15] dB, qubit DAC [5,4] dB, ADC 17 dB.
- Every calibrated trace passed the 0.15 review-span threshold. Q4 spans are
  0.9390-1.0064 with fitted first extrema 0.577062-0.994934. Q6 spans are
  0.8718-1.3138 with fitted extrema 0.363151-0.925288. Q6 -7.5 mA is a clear but
  noisier response with low SSF separation and affine population overshoot; it
  is retained as measured rather than normalized. Q4 -10 and -5 mA and Q6
  -7.5 mA reach their first extrema near the requested gain ceiling, which is
  expected evidence from a fixed-sigma map rather than a per-current sigma tune.
- Added an analysis-only builder for current-x/gain-y/calibrated-population-z
  heatmaps plus nine-panel physical-population curves. Both maps use one shared
  -0.30 to 1.50 color scale and only a fixed three-sample display mean; no row
  offset or amplitude normalization is applied, and canonical HDF5 is unchanged.
- Updated the reusable PowerPoint from 38 to 42 slides. Slides 39-42 contain the
  Q4 map and full frequency/readout/settings table, Q4 curves, Q6 map/table, and
  Q6 curves. PowerPoint COM reopened the deck and exported all four new slides
  at 1600x900; visual inspection passed. The original 38-slide deck is retained
  as `PUCQ4_first_light_characterization_pre_fixed_rabi_20260907.pptx`.
- A valid 0 mA pilot was initially rejected only because the realized QICK gain
  endpoint is 0.99493377 rather than nominal 1.0. The validator now accepts a
  bounded two-step shortfall and recovered that canonical row without a second
  hardware acquisition. The durable failure-mode rule is recorded separately.
  The production controller then exited with empty stderr and final independent
  readback of both Yokogawas at exactly 0 A/output off.

## 2026-09-07: Ramsey-corrected common-60-ns Rabi maps

- Objective: use the prior wide-qspec frequencies and prior fitted pi gains as
  inputs to the canonical round-robin Ramsey frequency-correction path, then use
  each accepted corrected frequency in a new current-by-gain Rabi map with one
  0.060 us sigma for both tunable devices. Preserve the already validated
  current-specific shifted resonator readout frequencies and document all
  results in new end slides.
- Added `pucq4_10_ramsey_corrected_rabi_maps.py`. It verifies the source qspec
  and Rabi manifests, calls `T2RMeasurement.adjust_qspec`, validates the Ramsey
  result, and acquires/saves the corrected-frequency Rabi row using the normal
  PUCQ4 canonical HDF5 helpers. `round_robin_benchmark.py` gained an opt-in
  standalone `t2r_correction` route and explicit Ramsey setting overrides while
  preserving ordinary benchmark defaults.
- Ramsey settings: 0-1 us, 50 points, 4 MHz intentional detuning, 1,000 reps x
  3 rounds, 4,000 us relaxation. The correction pulse used the prior pi gain
  with its matching calibration sigma (Q4 0.060 us, Q6 0.090 us). Final Rabi
  settings: one common 0.060 us sigma, requested gain 0-1/201 points, 1,200 reps
  x 2 rounds, 250 us relaxation, and 4,000 SSF shots per state at every current.
- Completed 18/18 rows over -10 to +10 mA in 2.5 mA steps. Every manifest row is
  `complete`; all 36 HDF5 and all 36 canonical plot paths exist. Corrected qfreq
  ranges are 6038.096816-7143.693599 MHz for Q4 and
  4375.815431-7211.196119 MHz for Q6. The largest absolute corrections were
  +0.672790 MHz for Q4 at +5 mA and -0.731300 MHz for Q6 at -7.5 mA.
- The common realized Rabi gain axis is 0.00000000-0.99493377 with 0.00497467
  median spacing. Minimum calibrated 5-95% response span was 0.9573 for Q4 and
  0.5888 for Q6. Maximum independent Ramsey frequency standard error was
  0.00601 MHz for Q4 and 0.02699 MHz for Q6.
- Q6 -10 mA produced one intentional pre-Rabi abort: the endpoint-based
  canonical quadrature selector saved a weak Q fringe even though the I fit was
  strong. Sources were returned to 0 A/output off. After adding independent
  both-quadrature validation, the same row was reacquired; the accepted repeat
  selected I and passed saved and independent criteria. No failed Rabi data was
  promoted.
- Added `tools/build_pucq4_ramsey_corrected_rabi_maps.py` and generated per-device
  Ramsey galleries, current-x/gain-y/calibrated-population heatmaps, companion
  Rabi galleries, and `ramsey_corrected_rabi_provenance.json`. The heatmaps use
  one -0.30 to 1.50 physical population scale and a disclosed fixed three-point
  display mean; no row normalization changes the canonical HDF5.
- Extended `tools/build_pucq4_first_light_ppt.py` with
  `--append-ramsey-rabi`. The canonical deck increased from 42 to 48 slides;
  slides 43-48 hold both devices' Ramsey evidence, heatmaps, complete corrected
  qubit/readout-frequency tables, and Rabi curves. PowerPoint COM reopened the
  48-slide deck and all six 1600x900 slide exports passed visual inspection. A
  pre-update backup was retained.
- Final controller readback verified Yoko3 and Yoko4 in current mode at exactly
  0 A with output disabled. The accepted production controller logs have empty
  stderr; the earlier Q6 -10 controller stderr records the intentional validation
  abort and is retained for provenance.

## 2026-09-08: Ramsey-corrected sigma tuning and coherence maps

- Objective: rerun the complete two-Yokogawa current grid, obtain a fresh
  round-robin Ramsey correction at every current, tune a separate Gaussian
  sigma so the fitted pi amplitude is near 0.65, then use the corrected
  frequency and selected sigma for T1/T2R/T2E maps with delay in microseconds.
- Added the resumable hardware controller
  `pucq4_11_sigma_coherence_maps.py` and analysis builder
  `tools/build_pucq4_sigma_coherence_maps.py`. The controller completed all
  18 rows and returned Yoko3/Yoko4 independently to current mode, 0.010 A
  range, exactly 0 A, output off. Production and retry stderr logs are empty.
- Ramsey used 0-1 us/50 points, 4 MHz detuning, 1,000 reps x 3 rounds, and
  4,000 us relaxation; Q6 -7.5/-10 mA used 2,000 x 4. Sigma attempts used
  gain 0-1/201 points, 1,200 x 2, 250 us relaxation, and 4,000 SSF shots/state.
  All accepted pi gains are within 0.625-0.675: Q4 spans
  0.646707-0.651682 and Q6 spans 0.641732-0.656656. Selected sigma ranges are
  0.053267-0.096998 us for Q4 and 0.049594-0.132845 us for Q6.
- T1 used 0.01-60 us/241 points; T2R used 0-30 us/301 points at 0.5 MHz;
  T2E used 0.01-60 us/301 points at 0.5 MHz. Standard coherence averaging was
  1,200 x 2 with one saved 4,000-shot/state SSF calibration per curve. The
  targeted Q6 -7.5/-10 mA 2,000 x 4 retry preserved both candidates and chose
  the higher saved-population R2 independently for T1/T2R/T2E: all -7.5 mA
  priors were retained and all -10 mA high-statistics candidates were selected.
- Final manifest audit: 18/18 complete points, all 180 expected HDF5/plot
  artifact paths present, every saved SSF array length 4,000. Lowest selected
  R2 is the retained Q6 -7.5 mA T2R value 0.553; this is explicitly documented
  rather than hidden by normalization or an inferior retry.
- Heatmaps place current on x, actual QICK delay in us on y, and saved signed
  SSF population on color. Native pulse-dependent delay centers are rendered
  with one explicit midpoint-edged column per current, with no interpolation.
  A fixed -0.30 to 1.50 color scale and fixed three-point display mean are used;
  canonical HDF5 remains unchanged.
- Replaced the provisional entries in
  `system_config.PUCQ4_FLUX_PULSE_CALIBRATION` with all 18 final corrected
  frequencies, selected sigmas/pi gains, readout parameters, and coherence
  values. The six-row zero-current pulse arrays were not changed.
- Extended the reusable PowerPoint from 48 to 60 slides. Slides 49-60 contain
  the fresh Ramsey corrections, per-current sigma plots/tables, final Rabi
  curves, and T1/T2R/T2E maps/curves for both devices. The builder initially
  failed because the new Ramsey averaging summary was referenced before local
  assignment; the summary is now computed from each manifest row before slide
  creation. The canonical deck was open but fully saved, so only that exact
  presentation was briefly closed, rebuilt from its retained 48-slide backup,
  reopened, and exported. All 12 1600x900 renders passed visual inspection.

## 2026-09-08: promote current-indexed calibration into runtime configs

- Objective: make the final two-Yokogawa calibration reusable by normal
  scripts while preserving the established six-qubit config shape seen by the
  QICK programs. This was an offline configuration task; no source, attenuator,
  QICK, Pyro, Visdom, or `M:` data write was performed.
- Made `system_config.PUCQ4_FLUX_CALIBRATION` the final authoritative 18-row
  table and added the qspec gain/length fields from
  `single_sweep_qspec_manifest.json`. Added strict supported-current checking,
  base+offset consistency checks, and a resolver that combines current-specific
  Q4/M5 and Q6/M6 values with the four fixed rows into six-element lists.
- Added independent `YOKO3_CURRENT_MA` and `YOKO4_CURRENT_MA` selectors to
  `round_robin_benchmark.py`, plus matching environment controls. The selected
  values now set corrected f01, qspec gain/length, sigma, pi gain, resonator
  base, and readout offset/gain/length. The benchmark appends the currents to
  its notes and logs the complete selected calibration before acquisition.
- Added experiment-config recentering for normal qspec plus the existing
  f01-centered spectroscopy and chevron variants. Updated `build_state.py` so
  current-indexed qspec/readout lengths and other readout lists become scalars
  for the selected qubit; `res_length_ge` is copied to the QICK-facing
  `res_length` key.
- Updated the self-contained `pucq4_config.py` bootstrap defaults to the same
  final 0 mA M5/M6 frequencies, readout gains/offsets/lengths, while keeping
  the nonzero-current table centralized in `system_config.py`.
- Updated the shared flux subprocess launcher, all map controllers that inherit
  it, the alternating readout optimizer, and the explicit Q6 cross-readout
  diagnostic to propagate the selected currents. The older literal campaign
  seeds are overwritten at import from the shared final table so future resume
  or recalibration work cannot silently start from the superseded fits.
- Zero-current defaults changed for Q4/M5 from 7084.77785 to
  7084.280775100 MHz f01 and from 9014.95 MHz base / 9015.39 MHz physical
  readout to 9014.74 / 9014.94 MHz. Readout length/gain/offset changed from
  5 us / 0.33 / +0.44 MHz to 6 us / 0.65 / +0.20 MHz; sigma/pi changed from
  0.055 us / 0.6486907 to 0.056940825 us / 0.646706952.
- Zero-current defaults changed for Q6/M6 from 6522.54018 to
  6522.416914100 MHz f01 and from 9059.45 MHz base / 9059.41 MHz physical
  readout to 9059.26 / 9058.86 MHz. Readout length/gain/offset changed from
  3 us / 0.60 / -0.04 MHz to 10 us / 0.95 / -0.40 MHz; sigma/pi changed from
  0.062 us / 0.6486907 to 0.062451228 us / 0.646706952.
- Verification: all 18 runtime rows matched the final Ramsey/sigma/coherence
  and single-sweep-qspec manifests within the stored precision. Pure resolver
  checks covered all grid points, unsupported-current rejection, experiment
  recentering, campaign seed synchronization, and scalar QICK-state
  specialization. A monkeypatched constructor check proved invalid current is
  rejected before the proxy call. `py_compile` passed for the changed runtime
  path and all four downstream map controllers; the offline harness parsed all
  123 Python files with no imports or hardware contact. Focused IDE inspection
  found no errors (the two largest files reached the inspection time budget
  after returning no errors), and `git diff --check` reported no whitespace
  errors.

## 2026-09-08: append corrected-frequency lookup summaries

- Objective: add final-deck plots that match sigma, T1, T2R, and T2E to the
  corrected qubit frequency measured at the same current for both tunable
  devices, while retaining current as the primary x axis and showing the
  nonlinear frequency correspondence explicitly.
- Extended `tools/build_pucq4_sigma_coherence_maps.py` with a manifest-only
  summary path. It verifies that each row's selected-Rabi and coherence drive
  frequencies match its Ramsey-corrected `f01`, then creates one plot per
  metric/device. Bottom ticks are current in mA; top ticks are the exact
  same-row corrected frequencies in MHz and are labeled as a nonlinear lookup,
  not a coordinate transform. `frequency_axis_summary_provenance.json` stores
  the full 18-row pairing and confirms that no raw data were modified.
- Extended `tools/build_pucq4_first_light_ppt.py` with the idempotent
  `--append-frequency-summary` path. The canonical deck grew from 60 to 64
  slides: Q4 sigma/T1 and T2R/T2E are slides 61-62, and Q6 equivalents are
  slides 63-64. The original 60-slide file was copied to the dated pre-update
  backup before the canonical path was replaced.
- Validation: all eight source plots passed visual inspection; PowerPoint COM
  opened the canonical deck as 64 slides and exported slides 61-64 at
  1600x900. All four native renders were visually inspected with titles,
  current ticks, corrected-frequency labels, point values, units, and footer
  provenance legible. No experiment, QICK, Pyro, Visdom, Yokogawa, or raw-data
  operation was run.

## 2026-09-08: append frequency-only sigma/coherence summaries

- Objective: add a second final-deck view of sigma, T1, T2R, and T2E against
  corrected qubit frequency alone, ordered by frequency rather than bias.
- Added a manifest-only `--frequency-only-summary` analysis path. It preserves
  the same-row corrected-frequency/metric pairing, sorts each device's nine
  rows strictly by corrected `f01`, rejects duplicate or non-increasing
  frequency coordinates, and generates eight plots with MHz as the sole x
  axis. No current axis or current tick annotation is included.
- Added the idempotent `--append-frequency-only-summary` deck path. The
  canonical presentation grew from 64 to 68 slides: Q4 occupies slides 65-66
  and Q6 occupies slides 67-68. The 64-slide input was retained in the dated
  pre-frequency-only backup before replacement.
- Validation: the provenance contains 9/9 sorted rows and four existing plots
  for each device, with no raw-data mutation. PowerPoint COM opened the
  canonical file as 68 slides and exported slides 65-68 at 1600x900; all four
  renders passed visual inspection. The updated deck was reopened for the
  user. No experiment, QICK, Pyro, Visdom, Yokogawa, or raw-data operation was
  run.

### Renewed M4 search from the supplied VNA deck (2026-09-08, started)

- User explicitly authorized hardware spectroscopy, readout adjustments, and
  progress/final-curve updates to the existing first-light PowerPoint. Goal:
  reproducibly identify M4/index 3/displayed Q4, physical number unknown. Physical
  Q4 is already assigned to M5 and is not the missing row. No flux sweep, broad
  device recalibration, or changes to the underlying qspec program are in scope.
- Re-read both supplied decks. The Downloads deck contains the M4 5575 MHz fge
  seed, 5466 MHz gf/2, 218 MHz anharmonicity, 9000 MHz resonator, 20-30 MHz VNA
  linewidth, and -30 to -40 dBm VNA spectroscopy powers. It contains no original
  M4 two-tone trace. These powers are not calibrated QICK DAC gains. The live
  first-light deck is the 68-slide file under pucq4_first_light/ (the concatenated
  path in the request does not exist). Existing slides and data will be retained.
- Initial exact script: pucq4_12_m4_search.py, guarded main; explicit --execute
  required. Plan m4_s01: res_spec only, one pass; M1/index 0 control at
  8917-8922 MHz/101 points, gain 0.48/10 us/1000 reps; M4/index 3 at
  8995-9003 MHz/161 points for gains 0.30, 0.10, 0.03, 0.60, 0.95.
  Reps respectively 1000, 3000, 6000, 1000, 1000; one round; 10 us readout
  except 25 us at 0.03; 20 us relaxation. Duration limit 20 minutes, progress
  every 20 frequency points. Source queries require CURR/0 A/0.010 A range/output
  off and make no source writes. Nominal existing attenuators: resonator 10/15,
  qubit 5/4, ADC 17 dB; no attenuation writes. Finite pulses only.
- Canonical root M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/
  pucq4_first_light/m4_s01/<timestamp>/; longest preflight artifact 193 characters.
  Data_H5 labels Res_ge/QSpec_ge; every saved frequency/I/Q vector is read back
  and compared exactly before proceeding. Abort on unexpected source state,
  acquisition/connection errors, malformed arrays, or save failure.
- Acceptance requires a visible, sampled, in-bounds line reproduced at the same
  frequency, verified plot/HDF5/config provenance, then Rabi/readout calibration
  and canonical round-robin population spectroscopy for the final accepted plot.
  A fit candidate or noise-normalized fluctuation is not a found qubit.
- Context/configuration finding: build_state now derives res_length from
  res_length_ge, so new diagnostics set and assert both keys; changing only the
  scalar readout length silently retains the prior per-row value. Source code
  and nominal labels alone do not establish the hardware pulse length.
- Offline verifier passed (124 Python files); git diff --check passed. The first
  deck-text probe hit Windows cp1252 on Greek alpha; setting PYTHONIOENCODING=utf-8
  resolved this environment issue without modifying source artifacts.
- First hardware checkpoint: m4_s01/2026-09-08_12-50-20 completed six
  resonator sweeps. M4 has a resolved dip near 9000 MHz across gains 0.1-0.95;
  the 0.03-gain slope estimator was noise/edge dominated and was not accepted.
  All HDF5 I/Q/frequency arrays matched their acquisition; sources remained zero/off.
- m4_s02/2026-09-08_12-53-37: M3 weak-drive control (gain 0.002/20 us,
  1000 reps, 5554.33447-5560.33447 MHz/301 points, RO 8974.92 MHz/.41/12 us)
  was rejected: the joint-IQ width collapsed to one 0.01998 MHz bin. Layer:
  verification; no M4 null inference was drawn from this control.
- m4_s03/2026-09-08_12-55-46: M3 res dip was 8975.0 MHz; stronger M1
  control (gain .01/20 us, 2000 reps, 5513.69373-5521.69373 MHz/401 points,
  RO 8919.31 MHz/.48/10 us) resolved 5517.689621 MHz, FWHM .174420 MHz,
  joint-IQ amplitude/noise 56.94. The saved raw trace visibly resolves the line.
- Next bounded plan m4_s04: qspec only, M4/index 3, 5525-5675 MHz/3001
  points (.05 MHz requested), 1000 reps x1, drive .01/20 us, relaxation100 us;
  physical readout 8999.75 then 9000.25 MHz at .60/10 us. The two-tone parking
  comparison follows the measured M4 slopes; it is not a frequency fit result.
  Expected 13 minutes total, limit20 minutes, same sources/attenuators/save and
  abort rules. No hardware-independent calibration values were changed.
- Existing deck now has 70 slides; slides69-70 document the VNA starting point
  and fresh power-dependent resonator evidence. Pre-update copies retained.
- m4_s04/2026-09-08_12-59-26 completed both 3001-point M4 sweeps. Neither
  readout slope shows a resolved line. Maximum multi-width diagnostic scores
  were 4.86 and 4.82, compared with 83.2 for the M1 control; these filtered
  background scores are screening diagnostics, not calibrated populations.
  A weak near-seed fluctuation in the first trace fits 5575.4211 MHz,
  FWHM0.1147 MHz, joint-IQ amplitude/noise3.46; the opposite slope does not
  clearly reproduce it. It remains unaccepted.
- Next plan m4_s06 is one bounded independent repeat: M4/index3 qspec only,
  5573.4-5577.4 MHz, 401 points/.01 MHz requested, 8000 reps x1, .01 drive
  gain/20 us, 8999.75 MHz readout/.60 gain/10 us, 100 us relaxation. Expected
  about7 minutes, limit10 minutes; same zero/off sources, nominal attenuators,
  finite-pulse behavior, canonical paths, and abort conditions as above.
  A clean reproduction is required before Rabi or configuration promotion.
- m4_s06/2026-09-08_13-14-08 rejected the 5575.42 MHz candidate. The
  8000-average repeat is visibly flat; exploratory FWHM collapsed from .1147 to
  .01001 MHz (one realized .00999 MHz bin), and joint-IQ amplitude/noise fell
  to2.93. Matching a center while width collapses is not reproduction. Do not
  retry/promote this fluctuation as a qubit. Both source readbacks stayed zero/off.
- Next plan m4_s05 (prepared earlier, executed after s06): res_spec only,
  M4/index3, 8997-9003 MHz/121 points, gain.95, readout lengths1 and3 us,
  3000 reps x1, 20 us relaxation, estimated under1 minute/limit5 minutes.
  Purpose: choose measured readout-sensitive points for a shorter integration,
  testing whether a fast-decaying qubit response was diluted by10 us averaging.
  No qubit drive, source writes, or attenuation changes in this resonator stage.
- Short-readout validation: m4_s05/2026-09-08_13-22-25 resolves the M4
  resonance with 1 us and 3 us windows at readout gain 0.95. The 1 us slope
  points are 8999.65 and 9000.20 MHz. m4_s07/2026-09-08_13-23-47 uses the
  known M1 line as a matched 1 us control: 5515.69-5519.69 MHz/201 points,
  2000 reps x1, drive 0.01/20 us, readout 8919.31 MHz/0.48/1 us, relaxation
  100 us. It resolves 5517.693473 MHz, FWHM 0.149566 MHz, joint-IQ SNR 16.90.
- Next m4_s08: M4/index 3 qspec only, physical readout 9000.20 then
  8999.65 MHz, gain 0.95/1 us; qubit drive 5525-5675 MHz/1501 points
  (0.1 MHz requested), gain 0.10/20 us, 2000 reps x1, relaxation 100 us.
  Expected 12.2 minutes total, limit 16 minutes. Hypothesis is a weak/rapidly
  decaying excitation diluted by the longer readout. Same source checks,
  nominal attenuators, finite pulses, path limits, and abort/save rules apply.
- m4_s08 first short-readout trace (9000.20 MHz/.95/1 us, drive .10/20 us)
  saved at 13:32:49 shows a visibly resolved feature near 5594.4 MHz in both
  unsmoothed quadratures. Multi-width screening score is 20.90. This is a new
  candidate, distinct from the rejected 5575.42 MHz fluctuation. Opposite-slope
  acquisition and independent drive-off/power tests are required before acceptance.
- Next bounded plan m4_s09, after s08 completes: M4/index 3 qspec only,
  5589.4-5599.4 MHz/401 points, 2500 reps x1, drive gains .10, 0, .03, .01
  in that order, 20 us pulse, RO 9000.20 MHz/.95/1 us, 100 us relaxation.
  Expected 8.2 minutes, limit 12 minutes; same nominal attenuators, zero/off
  flux-source requirements, finite pulses, canonical HDF5, and abort rules.
  The zero-drive trace must lack the line, and a lower-power sampled line must
  reproduce its center; no shared parameter is promoted from this candidate.
- m4_s08 completed both readout slopes and saved both raw datasets. Independent
  shared-Lorentzian fits give 5594.453776 MHz/1.355167 MHz FWHM at 9000.20 MHz
  readout, and 5594.468818 MHz/1.468581 MHz at 8999.65 MHz. Both are resolved
  in unsmoothed I and Q. Source readbacks remained zero/off. Slide 75 records
  the candidate; m4_s09 validation started at 13:39:21.
- Corrected round_robin_benchmark.py's runtime readout-length handoff with one
  assignment to res_length_ge alongside the existing res_length assignment.
  This ensures the already-supported length override reaches build_state/QICK.
  No default values, run flags, paths, or low-level pulse programs changed.
  Offline harness passed (125 Python files), and the two-line focused delta
  against the pre-task working copy plus git diff --check were reviewed.
- s09 drive-off control has no resolved line. At the driven trace's fixed
  5594.424109 MHz center/1.364596 MHz width, off amplitudes are I=-0.03181
  +/-0.01633 and Q=-0.00182 +/-0.01660 a.u. (linear least-squares standard
  errors), compared with driven amplitudes -0.67588/+0.44522 a.u.
  The unconstrained off fit collapses to one bin and is rejected.
- s09 gain .03 reproduces 5594.494423 MHz with FWHM .657068 MHz and
  joint-IQ SNR 5.98, visibly resolved in raw I/Q. The gain .01 scan is ongoing.
- s09 completed at 13:47:29 with all four HDF5/plots verified and sources
  zero/off. Gain .01 has a weak feature consistent with the established center
  (5594.4663 MHz, exploratory width .3444 MHz, raw-IQ SNR 2.01); it is not
  sufficient by itself as the final clean low-power line. The independent .10
  repeat, two readout slopes, flat drive-off trace, and resolved .03 response
  justify proceeding to pulse calibration at 5594.49 MHz.
- Next exact hardware script: round_robin_benchmark.py via the temporary
  run_rr_m4.py source-checking launcher; one round, only rabi enabled,
  M4/index 3, sub-study m4_rabi01. Runtime qubit frequency 5575 MHz seed ->
  5594.49 MHz measured spectroscopy center; Gaussian sigma stays .100 us
  (existing seed, 4*sigma=.400 us). Rabi gain 0-.85/171 points, 5000 reps x1,
  100 us relaxation; RO base8999.95/offset+.25=9000.20 MHz, gain.95/1 us.
  Raw Rabi scaling is disabled; fitted pi must be visibly supported before
  SSF/population calibration. Retain the old .7 seed only if fitting fails so
  raw data persist, explicitly marked as uncalibrated. Expected 90 seconds,
  limit5 minutes; same source/attenuator/abort rules. Output canonical
  first-light/m4_rabi01/<timestamp>/; intended plot and HDF5 paths <260 chars.
- M4 coherent control confirmed by m4_rabi01/2026-09-08_13-49-23. Normal
  round-robin Rabi shows a clear oscillation in both I/Q with pi gain
  .512390893 at sigma .100 us; selected-I fit R2=.98146. Canonical rabi_ge
  HDF5 and normal plot both exist; all 171 raw/axis/fit values are finite and
  the stored config confirms 5594.49 MHz and actual 1 us readout. Sources zero/off.
- Next m4_rabi02: same one-pass round-robin Rabi configuration and physical
  readout; sigma .100 -> .0788 us (Gaussian length .400 -> .3152 us), chosen
  from .100*.512390893/.65 to target pi gain .65. Gain0-.85/171 points,
  5000 reps x1, relaxation100 us; scaling enabled with 50,000 fresh shots per
  state/100 us relaxation using the newly fitted pi. Expected under2 minutes,
  limit5 minutes. Plot/HDF5 paths max195 characters; no other device, flux,
  attenuator, or shared configuration changes.
- m4_rabi02/2026-09-08_13-53-16 confirms pi gain .646706952 at sigma
  .0788 us, inside the .6-.7 window. Fresh 50k-shot centroids are
  g=(-1.28017,-.05834), e=(-2.77801,.63346) a.u.; calibrated Rabi spans
  -.0669 to1.0563 before any clipping (none applied). Both raw and population
  normal-RR plots and canonical rabi_ge HDF5 were saved; sources zero/off.
- To improve final spectroscopy, next run normal round_robin_benchmark.py
  single-shot only, M4/index3, one pass for lengths1,2,3,5,8,12 us at fixed
  9000.20 MHz/gain.95, qfreq5594.49 MHz, sigma.0788 us/pi.646706952.
  20,000 shots per state/100 us relaxation; sub-studies m4_len01,...,m4_len12.
  New optional PUCQ4_SSF_SHOTS and PUCQ4_SSF_RELAX_DELAY_US overrides use
  the existing scalar/list helper and leave interactive defaults unchanged.
  The launcher runs stages sequentially with zero/off source checks each time;
  5-minute series bound, same nominal attenuation and canonical SS HDF5/plots.
  Choose the shortest measured length within .02 of the best fidelity, then
  measure a bounded gain/frequency grid at that length.
- Length survey completed and all canonical SSF files contain 20k finite I/Q
  values per state with the intended actual res_length_ge/res_length. Repository
  SSF contrast at1,2,3,5,8,12 us was .1562,.2146,.2594,.2904,.25725,.25695.
  Choose5 us (shortest measured length within .02 of best), not the1 us locator.
  Both sources remained zero/off at every stage. Full results/length plot:
  m4_len12/2026-09-08_13-57-27/documentation/ssf_summary.json, ssf_length.png.
- Next gain/frequency grid: fifteen sequential one-pass RR ss-only stages
  m4_gf01..15, M4/index3, frequencies8999.6,8999.8,9000.0,9000.2,9000.4 MHz
  crossed with readout gains.55,.75,.95, fixed5 us; same qubit5594.49 MHz,
  sigma.0788 us/pi.646706952, 20k shots/state,100 us relaxation. 5-minute
  series bound; source/attenuation/save/abort rules unchanged. Follow with a
  second length pass near the selected optimum to check convergence.
- The 15-point grid finished with full raw SSF/HDF5 and safe source readbacks.
  Best measured point: physical8999.8 MHz (offset-.15 MHz from8999.95 base),
  gain.95/5 us, repository SSF contrast.3138, versus.29545 at9000.2 MHz/.95.
  Full grid: m4_gf15/2026-09-08_14-00-41/documentation/ssf_summary.json.
- Second length pass m4_lenb03..07: 3,4,5,6,7 us at8999.8 MHz/gain.95,
  otherwise identical calibrated qubit and20k shots/state/100 us relaxation.
  Five sequential normal-RR ss-only stages,3-minute bound; unchanged source,
  nominal attenuation, save checks, and abort rules.
- Second length pass at8999.8 MHz/.95 gives SSF .2835,.29101,.31655,
  .31575,.31705 for3,4,5,6,7 us. Select5 us, the shortest point on the
  plateau. Next final local grid m4_gfb01..09:8999.7,8999.8,8999.9 MHz x
  gains.85,.90,.95 at5 us, identical20k-shot/state calibrated SSF settings.
  Nine sequential RR ss-only stages,3-minute bound, same safety/save checks.
  Stop alternating if improvement over8999.8MHz/.95/5us is below.02 contrast.
- Final local grid confirms8999.8 MHz/.95/5 us as best, SSF contrast.31815;
  neighboring8999.9MHz/.95 gives.3148. Improvement over the prior selected
  repeat.31655 is only.00160, so alternating readout optimization is complete
  at the stated.02 threshold. All sources zero/off and raw datasets verified.
- Final exact hardware plan: round_robin_benchmark.py via run_rr_m4.py,
  one pass on M4/index3 with only q_spec,rabi,ss enabled; sub-study m4_final.
  Seed5594.49 MHz, calibrated qspec gain.03/20 us, +/-1.5 MHz (5592.99-
  5595.99 MHz),301 points/.01 MHz requested,10000 reps x1,100 us relaxation.
  Fresh50k-shot/state SSF calibration uses sigma.0788us/pi.646706952;
  readout8999.8MHz/.95/5us. Rabi then uses the measured qspec center,
  gains0-.85/171 points,5000 reps x1,100 us relaxation and fresh50k-shot/state
  calibration; final standalone SSF50k/state follows. Nominal attenuators and
  zero/off flux states unchanged. Expected8 minutes, monitored limit12 minutes.
  Normal RR plot and Data_h5/qspec_ge,rabi_ge,ss_ge datasets required, with
  finite sampled line, calibrated peak population and independent raw-IQ fit.
  Report actual linewidth honestly: the current M4 power series is broader
  than the preferred.1-.2MHz and must not be rescaled to manufacture acceptance.
- m4_final/2026-09-08_14-06-43 completed with all raw HDF5 and plots saved,
  but failed final spectroscopy shape acceptance. Layer: measurement/analysis.
  Main fitted line5594.221022 MHz/FWHM.397565 MHz has an additional response
  near the high-frequency edge around5595.8 MHz and a poor global single-line
  R2=.55908. Independent IQ center5594.18960 MHz also differs from earlier
  ~5594.49MHz centers. Frequency drift/multiple-response cause is unresolved.
  The same-pass Rabi is excellent (R2=.99734, pi.646706952), and SSF=.32116;
  qubit identity remains confirmed, but the wide plot is not the final curve.
  Full report retained in documentation/final_validation.json; sources zero/off.
- Next m4_clean01: new one-pass RR q_spec,rabi,ss acquisition, seeded from
 5594.221022 MHz, qspec5593.221022-5595.221022MHz/201points (.01MHz),
  5000 reps x1, drive gain.03->.05/20us,100us relaxation. Fresh narrower
  acquisition isolates the resolved line; it does not crop or overwrite the
  prior raw trace. Readout remains8999.8MHz/.95/5us; sigma.0788us/pi.646706952,
  50k-shot/state calibrations; subsequent Rabi/SSF unchanged and use the new
  qspec center. Expected4.2minutes, monitored limit8minutes; same source,
  attenuation, canonical save, artifact-length and abort requirements.
- Final clean evidence accepted: m4_clean01/2026-09-08_14-17-43. Normal
  round-robin center5594.388459689 MHz, FWHM.723922232 MHz; independent
  calibrated-population fit peak.37266, baseline.03013, R2=.97065, center
  fit standard error.00351 MHz. Raw-IQ fit5594.39975 MHz, SNR23.19. Same-pass
  Rabi R2=.99674/pi.646706952 and SSF contrast.32834 pass. All frequency,
  raw-IQ/population arrays and50k-shot calibration/SSF arrays are finite;
  qspec_ge/rabi_ge/ss_ge HDF5 plus normal plots exist. Both sources zero/off.
- The preferred.1-.2 MHz linewidth was not reached; the final true width is
  .724 MHz. Centers shifted across5594.22-.49 MHz during this session. This
  is a found and controllable M4 qubit with a clean spectrum, not evidence of
  narrow/stable linewidth or measured T1/T2. Physical number remains unknown.
- Shared M4/index3 configuration handoff (only this row changed):

  | Parameter | Old | New | Unit / use |
  | --- | ---: | ---: | --- |
  | Qubit frequency | 5575 seed / measured NaN | 5594.38846 | MHz, g-e spectroscopy |
  | Qspec drive gain | 0.010 | 0.050 | DAC gain; 20 us pulse unchanged |
  | Readout gain | 0.30 | 0.95 | DAC gain |
  | Readout length | 10 | 5 | us |
  | Readout offset | +0.40 | -0.15 | MHz relative to unchanged8999.95 MHz base |
  | Physical readout tone | 9000.35 | 8999.80 | MHz |
  | Gaussian sigma | 0.100 seed | 0.0788 | us; pulse length0.400->0.3152 us |
  | Pi amplitude | 0.700 seed | 0.646706952 | DAC gain, measured Rabi |

  Promoted in system_config.py, expt_config.py and pucq4_config.py. Historical
  VNA arrays remain historical. Default qubit selection/run flags and all five
  other rows remain unchanged. No M4 ef/coherence calibration is inferred.
- Completion: the existing PowerPoint now has 81 slides, with accepted Rabi/SSF
  on slide 80 and the clean final calibrated spectroscopy curve on slide 81.
  Search/control/rejected-candidate progress is preserved on slides 69-79.
  Native PowerPoint rendering was inspected; a final title was shortened to
  avoid overflow. The original 68 slides and all 301 original slide/media/notes
  parts are byte-for-byte unchanged against the initial backup. The exact
  final deck SHA and config checks are in final documentation/completion_audit.json.
- Final offline verification passed: tools/harness_check.py parsed 125 Python
  files without importing acquisition modules or contacting hardware;
  git diff --check was clean. Focused pre-task snapshot diffs were reviewed.
  AST comparison confirms that all non-M4 code/settings in the three shared
  configs are unchanged; pure config resolution agrees with accepted raw
  results. Default RR selection/run flags remain unchanged. Prior user changes
  and all raw measurement artifacts were preserved.
- Final hardware validation was the completed m4_clean01 round-robin
  qspec/Rabi/SSF pass, distinct from the offline checks. Its documentation
  contains M4_handoff.md (old/new settings, limitations, exact raw paths),
  final_validation.json, completion_audit.json, plan/guard/stdout/stderr,
  the acquisition launcher and reporting/audit source. Sources finished zero/off.
  The requested discovery and clean-spectrum PowerPoint endpoint is complete;
  narrower linewidth, physical-number assignment and coherence are future work.

## 2026-09-08 M4 narrow-line/readout/coherence and chip summary

- User explicitly authorized M4 low-power spectroscopy, increased averaging,
  alternating readout-length and gain/frequency SSF optimization, Gaussian sigma
  calibration for pi near0.65, T1/T2R/T2E, updates to existing deck tables, and a
  separate complete chip-summary deck with all six traces and both Yoko maps.
- Identity remains M4/index3/displayedQ4, physical number unknown. Physical
  Q4 is M5/Yoko3; physical Q6 is M6/Yoko4. Use existing validated current-sweep
  artifacts for the requested Yoko-map summary. No Yoko current sweep is needed
  to measure this fixed-frequency row. Preserve earlier measurements and all
  pre-existing repository changes; do not replace low-level QICK programs.
- Acceptance: narrowed, resolved sampled spectroscopy with calibrated population;
  iterative SSF plateau, repeatable selected readout; measured Rabi pi0.6-0.7;
  bounded, resolved T1/T2 fits with uncertainty and >=4points per T2 fringe;
  final normal one-pass RR res/qspec/Rabi/SSF/T1/T2R/T2E plots and canonical HDF5.
  Report actual linewidth/quality if a preferred target proves unattainable.
  Both decks must have measurement-backed tables/configuration and exact source
  provenance, with visibly unknown fields where no measurement exists.
- First finite hardware envelope: round_robin_benchmark.py via the guarded
  run_rr_m4.py launcher; one pass, M4/index3 only, q_spec flag. m4_n01 uses
  seed5594.38846MHz,5593.38846-5595.38846MHz/201points, gain0.01 (old0.05),
  drive20us,16000reps x1,100us relaxation, fresh50k-shot/state calibration.
  RO8999.80MHz atgain0.95/5us; sigma0.0788us/pi0.646706952. Nominal unchanged
  attenuators res10/15dB, qubit5/4dB,ADC17dB. Both Yokogawas must read CURR,
  0A,10mA range,outputoff before/after; no source/attenuator writes planned.
  Output under canonical pucq4_first_light/m4_n01/<timestamp>/; launcher
  computes plot/HDF5 path lengths and rejects >=260chars. Expected7minutes,
  monitored limit12minutes, abort on source/acquisition/buffer/save errors.
  Save plot and raw HDF5 independently; reject unbounded or one-bin fits.
- Offline preflight passed:125Pythonfiles parsed; old interactive Python
  processes predate this task and have no QICK/Pyro connection. Existing browser
  connections to9090 are separate from acquisition. Config/deck backups retained.
- m4_n01/2026-09-08_15-49-05 completed: gain0.01/20us,16000reps,
  center5594.806873MHz, FWHM0.541814MHz, calibrated peak0.09331, baseline-0.01512,
  R2.91962; independent raw-IQ SNR14.11. Raw HDF5 and both normal plots verified;
  sources CURR/0A/off. Frequency shifted from the earlier5594.39MHz measurement.
- Next m4_n02: normal RR q_spec only, one pass/index3, gain0.005/20us,
  5594.006873-5595.606873MHz/161points,32000reps x1,100us relaxation,50k-state
  calibration at updated seed5594.806873MHz. Same8999.8MHz/.95/5us readout,
  sigma.0788us/pi.646707, nominal attenuators, source/readback/save checks.
  Expected11minutes; monitored limit15minutes. Purpose is to test power
  dependence of linewidth, not presume narrow-line acceptance.
- m4_n02/2026-09-08_15-57-30 completed atgain0.005/20us/32000reps:
  center5594.796624MHz,FWHM.586003MHz,peak.02389,R2.75177,rawIQ SNR5.40.
  Halving drive reduced signal without narrowing the line; this is a resolved
  low-SNR diagnostic, not the final clean spectroscopy. Source/arrays/saves pass.
  n01 gain.01 is the better current low-power tracking setting. Coherence will
  help assess intrinsic dephasing versus acquisition-time frequency variation.
- Next m4_rc01, one normal RR pass res_spec,rabi,ss, index3 at5594.796624MHz.
  Res8998.45-9001.45MHz/151points,.02MHz step,600reps x1,100usrelaxation;
  initial base8999.95MHz, gain.95/5us, offset-.15MHz after measured res minimum.
  Provisional Rabi sigma.0788us/pi.646707,0-.85gain/171points,5000reps x1,
  fresh50k-state calibration; SSF50k/state. Purpose is reliable preparation
  for readout optimization; final Rabi is repeated after optimization. Expected
  3minutes, limit6minutes. Same canonical path/source/attenuation/abort checks.
- Environment-only launch failure: PowerShell policy blocks dot-sourcing .ps1 helpers. No hardware started. Retried with the same reviewed helper definitions supplied inline; no policy setting was changed.
- Report-preview verification found a null SSF metric in a completed historical current row. Fixed optional-metric rendering to say not recorded; required current-row raw data/settings remain verified. No acquisition result was changed.
- m4_rc01/2026-09-08_16-11-21 passed:resminimum8999.91MHz, RabiR2.99632,
  measuredpi.646707atsigma.0788us, SSFcontrast.26542at8999.76MHz/.95/5us.
  Fullres/Rabi/SSF HDF5/plots verified, sources zero/off.
- Next m4_clen01..07:seven sequentialnormalRR ss-only stages,M4/index3,
  lengths2,3,4,5,6,8,10us atfixed8999.80MHz/gain.95,offset-.11MHz relative
  tonebase8999.91MHz. Each30000shots/state,100usrelaxation,qfreq5594.796624MHz,
  sigma.0788us/pi.646707. Expected2minutes,serieslimit5minutes; sources and
  canonical save checks before/after each stage. Choose shortest plateau,
  then gain/frequency grid. Existing nominal attenuation unchanged.
- Length comparison m4_clen01..07:contrasts at2/3/4/5/6/8/10us were
  .2072/.24277/.26940/.27390/.28580/.26764/.25690. Initial best6us; neighboring
  shorter lengths will be retested after the2D grid to choose the plateau.
- Next m4_cg01..15:3gains(.70,.85,.95)x5physical frequencies
  (8999.6,.7,.8,.9,9000.0MHz),length6us,30000shots/state. One bounded RR ss-only
  stage per point,index3, same5594.796624MHz/sigma.0788us/pi.646707 and100us
  relaxation. Offset recorded relative8999.91MHz. Expected3minutes,limit5.
  Sources/attenuation/canonical saving/abort checks unchanged.
- m4_cg01..15 grid complete and raw shots/configs verified. Best8999.70MHz/
  .95/6us givesSSF.31450; neighboring8999.6/.8/.9/9000.0 atgain.95 give
  .27774/.26570/.25824/.27557. The newbest needs an independent repeat.
- Next m4_dlen01..10:lengths4,5,6,7,8,8,7,6,5,4us,forward/reverseorder,
  fixed8999.70MHz/gain.95,offset-.21MHz from8999.91base. Each50000shots/state,
  100usrelaxation,qfreq5594.796624MHz,sigma.0788us/pi.646707. One normalRR
  ss-only/index3pass perstage,sourceszero/off. Expected2.5minutes,limit5.
  Choose shortest length whose repeated contrast is within.01 of the maximum
  mean and within observed repeat uncertainty; then a final local2D grid.
- Repeated m4_dlen01..10 means at4/5/6/7/8us: .251254/.277683/.269941/.286931/.284461.
  Select5us as the shortest length within.01 of the maximum repeated mean;
  its .01024 forward/reverse spread also exceeds the .00925 mean difference.
  The initial6us/.3145 grid maximum did not repeat and is not accepted.
- Next m4_dg01..06: final local2D grid at5us,8999.65/.70/.75MHz,
  gains.90/.95,50000shots/state,100usrelaxation. One normalRR ss-only/index3
  pass perstage, qfreq5594.796624MHz,sigma.0788us/pi.646707; offset from
 8999.91MHz base. Both Yoko sources must remain CURR/0A/off before and after;
  nominal attenuators unchanged. Expected1.5minutes,serieslimit4minutes.
  Canonical unique paths, raw/plot validation and previous abort conditions apply.
- m4_dg01..06 passed: final5us local grid again selects8999.70MHz/.95,
  contrast.29478, versus.25298/.26670at8999.65/.75MHz. Gain.90 gives
  .22566/.24322/.25996. Repeated5us evidence spans.27256-.29478, so stop
  alternating with no significant stable improvement above the plateau.
- Verification-schema failure: enhanced audit incorrectly assumed old qspec
  files saved their raw SSF calibration shots. They did not; Rabi/coherence do.
  Added the four existing in-memory qspec calibration arrays to canonical RR
  HDF5 collection, without changing the QICK pulse program or acquisition.
  Historical qspec is explicitly marked not reconstructable from HDF5 alone;
  final qspec will require complete shots and population reconstruction.
- Next m4_cp01: one normal RR pass flags rabi,ss,t1,t2r,t2e, index3 only.
  SelectedRO8999.70MHz/.95/5us,base8999.91/offset-.21MHz; qfreq5594.796624MHz,
  sigma.0788us,pi seed.646707. Rabi0-.85/171points,3000reps x1,
  100usrelaxation,50kstatecal; SSF50k/state. T1.01-40us/161points;
  T2R0-6us/241points,2MHzdetuning; T2E.01-20us/401points,2MHzdetuning.
  Eachcoherence2000reps x1,250usrelaxation,fresh20kshots/state affinecalibration.
  This is a pilot to resolve time scales; extend or shorten bounds if required.
  Expected8.5minutes, monitoredlimit15minutes. Abort on source/acquisition/
  fit/persistence failure; both sources CURR/0A/off before and after, unchanged
  nominal attenuation, unique canonical paths with longest-path guard.
- Reporting preview exposed two historical Q6 wide-map visibility flags below
  threshold4 at-10/-2.5mA. PUCQ4_STATE continuous-map closure already records
  their independent line fits and SNR3.76/3.78. Summary retains actual maps and
  explicitly marks these rows; no false all-threshold-pass claim or reacquisition.
  The map footer now records20MHz baseline,1500bins,3pixel line display and
  gamma.55; these are display processing, not raw data transformations.
- Existing reusable deck84slides:83-84 contain the measured initial and repeated
  readout optimization evidence. Final coherence/table update remains pending.
- m4_cp01 completed with all five canonical datasets/normal plots and source
  zero/off checks. RabiR2.99483/pi.646707; SSF.26788. Independent fits reproduce
  T1.879921us(+/-.03864 stored),R2.93795;T2R.457470us(+/-.03683),R2.75284;
  T2E1.205298us(+/-.10031),R2.59520. Pilot T2 curves fail final R2 acceptance,
  so increase averaging and acquire shorter, densely sampled delay ranges.
  Ramsey fringe2.781611MHz versus2MHz phase detuning gives a diagnostic
  frequency-correction seed5594.015013MHz; final spectroscopy must locate f01.
- Actual pilot coherence calibration was50kshots/state at100usrelaxation,
  despite the requested20k/250us: intervening Rabi/SSF overrides changed the
  shared Readout_Optimization config. Raw arrays were complete and population
  reconstruction passed. Moved coherence-specific SSF overrides to immediately
  before T1/T2, after Rabi/SSF. Audit records requested/actual count mismatch;
  final-report acceptance requires exact count match. No QICK pulse change.
- Next m4_f01: final candidate one-pass normalRR flags
  res_spec,q_spec,rabi,ss,t1,t2r,t2e, index3 only. Qfreq state-preparation seed
  5594.015013MHz comes from pilot Ramsey; qspec will measure and propagate f01.
  Res actual center8999.95MHz from shared fixed-row state, +/-1.5MHz/151points,
  .02MHzstep,600reps x1; foundminimum plus-.21MHz offset defines physicalRO.
  ExpectedRO8999.70MHz/.95/5us. Sources remain0A/off; attenuators unchanged.
  Qspec5593-5596MHz/201points,gain.01/20us,32000reps x1,100usrelaxation,
  fresh50kstatecal. Rabi0-.85/171points,5000reps x1,100usrelaxation,
  sigma.0788us/pi seed.646707,50kstatecal; SSF50k/state.
  T1.01-5us/201points,6000reps x1;T2R0-3us/201points,8000reps x1,
  2MHzphase detuning;T2E.01-5us/201points,8000reps x1,2MHzphase detuning.
  All coherence100usrelaxation and fresh50kshots/state at100usrelaxation.
  Shorter ranges and denser averaging replace the rejected pilot T2 tails.
  Expected25minutes, monitoredlimit32minutes; abort on acquisition/source/
  out-of-envelope/fit/save failure. Guarded unique canonical paths<260chars;
  all required datasets, calibrated plots and independent fits must pass.
- During f01 acquisition, copied its unchanged acquisition/config sources to
  %TEMP%/pucq4_m4_coh_20260908/f01_source_snapshot and recorded SHA256 hashes.
  This preserves the exact source state before later accepted config promotion.
  Source snapshot timing is after launch; no acquisition/config source changed
  between f01 launch and the snapshot. Reporting/audit utilities changed offline.
- Hardware-free regression check extracted only the coherence-override AST and
  setter. It verifies the block runs after SSF/before T1, restores20k/250us
  after prior50k/100us, and leaves settings untouched when coherence is off.
  No acquisition modules imported. This tests the diagnosed override-order bug.
- f01 spectroscopy finished: normal fit5594.046778405543MHz,FWHM.641605MHz,
  gain.01/20us/32000reps. The actual calibrated-population image is visibly
  resolved, peak roughly.07; independent HDF5 audit waits for batch save.
  Rabi/coherence now use this fresh frequency, not the earlier5594.80MHz seed.
  Linewidth0.1-.2MHz target is not reached and will be explicitly documented.
- f01 Rabi pi gain is.651681621193at78.8ns (target~.65 achieved); finalSSF
  contrast.308at8999.73MHz/.95/5us. T1/T2R/T2E use that newly measured pi gain.
  This revalidatesSSF after the frequency shift. Sharedconfigpromotion waits
  for complete raw/fit checks; no parameters have been promoted midacquisition.
- Final m4_f01 passes all raw/config/source checks and independent fits.
  f01=5594.046778406MHz; FWHM.641590MHz; peakPe.06947; qspecR2.90742/IQSNR11.83.
  RabiR2.99678; sigma78.8ns; measuredpi.651681621193; SSFcontrast.30808.
  T1=.826635+-.009267us(R2.99521),T2R=.479394+-.015215us(R2.95616),
  T2E=1.702556+-.058259us(R2.95306). T2fringes33.90/20.07points each.
  All five population curves reconstruct from exactly50kcalibration shots/state.
  Realized resaxis8998.20-9001.20MHz; measured minimum8999.94 and RO8999.73MHz.
  Both sourcesCURR/0A/off. Acquisition source snapshot and final settings JSON saved.
- Promoted M4 only: f01 5594.38846 ->5594.046778406MHz; qspecgain.05->.01;
  pi.646706952->.651681621193; resbase8999.95->8999.94MHz;
  offset-.15->-.21MHz, physicalRO8999.80->8999.73MHz. Sigma78.8ns,
  gain.95 and length5us revalidated. Other rows and RR selection unchanged.
- Saved echo plots contain a historical title typo T2R despite being T2E data.
  Source plotting strings corrected for future runs; original evidence images
  preserved unchanged and their PowerPoint captions explicitly identify the typo.
- Completion: updated the original characterization opening table and appended
  final M4 results plus all six normal trace types; the deck now has 93 slides.
  Its 86-slide pre-update backup is retained. The preservation audit checked
  398 previous slide/media/notes parts; only the requested opening table/note
  relationship changed. Subsequent viewport enlargement touched only new M4
  resonator slide 88, with another backup and exact ZIP-part comparison.
- Created `PUCQ4_chip_summary_20260908.pptx`, 39 slides: all 36 zero-current
  traces, six-row results/settings/averaging tables, final repeated M4 readout
  optimization, and all six requested maps for each of the two Yokos, plus
  selected sigma maps and per-current settings. All six resonator viewport
  crops preserve embedded image bytes, data and axes while removing unused
  panels. Every source trace image passed a SHA256 embedding check.
- Native PowerPoint exported all 39 summary slides and the eight updated/new
  characterization slides. Zero-current tables, all six trace galleries, M4
  optimization and all 12 requested Yoko maps passed visual inspection; flux
  settings tables also passed the earlier native preview. Provenance, sources,
  completion audit and render evidence are saved beside the new summary.
- Final offline verification passed in the configured Python 3.9.13 Conda SDK:
  `tools/harness_check.py` parsed 127 Python files, with manifest/structure
  checks passing and no hardware contact. `git -c core.safecrlf=false diff
  --check` passed. Focused diffs against pre-task snapshots confirm M4-only
  parameter promotion, the two bounded RR persistence/override fixes, and
  label-only echo corrections. Existing unrelated user changes were preserved.
- `M4_coherence_handoff_20260908.md`, the selected-settings JSON and
  `harness_engineering/PUCQ4_STATE.md` now carry the accepted frequency,
  coherence/errors, readout, sigma/pi, exact provenance and final source state.
  Hardware validation passed in `m4_f01/2026-09-08_16-45-24`; the preferred
  0.1-0.2 MHz linewidth and 0.3-0.5 population targets remain explicitly unmet.
  Both sources ended in current mode at 0 A, outputs off. No further hardware
  operation was needed for final reporting.

## 2026-09-08 PowerPoint clutter cleanup

- User requested cleanup of duplicate PowerPoints. Scope: only the PUCQ4
  first-light report folder; preserve the two active decks and all raw data.
- Moved 29 backup/intermediate decks (803.6 MiB) to `_PowerPoint_archive/`.
  All source/destination paths were checked within the approved folder, below
  the Windows path limit, and free of destination collisions. No files deleted.
- Verified every moved file and both retained decks by SHA256. Exactly two
  PowerPoints remain at the top level: `PUCQ4_chip_summary_20260908.pptx`
  and `PUCQ4_first_light_characterization.pptx`.
- `archive_manifest_20260908_180106.json` records old/new paths, sizes, hashes
  and completion. Current summary provenance and completion-audit references
  now point to archived backups; their prior JSON copies are in the archive.
  Persistent state directs future report backups into the archive folder.
- File organization only: no Python/acquisition commands, hardware contact,
  measurement changes or edits to either active PowerPoint.

## 2026-09-08 zero-current linewidth, sigma and coherence refresh

- User authorized lowering drive toward 0.1 MHz FWHM for the summary's broad
  Q4/Q6 traces, then new Rabi/sigma/pi and T1/T2R/T2E, with updated summary
  traces and configuration/result tables. Scope is zero current only; no
  nonzero-current map reacquisition, attenuation changes or pulse-program edits.
- Identity check: the 0.6 MHz curve is displayed Q4/M4/index 3 (physical number
  unknown), and the 0.3 MHz curve is Q6/M6/index 5 (physical Q6/Yoko 4).
  Physical Q4 is M5/index 4 and already has a displayed 0.1 MHz curve. An
  optional clarification was posted; M6 is common to both interpretations.
- Preserve raw data and both active decks; backups belong in
  `_PowerPoint_archive`. Acceptance: distinguishable, densely resolved qspec,
  independently fitted FWHM with uncertainty and repeated low-power evidence;
  aim 0.10 MHz (approximately 0.08-0.12), do not force fits if a floor remains.
  Recalibrate Gaussian sigma for pi gain near 0.65 (0.6-0.7 acceptance),
  revalidate SSF, then accept bounded independent coherence fits with R2 >0.85,
  relative fit error <20% and at least four points per T2 fringe. Final evidence
  must be a normal one-pass RR chain with complete HDF5 and population plots.
- First stage `line_6a`: reviewed temporary `run_rr_zero.py`, copied from the
  previous finite M4 wrapper, executes the normal `round_robin_benchmark.py`
  exactly once, index 5 only, flags res_spec,q_spec. It is hardware operation,
  not an import-safe test. Q6/M6/Yoko4, other Yoko3/M5 held at 0 A; wrapper
  checks both sources CURR, 0 A, 0.01 A range, outputs off before and after.
- Res seed 9059.26 MHz, -1.5..+1.5 MHz/151 points/.02 MHz step, 600 x1 reps,
  100 us relaxation. Refreshed minimum plus offset -0.40 MHz defines readout;
  gain .95, length 10 us. Qspec 6521.4-6523.4 MHz, 201 points, gain .003,
  20 us, 8000 x1 averages, 100 us relaxation; fresh 30k shots/state calibration.
  Preparation seed f01 6522.4169141 MHz, sigma .062451228 us, pi .646706952.
  Nominal attenuation res10/15 dB, qubit5/4 dB, ADC17 dB unchanged.
- Output `pucq4_first_light/line_6a/<timestamp>/`, canonical study_data/Data_h5
  and documentation. Wrapper calculates longest plot/HDF5 path and requires
  <260 characters before connection; labels are short and paths unique.
  Expected 4-5 minutes, monitored limit 9 minutes. Abort on instrument/source,
  missing/raw-save, out-of-envelope, or invalid-fit failures. Exact plan and
  stdout/stderr under `%TEMP%/pucq4_line_20260908`.
- Offline harness passed (127 Python files). Audit utility now accepts an
  explicit qubit index/guard filename, reports linewidth uncertainty, and
  checks all requested dataset labels persisted; old M4 defaults are preserved.
- `line_6a/2026-09-08_18-15-14` passed raw/calibration/source checks.
  Measured res minimum 9058.84 MHz -> physical readout 9058.44 MHz.
  Qspec gain .003 gives f01 6522.520687 MHz, FWHM .202220 +/-.008502 MHz,
  population R2 .92117, peak .17155, joint-IQ width .201157/SNR13.86.
  Actual qspec step .00999069 MHz; 30k state calibration fully reconstructs.
- Next `line_6b`: same guarded one-pass RR, M6/index5, flags q_spec,rabi,ss,
  both sources CURR/0A/off, unchanged nominal attenuation. Physical readout
  9058.44 MHz/.95/10us, base9058.84/offset-.40. Qspec6521.92-6523.12MHz,
  121points, gain.0015/20us,32000x1 averages,100us relaxation,30kstatecal.
  Rabi at newly measured qspec frequency: sigma.062451228us, pi seed.646706952,
  requested gain0-.85/171points,3000x1,100us relaxation,30kstatecal;SSF30k/state.
  Expected10minutes, limit14; same source/fit/persistence abort conditions,
  unique canonical paths<260 characters. No permanent calibration promotion.
- User additionally requested FWHM(MHz) versus current and actual qspec gain in
  both Yoko tables. Offline linewidth audit now examines both selected local
  spectra and continuous-map data. A wide-sweep fit must center on its observed
  peak, not a stale pre-sweep frequency seed; displaced low-SNR fits are flagged
  and never silently reported as validated linewidths. Nonzero currents are
  analyzed from existing raw data only. Newly created files can briefly be absent
  from PyCharm's index; query the SDK using an established module file.
- `line_6b/2026-09-08_18-20-32` passed. Gain .0015 gives f01 6522.527958 MHz,
  FWHM .195655 +/-.009207 MHz, R2 .94191, peak .12068, joint-IQ width .196533
  and SNR16.32. Rabi R2 .99719; measured pi .651681621 at sigma .062451228us.
  SSF contrast .3638; exact30k calibration shots/state reconstruct populations.
  Linewidth did not narrow significantly relative to .003; both sources0A/off.
- Next `line_6c`: final candidate one-pass RR, M6/index5, all seven flags.
  Gain .00075,20us,64000x1,6522.03-6523.03MHz/101points; 100us relaxation.
  Fresh res seed9058.84MHz +/-1.5/151/.02,600x1; minimum-.40MHz readout,
  gain.95/10us. Measured-area sigma prediction .062612797us (old sigma times
  measuredpi/.65); pi seed.651681621; fresh Rabi0-.85/171,5000x1.
  All population calibrations and SSF50kshots/state/100us. T1.01-30us/201,
  T2R0-10us/201, T2E.01-30us/301; all5000x1/100us; bothT2phase1MHz.
  Both sourcesCURR/0A/.01Arange/off before/after; same nominal attenuators.
  Expected24minutes, limit32; same instrument/source/fit/raw-save abort gates.
  Unique canonical paths<260 checked before hardware. Shared calibration is
  not changed until final fit/raw verification. Qspec title-only formatting
  now includes explicit MHz and three decimals; pulse program unchanged.
- Offline report preview caught duplicate slide-part names after replacing slide
  IDs: python-pptx allocates new slide names from the current count, so removal
  followed by another append reused a live part name. Verification-layer defect:
  renumber live slide/notes parts before save and assert ZIP names are unique.
  Repeated preview passed; native PowerPoint rendered all requested sample pages.
  No active deck was changed by either preview. Yoko width tables render with
  matched current/gain/pulse length and explicit weak/coarse-fit markers.
- line_6c stopped at18:48 after the population Lorentzian fit returned x0
  infeasible. Actual sweep span0.999069MHz is below the default initial
  half-width1MHz, outside the existing fit bounds. Raw-I/Q PNG shows0.192MHz,
  but RR skipped collection on fit failure and wrote None placeholder datasets.
  This is incomplete, excluded from accepted tables/configs. Both sources
  verified CURR/0A/off. Raw PNG and placeholder HDF5 retained unchanged.
- Failure layer: verification/analysis initialization and orchestration
  persistence. Clamp only the population fitter's initial half-width into its
  existing bounds; pulse programs unchanged by AST comparison to pre-task copy.
  Offline regression reproduced the failure for spans0.2 and0.999069MHz, then
  recovered exact0.1MHz synthetic lines after fix; line_6a/6b measured centers
  and widths agree before/after within0.0001MHz. Full audit rejects line_6c.
- Repeat line_6d: exact line_6c physical configuration, sweep bounds, nominal
  attenuators,50kstatecal, all seven flags, M6/index5, one RR pass, both Yokos0/off.
  Only sub-study and fit initializer change. Output unique line_6d/<timestamp>
  canonical hierarchy; path<260 before connection. Expected24min/limit32min;
  abort on source/instrument/fit/raw-save failure. No calibration promotion yet.
- Early RR collection now preserves res/qspec arrays immediately after their
  acquisition, before a rejected fit can continue the qubit loop. The existing
  end-of-round Data_H5 writer, dataset labels and hierarchy remain unchanged;
  unavailable fits stay invalid, and no seed is promoted. The already-running
  line_6d uses its frozen pre-persistence-fix source snapshot; subsequent M4
  uses the fix. Offline replay of the exact rejected-fit branch preserved raw
  line_6b IQ/population/calibration arrays in a temporary canonical HDF5, with
  dependent work skipped and fit saved invalid. No hardware in that check.
- Visual review of every archived current linewidth fit is complete (temporary
  q4/q6 narrow/wide audit panels). Q6 wide -10,-7.5,-2.5mA fits fail the
  conservative SNR gate; show unresolved in visible tables rather than promote
  diagnostic numeric widths. Candidate fits and exact raw sources stay in notes.
  Coarse but distinguishable widths have a sampling marker. Each fit now records
  its exact input seed/window so the reported width is reproducible.
- line_6d spectroscopy now fits normally: f016522.511432598MHz,
  FWHM.211486594MHz at gain.00075/20us/64000x1. Population PNG is clean;
  full independent audit awaits the remaining Rabi/SSF/coherence data. The
  low-power goal intentionally accepts populations below the usual0.3-0.5
  planning window when the line is well resolved; no population is rescaled
  to meet that window. Both current sources remain at the verified0/off state.
- Next prepared run line_4a (not concurrent with M6): displayedQ4/M4/index3,
  physical number unknown; one normal RR pass with res_spec,q_spec,rabi,ss,
  t1,t2r,t2e. BothYokosCURR/0A/.01Arange/off before/after. Resseed8999.94MHz,
  +/-1.5MHz/151points/.02MHz/600x1/100us; measured minimum-.21MHz readout,
  gain.95/5us. Qspec5592.85-5595.25MHz/121points/.02MHz, gain.005/20us,
  96000x1/100us; f0seed5594.046778406MHz. Sigma.079003864us from prior
  .0788us times measuredpi/.65; piseed.651681621. Rabi0-.85/171,5000x1/100us.
  All calibrations/SSF50kshots per state/100us. T1.01-5us/201/6000x1;
  T2R0-3us/201/8000x1/2MHz phase;T2E.01-5us/201/8000x1/2MHz phase.
  Nominalres10/15dB,qubit5/4dB,ADC17dB. Unique canonical line_4a/<timestamp>
  with bothplot/HDF5paths<260. Expected35minutes/limit45; abort on instrument,
  source, fit, envelope or persistence failure. Uses verified early collection
  fix and bounded population-fit initializer; pulse programs unchanged.
- line_6d/2026-09-08_18-51-46 completed19:16 with all seven canonical HDF5
  datasets and normal population plots. Source guards before/after pass at0A/off.
  Independent audit: res9058.88MHz,RO9058.48/.95/10us; f016522.511431896MHz,
  FWHM.211476902+/-.010673046MHz, R2.94611,peakPe.05836,jointIQSNR14.17.
  Sigma.062612796us, measuredpi.646706952329,RabiR2.99848;SSF.37234.
  T1=6.435511765+/-.077919841us,R2.99518;T2R=1.803285514+/-.046256470us,
  R2.96934;T2E=5.466466197+/-.107096618us,R2.97257. Independent fits agree,
  all50k state-calibration arrays reconstruct populations; T2fringes19.95/10.01
  samples. Width target0.10MHz not reached; lower-power runs plateau near0.20.
- Starting the previously recorded line_4a envelope after M6 fully completed.
- Promoted validated M6 zero-current values only: qfreq6522.416914100 ->
  6522.511432598MHz; qspecgain.006 ->.00075 (the prior summary trace used.01),
  sigma.062451228 ->.062612796167us; pi.646706952 ->.646706952329;
  resbase9059.26 ->9058.88MHz and physicalRO9058.86 ->9058.48MHz, offset-.40,
  gain.95/length10us unchanged. T1/T2R/T2E5.535403/1.534897/4.997961 ->
  6.435511765/1.803285514/5.466466197us. Width.211476902+/-.010673046MHz,
  qspec64000x1/101 andSSF.37234 retained as calibration metadata.
  Shared zero-current frequency aliases agree; every nonzero calibration row
  remains unchanged. Reusable acquisition plans and raw-fit provenance copied
  into summary artifacts/linewidth_validation; no new active PowerPoint.
- line_4a preliminary qspec at gain .005/20 us/96000 x 1 remains broad:
  normal population fit f01 5593.886652463 MHz, FWHM .676325783 MHz, peak
  population about .02. The weaker response is visibly noisy despite the higher
  averaging. Full raw-data audit awaits the end-of-pass writer; do not promote
  this candidate on the basis of the plot. Rabi gives pi .651681621 at sigma
  .079003864231 us, and SSF contrast is about .290. Coherence is still running.
- Prepared next bounded line_4b envelope (start only after line_4a ends):
  displayed Q4/M4/index3, physical number unknown; both Yoko sources verified
  CURR/0 A/.01 A range/output off before and after. One normal RR pass with
  res_spec,q_spec,rabi,ss,t1,t2r,t2e. Qspec gain .0075 is between the previous
  .01 and noisy .005 measurements; 128000 x 1 averages, 20 us pulse, 97 points
  5592.69-5595.09 MHz (.025 MHz requested step), relaxation 100 us; seed is the
  new 5593.886652463 MHz measurement. This repeat targets a clear final trace,
  not a claim that the .10 MHz target was attained. Res seed 8999.90 MHz,
  +/-1.5 MHz/151/.02 MHz/600 x 1; RO measured minimum-.21 MHz, .95 gain, 5 us.
  Retain the just-validated sigma .079003864231 us and pi seed .651681621;
  Rabi 0-.85/171/5000 x 1. T1 .01-5 us/201/6000 x 1; T2R 0-3 us/201/8000 x 1;
  T2E .01-5 us/201/8000 x 1; both T2 phases 2 MHz. All relaxation 100 us;
  all state calibrations and SSF 50000 shots/state. Nominal res attenuation
  10/15 dB, qubit 5/4 dB, ADC17 dB unchanged. Canonical line_4b/<timestamp>,
  maximum planned plot/HDF5 path 193 characters. Expected36 minutes/limit48;
  abort on instrument, source, fit, envelope or persistence failure. Exact plan
  prepared in task TEMP; no additional hardware launched concurrently.
- line_4a completed at19:52 with all seven HDF5 datasets, calibrated population
  reconstruction from50k shots/state, and both sources zero/off. Independent
  qspec FWHM .676300578 +/- .097119769 MHz, f01 5593.886650751 MHz, peak Pe
  .020235986, R2 .65147 and joint-IQ SNR4.216: retain as diagnostic, not the
  final clean-line result. Rabi R2 .99701/pi .651681621; SSF .28998.
  T1 .832705653 +/- .008973820 us; T2R .409754446 +/- .014489408 us;
  T2E 1.550060567 +/- .044891558 us; all independent coherence fits passed.
- Started the recorded line_4b plan at19:53:40, PID25376, after line_4a exited.
  Run line_4b/2026-09-08_19-53-42; maximum monitored deadline20:41:40.
  Pre-run source/config/plan snapshots and SHA256 records are preserved in its
  documentation/acquisition_source. Both fresh source readbacks are CURR,
  0 A/.01 A range, output off. No permanent M4 calibration promotion yet.
- line_4b completed20:30:28 with all seven canonical HDF5 datasets and normal
  plots, and both final source readbacks CURR/0 A/.01 A range/output off.
  Independent qspec f01 5594.109362449 MHz, FWHM .682091469 +/- .046102016 MHz,
  R2 .91257, peak Pe .034759, joint-IQ SNR10.306. The saved pulse frequency is
  5594.109362350 MHz. Rabi R2 .99728, pi .651681621 at sigma .079003864231 us;
  SSF contrast .30966. T1 .816978890 +/- .007846887 us (R2 .99654),
  T2R .471189184 +/- .016288197 us (R2 .94889), T2E1.757433500 +/- .053691661 us
  (R2 .96290). Independent coherence fits agree; every50k state calibration
  reconstructs its population, and T2 fringes have34.10/20.09 samples.
  All seven final M4 plots were visually inspected, as were all seven M6 plots.
- Promoted M4 qfreq5594.046778406 ->5594.109362350 MHz, qspecgain .01 ->.0075,
  sigma .0788 ->.079003864231 us. Pi .651681621193, resbase8999.94 MHz and
  RO8999.73 MHz/offset-.21 MHz/gain.95/length5 us are revalidated unchanged.
  Shared frequency aliases agree. The weak drive requires128000 x1/97 points;
  exact high-average plans remain separate from default all-row timing arrays.
  Other fixed rows and all nonzero flux calibration entries remain unchanged.
- Both requested .10 MHz linewidth targets were not attained. Lower gain gave
  M6 about .20-.21 MHz across .003/.0015/.00075, and M4 about .68 MHz in the
  latest .005/.0075 acquisitions. The clean final M4 curve uses .0075; noisy
  .005 data remain diagnostic. The final weak-drive stationary comparison
  1/(pi T2R) is .676 MHz for M4 and .177 MHz for M6, not an intrinsic-linewidth
  fit or an assertion that all sequential coherence fits form a stationary model.
- Final review copy audit passed:42 summary/114 characterization slides,
  all36 source trace images unchanged,14 original Yoko map/sigma images intact,
  92 unchanged historical slides, and every current/gain/length/FWHM pairing
  checked. New zero-current values and shared sigma/pi/readout aliases agree.
  Native PowerPoint previews reviewed, including larger labeled M4/M6 SSF plots.
  Active decks are ready for replacement; backups remain in _PowerPoint_archive.
- Replaced the two active decks in place at20:38:46. Independent audit of the
  saved files passes:42/114 slides,36 raw trace-image hashes,14 unchanged map/
  sigma images,92 preserved historical slides, all current/gain/FWHM pairings,
  sigma/pi/readout tables, and unchanged unrelated/nonzero shared settings.
  Native PowerPoint exported64 final review images into summary artifacts/
  native_render_20260908_linewidth. Reviewed final coherence galleries and
  Yoko pulse/wide-width tables; earlier native previews cover opening tables,
  qspec, labeled SSF, drive comparison, and refreshed historical slides.
  Only PUCQ4_chip_summary_20260908.pptx and PUCQ4_first_light_characterization.pptx
  remain active in the root. Backups and prior audits are in _PowerPoint_archive.
- Completion audit and reusable handoff are saved in summary artifacts;
  summary_provenance.json points to both and retains the previous deck-update
  metadata explicitly as history. Offline harness passed129 Python files and
  git diff --check passed; focused pre-task source diffs reviewed. New linewidth
  updater has an explicit39/93 source-count guard so this one-time migration
  cannot accidentally append duplicate tables/slides to the finished42/114 decks.

## 2026-09-08: user-authorized all-six round-robin reproduction (active)

- Exact task/envelope: docs/pucq4-round-robin-validation.md. User allows config
  and orchestration changes, expressly forbids underlying classes/QICK changes,
  requests documented Yoko-bias validation and one new PowerPoint.
- Baseline source audit found default M4 exclusion, coherence population scaling
  disabled, and low-gain/short-lifetime acquisition settings confined to prior
  temporary plans. Added reusable per-row/current averaging/timing profiles in
  expt_config; RR now selects all six, scales coherence, and snapshots Exp Config
  per row instead of retaining mutable references to later rows' settings.
- Added set_yoko_bias near RR's top. Only the documented grid; current-mode/range
  checks; 0.5 mA/s ramps; verified setpoints; zero/off cleanup. RR now owns both
  sources for each invocation. Existing external controllers must not assume a
  bias remains applied after RR returns. Retained Q4/M5 -10 mA's 600 s settling.
- Hardware-free verification: 81 selector combinations x six profiles, actual
  requested sampling checks, simulated ramp/zero/out-of-grid cases, protected
  section/driver/build_state hashes and unchanged QICK_experiment class AST.
  Safe harness currently passes131 Python files; diff whitespace check passes.
- Default all-six zero pass started21:14:32 in rr6_0mA with no PUCQ4 overrides,
  deadline9000 s, guard59668 / acquisition60780. Both GS210 sources verified
  current/10 mA range/0 A/output off. Old inactive Python windows preserved.
- Interim M1/M2 frequency, Rabi and T1 reproduce prior values. M1 Ramsey has
  late-time revival and needs repeat/residual review. M3 frequency/Rabi agree,
  but SSF0.176 versus0.372 at readout8974.94 MHz (prior8974.89 MHz) requires
  configuration retuning after this pass. Do not promote interim plot values
  before canonical raw HDF5 is saved and audited.
- M4 plot-level T1/T2R/T2E0.81/0.43/1.64 us match0.817/0.471/1.757 us;
  M5 plot-level12.83/5.96/14.64 us match12.44/6.12/13.85 us. M5 spectroscopy
  7084.477746 MHz is within the earlier qspec linewidth of7084.565737 MHz,
  although it differs from the later Ramsey-corrected pulse setting7084.280775.
  The comparison audit now pairs qspec with prior qspec and records pulse-
  frequency changes separately; unresolved old lines are not valid references.
- Verification/context finding: archived50-point0--1 us Ramsey corrections
  saved0.01953125 us actual steps. The installed QICK compiler independently
  truncates phase/time sweep increments. The inferred effective phase slope is
  4.179592 MHz versus the nominal4 MHz correction constant, potentially explaining
  much of the frequency offset. New raw Ramsey data must confirm this; inferred
  timing-adjusted candidates are not promoted to calibration.
- The archived M1 Ramsey trace already shows a revival. An exploratory two-
  frequency fit over the entire recorded window improves R2 from0.8525 to0.9128
  with separation0.01887 MHz. This does not identify a microscopic cause or
  replace canonical T2R. Its old HDF5 lacks Population; the diagnostic uses the
  saved-fit I quadrature and labels its affine display normalization explicitly.
- Follow-up bias helper validation now depends only on allowed current values,
  so zeroing cannot fail because unrelated pulse-calibration metadata is invalid.
  A simulated first-source mode error verifies that zeroing still attempts the
  second source. This follow-up code is not the already-running baseline process.

### 2026-09-09: round-robin reproduction checkpoint (still acquiring)

- Authorized scope is the six-row default RR, both documented nine-current flux
  grids and fixed-row endpoint controls, plus one new validation PPT. Underlying
  classes/programs/drivers and both reference decks remain unchanged by this task.
- The all-six zero-current pass (rr6_0mA/2026-09-08_21-14-32) saved all seven
  canonical datasets. Raw/fit comparison accepted M2/M4/M5/M6. M1's Ramsey
  revival and M3's weak baseline SSF required follow-up; no failed row was
  automatically promoted. The full M1/M3 repeat is now rrv13_0b/2026-09-08_23-57-09,
  controller26812, RR59244, 6000 s limit. Both sources verified0 A/output off.
- Verification layer: repeating M3's exact18 us/.95/8974.63 MHz settings changed
  SSF contrast.6924->.52705 with identical saved scalar configs. Ground-shot
  means/RMS stayed stable; the prepared-excited cloud moved. This is measured
  variability, not proof of a microscopic cause. At9 us, four repeats gave
  .6022 mean/.0289 sample SD;18 us .5978/.0852;21 us .6375/.0818. Retain9 us
  for full-chain validation instead of selecting a lone maximum. Repeated
  alternating length/gain-frequency grids and HDF5 sources are in the new
  validation artifacts. The measured dip at9 us/.95 is8974.84 MHz; offset-.21
  recreates the chosen8974.63 MHz physical readout tone.
- M1 repeated at historical500 Ramsey reps retains a strong late revival,
  with canonical T2R9.44 us; clean spectroscopy/Rabi/T1 still reproduce. A
  phase-detuning diagnostic is next after the current two-row pass. Do not
  shorten a fit window or change fit gates solely to force the old16.53 us.
- Environment layer: PowerPoint COM was blocked by an existing modal read-error
  dialog. Acknowledge only the verified error's OK action, preserving user
  presentations/processes/security settings. The new M: validation PPT then
  opened and rendered natively. Acquisition tables now use15-row chunks.
- Environment layer: forcing Git core.autocrlf=false makes this CRLF working
  tree appear as whole-file whitespace errors. Use the repository's normal
  configuration for git diff --check; it passes. No line-ending rewrite was
  needed. The safe harness parses131 files; protected hashes/class ASTs and81x6
  offline profile checks pass. UTF-8 BOM-aware manifest loading also supports
  metadata written by Windows PowerShell without losing later audit promotion.

- 00:24 verification correction: the M1/M3 acquisition finished successfully,
  with both sources0/off and all canonical HDF5 files. The OFFLINE joint-IQ
  audit failed because its historical half-width seed0.45 MHz exceeded M3's
  new~0.25 MHz half-window. Filter starting widths to the existing bounds and
  bound center offsets; do not widen the measured/fit window or change its
  objective. Fresh raw audit then passes M3 quality checks, with only the
  old-versus-new T2R comparison flagged (59.8376+/-1.0931 us, R2.98605). The
  independent fit agrees and has10.0 samples/fringe. Saved wider M4/M6 spectra
  reproduce their earlier joint-IQ centers/widths within1e-5 MHz after the fix.
  No experiment/QICK class or raw data changed. A targeted repeat follows.

Environment/tooling failure (2026-09-09): a read-only TEMP PowerShell monitor
.ps1 was blocked by the machine's script execution policy. Hardware acquisition
continued independently. Read the same JSON/log state with ordinary inline
PowerShell commands; do not change execution policy or invoke script contents
to bypass the block. The optional monitor file is unused.

2026-09-09 M5/Y3+2.5 reproduction recovered by configuration-only readout
changes. First full RR passed qspec/Rabi/T1/T2R/T2E but failed SSF. Two
alternating grids and a fresh resonator reference yielded4us/.95/+.14MHz.
Complete repeat rrv5_p025b/2026-09-09_01-21-43 passes every original gate:
f7143.883429323 MHz, FWHM.119323607+/-.001401703 MHz, pi.646706952329,
SSF.5841, T1/R/E10.1960/7.6056/15.5660 us. Raw/plots/source0-off checked;
validated values and linewidth/current/gain/averaging metadata promoted.
Protected source/classes and81x6 profiles still pass; no QICK program changed.
See the new readout-reference failure-mode entry and decision artifact.

### September9 RR readout-frequency reference correction
Configuration layer: fresh resonator dip plus the previous offset changed the
physical readout tone, even though a calibrated absolute frequency was stored.
M5/Y3+2.5 shifted9014.56->9014.16MHz and M6/Y4+2.5 shifted9059.94->9059.58MHz,
with contrast loss. SS-only grids measured recovery at corrected tones. Normal
RR now preserves the saved physical tone and writes its offset from the newdip;
base/offset diagnostics and pre_optimize retain their explicit semantics.
Ninety offline parsed-block cases pass;131-file harness and81x6profile/protected
source checks pass. Full M6+2.5 RR verification launched02:56:51,controller57332,
using the persistent22us/.80/9059.98MHz candidate. No classes/QICK programs changed.

2026-09-09: Verification/reporting. Pending or failed bias attempts must not leak fitted taus, linewidths, or dependent pulse calibrations into accepted summary tables or heatmaps. The RR report builder now applies the same per-experiment dependency gates to zero and bias tables, and leaves pending/review current rows blank on the complete nine-current map. Raw diagnostic attempts remain preserved. Native PowerPoint rendering verified the review row and blank map. All131 offline Python parses,81x6 profile/protected-source checks passed. This changes reporting only; no experiment class/QICK program edits.

2026-09-09 09:35: Verification. Routine qspec audit now also applies the existing strict-repeat raw-I/Q gates: vectorSNR>=6, residual RMS<=2noise sigma, population/raw center difference<=max0.1FWHM/3fitSE, and oneFWHM of baseline on both sides. A visible high-frequency shoulder in M6+7.5 motivates checking model residuals rather than relying only on aggregate R2. All10currentlyacceptedselectedaudits pass these added gates without rewriting their originals. All131offlinefilechecks/whitespace pass. No acquisition class/program changes. The ACTIVE queue_e imported the previous auditor at launch; independently audit its completed outputs with the current module until the queue exits/restarts. Subsequent bias groups use independent shared rows and remain bounded; do not promote M6+7.5 before this review. New sequence processes will load the stricter auditor.

2026-09-09 09:38 environment note: PowerShell5.1 native argument passing stripped embedded double quotes from an otherwise single-quoted Python -c expression, producing NameError in a read-only reference query. The same query succeeded when placed in a small .py file and run with the freshly queried SDK. Prefer file-based Python utilities to complex -c quoting; no hardware or data mutation occurred in the failed query.

### 2026-09-09 - Keep fixed-qubit bias controls separate from zero calibration

Layer: state management and verification. A fixed-row control has current_mA=0 for its own flux mapping even when another qubit's Yokogawa is biased. Review/promotion must key it by both actual source currents and its run; otherwise a confirmed coherence change can overwrite zero-current calibration. The multi-metric review helper now updates only the matching control entry, preserves original audits and repeat decisions, and supports combined independent confirmations with comparison-only issues. An isolated JSON fixture verifies this routing plus rejection of model failures. M1's repeated non-single-exponential Ramsey remains unresolved; a returned fit value is not accepted evidence.

### 2026-09-09 - Separate repeat agreement from historical-band overlap
The M2 endpoint-control T1 repeat agreed with its primary under the unchanged15%-of-mean/3-combined-SE tolerance but returned inside the older30% comparison band. The review records variation and both values, without claiming a persistent increase. Ramsey's two outside-band values agree and confirm that change. Quality/model/actual-axis/calibration gates remain mandatory. Matching-setting reviews now also compare post-ramp settling intervals; differing delays are not silently treated as the same configuration. Q4+10mA's new600s settling check is still a diagnostic configuration change pending hardware validation.

## Validate frequency tracking separately from a constant frequency shift

- Layer: task specification / verification / state. A small statistical fit
  error does not bound variation between successive bias-conditioned spectra.
  Q4/M5+10mA on September9 produced four clean lines spanning0.126345MHz;
  two matching600s-settling scans differed0.027733MHz, exceeding the original
  0.015045MHz fixed-shift test. The strict static test remains failed.
- A separately predeclared review confirmed the acquisition workflow: every
  raw/calibration/fit/edge gate passed, matched-settling lines were within one
  measured linewidth with consistent widths, and two full RR sequences
  reproduced pi/readout/T1/T2R/T2E while propagating each measured qspec center.
  This is validated_tracking, not a constant-frequency calibration. Store the
  full run's values, the observed frequency range, original comparison flags,
  both full chains and all repeats. Never promote a qspec-only frequency into
  the pulse/coherence table without its complete calibration chain.
- The shared row requires_qspec_each_pass. Normal RR enforces qspec before
  pulse measurements at that selected bias; explicit freshly measured carrier
  overrides remain available for diagnostics. No general auditor quality gate
  or experiment class was changed. The report explicitly distinguishes the
  observed frequency range from fit uncertainty and assigns no physical cause.

### September 9 — verification layer: phase changes need complete-chain evidence

Q6/M6 at Yoko 4 +10 mA reproduced historical numeric coherences at 0.5 MHz Ramsey phase, but the full-window residual had a structured component (peak power fraction 0.2827; residual RMS/fit span 0.0427). Do not clear a raw/model finding using a historical-comparison waiver. An otherwise identical 0.75 MHz Ramsey-only acquisition passed the unchanged model gates (0.1899 residual peak fraction), with T2R 5.94 +/- 0.22 us. This supports a configuration-only full-sequence trial, not immediate calibration promotion. Persist only the diagnosed phase setting, keep all delays and samples, and require a new normal RR chain plus any independent comparison check warranted by that chain. No microscopic cause was established and no class or fitter was modified. Full trial rrv6_p100b started 12:29:41; outcome pending.

September 9, context/source-of-truth correction: one offline diagnostic initially guessed its HDF5 timestamp from the echo plot and got FileNotFoundError. No data were changed. Resolve the exact HDF5 path from the saved audit instead; plot and batch-save timestamps can differ by a second. Rerun against audit.experiments.t2_ge.h5 succeeded.

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

### September 9, 16:46 local — Q4 -2.5 mA validated and promoted; Q6 echo repeat active

M5/physical Q4/index4 full rrv5_m025b/2026-09-09_16-25-00 completed in1003.85s, all7 canonical HDF5 and normal plots, both sources current mode0A/off on.01A range. Corrected auditor passes every raw/calibration/model/reference gate. Qspec center6945.112997197333MHz (actual pulse6945.1129971731425), FWHM.119485175735460+/-.001774532792MHz,R2.978698824; old qspec6945.111955770774 differs only.001041MHz. Pi.6467069523286333 at unchanged sigma.058777626us; SSF.5926 versus prior.6183. T1/T2R/T2E12.843022492/3.330588261/9.688102646us, changes+1.42%/-4.61%/+.04%. Normal echo plot reviewed. Selected complete pass promoted16:43:59: carrier6945.02924848->6945.1129971731425MHz,resbase9013.74->9013.70MHz,offset+.32->+.36MHz preserving absoluteRO9014.06/.73/6us,measuredpi precision,coherences/FWHM/counts/provenance. No sigma/drivegain/averaging changes. Earlier failed separate visits and held-bias diagnosis retained; this new complete result supports the old line and no added settling.

ACTIVE m6_m025_fullc.json guard57048/child27940, launched16:44:01, runrrv6_m025c (timestamp in output). Y3zero/Y4-2.5, all7normalflags1pass,2700sdeadline, .5mA/s ramp/noextrasettle/both0offafter. Prepared fullc envelope unchanged: only science difference from fullb is echo1200x2->2400x2; qspec10000x1 and Ramsey2400x2, cal/SSF20k250, same drivegain/sigma/RO and EXPLICIT fullb qspecbounds. Source snapshot includes promoted M5 row, no M6promotion. Both verified0/offbeforelaunch; no hardware overlap. Audit fullc atcompletion; only if soleSSFcomparison issue remains, create exact actual-carrier SSF confirmation and bootstrap/review. Old prepared m6_m025_ssrepeat.json uses fullbcarrier and must not be used blindly. Six remaining negative groups fresh innegative_remaining_6.json; M1 Ramsey/finalall6zero still required.

### September 9, 16:52 local — report verification and M1 diagnostic preparation

Safe 131-file harness and focused whitespace checks passed after Q4 -2.5 mA promotion. The 81 x 6 profile/timing and protected-source checks also passed; underlying classes remain unchanged. The single validation deck was rebuilt with 123 slides and seven outstanding findings. Native PowerPoint rendering passed; Q4 current/linewidth table and the repeated readout-length points were visually reviewed. Both reference PPT SHA256 hashes still match their recorded originals at16:48. The latest persistent scalar change table has167entries relative to the task-start snapshot.

Offline TEMP/predict_m1_midpoint.py predicts a phase-only diagnostic from three archived exploratory two-frequency fits. Corrected phase estimates are .007047132/.005797016/.006303606MHz using actual/requested sampling. Noiseless synthetic midpoint curves can be fitted by the unchanged automatic Ramsey method or its full population starts; frequency-only starts can return negative tau despite excellent R2 and must not be trusted. These are simulated predictions, NOT measurements or promoted parameters. No persistent M1 setting changed. Artifact M1_midpoint_model_prediction.json is diagnostic only, not included as a measured trace. Fresh M1 qspec/Rabi/high-phase Ramsey remains required before choosing the actual diagnostic phase. Preserve all full windows and require actual uncertainty/independent-fit checks.

Q6 fullc remains active (guard57048/child27940); qspec averaging was89% at16:52. No hardware errors reported. All other plans remain unstarted as listed in the16:46checkpoint.

### September 9, 17:04 local — prepare a longer M1 phase diagnostic without changing defaults

M1 timing-design simulation (M1_midpoint_window_prediction.json) uses an archived exploratory pair plus empirical residual resampling, explicitly simulated and not measured. At60us, automatic fits often returned a plausible but biased~15us despite the simulated34us decay; independent agreement remains essential. At120us/601points, full calibrated-population starts recovered~31-34us in all20 repetitions for each midpoint error-.001/0/+.001MHz. This supports a120us diagnostic window with the same.2us requested spacing after fresh M1 qspec/Rabi/high-phase Ramsey. No real M1 phase/sigma/coherence calibration has been promoted; simulation is not a measured PPT trace.

RR now accepts optional single-row PUCQ4_T2R_FIT_STARTS JSON for the unchanged public fitter: f>0, phase, amp, initial_offset, final_offset, all finite; T2/unknown keys are rejected. DefaultNone leaves all normal profiles unchanged. The recorded row profile receives these starts; normal configured-fit path preserves Fit Unseeded, original tau/error and actual final starting-values metadata. No QICK/class/model/pulse change. The finite guard validates the same restrictions before hardware. prepare_exact_repeat now copies recorded fit starts explicitly so diagnostic repeats cannot silently inherit a different persistent profile. FIXTURE_exact_repeat_fit_override.json is offline TEST ONLY and must never launch.

All14actual RR/guard selector cases passed (invalid types/nonfinite/negativef/unknownT2/multiple rows/disabledRamsey rejected; default and other pulse/timing settings unchanged). The existing16 measured-fixture RR fit/plot/raw-preservation checks,81x6profile/protected-source/timing checks,131-file harness and focused whitespace all passed17:04. Active Q6 fullc had imported and recorded its acquisition profile at16:44; its original launch-source bytes remain captured by its guard despite these later orchestration changes. Q6 Ramsey was halfway through its two averages at17:02 and remains healthy; echo follows. No new hardware run started.

### September 9, 17:08 local — require a fresh Q4 -2.5 mA spectrum before pulse measurements

The accepted Q4/M5 -2.5mA full pass matches the historical line, but earlier separate visits moved by more than one linewidth. Added requires_qspec_each_pass=True to this selected current row, using the existing RR precondition already used atY3+10mA. Ordinary all-seven RR behavior and physical parameters are unchanged. Disabling qspec while requesting Rabi/SSF/coherence now needs an explicitly fresh measured carrier; resonator-only remains allowed. This is a reproduction precondition, not a claim of a stable frequency or the separate +10mA tracking-review status. Extended existing guard tests to both Y3-2.5/+10; all81x6profile/protected-source/timing and guard cases passed17:08; focused diff whitespace passed.

Active Q6 fullc is on echo with2400x2 averaging. T1 normal plot shows~6.51us and is clean; all numbers await raw audit. No new Q6 calibration promoted. M1 diagnostic will use a fresh high-phase trace first, then its measured phase midpoint over120us/601points with existing free-fit population starts as justified by the simulation. Source changes are after current child import; old launch bytes remain archived by guard.

### September 9, 17:14 local — Q6 coherences pass; M1 fresh preparation active

Q6/M6/Y4-2.5 fullc rrv6_m025c/2026-09-09_16-44-04 completed in1725.574s with all7canonicalHDF5/normalplots and both sources0A/off,current mode,.01A range. Actualpulse6103.93669863587MHz; qspeccenter6103.936699237719+/-.004490210MHz,FWHM.360744246301313+/-.017382360MHz,R2.819017489 FAIL. RawIQcenter6103.937923772098,width.373386758,SNR7.6990097,residual.9661721sigma PASS; frequency comparison toold6103.798152153059 passes. Rabi pi.6516816211926998/sigma.056022425us passes. Res9057.86/RO9057.86MHz/.8/10us/offset0;SSF.323 versusprior.427 failscomparison. T1 6.506661995+/-.137503462us,R2.976424763;T2R1.204703866+/-.092004776us,R2.902600431;T2E4.260214376+/-.159111585us,R2.907524258. Allcoherenceindependent/residual/window/fringe/referencechecksPASS (+9.97%,+2.65%,+2.64%old). NormalT1/echoplotsvisuallyreviewed. Extraechoaveraging solved its qualityfinding. NO M6-2.5promotion or SSF-onlyreview while qspec qualityfails.

ACTIVE m1_midpoint_prepare.json guard57980/child60016 launched17:14:01. BothYokos0/off,M1/index0,normalqspec/Rabi/SSF/T2R onepass,1800sdeadline. Existingexplicitdocumentedenvelope: qspecseed5517.691628432926MHz +/-.8MHz301/.001gain/20us/5000x1/100us;Rabi0-.85/171/1200x1/250us/sigma.05;RO8919.25/.48/10us;Ramsey0-60us301/.5MHz/1200x1/250us;cal10k250. No fit-start override for this fresh high-phase preparation. Canonicalpaths<210,nominalattenuationargsunchanged,both0/offafter,abortacquisition/save/readbackerrors. Q6guard completed with0/off before launch. Once saved, analyze fresh full-window Ramsey with TEMP/ramsey_beating.py and derive phase_midpoint = programmed_phase - exploratory_mean_frequency*actual_step/requested_step. Plan next M1 diagnostic over120us601 with measured midpoint and optional existing fit starts f approximatelyhalfmeasuredsplitting, phase0,amp1,offsets.5/.5; NO T2 constraint or promotion from simulation. Actual raw/model/uncertainty checks and freshcompletepass remain required.

PREPARED m6_m025_fulld.json (unstarted). Onlysciencechange fromfullc: qspec401x10000->101x40000 atsamewindow/gain/length/statecal. Requested .020MHzspacing still provides>12samples acrossold.253MHzFWHM,18acrosscurrent.361; net4.04Mvs4.01Mshots keepsdurationaboutsamewhilehalvingexpectedperpointnoise. Allotherfullc pulse/readout/averaging/timing values and PRE-Rabi seed are unchanged. All7onepass,2700sdeadline,196charcanonicalpaths,.5mA/sramps,noextrasettle,0/offafter. Do notlaunchduringM1active. NewfullpassandexactSSFconfirmationmayfollow; neverreuseoldpreparedSSrepeatcarrier. Sixremainingnegativegroupsinnegative_remaining_6.json andfinalall6defaultzero remainunstarted.

### September 9, 17:25 local — fresh M1 Ramsey confirms the model issue; longer phase-frame test active

M1 midprep rrv1_midprep/2026-09-09_17-14-06 completed in385.894s, four normalRR canonicalHDF5/plots andboth0A/off. Freshqspec5517.701453579536MHz (actualcarrier5517.701453768203),FWHM.0517816054033125MHz,R2.980807038; Rabi pi.6865043032411646 at sigma.05us, RO8919.25MHz/.48/10us/-.52,SSF.664. Qspec/Rabi/SSF raw/calibration/model/referencechecksPASS. Usualphase.5MHz Ramsey0-60us301/1200x1/250us has savedtau19.610808294+/−1.046682513us,R2.838597974 and structured residuals, so NOTaccepted despite historicaltaucomparisonpassing. Freshdiagnostic M1_midprep_ramsey_diagnostic.json/png uses fullrawtrace: exploratorypair frequencies.499607200705948/.518102371221001MHz,center.508854785963475,splitting.0184951705150529,R2.973927031,no fittedparameterboundhit. Exploratorytau31.636728438us is NOTpromoted or substituted for the canonicalfit.

ACTIVE m1_midpoint_120.json guard45984/child60792 launched17:23:34. BothYokoszero/off,M1/index0,normalRRt2r-only1pass,900sdeadline. Exactfreshcarrier5517.701453768203/pi.6865043032411646/sigma.05/RO8919.25/.48/10/-.52;cal10k250;Ramsey1200x1/250us. Extend0-60us301 to0-120us601 atsame.2usrequestedspacing. Phase .5 -> -.005210647856533801MHz from .5-.5088547859634747*.19856770999999895/.2. Fitstarts JSON f.009247585257526428MHz(halfsplit),phase0,amp1,offsets.5/.5;noT2guess/constraint. ExistingnormalRRfitpath retainsoriginalautomaticfit/tau/error and uses fullrawdata. Neither this phenomenologicalphasechoice nor the exploratoryfit is a calibrationuntilrealquality/repeat/fullchainchecks. Predicted absolute frame5517.696205535389MHz is only arithmetic context, not a runtimeconfig. NO adaptivephase/frame or pulsecarrier override feature was added; currentprofilesstilldefaultM1phase.5,60us,andautomaticfit. Allsettingchangesinthisrunareexplicitdiagnosticenvironmentoverrides. Acquisition45%at17:25:30;noerrors. Waitforguard0/off/HDF5thenauditnormalresult. If clean, requirefreshcompleteM1sequenceandconfirmation beforepersistingphase/window/fitstarts. AnewM1Y3+10fixed-rowcontrol is also needed because its earlierRamseymodelcheckfailed.

Preparedm6_m025_fulld.json remainsunstarted and should follow currentM1diagnostic: completeQ6Y4-2.5 with101x40000qspec,samegain/span,totalcount~unchanged; otherfullcparametersunchanged. PriorQ6coherencechecksnowPASS; qspecR2.819/SSF.323remainreview. NeverreuseoldSSFconfirmationcarrier. Sixfreshremainingnegativegroups andfinalall6defaultzero stillrequired. SameonevalidationPPTcurrently123slides; latestM1diagnostic/newQ6fullc/captionrevisions neednextbuild. Task remainsactive;do nothandoffpartialcompletion.

### September 9, 17:36 local — M1 diagnostic passes; Q6 revised complete sequence active

M1 midpoint diagnostic rrv1_mid120/2026-09-09_17-23-44 completed17:27:45 with canonicalT2R HDF5/normalplot andboth0/off. SavedT2R36.696927198911+/−1.715973201594us,R2.973946705; independent36.696882911395+/−1.716014324815us,R2.973946705,fringe.009873599522MHz/510actualsamplesperfringe. Residualpeakfraction.051844908,RMS/fitspan.033728306; all raw/calibration/model/window/uncertaintygatesPASS. Automaticfit36.696492483243+/−1.716175009301us,R2.973946705 alsoagreesandremainsstored asFit Unseeded. Solefinding: +121.94%versusprior16.534856784us. Diagnostic-only,notcompletechain; noM1phase/frequency/pi/coherencepromoted. Sameactualcarrier/pi/sigma/RO asmidprep,phase-.005210647856533801MHz,0-120us601/1200x1/250us,cal10k250,fstart.009247585257526428/fullpopstarts(noT2guess). Normalplotvisuallyreviewed.

ACTIVE m6_m025_fulld.json guard46256/child25612 launched17:28:19; runrrv6_m025d (timestamp in guard/ROOT). CompleteQ6atY3zero/Y4-2.5,all7onepass,2700sdeadline,.5mA/sramp,noextrasettle,both0/offafter. Fourfoldqspecperpointaveraging:101pointsx40000 instead401x10000 atsameexplicit6102.73607274-6104.73607274MHz/gain.002/20us/100us andsamePRE-Rabi seed6103.945208979526/pi.651681621193/sigma.056022425. KeepRO9057.86/.8/10,allstatecal20k250,Rabi1200x2,T11200x2,R/E2400x2,unchangedfullwindows. Bothsourcesverified0/offafterM1beforelaunch. Spectrum81%at17:35:58; noerrors. Afterfullsaved/audited, onlyifsoleSSFcomparisonremains create matchingactualcarrier/pi/ROoffset SSconfirmation,bootstrapandapplypredeclaredreview; never useoldpreparedSSplan blindly. Ifqualityfails, diagnoseconfiguration; do notwaive.

PREPARED/unstarted m1_midpoint_fulla.json, labelrrv1_midfulla: fresh all7normalRRatbothzero,1800sdeadline. Qspecseed5517.701453768203/pi.6865043032411646 fromfreshpreparation, normalnewqspecthenRabi propagateactualcarrier/pi. PersistentRO/base8919.77+offset-.52=tone8919.25/.48/10,sigma.05,cal10k250; noREADOUT_FREQoverride,soresprobe8918.27-8921.27MHz151/.02MHzspacing/gain.48/10us/600x1/100us. Planproseinitiallyconfused.02spacingwithgain; correctedbeforeanylaunch and verifiedgain.48againstrawbaselineaudit. Qspec+/-.8MHz301/5000x1/.001/20us/100;Rabi0-.85/171/1200x1/250;T1.01-100/201/1200;Ramsey0-120/601/1200/phase-.0052106478565/fullfitstarts;Echo.01-100/501/1200/phase.5;all250usrelax. OnlyRamseyphase/window/starts differfrompersistentM1; noadaptiveframe orcarrierlocking implemented. RequirefreshfullchainandindependentT2repeat beforepromotion, thennewM1Y3+10fixedcontrol. OtherM1coherence/frequencyconfigsremainunpromoted.

Offline selected_profile_check_1734.json compares all14currentlyselectedacceptedcompletepoints againstlive reusable pulse/readout/averaging/timing/SSFcalibrationprofiles: ZERO mismatches. Includes all70calibrationrecords, allcoherencewindows/phase andqspec spans/counts; seed recenteringintentional. TEMP/check_selected_profiles.py canrerunwithafreshoutputafterlaterpromotions. SameonePPTrebuilt124slides/7outstandingfindings,configurationchanges168. AddedactualM1highphaseand120usmidpointnormaltracessidebyside; native124renderchecked17:35 andslide84visuallyreviewed. RevisedfivehistoricaldiagnosticcaptionspointtolateracceptedQ4/Q6results. Captionspacingimprovedinhelperafterrender; rerunhelperandnextbuildtoapply. Allraw/oldPPtsunchanged. Latestsourcearchivecheckpoint_1716hascurrentcore/guards; newM1fullplan/helper/checkselectedprofile neednextarchive.

Remaining: finishQ6-2.5fulld+SSFconfirmation/review/promotion; M1complete/repeat/control; sixunstartednegativegroupsinnegative_remaining_6.json; finalall6defaultzero; finalconfiguration/source/reference/PPTQA. Continuecompleteauthorizedtask,notpartialhandoff.

### September 9, 17:59 local — Q6 complete sequence passes quality; readout confirmation active

Q6/M6/index5 at Yoko4 -2.5 mA, Yoko3 zero completed rrv6_m025d/2026-09-09_17-28-22 in 1739.288 s. Guard verified both current-mode sources at 0 A/output off, 0.01 A range. All seven normal plots and canonical HDF5 files persisted. Corrected audit validation_M6.json has only the historical SSF comparison finding; no raw/calibration/model/coherence finding remains. Qspec center 6104.000262104902 MHz, actual pulse 6104.000263479161 MHz, FWHM 0.223003723301880 +/- 0.008982134663578 MHz, R2 0.962611168. The 101-point/40000-rep scan resolves about 11 points per FWHM; this improves point averaging without attributing a physical linewidth change to sampling. Rabi pi gain 0.6566562900567662 at unchanged sigma 0.056022425 us. Res center 9057.84 MHz; absolute readout remains 9057.86 MHz/gain 0.8/10 us, now offset +0.020000000000436557 MHz. SSF contrast 0.2795 versus historical 0.427.

T1 6.074461531776 +/- 0.118818809738 us (R2 0.979254272), T2R 1.255037936880 +/- 0.055917038091 us (R2 0.894696588), T2E 4.198881382411 +/- 0.148789932392 us (R2 0.915930600). Independent fits, residuals, full windows, calibration reconstruction and fringe sampling pass. Changes from prior coherence are +2.66%, +6.94%, +1.16%. Normal T1, Ramsey and echo plots visually reviewed. No Q6 -2.5 mA promotion yet.

ACTIVE m6_m025_d_ssrepeat.json, guard 49540, launched 17:58:53. Exact normal SS-only independent confirmation at the full-d measured carrier/pi/sigma/readout/offset above, 20000 shots per state and 250 us relaxation. Yoko3 zero/Yoko4 -2.5 mA, 0.5 mA/s ramps, no extra settling, both zero/off afterward, 240 s deadline, longest projected artifact 197 characters. Nominal attenuation arguments unchanged 10/15,5/4,17 dB; abort acquisition/readback/save errors. Previous full guard completed safely before launch. The old full-b SS plan is unused. After this confirmation: audit, exact-histogram bootstrap, and predeclared repeated-SSF review only if every unchanged gate passes. Never waive a failed quality gate. If accepted, promote this row and persist qspec 101/40000, state calibration 20000/250, Ramsey and echo 2400x2 in expt_config; Rabi/T1 remain 1200x2. Promotion helper alone does not set acquisition averaging.

Then run prepared m1_midpoint_fulla.json, matching independent Ramsey confirmation, and M1 Yoko3 +10 mA fixed-row control. Six negative groups in negative_remaining_6.json and final_all6_default.json remain unstarted. Source/plans/helpers archived under execution/checkpoint_1753. Top state paragraph cleaned to remove the obsolete active Q4 run. Same single report remains 124 slides; latest D audit and any accepted review must enter next build. Continue the complete authorized task.

### September 9, 18:06 local — Q6 SSF repeat fails consistency; M1 complete sequence active

The exact Q6 full-d SS confirmation rrv6m25_dsr/2026-09-09_17-58-56 completed safely with canonical raw/plot and both sources zero/off. It gives SSF 0.3335 versus full-d 0.2795, difference 0.054. Unchanged histogram bootstrap (300 paired I/Q resamples, seed1729) gives SE 0.004325706/0.004728536; the predeclared agreement limit remains 0.03. Both values remain below historical 0.427. This fails repeatability: do NOT apply review_repeated_ssf or promote full d. Preserve both audits and M6_m025_D_SSF_confirmation.json/png. Current configuration/averaging remains unpromoted.

Offline inspect_ssf_variation.py confirms unbinned rotated KS 0.27955/0.33430, so the difference is not a histogram-bin artifact. Twenty 1000-shot blocks show excited-state mean scatter 4.35/3.42 times the mean within-block SE, while ground-state ratios are 0.93/1.29. This is observed variation, not an assigned physical cause. Raw distributions and block plot are saved as M6_m025_D_SSF_raw_variation.json/png. Ground and excited blocks were acquired separately. Actual SS plot times are17:39:59 and17:59:18; HDF5 Dates are later batch timestamps and must not be described as SS-stage times. Future bootstrap x-axis wording now says recorded HDF5 timestamp; existing outputs preserved.

ACTIVE m1_midpoint_fulla.json, guard37696/child59760, launched18:01:18. BothYokoszero/off, M1/index0, all seven normal experiments, onepass,1800sdeadline. Its full envelope is in the prepared plan and17:36 entry. Only diagnostic Ramsey phase/window/fitstarts differ from persistent M1:0-120us601/1200x1/250us,phase-.005210647856533801MHz,fstart.009247585257526428 with free population starts; no T2guess. Fresh qspec/Rabi propagate carrier/pi. No M1 promotion before complete raw audit and independent confirmation; endpoint control remains required.

PREPARED m6_m025_ss50k_repeats.json: three predetermined matching SS-only normalRR records at full-d carrier/pi/sigma/RO,50000 shots per state (old20000),250us. The readout location is unchanged after six optimization grids; longer records test averaging of the observed shot-block variation. Every pair must agree within max(.03contrast,3combinedbootstrapSE); keep all three, not a selected maximum. Each240sdeadline,canonicalpaths<210,.5mA/sramps,noextrasettle,both0/offafter,nominalattenuationsunchanged. Do not launch before activeM1 completes safely. If all three agree, newfullQ6 sequence with50kstatecal/SSF andfulldqspec/coherence averaging stillrequired. No underlyingclasses/QICK or acceptance thresholds changed. Sixnegativegroups, M1repeat/control, finalall6zero and finalreport remain.

### September 9, 18:17 local — M1 complete pass passes quality; exact Ramsey repeat active

M1/index0 full zero-current rrv1_midfulla/2026-09-09_18-01-21 completed in802.918s, all7normalplots/canonicalHDF5 andbothsources0A/off. Audit has only the T2R historical comparison finding. Res8919.79MHz,RO8919.25/.48/10us/offset-.5400000000008731. Qspeccenter5517.695088474854MHz (actualpulse5517.695088027188),FWHM.049070099274311+/- .000802723341904MHz,R2.980551048. Rabi pi.6865043032411646/sigma.05us;SSFapproximately.665. T1 31.024767903132+/- .500645456511us,R2.994440966;T2R31.204608122321+/- .634875824376us,R2.961021457;T2E43.293840787587+/- .562129424119us,R2.983919286. All independent/raw/calibration/residual/window/fringe gates PASS; Ramsey independent31.204610183344us/f.0191053842583MHz/263.59actualpointsperfringe. OriginalautomaticRamsey31.204529041444us alsoagrees andisretained. Changesfromoldcoherences:-1.35%,+88.72%,+16.61%. NormalRamseyplotvisuallyreviewed. Acleanfulltracewasobtaineddespitefreshcarrier6.37kHzbelowtheearlierpreparation; noadaptiveframeorcarrierlockingfeaturewasadded.

ACTIVE m1_midpoint_full_repeat.json,guard47452/child55344,launched18:15:32. Exact T2R-only normalonepass fromfulla actualcarrier/pi/sigma/ROoffset,10000statecal250us,0-120us601/1200x1/250,phase-.005210647856533801MHz andidenticalfreefitstarts f.009247585257526428/phase0/amp1/offsets.5/.5. Actualaxis fromprimary.20996094-119.35058594us/.19856771step. BothYokos0/offbefore/after,1800sdeadline,longestHDF5path190characters,nominalattenuationargsunchanged;abortacquisition/readback/saveerrors. Oncecompleteandaudited, applyreview_repeated_metrics ONLYifprimary/confirmationqualityallpassandmatching-repeatgate15%meanor3combinedSE passes. OnlythenpromoteM1fixedcarrier/resbase/offset/pi plusRamseyphase/window/starts. NozeroM1promotionyet.

PREPARED m1_p100_fullb.json: after accepted M1 zero-profile promotion, rerun its fixed-row endpointcontrolatYoko3+10mA/Yoko4zero usingpersistentdefaults,all7onepass,600ssettle,.5mA/sramps,3600sdeadline,both0/offafter,canonicalpaths<210. Earlierfailedcontrolmuststayinhistory;selectnewcontrolonlyifvalidated. Q6three50kSSrepeatsremainunstarted andshouldfollowcurrentM1repeat; conditionalm6_m025_fulle.jsonpreparedwithallstatecal/SSF50k andfulldqspec/coherencecounts. SamePPT125slides/6currentfindings,native125QA18:08andnewreadoutvariationslide85visuallyreviewed. Coreclassesunchanged;focusedwhitespacepassed18:12. Sixnegativegroupsandfinalall6defaultzero remain;continuecompleteauthorizedtask.

### September 9, 18:26 local — M1 repeat remains unresolved; Q6 longer records pass diagnostic agreement

M1 exact Ramsey repeat rrv1_midfr/2026-09-09_18-15-35 completed safely with canonical T2R raw/normalplot and bothsources0/off. T2R25.808147134404+/- .577750997049us,R2.957883837; independent25.808053377426+/- .577748253511us,f.0191735709454MHz/262.66samplesperfringe. Allqualitygatespass;old16.5348568uscomparisonfails. Differencefromfulla31.2046081us is5.39646099us,largerthanpredeclared4.27595664usrepeatlimit(max15%mean,3combinedSE). Do NOT applyreview_repeated_metrics orpromoteM1phase/window/carrier/pi. The two automatic/canonical/independent fits agree within each record; fringe frequencies.0191054/.0191736MHz areclose. Fullraw exploratory two-frequency fits yield only smallR2improvements(.96505vs.96102; .95920vs.95788), andthe repeatfit'snearlydegenerate pair isnotameasuredsplit. Do notpromoteexploratorytaus orassignphysicalcause. Artifacts M1_midfulla_ramsey_diagnostic andM1_midfull_repeat_ramsey_diagnostic preservetheseanalyses.

Q6 three predetermined50kshot records completed18:23:42 with normalSSplots/HDF5andboth0/offaftereach: rrv6m25_S50/18:21:08,18:22:07,18:23:03. Contrasts.30804/.28324/.30724; exactunchangedhistogrambootstrapSE.003087869/.002894868/.002842671. Everypairpasses thePREDECLARED max(.03,3combinedSE) gate: differences.0248/.0008/.0240, eachlimit.03. AllthreekeptinM6_m025_SSF50k_repeats.json/pngandM6_m025_SSF50k_repeat_gate.json. Longerrecords satisfythe diagnostic repeatcriterion buttemporalvariationbeyondcountingnoise remains. Thisdoesnotrestorehistorical.427 orpromotea fullcalibration. Preparedm6_m025_fulle.json cannowbetested aftercurrenthardwarefinishes: allstatecal/SSF50k250us, samefulld101x40000qspecand2400x2R/E,all7normalonepass. AnewexactSSconfirmationatthe NEWfull-Ecarrier/pi/ROoffset remainsrequiredifSSFcomparisonissolegroundfinding.

ACTIVE m1_postfull_phase50.json,guard20808/child34336,launched18:24:56. M1bothYokoszero/off,normalT2R-onlyphase diagnostic at SAMEfulla/repeatcarrier5517.695088027188MHz/pi.6865043032411646/sigma.05/RO8919.25/.48/10/-.54;10kstatecal250us. Usepriorhighphase.5MHz/0-60us301/1200x1/250,automaticfreefit(no fitstartoverride). Thisisolatesthecurrentfringestructureafterthelowphaserepeatdisagreed. 1800sdeadline,canonicalpaths<210,nominalattenuationsunchanged,both0/offafter,abortacquisition/readback/saveerrors. Do notpromotethisdiagnostic'sscalar. Afterfullsave,auditandexploratoryanalysiscaninformthenextboundedconfigurationtest. NOadaptiveframe/carrierlocking/pilotprogramfeaturehasbeenimplemented;protectedclasses/coreconfigremainunchanged. M1_p100_fullb.json staysconditionalonavalidatedM1zeroprofile,notreadytolaunch. Sixremainingnegativegroupsandfinalall6zero remainunstarted. Continuecompleteauthorizedtask.

### September 9, 18:35 local — Q6 full E active; M1 phase test derived from two high-phase scans

ACTIVE m6_m025_fulle.json, guard7740/child14000, launched18:27:59. Complete normal Q6/M6/index5 atYoko4-2.5mA/Yoko3zero, all7onepass,2700sdeadline. Three predetermined50kSSrepeats passed everypair beforelaunch, and preceding M1 phase diagnostic guard verifiedboth0/off. FullEonlysciencechangefromfulld: allstatecal/SSF20000->50000shots/state, same250usrelax. Keep101x40000qspec at explicit6102.73607274-6104.73607274MHz/.002gain/20us/100us, Rabi/T11200x2, Ramsey/echo2400x2, samefullwindowsandphase.5MHz. PRE-Rabiseed6103.945208979526/pi.6516816211926998/sigma.056022425;RO9057.86/.8/10us withoffsetfromfreshres. .5mA/sramps,noextrasettle,both0/offafter,canonicalpaths196chars,nominalattenuationargsunchanged10/15,5/4,17dB;abortacquisition/readback/saveerrors. NoQ6current-specificparameters/countspromoted. AuditfullEthencreateNEWactualcarrier/pi/offsetSSconfirmationifSSFhistoricalcomparisonissolegroundfinding;neverreuseDconfirmationcarrier blindly.

M1 high-phase diagnostic rrv1_pf50/2026-09-09_18-24-59 completed withnormalraw/plotandboth0/off. Canonical13.038335916908us fallsinsideoldcoherencebandbutfailsstructured-residualgate;donotpromote. Initialexploratoryhelperstartedpaircenteronlyatthe strongestFFTbinandreturnedcollapsedpairwithamplitudeboundhit,R2.88755. Newoffline ramsey_beating_multistart.py keepssamefullraw/model/boundsbutadds five data-derived centerstarts(FFTpeak+[-1,-.5,0,.5,1]/span),60pairfits. Best8solutionsagreewithoutboundhits: center.502266404580632MHz,split.0148211705972694MHz,components.494855819282/.509676989879MHz,R2.965675193,exploratorytau29.198646us. Thisisa diagnostic design aid only. Repeatingthenewmultistartontheearliermidprephightrace reproducesitsresolvedpair(center.508854785791356,split.018495171189941,R2.973927031),showingtheanalysisfixdoesnotreplaceknowncalibration.

M1_frame_design.json derives phase_mid=phase_high-pair_center*actual_step/requested_step andframe=carrier+phase_mid/ratio. Bothhighphaserecords haveactualstep.19856770999999895us/requested.2us,ratio.9928385499999948. Earliercarrier5517.701453768203/phase_mid-.005210647685648 impliesframe5517.696205535562MHz. NewSAMEfulla/repeatcarrier5517.695088027188 requiresphase_mid+.0013305511624549515MHz andimpliesframe5517.696428175757MHz; framechangeonly.000222640196MHz. The oldconstantnegativephasewasreusedafterthecarrierchanged6.37kHz;theproposednewtestcorrectsthisrelativephase. Noadaptiveframe/carrierlocking featurehasbeenimplemented orpromoted.

Freshpair/residual simulation inM1_current_midpoint_window_prediction.json favors180us/901points over120us/601 for~.007410585299MHzmidpointfringe(~135usperiod). Withtrue simulatedtau29.198646us andmidpointerrors-.001/0/+.001MHz,180usautomaticandfreepopulation-startfits passall20trialsineachcase,median~27-29us;120usautomaticusuallyfails. Simulationisnotmeasurementorevidencefortables. PREPARED m1_frame180_a.json andm1_frame180_b.json: two predetermined identicalnormalT2R-onlyrecords afteractiveQ6finishes. Actualcarrier5517.695088027188/pi.6865043032411646/sigma.05/RO8919.25/.48/10us/offset-.54; phase+.0013305511624549515MHz,0-180us901/1200x1/250us,10kstatecal250us;existingfreefitstarts f.007410585298634704/phase0/amp1/offsets.5/.5,NO T2guess. Each1200sdeadline,bothYokos0/offbefore/after,canonicalpaths<200,nominalattenuationsunchanged. Keepboth;requireunchangedraw/modelgatesandpairagreementmax15%meanor3combinedSE. AnewcompleteM1pass andmatchingconfirmationwithanyfuturecarrier-dependentphaseconfigurationremainrequiredbeforepromotion. M1_p100_fullb staysconditionalandmustbereconciledwiththeeventualvalidatedprofile,notlaunchedas-isnow. Sixremainingnegativegroupsandfinalall6zero remainunstarted. Continuefullauthorizedtask.

### September 9, 18:49 local — user requests closeout within a few hours

Recovered the existing authorized validation task and confirmed guard7740/child14000 are the expected active Q6 full-E processes. No competing acquisition started. User requested "finish this up within the next few hours"; target approximately22:00local including report and zero/off cleanup. Prior full-six run took111.7minutes; remaining full matrix exceeds three hours at its saved averaging. Priority question (final all-six baseline versus remaining negative coverage) is pending; continue independent audit/report preparation. No acceptance gate is relaxed and no failed M1/Q6 result is promoted. Safe harness passes131Pythonfiles; git diff --check passes. Single report was already rebuilt18:40 with130slides; saved state125-slide count was stale. The live report_build.json remains source of truth until final rebuild/QA.

### September 9, 19:00 local — Q6 full E complete; exact readout confirmation

rrv6_m025e/2026-09-09_18-28-03 completed with all7canonicalHDF5 and normalplots; both sources verified CURR/0A/0.01A-range/outputoff afterward. Audit passes every raw/calibration/model/coherence gate; solefinding historicalSSF. Measured qspec6103.731047905MHz/FWHM.328195116+/- .013409379MHz/R2.962903206; actualcarrier6103.73104802091MHz; pi.651681621193/sigma.056022425us. Res9057.90MHz, readout9057.86MHz/.8/10us, offset-.04MHz. SSF.32474. T1/T2R/T2E6.143175070+/-.119192534,1.015747031+/-.070154168,4.291727696+/-.151317026us; historicalchanges+3.82%,-13.45%,+3.40%. No promotion yet.

Next m6_m025_e_ssrepeat.json: exact normalRR SS-only onepass at physicalQ6/M6/index5, Yoko4-2.5mA/Yoko3zero. Uses all measured values above,50000shots/state/250us, trigger.4us/zero phases; .5mA/sramps,noextrasettle,bothzero/offafter. Nominal attenuation args10/15,5/4,17dB unchanged,197-character maximum path,240sdeadline; abortreadback/acquisition/saveerrors. Exact bootstrap and predeclared .03-or3SE repeatgate apply. Nine legacy RR Python processes (2025--February2026) have unchanged CPU and no established TCP connections; preserved. Current task has no competing acquisition. Final-all-six priority is the stated default pending optional user reply; target22:00local remains.

### September 9, 19:02 local — detached-launch quoting corrected

Environment-layer failure: Start-Process passed doubled cmd quotes literally; initial controller41824 exited before Python or any new hardware/result file. Corrected with reviewed TEMP/launch_guard.cmd: activate the configured Conda runtime, stop on activation failure, invoke the configured interpreter with normal path quoting. Hidden cmd controller13540 now has verified guard23620/child32252; m6_m025_e_ssrepeat.result.json confirms both sources zero/off before the normal SS-only repeat. Failed controller logs retained, second launch logs use launch2. No physical setting changed in recovery. The existing nine legacy RR processes again had unchanged CPU and no established TCP connections. Harness failure-mode note added. Native report131slides and representative coverage/new diagnostic plots passed preview; both reference deck hashes unchanged.

### September 9, 19:04 local — Q6 confirmation consistent; M1 diagnostic pair begins

Exact Q6 SS confirmation rrv6m25_esr/2026-09-09_19-02-12 saved normalplot/HDF5 and bothsourcesCURR/0A/off. Contrast.30334 versus fullE.32474, difference.02140 below predeclared.03 minimum repeatlimit; bootstrap and unchanged repeated-SSF review remain necessary before promotion.

Next m1_frame180_a.json followed by predetermined b if acquisition saves safely: M1/index0, bothYokoszero/off, normalRR T2R-only onepass. Actualcarrier5517.695088027188MHz, pi.6865043032411646, sigma.05us, RO8919.25MHz/.48/10us/offset-.5400000000008731MHz. Phase+.0013305511624549515MHz,0--180us/901points,1200x1/250us;10000statecal/250us. Existing free starts f.007410585298634704MHz,phase0,amp1,offsets.5/.5; noT2guess. Bothrecords retained; fullquality and max15%mean-or3combinedSE repeatgate unchanged. Each1200sdeadline,canonicalpaths<=200characters,nominalattenuationargs10/15,5/4,17dB unchanged,bothzero/offafter; abortacquisition/readback/saveerrors. No competing task guard. Diagnostics alone cannot promote M1; finaldefaultstillrequired withincloseoutbudget.

### September 9, 19:07 local — Q6 -2.5 mA accepted with repeat-checked lower readout contrast

Unchanged exact-histogram bootstrap: fullE.32474+/- .0031393396SE and confirmation.30334+/- .0028934526SE. Difference.02140 satisfies the predeclared max(.03,3combinedSE)=.03 gate. review_repeated_ssf.py verified all7full-chain raw/calibration/model checks, matching pulse/readout/shot settings,3lengthscans+3gain/frequencygrids and4selected-point measurements. New validation_M6_ssf_reviewed.json preserves originalSSFcomparison and bothrecords, statusvalidated_change; lowercontrastrelative.427 remains, causeunassigned. Selected fullE and applied exactly its measured row to system_config.

PhysicalQ6/M6/index5/Yoko4-2.5mA changes: carrier6103.73607274->6103.73104802091MHz; resbase9057.78->9057.90MHz; ROlength12->10us; gain.7967->.8DAC; offset+.08->-.04MHz; absoluteRO9057.86MHz andsigma.056022425us unchanged; pi.651681621->.6516816211926998DAC (precision only). Stored T1/T2R/T2E5.91692/1.173581/4.150655 ->6.143175070/1.015747031/4.291727696us; FWHM.328195116+/- .013409379MHz, target.1--.2MHz not reached. Persistent acquisition profile: qspec401pointsx5000 ->101x40000; all statecal/SSF10000->50000shots/state atunchanged250us; Ramsey/echo1200->2400reps,2rounds; Rabi/T11200x2 unchanged. These are shared-profile oldvalues, distinctfromintermediate diagnostic overrides. Existingconfig-promotionsJSON records every scalarold/new/unit.

All81x6profile/source/timing checks,131-fileharness and selected_profile_check_1907.json pass:15acceptedcompletepoints,0mismatchrows. M1frame180a is active guard25596/child55560; its launch-time sourcebytes captured before Q6-only profile promotion, and its row settings are unaffected. No other hardware launched. Finaldefault/report priority and22:00target remain.


### September 9, 19:24 local: M1 pair rejected; final default pass

The two predetermined identical M1 180us/901-point records finished with both
sources verified0A/output off. T2R31.038210 +/-0.774484us and19.473528
+/-1.153028us; both raw/model quality audits pass but their difference 11.564683us
exceeds the unchanged repeat gate 4.166975us. Neither is promoted; no stable new
M1 T2 or physical explanation is asserted. Automatic and independent fits agree
within each record. Evidence: M1_frame180_repeat_review.json in validation artifacts.

Next authorized run: final_all6_closeout.json, normal round_robin_benchmark.py,
empty PUCQ4 environment, one pass on indices0..5, flags res_spec,q_spec,rabi,ss,
t1,t2r,t2e. Both Yoko sources at0A, current mode/.01A range, .5mA/s ramp policy,
zero/off cleanup. Exact row frequency/gain/pulse/readout/sweep/repetition settings
are snapshotted in final_all6_closeout_envelope.json; no M1 profile change.
Nominal attenuation arguments10/15,5/4,17dB unchanged. Output normal rr6_0mA
hierarchy; longest projected artifact193characters. Expected112min,8100s hard
deadline; abort readback/acquisition/save errors. Retain final failures alongside
earlier accepted runs. Negative grid -5,-7.5,-10mA on both flux qubits remains
unmeasured in this campaign, pending time-limited closeout near22:00local.

September 9, 19:31 local report verification: the new explicit spectroscopy-target
table exposed that older serialized audits lack audit_path (the analyzer adds it
to the returned in-memory object after writing). Classification: verification /
artifact-schema assumption. The report now records the original run and canonical
qspec HDF5 directly; it does not guess a reviewed-audit filename. The failed build
never replaced the existing PowerPoint because replacement occurs only after a
successful staged save and reopen. Native verification follows the corrected build.

September 9,19:54local continuity: final RR remains active guard36224/child58200,
currently M3 echo. A finite offline audit/report watcher is now waiting:
controller58720/Python60764, TEMP/audit_after_final.py. It waits for final cleanup
and all four copied source hashes before auditing, then records every final row
and refreshes the one deck. No hardware access or config promotion. Do not run a
second manual audit on this final run while the watcher owns the fresh audit paths.
Prepared promote_final_zero.py can preview accepted M2--M6 literal updates after
completion; M1 is excluded. The137-slide interim report passed native rendering;
explicit spectroscopy target deviations and both M1 repeats are visible. Both
reference decks still match their original hashes;131-file offline harness passes.

September 9,20:05local offline preparation: promote_final_zero.py previewed the
four accepted rows from the earlier all-six run (M2,M4,M5,M6). Proposed15 changes
were rounding precision only; all three source hashes remained unchanged and
function/class AST checks passed. M3's selected zero record comes from a different
reviewed complete run, so it was excluded from this same-pass preview rather than
misrepresented as part of the earlier all-six run. Current final M3 plots show
T1~31.46us,T2R~45.23us,T2E~54.09us; T1 may need a new exact confirmation against
historical45.79us once the canonical HDF5 audit completes. This is a provisional
plot observation, not a new validation. M4 qspec22%; final guard/watcher unchanged.


September 9 final default pass automatic audit: complete_with_findings; 3/6 complete chains pass the unchanged reproduction gates. Elapsed 113.17min. Both sources verified0A/output off. Exact audits and all findings: validation artifacts/final_default_audit_summary.json. The one validation deck has been refreshed with every final-run result. Shared config promotion, final visual review and time-bounded remaining work await review.

2026-09-09 21:24 local, PUCQ4 closeout: normal all-six zero-current pass finished with seven canonical HDF5 datasets, all normal plots, and both Yokogawas0A/off. FinalM4--M6 pass. M1 retains structured Ramsey residuals/independent-fit and coherence comparisons; M2 retains SSF historical comparison; M3 retains T1 comparison. One exact M3 T1 repeat37.915038+/-0.512876us versus31.460196+/-0.331662us fails the unchanged repeat gate (6.454842>5.203143us), despite both individual fits passing. One exact M2 SSF repeat.2853 versus.2678 agrees within.03, with bootstrapSE.006634/.005808, and straddles the historical band; no optimization waiver or configuration promotion. Failure class: measurement repeatability/comparison, not software syntax or a proven physical cause. Preserve original full-pass audits and both confirmations; do not substitute a favorable single-metric repeat for the complete pass. See final_default_confirmation_review.json in validation artifacts. Last bounded acquisition is Q4/M5/Y3-5mA; no further negative groups under22:00target.

2026-09-09 21:49 local, final Q4/-5mA review: full and exact qspec repeat centers6725.225762893/6725.099256569MHz, difference.126506324MHz >.015665516MHz preset repeat limit. Both line fits pass raw/model gates and both exceed the original reference-frequency comparison; retain measured variability without assigning a physical cause. The full run's Rabi/SSF/T1/T2R/T2E gates pass, but the current point is not promoted. All eight plots (seven full stages plus frequency pair) are included in the final report. Final zero M4--M6 promotion applied39localized literals exactly matching its reviewed preview; M1--M3 unchanged in this closeout promotion. Two offline helper errors were corrected before final checks: duplicate anaconda3 path stopped Conda activation before Python ran; config-change alias extraction initially inspected only module.body and missed assignments inside the FRIDGE branch. Use the configured SDK root exactly and ast.walk for uniquely named assignments; capture now records218task-start-to-final changes including frequency aliases. No raw data or hardware configuration was affected by these helper errors.

2026-09-09 21:52 local CLOSEOUT: campaign closed_with_gaps. Final all-six3/6pass(M4--M6); selectedzero5/6; eachflux5/8nonzero accepted. M1 coherence/model and endpoint, M2 finalSSF, M3 finalT1, Q4/-5mA frequency repeat unresolved; remainingnegative groups/endpoints and spectroscopy targets explicit. Q4/-5 exact centers6725.225762893/6725.099256569MHz differ.126506324>.015665516MHz; no promotion. Shared finalzeroM4--M6 update39literals matches preview. All15selected profilesmatch;131Pythonfiles and81x6routing pass;33protectedsources+2referencePPTs unchanged; focused diff pass. Final149slidePPT rendered freshly in PowerPoint. BothYokos current mode/.01Arange/0A/outputsoff freshly read21:50:26local. Noactive taskacquisition/guards; preserve legacyPIDs. Finalnotes docs/pucq4-round-robin-closeout.md and artifacts/closeout.md; artifacts/closeout_checks.json. Execution helpers/plans/logs/baselines are copied with byte/hash verification into artifacts/exec_0909. No more hardware work scheduled.

2026-09-09 21:54 local archive completion: copied1460task files/22283308bytes into validation artifacts/exec_0909, every source/destination SHA256 matched; archive_index.json records all relative paths and hashes. The prepared local .ps1 could not run because this machine disables script-file execution. Permitted native Copy-Item/Get-FileHash commands completed the authorized archival copy without changing execution policy or moving/deleting originals. Failure class: environment execution restriction, resolved with supported file operations. Final report hash still matches native-render proof; final documentation diff check passed. Task closed with the explicit measurement/coverage gaps in docs/pucq4-round-robin-closeout.md; no active or scheduled task acquisition.

September 9, physical Q5-Q6 avoided-crossing task resumed with explicit autonomous lab authorization. Downloads deck physical Q5 is unresolved among fixed readout rows M1-M4; physical Q6 is M6/index5/Yoko4. Do not substitute displayed Q5/M5 (physical Q4). Established shared drive DAC1, readout DAC0/ADC0, sequential tones. Both sources freshly read CURR/.01A/0A/off/30V, serial Y3 90Z114063 and Y4 90Z114062. New bounded controller pucq4_13_avoided_crossing.py uses existing qspec/res-spec programs and canonical create_data_dict/save_h5; source hashes copied, actual bias saved separately from -5mA config anchor. Independent tools/pucq4_crossing_guard.py bounds child and verifies zero/off after exit. Hardware-free complex-IQ HDF5 roundtrip/current-envelope checks PASS;132-file static harness PASS before guard/helper addition. Initial plan TEMP/pucq4_cross_20260909/locator1.json: Y4-5mA/Y3zero; M6 qspec5576.10371144-5600.10371144MHz601/3000x1/.006DAC/20us/100us, equal drive-off reference; res9056.91MHz anchor/offset-.28/12us/.8767 tracked with original resspec600x1/120/.05MHz/100us. .5mA/s ramps,noextraDCsettle,0/offafter;1800sdeadline+180sguardgrace. Nominalattenuationargs10/15,5/4,17dB unchanged; settersdisabled so no freshattenuatorreadbackclaimed. Rawandplots underfirst-light/ac_loc1/timestamp. Physicalassignment andfit results awaitdata; fullcoarse/fine/powerchecks/reportstillrequired. Minor preparation patch used nonexistent log heading and was rejected; append durable log with native Add-Content instead. No hardware or raw data affected.
September9 22:31local: locator1 complete536.06s with canonicalres/qspec-driveoff/qspec-driveonHDF5 andplots; guard+child independentlyverifyboth0A/off. M6resminimum9056.91unchanged;RO9056.63/.8767/12us. Complexjointfitcenter5587.968909939+/- .015142285MHz,FWHM.419452536+/- .071924358MHz,SNR7.9445,residual.9285noise,10.49samples/FWHM;PASSsingleline,notcrossing. Currentandreferenceaxes/rawmetadataverified. Nextcoarse1/ac_c1 testsnearestfixedcandidateM4withoutassigningphysicalQ5: Y4-4.995,-4.975,-4.955mA,Y3zero. M6commonaxis5580-5608MHz401/3000x1/.006DAC/20us/100us/equaloff. MiddlepointM4additionalcommonaxis401/5000x1/.03DAC visibilitycheck (higherpowerdiagnostic;weakpowerconfirmationmandatory). AllROsettingsinitialsharedanchor/trackedviaunchanged600represprogram. .5mA/sramps,noextraDCsettle,0/offafter,3000sdeadline+180guardgrace,aborterrors. Canonicalpathsame<200chars. Latest137fileharness,scientificsyntheticfit/driftchecks,andfocuseddiffpass. Nohardware/programclass/sharedconfigchanges.
September9 22:35local: coarse1 live guard33944/child50808, initialreferencehealthy. LocatorPowerPoint created at first-light/PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx (6slides, explicitlyinprogress), nativePowerPointrenderpassed; first/locatorplotsvisuallyreadable. DownloadsreferenceSHA256626B42201DD6C2E02417806AAE34EF94122196F2D4E88D2DAC59B9492B44EC71 unchanged. OfflineanalyzerfitchecksrecoverknownJandflagfixedfrequencydrift. Maps use actual sample/bias bins without frequencyinterpolation; fits/plotsseparateacquisitionplans toavoidcombiningvisitsoraveragingchanges. ExactearliercalibrationrawprovenancecopiedtoTEMP/calibration_provenance.json; sourcesareSeptember7ramsey_sigma_coherence_manifest q6.map.points corrected_qfreq_MHz and sigma_calibration.selected.h5. Directory/sourcekeys must be inspected rather than guessed (initialpointfieldlookupswereempty, correctedwithoutdata/hardwaremutation).
September 9, 23:00 local: ac_c1 completed all four paired qspec traces and four res-spec files; guard exit0, 1557.328s, independently verified both sources CURR/.01A/0A/off. Third Q6 point at-4.955mA is5598.045399082+/- .019326400MHz, FWHM.277508371+/- .067123608MHz, SNR5.7720. Candidate M4 at-4.975mA is5594.507302995+/- .014368989MHz, FWHM.833709725+/- .072826818MHz, SNR14.2418 at.03gain. M4 resonance8999.92MHz/readout8999.69MHz/.95/5us remained stable; all M6 tracked tones9056.63MHz. Only one line in each coarse readout, no accepted splitting or physicalQ5 assignment. Next bounded visibility1.json/ac_v1 tests masked splitting with lower power at local interpolation seed-4.9709mA (not a resonance estimate): M6 and M4 each5592.5-5596.5MHz201points, paired off/on,20us/100us/1round. M6 gains.003/16000reps and.006/4000reps; M4 gains.0075/16000reps and.03/4000reps. Same readout tracking/nominalattenuation; Y3zero/off,Y4.5mA/s,zeroextraDCsettle,4000sdeadline+180guard,zero/offcleanup,abortcurrent/acquisition/savefailure. Longest artifact194characters. Coarse controller is terminal before next launch.
September9 22:59:26local visibility1/ac_v1 launched hidden guard60216/child17112; run timestamp22-59-27. Analysis verification: added a close-pair synthetic case at0.36MHz separation/0.12MHz linewidth/0.02MHz grid, joint complex-IQ fit with a single-line null model, deltaBIC>10 and width/separation/SNR gates. Existing distant-line, known-J, zero-J and fixed-drift checks all pass. Joint center covariance is retained and used in the coupled-mode fit. This is offline analysis only; hardware programs remain unchanged.
September9 23:06local progress report refreshed to17slides from completed ac_loc1/ac_c1 only; title explicitly measurement in progress. Native PowerPoint render/reopen passed. Measured-line table and readout-tracking table visually inspected, readable and unclipped. Derived evidence is ac_review/progress_2306/analysis.json plus analysis source snapshot/hash. Static harness137files and git diff --check passed after report/analysis improvements; no shared experiment/config changes in this task.
September9 23:14local visibility1 first paired trace complete. Q6/M6 atY4-4.9709mA, gain.003/16000reps/20us, actual19.981384kHz bins: one accepted line5594.548946939+/- .010376561MHz, FWHM.291832159+/- .040215801MHz, SNR6.0090, residualRMS.95646noise,14.605samples/FWHM. Source-level readback exactly-4.9709mA. Both canonical raw references saved. This is a resolved single line, not a doublet/coupling estimate. Higher-power M6 followed by both M4 powers remains active under guard60216/child17112.
September9 23:18local visibility1 M6 higher-power paired trace complete. Gain.006/4000reps at same-4.9709mA and19.981384kHz grid: center5594.593315964+/- .011495885MHz, FWHM.411295728+/- .049908499MHz, SNR6.1005, residualRMS.86381. Single line at both powers. Lower/higher linewidth difference.119464+/- .064095MHz (~1.86SE); center difference.044369+/- .015486MHz. Sequential traces do not separate drive shifts from temporal drift; no cause or coupling inferred. M4 weak-drive trace is acquiring. Prepared but NOT launched candidates1/ac_cs1 to survey M3/M1/M2 fixed-row candidates with measured drive gains and grids resolving their prior widths; validated max path195characters. Current seeds-5.135/-5.305/-5.63mA within authorized survey envelope; inspect exact JSON and previous guard before any launch.
September9 23:22local offline preparation for candidate survey: fixed an initial-state alias in the new crossing controller so future readout tracking cannot rewrite the initial snapshot; clarified :SOURCE:LEVEL? metadata and shared drive labeling. Completed acquisitions now save before honoring a pending stop at the next acquisition boundary, preventing loss of a returned trace. Offline current-envelope/canonical-IQ/stop-during-acquisition persistence check passes. Active visibility1 child retains its launch code/source snapshot and is not restarted. No underlying qspec/res-spec program or shared calibration edits.
September9 23:26local reference audit: all seven completed paired traces (ac_loc1,ac_c1,first two ac_v1) have no accepted drive-off spectral feature. Analyzer records raw-reference metrics and prevents a coupled-mode fit where a reference feature overlaps a fitted signal. Final line windows scale with detected linewidth to avoid rejecting broad high-power lines solely because frequency sampling is finer; narrow-window preliminary snapshots are retained. Wider consistent fit windows move the coarse M4 center from5594.5073 to5594.4935MHz (within prior uncertainty) and its width from.8337 to.6969MHz; window/model systematics are separate from conditional statistical errors. New snapshot ac_review/progress_2326 records the exact analyzer source/hash.
September9 23:31local visibility1 M4 weak-drive paired trace saved, but NOT accepted: candidate5594.615149MHz, FWHM.551332MHz, vectorSNR4.555 below preset5 gate. Raw/plot/reference retained; no accepted fixed-mode frequency or width from this trace and no zero-coupling inference. Drive-off control has no accepted feature. Same-bias gain.03/4000reps comparison is acquiring; guard60216/child17112 remains authoritative. Candidate survey will prioritize identifying a resolved partner; any later M4 weak-power validation needs more averaging justified by this measured SNR.
September9 23:37local visibility1 completed exit0 in2143.188s; independent guard verified both sourcesCURR/.01A/0A/off. Final M4 .03gain trace accepted center5594.606620780+/- .008942088MHz, FWHM.732032016+/- .041970152MHz, SNR13.0603. Its same-power center differs.113161MHz from coarse-4.975mA, so candidate survey candidates1 remains PREPARED, NOT LAUNCHED. Next biascheck1/ac_b1 is an A-B-A control at-4.975,-4.9709,-4.975mA, M4only201points5592.5-5596.5MHz4000reps.03gain20us100us1round, pairedoff/on. Both sourcesparkedpreflight; sameROtracking/.5mA/s/noextraDCsettle;2000sdeadline+180guard,zero/offafter,abortreadback/acquisition/savefailures. A reproducible bias contrast relative to the bracketing A visits would justify a local fine map; a drift-like result redirects to the other fixed candidates. No physical assignment or coupling estimate is accepted.
September9 23:39:27local biascheck1/ac_b1 launched hidden guard11156/child54288, timestamp23-39-28. Before its first completed result, quantitative review defined: one accepted/reference-clean line each, A repeat difference <=max(.05MHz,3combinedSE), B contrast relative to time-interpolated A visits >max(.05MHz,3SE) to pass the local fine-scan follow-up gate. The .05MHz floor targets roughly half the prior.113MHz displacement; this is a follow-up gate, not a zero-coupling bound. Abrupt/large A drift is flagged separately. Repeated-current visits will be shown individually/control plot; no map may silently overwrite one visit with another. Future controllers now log under documentation and the guard archives stdout/stderr/final guard metadata after cleanup. Static137-file verifier and scientific/negative-control checks passed before launch.
September9 23:51local ac_b1 complete exit0 in686.079s, bothsourcesCURR/.01A/0A/off independently verified and logs archived underdocumentation. Accepted A1/B/A2 centers5594.510200336+/- .008984081,5594.513145134+/- .008942393,5594.526079236+/- .008838655MHz. B minus time-interpolated A =-.004995777+/- .010939611MHz; A repeat difference.015878900+/- .012602997MHz. Stable repeat, bias contrast below.05MHz/3SE follow-up gate: earlier.113MHz displacement was not reproduced as a bias response. This does not establish zero coupling. Proceed with prepared candidates1/ac_cs1 (no prior guard): Y4-5.135M6/M3 5542.3-5572.3MHz; -5.305M6/M1 5502.7-5532.7MHz; -5.63M6/M2 5423.7-5453.7MHz. EachM6:601/3000/.006. Fixed M3:3001/600/.002; M1:3001/600/.001; M2:1501/1500/.007 (points/reps/gain), calibrated drives and linewidth-resolving grids. All20us100us1round, pairedreferences, originalres tracking; Y3zero/off,Y4.5mA/s/noextraDCsettle,6000sdeadline+180guard,park0/off,aborterrors. Max path195characters; full reviewed envelope in TEMP/pucq4_cross_20260909/candidates1.json. Expected duration about50minutes.
September9 23:52:00local candidates1/ac_cs1 launched guard41004/child14820, run23-52-02. First M6 reference near M3 is healthy. Completed four plans now yield12paired spectra; all references have no accepted overlapping feature,11accepted single lines and1low-SNRM4 candidate, zero accepted doublets. Report updated to26slides using ac_review/progress_2353/analysis.json; nativePowerPointreopen/export passed. Bias-reversal control and branch-resolution summary were visually inspected and fit correctly. This remains an in-progress report while candidate survey and final fine-map decision remain outstanding.
September9 23:59local offline chronological-IQ audit: inspected installed QICK loop/get_raw implementation; expected shape is reps/frequency/readout/IQ, with reps outermost and time ordered. New analyze_pucq4_crossing_drift validates this shape and exactly reconstructs all saved normalized full IQ arrays from raw integrated integers; measured normalization2765forM4, no assumedclock. All20quarters across5high-powerM4traces pass line-fit gates. Coarse fixed-setting M4 center moves5594.40895 -> .47236 -> .52740 -> .57949MHz acrossquarters (errors.020-.026MHz); later ABA traces also show roughly.1MHz within-trace changes. This temporal variation limits interpreting whole-trace frequency differences as bias response; no physical mechanism, independent drift significance, or coupling is inferred. Analysis uses a shared full off reference across on blocks, so their errors are correlated. SDK initially did not index the new utility filename; filesystem existence was verified and SDK queried for its existing same-module neighbor before execution. No hardware was affected.
September10 00:01local candidates1 first paired spectrum complete: physicalQ6/M6 atY4-5.135mA has one accepted line5557.872145514+/- .020467925MHz, FWHM.457962758+/- .078884168MHz, SNR5.8281, residualRMS.90709,9.16samples/FWHM. No drive-off feature overlaps it. M3 fixed-row readout at same current is acquiring. Do not infer coupling or a bare-frequency correction from the single Q6 line; compare the direct M3 spectrum before choosing a finer current.
September10 00:09local candidates1 M3 direct-readout trace complete atY4-5.135mA. One accepted narrow line5557.323579761+/- .001316134MHz, FWHM.033840379+/- .003878603MHz, SNR11.8164,3.387samples/FWHM; no overlapping off-reference feature. Prior zero-current raw center5557.325410MHz agrees at this precision. Q6 at same bias is5557.872146MHz (gap~.54857MHz across distinct readouts/drives). One bias does not establish branch repulsion or a coupling estimate. M6 reference near M1 at-5.305mA is now acquiring; current guard41004/child14820 remains active.
September10 00:17local candidates1 Q6/M6 at-5.305mA: no accepted spectral line in5502.7-5532.7MHz, projectedSNRmax2.415 (paired IQ noise~.040/.044ADC). Raw/reference/plot saved. Readout tracking found9056.81MHz and moved tone9056.63 ->9056.53MHz at unchanged.8767gain/12us. This signal loss requires a direct M1 control and, if needed, higher-visibility/wider-frequency localization; it is not evidence of zero coupling. M1 reference is acquiring; no new plan launched.
September10 00:30local: ac_cs1 M1 pair complete atY4-5.305mA. Accepted5517.685971109+/- .001137229MHz, FWHM .055872355+/- .003461784MHz,SNR17.1227,5.592samples/FWHM, clean driveoff. M6 same bias has no accepted line at.006gain within5502.7-5532.7MHz; its res tracking remained healthy (9056.81MHz resonance, selected9056.53MHz tone). No coupling/null inference. M2-region Q6 acquisition continues. Prepared fine_m3_1.json/ac_f3a but NOT launched: first wider/higher-power M6 locator5480-5560MHz801points1500reps.03gain at-5.305mA, then seven M3-region biases-5.1454,-5.1294,-5.1414,-5.1334,-5.1384,-5.1364,-5.1374mA. Both M6/M3 use common.006drive,5554.3-5560.3MHz; M6:301points4000reps, M3:601points1000reps. Last center adds.003power M6:201/16000,M3:601/3000. This frequency-match seed uses measured Q6/M3 lines, not an accepted resonance bias. All20us100us1round/equalrefs/originalROtracking; nominal10/15,5/4,17dB unchanged, Y3zero/off,Y4.5mA/s/noextraDCsettle/zero-offcleanup,10000sdeadline+180guard,abortreadback/acquisition/savefailure. Validation-only passed, longest canonical artifact195characters, expected~85minutes. Inspect M2 results and confirm terminal parked guard before launch; no overlapping controller. Static harness138files and diffcheck pass after narrow-line detail plot addition. Source setting readback remains distinct from an independent measured current.
September10 00:35local: Q6 M6 at-5.63mA has no accepted line over5423.7-5453.7MHz at.006gain/3000reps; smoothed projectedSNRmax2.15, noise .0434/.0409ADC units, clean reference. Resonator9056.71MHz tracked, tone9056.43MHz. M2 direct scan is underway. Prepared fine_m3_1 revised before launch to add M6 visibility locator at-5.63mA5400-5480MHz801points1500reps.03gain/equalrefs, following corresponding M1-region locator. Same bounded protocol, total~90minutes and max path195chars; validation-only passes. Single-line analyzer now explicitly rejects failed optimizers, rank-deficient fits and active parameter bounds, retaining diagnostic fields; synthetic known-mode/broad-line/drift controls pass. No physical Q5 assignment, branch splitting or null coupling is accepted.
September10 00:45local: ac_cs1 completed exit0 in3064.656s; both sourcesCURR/.01A/0A/off independently verified, no previous process remains. Six paired traces persisted. Final M2 at-5.63mA also has no accepted line at.007gain/1500reps over5423.7-5453.7MHz (projectedSNRmax2.65, noise.0610/.0608), clean reference. Its resonator is healthy8951.61MHz, selected8951.21MHz/.56/5us unchanged. Simultaneous loss of the two nominal lines within the limited spectral window makes a wider M2 search a priority; this is a visibility clue, not coupling evidence. Defer fine_m3_1/ac_f3a (NOT launched). Next visibility2/ac_v2: -5.63mA M6 and M2 both5380-5500MHz2401points1000reps.03sharedgain; then-5.305mA M6 5480-5560MHz801points1500reps.03gain. Gain rises from.006(M6)/.007(M2) to.03 for localization within prior envelope, not shared calibration. All20us100us1round/equaloff/on, originalROtracking and nominal10/15,5/4,17dB;Y3zero/off,Y4.5mA/s/noextraDCsettle/zero-offcleanup,5000sdeadline+180guard;abortreadback/acquisition/path/saveerrors. Max canonical artifact194characters;expected26minutes. The exact plan is TEMP/pucq4_cross_20260909/visibility2.json and validation-only passed. Latest draft PowerPoint40slides has native-rendered readable drift/detail plots; it remains explicitly in progress.
September10 00:52local: visibility2/ac_v2 guard7192/child7408 active, run ac_v2/2026-09-10_00-45-39; no parallel controller. Offline analyzer now supports evidence from sequential M6/fixed-row readouts at a common drive gain/pulse. It requires unique visits, clean references, common frequency coverage and separation beyond linewidth/center error; coincident lines are never counted twice, conflicting doublets and additional unmatched lines are rejected. A resolved doublet in one readout is preferred over combining independent singles. Paired readout timestamps/selection/raw paths are retained. Existing seven-bias/interior-minimum/J-significance/fixed-frequency tests remain mandatory, and temporal shifts between readouts remain outside conditional fit errors. Synthetic known-J/duplicate/mixed-power/repeated-visit/conflicting-doublet checks pass. Accepted branch fits will include measured centers/error bars and fitted coupled/bare-mode curves in the report; none has resolved in measured data yet. No acquisition program or shared device calibration changed.
September 10, 01:03 local: corrected a verification-feedback failure in the offline peak detector. The ac_v2 M6 wide scan at -5.63 mA/.03 gain has a real repeatable IQ feature near sampled bin 5434.644157 MHz. Four chronological on quarters give peak I 0.8554, 0.5494, 0.6979, 0.5662 ADC units with conditional SE about 0.124-0.135; all use the full off reference. Raw normalization 6636 is verified against saved averaged IQ. A first-global-PC detector missed the feature because the sparse line did not dominate full-scan covariance. Detection now uses both smoothed/noise-scaled components and still fits signed IQ. Vector score 9.18 versus old projection 3.00; candidate center 5434.643611 +/- .006728 MHz, FWHM .138036 +/- .020473 MHz, SNR 10.214, only 2.761 frequency samples/FWHM: NOT ACCEPTED pending a finer grid. All earlier task traces were reanalyzed; no additional accepted lines, no splitting or physical assignment. Updated estimates/source hashes are in ac_review/progress_0102/analysis.json; previous snapshots remain intact. Regression and existing scientific checks pass. The detector failure and repair are recorded in FAILURE_MODES.md. This was an analysis correction; acquisition code and raw data are unchanged. M2 wide drive-on is running under guard7192/child7408. Fine M3 remains deferred; the newly localized feature requires an evidence-based finer scan.
September 10, 01:14 local: ac_v2 completed exit0 in 1647.672 s; both Yokogawas independently verified CURR/10 mA range/0 A/output off, previous processes gone. M6 and M2 show the same localized feature near 5434.65 MHz at -5.63 mA/.03 gain, but both candidates fail the frequency-sampling gate (2.76 and 2.99 samples/FWHM). The wider M1-region M6 scan at -5.305 mA/.03 gain,5480-5560 MHz with100 kHz steps, still has no resolved line (vector score3.30); its limited sampling/sensitivity does not exclude a narrow line or coupling. Next bounded map_m2_1/ac_m2a scans Y4 -5.67,-5.59,-5.63 mA, each with M6 and M2 readouts at shared gain.03 over5420-5455 MHz,1401points1000reps (25 kHz steps). The final bias adds lower gain.015 over5433.6-5435.6 MHz,201points: M6 8000reps, M2 16000reps. All20 us pulses/100 us relaxation/1round/equal off-on references, original programs and resonator tracking; nominal attenuations10/15,5/4,17 dB unchanged. Y3 stays0/off;Y4 ramps.5 mA/s with established zero extra settling and returns0/off. Deadline7000 s plus180 s guard; expected57minutes; abort readback/acquisition/path/save failure. Validation-only passed; longest canonical artifact195 characters. The plan uses observed candidates as scan seeds, not accepted calibration, splitting, or physical identity. Fine M3 remains unlaunched/deferred. Static harness138files and focused diffcheck pass; source-projection regression and scientific checks pass after the detector correction.
September 10, 01:30 local: ac_m2a first bias (-5.67 mA) is saved on both readouts. M6 candidate5430.250887MHz/FWHM.311885MHz/SNR4.547 and M2 candidate5430.128080MHz/FWHM.204398MHz/SNR3.891 both fail the SNR5 gate despite adequate grid sampling. Neither center/linewidth is accepted, and the provisional readout discrepancy is not combined into a mode pair. References are clean. Positive-bias (-5.59 mA) scans are underway. This map will be followed by its programmed lower-drive comparison. A literature check (Filipp et al., PRA83,063827; https://arxiv.org/abs/1011.3732, Secs.V-VI) confirms that shared-drive dark branches and additional two-photon transitions can complicate spectra; these remain hypotheses here, not identified PUCQ4 mechanisms. No coupling/physical assignment is accepted. At01:22 all61 HDF5 files and40 expected plots from the six completed plans existed (1.049GiB raw HDF5); all seven protected configuration/program files and the original Downloads deck hash were unchanged. The latest draft has50 native-rendered slides; a subsequent build will include the concise interpretation/reference slide. M4 reanalysis with the corrected detector retains all20 accepted time-block fits and up to.166MHz within-trace center span (ac_review/M4_time_blocks_v2).
September 10, 01:42 local: ac_m2a at -5.59 mA now has one accepted line on each readout. M6:5436.399576929 +/- .004762979 MHz, FWHM .212030102 +/- .017851823 MHz, SNR12.6627,8.486samples/FWHM. M2:5436.409615797 +/- .005204478 MHz, FWHM .156549213 +/- .018157786 MHz,SNR10.0451,6.265samples/FWHM. Both references are clean; centers agree within conditional errors, and the coincident lines are not counted as a doublet. Central-bias high/low-drive scans continue. Prepared upper1/ac_up1 (NOT LAUNCHED) to extend coverage at -5.59 mA above the present5455MHz ceiling: both M6/M2,5450-5480MHz601points4000reps.03gain,20us100us1round/equalrefs. The upper-frequency search is motivated by the observed lower-frequency feature and the fixed-row seed, not an accepted single-branch estimate of J. Fifty-kHz sampling is a locator; narrower candidates require refinement. Same res tracking, nominal attenuations, Y3zero/off,Y4.5mA/s/noextraDCsettle/zero-offcleanup;4000sdeadline+180guard,aborterrors;22minutes expected and195character maximum HDF5 path. Validation-only passed. Review the remaining map results and verify terminal parked guard before this next launch. No accepted exchange splitting or physical Q5 assignment exists.
# September 14: resume physical Q5-Q6 spectroscopy from completed artifacts

The stale September 10 live checkpoint was resolved against the actual guard:
map_m2_1 completed at 02:14 with exit 0 and both sources 0 A/off. Full saved
lower-power data resolve coincident M6/M2 lines at 5434.597693/5434.595834 MHz,
FWHM .107013/.113182 MHz, not a doublet. Complete analysis is in that run's
documentation/analysis_complete_0914. Combined progress_0914 contains 29 paired
traces; all 83 canonical HDF5 files exist (1.444 GiB), expected paired plots
exist, and the original Downloads PowerPoint SHA256 is unchanged. The updated
66-slide draft passed native PowerPoint rendering. Static harness (138 files),
scientific regression checks and git diff --check passed.

User renewed the existing autonomous lab-run authorization. No recent Python
controller existed before upper1 launch at 11:06:01 local. Fresh parked-source
preflight passed; guard61368/child47680 runs ac_up1/2026-09-14_11-06-02.
M6 5450-5480 MHz/4000 reps/.03 gain has no accepted line (vector score2.757,
clean reference); M2 still acquiring. Prepared fine_m2_0914 is not launched:
seven biases -5.67 to -5.59 mA, central bias first, M6/M2 at each, locator-based
2.4 MHz windows/121 points/4000 reps/.03 gain, same original programs and
guarded current protocol. Validation-only passed (195-character maximum path).
Review M2 and terminal cleanup before launch. The narrow fine scan follows the
observed line; it cannot exclude unseen modes outside its windows or establish
J from a single curved branch. This is a state-recovery lesson: elapsed time
and a stale ACTIVE label never establish current controller or hardware state.
# September 14: physical Q5-Q6 spectroscopy closed with unresolved coupling

Completed the fine map, wider visibility check and final localized confirmation.
Final confirmation accepts overlapping single lines on M6/M2 at5431.742654/
5431.757418MHz, widths.193551/.220711MHz, atY4-5.656667mA. Increased averaging
resolved the weak feature but no second branch. All49paired traces from11runs
yield33single-line acceptances and16unresolved traces, no doublet. J, minimum
splitting, resonance bias and physical Q5 row remain unassigned. See final
results and exact settings in docs/pucq4-avoided-crossing.md.

Final source cleanup passed on all runs; last guard58892/child54820 are gone,
both Yokogawas0A/off inCURR/10mArange. No new run is queued. Final audit hashes
143canonicalHDF5 files (2.029700GiB), verifies94acquisitionplots,11archived
guard records, seven protected sources and unchanged original Downloads deck.
Final evidence is ac_review/final_20260914; report is94slides and all94were
native-rendered. Visual review caught findings-page overflow and a table/footer
overlap; the builder now paginates findings three paragraphs/page and caps
tables at5.1inches. Corrected native rendering and visual checks pass.

Verification feedback: one offline check invocation omitted its required plan
argument and failed with IndexError before any instrument access. Reading its
CLI and rerunning with gap_m2_0914.json passed canonical lossless IQ/reference
persistence and unsafe-current rejection. Scientific regressions passed;
final static harness parsed138files and diffcheckpassed. Read the check's
required arguments before reusing it. Raw artifacts and user changes preserved;
no shared calibration promotion. The saved final outcome supersedes old ACTIVE
checkpoints and prepared plans; do not restart those plans automatically.

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

## September 15 final chronological-block follow-up

Scope: offline inspection of the saved ac_r6new M6/index5 upper trace; no hardware,
configuration, detector thresholds, raw data, or accepted full-trace fits changed.
Acceptance: inspect four equal ON quarters using the established analyzer and
confirm the lower-quarter results reproduce the earlier lower-only analysis.
Evidence: renewal_20260915/right_m6_full_blocks/block_analysis.json.
The upper raw shape is (32000, 301, 1, 2); each quarter has 8000 repetitions.
Upper quarters 1/2 have no candidates; quarters 3/4 have rejected candidates at
5455.31485412084/5455.356182534796 MHz, SNR 4.093185262/3.136992724.
Zero upper quarters pass. All four lower quarters pass; their complete block
metrics equal the earlier lower-only analysis. The shared full OFF reference
correlates errors and quarter averaging reduces sensitivity. This cannot assign
a drift mechanism or establish a new pair. The 157-slide published report already
records the full-trace rejection; this supplementary null check does not change
its validated scientific results. Hardware stop rules remain in force. Further
scientific progress needs device/wiring evidence or a justified new measurement
protocol, rather than another repetition of the exhausted visibility scans.

## September 15, 11:31 — profile sensitivity diagnosis and prospective pilot

User requested a new debugging plan and execution. New plan:
docs/q5-q6-profile-debug-plan.md. Existing detector's SNR is peak height divided
by point noise, distinct from profile-amplitude uncertainty. A separate diagnostic
uses64 chronological blocks/state and alternating discovery/validation blocks,
with independent OFF partitions. All8 predefined windows pass retrospective
controls; upper t=12.2415,10.3063,6.8923,11.7380, clean OFF controls and no flagged
block dependence. Every old metric/rejection is retained. Synthetic320trials:
0/80 white and0/80 frequency-correlated nulls qualify;65/80 planted lines qualify;
all80 drifting cases flagged and none qualify. This is method validation with
finite examples, not a physical-null tail guarantee or prospective confirmation.

Fresh fixed-profile pilot started11:31:36 under guard53552/child59464/session54867.
Run FIRST_LIGHT/ac_prof2/2026-09-15_11-31-36. Plan/spec/source snapshots live in
IDE workspace q5q6_profile_pilot_0915. PlanSHA D7D9C51522333B79576EC9D45DE0205B92BBB3254CC6500FFC099E05278BBC4C;
specSHA0F8D713B85A54581742FE4D147F648ED95CF15D1A6CC184141A9985C7A128FB6.
Two M2/index1 lower/upper visits atY4-5.6566mA: lower5430.7-5432.7MHz/101,
upper5443.2-5445.8MHz/131;16000reps/state/window,.03gain,20us,100us,one round.
PhysicalQ6=M6/index5/Yoko4/A5;M2 physical identity unassigned;sharedDAC1.
Y3zero/off;Y4ramp.5mA/s,noextra hold,zero/offcleanup. Existing guard/controller and
QICK programs unchanged;nominalattenuations10/15,5/4,17dB unchanged, settersdisabled.
3600s+180guard;197character canonical HDF5 path. Source/acquisition/save/path/
deadline or invalid-control failure aborts dependent work. All4windows must pass
frozen t>=5, adjusted signp<.01, clean references, stationarity, fit and center
repeat gates before a new map. No automatic map. Preflight expectedserials/CURR/
.01A/0A/off/30V and8 unchanged acquisition sourcehashes passed. Offline140file
harness, canonicalrawroundtrip, frozenpositive/null/phase/reference tests passed.

New context discovery while pilot runs: local C:/Users/Ma Quantum Lab/Dropbox
exists and contains a much broader design archive than M:/malab. Prior filename
searches of Downloads/Documents/Desktop/C:/_Lib/M:/malab did not cover it.
Exact PUCQ4 source Device04V2.py and output_dxf/PUCQ4.dxf are under
Dropbox/SQUILL/FUN15-12052025; sourceSHA9804f5579987eb44a55d352d27b000ba5e74c6c107454891e33af33fb5bc0468,
DXFSHA4ee0d705fb196686c64c9f20c2f187519f38d9cee39e40aa2f55a4a313fb7b08.
AST-read nominal physicalQ1..Q6 frequencies match the original deck's nominal
entries in order M3,M4,M1,M5,M2,M6. SQUIDs are indices3/5, consistent with the two
validated flux identities. Rendered4114DXFpolylines visually match deck geometry;
submission note Dec5,2025 namesPUCQ4. This establishes intended design mapping,
including Q5/M2, but the original measured-index uncertainty is retained.
Evidence: IDE q5q6_profile_pilot_0915/design_review/design_evidence.json.
No source notebook, design or original deck was edited/executed.

Notebook-tool limitation: older notebook cells without persisted IDs receive new
IDs on each read; read_notebook_cell could not address IDs just returned by
read_notebook, even after path normalization. No notebook mutation was attempted.
The actual PUCQ4 design was subsequently found as a plain Python source, so it
could be inspected safely as text/AST without executing the mask generator.

## September 15, 12:09 — prospective pilot passed; map launched

Pilot1943.218s/exit0,allfour fixed-profile windows pass. Lower centers
5431.778959825/5431.754416540MHz,t7.931/9.561; upper5444.514833207/
5444.533176441MHz,t9.438/8.237. Repeatdifferences24.543/18.343kHz pass frozen
50/116.880kHzlimits. Bothsourcesexpectedserials/CURR/.01A/0A/off/30V;
guard53552,child59464 and monitor exited. Full assessment saved in IDEpilotfolder.

Mapac_pmap/2026-09-15_12-09-11 launched after fullpilotpass and freshpreflight,
guard13836/child60040/session36347;monitor session65902. Exact16windowtable,
seeds,limits,checks and plan/specSHA are in docs/q5-q6-profile-debug-plan.md.
All8acquisition source hashes unchanged; safe142fileharness and canonicalraw
roundtrip pass. Mapalternatessides, repeatsstartingbias, stops onfailedfrozen
control. First7unique pairs only enter unchangedcoupledfitgates afterfinalstability.
No completedmap or acceptedJyet. Preserve all raw and oldanalysis labels.

Additional sourcecrosscheck: originaldeckcreatorShishirSudhaman,createdAug1,2026,
lastmodifiedOliviaSeidelSep3,2026. His08042026_WeeklyUpdate.pptx explicitlylabels
M2=Q5? and says couplingidentitywasguessedandneedsconfirmation. Laterseven-qubit
PUR1D4drive-line results cannot establish PUCQ4wiring. LocalDropbox remains the
correct expandedsourcearchive; metadata/text saved in IDEpilot/design_review.

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

## Prospective discovery map (frozen before acquisition)

The fixed-center ac_pmap failed and will never be relaunched. Its conditional
center moved0.131MHz; disjoint-half discovery gave validationt4.746<5. The new
method separates unknown-center discovery from validation and doubles total
repetitions so validation retains16000. No lowered gate or pooled failed visit.
400 synthetic cases passed: white/correlated nulls0/160, discovery-only spurs0/80,
drift0/80 accepted; shifted lines78/80. All80 drift cases were flagged.

Plan: IDE workspace q5q6_discovery_map_0915/map.json, SHA256
E34666C3664743B801E75AA92B7CA720ED53FA6EA375DE1621FB367E22C47F7D.
Exact18frequency windows and analysis hashes are in frozen_spec.json. Nine paired
visits at Y4 currents[-5.652,-5.652,-5.685,-5.634,-5.670,-5.640,-5.660,-5.646,-5.652]mA.
Second visit shifts both windows+0.25MHz; last repeats first grid. Sevenunique
biases bracket locator minimum-5.65155mA with at least two points on eachside.
Eachwindow101points,32000reps per OFF/ON state,gain0.03,pulse20us,relax100us,
one round. Lower2MHz/upper2.6MHz search windows, overall5426.360..5448.782MHz.
128 chronological blocks split alternating64discovery/64validation; t>=5,
family18 signp<0.01,OFFp>0.01,no dependence,conditional fit quality. Repeats must
agree within min(0.1,max(0.05,3combinedSE))MHz. Only seven unique pairs enter the
unchanged coupled-model gates after all18windows and both repeat controls pass.

Use tools/pucq4_crossing_guard.py on this fresh plan only. Controller remains
pucq4_13_avoided_crossing.py (safe import); existing QICK programs and readout
tracking unchanged. Q6/M6/index5/Yoko4/A5 changes current; M2/index1 readout is
intended-designQ5 with experimentalidentityuncertain. Y3/Q4/M5 remains0/off.
Ramp0.5mA/s,anchor-5mA,no extra settling;10mArange/30Vcompliance. Nominal
attenuations10/15,5/4,17dB,settersdisabled. Canonical root pucq4_first_light/ac_dmap,
longest196characters. About4hours,21600s deadline+180guard. Monitor requests
boundarySTOP on anywindow/grid failure or monitor error; controller/guard abort
on source/acquisition/save/path/deadline errors and verify both0A/off. Preserve
partial outputs. No automatic repeat or averaging escalation if this fails.
## September 15,12:39 - ACTIVE discovery/validation map

New ac_dmap/2026-09-15_12-39-29 active under guard45620/child56568,
acquisition tool session5481, monitor session8966. Do not relaunch. Exact plan,
spec, frozen analyzers, monitor and guard/runtime records: IDE workspace
q5q6_discovery_map_0915. Read docs/q5-q6-profile-debug-plan.md for envelope.
Fresh preflight passed all8acquisition hashes and both expected sources
CURR/.01A/0A/off/30V.143-file offline verifier and canonicalIQ roundtrip passed.
This supersedes the no-active-run statement below. The old fixed-center map
remains failed and preserved; no J accepted. Published report remains157slides.

### Unknown-center mapping cannot reuse a pilot's frozen detection center (September15)

Verification/specification: ac_prof2 passed fresh central profile tests, but the
first ac_pmap lower line moved0.131MHz from its locator prediction and failed
fixed-profilet2.764. Discovery on half the same data gave4.746<5. Preserve that
failed plan and do not relabel it. Model predictions may locate search windows;
unknown line detection must learn on separate discovery data. Account for the
held-out sample budget before acquisition and validate the learner under null,
correlated-noise, discovery-only-spur and drift scenarios. Early shifted-grid
and final temporal repeats test additional artifacts. A new method does not
waive independent physical identity or the unchanged coupled-model gates.

### Bitwise legacy reproduction depends on numerical-library threads

Environment/verification: forcing MKL/OMP to one thread produced tiny numeric
changes (maximum3e-7 in samples-per-FWHM) in prior fits with no acceptance change.
Repeating with the original thread environment reproduced all78prior metric
records exactly. Preserve the first output, explain the cause, and use the
original environment for claims of exact reproducibility. A tolerance-based
comparison must be labeled as such.

### Source coverage and script provenance

Context: local Dropbox contains the actual PUCQ4 Device04V2.py and DXF, absent
from the smaller M:/malab mirror. Search available source archives before
concluding that chip-specific design evidence is missing. Design frequency rank
is still not independent measured identity; the original August4 author deck
explicitly labels Q5? and calls its spatial assignment a guess.

Environment: automatic approval review rejected execution of an unread external
helper and an unsigned native-report validator. Read their full source, review
side effects, and pin hashes before resubmission. Both reviewed operations were
subsequently approved. Never treat a prior denial as permission to run an
unreviewed equivalent through another interpreter.
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

## September15,13:01 - both maps terminal; hardware parked

ac_dmap guard45620/child56568, acquisition session5481 and monitor8966 exited.
Guardfailed/exit1 after1282.156s because firstlower held-outt3.963223<5.
Both expected Y3/Y4serials verified CURR/.01A/0A/outputoff/30V. The already-started
upperOFF reference saved; no upperON followed. ac_pmap is also terminal.
All previous ACTIVE/STOP-pending headings are historical. Do not relaunch either
failed plan. No paired map, repeat-grid test, resonance minimum or J accepted.
Final assessment: IDE q5q6_discovery_map_0915/final_analysis/map_assessment.json.
Artifact audit and final PowerPoint publication are in progress; nohardwareactive.

### Report validation must compare content after slide reordering (September15)

Verification: python-pptx reassigns package slide filenames when loading a deck
whose presentation order was previously changed. Comparing identical ZIP entry
names then falsely reports changed historical slides. The corrected audit matches
slides in presentation order, compares slide and notes XML, and verifies retained
media hashes. All157historical slides and133media items are preserved in the
176-slide report. Preserve the failed filename-based audit as diagnostic evidence.
Native review also caught one footer-overlapping caption; fixed spacing/font and
re-rendered all176slides with zeroissues, followed by visualreview.

State: a hand-copied old SHA256 was truncated in a publication command. The guard
rejected it before anywrite. Bothactualfile hashes matched the preservation audit;
the corrected command reads fullhashes from that machine-readable audit andchecks
64-characterlength. Never override a hashguard based on assumed harmless changes.
## September15 - debugging execution closed out; report published, science unresolved

No acquisition, guard, monitor or report-render process remains active. Both
Yokogawas verified CURR/.01A/0A/outputoff/30V after the stopped discovery map.
Do not relaunch any historical pilot/map plan. All earlier ACTIVE headings below
are historical. User's debugging plan was executed; full Q5/Q6 scientific closure
was not achieved and must not be described as complete.

Published176-slide PowerPoint:
M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/pucq4_first_light/PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx
SHA25650065883A3A09A32DED465B46EAA4D80C2A3347ECAA0C1BFB9D773C7E57D0F36.
All176slides native-rendered,zero layoutissues;19newslides visuallyreviewed.
All157historical slide/notes contents and133mediaitems retained. Prior157-slide
report backedup in ac_review/profile_debug_20260915/prior_157_slide_report.pptx.
Evidence archiveac_review/profile_debug_20260915 contains876hash-verified files,
publication.json and archive_index.json; final documentation/source supplement
is saved separately there after publication.

Results: fresh ac_prof2 pilot passed4windows and bothcenter repeats. ac_pmap first
lower failed frozen-profilet2.764; ac_dmap firstlower failed held-outt3.963<5.
Both aborted at the next boundary, preserving upperOFF references alreadyunderway.
Neither map produced a paired bias or completed grid/temporalrepeat. No accepted
minimumsplitting,resonancebias orJ. No sharedconfig,attenuator orQICK change.

Auditpasses228previousrawhashes plus8newHDF5 and4newplots; previous218rawhashes
areidentical to priorrenewalaudit. All78legacy metricrecords reproduce exactly
under original numerical-library threadsettings. Failed controls/fitcandidates
and first failed report/audit attempts are preserved, notpromoted oroverwritten.

RecoveredactualPUCQ4 Device04V2.py/DXF establishes intendedQ5/M2 designmapping.
Originalauthor08042026_WeeklyUpdate.pptx labelsM2Q5? and calls the assignmenta
spatialguess. ExperimentalQ5identity and independentlywiredmicrowavedrive remain
unverified. Savedreadouttracking is unchanged acrosspilot/maps and supplies no
specific readout-setting explanation forweakerspectroscopy.

Next evidence required: a physically supported way to obtain repeatable paired
lines acrossbias, plus independent device/index/wiring confirmation. Do not
increaseaveraging,change thresholds,or repeatfailedplans merely toobtain a pass.
The physical cause of the weak held-out response is notestablished. Useraskedfor
autonomousprogress; routinepermissionprompts areunnecessary, but missingdevice
parameters or an undocumentedport mustnever beguessed.

### Optional source supplement was not exported

Automatic approval review rejected copying repository documentation and the
original Dropbox design/meeting files into an additional M: closeout folder.
Its stated reason was potentially sensitive export to a destination whose trust
and authorized data scope were not established. The command was rejected before
execution. No workaround or equivalent export was attempted. Keep the originals
at their existing paths and use the source references/hashes already present in
the approved evidence archive. The published176-slide report, prior-deck backup,
and876-file main archive are complete and verified. Final handoff remains local
in docs/Q5_Q6_NEXT_CHAT_HANDOFF.md and harness_engineering/PUCQ4_STATE.md.
## September 15, 15:44 - renewed Q5/Q6 recovery plan and A-B-A execution

User requested saving and executing a plan to overcome the latest failed Q5/Q6
work. Saved docs/q5-q6-recovery-plan-20260915.md before new analysis/hardware.
New offline visibility budget reconstructs all canonical means and compares
absolute line amplitudes rather than differently normalized significance scores.
M2 lower validation amplitudes: pilot .08780/.08049, fixed map .05200,
discovery map .02975 ADC units; roughly similar per-shot noise. Current and time
remain confounded; no microscopic mechanism assigned. Earlier M6 central lower
has .14419+/-.01280 ADC amplitude at16000reps, motivating complementary readout.
Raw/legacy metrics unchanged. Diagnostic output in IDE q5q6_recovery_0915.

New guarded ac_roaba/2026-09-15_15-44-33 ACTIVE, guard43468/child54148,
acquisition session87754, monitor52719. Six lower windows A-B-A at
-5.6566,-5.652,-5.6566mA, rows(6,2),(2,6),(6,2),16000reps/state,
101points,gain.03,20us,100us,one round. Exact windows/controls in saved plan/spec.
197charpath,7200s+180guard,Y3zero/off,Y4.5mA/s/zerooffcleanup.
Fresh bothserial/CURR/.01A/0A/off/30V preflight passed;8acquisition hashes unchanged;
146-fileharness,7decision checks and whitespace pass. No QICK/config changes.
First M6 positive-control failure or invalid reference/dependence stops; a B line
failure remains diagnostic and permits the predeclared A return. No automatic map.
Initial M6 tracking selected9056.43MHz versus earlier9056.38MHz; retain this
50kHzdifference as a readout covariate. Gain.8767/length12us unchanged.

Additional source review: FUN15 submission folder has only the known submission
deck among document formats searched. Device04V2 active launchers define readout
input/output and two flux ports; no independently wired Q5 drive is established.
Local Dropbox filename search found older/generic wiring documents, no new PUCQ4
packaging record. Source-only evidence cannot resolve experimental Q5 identity.
### Recovery A-B-A checkpoint: both readouts lose the lower line at B

M2 atB=-5.652mA failed held-outt2.175437; M6 atB failedt2.231347 and
conditionalfit quality. Bothreferences pass. M6 returned toA=-5.6566mA with
t13.409682,center5431.774630MHz,FWHM.147757MHz. Signal recovered, but its
67.724537kHz center difference from firstA exceeds50kHz repeatlimit. Preserve
allfailedcontrols; no map accepted. FinalM2A is still acquiring under the same
activeguard. Newly prepared short-block timing control is NOT started:
IDE q5q6_interleave_0915/interleave.json, exact same total16000reps/state/readout,
32x500blocksalternatingM6/M2order,oneheldAcurrent. Sixofflinechecks and147-file
harnesspass. NoQICK/acquisition/configchanges. Fullenvelope in recoveryplan.

Offline200synthetic coupled-fit trials show a seven-point map bracketing a narrow
visibility gap can satisfy unchangedmodelgates; no actual pairedmap exists.
A1centerbootstrapSD7.09/12.72kHz doesnoterase54.45kHzcross-readout mismatch.
Imageviewer couldnot directlyreadM:; approved copies oftwo newplotsintoIDE
workspace passedSHA checks and werevisually inspected,raworiginals unchanged.
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
### September 15 - recovery diagnostics terminal; publication blocked

Plan: docs/q5-q6-recovery-plan-20260915.md. User-requested recovery used prior
failed pilots/maps, absolute signal/noise analysis and fresh controlled runs.
A-B-A completed six windows; both B readouts failed detection while four A
windows passed. M6 return 67.725 kHz and initial cross-readout 54.450 kHz failed
50 kHz limits. Counterbalanced timing run ac_roint/2026-09-15_16-29-21 completed
all 64 blocks in 961.734 seconds: M6 t=9.684 passed, M2 t=3.945 failed. Both
within-run time-half checks passed; independent paired detection did not.
The predeclared full-map prerequisite failed, so no map/J was accepted or run.

Both guard/monitor pairs exited code 0; both expected Yokogawas verified CURR,
10 mA range, 0 A/output off, 30 V compliance. No task acquisition remains active.
All 148 new HDF5 files and 78 acquisition plots audited; eight acquisition/QICK/
shared source hashes unchanged. Raw artifacts and all previous failed decisions
preserved. No calibration or attenuation changes. No microscopic cause assigned.

Verification-feedback/state-management lesson added to FAILURE_MODES.md:
passing pilots or fitted centers do not override failed detection/repeat controls;
readout switching and shorter timing did not establish a reliable map protocol.
M6's between-run tracked tone changed 9056.43 to 9056.38 MHz; retain that covariate.
Future work requires justified intervention plus paired visibility/repeat controls,
and independent physical Q5 routing/drive evidence. Do not relaunch failed plans.

Local report IDE/q5q6_interleave_0915/recovery_final.pptx has 195 slides; SHA256
D55F2C5180EE970DE78A49D965B3C86101684377C5DAFC94BB3C0D859E27094B.
All 195 native renders passed layout checks; 19 new slides visually reviewed;
176 historical slides/notes and 141 media hashes preserved. Automatic approval
review rejected M: archival/report replacement for lack of explicit publication
authorization. The script did not execute and the existing 176-slide publication
is unchanged. User approval question pending; no workaround attempted.

Final safe verifier: 147 Python files parsed, structure/manifest checks passed;
no imports/hardware access. git diff --check passed (existing CRLF warnings only).
Earlier targeted checks: signal/baseline reconstruction, seven A-B-A decision
cases and six interleaved aggregation/control cases passed. The numerical
200/200 synthetic gap-map fit exercise is not device validation.

## September 16 - publication completed with explicit user authorization

The user explicitly requested saving the PowerPoint with the previous reports.
Publication succeeded and was hash-verified: 195-slide report at
`M:\_Data\20250822 - Olivia\pucq4_run_started_Oct_7\PUCQ4\pucq4_first_light\PUCQ4_Q5_Q6_avoided_crossing_20260909.pptx`.
SHA256: `D55F2C5180EE970DE78A49D965B3C86101684377C5DAFC94BB3C0D859E27094B`.
The previous 176-slide report is preserved at
`FIRST_LIGHT/ac_review/recovery_20260915/prior_176_slide_report.pptx` with its
original hash. The same archive contains 265 verified evidence files and
`publication.json`. All earlier publication-pending statements are historical.
Scientific status remains unresolved: diagnostic progress, no accepted full
crossing map or J. This publication task contacted no hardware.

## October 8 - honor resonator spectroscopy settings in round robin

Goal/acceptance: edits to `expt_cfg["res_spec"]` must survive automatic PUCQ4
profile application and reach the frequency loop and acquisition averaging.
Scope is offline configuration precedence; no device calibration, run flags,
data paths, other experiment profiles, or hardware operations changed.

Root cause (context provision/state management):
`apply_pucq4_round_robin_profile` ran before each row and silently replaced the
main resonator settings with a second hard-coded dictionary. Removed only that
profile entry; explicit `PUCQ4_RES_*` environment overrides remain supported.
For all six resonator rows (Python indices 0..5), current effective defaults now
follow the existing main dictionary: start -1.5 -> -3 MHz, step 0.02 -> 0.05 MHz,
points 151 -> 120, reps 600 -> 200, rounds 1 -> 1, relax delay 100 -> 1 us.
These are the user's existing settings, not newly selected calibration values.
The last frequency offset is start + step_size * (steps - 1).

Added `tools/check_res_spec_config.py`: AST-selected definitions and stubbed
acquisition verify two setting sets, six rows, two rounds, and six current-profile
branches without importing hardware modules or the orchestrator. It passes.
An initial verification-fixture error used an extra dimension in fake IQ data;
corrected the stub to match the wrapper's IQ indexing and reran successfully
(verification-feedback layer; no production acquisition change needed).
Updated REPOSITORY_MAP.md with configuration ownership and override precedence.
The safe repository verifier passes. Hardware validation was not run and requires
a separately authorized lab session.

## October 8 - restore prior resonator acquisition configuration

User requested restoring the previously successful resonator setup for their
next round-robin run. Compared September 29 commit `3bc429e` with its parent,
the prior automatic RR profile, and the persistent zero-current readout table.
Restored the accepted profile's settings directly in the main `res_spec`
dictionary, preserving the preceding fix that prevents hidden profile resets.
For M1..M6 / Python indices 0..5: start -3 -> -1.5 MHz; step 0.05 -> 0.02 MHz;
points 120 -> 151; reps 200 -> 600; rounds remain 1; delay 1 -> 100 us.
WIDE_RES_SCAN remains False. Its optional point count is restored from 60 to
the pre-September-29 value 300 (offsets -15 to +14.9 MHz at 0.1 MHz spacing).

Restored VNA_res in MHz, in M1..M6 order:
[8920, 8351.61, 8574.80, 8299.92, 9214.66, 9658.88] ->
[8919.77, 8951.61, 8974.80, 8999.92, 9014.66, 9058.88].
Round robin actually seeds its scans from system_config's resolved readout
table, which already contains those zero-current values. The VNA_res repair
also restores consistency for other consumers. Qubit spectroscopy edits are
outside this resonator-only request and were preserved. Gains, lengths,
attenuations, run flags, Yoko selectors, and output hierarchy are unchanged.

Acceptance: static evaluation confirms the restored tracking and wide sweep
ranges/averaging, and the hardware-safe syntax verifier passes. No experiment
or instrument connection was run; present-day resonance visibility remains
to be confirmed by the user's lab run.

## October 8 - user-requested run-folder rename

Replaced the previous August run-folder component with
`pucq4_run_started_Oct_7` in 46 references across 25 repository source and
documentation files, including historical path references as explicitly requested.
The new run-folder name is user-authorized. Existing measurement directories
were not moved or renamed; historical documentation paths now use the requested
name and may require corresponding external data relocation before use.
Acceptance: no old-name matches in repository text (including ignored files,
excluding Git internals and bytecode caches), clean whitespace diff, and safe
offline verifier. No hardware run or M: drive write.

## 2026-10-20 - fresh cooldown benchmark: resonator preflight

User authorized fresh hardware measurements after warm-up/cooldown, destination
M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7. Acceptance: fresh res/qspec,
Rabi pi gain/sigma, SSF, T1/T2R/T2E, canonical raw/plots and a new PowerPoint;
old metrics are references only. Preserve user edits and historical raw paths.

Read-only Yoko preflight passed both expected serials, CURR, 0 A, output off,
0.01 A range, 30 V. Offline harness passed 148 Python files. Existing Python
experiment processes predate this campaign; no Python connection to the QICK
host was present in the TCP snapshot (Chrome had port-9090 connections).

Ran round_robin_benchmark.py with res_spec only, indices 0-5, n=save_r=1,
500 reps x 1, +/-1.5 MHz / 151 points / 0.02 MHz, relax 100 us. Existing
readout gains .48/.56/.95/.95/.65/.95 and lengths 10/5/9/5/6/10 us retained.
Nominal attenuation arguments readout 10/15, qubit 5/4, ADC17 dB retained;
these are not new attenuation readbacks. 900-second process timeout; abort on
acquisition/save error. Checked representative longest normal path 199 chars.

Run: PUCQ4/find_resonators_sandbox/rr6_0mA/2026-10-20_16-23-15 beneath the
selected root. Completed 69.2587 s, exit 0; both sources verified zero/off.
res_ge/2026-10-20_16-24-24_res_ge_results_batch_1_Num_per_batch1.h5 contains
all six finite 151-point traces and required metadata; six PNG/PDF pairs exist.
M1-M6 minima MHz: 8919.07, 8950.91, 8974.12, 8999.14, 9013.72, 9057.70.
All minima interior. Visually reviewed all six plots; M6 is asymmetric.
No shared calibration or run flag edits made, and no qubit drive acquired.

New interim report: PUCQ4_benchmark_20261020.pptx in selected run root, 9 slides,
reopened successfully; SHA256
71cd830672f9485e499dd292d550a40e6f610a21923f4d3e355dc885918bb2ff.
Remaining metrics are explicitly pending. Await user clarification on whether
expt_config's 100-300 MHz carrier changes are measured or search guesses.
RR overrides these with system_config carriers; it also preserves old absolute
readout tones after remeasuring resonators. Both need reconciliation before
fresh population calibration. No hardware acquisition remains active.

State-management discovery: September artifacts are under pucq4_run_started_Aug_3,
not the October path copied into historical docs. Do not rename historical raw
provenance as part of setting a new acquisition destination.
## 2026-10-27 - fresh cooldown search resumed

User confirmed expt_config.VNA_qubit are guesses and explicitly requested
continuing qubit search, Rabi, SSF and coherence. Destination and hardware
permission remain the October run. No nonzero flux sweep is requested.
Both Yokogawas freshly verified expected serials/CURR/0 A/off/.01 A/30 V.
No Python TCP connection to QICK was observed before launch (Chrome port9090).

First diagnostic: normal round_robin_benchmark.py, M1/index0, res_spec+q_spec,
one round; res151 points +/-1.5MHz/500reps/100us, then qspec5017.691628-
6017.691628MHz/4001points/100reps/20us/.03gain/100usrelax. Raw IQ scaling
false because no new pi pulse exists. Keep-seed-on-fit-failure is diagnostic
persistence only; no Rabi/coherence enabled. Existing readout .48gain/10us,
explicit -.52MHz offset from newly measured dip. Nominal attenuation args
10/15,5/4,17dB unchanged. Zero/off final;900s timeout; abort on hardware/save
error. Canonical projected qspec filename204chars. Label hunt_m1_1027 under
existing find_resonators_sandbox study; timestamps prevent overwrite.
Controller log/plan: IDE/benchmark_1027/hunt_m1.log and hunt_m1_plan.json.
All future stages depend on review/repeat of actual acquired features.

October27 discovery progress: hunt_m1_1027/2026-10-27_16-26-32 completed,
raw I candidate5515.42947MHz; reject global fitter5655.11696MHz/Q noise.
fine_m1_1027/2026-10-27_16-28-44 repeated5515.334706MHz/FWHM.0834629MHz,
raw joint-IQ SNR19.7347/residual1.1282, .003gain/20us/401pts/1000reps,
5513.43-5517.43MHz, readout8918.55MHz/.48/10us. Candidate resolved,
not yet population-calibrated. M2 broad scan noise only; M3 weak5551.3MHz;
M4 broad inconclusive; M5 broad noise only. No broad fitter promoted.
M2-M6 queued sequential independently, res+qspec each, priorcenter+/-500MHz,
4001/100/.03gain/20us/100us; old offset applied to newly measured res.
All completed raw/plots saved and sources0/off. Exact plans/receipts in
IDE/benchmark_1027. Next: inspect M6; M1Rabi and finer local searches.

October27 16:43 progress: M1 Rabi/SSF completed rabi_m1_1027/2026-10-27_16-36-06,
qfreq5515.334706157904MHz/sigma.05us/pi.6865043032411646, RO8918.55/.48/10us.
Rabi R2.99760711, SSF contrast.6445 with10000shots/state; raw and population
reconstruction audit passed. Normal calibrated Rabi plot visually reviewed:
just over half oscillation, pi in target. No shared config promotion yet.
M2 local_m2_1027/2026-10-27_16-38-10 resolved5438.51616267MHz/.33006244MHz
at.03gain,1201points/1000reps; lower-power repeat prepared, not acquired.
M3 fine_m3_1027/2026-10-27_16-40-46 resolved5551.35452734MHz/.03969838MHz,
SNR52.4184/residual1.07427 at.003gain/20us/401points/2000reps. Rabi prepared.
Local4 thenlocal5 thenlocal6 controller remains active; each2000reps/1201pts,
.03/20us;M4/M6 priorcenter+/-15MHz,M5+/-30MHz,100usrelax,rawIQdiagnostic.
All completed sourceszero/off, raw persisted. PowerPoint24slides rebuilt from
initial9+freshnormalplots; sourceplans/logs/receipts in root/benchmark_1027.
No fresh coherence acquired yet. Do not relaunch completed labels.

October27 presentation QA note (environment layer): native PowerPoint automation
hung opening the interim M-drive PPTX before rendering any slide. The dedicated
/AUTOMATION -Embedding process created by this task (PID68572,16:43:42) was
terminated after waiting; read-only open, no report write or user presentation
changed. COM returned0x800706BE after termination. PPTX package reopens with
python-pptx; source scientific plots are visually reviewed. Native rendered-slide
QA remains pending; do not claim it passed. Hardware acquisition continued
independently and was not interrupted.

October27 16:54 progress: local_m4_1027/2026-10-27_16-42-35 completed,
independent center5594.64537865MHz/FWHM.38584387/SNR20.70184/residual1.00327.
local_m5_1027/2026-10-27_16-47-44 completed, candidate7064.93453384MHz,
FWHM.69311654/SNR9.74985/residual.90827 at.03gain/20us. Both normal raw/plots
persisted; lower-power repeats prepared (fine_m4:.01gain,+/-1.2/241/5000;
fine_m5:.003gain,+/-1/201/5000), not yet launched. M6 local acquisition ACTIVE
under session63757 queue (last stage). All completed source readbacks0/off.
Progress report28slides/12runs, broad automatic fits visibly marked unaccepted.
Source snapshot with hashes archived at runroot/benchmark_1027/sources_initial.
Prepared unstarted: fine_m2, rabi_m3, fine_m4, fine_m5, m1_len4/6/8/10/12.
Continue actual calibration, do not present current partial metrics as final.

October27 17:09: fine_m4 lower-power repeat5594.640661844MHz/FWHM.173653049,
SNR37.1868,residual1.24377; fine_m5 lower-power repeat7064.931216765MHz/
.11414612,SNR12.3632,residual1.02328. Both saved/zero-off. Current controller
session97923: m1_len16/20/12r, rabi_m2/m4/m5, then hunt_m6ro. Plans record
exactmeasuredcarriers/sigmas/readout; defaultRabi0-.85/171, M2 1200x1,
M4 5000x1, M5 1200x2. M6 retry uses independently measured positive RO slope
9058.74MHz/.30/10us (slope maxima9058.72/.78/.74 for11/21/31-point smoothers),
qspec6022.489876-7022.489876MHz/4001/500/.10/20us/100usrelax. Diagnostic
power broadening and relocated readout, not calibration.900s timeout each.
M6 old operating point yielded null broad/local scans; no coherence launched
on an unverified carrier. Lower-power res dip9058.34 vs.95gain dip9057.72.

### 2026-10-27 17:16 - fresh Rabi and readout tuning continue
M2/M4/M5 Rabi+SSF receipts and canonical HDF5 audit passed; raw finite and
population reconstruction verified. M2 pi .666605628 at sigma .101us (R2 .98551,
SSF .2635); M4 pi .572086919 at sigma .079003864231us (R2 .99795, SSF .43026);
M5 physical Q4 pi .631782946 at sigma .056940825us (R2 .98735, SSF .1801).
M2/M3/M4 calibrated Rabi plots visually checked. M4 sigma retune .0695us planned.
M1 length16/20/12repeat SSF .6739/.6622/.6648;12us selected shortest plateau.
Controller97923 executing M6 broad readout-corrected diagnostic; controller94636
waits for successful final receipt before M4 Rabi repeat, M1 grid (.40/.48/.56
x8918.35/8918.55/8918.75MHz,length12us), M2 lengths3/5/7/9/11us,
M3 lengths5/7/9/11/13us and M5 lengths2/4/6/8/10us. All at measured carriers/pi,
zero Yoko current, normal RR attenuation10/15dB readout,5/4dB qubit,ADC17dB.
Stage plans explicitly record values; 900s deadline per stage;abort on acquisition,
source readback, save or audit failure. Max raw paths remain below260characters.
No shared calibration promoted; working calibration.json contains measured seeds
plus explicitly provisional qspec gain changes for M1/M3. Coherence still pending.

### 2026-10-27 17:21 - M4 target pulse and M6 null
M4 sigma .0695us repeat gives pi .646706952 and SSF .43186, canonical raw/plots
and audit saved under rabi_m4b_1027. M6 low-power positive-slope broad scan is
noise-only (largest pointwise score4.09 over4001points); no frequency accepted.
Next bounded M6 alternative is dip9058.34MHz,gain.30,length10us, drive.03,
6022.49-7022.49MHz4001points1000reps. Prepared, not yet queued.
Offline helper attempted to read an HDF5 during save and got bad-object-header;
no writer failed and completed file is readable. Verification-layer fix:
summarize_ss now requires the successful stage receipt before opening raw files.

### 2026-10-27 17:30 - iterative readout optimization
All length scans and M1 two gain/frequency grids completed, audit+raw/plots saved.
M1 best grid point .64gain/8918.35MHz/12us gives SS contrast .7818; neighboring
.72gain gives .7732. Next length repeat6/8/10/12/16us at that new point.
M2 provisional length9us (.3172;11us .3086), M3 provisional13us (.64655;
11us .63515), M4 provisional7us (.44066;5us .40476), M5 provisional2us
(.2583 versus10us .1646). Need repeat length and2D until plateau; these are
working selections, not final shared calibration.
Queue chain:94636 complete;63265 complete;36204 complete;6146 active M2 grid;
79927 waiting then M3/M5 grids;99078 waiting then M4 grid. Each waits for its
predecessor's successful final receipt; no concurrent acquisition. M6 next
alternative remains prepared only. Shared configs unchanged. Interim report
refreshed to83slides; final grouped benchmark report still pending coherence.

### 2026-10-27 17:45 - plateau repeats and wider M6 diagnostic
M1 at.64gain/8918.35MHz gives SSF.7127/.7420/.7767/.7787/.7751 for
6/8/10/12/16us; choose10us shortest plateau. M2 at.56/8950.13 gives
.1499/.26115/.27035/.32911/.3014/.28845 for1/3/5/7/9/13us (20kshots);
choose7us. M3 at.95/8973.43 gives.6791/.71285/.7113/.72505 at9/13/17/21us;
13us provisional, need final longer-length repeat. M5 at.85/9013.56 gives
.3681/.5167/.5738/.6097/.631 at1/2/3/4/6us;6us provisional, extend later.
M4 at.95/8999.25 gives.43352/.4496/.43314/.42302/.41974 for5/7/9/11/15us;
choose7us. Fine2D plans queued (M1/M2 9points each,M3/M5/M4 6points each).
Controller77845 has completed M4 length sweep and now executes hunt_m6wide:
4300-7300MHz6001points600reps,gain.03,length20us,RO9058.34/.30/10us,
1200sdeadline; derived bounds cover historical Q6 tuning4375-7211MHz plus margins,
without changing current. NormalRR attenuation,zero sources,abort on error/save failure.
Initial offline PowerShell plan assignment lacked newproperty keys; caught before
execution and corrected with Add-Member; reviewed complete JSON has explicit bounds.
Queue after M6wide:58984 M1fine ->26698 M2fine ->17202 M3fine ->45806 M5fine;
next M4fine controller ID recorded in active tool history. No concurrent hardware.
Final-full plan generator preserves delay windows/averages but reduces redundant
points: M1/M2/M3/M4/M5 Ramsey201/135/268/121/101 andEcho251/151/376/101/151;
nominal >=5samples/fringe; actual fitted fringe sampling must be checked.
Full benchmarks/coherence and shared config promotion remain pending.

### 2026-10-27 17:55 - M1 readout convergence; first full pass queued
M1 fine grid at10us gives .7766 at.64/8918.35MHz, reproducing .7767;
maximum .7793 at.68/8918.45 is not a significant improvement. Retain lower
readout gain .64,8918.35MHz,10us. Length plateau and two2D grids reviewed.
M6 wide4300-7300MHz scan completed raw/plots,zero/off sources, no resolved line;
ranked fluctuations5257.9766(score4.55) and5411.4729(score4.38) will receive
+/-3MHz241point5000rep local repeats after finegrid queue (controller98132).
Do not promote either rank; broad visual IQ is noise-like.
Queued full_m1a normal round_robin_benchmark.py after check_m6c2:
M1/Python0,all7flags=res_spec,q_spec,rabi,ss,t1,t2r,t2e,n1;
res8919.07+/-1.5MHz151points500reps; explicitRO8918.35/.64/10us;
qspec5515.334706+/-0.6MHz241points5000reps,gain.004,length20us;
Rabi sigma.05us,0-.85gain171points2000x1; qspec thenRabi update fittedcarrier/pi.
T1 .01-100us201points,Ramsey0-60us201points/.5MHz,Echo.01-100us251points/.5MHz,
1200x1 averages,250usrelax; SS10kshots/state. Standard attenuation10/15dB readout,
5/4dB qubit,ADC17dB. Normal canonical October folder full_m1a_1027;
raw path205characters,plots also below260;1800sdeadline;abort on hardware,
source readback,save error. Final safe source state remains0A/outputoff.
This is a measurement pass, not yet accepted calibration; quality audit follows.

### 2026-10-27 17:58 - M3 readout refinement stable
M3 fine grid max.7167 repeats prior.71285 at.95/8973.43MHz/13us;
neighbors .6596-.6773 give no further improvement. Retain13us near length
plateau (17us.7113,21us.72505); longer integration's small gain is not robust
relative to repeats. Full_m3a queued after M2 next length check:
normal7flagRR,M3/index2,qfreq5551.354527+/-0.6MHz241points5000reps,
qgain.008/20us to widen previous.0397MHzline toward.1-.2MHz;
sigma.124us,RO8973.43/.95/13us,res8974.04+/-1.5MHz151x500;
Rabi0-.85/171/2000x1;T1.01-150us201points,Ramsey0-80us268points/.5MHz,
Echo.01-150us376points/.5MHz,1200x1,250usrelax,20kSSshots/state.
Same recorded attenuation/zero sources/save envelope;1800sdeadline.
Full passes remain unaccepted until qspec population/linewidth and coherence audited.
M2 finegrid improved to.3775 at.64/8949.93MHz/7us; next3/5/7/9us check
queued after full_m1a, then choose nextgrid. M1full controller10843,
M2length controller83665; M3full controller in current tool history.

### 2026-10-27 18:01 - M4 readout converged and full pass queued
M4 finegrid gives.45374/.45314/.44454 at.95gain and8999.05/.25/.45MHz;
no significant gain over.4496 repeat at8999.25/7us. Retain .95/8999.25/7us.
Full_m4a normal7flagRR queued after M5 next length scan (controller49323).
M4/index3 physicalunknown, res8999.08+/-1.5MHz151points500reps;
qspec5594.640662+/-0.6MHz241points5000reps,.01gain20us;
sigma.0695us,Rabi0-.85/171/2000x1;SS50kshots/state;
T1.01-5us101points6000x1,Ramsey0-3us121points8000x1/2MHz,
Echo.01-5us101points8000x1/2MHz,100usrelax;normal recordedattenuation,
0A/off sources before/after,canonicalOctober output,1800sdeadline,
abort on hardware/readback/save failure. All numbers remain pending full audit.
M5 finegrid best.6671 at.95/9013.56/6us; next lengths4/6/8/10/12/16us queued.

### 2026-10-28 16:33 - resume fresh benchmark
User explicitly requested continuation. All queued Oct27 stages completed successfully;
no active recent acquisition controller remains (listed RR processes date2025/Feb2026).
M1 full fresh RR: qfreq5515.336341MHz,FWHM.168887MHz,pop.32099,
pi.6865043,SSF.77324,T1~33.60us,T2R~41.44us,T2E~45.09us;
M3 full: qfreq5551.357873MHz,FWHM.154803MHz,pop.33402,pi.6566563,
SSF.69255,T1~57.37us,T2R~50.09us,T2E~58.55us. Coherence independent fits agree;
qspec R2.83/.86 requires visual review rather than automatic rejection.
M4 full has decays beyond old windows (fit~7/12/15us vs5/3/5us acquired), so no
coherence values accepted yet. Rabi/SSF/qspec normal and stable.
M2 third length scan3/5/7/9us gives.2922/.3389/.3883/.3316; retain7us.
M5 third scan4/6/8/10/12/16us gives.6405/.6665/.6753/.6682/.6561/.6351;
choose6us shortest plateau; 2D repeat at6us .6671 reproduced.6665.
Next sequential normal RR stages: M2 sixpointgrid(.64/.72 x8949.73/.93/8950.13MHz,
7us,10kSSF); fullM5 all7 flags atRO9013.56/.95/6us,carrier7064.931217MHz,
sigma.056940825/pi.631783,qgain.003/20us,+/-.6MHz241points5000reps;
T1.01-100us201points,Ramsey0-60us201points/.5MHz,Echo.01-100us251points/.5MHz,
1200x2 coherence averages,250usrelax; then fullM4 repeat all7flags at
RO8999.25/.95/7us,carrier5594.640662MHz,sigma.0695/pi.646707,qgain.01,
T1.01-30us121points,Ramsey0-40us201points/.5MHz,Echo.01-60us301points/.5MHz,
1200x1 averages,250usrelax. Rabi0-.85/171/2000x1;res+/-1.5/151x500.
Normal10/15dB readout,5/4dB qubit,ADC17dB attenuation. Canonical Oct root;
new stage labels1028, no existing artifacts overwritten, paths<260, full1800sdeadline.
Abort on readback/acquisition/save error; both Yokos must remainCURR/0A/outputoff.
Shared configs not yet promoted; report still interim. New day measurements remain
fresh post-cooldown evidence; Oct27 full runs preserved.

### 2026-10-28 16:35 - model review and diagnostic continuation
M1/M3 qspec20us traces have coherent fringes; current Lorentzian width is an
envelope estimate. Good coherence preserved; clean-spectrum acceptance pending.
Queued longq_m1/longq_m3 after fullM4b: q_spec,rabi,ss,100us atsame .004/.008 gains,
151points+/- .45MHz2000reps; then Rabi andSSF. No underlying program changes.
M6 candidates failed reproduction (one-sample noise fits), no frequency accepted.
Queue res_m6lin afterlongq_m3: gain.05 instead of.30,measured seed9058.34MHz,
+/-3MHz301points2000reps10us readout, to test lower-power linear response.
Then M2 length5/7/9us at.72gain/8949.93MHz,20kshots,state source guards unchanged.
M2 newday grid best.3582 atthispoint; retain only after length/repeat confirmation.
A missing PowerShell object property was caught by ErrorActionStop before writing
res_m6lin plan; corrected with Add-Member and complete JSON reviewed before run.

### 2026-10-28 16:38 - M5 spectrum shifted
Completed M5 qspec plot reports7065.204MHz versus7064.931MHz yesterday, FWHM.189MHz,
peakpopulation~.10. Full RR already updates carrier and remeasuresRabi/SSF/coherence;
normal plot reviewed while immutable completed phase; do not read live HDF5.
Long-drive M5 test added afterM2 lengthchecks: q_spec,rabi,ss,seed7065.204MHz,
+/- .6MHz151points2000reps100us,samegain.003; thennormalRabi/SSF. Purpose is
empirical population/linewidth tuning, not assumption that longdrive guarantees success.
No acceptance or shared promotion until full saved-data audit.

### 2026-10-28 16:40 - M2 full pass queued with planned review
Normal all7flag full_m2a queued afterlongq_m5; readoutcurrently.72/8949.93/7us,
with prior multiple lengthscansfavoring7us. New5/7/9us check occurs beforelongq_m5,
so its result can update fullM2 plan before execution if needed. Carrier5438.537411MHz,
qspec+/- .6MHz241points5000reps,.007gain20us,sigma.101/pi.666606;
Rabi0-.85/171/2000x1;T1.01-60us121points,Ramsey0-40us135points/.5MHz,
Echo.01-60us151points/.5MHz,1200x1/250usrelax. Allfourphase SSFcentroid
calibrations andstandaloneSSF explicitly20kshots/state. Normal recordedattenuation,
zero sources,canonicalOct folder,new1028label,1800sdeadline;hardware/save abort.

### 2026-10-28 16:47 - M5 audit and adaptive short-coherence plan
FullM5a completed withallcanonicalHDF5/plots andzero/off sourceguards. Fresh carrier
7065.203976783MHz,pi.646706952,sigma.056940825,SSF.5289,res9013.4MHz.
T1=4.20076+/-.08269us,R2.98193,independent fit agrees. T2R candidate1.59274us
R2.75994 andT2E6.86431us R2.79170 are NOT accepted; windows60/100us mostly
baseline. Next fullM5 must use T1.01-25us101points,T2R0-10us201points/2MHz,
Echo.01-30us301points/1MHz,2400x2 coherence averages,250usrelax.
Before it, fresh readout length2/4/6/8us afterfullM2 then2Dsweep ifneeded.
LongqM5 test updated before execution:gain.003->.006,length100us,piupdated.646707,
newseed7065.204; observedpeak .10 with natural-width-scale .189MHz motivates
power test. Targetpopulation/linewidth feasibility must be judged from data,
not assumed from simply increasing drive length. Original fullM5a unchanged.

### 2026-10-28 16:56 - M4 extended fits and physical consistency control
FullM4b raw/plots/sourceguards passed. qfreq5594.645618245MHz,width.178571MHz,
peak.47017,RabiR2.99682,pi.646706952,SSF.43138. T1=7.16306+/-.17887us,
T2R=11.05849+/-.38475us,T2E=16.25898+/-.42897us; independent fits agree,
R2.989/.946/.953,>=10samples/fringe. Echo exceeds2T1 by1.93us (~3.5 combined
conditional-fit errors), so stationary physical consistency remains unresolved.
Do not silently markcomplete fromfitR2. Queue T1pre/Echo/T1post atfixed measured
carrier5594.645618/pi.646707/sigma.0695/RO8999.25/.95/7us afterM5lengthscan:
T1.01-30us121points2400x1,Echo.01-45us226points2400x1/.5MHz,100usrelax,
50kstatecal shots (100usrelax). Samezero/attenuation/saveguards,1800sdeadlines.
100usrelax is >13x currentmeasuredT1; bracket sequence assesses temporal variation
or fitmodel; it is not a replacement for a final accepted full run if needed.

### 2026-10-28 16:58 - long-drive M1 test successful
longqM1 saved raw/plots/sourceguards;100us driveat.004 givesqfreq5515.333004MHz,
FWHM.093119+/-.001510MHz,pop.49435,R2.99062,rawIQ SNR46.25,
RabiR2.99869,SSF.7335. Fringes removed. QueuefullM1b normalall7 afterM4pair,
using100us/gain.0045 (smallincrease to target.1MHz),151points+/- .45MHz2000reps;
measuredcarrier/pi frompilot, unchangedsigma.05/readout8918.35/.64/10us,
T1.01-100us201,Ramsey0-60us201/.5MHz,Echo.01-100us251/.5MHz,
1200x1/250usrelax;Rabi0-.85/171/2000x1,normalSSF10k,res151x500.
Samecanonicalfolder,0A/off sources,attenuation,1800sdeadline,abortconditions.
Gain.0045 is a tuning test pendingreview, not promotedcalibration.

### 2026-10-28 17:01 - M3 long-drive pilot and final full plan
100us/.008 qspec pilot: center5551.359873MHz,FWHM.089431+/-.001960MHz,
pop.54490,R2.98297,rawIQ SNR39.81,RabiR2.99764,SSF.66865. Plot reviewed;
fringes largely removed. FinalfullM3b queued afterfullM1b: all7flags,
150usdrive/.009gain,151points+/- .45MHz2000reps; longerdrive justified by
M3 measured~50uscoherence; smallpowerincrease targets.1MHzwidth.
Measuredpilotcarrier/pi,unchangedsigma.124/RO8973.43/.95/13us;
T1.01-150us201,Ramsey0-80us268/.5MHz,Echo.01-150us376/.5MHz,
1200x1/250usrelax,normalRabi2000x1/171,SS20k,res151x500;sameguards/paths,
1800sdeadline. Population/width/physics gates remain pending the acquired result.

### 2026-10-28 17:03 - low-power M6 resonator needs more SNR
res_m6lin completed withplots/HDF5/sourceguards. Gain.05 scan hasvisiblebroad
asymmetricenvelope,butrawminimum9058.02MHz isanoiseoutlier; smoothedenvelope
minimumnear9058.6MHz andrise near9058.95MHz. Noise .0357,fullspan/noise7.96;
7/11/21point slope estimates unstable. DoNOTpromote rawminimum or a noise slope.
Intermediategain .10/.15 res-only scans queued afterfullM3b: same9058.34seed,
+/-3MHz301points2000reps10us,240sdeadlines; determine a reproducible slope before
anotherqubitsearch. These are readout diagnostics,not accepted qubit calibration.

### 2026-10-28 17:05 - active-plan race prevented
M2 lengthcheck5/7/9us gives.36535/.35976/.34395; choose5usshortestplateau.
FullM2a had juststarted at7us before planned update; guard rejected mutation.
Its plan/rawprovenance remainunchanged and acquisitioncontinuesnormally.
M2 highergain2Dgrid (.72/.84/.95 x8949.53/.73/.93MHz) at5usqueuedafterM6maps.
Gridplans corrected to5us BEFORE execution; workingcalibrationlength5us.
FullM2a is a useful baseline; final5usfullrepeat will be needed aftergridreview.

### 2026-10-28 17:06 - M5 long-drive/power test meets spectroscopy targets
LongqM5 gain.006,length100us yieldscenter7065.092268MHz,FWHM.186271+/-.003444MHz,
peakpopulation.39456,R2.98800,rawIQ SNR36.66,RabiR2.99880,SSF.6577.
Bothqspec targets achieved empirically. Carrier shifted-.1117MHz since earlierfull;
retain per-passqspec tracking. FutureM5lengthplans updated BEFOREexecution with
latestcarrier/pi; workingqgain.006. FinalM5fullshould use100us/.006,
T1.01-50us201points (covers potentialT1variation),Ramsey0-12us241points/2MHz,
Echo.01-30us301points/1MHz,2400x2coherenceaverages/100usrelax.
100usrelax is24x latestT1 and>7x priorzero-current13us; revalidatecurrentT1.
These areplanned windows; acceptonlyafterfull raw/plots/fitreview.

### 2026-10-28 17:14 - M2/M5 fresh quality review
FullM2a savedall7datasets/plots/guards: qfreq5438.522429326MHz,FWHM.13449MHz,
peak.27215,R2.8845 (20usfringes/shoulder),RabiR2.99434,pi.666606,SSF.33885.
T1=6.47218+/-.16816us accepted; Ramsey candidate7.1404us R2.80984 notaccepted;
Echo12.4903+/-.7285us R2.89134 independentlyaccepted but noisy, needs betterfullpass.
FinalM2 should use100us qspec/.009gain/151points2000reps plusselectedreadout,
T1.01-40us161points,Ramsey0-25us251points/1MHz,Echo.01-50us251points/.5MHz,
2400x1coherenceaverages/100usrelax (15xmeasuredT1),20kstatecalshots/100usrelax.
NewM5length atlatest7065.092MHz/pi.6467 gives2/4/6/8us .5236/.6471/.6480/.6658;
8usworkingselection, needs2Drefinementandrechecklengthplateau.
QueueM5gfc .9/.95 x9013.46/.56/.66MHz at8usafterM2highgaingrid.

### 2026-10-28 17:22 - matched M4 control and model-dependent acceptance
MatchedT1pre7.26358+/-.12331us,echo16.02308+/-.32428us,T1post7.53197+/-.13965us;
raw/plots/sourceguards/audits passed. Repeat confirms effectivefitvalues, not a
single-rate stationary2T1 interpretation. Independent offline modelcheck:
stretchedT1 improves AIC by20.4(pre)/14.6(post) overfree-offset exponential,
beta=.865+/-.027 and.874+/-.031. SeparateT1-vsT2 DC-offset model leavesecho~16.06us.
No forced2T1 bound orzero-offset correction applied. Report M4 fullRR effective
T1/T2R/T2E withasterisk/modelqualifier andpairedcontrol slide; no microscopic
mechanism inferred. Pulse/readoutcalibration accepted; conditionalfiterrors are
not modeluncertainty. m4_model_consistency.json retains fits/rawprovenance.
M2future5usgridplansupdatedbeforeexecution to latestfullRR carrier5438.522429MHz;
piunchanged. CurrenthardwarefullM1b ->fullM3b ->M6midpowerres ->M2grid ->M5grid.
All shared sourcecalibration stillunmodified; finalotherrowsawaitfullreview.

### 2026-10-28 17:25 - finalM1b accepted near linewidth boundary
All7normalRR datasets/plots/statecal reconstructions/sourceguards passed.
qfreq5515.333075600MHz,FWHM.094760+/-.001347MHz,pop.49770,R2.99277,
rawIQ SNR45.15;pi.686504303,sigma.05,RO8918.35/.64/10us,SSF.7280.
T1=15.55596+/-.20530us,T2R=23.89569+/-.45763us,T2E=25.27887+/-.53021us;
independentfitR2.993/.985/.974,>=4.99samples/fringe,physicalboundsplausible.
Manualreview accepts .0948MHz asnear.1 target (5% below); stricthelperflag remains
recorded, no value altered. Use thislatestfullpass,not yesterday's larger33.6/41.4/45.1us.
Temporalchange is measured; no mechanism assigned. final_runs row1 selectedfullM1b.
Shared code configs remainunmodified until all pending acquisitions complete.

### 2026-10-28 17:26 - QICK pulse-length limit failure and recovery
FullM3b failedatqspec pulse construction:150us ->92160cycles exceeds16bit65535.
Thisrun firmwareclock614.4cycles/us supports atmost~106.665us;100us=61440cycles
was alreadyvalidated. SourcesfinalCURR/0A/off both,returncode1. Partialresplot
exists,but no canonicalHDF5 persisted; do notusefailedrunas measurement evidence.
Failedlabel/log/receipt preserved. Classification: execution-environment/verification
preflight gap (plannedpulseexceededhardwareencoding). Narrowfix: newfullM3c
sameplanbut100us,unchangedunderlyingprogram. Addlocalplan guard<=100us for
thisrun anddurablelesson. M6mapcontroller76747 stoppedonpredecessorfailure;
resmid1/2neverstarted. Restartlinearqueue fullM3c ->resmid1 ->resmid2;
existing64383/49575waiterswillresumeonnewsuccessfulresmid2receipt. No concurrentacquisition.

### 2026-10-28 17:38 - finalM3c accepted near spectroscopy targets
Recovered100usfullpass savedall7HDF5/plots/calibrationreconstruction/zero-sourceguards.
qfreq5551.361657087MHz,width.093234+/-.001621MHz,pop.52483,R2.98922,SNR46.01;
pi.656656290,sigma.124,RO8973.43/.95/13us,SSF.70575,res8974.09MHz.
T1=39.04880+/-.67486us,T2R=39.30391+/-.72878us,T2E=48.71536+/-.83897us;
independentR2.992/.984/.975,>=4.99samples/fringe,physicalboundsplausible.
Manualreview accepts near-targetwidth.0932 (7%below.1) andpop.525 (5%above.5),
withactual values/stricthelperflags retained; no clipping/forcedfit or further
powerchasingwhilecoherencechanges. Latestfullreplacesyesterday57.4/50.1/58.6us.
Sourcecalibrationnotpromotedyet. RemainingM2/M5finalsettings andM6discoveryqueued.

### 2026-10-28 17:44 - highgainM2 and repeatM5 readout
M2gfd at5us: .95gain/8949.73MHz .5062,8949.93 .5052 versus .84gainbest.4378
and.72gain~.31. Select .95/8949.73 workingpoint; rechecklength3/5/7/9us then
fine2D .9/.95 x8949.63/.73/.83 ifneeded beforefinalfull.
M5gfc at8us max.6589 at.95/9013.56 reproducesprior.6658 withinshotvariation;
neighbors .6357-.6518. Rechecklength4/6/8/10/12us to confirmshortestplateau.
Same measuredcarriers/pi from latestpilots,normalSSF10k/state,sourceguards,
normalattenuation,canonical1028folders,900sperstagediagnosticdeadline.
M6mid.15 res hasspan/noise18.68 andrisingedge9058.94MHz (7/11/21smooth
9058.98/.94/.84); lowerpowerdiagnosticreadoutselected9058.94/.15/10us,
not a qubit calibration. New boundedfrequencywindowsearch may followfinal5row
passes; do notuseitsautomaticnoise fits as carriers.

### October 28 17:49 - final M5 / bounded M2 readout checks

All previous controllers completed. M5 length4/6/8/10/12us at gain.95,
9013.56MHz gives SSF .6472/.6682/.6986/.6863/.6840; choose8us shortest
plateau. Prior2D at8us already selectedsame .95/9013.56 withno meaningful
improvement, so alternatingoptimizationconverged. M2 repeated3/5/7/9us
.3829/.4213/.4224/.4520 variesfromprior .5062 at5us; retainall evidence.
Next sequentialqueue: m2_lenf1..4 (7/9/11/13us), m2_gfe1..6 (9us,
gain.9/.95 x8949.63/.73/.83MHz), then full_m5b. All acquisition remains
round_robin_benchmark.py. M2 SS only10kshots; M5 all7flags, qspec7065.092268
+/- .6MHz151points2000reps,gain.006/100us; sigma.056940825us,pi.646707;
RO9013.56/.95/8us. Rabi0-.85/171/2000x1; T1.01-50us201,
T2R0-12us241/2MHz,T2E.01-30us301/1MHz,2400x2 coherence/100usrelax,
20k statecalshots/100usrelax. Attenuators unchanged10/15dB readoutDAC,
5/4dBqubitDAC,17dBADC. BothYokos mustreadCURR0A/outputOFF before/after.
CanonicalOctoberroot; path<260chars. SS stages900sdeadline/full1800s;
abort on source mismatch, acquisition/save failure. No nonzero flux or
attenuator change. FinalM2full remains pending nextreadoutreview.

### October 28 17:53 - final M2 selection and bounded M6 follow-up

M2 repeated7/9/11us SSF.48665/.4779/.4077 (13us also acquired).
9us 2D gain.9/.95 x8949.63/.73/.83 gives .4118/.4146/.4803 and
.44164/.4897/.4305. No significant improvement over earlier5us .5062;
readout/time variation retained. Choose7us shortest7-9us plateau, .95gain,
8949.73MHz. full_m2b all7flags follows full_m5b, measuredcarrier5438.522429,
sigma.101us,pi.666606;100us/.009 qspec +/- .45MHz151points2000reps;
T1.01-40us161,Ramsey0-25us251/1MHz,Echo.01-50us251/.5MHz,
2400x1coherence100usrelax,20kcalshots. Full1800sdeadline.
Then hunt_m6win1..6 via normalRR q_spec only, rawIQ/no scaling. M6index5,
physicalQ6,Yoko4 remains0A/off. Bounds4300-4810,4790-5310,5290-5810,
5790-6310,6290-6810,6790-7300MHz; .5MHzsteps,1000reps,100us/.1drive;
readout9058.94MHz/.15gain/10us from fresh res-only diagnostic. Each600s
harddeadline, expected~220s, write HDF5/PNG aftereach. Maxartifact208chars.
Samefixedattenuators10/15dBreadout,5/4dBqubit,17dBADC. CanonicalOctoberroot.
Abort on QICK/save/source mismatch; no guessed Rabi/coherence fromnullqspec.
Bothsourceguards0A/CURR/OFF/.01Arange before/aftereach. No flux changes.

### October 28 18:04 - M5 fullb review flags retained

FullM5b saved all7 HDF5/plots, sources0/off. Carrier7065.146080MHz,
width.225403+/-.007441MHz,pop.226985,R2.96368; RabiR2.99741,
pi.646707,SSF.5696. Normal/independent coherence agree: T1 5.40334+/-.09070us,
T2R2.16354+/-.05025us,T2E11.43937+/-.14180us,R2.987/.970/.991,
~10pts/fringe. All metrics measured; preferred spectroscopy targets missed.
Echo exceeds2T1 by.63us (~2.75combinedfiterrors); do not silentlycallstationary
or eraseflags. M5 fullb remains reviewed but calibrationnotcomplete.
Prepared two bounded qspec/Rabi/SS checks m5_powerlo/.0045 andpowerhi/.008,
100us/+-.6MHz151pts2000reps, thennormalRabi0-.85/171/2000x1 and20kSS,
readout9013.56/.95/8us, measuredsigma/pi,carrier7065.146080 seed only.
Queue follows hunt_m6win6; no concurrency. Same attenuators/root/sourceguards,
path<260,1800sdeadline,abort onhardware/save/source mismatch. Power tradeoff
is empirical; no attempt to forcepop/linewidth target by altering QICK program.

### October 28 18:13 - M2 interpretation resolved as structured spectrum

FullM2b all7raw/plots finite, 0/offsourceguards. T1 6.09292+/-.10815us,
Echo10.42431+/-.30813us independentlyaccepted,R2.991/.950.
NormalRR T2R7.33859+/-.37281us,R2.8697 is an EFFECTIVE singlefrequency fit,
not a clean stationary singlefrequency decay. All~10pts/fringe.
Offline comparison from actualfullb population: twoLorentzians R2.9681 vs
single.8611, AICimproves216.2. Peaks5438.399967+/-.002319 and
5438.533518+/-.001049MHz, widths.08947/.09145MHz, amplitudes.1437/.3218.
Earlier independentfulla repeats peaks5438.400511 and5438.532987MHz,
mainwidth.09189, twoLorentzR2.9482 vs.8845, AICimproves187.1.
Fullb Ramsey twofrequencyshared-envelope R2.9562 vs.8697,AICimproves267.4,
frequencies.985647/1.123891MHz, separation138.24kHz vs spectroscopy133.55kHz
(recorded-delay scale correction~1.0072). Modelenvelope8.26976us is supplemental,
not substituted for normalRR7.33859us. No microscopicmechanism inferred.
Accept empiricallyvalidated pulse/readout and T1/Echo; retainnormalRR carrier
5438.517519MHz as EFFECTIVE workingcarrier, explicitly notunique spectralpeak.
RabiR2.99597,pi.666606,sigma.101us,SSF.4958 at8949.73/.95/7us.
Maincomponent width~.091MHz/pop.322 is near target and distinguishable; further
singlelinepower-tuning cannot be assumed to remove repeatable intrinsicstructure.
Keepnormalplots, addcomparison slide andstarcarrier/T2R. m2_structure.json/png
and earlierindependentcomparison retain fullprovenance/fitassumptions.

### October 28 18:21 - M5 offline relaxation model check

Separate fulla/fullb T1 population traces both show strong curvature. Stretched
exponential beta.6734+/-.0166 (a),.65497+/-.01045 (b), AICimproves168.4/344.2
over singleexponential. Fullb R2 rises.98742->.99775. This is phenomenological,
not a matchedcontrol or microscopicmechanism. Extrapolated amplitude isfree
and >1, so do not interpret stretched tau/amplitude as validated physical
population/lifetime. RetainnormalRR effective5.40334us T1 and11.43937us echo;
stationarysingle-rate2T1 comparison isnotestablished. m5_relaxation.json/png
retain assumptions andallfitparameters. Powerchecks stillpendingafterM6.

### October 28 18:27 - bounded M6 sensitivity follow-up prepared

Windows1..5 rawIQ maxvector scores3.35/3.72/3.92/3.54/3.41, all noise-like;
no acceptedcarrier, including historicalregion6290-6810MHz. Window6 stillruns.
After queuedM5powerhi, res_m6high usesnormalRR res_spec only,M6/index5,
seed9057.72MHz (measuredOct27gain.95), +/-2MHz201points/.02MHzsteps,
1000reps,readoutgain.95/5us. This tests higher-gain/shorterreadout sensitivity;
currentlowergain.15/10us mayhideaweakline. Reviewfreshresonatorslope before
choosingonefocusedhighgain qspec test; do not promotea guessedqubitfrequency.
Sameattenuators10/15dBreadoutDAC,5/4dBqubitDAC,17dBADC,canonicalOctoberroot,
path<260chars,900sdeadline, bothYokosCURR0A/outputOFF/.01Arange guards.
Abort onsource/acquisition/savefailure. No flux orattenuatorchanges.

### October 28 18:36 - fresh M6 high-gain edge and final queue

All6 lowergainwindows4300-7300MHz saved raw/plots, maxvector scores<4,
noacceptedqubit. Freshres_m6high atgain.95/5us findsdip9057.72MHz andVERY
steep positiveedge9057.78MHz, stable across7/11/21smoothwindows; previous
highgainRO9057.30 was onmuchshallowerleftside. Use measured9057.78 for
ONEfocusedstrongrawqspec: hunt_m6strong, M6index5, historicalsearchcenter
6522.489876+/-75MHz =6447.489876-6597.489876,601points/.25MHz,3000reps,
drive.1/100us,RO9057.78/.95/5us,rawscalingfalse,900sdeadline (~370s).
CanonicalOctoberroot, maxrawpath210chars; noRabi/SSF/cohfromseed.
Then full_m5c all7 normal flags: carrierseed7065.186369 fromlastpowercheck,
qgain.006/100us,151points+/- .6MHz2000reps; retainreadout9013.56/.95/8us,
measuredsigma/pi,Rabi0-.85/171/2000x1,20kstatecal/100usrelax;
T1.01-50us201,T2R0-12us241/2MHz,Echo.01-40us401/1MHz,2400x2coh100usrelax,
1800sdeadline. Powerlo .0045 width.19690/pop.11957; powerhi.008width.26695/
pop.23598: highergaindidnotmeaningfullyimprovepopulation vs.006fullb but
broadenedline. Retain.006compromise andexplicitlyflaganytargetmiss.
Samefixedattenuators10/15readout,5/4qubit,17ADC dB; bothYokosCURR0A/OFF/.01A
before/aftereach. Abort onacq/source/savefailure. No current/attenuatorchange.

### October 28 18:44 - M6 sensitivity follow-up remains null

Freshhighgain9057.78MHz/.95/5us,drive.1/100us,3000reps focused6447.49-6597.49
scan savedraw/plot, allfinite, maxvectorscore3.515 at6507.238MHz isnoise-like,
noacceptedline. Sources0/off. Do notturnthispointorhistorical6522.489876
seedintoa pi/SSF/coherencebenchmark. M6resonly ismeasured9057.72MHz atgain.95,
and9058.38MHz atgain.15; readoutpowerdependencemustremainexplicit.
FullM5c nowacquires normalall7. Useroptionalwiringquestion remainsunanswered.
NewreusablebenchmarkPowerPoint belongsdirectlyin the explicitlyrequested
October7runroot; this directuserfolderinstruction supersedes thehistorical
pucq4_first_light runfoldername. KeepallnewrunpathsOctober7; nohistoryrenaming.

### October 28 18:58 - final hardware review complete, qualifications retained

Controller28579 finished allstages; sourceguardsCURR0A/OFF/.01A afterfinalC.
FullM5c all7raw/plots audited finite. Carrier7065.155041601MHz,sigma.056940825us,
pi.641732283465,RabiR2.99813,SSF.5654,RO9013.56/.95/8us.
Qspecwidth.266190+/-.008193MHz,pop.226685,R2.96912: targetsNOTmet despite
boundedpowerchecks. Reviewcomplete doesnotmean nominalcalibrationcriteria met.
NormalC T1 7.390912+/-.131847us,T2R3.510375+/-.187661us,
Echo7.539772+/-.157073us; independentfitsagree,9.33/10ptsperfringe.
RamseysingleR2.86324; twofrequencyshared-envelopeR2.97011,AICimproves360.5,
freq2.043808/2.181642MHz,separation137.834kHz, supplemental tau4.16827us.
Do notreplace normal3.510375 withthismodeltau orclaim a microscopicmechanism.
T1stretchedbeta.6291+/-.0110,AICimproves367.3; extrapolatedamplitude1.286
isphenomenological,notphysicalp0. RetainnormalRR effectiveT1/T2R withwarnings;
EchoCwithin2T1butthisisnotproof ofstationarity. EarlierBvalues5.40/2.16/11.44
remainhistory; latestCselected, no cherry-picking. M2/M5 Ramsey separationsboth
~138kHz; possiblecommoninstrument/systemcontribution isnotexcluded. No device
mechanism assigned. Change earlier logwordintrinsicstructure toobservedstructure.
M6 remainsresonly; no confirmedline afterbroadandfocusedreadoutchecks. User
optionalwiringquestionunanswered. No nonzero flux authorized/acquired.
Proceedreviewed-sourcepromotion, safeofflinechecks, finalPPT/manifest/archive.


### October 28 - configuration promotion and report closeout

Final49-slidePowerPoint saved directly in the user October root. Selected
fullM1b/M2b/M3c/M4b/M5c results used; freshM6res-only and raw search evidence
included. Source promotion applied only after all acquisition finished.
PureAST extraction verified shared arrays and acquisition profiles against
selected HDF5, integerloopcounts, allnonzero profiles/tables unchanged and
privatehistoryunchanged. 161-entryold/new/unit ledger produced. Safeharness
passed148Pythonfiles; noimports/hardware. Finalwhitespace/package/archive
checks follow. Nohardware rerunafterpromotion claimed. M5 nominaltargets
remainmissed; M2/M4/M5 interpretation flags remain visible. M6 unresolved.

### October 28 19:05 - final package verified

The 49-slide PowerPoint package reopened successfully and passed ZIP CRC and
slide-bound checks. The report, 161-entry old/new/unit change ledger, and
benchmark manifest are in the user-selected October run root. Completed plans,
receipts, logs, analysis figures, and initial/before-promotion/promoted source
snapshots are archived under benchmark_1027. No raw measurement was deleted
or overwritten. Both final acquisition guards verified 0 A and outputs OFF.
The offline harness passed 148 Python files; pure AST checks matched shared
configs and profiles to selected HDF5 data, including integer counts and
preserved historical/nonzero settings. git diff --check passed. Native Office
rendering was not performed; package/geometry checks and source-plot inspection
were performed. No hardware run after source promotion is claimed.
M6 qubit metrics remain unresolved. M5 spectroscopy targets remain missed and
qualified coherence fits remain explicitly flagged; this is not a claim of a
fully calibrated six-qubit device.
# October 29: authorized remaining-qubit offset-readout retry

18:30: M6 readout confirmation accepted at9057.695MHz/.95/45us, independent
SSF.77798/.77668 (50k/state), length plateau bracketed and 2Dgain/frequency
rechecked at45us. Earlier13us point was not a plateau; extending to57us and
then re-optimizingfrequency was necessary. Track actual last-grid length rather
than the temporarily updated working length, so endpoint extension cannot skip
the required gain/frequency recheck. Allraw/plots retained. Free-decayT1 control
3.1145us justifies100usreset; no readout-induced mechanism inferred.
Only18969 now acquires two lower-drive qspec checks before full M6benchmark.
No shared sourcecalibration promoted; allpriorcontrollersended.

18:09: first readout pass through all six rows ended. Controller6548 stopped
on M4 final-vs-grid mismatch: examples.40294/.42094 at8999.25/.95/5us versus
priorgridmax.48645; pairdifference.018 passes repeat gate but selected maximum
does not represent the later state. M3 examples.65476/.64814 pass. Need fresh
M2/M4 pulse controls and reference-interleaved setting comparisons; do not
increase averaging or select a high outlier to conceal time variation.
Only controller80875 now takes fresh M6 T1 with1000usreset, bounded160uswindow
and600sdeadline, before choosing finalSSFreset/window. NormalRR/HDF5/plots.

17:59 M2 raw diagnostic: identical8949.73MHz/.95/5us measurements show
empirical-CDF contrasts.46745/.41815/.42360/.39574, matching reportedhistogram
values. Projected separation falls2.219->1.877 whileg/e noise stays~1.63/2.09.
Ground centroid is nearly stationary; excited centroid shifts. This is actual
state-cloud variation, not histogram-binning error. No physical mechanism is
identified. Fresh qspec/Rabi/SSF control pre2_b prepared before acceptance.
Diagnostic m2_readout_stability.json/png retains all four raw paths.

17:53 verification/state failure: M2 selected5us/.95/8949.73MHz did not reproduce
the single best grid value. Independent50k/state repeats.42336/.39546 differ
.02790 and miss gridmax.46731 by more than allowed. Original optimizer43902
stopped before M3/M4 as intended; both source guards0A/OFF. No promotion of
M2 readout. Resume independent M3/M4 optimization, then recheck M2 pulse and
temporal readout behavior; do not increase shot count to mask nonstationarity
or publish the maximum as representative. M1 examples.80604/.82842, M5
.68410/.68488, M6.53308/.52706 saved, all still require fixed-setting length
plateau confirmation. A chosen endpoint is not proof of a plateau; longer
length checks prepared before final acceptance.

17:27: ONLY controller43902 optimizesall6 rows6,5,1,2,3,4 after all fresh
qspec/Rabi/SSF preludes. M5 meetsnominalqspec targets inpre5_a:
7065.135356MHz,width.152162MHz,pop.457623,R2.986915,pi.626808277,
baselineSSF.68635. M6 adjusted sigma.0555868301534722us givespi.641732283465,
RabiR2.994258,SSF.31325; qspecwidth.266709MHz remains a targetmiss.
M6 intermediate gain sweeps .75/.85 reproduce dip9057.98/9057.86MHz;
readoutgrid uses per-gain measured dipoffsets to compare gains. LengthA improves
SSF to.4164 at9us; gridA reaches.46510 at9057.72/.95/9us. Allplots/H5saved,
sourceguards0A/OFF. Optimizer final examples2x50000/state and repeat checks;
normalfullRR planned after selected readout toverify calibratedpopulation/coherence.
No shared configuration promoted during acquisition. Earliercontrollersended.
Fixed one cp1252 en-dash byte introduced by my PowerShell log append; original
bytes archived locally, only invalid UTF8 byte repaired. Use UTF8-aware edits
for repository logs. All source/raw-acquisition files retained.

16:50 adaptive update: first off6_ha at9057.60/.95/5us reveals a strong
multi-point line near6404.600482MHz (jointIQ SNR14.83, FWHM1.949MHz,
residualRMS1.004noiseunits); visually reviewed rawI/Q/HDF5. Independent
repeat/lower-drive tests queued before any pulse calibration. Initial remaining
wide plans archived then narrowed6398.6-6410.6MHz, full settings in task_brief.
No shared qubit config promoted yet. Previous single-edge nulls do not rule out
qubit visibility at another resonator offset.

User requested fresh M6/Q6 discovery and remaining calibration, suggesting an
offset resonator readout. Re-read safety/state/current source and preserved all
existing changes. New IDE workspace benchmark_1029 archives initial sources and
the task/hardware envelope. Only normal round_robin_benchmark.py acquires data.
Fresh res6_hi/mid/low_1029 completed with raw HDF5 and PNG; minima9057.72MHz
at .95/5us,9058.40MHz at .3/10us,9058.30MHz at .15/10us. Both Yokogawas
verified CURR0A/outputOFF/.01Arange/30V before and after each stage.
Sequential controller23131 now tests measured left/dip/right readout offsets;
6372.49-6622.49MHz raw qspec,1001points,1000reps,.1/100us pulse,600s/stage.
Historical qubit carrier remains unvalidated. No current sweep or guessed
Rabi/coherence. Previous Oct28 artifacts/source calibration remain intact.

October29 off6_hb independently repeats the M6 line at RO9057.70/.95/5us: joint-IQ center6404.544778MHz, width1.765070MHz, vectorSNR9.40, residualRMS.844noiseunits. off6_ha center6404.600482 differs56kHz; both broad strong-drive lines. Lower-drive repeats running before pulse calibration. Readout-offset dependence is being measured; no sole-cause attribution from changing both power and offset.
October29 17:05: all7offset spectra saved PNG+H5. M6 repeated at lowdrive .01/100us center6404.640241MHz (normal width.273689MHz; jointIQ center6404.644829,width.295489,SNR16.96). Initial Rabi6_a confirmed coherent drive R2.992887,pi.577061588 at sigma.062612796167us,SSF.31215. Shorten sigma by measuredpi/.65 and rerunqspec/Rabi/SSF; gain.0075/100us trial below.01. Activecontroller75081 pre1–5;44774 waits for pre5 thenrabi6_b;23131 ended. User expanded all-six readout optimization + independentSSF examples + otherdeck update. Pendingdeckclarification. Readout optimizer prepared but not launched untilpulse readiness reviewed.

## November 11 - Q2 SSF failure repair, report and authorized restart

Scope: user requested results/PPT, then explicitly repair failed24h and restart.
Prior Nov4 series failed at2660.5s,2974raw+plots: WinError5 replacing status JSON.
Acceptance: preserve old raw evidence; nonfatal status locks with strict raw-save
checks; full86400s timer; exact prior calibrated Q2 settings; realraw+PNG verification;
independent watchdog and report with measured coverage clearly labelled partial.
No new calibration, source parameter changes, unrelated process termination or
claims of full-day completion. Hardware envelope in IDE/ssf_restart_1111/RUN_BRIEF.txt
and tracking.json; script remains existinground_robin_benchmark.py, ss-only/index1.

Fixed tools/ssf_progress.py and three RR status-write sites. Hardware-free injected
transient/permanent locks, valid oldJSON, recovery, independent localmirror and
cleanup passed;149-file harness passed;diff-check passed. Shared configs byte-identical
to November4 snapshots; old Python acquisition processes zeroCPU/write changes and
no lab connections. Plenty of M: space; projected HDF5 path201characters.
Controller copied/reviewed from prior wrapper; local progress watchdog600s, total
24h+15min bound; nonfatal supplemental receiptcopies/status read errors. No rawdata
or experiment class/schema changes. Both Yokos read CURR0A/OFF/.01A/30V before launch.

Launched controller20404 / RR17952; canonicalssf_24h_q2/2026-11-11_12-52-52.
Start1794419573.5911183, deadline1794505973.5911183 (Nov12~12:53local).
Live Windows network-status readlock3s reproduced sameWinError5: rounds173-175 still
saved, local status advanced, network sidecar then recovered. Round211 raw verified
4finite2000samplearrays, exact pulse/readout/relax settings andcorrespondingPNG.
Evidence restart_verification.json. Running, not yet complete. Original failed
Nov4 run and original stale status/tmp preserved. Failure mode/state docs updated.

November11 report complete: report_1111/Q2_SSF_partial_results_20261111.pptx
and matching PDF under original Nov4run/documentation.12slides, nativePowerPoint
render/export inspected. All2974 HDF5 files validated, contiguousrounds, each
4finite2000pointarrays, fixedsettingsmatch,2974normalPNG paths. Recomputed
canonicalcontrast exactlymatches all2974 storedvalues. Median.1100413,
p05.0740120,p95.1462462; first5minmedian.1090545,last5min.1130283.
11,896,000preparedshots across44.34minutes, not24h. RawIQcentroidtrends and
originalearly/middle/final RRplots included; noTLSswitching/assignmentaccuracy
claim. CSV,summaryJSON,analysiscode andfigurescopied alongside/hashverified.
Restart remains active; no hardwarefaults observed during initialverification.

## November 16 - verify completed Q2 SSF and package like supplied ZIP

User asks completion status and a package matching Downloads/Single_Shot_Fidelity_package.zip.
Read-only source/controller review: repaired run completed64833 rounds in86401.234s,
returncode0, statuscomplete; finalNovember12sourcesCURR0A/OFF/.01A/30V. No hardware
query/restart/config change. Intended export preserves allrawHDF5/plots, accepted
Nov4calibrations, source provenance and reference hierarchy; documentationtrio
(two self-contained notebooks plusWordguide) updatedfornewrun. Acceptance: enumerate
all64833 HDF5/PNG pairs; validrounds/finite2000arrays/settings; preservebytes; verify
allZIP CRC/SHA256; openWordguide; delivernewZIPwithoutoverwritingreference.

IDE notebook creation rejects outside-project paths. Created package notebooks
through IDE under temporary _ssf_package_docs_1116; adjusted samplepaths/contrast
labels and removed oldoutputs. Notebook execution attempted but kernel stayeddown/
queued; interrupted ownqueuedcell. Do not claim end-to-end kernel execution.
Word compatibility test exposed ElementTree removal of unused namespace declarations
while retaining mc:Ignorable prefixreferences; remove staleIgnorable before saving
rewritten WordXML. Corrected guide test opened in native Word. Package tail repair
preserves all earlier compressedrawbytes and rechecks archive members against manifest.

November16 export delivered: Downloads/Single_Shot_Fidelity_Q2_24h_20261111_package.zip,
19100022814bytes/129721members. SHA256 sidecar saved alongside. Everyarchivedsource
SHA256 andZIPCRC verified. Fullrawaudit64833rounds,259332000preparedshots,allsettings
matching;largesttimestampgap6s. Corrected2pageWordguide opened/exported natively.
Notebooks created viaIDE, freshpaths/emptyoutputs/syntaxvalidated; no activekernel
was available so end-to-end notebookexecution is not claimed. Originalarchive and
rawdata unchanged.149-fileofflineharness andfocused diff-check passed. Fullreceipt,
reproduciblepackagingcode andguidepreview in IDE/ssf_package_1116. No acquisition,
currentquery or configurationchange. Whole-runmediancontrast.08254127,p05.04852426,
p95.119; firsthour.08952238,lasthour.08852213. Report as contrast, not accuracy.
## 2026-11-16: Q2 multi-day Zeno setup and launch

- User requested the existing longitudinal_qze_005_ge_ramsey script, calibrated
  Q2/M2/index1, correct PUCQ4 paths and days of acquisition. Task-start source
  backup, task diff, RUN_BRIEF and controller are in IDE codex/zeno_q2_1116.
- Preserved existing user-enabled baseline and Zeno stages. Replaced stale
  squill/index5/path and hard-coded frequencies with shared measured PUCQ4 row;
  Q2 gain .25->.192 DAC, length9->7 us, offset0->-1.014313 MHz. Shared sigma,
  pi and frequency settings were already correct, so no shared-config edit.
- Applied existing baseline acquisition profile without altering Zeno grids.
  Added72h timer after baseline, cycle-boundary STOP and progress markers.
  Fixed one bare T2R exception to honor debug_mode and expose save/hardware errors.
- Offline harness:149 tracked Python files parsed; task-start diff reviewed;
  focused whitespace check passed. Longest calculated raw path218 chars.
  M: free4.6TB. Existing acquisition-like processes had no CPU/write movement
  or lab connections in read-only sampling; none were terminated.
- Both Yokos verified CURR,0A/OFF,.01A,30V before launch. Controller4844,
  child40624. Fresh folder qze_005/q2_2026-11-16_17-18-08. Baseline resonator
  PNG and HDF5 exist, measured dip8951.05439MHz; baseline review ongoing.
- Long-run completion remains pending. Watchdog fails on child error,6h without
  saved raw data, source-state violation, <5GiB free or82h total; no automatic
  hardware retries. STOP at timestamped sub-study finishes the active cycle.
### November16 Zeno first-attempt calibration rejection

Stopped owned child40624 before Zeno after short Ramsey correction moved
qspec5438.3663669MHz to5430.904935MHz. First Rabi .696454 was plausible;
post-correction .562138 failed the accepted window. Preserved all first-run
data. Controller recorded failure and freshly verified both sources0A/OFF.
Short Ramsey is now diagnostic-only for Q2, preserving qspec frequency;
second Rabi still runs and final .6-.7 pi gate precedes Zeno. Offline harness
passed again. Restart uses IDE codex/zeno_q2_1116/attempt2 controller and a
new timestamped M: folder; inspect that receipt for current ownership.
### November16 Zeno restart passed baseline and entered acquisition

Attempt2 folder `PUCQ4/qze_005/q2_2026-11-16_17-22-26`; controller52228,
child46100. Res8950.91439MHz, readout8949.900077MHz at .192DAC/7us,
qspec5438.387131361903MHz/FWHM .163MHz, Rabi .686504 then .676555 at
sigma .101us. Baseline SSF contrast8.61%, consistent with earlier tracking.
Q-quadrature Rabi fit-shape R-squared .902/.889; noisy but repeated pi agrees
with prior .676555 and both pass .6-.7 acceptance. Plots and readable HDF5
verified for baseline;10 files include the script's inherited unused second
Ramsey-correction placeholder. No accepted lifetime fits claimed (fit_data=False).
First Zeno high-gain qspec began17:28:44; unchanged gain .000001-.04/50points,
qubit5430.4996816-5439.4996816MHz/500points,20reps*20rounds. Normal readout
.192,7us; Zeno tone8949.900077MHz. Baseline review JSON in attempt2 workspace.
72h timer reaches November19 17:28:44 local, then finishes active cycle.
First Zeno sweep raw-save and full-duration completion remain pending.
Live QICK acquisition returned averaging round1/20 after58.96s without an
exception; acquisition46100 and controller52228 remain running at handoff.
Baseline10 HDF5/12 PNG verified, actual maximum artifact path212 characters.
Review JSON and task diff copied into the new timestamped run folder.
## 2026-11-17: Investigating weak Zeno qspec and coherence

User requests minimal fixes and Ramsey/echo settings that resolve prior beating.
Original Q2 run completed five Zeno cycles and round5 high-gain qspec. Stopped
owned46100 at10:34:11 only after that HDF5 closed; controller confirmed0A/OFF
sources afterward. Original data preserved, not a completed72h run.
Workspace IDE codex/zeno_fix_1117 contains task-start backups and exact diff.
Confirmed wrapper defects: magnitude scaling differs from baseline signed IQ
projection; qspec ignores configured reset with final_delay=.5us. Archived
raw projected spectra reveal a weak line5438.18-5438.41MHz; gain-averaged
comparison is diagnostic, not proof of adequate individual-gain SNR.
Changed three plot projection lines, one wrapper final_delay argument, and
local script settings: qspec±.45MHz/151pts/2000x1/100us reset, Ramsey0-25us/
251pts/2400x1/1MHz/100us,echo .01-50us/251pts/2400x1/.5MHz/100us. No shared
config, bias, gain grid, readout-power or low-level pulse-program edit so far.
Offline149file harness, focused diff, projection regression passed.
Fresh two-gain0/.04 diagnostic check1117/2026-11-17_10-34-45 still has weak
qspec and near-ground Ramsey, so long-run restart is withheld pending timing
comparison. Existing5us delay also separates final qubit pulse from readout,
versus baseline.01us. Next bounded diagnostic compares baseline and Zeno with
.01us pre/post delay, preserving all other pulse/calibration values. This is
not yet a proven cause or a permanent timing recommendation. Both source
identities/mode/zero/output/range/compliance checked before each acquisition.

November17 validation/restart follow-up: four bounded checks completed,11HDF5 and11canonicalPNG, both sources CURR0A/OFF/.01A/30V before/after. Projection/reset/averaging alone did not restore contrast; .01us final-readout tests recovered qspec/fringes. Final optional readout_ringdown_delay=.5us preserves5us ring-up and original Zeno tone; changed only final wait in three active programs, defaults unchanged. Main local qspec carrier-1 to+.5MHz/251pts/2000x1/100usreset; Ramsey0-25us/251/2400x1/1MHz/100us;echo.01-50us/251/2400x1/.5MHz/100us. State references2000->10000shots/state. ROpower/sigma/pi/attenuators/sharedconfigs/gain-grid unchanged.
Final confirmation confirm1117/2026-11-17_11-01-37: zero-gain qspec5438.251226+/-.008845MHz/FWHM.131178MHz; gain.0399805 center5438.107154+/-.023145MHz/FWHM.421692MHz. Finite per-point noise remains (amplitude/residual2.53/1.74); these are diagnostic single-line fits, no shared-calibration promotion or TLS doublet claim. Early Ramsey/echo oscillations recovered, but prior beating not convincingly reproduced. All low-level statements except3optional finalwait expressions match task-startAST.149-fileharness and whitespace passed.
Restart11:05:47: qze_005/q2_2026-11-17_11-05-47; controller67668, acquisition29564, freshbaseline then72hZeno interval. Six-hour noHDF5/82hwall/disk/source guards. Projectedlongestpath218. FreshresHDF5 saved; remainingbaseline and firstfullZeno map pending at this entry. Repairreport, diagnosticscripts, raw-reference receipts, beforeafterplots and taskdiff in validation_1117 withinrun. Original raw preserved. Completion must be checked from live receipt/log/raw coverage; restart is not72hcompletion.

11:12:05 baseline completion verified:10HDF5/10PNG in newrun, freshcarrier5438.247946163877MHz, finalpi.6715802966489653, RO8950.000077MHz (res8951.01439MHz plus unchanged-1.014313offset). ZENO_CYCLE_START0 recorded and first25.1M-shot high-gain sweep advancing normally. 72h timer started11:12:05Nov17; full-cycle finish may extend pastNov20same time. FirstfullZenoHDF5 not yet saved (~89min sweep); bounded finaltwo-gain diagnostics already saved/validated. Controller67668/child29564 active.

PROVENANCE CORRECTION from final live-config audit: longitudinal script line205 assigns bare_non_offset_res_freq=offset_res_freqs[0], so the name is misleading. Original long run used Zeno tone8949.900077MHz (offset RO), and resumed run preserves this choice at freshly measured8950.000077MHz. The ring1117 and confirm1117 diagnostics used8950.91439MHz, the non-offset resonance; earlier descriptions calling this the original tone were incorrect. These are additional non-offset-tone tests, not exact reproductions of every resumed setting. timing1117 used the original offset tone with.01us pre/post waits and recovered signals. Zero-gain comparisons establish the timing-related recovery independently of tone frequency. The full50-gain,.5us-finalwait,5us-ringup sweep at the preserved offset tone is NOW acquiring; its first complete map remains pending. No hardware/source-file tone change was made in response to this audit. Do not misrepresent the non-offset diagnostic as exact-condition validation of the running map. Raw configs are authoritative.

## November18 - repeated blank Zeno maps, repair in progress

Task user asks fix qspec/T1/Ramsey/echo keeping overall script structure. Currentbaseline/raw reviewed, task-start backups in IDE/zeno_fix_1118. Lastworkingrun q2_2026-11-17_11-05-47 preserved28HDF5, through round3T1; ownedchild29564 terminated10:01:34 with partialround3Ramsey unsaved. Controller67668 recorded failure/manualstop and bothYokos CURR0A/OFF/.01A/30V final. No originalraw artifacts deleted/rewritten.
Fullmaps containnoise, not emptyfiles. Earlyqspecgain-averagedline presentcycle0/1, lostcycle2/3. Mainheldstartupcarrier5438.247946/pi.6715803 for~23h. ReferenceSNR fromraw10kshots declined Qspec11.43(cycle0)->1.65(cycle3), T111.17->.65; dividing by unresolvedcentroids inflatesnoise. Q2currentfrequencydrift is NOT quantified: a freshwidefitcandidate5436.438 andnarrow5436.269 hadno reliableline andfailedRabi; theyremainrejected. Earliercommentarycandidate language corrected beforepromotion. Neverread these asvalidated1.8MHzshift.
Freshrecal1118/2026-11-18_10-04-51 savedRes/QSpec_wide/QSpec_repeat/Rabi; res8950.97439,RO8949.960077/.192/7us,sigma.101. Qspecwide�2MHz/401/5000x1/.009gain100us; rawreviewweak/noise, Rabi .6-.7gatefailed. No sharedconfigs changed. survey1118/10-16-59 completed85SS2000shotspoints(17qubitfreq5434.247946-5442.247946,5RO8949.500077-8950.500077); max.052/refSNR~3 notaccepted. ro1118/10-20-37 completed55SS20kpoints(5qubit5437.247946-5439.247946,11RO8947.500077-8952.500077),max.0227/refSNR4.17, no acceptedcontrast. FixedRO.1927us/pi.67158/sigma.101, no power/current change. Bothsources0/offbefore/after everyrun.
ACTIVE search.pyPID61004 search1118/2026-11-18_10-26-27: existingbaselineQspecprogram,highgain.02asoriginalZenoscript,100us,5413.247946-5463.247946MHz501/2000x1/100usreset,3RO8950.000077/8949.500077/8950.97439,.1927us. Firstsavedraw atlastworkingROhasno convincingline (matched-candidate max4.06 overmanyfrequencies/widths, not accepted); twootherspending. CanonicalHDF5/plot saving,20minbound,sourcechecks. Allcandidatefitsremainunpromoted. No longrunrestartuntilsignalvalidatedall4classes. Askpendingwhetheruser'sworkingbaselineplotsareNov17startuporfreshtoday.
Codechangesonlylongitudinalscript andT1Zeno loopclass thisturn: normalqspec+Rabi refresh beforeeachmap; qspec centerinside�1MHz/FWHM.05-.8/R2>=.3,pi.6-.7/RabiR2>=.75/referenceSNR>=5; allrawcalibrationssavedQSpec_cal/Rabi_cal inexistingoptimizationHDF5hierarchywithstagetag. Carryacceptedfreq/pi tonextcycle. Post-mapreferenceSNR>=5 checkedonlyAFTERrawsave. Full50gain grid/order preserved, noexistingpulseprogramrewrite. Statecal10k->20k. LocalT1 .01-100us/30/600/4000usreset ->.01-40us/161/2400/100usreset; postwait5->.5us(defaultwithoutoptionalkeyunchanged), magnitude->signed. OtherthreeZenomodulesunchangedfromtaskstart. Sharedexpt/systemhashesmatchbackups.
Offline149fileharness passed; replaycheck_repair.py acceptsactualearlierbaseline(RabiR2.906),rejectsactualnoisyspectrum/ref,checksmetadataandT1lowlevelASTunchangedexceptoptionalfinalwait. Initial6sigmareferencegatewasoverstrictforaccepted2000shotbaselineSNR5.32; chosen5sigmaqualitygatepassesacceptedfixtureandstillrejectsnoiseSNR.65. No freshhardwarevalidation of repairedlongrun yet.

November18 final diagnostic checkpoint10:43: search3/3 HDF5 and normalplots complete; rawIQ5413.247946-5463.247946MHz has no convincingline at3readouttones. Candidate-scoremax4.06/3.72/3.47 overmanywidths/positions rejected. shortro1118/10-37-07 complete36SS20kpoints with.5/1/2/7us:bestcontrast.0152/SNR2.77, no recovery. All owned acquisitions stopped; bothYokos0A/OFF/.01A/30V andQICKstop_tproc final. No long-run restart queued; repairedsource NOT hardwarevalidated. Pending userworking-baselinepath/instrumentchanges; do notclaimfixcomplete orcarrierdriftmeasured.
Maincalibrationlabels corrected BEFOREANYrepaired-mainrun fromdraftQSpec_cal/Rabi_cal toSTANDARDQSpec/Rabi withstagefilenamesuffix, preservingexistingoptimizationHDF5labels/loaders. Protected3Zenomodules/sharedconfigsbyteidenticaltotaskstart. Finalreport/snapshots/taskdiff inIDE/zeno_fix_1118/RESULTS.txt. Broadernewdiagnosticlabelsremainpreserveddiagnosticonly, notmainruntimeformat. Hardwarefree149fileparser/replaypass; finalchecksandartifactarchivepending.

November18 10:49:30 LIVE repaired-helper validation complete: guard1118/2026-11-18_10-45-53 saved1standardQSpecHDF5/2normalPNGs, rejectednoisebeforeRabi/Zeno withcenter5437.272303/FWHM.022800MHz/R2.044116, statusrejected_calibration. Actualmainhelper execution/datawrite/qualitystop verified; freshQ2signalrecoveryNOTverified. QICKstopped, bothYokosCURR0A/OFF/.01A/30V, ownedguard45740exited. No longrunqueued. Full183diagnosticHDF5 plus1liveguardHDF5 preserved withnormalplots. Alloriginal28HDF5 retained. Final149fileharness/replay/whitespacechecks pass. Review in guardrun/documentation/repair1118/RESULTS.txt; requeststillneedsvalidfreshworkingbaseline/instrument-sideinformation beforeclaiminghardwarefix orresuming72h.

## 2026-11-18 afternoon - Q2 exact singles comparison; startup noise-fit guard

User requested pre/post commit+push, working singles, actual Zeno2D validation,
then conditional72h restart. Checkpoint3637e98 committed before hardware.
Auto-review rejected push to origin https://github.com/CosmiQuantum/MaLab_fridge_QZE_cooldown.git
(branch harness_engineering_experiment) because destination/payload trust was
unconfirmed; explicit destination approval requested. No push workaround used.
Unrelated preexisting changes/generated data stayed outside the commit.

Q2=M2/Pythonindex1, physicalID unassigned, BOB RFSoC4x2. Repeated exact current
startup statements once (AST truncated before Zeno), then separately held exact
Nov17 accepted carrier5438.247946MHz/pi.6715803/sigma.101us and
RO8950.000077MHz/.192DAC/7us. Shared expt/system byte-identical to Nov17 snapshots.
Fresh startup qspec/Rabi R2 .014/.024 versus prior .476/.906. Existing script
accepted repeated noise-fit pi .179 and proceeded through coherence before its
late pi check. Fixed settings did not recover qspec/Rabi/T1/Ramsey/echo. Raw
complex Rabi template amplitude .2657->.0205 with residual noise .0304->.0308;
loss is not solely population normalization. Prior singles really had signal.

Bounded lower RO gains.024/.048/.096/.144/.192 at5frequencies/20kstate pairs
plus5Res sweeps gave maximumreferenceSNR3.969, no accepted pulse. Reapplying
configured qubitDAC Nyquist2 while output idle gave fixedRabiR2 .108before/
.020after, no recovery. Existing normalqspec program, short constant pulse
.6715803DAC/.24165018us derived from prior Gaussian pi area, searched
4938.247946-5938.247946MHz/1001pts2000reps and repeated5423.247946-5453.247946MHz/
301pts6000reps. Weak broad candidate near5434-5436MHz did not support normal
.02DAC/100us qspec and GaussianRabi at5433.700MHz (RabiR2 .02088). No new qubit
frequency, linewidth/lifetime, sigma, readout optimization, or TLS result accepted.

Only new experiment-source change:25 added lines in longitudinal script.
Startup reference count2000->20000shots/state. AFTER raw QSpec/Rabi save,
require referenceSNR>=5 and fitR2>=.3/.75; enforce existing pi.6-.7DAC window
immediately after both startupRabi saves. Pulse programs/shared calibration/
powers/phaseorder/50gain grid/HDF5format unchanged. Offline149file harness and
saved-data accept/reject replay pass. ACTUAL modified main launched once with
STOP before baseline could finish, rejected qspecSNR1.217 after rawsave. This
validates early rejection only, not recovered measurement control or 2Dmaps.

Seven diagnostic runs retained52HDF5/60PNG, normal canonical labels and layout,
longest raw/PNGpath212chars. Main runs q2_2026-11-18_14-14-48 and guarded
q2_2026-11-18_14-51-02; other substudy labels ro_power1118,fixed1118,
pi_search1118,pi_repeat1118,candidate1118 under October7/PUCQ4/qze_005.
All owned sessions exited, tProc stopped, clocks locked, both sources read
CURR0A/OFF/.01Arange/30V finally; no source writes or attenuation changes.
Constructor attenuation10/15RO,5/4qubit,17ADC dB is metadata: setters commented,
remote RF readbacks unavailable. Fridge/RF status clarification requested.
No new full Zeno map or72h run started. Need fresh valid singles, exact-setting
zero/highgain2D then full50gain validation before restart; existing authorization
persists. Workspace IDE codex/zeno_compare_1118; RESULTS.txt/receipts/plots and
source snapshots archived under guarded main run/review1118. Existing raw saved.
Legacy log/state files contain non-UTF8 bytes; appended/prepended ASCII only,
preserving existing bytes instead of converting historical user documents.
## November 19 - user-authorized cross-qubit trace recovery (in progress)

Task: recover fresh curves on any PUCQ4 qubit using the validated benchmark
PowerPoint as a reference; verify actual spectroscopy and coherent control.
Authority: user explicitly requests fresh trace recovery. No nonzero Yoko bias,
RF attenuator changes, shared-config promotion, or long Zeno restart in scope.
Preserve prior raw data/user changes. M1..M6 are rows/display Q1..Q6, indices0..5;
only M5=physicalQ4/Yoko3 and M6=physicalQ6/Yoko4 assigned.
Reviewed actual ROOT/PUCQ4_benchmark_20261029_final.pptx and manifest plus
Nov4 low-power evidence. Reuse existing round_robin_benchmark.py, separate
one-row/one-pass stages, canonical phase-checkpoint Data_H5 and normal plots.
Workspace IDE codex/recover1119 stores RUN_BRIEF, source snapshots, exact env
plans, source readbacks and watchdog receipts; output run subtree is
M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/
find_resonators_sandbox/recover1119/<timestamp>/. Planned max path203 chars.
RO .10DAC for M1/3/4/5/6 (qualified conservative cap .10); M2 .192DAC/7us
under independently qualified cap.24. Do not restore October .90/.95 gains.
Lengths8/7/13/5/6/45us, all reference carriers/sigmas/pi from documented
settings; M2 exact Nov17 carrier5438.247946163877MHz/pi.671580296649.
Res +/-3MHz/.05MHz/121points/2000reps; qspec +/-5MHz/.05MHz/201/4000x1,
existing row drive gains/pulse lengths. Readout follows measured dip preserving
known offset;10k g/e shots/state. Actual attenuation unavailable: constructor
10/15 RO,5/4 qubit,17 ADC dB are metadata; setters remain commented.
BothYokos read/require CURR0A/OFF/.01A/30V before/after every stage.
Abort on hardware/save failure, unqualified power, unresolved resonator before
qspec;1800s stage budget/600s log watchdog. All raw fits remain diagnostic
until repeat/fine qspec and real Rabi. SDK3.9.13 obtained from PyCharm;
149file hardware-free verifier passed before launch. First offline PPT-text
print hit console cp1252; extraction saved, UTF8 console used subsequently.
Early M3 res8974.268017MHz SNR21.8 and real narrow qubit response near
5551.341583MHz (raw vectorSNR9.59,g/e meanSNR19.55) recovered; initial.05MHz
sampling insufficient for linewidth acceptance, fine scan required. M5 res
9014.155328MHz SNR10.8 and qubit candidate7064.915555MHz (SNR6.21,g/e16.54),
not yet repeat-confirmed. Remaining screen ongoing. Queued serialized M3
fine qspec then gated Rabi/SSF/T1/T2R/T2E; full envelope in RUN_BRIEF.
No success claim for coherent control, final optimization, or Q2Zeno yet.
November19 10:29 fresh singles recovery update (not a final all-six benchmark):
All six low-power resonator traces persisted. Coarse real lines M1~5515.332,
M3~5551.342,M5~7064.916MHz; M4 weak candidate,M6 unresolved at .10DAC.
Q2 also returned with old carrier/pi for its initial references: SNR17.01,
so return cannot be attributed solely to our later frequency retune. Cause of
Nov18 loss versus Nov19 recovery remains unidentified; no RF attenuator,
nonzero bias, clock reset, or low-level pulse-code change was made today.
Fine Q2 spectrum5438.770200MHz,FWHM.17115MHz,rawSNR5.20,R2.503,mean-refSNR24.82.
Rabi at that carrier sigma.101us gives pi.671580296649DAC,R2.9748,refSNR22.97.
Fresh20kSSF lab contrast.09995; normal T1/T2R/T2E conditional fits
4.3745+/-.3194 /2.7324+/-.3718 /8.8530+/-1.1102us (R2.841/.470/.510).
Independent fits agree; all fail stricter R2>.85 final-lifetime acceptance.
Visible early decay/fringes are real recovery evidence; not high-quality
final lifetime calibration or reproduced two-frequency/TLS beating proof.
Q2 coherence run recover1119/2026-11-19_10-25-01; all plots and HDF5 saved.

M3 fine repeats center5551.348420MHz,FWHM.094389MHz,rawpointSNR6.41,
matched19.40,R2.674,ref24.24. Initial conservative pointSNR8 gate STOPPED.
Original source/log/rejection preserved. Recorded scientific review used
repeat agreement<20kHz,resolved~19points/FWHM,matched>=15,point>=5,
R2>=.6,ref>=5 solely to permit an independent diagnostic Rabi (no promotion).
Rabi sigma.124us,pi.651681621193DAC,R2.9367,ref16.66 confirms coherent control.
At .10DAC/13us/8973.488017MHz,SSF labcontrast.07485; T1/T2R/T2E
24.0468+/-2.4588 /28.0779+/-2.9114 /34.8048+/-3.8424us,
R2.705/.615/.474. Independent fits agree but do not pass final R2>.85.
Population normalization overshoots1 in Rabi; signed raw data retained.
M3 coherence run recover1119/2026-11-19_10-11-11. BothYokos0A/OFF before/after
all normal stages. Shared expt/system configs unchanged; new carriers/pi only
explicit per-run values. No claimed reoptimization or final calibration.

18-slide PUCQ4_recovery_20261119.pptx saved in run root; zipCRC,shape bounds,
raw/source hashes checked; native Office render not performed. All24 canonical
normal HDF5 and normal plots preserved; recovery_review under M3 coherence
documentation holds overview,manifest,source snapshots,launchers and summary.

Original Zeno-debug task continued with fresh Q2 pulse: bounded3-gain diagnostic
zeno_once.py/launch_zeno_once.py, exact envelope in IDE/recover1119/ZENO_BRIEF.txt.
Output qze_005/map1119/2026-11-19_10-35-37. Same existing five map classes/calls,
map order highgainQspec.02,normalQspec.009,T1,Ramsey,Echo. Zeno1e-6.. .04DAC
three gains (50->3 diagnostic only), same offset RO tone8950.000077MHz,5us
ringup/.5us final wait,main delay/count/reset profiles,20krefs. Holds fresh
RR pulse during this bounded diagnostic rather than main per-map qspec/Rabi
refresh.45min total/600s progress watchdog; STOP_2D after saved map; parent
stops owned tProc and reads clock/Yokos after child exit. No full50/72h claim.
First highgainQspec raw3x251 saved10:41:05,refSNR25.32,all3 visible lines:
centers5438.775787/5438.706631/5438.554961MHz; FWHM.2475/.4412/.7507MHz,
R2.598/.638/.558. Normal Qspec and three coherence maps remain in progress.
November19 bounded Zeno check completed10:57:18,child clean0,parent stopped
owned tProc,clockslocked,bothYokos0A/OFF/.01A/30V verified10:57:24.
All5 maps raw/plots retained under map1119/2026-11-19_10-35-37.
Mean-refSNR highQspec25.32,normal21.77,T1 18.56,Ramsey23.91,Echo20.83.
Zero-gain map fits: T1~5.1746+/-.4576us,R2.785;T2R3.1884+/-.4497us,
R2.452;T2E6.9188+/-1.2497us,R2.335. Highgain Ramsey/echo have weaker,
faster loss of fringes and do not provide accepted lifetime fits. Normal
Qspec zero-gain center5438.772803MHz/FWHM.14398;middle/high centers
5438.715012/5438.526441MHz with broader weak-per-point features. No new
microscopic TLS/beating claim. Full50 grid not validated by3gain tests.

Q2 working seed promotion in expt_config.py/system_config.py only:
res8950.93439->8951.01439MHz;qubit5438.499681577915->5438.770199633722MHz;
pi.6765549655130318->.6715802966489653DAC. Derived RO8949.920077->8950.000077
MHz; offset-1.014313MHz,gain.192,length7us,sigma.101us,ceiling.24 unchanged.
Numbers come from repeated normal RR and independently confirmed Rabi,not
unvalidated noise candidates. Lifetimes/readout are not promoted as final
benchmark optimization. All other rows and pre-existing user changes remain.
Safe149-file verifier passed after edits; focused5 numeric substitutions
reviewed. Saved-data operational gates replay passed on all5 actual3gain maps,
including zero-gain coherence; native legacy50gain HDF5 parser replay passed.
These offline checks did not operate hardware.

Prepared native continuation: full50 one-cycle validation then conditionally
native72h, exact envelope in recover1119/NATIVE_BRIEF.txt and pipeline_plan.
Only validation's outer repeat_round range is changed in compiled AST; actual
script and all50gain/native phase/call/data hierarchy remain unchanged. Last
accepted full-cycle carrier/pi seeds are runtime overrides for72h startup.
Working configs are not automatically rewritten later. Whole-cycle metadata,
50gain shape,mean-refSNR>=5 and zero-control gates must pass before72h child.
These are operational nonblank-data gates,not final metrology acceptance.
Longest intended round999 artifact218chars.14h/96h stage deadlines and600s
log watchdog,owned-child cleanup,local atomic state. Await launch below.
November19 native continuation launched11:12:57. Local checkpoint4418de6
before hardware start. Parent26112/child19052; actual native output
qze_005/q2_2026-11-19_11-12-57. Startup qspec and both Rabi guards passed;
cycle0 started11:19:47,first per-map normal qspec/Rabi refresh now running.
Fresh native res tracking choseRO8950.080077MHz;startup carrier5438.777724MHz
andpi.661631DAC (sigma.101us). These are runtime native calibrations,not
unrecorded source edits; working source seeds remain the reviewed values.
Full50 maps pending;72h not started. Parent will stop on any native/quality/
save/source-hash error,and only launch72h after all5 full maps pass. Native
maxcycles1 is compiled outer-loop limit only; original source unchanged.

Final26slide report saved PUCQ4_recovery_20261119_with_zeno.pptx in run root,
SHA2561e1e38adccf8a242a97a0f56a3f719daaec6eecfe611e50e7d988d3499deb284.
Contains reviewed normalQ2/M3 curves,all-six screens,conditional fit flags,
five bounded native maps and raw curves. Corrected earlier coarse-screen
subtitle to reflect subsequent Q2 as well as M3 confirmation. Earlier deck
preserved. ZIPCRC,shape bounds,images,raw hashes verified; no Office render.
Q2/Zeno report sources/launchers/promoted config snapshots/calibration-change
JSON under map1119/2026-11-19_10-35-37/documentation/recovery_review.
Native pipeline status remains local recover1119/pipeline_status.json to avoid
M-sidecar stalls. User-facing latest paths in RECOVERY_RESULTS_with_zeno.txt.
Git push remains blocked by prior automatic approval rejection of unconfirmed
remote export destination; no retry/workaround or remote write attempted.

Final27slide report status annotation, SHA256
31df0753e81b40a6a4b2b4bda791d63f0d8d0e2b9602ce7068ecd54f50564055.
Prior26slide version preserved as deck_before_status_update.pptx. Corrected
pre-promotion caption and added old/new/unit table plus native validation
status. Final audit verified24normal+5bounded HDF5,all canonical metadata/plots,
finite averaged arrays,27slide package/images/bounds and live source hashes.
Historical state bytes preserved exactly;149file offline verifier passed.

Native actual first full50 map entered:25,100,000 readouts expected
(50*251*2000),progress increasing,no controller stderr. Full raw map files
not complete;72h remains queued,not started. Added native_validation_plan.json
and NATIVE_BRIEF under native qubit_1round0/documentation;local
full_validation_started.json records exact folder and owned PIDs.

Auxiliary metadata-only PowerShell command spun CPU with no output/local
marker before Git; identity-verified helperPID65316 terminated and replaced
by standard-library JSON writer. No lab process killed/restarted; native
parent26112/child19052 and atomic heartbeat continued uninterrupted.
Nested handoff metadata is generated with Python JSON; no hardware conclusion
follows from this helper failure. No experiment/source changes in that helper.

## November20 Q2 SSF recovery and native Zeno crash repair (in progress)
User authorized readout recovery/reoptimization and full native restart after fresh validation. Prior Nov19 native full50 cycle passed; subsequent72h child failed during normal QSpec_zeno fit: scipy maxfev1000 then ValueError seven None values unpacked into five. Failure is optional gain-averaged fit handling, not an acquisition connection error. Raw prior files retained; failed map did not reach caller raw save.
Minimal fix section_004_qubit_spec_ge_zeno.py returns correct five-value failure, still plots map without fit, and preserves return_fwhm arity. Main logging accepts absent optional fit center; main explicitly sets res_length_ge alongside res_length because build_state derives actual length from the former. AST-isolated regression injects nonconvergence and verifies both return signatures and PNG without hardware imports.149file offline verifier passed before acquisition; final checks pending.
Historical Oct29 deck Q2/M2 row1 actual readout8949.730MHz/.95DAC/7us, histogram contrast48.87% finalpass and44.47/45.25% independentexamples. Those are not60% contrast. Balanced assignmentaccuracy roughly(1+contrast)/2; heldout accuracy to be measured separately. November4 subsequent punch-out qualification lowered cap.24 and selected.192DAC/7us/offset-1.014313MHz, yielding~.1contrast. Do not silently replay.95 above the qualified cap.
Workspace IDE codex/ro1120: RUN_BRIEF.txt records envelope; source snapshots retained. Active bounded normalRR qspec/rabi/SSF parent39272 followed by saved-data review parent73620 and conditional bounded alternating SS length/gain/frequency optimization. No72h restarted yet. Data normalRR find_resonators_sandbox/ro1120, optimization qze_005/ro1120. Gain<=.24,8948.5..8952MHz,.5..15us; bothYokos readback0A/OFF required; no current/attenuation changes. Latest status/log/raw must be inspected before assuming success. Preserve historical old IDE Python processes; no recent competing acquisition found.

November20 selected working readout after126 raw SS points/126plots, all finite/canonical:
RO8950.325077MHz,gain.24,length7us,reference resonance8950.96439MHz,offset-.639313MHz.
Independent candidate contrasts.08770/.08305/.09160; heldoutaccuracy.54445/.54135/.54665.
Baselineaccuracy.53025/.55030/.53440; analogdprime meanratio1.14046,timevarying.
High-power historical fidelity NOT recovered within cap; no reducedaverages accepted.
Fresh normalRR carrier5438.856540689225MHz/pi.6765549655130318/sigma.101us; qspecR2.48989, RabiR2.96797,refsSNR21.2/22.7. Final resdepthSNR40.37 at.24.
Promoted ONLYQ2/M2/index1:res8951.01439->8950.96439MHz;offset-1.014313->-.639313MHz;
gain.192->.24DAC;carrier5438.770199634->5438.856540689MHz;pi.671580296649->.676554965513DAC.
Length7us/sigma.101us/otherrows/gaincap.24 unchanged. expt/system/main consistent.
149fileharness/AST-isolatednonconvergence/126HDF5+plots/unchangedotherrows/diffchecks passed.
Native72h restart launched15:33:25: qze_005/q2_2026-11-20_15-33-25, controller60196 child65592.
Startupcalibration acquiring; loop/data acceptance still pending. Supervisor ro1120/restart_status.json;
600s progressguard/96hwallbound/bothYokos0AOFF required. No72hcompletion claim.

November20 15:40:13 native restart entered timed Zeno cycle0 after full startup.
TenHDF5 saved: nineactual measurements plus native unusedsecondRamseycorrection
placeholder. Startupres/qspec/Rabi/SSF/T1/Ramsey/echo raw/plots verified; refsSNR
~22-24 for reviewed startupqspec/Rabi/T1. NativeRabiR2.920/.939; SSFcontrast.10270.
Actual nativeRO8950.365077MHz/.24DAC/7us after freshres tracking, same-.639313offset.
Currentpermapqspec/Rabirefresh acquiring before first50gainmap.72h targetNov23
15:40:13local then finishactivecycle; nofirst2Dsave/full72hcompletionclaim yet.
Firstaudit native_verification.json distinguishes unusedplaceholder fromactualdata.

## 2026-11-20 - Persistent measurement workflow

- Request: persist the user's measurement sequence and existing class, plotting,
  and saving conventions for future repository sessions; offline documentation only.
- Added mandatory MEASUREMENT_PLAYBOOK.md and docs/MEASUREMENTS.md navigation,
  routed from AGENTS.md and the harness index and required by the manifest.
- Gates: qspec <=0.5 MHz (tighter PUCQ4 target retained), measured punch-out
  readout ceiling, SSF >50%, eight points/fringe replacing four, and 5-10*T1
  relaxation for Rabi/SSF/coherence. Added averaging optimization, bias-indexed
  calibration, propagation, recovery and per-result documentation requirements.
- Acceptance: safe harness verifier and focused diff review; no hardware run,
  config changes, data writes, or claims of new validated calibration.
- Maintenance is performed by future agents during tasks, not a background job.
- Verification result: offline harness passed (149 Python files parsed), required documents passed, and git diff --check passed. Focused instruction diff reviewed; hardware not run.

November20 live2D verified after restart: ZENO_CALIBRATION qspec_hi
5438.845378047941MHz/pi.6765549655130318DAC passed, then full50gain highdrive
map actively advanced227682/25100000readouts (~48s elapsed,~88minremaining).
Controller60196/child65592 statusrunning; firstcomplete2DHDF5 stillpending.
NativeactualRO8950.365077MHz/.24DAC/7us. Primaryrepair/configcheckpointbb2222a.

## 2026-11-20: user stops low-SSF Zeno run; recovery gate tightened

User explicitly requested stop and SSF >50% in existing histogram contrast before restart. Stopped owned native child65592; supervisor60196 exited with expected nonzero child return, stopped tProc, clocks locked, both Yokos CURR0A/OFF/.01A/30V. This was deliberate interruption, not a new hardware fault. Current 2D map was not persisted; previous HDF5 preserved.
Read-only audit confirmed saved startup SSF pi_amp .666605627784899 equals preceding corrected Rabi result, carrier5438.859978368855MHz, sigma.101us, full4sigma envelope.404us, reset100us. RabiR2 .9394. No missing/half-pi/stale amplitude established. Averaged2000-shot signal is not single-shot validation. Low gain remains hypothesis, not proven sole cause.
Recovery controller IDE codex/ssf1120b/campaign.py PID62012 uses existing RR and canonical find_resonators_sandbox/ssf1120b hierarchy. Exact envelope RUN_BRIEF.txt. Sigma .101/.097/.08 comparisons and interleaved resonator power qualification; raw and plots checked. No automatic Zeno restart. Historical .24 limit was conservative, .30 stable/.60 movement; fresh qualification required before any upward promotion. Acceptance strictly contrast>0.50 independently repeated, not (1+contrast)/2>0.50. Shared configs unchanged pending evidence. Layer: verification/context. Next: review pulse scans and power spectra, then alternate readout length and gain/offset optimization under newly verified ceiling.

## 2026-11-20 SSF recovery completed unsuccessfully; Zeno remains STOPPED

Fresh RR sigma.101/.097/.080us gives pi.671580/.696454/.835744, RabiR2 .977/.978/.981 but contrast7.51/9.22/10.73%. Shortest diagnostic exceeds preferred pi-gain range; none recovers target. Carrier5438.856434632MHz; qspecFWHM.148956MHz. FreshT1 4.066998+/-.209365us,R2.941745, with sigma.097/full.388us and100usreset. October29 deck effectiveT1 5.35us, so lifetime also shorter; no proven sole cause.
Power .30 at7us qualified through repeated fitted spectra and low returns; .40 rejected (30.77kHz versus30kHz limit). SSF101points/101normalplots/101HDF5, twoalternatingcycles plus5pairedconfirms:2us/.30/8950.515077MHz candidate9.35-10.61%,mean10.00%; baseline7us/.24/8950.365077MHz7.43-11.29%,mean9.07%. This is NOT >50%contrast. Candidate not promoted; same-length finalpowerqualification also still required before future promotion. Sharedcalibration/averaging unchanged. FinaltProcstopped,clockslocked,bothYokos0A/OFF.
Minimal code: RR explicit boundedres-only characterization mode preserves normal persisted ceiling; native saves SSF then aborts<=.50/nonfinite.8guardcases+5SSFgatecases passed offline. Calibrationcontroller initially hitpersisted.24 guard beforeacquisition; newmode was required for user-authorized spectroscopyonly qualification, not ordinaryreadout. No lowlevelQICKchanges.
Canonicalreadout qze_005/ssro1120b/2026-11-20_16-27-12; pulse/power/T1 find_resonators_sandbox/ssf1120b. Report in readoutrun documentation/recovery_review/RESULTS.txt. IDEworkspace ssf1120b holds exactplans/reviews/scripts. No automaticrestart; require reproducibleexistingSSFmetric>.50 first. RFboard attenuatorsetter lines arecommented inQICK_experiment; constructorarguments must notbe represented as instrumentreadbacks. Nextdiagnosis measurementchainSNR and prep/readoutseparation withinverifiedhardwarelimits.

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

## 2026-11-20 SSF improvement campaign ACTIVE
User requests higher SSF on all six and authorizes higher resonator gain.
Work controllers: IDE codex/ssup1120. Current phase power.py (power2 PID46592),
canonical find_resonators_sandbox/ssup1120; monitor power_status.json/logs.
Sequential higher-gain characterization .1.. .95 with stop at reproducible
>30kHz dip shift; all Yokos zero/OFF, RF chain unchanged. No Zeno restart.
Current permission retains prior below-shift rule; optional async question asks
whether user also wants above-shift tests. Do not infer approval from no answer.
Original5000rep/201point scans retained; remaining2000rep/101points/.04MHz,
fitR2>.95,SNR>=10, independent returns and confirmation. Planned fallback.py
reviews lower ceiling, then prep.py rechecks qspec/Rabi; optimize.py prepared
for gain/frequency grids at qualified original length. None of these follow-up
controllers launched yet. Updated run envelope in ssup1120/RUN_BRIEF.txt.
All sources/configs unchanged. FirstQ1 .4 shift repeated~60kHz; .3 failed
return-reference comparison, .2 awaits independent fallback confirmation.

## SSF improvement campaign continuation, Nov20 22:47
Initial power controller46592 completed and cleaned up. Sequential pipeline68776
is now prep.py; its first phase pairs.py repeats M1/M4 at.1/.2/.1/.2,6000reps
without relaxing30kHz maximumdifference. Current files IDE codex/ssup1120:
power_review.json, paired_m*.json, pipeline_status.json. No Zeno restart.
Power operating candidates before paired update: M1.1,M2.4,M3.2,M4.1,M5.1,M6.1.
M4/M5 initial low reference noisy;6000rep repeats resolved them. All rejected
higher scans retained. Boundary flags describe conservative tolerance, not exact
physical punch-out thresholds. No above-shift permission received as of this entry.
Pipeline next: prep.py qspec+Rabi+SSF allrows, then optimize.py eachrow. Optimize
uses existing SingleShot and canonical Data_H5, fixed-length gain/offset grids,
then2/4/8us candidates with independent same-length gain qualifications, final
finegrid and3paired20k-shot confirmations. Row gain enforced perqualifiedlength.
Originallength results from power_review; candidatepulse updates remainlocal.
Output readout folder find_resonators_sandbox/ssopt1120; power/prep inssup1120.
No configs or low-level pulseprograms modified. Local script syntax allpassed;
latest optimize.py import explicitly resolves local screen before legacyhelpers.
Longest intendedopt artifact204characters. Currentrunbrief copiedeachRRstage.
Continue monitoring pipeline; do not launch competing acquisition. Pipeline has
STOP between phases; perRR watchdog600s/1800s; directSSF Pyrotimeout60s,onehour
perrow. Finalhardwarecleanup insideeachstage. A new chat must inspectstatus,
not restart scripts or overwrite labels. Remaining: reviewallraw/plots, improve
unresolvedrows as warranted, independentfinalRR evidence, report/promotiononly
for validated settings; persist failure/success and safelyleaveZeno stopped.

## SSF campaign status Nov20 23:17
Pipeline68776 continues prep.py. Power work is done. Tightly paired6000rep
M1/M4 checks still failed strict30kHz maximumrange (reference drift); original
length ceilings remain M1.1,M2.4,M3.2,M4.1,M5.1,M6.1, allDAC.
Fresh preparation: M1freq5515.3455257,sigma.05,pi.6865043,RabiR2.98698,
SSF.14685; M3freq5551.3484616,sigma.124,pi.661631,R2.98756,SSF.153304;
M5freq7065.0944599,sigma.056940825,pi.63178295,R2.96778,SSF.0985.
M4 feature5594.644875MHz rawSNR3.50/R2.147 notaccepted; targetednarrow
confirmation andRabi queued viarefine_prep.py atstartofitsoptimize.py stage.
M6 hasNOresolvedqspec: rawSNR1.56,R2.060,referenceSNR1.84. Native broadfit
6404.699MHz isnoise and NOT promoted. Retain6404.639622 seedforreadoutsearch.
Usertold M6needs readoutimprovement thenqspec retry; ensurethatretry happens
beforeclosingtask (pipelinecurrentlydoesnotautomaticallyretryafterreadout).
Q2prep remaining. Allnativeqspec/Rabi plots reviewed forM1/M3/M5; records
visual_review.json. No missing/pi2pulse indication; poorSSF persistswithpi.
Pendingoptimize.py uses ownqualifiedlengthcaps,2/4/8us+original,loops gain
andfrequency thenpairedconfirms. Localcontrollers syntaxpassed. Newoffline
review_results.py andbuild_deck.py arepreparedbutnotrun; reviewrequiresall6
completedreadoutstates; rememberindependentfinalRR whereappropriate andQ6retry.
No sharedconfigchanges,nolongrunrestart,noabove-shiftpermissionreceived.

Nov20 23:35 continuation: initialpipeline68776 onM3; M1candidate14.780% vsbaseline14.712%, no meaningfulimprovement. Finegain queue26584 waits for initialpipeline; .05DAC increments require new four-scan same-length qualification. Localfine_gain.py/fine_pipeline.py parsed withall17controllers. Planned retry.py then waits forfinequeue, retriesunresolvedQ2/M6/M4 usingboundedprior+/-1 then5MHz and independentnarrowconfirmation/Rabi/SSF. No above-shiftpermission, noZeno restart, nocoreconfigchanges. SeeIDE/ssup1120/RUN_BRIEF.txt andstatusJSONs; do notstartcompetinghardware.

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

Nov20 offline audit correction (verification/context): directSS canonical save_h5 stores Syst Config as JSON, whereas existing RR records inspected here use Python repr. Audit initially assumed single-quoted repr and failed before report creation; replaced directSS parsing with json.loads. M1 complete integrity audit passes28HDF5/28plots, finite IQ/counts, exact gain/length/frequency and fidelity metadata. No acquisition data changed. 149file offline harness passed again; allfive protected source hashes unchanged fromtask-start snapshots. Reportqueue48992 waits forretry completion; do not claim reports already exist.

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

On Nov 24, a transient Windows reader lock on zb1124a/power_status.json interrupted the controller after the M3 ceiling confirmation was already saved. Both sources were verified zero/off, clocks locked. Recovery uses tools.ssf_progress.publish_status with bounded retries and treats a permanently missing required record as an error. Reuse the complete per-scan result and receipt when resuming. The resumed controller reconstructed M1 and M3 from disk before advancing to M4.

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

November 24 zero-bias campaign closed after six independent preparation checks. In row order M1-M6 the optimization means were 31.34%, 13.08%, 24.78%, 18.15%, 11.63%, 7.05%. Fresh checks were 31.57%, 12.00%, 23.39%, 16.00%, 12.08%, 5.28%. All six pi gains were within 0.6-0.7 DAC. M2 peak population and M6 linewidth missed preferred device targets. No row qualified for a final one-pass T1/T2R/T2E benchmark or shared setting promotion. Both sources verified at zero/OFF, QICK stopped and clocks locked. See the November 24 closeout table in docs/MEASUREMENTS.md and dated 19-slide PUCQ4_zero_bias_20261124.pptx under pucq4_first_light. Next step is a physical readout-chain check against the last documented working setup, followed by the M1 reference.

November 25. Prepared round_robin_ramsey_drift.py for one all-row calibration batch and repeated fixed-carrier Ramsey batches until 06:00 local. Hardware launch pending the November 24 readout-chain follow-up. No new calibration or drift measurements.

November 25 overnight launch envelope: round_robin_ramsey_drift.py, BOB/PUCQ4 M1-M6 indices 0-5, Y3/Y4 zero/off. Authorized diagnostic contrast. November 24 evidence.json fresh_check.settings row-specific gains/lengths/tones/sigma. Nominal attenuation arguments 10/15, 5/4, ADC17 dB. Startup res_spec/q_spec/rabi once per row (persistent zero-current sweep windows around fresh seeds, Rabi to 0.85 DAC); then t2r only with 4800/9600/4800/7200/9600/16000 reps, 10 samples per programmed fringe minimum, normal per-row windows and delays. End after last cycle begun before 06:00. Abort on readback/acquisition errors or incomplete startup, preserve data. Canonical Oct_7/PUCQ4/find_resonators_sandbox/ramsey_drift output. Review initial frequency resolution.

21:48:10 started acquisition controller PID 37496; frequency-analysis PID 45632. Live zero-current readbacks and M1 resonator acquisition verified. End-of-run and first Ramsey quality must be checked against the saved batches. Original startup logs in TEMP/ramseydrift_20261125_214807.out and .err.
