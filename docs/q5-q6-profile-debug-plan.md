# Q5/Q6 profile-sensitivity debugging plan — September 15

## Objective and scope

Resolve the branch-visibility bottleneck with a falsifiable diagnostic, then
acquire a bracketed paired-bias map if a prospectively validated protocol exists.
Physical Q5 identity remains a separate device-mapping requirement. This work
does not assume M2 is Q5 or invent an independently wired drive.

The user requested a new debugging plan and its execution after the prior bounded
investigation stopped. Existing acquisition stop rules and device limits remain
in force. Start with saved raw data. Preserve all prior fits and their rejection
labels, every raw artifact, shared calibration, QICK programs and user changes.

## Hypotheses and decisions

1. **Detector sensitivity:** the existing `snr` is fitted peak amplitude divided
   by point noise, not uncertainty of a fitted line amplitude. Test whether a
   whole-profile statistic reproducibly recovers the broad upper feature.
2. **Correlated noise or drift:** test independent ON and OFF partitions, rather
   than reusing the full OFF average for every ON block. Keep frequency-correlated
   noise within each block; inspect chronological block amplitudes and lag
   correlation. A reproducible offset between sequential OFF/ON acquisitions
   can mimic a line; retrospective statistics alone cannot exclude that cause.
3. **Readout suppression:** only if the profile test fails, inspect the saved
   resonator response and readout settings for a specifically testable readout
   hypothesis. Do not launch another identical averaging escalation.
4. **Physical identity:** require chip-specific layout/resonator connectivity or
   a uniquely identifying measurement. Coupling and spatial proximity alone do
   not assign Q5. No undocumented drive port may be enabled.

## Stage A: offline implementation and validation

Use chronological raw repetitions from the complete lower and upper windows in
ac_ctrhi, ac_cmax, ac_m6hi and ac_r6new. Include every selected window, even nulls.
Split each ON and OFF into 64 equal chronological blocks, then use alternating
blocks for discovery and validation. Verify raw normalization separately against
each saved ON/OFF average and verify recombination. No full OFF reference is
shared between discovery and validation.

Select a Lorentzian center, width and signed IQ direction using discovery only,
with a linear IQ baseline. Test that frozen profile against validation block
amplitudes. Report block uncertainty, sign-randomization diagnostics, independent
OFF-only controls and chronological dependence. The sign-randomization reference
requires symmetric independent block errors; it is not an exact physical-null
test for these sequential acquisitions. Correct the eight-window family with
Bonferroni when displaying conditional diagnostic p-values.

Validate implementation on seeded synthetic white noise, frequency-correlated
noise, planted Lorentzian signals and drifting artifacts. Controls must show
that the routine can withhold a conclusion when stationarity fails. The real
data have already been inspected, so all Stage A results remain retrospective
diagnostics. They do not replace the original acceptance gate or establish J.

Acceptance: normalization/recombination pass; deterministic validation succeeds;
all eight windows and controls retained; uncertainty assumptions explicit; no
raw or existing analysis modified; safe harness and focused diff pass.

## Stage B: prospective measurement, conditional on Stage A

If Stage A demonstrates a useful profile method and acceptable controls, freeze
its method and thresholds before new data. Design a short independently repeated
ON/OFF control sequence that separates reproducible drive response from temporal
baseline structure. Record exact bounds, settings, duration, paths, source hashes
and stop rules before acquisition. Use the existing guarded orchestrator and
unchanged spectroscopy program, within the documented current/frequency/gain
limits. Do not launch the old exhausted plans or call retrospective detection
prospective confirmation. Abort on failed positive control, reference structure,
inconsistent repeats, source/acquisition/save failure or deadline.

## Stage C: scientific closure and reporting

### Executed pilot and frozen map envelope

ac_prof2/2026-09-15_11-31-36 completed exit0 in1943.218s; all4 prospective
windows pass and both sources are verified0A/off. Lower centers
5431.778959825/5431.754416540MHz; upper5444.514833207/5444.533176441MHz.
Repeat differences24.543/18.343kHz pass their frozen50/116.880kHz limits.
Full assessment is IDE q5q6_profile_pilot_0915/pilot_assessment.json.

Fresh map is prepared in IDE q5q6_profile_map_0915/map.json, labelac_pmap.
PlanSHA BFFAC67AE4A1E41F81D5196597E8D8F240D4FD21EE4C3DBA9965F092F9D32AB5;
specSHA FA20A8E4A6FB79DE8671039CFA2151D23B925C8B0142D3643B80AC4AFD5340E4.
Readout M2/index1 throughout; sharedDAC1. Seven unique paired biases plus repeated
first bias, in acquisition order below. Bounds derive from a locator model of
the old two accepted detuned pairs and mean fresh pilot centers (15.3kHz locator
RMS). That mixed-power seed model is not an accepted coupling result.

| Visit | Y4 mA | Lower MHz | Upper MHz |
| --- | --- | --- | --- |
| 1 | -5.655 | 5430.982..5432.982 | 5443.406..5446.006 |
| 2 | -5.700 | 5423.542..5425.542 | 5440.287..5442.887 |
| 3 | -5.625 | 5433.792..5435.792 | 5447.636..5450.236 |
| 4 | -5.685 | 5426.360..5428.360 | 5440.989..5443.589 |
| 5 | -5.635 | 5433.052..5435.052 | 5446.030..5448.630 |
| 6 | -5.670 | 5428.876..5430.876 | 5441.992..5444.592 |
| 7 | -5.645 | 5432.122..5434.122 | 5444.613..5447.213 |
| 8 | -5.655 | 5430.982..5432.982 | 5443.406..5446.006 |

Each window101points; lower16000,upper32000reps/state; gain.03,20us pulse,
100us relaxation,one round. Y4ramp.5mA/s,noextra hold,Y3zero/off,Y4zero/offcleanup.
Existing guarded controller and unchanged section004/002 programs; calibrated
M2 readout/tracker, nominalattenuations10/15,5/4,17dB unchanged, settersdisabled.
Expectedabout3hours;14400s+180guard;196char maximum canonical HDF5 path.
All16windows require frozen profile/reference/dependence/fit gates with family16;
failed window requests boundary STOP. First7 unique pairs enter unchanged
coupled_fit only after all windows and the final repeated-center control pass.
No failed-window pooling, automatic retry, or gate adjustment. Stop on source,
acquisition,save,path,deadline or control failure. Safe142-file harness and canonical
IQ persistence check pass; preflight/launch status must be checked separately.

### New design provenance found during the pilot

The earlier missing-source searches omitted the local Dropbox archive. It is
available and materially broader than M:/malab. Exact source:
`C:/Users/Ma Quantum Lab/Dropbox/SQUILL/FUN15-12052025/design codes/Device04V2.py`.
It declares ChipID PUCQ4, design resonator frequencies by physical qubit
Q1..Q6 = [7.501,7.528,7.437,7.562,7.467,7.598] GHz, identity indices [0..5],
and SQUID junctions only at indices3/5. The `output_dxf/PUCQ4.dxf` geometry
visually matches the supplied characterization layout. The submission note is
dated December5,2025 and explicitly lists PUCQ4 among eight submitted designs.

Combining these design assignments with the characterization table's nominal
frequency entries gives intended Q1->M3,Q2->M4,Q3->M1,Q4->M5,Q5->M2,Q6->M6.
The original table explicitly warned that higher measured resonator frequencies
caused indexing uncertainty. Therefore this is source-backed **intended design
mapping**, not permission to silently promote every measured physical identity.
Q4/M5 and Q6/M6 retain their separately demonstrated flux identities. Review
additional packaging/measurement provenance before claiming Q5/M2 experimentally
confirmed. This source removes the absence of chip-specific design evidence.

The source/geometry/deck extraction and SHA256 provenance are saved under
IDE workspace `q5q6_profile_pilot_0915/design_review/design_evidence.json`.
No mask-generation code was executed, and original design/deck files are unchanged.

Original-author cross-check: local Dropbox/People/Shishir/Meeting Slides/
08042026_WeeklyUpdate.pptx slide2 explicitly labels M2 as Q5? and M1 as Q3?,
while slide4 says the nearer coupled qubit was guessed and needs confirmation.
The original two-slide deck's creator is Shishir Sudhaman; it was created
August1,2026 and last modified by Olivia Seidel September3,2026. This confirms
that the intended mapping was known but not experimentally settled. Later
August/September decks in that folder concern the different seven-qubit PUR1D4
device; their dedicated-drive experiments cannot be reused as PUCQ4 wiring.
Extracted slide text is in design_review/meeting_slide_text.json.

A successful fresh paired-visibility protocol permits a predeclared map with at
least seven distinct paired biases and two points either side of the minimum.
Preserve the physical model's identifiability/residual gates and reject a failed
model. Report the new analysis and controls in the reusable PowerPoint with raw
provenance; keep old rejected results visible. Finish with verified source parking,
canonical HDF5/plot audit, native slide validation and updated handoff. Do not
declare Q5 identity, independent routing, J or mechanism solved without evidence.

Method references: SciPy's `signal.correlate` matched-filter example and
`stats.permutation_test` paired-sign null assumptions:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html

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