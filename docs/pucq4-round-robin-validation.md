# PUCQ4 round-robin reproduction, 2026-09-08

Closed with gaps on September 9 at approximately21:51local, within the requested
closeout window. See [the final handoff](pucq4-round-robin-closeout.md) for accepted
coverage, unresolved measurements, report/check links and exact configuration
changes. Both sources are0A/output off; no task acquisition remains scheduled.
The historical progress entries below are retained as provenance.

September 9, 18:49 local closeout update: user requested "finish this up within
the next few hours." Target approximately 22:00 local including report, offline
verification and both sources at zero/output off. The prior all-six pass took
111.7 minutes; completing all remaining negative groups, controls, M1 work and
another all-six pass exceeds that allowance. With no optional preference reply,
the final all-six default benchmark and report took priority. Time pressure does not change quality gates,
promote rejected diagnostics, or turn unmeasured rows into validated results.

User-authorized hardware task: make `round_robin_benchmark.py` reproduce the
six-row zero-current characterization and the nine documented bias points for
each flux-tunable qubit. Adjust configuration and orchestration only; preserve
all underlying experiment classes, QICK programs, drivers and raw measurements.
The starting source hashes and dirty-file snapshots are in
`%TEMP%/pucq4_rr_verify_20260908`.

## Inputs and outputs

Reference: `PUCQ4_chip_summary_20260908.pptx` and its
`PUCQ4_chip_summary_20260908_artifacts/summary_provenance.json`, plus
`yoko_flux_campaign/ramsey_sigma_coherence_manifest.json`, under
`M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light`.
Preserve both existing active decks. Create one new round-robin validation
PowerPoint in that same run folder, with actual new traces, comparison tables,
current-indexed linewidths, all acquisition settings, and reproducible usage.

M1--M6 mean resonator rows/displayed Q1--Q6/Python indices 0--5. Physical Q4 is
M5/index 4/Yoko 3/192.168.1.73/D4. Physical Q6 is M6/index 5/Yoko 4/
192.168.1.77/A5. M4's physical number remains unassigned.

## Acceptance

- One normal round-robin pass at zero with all six rows and res spec, qspec,
  Rabi, SSF, T1, T2R and T2E, followed by each flux row at every documented
  nonzero current with the other Yoko at zero. Fixed-row controls at endpoint
  biases check that the six-row configuration remains usable away from zero.
- Every promoted trace has canonical HDF5, its normal RR plot, finite raw
  arrays, recorded configuration, and recoverable bias state. Population plots
  use measured ground/excited calibrations. No seed or failed fit counts as a
  found line. Resolve each qspec peak with several frequency samples; report
  FWHM in MHz with the associated current and drive gain.
- Rabi pi gain 0.6--0.7, with adequate fitted contrast and no extra unexplained
  oscillations. Coherence fits must be resolved, have finite uncertainty,
  relative uncertainty below 20%, and at least four actual samples per fringe.
  Independent fits/residuals distinguish poor configuration from physical drift.
- Flag coherence changes exceeding 30% of the reference or three combined
  standard errors, whichever is larger. Investigate flagged results; preserve
  genuine repeatable changes rather than forcing agreement by selecting runs.
- Set the two desired currents near the top of RR; running normally selects
  the correct persistent pulse, readout, frequency, averaging and timing profile.
  No diagnostic environment override is required for the final evidence.
- Safe offline harness and focused diff pass; protected source hashes and the
  QICK_experiment class remain unchanged. Record exact changed settings and
  all actual bias outcomes in the final handoff.

## Hardware envelope

Exact acquisition entry point: `round_robin_benchmark.py` (unsafe to import;
module-level acquisition). One round, normal experiment ordering. Allowed
flags: res_spec,q_spec,rabi,ss,t1,t2r,t2e; diagnostics may select a subset.
The profile in `expt_config.py` records each row's exact reps, rounds, scan
bounds and state-calibration counts before hardware operation.

Two GS200 sources only, current mode, 10 mA range, hard current grid
-10,-7.5,-5,-2.5,0,+2.5,+5,+7.5,+10 mA. Ramp at 0.5 mA/s; verify mode, range,
setpoint, identity and output. Enable at zero before nonzero ramps. Retain the
documented 600 s settling period at Yoko 3 -10 mA, also applied to +10 mA after
the frequency-tracking investigation below, with periodic progress.
Return both sources to 0 A/output off on completion or error.

Nominal attenuation arguments remain resonator DAC 10/15 dB, qubit DAC 5/4 dB,
ADC 17 dB. The current QICK_experiment constructor stores these arguments; its
RF-board setter calls are commented out. Preserve the existing RF path and do
not claim these constructor arguments are new instrument readbacks or writes.
Frequency centers, qspec gains/lengths, Gaussian sigma, pi gain and readout
settings come from the saved six-row and current-indexed calibration tables.
Resonator scans initially +/-1.5 MHz in 0.02 MHz steps; qspec spans and all
coherence delays are explicitly recorded in the runtime profile. Gain tuning
and frequency recentering remain bounded by measured local features.

Use the normal study/substudy/timestamp/optimization, study_data/Data_h5, and
documentation hierarchy. Preflight the longest plot and timestamped HDF5 path
below 260 characters. Zero pass is expected to take 1--2 hours; each flux point
15--45 minutes (longer for diagnosed weak readout). Finite subprocess deadlines,
RR logs, raw-data audits and source readbacks provide progress/health signals.
Abort on source mismatch, current/range violation, nonfinite/flat/saturated
measurement, failed save or deadline; diagnose before repeating.

## Progress

- Baseline audit: RR excludes M4 and defaults coherence population scaling off.
  Successful M4/M6 low-gain averaging and per-row coherence timing are currently
  only in prior temporary run plans. Persist these settings before validation.
- 21:14 local: launched the all-six default pass, with NO PUCQ4 environment
  overrides, in `rr6_0mA/2026-09-08_21-14-32`. Finite guard PID 59668 / RR PID
  60780, 9000 s deadline. Both sources verified current mode, 0 A, output off,
  10 mA range. Ten old Python windows (2025--February 2026) have unchanged CPU
  time and no established hardware TCP connections; preserved them.
- Offline: 81 combinations of documented bias selectors x six acquisition
  profiles checked; simulated ramp/zero/out-of-grid behavior checked without
  hardware. Protected sections/drivers/build_state hashes and system_config
  experiment-class AST unchanged. Safe harness passes (130 Python files).
- First M1 evidence: resonator 8919.79 MHz, qspec 5517.6901796 MHz / FWHM
  0.0487444 MHz, pi gain 0.691479 at sigma 0.050 us, SSF contrast 0.640.
  T1 plot gives 31.41 us versus prior 31.45 us. Raw HDF5 audit follows normal
  end-of-round batch persistence; these are interim plot/log observations.
- M1 Ramsey shows a late-time revival; RR's single decay fit gives about
  11.59 us, versus prior 16.53 us. The earlier reference trace also has late
  residual oscillations. Flag for repeat/residual review, without changing
  the experiment/QICK classes or selecting a fitting window to force agreement.
  M1 echo is 45.25 us versus prior 37.13 us. M2 qspec is 5438.7108871 MHz /
  0.0838350 MHz FWHM; pi gain 0.666606 at unchanged 0.101 us sigma; SSF 0.307.
- M2 T1 8.85 us, T2R 11.83 us, T2E 18.09 us (prior 8.77/11.71/15.88 us).
  M3 qspec 5557.3331106 MHz / 0.0380981 MHz FWHM and pi gain 0.661631 at
  0.124 us sigma agree, but SSF 0.176 versus prior 0.372 requires readout
  retuning after the current all-six pass. Actual tone 8974.94 MHz is 0.05 MHz
  above the previous optimum 8974.89 MHz. Do not accept the SSF loss silently.
- M4 completed: plot-level T1/T2R/T2E are 0.81/0.43/1.64 us, versus
  0.817/0.471/1.757 us. Qspec 5594.075854 MHz / 0.670623 MHz FWHM;
  pi 0.651682 at unchanged sigma 0.079003864231 us; SSF 0.299. Await the
  same end-of-round raw audit before promotion.
- M5 zero: qspec 7084.477746 MHz / 0.106667 MHz FWHM, pi 0.646707 at
  unchanged 0.056940825 us sigma, SSF 0.535. Compare spectroscopy with the
  archived spectroscopy center 7084.565737 MHz, not only the later pulse
  frequency 7084.280775 MHz produced by a Ramsey correction.
- Timing investigation (offline, provisional): all 18 archived short Ramsey
  corrections requested 0--1 us / 50 points / 4 MHz phase detuning, but saved
  actual delay spacing is 0.01953125 us. QICK's installed `QickRawParam.to_steps`
  truncates time and phase sweeps independently. Using the requested phase
  increment with actual time spacing implies 4.179592 MHz effective phase
  detuning, a +0.179592 MHz difference from the nominal correction constant.
  For M5 zero, that timing-adjusted candidate is 7084.460367 MHz, close to the
  new qspec. This is an explanation to check against fresh Ramsey data, not a
  validated frequency to promote. No archived raw data or program is changed.

## Prepared M3 follow-up envelope

After the all-six baseline completes and sources are verified zero/off, use
`run_readout_grid.py` / `m3_readout_frequency.json` in the task TEMP folder to
invoke normal RR with only `ss` enabled, index2/M3, both Yokos0. Eleven tones
8974.69--8975.09 MHz at0.04 MHz spacing; fixed gain0.41, length12 us,
sigma0.124 us, freshly measured qubit frequency5557.333110551 MHz and
pi0.661630958921. 20000 shots/state,250 us relaxation, unchanged attenuators.
Output `rrv3_roa/<timestamp>/` in the normal study hierarchy; longest planned
artifact path194 characters.180 s deadline per point. Abort on source mismatch,
QICK/save error or nonfinite raw data; return both sources to0/off. Then inspect
the frequency scan before choosing a length/gain-frequency follow-up and a full
chain with persistent corrected settings. These diagnostics cannot replace the
final full-chain evidence.

First M3 frequency diagnostic completed11 points: contrast0.264--0.324, with
a broad plateau near8974.89--8975.01 MHz. Because the exact scalar settings
differ from baseline only by0.01 MHz and shot count near the maximum, do not
attribute the entire contrast recovery to frequency without a repeat. Next
`m3_readout_length.json` first repeats baseline8974.94 MHz/12 us/10000 shots,
then measures6,9,12,15,18 us at8974.93 MHz/gain0.41/20000 shots per state.
Other pulse, bias, relaxation and attenuation arguments unchanged. Output
`rrv3_rob`, same194-character maximum path and180 s per-point guard/abort rules.
The exact baseline repeat gives0.3205, so the original0.1759 drop was not solely
the tone choice. A follow-on `rrv3_roc` sweep at21/24/27 us gives0.3443/0.3490/
0.3490, establishing the plateau from18 us (0.3507). These lengths are below
the saved ADC buffer duration29.630 us. Next `m3_readout_gain_frequency.json`
uses18 us, gains0.30/0.41/0.52/0.63 and tones8974.83/8974.93/8975.03 MHz,
20000 shots/state and250 us relaxation. Other pulse/bias settings unchanged;
output `rrv3_rod`, same194-character path and180 s per-point bounds. Recheck
length at the selected gain/frequency before accepting a full-chain repeat.

## Prepared nonzero-current sequence

`bias_queue_a.json` in the task TEMP folder selects both tunable rows separately
at+2.5,+5,+7.5,+10,-2.5,-5,-7.5,-10 mA, alternating M5 then M6 at each current.
The other source stays0. Only row/current/output-label selectors are overridden;
acquisition settings come from persistent profiles. Each point has a5400 s
deadline and a post-run raw/fit audit; the sequence stops on a failed audit.
The Yoko3+10 mA point additionally measures full chains for fixed M1/M2
(7200 s deadline), and Yoko4-10 mA measures fixed M3/M4 (10800 s deadline).
These provide a nonzero endpoint control for each fixed row and test profile
transitions between fixed and tunable rows. This is not an81-pair simultaneous
Yoko matrix, nor a claim that every fixed row was measured at every current.
All use the previously recorded source ramps, attenuator arguments, canonical
folder hierarchy and zero/off cleanup. Exact settings and source hashes are
saved in each guard/profile record.
`rrv3_rod` completed: gain0.63 at8974.83 MHz gives0.44975 contrast, above the0.41 branch. The next bounded SS-only grid `rrv3_roe` tests gains0.60/0.63/0.66/0.70 x8974.73/8974.83/8974.93 MHz at18 us. Same M3 pulse values,20000 shots/state,250 us relaxation,both sources0/off,nominal attenuation arguments,194-character maximum artifact path and180 s per-point deadline/abort rules.

## Baseline completion and report check (23:06--23:35 local)

The all-six `rr6_0mA/2026-09-08_21-14-32` pass finished with exit0 after
6697.757 s. All seven canonical HDF5 datasets and original RR plots exist for
all six rows. Both sources were verified CURR/0 A/10 mA range/output off before
and after. Independent audits promote M2, M4, M5 and M6. M1 remains review for
structured Ramsey residuals (canonical T2R11.59 us vs prior16.53 us); M3 remains
review for qspec R2.743, SSF.176 vs prior.372, and T2R55.93 vs42.85 us. Complete
metrics and exact source paths are in the validation manifest/audits.

M1 full-window exploratory two-frequency fits improve R2 from.771 to.970 in
the new trace, with fitted separation0.0160 MHz. The old trace also has a weaker
revival. These are diagnostic models, not a confirmed microscopic cause or a
replacement for the unchanged RR T2R fitter. Repeating the full delay window
with the historical500 reps is planned; no time-window selection to force the
old coherence number is allowed.

The one new validation PowerPoint was built with57 slides. Native PowerPoint
opened a byte-identical local preview and exported14 representative slides.
The existing PowerPoint application was initially blocked by an existing modal
error reporting it could not read the reference characterization path. Only
that verified error's OK button was acknowledged; no user presentation or
Office process was closed and no security setting changed. Native open/export
then succeeded. Visible acquisition-table crowding led to15-row chunks instead
of18; final rebuild/render remains pending. Both reference decks are preserved.
M3 fine grid reaches0.54475 contrast at the boundary0.70 gain/8974.73 MHz. Next `rrv3_rof` brackets this branch with0.70/0.80/0.90/0.95 gain x8974.43/8974.63/8974.83 MHz at18 us. Same pulse/calibration/zero-bias settings,194-character maximum path,180 s guard per point; inspect all raw SS results before selecting a length scan.
Expanded grid completed: maximum0.6924 SSF contrast at0.95/8974.63 MHz/18 us; neighboring0.90/8974.63 gives0.6734. Next `rrv3_rog` rechecks6/9/12/15/18/21 us at0.95/8974.63 MHz. Same M3 pulse parameters,20k shots/state,250 us relaxation,zero/off sources and nominal attenuation arguments. Same194-character path/180 s deadline; 21 us is below29.63 us ADC buffer.
High-gain length results are nonmonotonic:6/9/12/15/18/21 us gives.557/.620/.606/.560/.527/.684. The exact18 us value did not reproduce the earlier.692, so no single maximum is promoted. Next `rrv3_roh` tests0.80/0.90/0.95 x8974.53/8974.63/8974.73 MHz at9 us, with all other pulse/shot/relax/zero-bias values unchanged and180 s per point. The guard now also saves its launch-time four configuration/orchestration source files in documentation/rr_source; maximum intended path remains194 characters including these snapshots.
At9 us the second gain/frequency grid selects0.95/8974.63 again, contrast0.6249 vs the independent0.62045 repeat. Raw18 us comparison shows identical scalar settings, stable ground means and changed prepared-excited mean. No cause is assigned. `rrv3_roi` now interleaves9/21/24/27/21/9/18 us at0.95/8974.63 to separate repeatability from length dependence and bracket the upper plateau. Same20kstate/250us/pulse/bias/attenuation/path/180s constraints; maximum27 us remains below29.63 us.
Repeated9 us scores.620/.625/.561/.602; repeated21 us.684/.685/.543, with24/27 us.649/.607. Longer pulses provide no reproducible plateau advantage above run-to-run variation. Select9 us/0.95/8974.63 MHz for full-chain validation, with the measured variability explicitly retained. First measure the res dip via normal res_spec-only RR, index2, both sources0/off, rrv3_resb,151 points centered8975.00 MHz over+/-1.5 MHz,0.02 MHz spacing,600x1,100 us relaxation,nominal attenuation arguments unchanged.300 s deadline, maximum path195 characters, same abort/zero-off rules.

## M1/M3 full-chain repeat (23:57 local preparation)

M3 resonator-only RR completed with canonical PNG/HDF5, finite arrays, dip
8974.84 MHz and contrast/noise65.3. At9 us/0.95 gain, the measured optimal tone
8974.63 MHz therefore requires offset-0.21 MHz. Persistent M3 candidates:
res base8975.00->8974.84 MHz; offset-0.08->-0.21 MHz; readout gain.41->.95;
length12->9 us; f01 seed5557.33447->5557.333110551 MHz; pi seed.6586706->
.661630958921, sigma.124 us unchanged. Shared zero-current aliases agree.
Qspec half-span.8->.25 MHz,301->201 points,5000->10000 reps,one round/100 us
relaxation; state calibration10000->20000 shots/state,250 us relaxation.
M1 changes only Ramsey1200->500 reps, retaining0--60 us/301 points/0.5 MHz
phase detuning/250 us relaxation. Its other settings are unchanged.

Next `zero_repeat_b.json` calls the normal one-pass RR for indices0,2, with
both sources0/off and all seven default flags. Only row/current/substudy
selectors are overridden; all acquisition/pulse/readout values are persistent.
Rabi0--.85/171/1200x1/250 us for both. M1 T1.01--100/201/1200x1; echo.01--100/
501/1200x1/.5 MHz; M3 T1.01--150/301, Ramsey0--80/401/.5 MHz, echo.01--150/751/
.5 MHz,all1200x1/250 us. M1 readout8919.85 MHz seed with-.52 offset/.48 gain/
10 us, sigma.05 us. Res spec both+/-1.5 MHz/.02 MHz/151/600x1/100 us. Existing
nominal attenuation arguments retained. Outputrrv13_0b/<timestamp>,max195
characters;6000 s upper limit and logs/plot/raw checks. Abort on instrument,
source-state, finite-data or save failure; return both sources0/off. The
sequence audits both full chains before any final promotion.

Prepared post-run diagnostics (do not overlap the active full chain):
`m1_phase025.json` / `m1_phase075.json` use normal RR t2r-only at M1/index0,
phase detuning.25/.75 MHz,0--60 us/301/1200x1/250 us. Preserve the full window,
fresh measured carrier5517.691628432926 MHz, pi.6815296343770982,sigma.05 us,
actual readout8919.25/.48/10 us,10kstate/250 us. These test carrier dependence
of the persistent revival, not fit-window selection. `m3_ramsey_repeat.json`
uses M3/index2,0--80 us/401/1200x1/.5 MHz/250 us, fresh carrier5557.335688295514,
pi.666605627784899,sigma.124 us and actual9 us/.95 readout from the current
Rabi log,20kstate/250 us. It checks the clean59.84 us Ramsey estimate versus
old42.85 us and initial55.93 us. All three retain nominal RF arguments and
both sources0/off;600 s per run, canonical short substudy labelsrrv1_r25,
rrv1_r75,rrv3_rrep and maximum195-character artifact path. Same abort/readback
rules. Await raw audits before treating any result as validated.

## Zero-current follow-up decision and bias launch

M3 full-chain raw/fit/calibration checks all pass; only its reference T2R band
was exceeded. An independent repeat at the SAME carrier/pi/sigma/readout/phase
settings gives58.3425 us versus59.8376 us in the full chain, both with clean
independent fits and both outside the prior42.8505 us comparison band. Preserve
this confirmed change. `validation_M3_reviewed.json` retains the original
issues/audit and links the confirmation; no fit-quality threshold or raw value
was changed. M1 remains model review after500/1200 reps and.25/.5/.75 MHz phase
checks. Its numerical16.65/15.54 us matches at alternate frames are not valid
single-decay fits. Restore M1's1200 reps at the standard.5 MHz for future RR;
keep the complete60 us trace and explicit model flag.

Fresh zero-current resonance/frequency/pi settings were propagated across
system/expt/pucq4 aliases from their passing individual calibrations. M5/M6
zero-current coherence/linewidth metadata now use the accepted all-six run.
M1's Ramsey limitation does not invalidate its checked spectroscopy/Rabi/T1/
echo. M3's actual full-chain dip8974.80 and offset-.21 give8974.59 MHz readout.
All nonzero-current calibration entries remain unchanged before their tests.

Launch the already documented `bias_queue_a.json` now:16 tunable-row points
on the exact nine-current grid, with fixed-row controls at Y3+10 andY4-10 mA.
Only selection/current/substudy labels are overridden; persistent acquisition
profiles and pulse tables are used. Per-point guard/audit and deadlines remain
as recorded above; stop the queue on an unresolved finding. No other hardware
run may overlap. Each point returns both sources to0/off.

M5/Yoko3+2.5 mA first full chain completed with all canonical files and both
sources returned0/off. f7143.888395659 MHz/FWHM.11867654 MHz/R2.9850,
pi.646706952329 at sigma.053267224 us; T1/R/E10.5903/7.2870/16.0775 us
all compare within tolerance. Only SSF.4706 versus prior.615 fails. Res dip
9014.64 MHz (40.6 contrast/noise) gave readout9014.16 with offset-.48 MHz.
Pause bias queue. Next normal SS-only length scan rrv5p25_la atY3+2.5/Y4zero:
3/5/7/9/11 us at.93 gain/9014.16 MHz, fresh above carrier/pi/sigma,20k shots
per state/250 us relaxation. Nominal RF arguments unchanged; no RF setters.
180 s per point, existing canonical hierarchy/max196-character artifact path;
abort on source/instrument/data/save failure and return both sources0/off.
Choose short plateau then 2D gain/offset scan; full chain must validate changes.
M5+2.5 length scan finds3/5/7/9 us contrasts.4644/.4564/.4306/.4043;
11 us is finishing. Next rrv5p25_ga holds3 us and scans gains.73/.83/.93
by absolute frequencies9014.16/.36/.56/.76 MHz (offsets-.48/-.28/-.08/+.12
from measured9014.64 MHz). Same fresh carrier/pi/sigma,20kstate/250us,
Y3+2.5/Y4zero, nominal RF args and180s per-point guard/path/abort conditions.
Recheck length at the selected gain/frequency, then repeat a full RR chain.
Gain/frequency grid at3 us improves M5+2.5 SSF to.5367 at.93/9014.56 MHz
and.5446 at.93/9014.76 MHz. Next rrv5p25_lb tests1/2/3/4/5/7 us at the
latter tone/gain. Same pulse settings,20kstate/250us,Y3+2.5/Y4zero,nominal
RF args,180s and196-character guard envelope. Fine gain/frequency scan will
bracket the selected frequency after the length decision; no maximum is yet
promoted into shared config.
M5+2.5 length repeat at.93/9014.76 MHz gives1/2/3/4/5/7 us contrasts
.3439/.4775/.5388/.5794/.5854/.5844. Select4 us as the shortest plateau.
Next rrv5p25_gb scans gains.83/.93/.95 by9014.56/.76/.96 MHz at4 us,
with unchanged fresh pulse settings,20kstate/250us,Y3+2.5/Y4zero,nominal RF
args and180s/196-character guard envelope. This brackets the previous best
tone and reaches the previously documented .95 maximum readout gain.
Final4-us grid selects gain.95/tone9014.76 MHz, SSF.58675; neighboring
9014.56/.96 give.5690/.5355. Gain.93 at9014.76 repeats.5794/.57285;
.95 adds a small improvement while preserving the4-us plateau candidate.
Next normal res_spec-only RR rrv5p25_rb atY3+2.5/Y4zero, gain.95/length4us,
center9014.64 MHz +/-1.5 MHz/151 points/.02 MHz/600x1/100us relaxation,
nominal RF args unchanged.300s deadline/196-character guarded path. Verify
raw/PNG and dip, then store tone9014.76 minus measured dip as new offset.
Full persistent-profile RR rrv5_p025b is next, all7flags; bias_queue_b then
continues15 remaining points only if every audit passes. No overlap.
M5+2.5 res-only completed: dip9014.62 MHz, contrast/noise39.05, interior74
points, canonical HDF5/PNG and both sources0/off verified. Persistent candidate
changes in the existing M5+2.5 row: f7143.681614020->7143.888395658809 MHz;
pi.651681621->.646706952329; res_base9015.04->9014.62 MHz; readout tone
9014.56->9014.76 MHz; length5->4 us; gain.93->.95; offset-.48->+.14 MHz.
Sigma.053267224 us and qspec.0045/20us unchanged. Status explicitly pending
full RR; old coherence metadata is retained until a complete pass is accepted.
Offline131-file harness, protected source/class checks,81x6 profiles and
focused whitespace diff all pass. Launch bias_queue_b now: first rrv5_p025b
is all7flags/index4/Y3+2.5/Y4zero with only selector/substudy env overrides;
all acquisition/readout/pulse settings are persistent. Existing5400s per-row
bounds, endpoint7200/10800s bounds and zero/off guard remain. Stop on issues.

Prepare M5/Y3+5 mA qspec window before its queued run: persistent half-span
1.0->2.0 MHz, retaining401 points/5000x1/100us relaxation and.03 gain/40us
pulse. Point spacing becomes.010 MHz (previous.005). The prior selected qspec
is7121.575914 MHz/FWHM.869150612 MHz, while the old pulse seed is7122.148333
MHz. A +/-1 MHz scan would leave only.427581 MHz below the prior peak, about
one half-width, so use more baseline on both sides. No frequency candidate is
promoted by this change; the normal RR will measure the center before Rabi.
The exact planned bounds from the current seed are7120.148333--7124.148333
MHz, inside the reference campaign's frequency neighborhood. Current remains
Y3+5/Y4zero, all other profiles/guard envelopes unchanged. Active M6+2.5
process has already loaded its independent configuration and is unaffected.

M6/Y4+2.5 current pass has a clean line near6841.813506590 MHz/FWHM.207
MHz (previous qspec6842.101132 MHz), Rabi pi.656656290057 at sigma.053726424
us, and SSF.269 versus prior.5753. T1 is about7.30 us versus prior7.219;
Ramsey/echo are still running. Let the pass persist all data and stop normally.
Prepared next m6_p025_length_a.json (NOT RUN YET): normal SS-only RR index5,
Y3zero/Y4+2.5, fresh above f/pi/sigma, tone9059.58 MHz (dip9060.10/offset-.52),
gain.7967, lengths4/6/8/10/12/14 us,20kstate/250us. Nominal RF args retained,
180s per point and196-character path check. Same zero/off and abort rules.
After this length scan, scan gain/frequency and recheck the plateau; require
a full RR repeat before accepting. Investigate the qspec shift separately;
do not explain it as readout-induced without evidence. The normal qspec
program schedules readout after the probe pulse; it has not been modified.
M6+2.5 first full pass completed with all canonical files; both sources0/off.
Audit flags only qspec center shift-.287624630 MHz from previous qspec and
SSF loss.2693 vs.5753. Rabi and all coherence model/raw checks
pass: T1/R/E7.298445095/2.292438891/6.344210298 us, all within3% of prior.
Launch the already documented m6_p025_length_a scan now. All environment
values/bounds/guards remain exactly those prepared above; no other hardware
acquisition may overlap. Frequency comparison remains unresolved for follow-up.
M6+2.5 length4/6/8/10/12/14 us at.7967/9059.58 MHz gives
.2512/.27285/.2698/.24875/.2482/.2469. Choose6 us as the shorter plateau
for the next gain/frequency grid. rrv6p25_ga scans gains.65/.80/.95 by
9059.38/.58/.78/.98/9060.18 MHz (offsets-.72/-.52/-.32/-.12/+.08 from
measured9060.10 MHz), fixed6us. Same f6841.813506589753/pi.6566562900567662/
sigma.053726424,20kstate/250us,Y3zero/Y4+2.5,nominal RF args.180s per point,
196-character path guard, same abort and zero/off behavior. Recheck length
and final gain/frequency afterward; no candidate is yet promoted.
M6+2.5 coarse gain/frequency grid selects.80/9059.98MHz at6us,SSF.4184.
Its9059.78/9060.18 neighbors give.2932/.1371; gain.95 peaksat9059.78 with
.38365 and declines to.17465 at9059.98. Next rrv6p25_lb checks4/6/8/10/12/
14/16us at.80/9059.98 (offset-.12relative9060.10). Samefresh f/pi/sigma,
20kstate/250us,Y3zero/Y4+2.5,nominal RF args,180s/196-character guard/zero-off
conditions. A finer gain/frequency grid follows the measured plateau.
M6+2.5 at.80/9059.98 gives4/6/8/10/12/14/16us contrasts
.3485/.42705/.44455/.5044/.51415/.54025/.5707. The last point is still
rising, so rrv6p25_lc extends length to18/20/22/24us at the same frequency/
gain/freshpulse/current/20kstate/250us.24us remains below the measured29.63us
ADC buffer limit. Same180s per-point/196-character path/zero-off/abort guards.
Select the shorter plateau only after these data, then refine gain/frequency.
M6+2.5 extended lengths18/20/22/24us gave.57265/.64115/.67225/.683.
Next rrv6p25_gb uses the shorter22us plateau and gain.75/.80/.85 by
9059.90/9059.98/9060.06MHz. Fresh f6841.813506589753/pi.6566562900567662/
sigma.053726424,20kstate/250us,Y3zero/Y4+2.5 unchanged. Normal SS-only RR,
180s per point,196-character path guard, same RF args and zero/off/abort.
After refinement, repeat the selected shorter lengths before res anchor/full RR.
M6 fine22us grid: .80/9059.98 repeated atSSF.5859 (earlier.67225), nearby
.80/9060.06=.59445 and.85/9059.90=.61655. Improvement among nearbysettings
is smaller than same-setting variation; retain.80/9059.98. Next rrv6p25_ld
interleaves16/22/24/18/20/22us to check the shortest reproducible plateau.
All freshpulse/current/20kstate/250us/180s/196-character/zero-off bounds
unchanged. Launch only after rrv6p25_gb completes and its guard returns0/off.
M6 length repeat16/22/24/18/20/22us gave.6095/.66687/.6783/.6252/.63187/
.65065. Select22us at.80/9059.98MHz; the24us gain is small and shorter
16--20us generally lower. Normal res-only anchor rrv6p25_rb now: index5,
Y3zero/Y4+2.5, center9060.10MHz +/-1.5/151/600x1/100us at.80gain/22us.
Nominal RF arguments unchanged;300s deadline and196-character path guard;
abort on acquisition/save/source errors and returnboth0/off. No concurrentjob.
Orchestration correction before subsequent full chains: preserve the calibrated
absolute readout_frequency_ge after a new res scan and record the offset from
that measured dip. Previously old offset + new dip shifted M5+2.5 readoutfrom
9014.56 to9014.16MHz and M6+2.5 from9059.94 to9059.58MHz; scansdemonstrated
contrast loss at those shifted tones. Only normal RR selection changes;
explicit base/offset diagnostics and pre_optimize retain their previous meaning,
and an explicit absolute readout override wins. No section/QICK code changes.
Next full RR must verify the corrected path using only persistent settings.
M6+2.5 res anchor completed:9060.08MHz,contrast/noise26.38,interior74points,
canonicalHDF5/PNG and bothsources0/off. Shared pending rowchanges:
f6841.879742850->6841.813506589753MHz;pi.651681621->.656656290057;
resbase9060.46->9060.08MHz;readout9059.94->9059.98MHz;offset-.52->-.10MHz;
gain.7967->.80,length12->22us. Sigma.053726424us,qspecgain.006/20us unchanged.
Coherence metadata awaits accepted completepass.90 parsed configurationcases
verify calibrated-tone preservation and explicitdiagnostic override semantics.
Launch bias_queue_c: first rrv6_p025b all7normalflags/index5/Y3zero/Y4+2.5,
only selector/substudy overrides. Remaining14 bias/control points followonly
if audit passes.5400s ordinary,7200/10800s endpointdeadlines;previousdocumented
frequency/gain/timing/current/RF/path/abort/zero-offbounds remain. New code
preserves stored absolute readout at every point and records measuredoffset.
Prepared, NOT launched: m6_p025_qrep.json / rrv6p25_qr, independent qspec-only
repeat if complete rrv6_p025b has only the old-frequencycomparison flag.
Clone the actual full-pass PRE-qspec calibration seed6841.813506589753MHz,
pi.6566562900567662/sigma.053726424us, readout9059.98/.80/22us; Y3zero/Y4+2.5.
Persistent qspec.006/20us,+/-1MHz/401/5000x1/100us,10kstate/250us; nominalRF
args unchanged,1200s deadline/196charpath/sameabort and0/off. This repeatmust
match full qspec settings (not a later updatedRabi pi), passraw/modelchecks,
and confirm the line before accepting a changed frequency. No causes inferred.
Full rrv6_p025b completed1062.7s, all7canonicaldatasets/plots andbothsources0/off.
Onlyremainingissue: qspecfrequency difference-.325634MHz fromold6842.101132.
Newqspec6841.775497231MHz/FWHM.205259985/R2.99055; pi.651681621193,
SSF.6575; T1/R/E6.796211059/2.320281858/6.533448775us (within6%prior).
Launch alreadyprepared m6_p025_qrep.json now. AuditedfullPRE-qspecpi/frequency/
readout/axis/averaging exactlymatchthatplan; laterRabi pi is deliberatelynot
substituted. Same1200s/196char/sources0off/abort envelope. No otherhardwarejob.
Q6confirmation rrv6p25_qr/2026-09-09_03-15-36 completed303s,allraw/plots/0off.
Independentqspec6841.779011697MHz/FWHM.204226/R2.98970 agreeswithin.003514MHz
withfullchain6841.775497231; rawIQcenters andallstrictsettings/qualitychecks
pass. New validation_M6_reviewed.json accepts replicatedfrequencychange and
retainsoriginalcomparisonflags; no causeassigned. SharedM6+2.5rowpromoted:
pulsef6841.775497163998MHz,pi.651681621193,sigma.053726424us,SSF.6575,
T1/R/E6.796211059/2.320281858/6.533448775us,FWHM.205259985+/- .002032026MHz,
qspec5000x1/401points. Readout9059.98MHz/.80/22us,res9060.08/offset-.10.
Profile/protectedsource andfocusedwhitespacechecks passafterpromotion.
Launch bias_queue_d now:14 remainingnonzero points,firstM5/Y3+5/Y4zero,
rrv5_p050a all7normalflags/persistentprofiles. M5+5qspecseed7122.148333MHz,
+/-2MHz/401/.03gain/40us/5000x1/100us aspreviouslydocumented. Othergridbounds,
endpointcontrolrows,nominalRFsettings,5400/7200/10800s deadlines,196charpath,
600s Y3-10settle,abort/save checks andreturnboth0/off unchanged. No overlap.
M5/Y3+5 completepass rrv5_p050a/2026-09-09_03-22-35 passed allchecks,canonical
raw/plots and0/off. Promote pulsef7122.148333110->7122.257671006202MHz;
pi.651681621->.651681621193,sigma.058777626us unchanged. Res9014.59MHz,
RO9014.11/.92/6us/offset-.48 unchanged. Qspec.03gain/40us,FWHM.624343669+/-
.005091870MHz,R2.993725,5000x1/401;SSF.5782 (prior.512). T1/R/E now
5.174480411/1.131040093/8.360387540us,prior5.681098/1.303506/8.890440.
Activequeue_d advancesnormallyto M6/Y4+5 rrv6_p050a;foundres9060.78 and
preservedRO9060.22MHz,derivedoffset-.56MHz (old-.76). M5promotion occurred
afterM6loadeditsindependentcurrentprofile. No overlap orclass/programchanges.
M6/Y4+5 rrv6_p050a/2026-09-09_03-40-19 passedallchecks andwaspromoted03:58:40.
Pulsef7062.539018770->7062.612211354989MHz,pi.646706952->.651681621193,
sigma.053267224us unchanged. Res9060.98->9060.78MHz,offset-.76->-.56MHz;
actualRO9060.22/.9167/12us preserved. FWHM.171174610+/- .002246003MHz,
qspec.006/20us/5000x1/401,R2.983418. SSF.4262vsprior.491;T1/R/E
7.120905648/2.452151628/7.949307822us vs6.308351/3.342414/7.630265;
Ramsey-26.6% remainsvisiblewithin30%criterion,allmodel/rawchecks pass.
Queue_d38236 remainsactive: rrv5_p075a/2026-09-09_03-57-51 M5/Y3+7.5/Y4zero.
Its acquisition_profile alreadyloadedbeforeM6+5sharedrowpromotion.12nonzero
pointsincludingactive remain; finaldefaultall6stillrequired. Nooverlap.
Upcoming negative-bias acquisition preparation (notmeasuredyet;activeM5+7.5
alreadyloadedandunaffected): M6/Y4-7.5 qspechalfspan1->2MHz,401points retained,
fromsavedpulse4839.206595110MHz =>4837.206595110--4841.206595110MHz,
.010MHz requestedspacing. Old tentativeqspec4839.816961MHz/FWHM.420299
wouldleaveonly.389634MHz uppermarginina+/-1window. M6/Y4-10 qspecreps
5000->20000,rounds1/401/+/-1MHz retained,fromsaved4375.850089710MHz =>
4374.850089710--4376.850089710MHz,.005MHz requestedspacing. Prior500rep
spectrumSNR1.224wasunresolved; moreaveragingisrequiredtoassessanactualpeak.
Both.03gain/20us/100usrelaxation,10kstate/250us/readout/pulses/RFvalues
unchanged. Theseareacquisition-onlychanges;no tentativefrequencyispromoted.
Allfrequencyboundsremaininsidethearchived4000--7300MHzmap. Currentbounds,
.5mA/sramp,sourceguards/0off,pathguardandqueue5400/10800sdeadlines unchanged.
The-10qspecaveragingaddsabout14minutesversus5000reps,withinendpointdeadline.
M5/Y3+7.5 ongoing rrv5_p075a has cleanqspec7022.356824934MHz/FWHM.120562,
pi.651681621193/SSFabout.725, T1plotabout12.13us versusprior7.2141;
Ramseyabout3.55us versusprior3.7675. Echoactive04:12. T1change requires
independent same-settingrepeat aftercompleteaudit;do notpromoteorforceoldtau.
Nohardwareoverlap;queue_d38236 willstoponthiscomparisonfindingafterfullsave.
PreparedNOTRUN m5_p075_t1rep.json / rrv5p75_t1 normalT1-onlyRR: index4,
Y3+7.5/Y4zero,f7022.356824933893MHz,pi.6516816211926998,sigma.055104024us,
RO9013.80MHz/.95/10us/offset-.20. PersistentT1.01--60us/241/1200x2/250us,
10kstate250us,nominalRFargs unchanged.900sdeadline,196charpath,source.5mA/s
and0/offguards/abortonacq/save/sourceerror. Launchonlyafterfullruncompleted
anditsrawauditconfirmswhichmetricsneedrepeat. No otherhardwaremayoverlap.
M5+7.5 completeaudit flagsONLYqspecfrequencycomparison(+.236584MHz vsold
7022.120241MHz) andT1comparison(+68.09%). Newqspec7022.356825833MHz/
FWHM.120574781,R2.976617;pi.651681621193,SSF.7245;T1/R/E12.126387325/
3.550125101/10.560902606us. Allraw/model/calibration/Rabi/SSFchecks pass.
Queue_d38236 exited,canonicalraw/plots/0offverified. LaunchpreparedT1-only
m5_p075_t1rep.json now;its exactpulse/readout/acqvaluesmatchfullT1config.
Independentqspec-onlyrepeatwillfollowwithfullPRE-qspecseed/settings. Both
comparisonchanges mustbeconfirmedbeforepromotion; noqualityfindingwaived.
PreparedNOTRUN m5_p075_qrep.json / rrv5p75_qr qspec-onlyRR to followT1repeat:
index4/Y3+7.5/Y4zero,actualfullPRE-qspecseed7022.138348760MHz/pi.651681621,
sigma.055104024us,RO9013.80/.95/10us/- .20. Qspec.002/20us,+/-1MHz/401/
5000x1/100us and10kstate250us matchprimaryexactly.1200s/196char/nominalRF/
sourceguards0off unchanged. No candidatefrequency substitutioninto calibration.
M5+7.5 T1-onlyrepeat rrv5p75_t1/2026-09-09_04-16-54 completedwithcanonical
raw/plot/0off. T1=11.044569115us, onlyoldcomparisonflag; allraw/modelchecks
pass. Differencefromfull12.126387325 is9.34%ofmean,within15%repeatgate.
Launchpreparedm5_p075_qrep.json now,exactpreviouslydocumentedPRE-qspecseed/
pi/axis/calibration/readout/currentsettings. Nootherhardwarejob. Requireqspec
agreementandcompletequalitygatesbeforecombinedfrequency/T1reviewpromotion.

2026-09-09 09:03 local: M5/Y3 +7.5 mA qspec confirmation rrv5p75_qr (04:21:41) is clean: 7022.336430607398 MHz, FWHM 0.114938591509 MHz, R2 0.97553. It differs from complete primary by 20.395 kHz, exceeding the predeclared 0.1-linewidth repeat gate. Do not execute/promote the prepared multi-metric review. T1-only repeat is 11.044569 us versus full-primary 12.126387 us, confirming the longer T1, but the complete row remains review. Both sources ended zero/off.
Next authorized run: normal round_robin_benchmark.py, all seven flags, M5/index4 only, Y3 +7.5 mA/Y4 zero, same shared calibration/profile; output rrv5_p075b under existing first-light hierarchy. Qspec seed 7022.138348760 MHz, +/-1 MHz/401/5000x1, gain .002/20 us; sigma .055104024 us; readout 9013.80 MHz/.95/10 us. T1 .01--60 us/241, Ramsey 0--30 us/301/.5 MHz, echo .01--60 us/301/.5 MHz, coherence 1200x2/250 us; normal Rabi 0--.85/171/1200x2/250 us; statecal 10k/state/250 us. Nominal attenuation arguments remain res10/15, qubit5/4, ADC17 dB with no setter changes. Abort on instrument/readback/raw-save failure or 5400 s deadline; both sources ramp zero/off afterward. Guard verifies no simultaneous acquisition, n=1 and longest intended path below 260 characters before hardware loop (normal substudy label length unchanged).

09:14 local: the new full M5/Y3+7.5 run rrv5_p075b/2026-09-09_09-01-35 found qspec7022.162967054066MHz/FWHM.122317MHz (close to old reference), unchanged sigma.055104024us and measured pi.6516816211926998, SSF.7133. T1 plot is12.76us; full saved-data audit awaits Ramsey/echo. Prepared NOT LAUNCHED m5_p075_t1rep_b.json: normal T1-only RR/index4, Y3+7.5/Y4zero, exact new f/pi/sigma/RO9013.80MHz/.95/10us. T1 .01--60us/241/1200x2/250us, calibration10k/state250us, unchanged nominal attenuation, output rrv5p75_tb under canonical first-light root,900s deadline, both0/off. Launch only after the full pass has exited and its T1 config is verified against this plan. Earlier 03:57/04:21 frequency scans remain preserved; no cause assigned to run-to-run shift.

09:26 local: exact-setting T1 repeat rrv5p75_tb/2026-09-09_09-20-33 measured12.337499645+/-.094050848us, all original quality gates pass and agrees with full12.757370733 within the15%repeat criterion. Reviewed fresh full-chain validation_M5_reviewed.json preserves original prior7.214108us comparison; no cause assigned. Shared M5+7.5 row promoted: pulse7022.138348760->7022.162967054066MHz, res9014.04->9013.96MHz, offset-.24->-.16MHz (absolute RO9013.80MHz unchanged), T1/R/E7.214108/3.767455/8.624310 ->12.757370733/4.038153856/10.907325683us. Sigma.055104024us, pi.651681621193, gain.002/20us, RO.95/10us retained. FWHM.122318558+/-.001824836MHz and SSF.7133 saved with5000x1/401 metadata.81x6 profiles/protected-source/whitespace checks passed. Both sources0/off.
Resume bias_queue_e:11remaining bias/control groups, all seven normal flags, current selectors only, documented grid and persistent profiles. First M6/Y4+7.5/Y3zero (qseed7185.390636570MHz, +/-1MHz/401/5000x1, qgain.006/20us, sigma.057859226us, pi.651681621, RO9061.49MHz/.8334/11us); T1.01--60us/241, Ramsey0--30us/301/.5MHz, echo.01--60us/301/.5MHz,1200x2/250us/cal10k250us. Subsequent points are in archived queue_e plan. Ordinary5400s; Y3+10 controls7200s; Y4-10 controls10800s; Y3-10 settles600s. Nominal attenuation unchanged, canonical pathguard<260, abort on audit findings/failed save/readback/deadline,0/off after each point. Final all-six default pass still pending.

09:42 local: M6/Y4+7.5 rrv6_p075a/09:26:40 has main qspec7185.488649831208MHz/FWHM.129291MHz with a visible high-side shoulder. Rabi pi.6516816211926998 at unchanged sigma.057859226us, SSF.4289 vsprior.474, T1plot6.52us vs5.60195, Ramsey5.52us vs4.92746; echoactive. Do not promote before full raw-I/Q review. Routine audit now applies the prior strict-repeat qspec gates (SNR>=6,residualsigma<=2,raw/pop centers withinmax.1FWHM/3SE,baseline>=oneFWHM peredge). All10previousacceptedselectedaudits satisfy them.
Activequeue_e loaded the old auditor before this update. After its completedpass, run current auditor with --audit-suffix _iq to preserve the original audit. The suffix produces a separate documentation audit, and record_attempt removes same-run approval if a new quality finding appears, preserving earlier reviewed comparison decisions and accepted different runs. Isolated JSON-fixture tests verified these cases plus suffix containment;131-fileharness andfocusedwhitespace pass. This is analysis/orchestration bookkeeping only; no hardwareprogram/class changes. Independent later bias groups can continue while a different current's spectroscopy is reviewed; each uses its own documented row. Subsequent sequence launches load current gates.

09:46 local: M6+7.5 complete rrv6_p075a/09:26:40 and versioned validation_M6_iq.json pass all raw/model/calibration/coherence gates, including rawqspecSNR30.4676/residual1.1553sigma/raw-popcenter0.644kHz agreement. Maincenter7185.488650128438MHz/FWHM.129294582766MHz/R2.956205; oldcenter7185.659470149740 gives-.170820MHz versus.146969MHz comparison tolerance, the ONLY finding. T1/R/E6.524914838/5.515944113/10.074508006us,SSF.4289,pi.651681621193,sigma.057859226us,res9061.47/RO9061.49MHz/.8334/11us/offset+.02. Queue_e57136 stopped normally; bothsources0/off.
Next authorized normalqspec-only repeat m6_p075_qrep.json, index5/Y3zero/Y4+7.5; EXACTPRE-qspec seed7185.390636570MHz/pi.651681621/sigma.057859226us, sameRO9061.49/.8334/11us/offset+.02. Qspec gain.006/20us,+/-1MHz/401/5000x1/100us;cal10k/state250us. Nominalattenuationsunchanged;canonicaloutputrrv6p75_qr,maxpath196,1200sdeadline,abortinstrument/save/readbackfailures,both0/off. Freshqueue_f10remaininggroups preparedNOTRUN and will load new stricter auditor. Do not promote until independentfrequencyrepeat gate passes.

09:54 local: M6+7.5 qspec repeat rrv6p75_qr/09:46:23 passed all current audit gates: center7185.521795555906MHz,FWHM.135945514821MHz,R2.954388, within old-reference linewidth. Difference from the firstscan is33.145kHz, so the strict0.1-linewidth repeat criterion did NOT confirm a fixed shift; no review waiver was performed. Bothsources0/off. Shared starting qfreq7185.390636570->7185.521795227345MHz now follows this measured clean line, explicitly frequency_candidate_pending_rr with seed-run provenance. Other pulse/readout/coherence settings retained; no measured coherence metadata promoted.81x6profiles/protected/whitespace checks pass.
Next: queue_f starts full normal RR M6/Y4+7.5/Y3zero, all7flags, only selectors/substudyrrv6_p075b, seeded7185.521795227345MHz +/-1MHz/401/5000x1/gain.006/20us/100us, unchanged sigma.057859226us/pi.651681621/RO9061.49/.8334/11us. T1.01--60us241;R0--30us301/.5MHz;E.01--60us301/.5MHz;1200x2/250us,cal10k250us. Canonicalpath196,5400sdeadline,nominalattenuationunchanged,both0/off; abortfailedsave/readback/instrument/auditfindings. If accepted, continue original10remaininggroups. Freshcontroller loads stronger raw-IQ audit. Earlier shifted fullpass remains a diagnostic, not accepted calibration.

10:16 local: M6/Y4+7.5 fresh full rrv6_p075b/09:54:43 PASS all current gates, source0/off and canonicalraw/plots verified. Promoted10:13:28: pulse7185.513939668361MHz (qspecindependent7185.513938427711),FWHM.136510718884+/-.002942536122MHz,R2.956286,rawSNR28.734/residual1.10897sigma;pi.6467069523286333,sigma.057859226us,res9061.47,RO9061.49MHz/.8334/11us/offset+.02,SSF.433;T1/R/E5.451795272994/5.723199056463/9.238789592754us. Qspec.006gain/20us/5000x1/401metadata and calibration_run saved. Originalpulse7185.390636570->latest7185.513939668361MHz; oldpi.651681621->.646706952329; offset-.04->+.02 preservesabsoluteRO. Allprofile/protected/whitespacechecks passed afterpromotion. Earlier frequency variation remains in M6_p075_frequency_variation.png/json,3actualscans,no causeassigned.
Queue_f30964 continues rrv5_p100a/2026-09-09_10-12-32, M1/M2/M5 indices0,1,4 atY3+10/Y4zero,normalprofiles,7200sdeadline. M1qspecactive10:15. Nooverlap;nextgroups remaininqueue and pauseonfindings. Report93slides/121configchanges rebuilt10:14,onlyknownM1zeroRamseyfindings;native8slidesexported10:15 withnewusagescope. Eightof18fluxoperatingpointsincludingzero nowaccepted. Finalall6defaultstillpending.

10:23 local: M1 control atY3+10/Y4zero found5517.675162084726MHz/FWHM.05642193MHz and T1plot31.42us matching old31.45us. Its normalRamseyplot at10:21:04 again shows pronouncedrevivals; single-decayfit7.93us is not accepted as a T2R. This repeats the documentedzero-currentmodel limitation. CurrentpasscontinueswithM1echo,M2,andM5 so other validmeasurements are preserved; do notalterclasses/programs orcropawayrevivals. Reportbuilder now includes original flaggedcontrolplots nexttoitscontrolaudit table, aftercompletecanonicaldata exist.

### September 9 10:52 local - +10 mA review and exact-setting confirmations

The all-seven-experiment M1/M2/M5 control group rrv5_p100a/2026-09-09_10-12-32 completed normally in 2182.95 s; all canonical raw data and plots persisted and both Yokogawas returned to 0 A/output off. M1 repeats the known structured Ramsey revivals (single-model 7.9329 us is withheld), while T1 31.4180 us and echo 32.0112 us match the original comparison. M2 has clean longer T1 12.05570 +/-0.42916 us and Ramsey 15.60720 +/-0.66088 us, versus 8.76716/11.70821 us in the reference. Echo20.45527us passes its comparison. These are biased controls, not zero-current calibration replacements.

M5/physical Q4 at Yoko3 +10 mA has center6841.583959579MHz, FWHM0.160779538MHz, sigma0.056481625us, measured pi0.651681621193, res9013.07MHz, RO9012.97MHz/.95/6us/offset-.10MHz, SSF.6415. T1/R/E14.718917807/2.435159840/8.426011469us all pass. Only qspec old-reference shift+0.383215MHz is flagged; this row is not promoted yet.

Queue_f exited and paused correctly. ACTIVE m2_p100_trrep.json controller57472 (10:50:50), normal RR flags t1,t2r, M2/index1 at Y3+10/Y4zero, f5438.719629589458MHz/pi.661630958921/sigma.101us, RO8951.21MHz/.56/5us/offset-.46MHz. T1 .01-60us/121; Ramsey0-40us/201/.5MHz; both1200x1/250us, statecal10k/250us, 900s deadline, .5mA/s ramps, normal raw hierarchy, both0/offafter. Exact repeat plan m5_p100_qrep.json is prepared, NOT STARTED: M5 qspec at same bias with PRE-qspec seed6841.22106308MHz/pi.646706952/sigma.056481625us, same RO, qgain.002/20us/401/5000x1/100us,1200s deadline. Nine remaining groups are prepared in bias_queue_g.json, NOT STARTED. Do not overlap hardware.

Offline review helper now supports multiple comparison-only changes confirmed by a single independent run and fixed controls at their actual two-source biases. It never waives raw/model/quality findings. Isolated JSON fixture passes (10:51:56), verifies two decisions, preserved originals/history, no zero/flux promotion and model rejection. Report builder includes unresolved control traces and actual-bias confirmed-change rows. Final all-six normal default zero-current run remains required and not started.

### September9 10:57 local - M2 endpoint-control review

Exact-setting independent control rrv2p10_tr/2026-09-09_10-50-52 completed with normal raw/plots and both sources0/off. T1 10.620643970+/-0.367919740us passes old comparison, while T2R15.705895870+/-0.625004097us repeats the primary15.607197551us increase. Primary T1 12.055699858us and repeat differ1.435056us, within the pre-existing15%-of-mean repeat tolerance1.700726us. T1 is recorded as consistent repeat with reference-band overlap, explicitly NOT a persistent increase. Both traces pass original calibration/raw/model gates. Helper now allows a coherence confirmation inside the old comparison band only when the primary and repeat still agree under unchanged uncertainty/15% rules; unexpected/quality/model findings still reject. Qspec retains the stricter outside-reference repeat requirement. Fixture exercises this mixed case and control-only selection, PASS10:56:01.

Selected validation_M2_reviewed.json stays in manifest.controls at Y3+10/Y4zero; shared zero config is unchanged. Reviewed decisions preserve both measurements and original comparisons. New report wording distinguishes repeat-within-old-band from confirmed change; M2 full/repeat T1 and Ramsey traces are included. Current only hardware: m5_p100_qrep.json controller56736, start10:54:44, qspec-only Q4/M5+10mA; queue_g remains prepared/not started.

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

### September9 11:44 local - frequency repeat interpretation (before result)
The matching600s-settling qspec repeat is still acquiring (~30% at11:44). The unchanged strict0.1-FWHM/3-fit-SE test will decide whether a fixed frequency shift is confirmed. Do not force the center to an old reference or call a variable center stable.
If that static-frequency test fails while the raw line remains clean, distinguish it from failure of the requested acquisition workflow. A separate, explicitly labelled frequency-variation review would require: both matching-settling scans pass every raw/calibration/fit/edge gate; center separation no more than one smaller measured linewidth; widths agree within20%/3SE; the complete settled RR passes every Rabi/SSF/coherence gate; its pi/readout/coherences also agree with the earlier complete RR using the same pulse/readout settings (15%-of-mean/3SE coherence tolerance); and the normal default sequence retains qspec before Rabi/coherence and propagates its measured center. Preserve all different settling histories and the old frequency comparison. Such evidence could validate reproducible tracking of a varying transition, but would not validate a constant shifted frequency. Record the observed range and mandatory qspec step in any resulting calibration/report; do not relax the general auditor or waive a quality finding.

### September9 11:54 local - resume remaining bias queue
Queue_g contains nine remaining documented current/control groups and starts with Q6/M6/index5 atYoko4+10mA,Yoko3zero. All7normalRRflags, one round, persistent profiles; Qspec7210.22085715-7212.22085715MHz/401/5000x1/.006gain/20us/100usrelax; res9061.68+/-1.5MHz/151/.02step/600x1/100us. RO9061.52MHz/.8722/12us, sigma.049593622us, pi seed.641732283. Rabi0-.85/171/1200x2/250us; T1.01-60us/241,Ramsey0-30/301/.5MHz,Echo.01-60/301/.5MHz,coherence1200x2/250us,statecal10k250us. Nominal attenuator args10/15,5/4,17dB unchanged. Sources ramp.5mA/s within documented+/-10mA; no extra Yoko4 wait, maxartifactpath<210,5400sdeadline for this point,0/offbetweenpoints. Queue stops on any raw/fit/comparison/save/readback issue. Subsequent documented negatives retain prepared profiles and Yoko3-10mA600s settling; endpoint M3/M4 controls retained. No overlapping acquisition. Latest131-file harness,81x6profile/protectedsource/trackingguard andfocusedwhitespace checks pass.

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

### September 9, 19:00 local — Q6 full E complete; exact readout confirmation

rrv6_m025e/2026-09-09_18-28-03 completed with all7canonicalHDF5 and normalplots; both sources verified CURR/0A/0.01A-range/outputoff afterward. Audit passes every raw/calibration/model/coherence gate; solefinding historicalSSF. Measured qspec6103.731047905MHz/FWHM.328195116+/- .013409379MHz/R2.962903206; actualcarrier6103.73104802091MHz; pi.651681621193/sigma.056022425us. Res9057.90MHz, readout9057.86MHz/.8/10us, offset-.04MHz. SSF.32474. T1/T2R/T2E6.143175070+/-.119192534,1.015747031+/-.070154168,4.291727696+/-.151317026us; historicalchanges+3.82%,-13.45%,+3.40%. No promotion yet.

Next m6_m025_e_ssrepeat.json: exact normalRR SS-only onepass at physicalQ6/M6/index5, Yoko4-2.5mA/Yoko3zero. Uses all measured values above,50000shots/state/250us, trigger.4us/zero phases; .5mA/sramps,noextrasettle,bothzero/offafter. Nominal attenuation args10/15,5/4,17dB unchanged,197-character maximum path,240sdeadline; abortreadback/acquisition/saveerrors. Exact bootstrap and predeclared .03-or3SE repeatgate apply. Nine legacy RR Python processes (2025--February2026) have unchanged CPU and no established TCP connections; preserved. Current task has no competing acquisition. Final-all-six priority is the stated default pending optional user reply; target22:00local remains.

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

September 9,20:10local: prepared-only optional -5mA envelopes are archived as
optional_closeout_bias_envelopes.json. Plans m5_m050_closeout.json and
m6_m050_closeout.json use normal one-pass RR/all7flags and persistent profiles;
Yoko3/Q4/M5 -5mA with Yoko4zero, then Yoko4/Q6/M6 -5mA with Yoko3zero.
Q4 seed6724.84481367MHz,sigma.084033637us,pi.651681621;
Q6 seed5588.10371144MHz,sigma.070716831us,pi.65665629. Complete readout/frequency/
gain/sampling/averaging bounds are in the envelopes. Each maximum1500s, expected
17-18min; paths196characters, .5mA/s ramps, both0/off cleanup, unchanged nominal
attenuation arguments10/15,5/4,17dB. No hardware launched by preparation.
Only launch after final default audit/required confirmations if the remaining
closeout time covers the bounded acquisition and final reporting. Do not overlap
hardware. These are additional coverage, with unchanged acceptance gates.


September 9 final default pass automatic audit: complete_with_findings; 3/6 complete chains pass the unchanged reproduction gates. Elapsed 113.17min. Both sources verified0A/output off. Exact audits and all findings: validation artifacts/final_default_audit_summary.json. The one validation deck has been refreshed with every final-run result. Shared config promotion, final visual review and time-bounded remaining work await review.

September9 21:17local: final all-six run completed with canonical data and sources0A/off. M4--M6 pass; M1 coherence and M2 SSF findings remain. M3 sole T1 comparison:31.460196us versus45.792778us reference. Prepared one exact T1 repeat (rrv3_ft1r, m3_final_t1repeat.json), carrier5557.32523493106MHz/pi.661630958921/sigma.124us, RO8974.59MHz/.95/9us/offset-.25MHz; .01--150us/301points/1200x1/250us, statecal20k/250us; n1, t1 only, bothsources0/off,300sdeadline, longestH5path189characters. Unchanged 15percent/3combinedSE repeat gate; original audit retained.

September9 21:19local: prepared one exact M2 SS-only confirmation after final contrast.2678 versus historical.343171585793. Plan m2_final_ssfrepeat.json/rrv2_fssr: carrier5438.708250302973MHz, pi.661630958921, sigma.101us, RO8951.21MHz/.56/5us/offset-.46MHz,10k shots/state/250us; bothsources0A/off,n1,180sdeadline. Repeat limit remains max(.03 contrast,3combined bootstrap SE); retain comparison and optimization gap, no automatic promotion.

September9 21:22local: last closeout hardware acquisition is physicalQ4/M5/index4 at Yoko3 -5mA, Yoko4 zero. Plan m5_m050_closeout.json/rrv5_m050a and artifact optional_closeout_bias_envelopes.json record every pulse/readout/sweep parameter; normal RR/all7 flags/n1, source mode/range/identity checks,.5mA/s ramps,0s settle, both0A/off afterward,1500s deadline, longest artifact196characters, nominal atten10/15,5/4,17dB unchanged. Expected finish21:40, leaving audit/report time to22:00. Do not launch additional negative groups. M3 exact T1 repeat37.915038us disagrees with final31.460196us under the unchanged15percent/3SE gate; both individual fits pass, no promotion.

September9 21:40local: Q4/M5/-5mA full pass saved all seven datasets and normal plots; sources0A/off. Sole finding is qspec center6725.225762893MHz versus reference6724.886395756MHz (linewidth tolerance.169327977MHz). T1/T2R/T2E14.728369/2.195412/7.209698us all pass reference/raw/model gates; pi.656656290057. Reserve one final exact qspec repeat rrv5m50_qsr, m5_m050_qspecrepeat.json,360smaximum (~5min), same original spectroscopy seed6724.84481367MHz, pulse/readout/sweep/statecal. This short confirmation is the only addition to the prior last-run plan and fits22:00closeout; no additional current groups or retries. Accept only unchanged raw/model gates and max(.1linewidth,3combined centerSE) repeat agreement.
