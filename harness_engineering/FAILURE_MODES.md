# Diagnostic loop and known failure modes

## An optimization iteration limit is not proof of a plateau

November4 Q2 capped-power cycles continued improving peak SSF contrast
.0962->.12605->.1518, so the three-cycle budget correctly stopped promotion.
Reuse identical completed plans/raw results and extend a bounded number of
cycles inside the already qualified gain/frequency/length envelope. Do not
restart completed hardware scans, call the last increasing point optimized,
raise the power ceiling, or relax data-quality gates just to finish. If the
apparent improvement is temporal variation, preserve the interleaved repeats
and qualify a working pulse rather than claiming a stationary optimum.

## Distinguish resonator visibility floor, movement onset and SSF power

November4 Q2 measurements exposed an overly conservative derived0.10DAC
ceiling: normal SSF contrast only~.06-.10 despite a repeatable resonator. The
actual power survey had stable0.30 spectra and first resolved movement0.60.
User asked the lowest reliably visible resonator power, then readout optimized
BELOW movement onset; these are separate operating powers. Do not impose an
extra full-step margin as though it were a measured physical onset or accept
weak single-shot contrast merely because a resonator is visible. Requalify
any revised ceiling against fresh low/cap/cap/low spectra, preserving measured
movement/uncertainty criteria and historical dates. Only accepted unchanged
resonances permit readout optimization at that power; no increase above onset.
Q2 bounded0.24=80%of stable0.30 was scheduled for this check, not assumed safe
calibration. Retain original low-power/noisy traces and their qualifications.

## A valid low-power full benchmark can exceed a one-hour launcher budget

Layer: execution environment / verification feedback. October30 M1 full1_low
hit its3600s subprocess timeout while Echo49% complete after successful prior
experiments. Weaker readout needed16000reps; timeout was not evidence of a QICK
fault. The batch collector had not persisted its final HDF5 files. Keep partial
plots diagnostic; do not promote them. Budget complete calibration jobs from
actual point/repetition/reset counts (November4 bound10800s), with a separate
600s no-log-progress watchdog. Terminate only the owned child tree on failure
and always record the error/returncode and final source readback.

Opt-in PUCQ4_PHASE_CHECKPOINTS persists each completed experiment once through
the same Data_H5 writer and schema before the next phase. It is restricted to
single-row/single-pass/save_r1/debug/standard GE jobs and excludes timed SSF.
End-of-round writers skip those already persisted types. Default runs retain
batch behavior. Verify actual checkpoint calls, canonical HDF5 field sets and
batch numbering, duplicate rejection, and the disabled no-op offline; never
import RR to test it. A partial seven-experiment pass still fails final review.

## A marginal low-power resonance needs matched-statistics repeats

October30 all-six readout-power work: Q2 gain.03 resolved with16000averages
but a10000-average return did not; M1 gain.03 also failed its weaker repeat.
Classify as verification/context, not confirmed physical drift or punch-out.
Record averaging and frequency sampling with every resonance-SNR criterion.
Repeat the lowest candidate at matching bounded statistics; require a resolved
reference before comparing its frequency with higher-power traces. Fresh
paired references distinguish drift from power dependence. Preserve rejected
raws and never accept an arbitrary minimum from a noise trace. The queued
follow-up uses two16000-average fine spectra with stricterSNR>=15/fitR2>.95.

## Long SSF series must verify exact saved paths without growing-folder scans

At roughly1s cadence a24hseries can create tens of thousands of plot/HDF5
files. Per-round glob scans then grow with accumulated history and distort
cadence. Data_H5.last_saved_path is set after successful close and
SingleShot.last_plot_path after savefig; timed RR verifies those exact files
and decodes all raw samples before proceeding. The schema/filenames remain
unchanged. A hardware-free2000sample canonical HDF5 roundtrip passed.

## Flexible Ramsey alternatives can become degenerate

October29 fresh M1 normal RR T2R20.16075us/R2.97597 is reproducible in the
independent single-frequency fit. An unconstrained diagnostic two-frequency
model reduced residuals slightly by fitting frequencies about2Hz apart with
opposing component amplitudes around730 for a population range near1.
Classification: verification. Reject this ill-conditioned cancelling solution;
it is not evidence of a resolved doublet and its envelope must not replace the
normal RR value. The external model-review helper now flags near-coincident
frequencies/large cancelling amplitudes and omits rejected curves from plots.
Keep the rejected numerical result in diagnostic JSON for provenance.

## Readout settings vary faster than a serial optimization grid

The October29 M1 higher-gain confirmation initially compared its new maximum
with two-hour-old independent examples. That can reject a useful current
setting simply because the entire response moved. The external confirmation
helper now compares with the matching reference point in the same new grid
(and includes the current gain explicitly). The running controller had already
loaded the old logic; review its result before any source promotion, and use
the corrected helper for a further M1 confirmation if needed. Preserve the
earlier high examples as dated observations, not current performance.

October29 M2 fresh controls pass Rabi, but 7us/.95/8949.73MHz references
span SSF.38836-.48430 in two minutes; adjacent-reference stability checks fail.
Classification: verification/state. Preserve failed gates and raw families;
do not promote the earlier maximum or relax a repeat threshold. Use balanced
repeated settings in reversed/rotated blocks, report between-block variation
and paired differences, and qualify a working choice if a stationary optimum
remains unresolved. Stable pulse calibration does not establish stable SSF.
Continue independent rows only after the previous controller has ended and
all instrument/save receipts are successful.

When something fails, first attribute it to one of the five harness layers. Fix
the narrowest layer, re-run, and record the result. Do not compensate for a
missing rule by increasing power, scan width, repetitions, or model effort
without evidence.

## Layer checklist

| Layer | Ask | Typical repair |
| --- | --- | --- |
| Task specification | Are outcome, non-goals, units, indexing, authority, and acceptance objective? | Fill `TASK_TEMPLATE.md`; make the hardware envelope bounded. |
| Context provision | Did the agent see the right architecture/config/data convention? | Update the repo map/contract; link the exact analogous file. |
| Execution environment | Is the IDE SDK activated, dependency/runtime correct, and hardware state known? | Fix setup or preflight; do not alter experiment physics to mask it. |
| Verification feedback | Can success/failure be observed independently? | Add a safe check, positive control, raw-data check, or fit quality gate. |
| State management | Did a prior result lose provenance or become a false fact? | Label evidence, update PUCQ4 state, and log rejected approaches. |

## Recorded failure patterns

### QICK constant-pulse length must fit its hardware encoding

- Layer: execution environment / verification preflight (October 28).
- The current generator uses 614.4 cycles/us and a 16-bit pulse-length field.
  A 150 us spectroscopy pulse required 92160 cycles and was rejected before
  qubit acquisition. The already-tested 100 us pulse is 61440 cycles and fits.
- Check pulse length against the current firmware before acquisition. For this
  benchmark retain <=100 us and the unchanged QICK program. Preserve failed
  logs/partial plots, verify canonical HDF5 independently, and restart under a
  new label. Do not split pulses or use periodic mode to evade this limit.

### Delay quantization can resemble a Ramsey frequency error

- Layer: verification feedback (October 28 fresh benchmark review).
- M4's old nominal 2 MHz Ramsey sweep requested 0–3 us but recorded only
  0–2.9296875 us. Phase is programmed from the requested delay sweep, so the
  expected fringe frequency against recorded time is about 2.048 MHz. The
  observed 2.045 MHz is not evidence of a 45 kHz carrier error.
- Check the requested phase increment against actual delay spacing before
  applying a frequency correction. Preserve both axes/configuration and always
  verify fringe sampling against the measured frequency, not just the nominal
  detuning. This review does not change the QICK pulse program.

### A high fit R2 does not validate coherence beyond the recorded window

- Layer: verification feedback (October 27–28 fresh cooldown).
- M4's former short-coherence profile sampled only 5/3/5 us for T1/T2R/T2E.
  Fits returned about 7/12/15 us with R2 above .99, but none of these decays
  was resolved within its recorded window. The independent audit rejected them.
- Extend the acquired delays using the observed scale, retain adequate fringe
  sampling, and repeat the normal full round robin. Keep the short-window raw
  data, but omit its extrapolated coherence values from accepted tables.

### Coherent spectroscopy fringes can invalidate a Lorentzian linewidth fit

- Layer: verification feedback (October 28 fresh cooldown).
- M1/M3 20 us rectangular-drive spectra show repeatable fringes, despite clean
  Rabi and coherence data. A Lorentzian envelope gives plausible centers and
  widths but R2 around .83–.86. More averaging alone does not fix the model.
- Review the actual population curve. Test the exposed spectroscopy drive
  length/power using the unchanged QICK program, then remeasure Rabi and SSF.
  A longer-drive test is an intervention to evaluate, not automatic acceptance
  of a steady-state line or a replacement for a full validated pass.

### Broad spectroscopy fit can select noise in the weaker quadrature

- Layer: verification feedback (October 27 fresh-cooldown search).
- M1's 1 GHz raw-IQ scan contained a strong I response near 5515.43 MHz,
  but the normal reported frequency was 5655.12 MHz from the weak Q trace.
  A local independent joint-IQ fit and a lower-power repeat resolved
  5515.334706 MHz (0.08346 MHz FWHM, SNR 19.73). Do not promote the broad
  automatic fit or pass it into Rabi merely because the script completed.
- Review both quadratures and repeat candidates with resolved local sampling.
  A 0.25 MHz discovery step can undersample a 0.1 MHz line; an inconclusive
  coarse scan is not proof of absence. Keep raw data and rejected fit labels.

### Passing discovery pilots do not establish a stable mapping protocol

- Layer: verification feedback and state management (September 15 recovery).
- Dual-readout A-B-A recovered the lower line through both M6/M2 at -5.6566 mA,
  lost it through both at -5.652 mA, then recovered it. Nevertheless M6 return
  and initial cross-readout center controls failed. A detected line and a
  completed acquisition do not establish stable calibration.
- Same-total-shot counterbalanced short blocks passed M6 detection but failed
  M2 (held-out t=3.945<5), despite passing time-half center checks. Conditional
  fitted centers cannot override failed independent detection. Preserve every
  prospectively required block and the unchanged acceptance gate.
- Readout switching and shorter separation therefore did not solve this task's
  visibility hurdle. Do not retry maps, pool failed visits, increase averaging,
  or infer a microscopic dark-state mechanism from these results. Require a
  justified intervention and fresh paired visibility/repeat controls first.
- Between-run tracked M6 tone changed 9056.43 to 9056.38 MHz, while within-run
  tones were constant. Retain this covariate in any causal timing interpretation.

### Windows PowerShell text and atomic-publication pitfalls (September 15)

- Layer: execution environment. PowerShell5 Get-Content without Encoding UTF8
  decodes UTF-8 files as Windows-1252. A read/modify/write can corrupt existing
  non-ASCII text. Use IO.File.ReadAllText and an explicit UTF-8 writer for whole
  files. Repair only after proving the transformation against a preserved source;
  do not overwrite user edits from an assumed baseline.
- File.Replace invoked from PowerShell with a null backup path produced an
  invalid-path error. Use an explicit, fresh backup path and terminating errors;
  check source, destination and backup hashes before and after replacement.
  The September15 report publication passed with the explicit backup path.

### Extra averaging does not guarantee branch visibility

- Layer: verification feedback and state management.
- Central M2 candidate5444.489811 MHz had SNR4.2007 at32000 averages.
  Its targeted64000-average repeat found5444.498749 MHz but SNR3.7118;
  both lower controls passed and both upper references were clean. The feature
  remains a rejected candidate. Do not promote it because its fitted center
  repeats, pool visits/powers without justification, or assume square-root
  averaging improvement when the measured response does not show it.
- Retain the demonstrated power/averaging ceiling and explicit stop criteria.
  Report a readout/window/power visibility limit, not global mode absence,
  zero coupling, a diagnosed dark state, or an accepted minimum/J.
- Conditional whole-trace errors also omit temporal variation: saved M2
  repetition blocks at -5.74 mA span27.215kHz in the lower line. Blocks share
  their off reference, so report descriptive spans and preserve raw provenance
  rather than assigning independent drift significance or a cause.

### Recorded current versus requested current (verification, 2026-09-14)

GS210 source settings can differ from requested crossing currents by up to
0.000067 mA while remaining within the acquisition tolerance. At 235 MHz/mA
this is about 15 kHz, larger than some conditional line-center errors. Analyze
the saved `actual_yoko4_current_mA` setting, retain `requested_current_mA`, and
require finite matching drive-on/off settings. Do not relabel `SOURCE:LEVEL?`
as an independent ammeter measurement. Preserve raw data and original plot
titles; explain any requested-versus-recorded axis difference in the report.

### Reduced-power null spectra also lose the positive control

- Layer: verification feedback.
- In `ac_ctrlo`, gain 0.007 recovered neither the known lower line nor an upper
  line through M6 or M2 near recorded -5.656600 mA. Clean drive-off references
  do not establish sufficient excitation/readout sensitivity.
- Do not use the absent upper line to reject a branch model or infer zero
  coupling or a dark state when the lower positive control also disappears.
  Revalidate sensitivity at a demonstrated drive setting before interpreting
  an upper-window null. Keep unsuccessful lower-power measurements visible.

### A coupled measured row is mistaken for a physical qubit identity

- Layer: context provision and verification feedback.
- The PUCQ4 reference explicitly leaves physical identities for M1-M4 unknown.
  Its layout proximity and a response shared with Q6 do not independently map M2
  to physical Q5. Preserve measured-row labels in spectra and configuration.
- Require a design-to-measured-resonator routing record or a uniquely identifying
  measurement before assigning physical Q5. Separately record that the current
  qubit and ef software channels both use DAC1; frequency selection does not
  establish a second spatial drive port.
- Do not interpret a frequency difference between different-current single
  lines as a simultaneous branch splitting. The September14 endpoint recovery
  gives such different-current lines; use paired branches and the existing
  model-quality gates for J and resonance-current claims.

### A global I/Q projection hides a sparse spectroscopy line

- Layer: verification feedback.
- Symptom: the September 10 `ac_v2` wide M6 scan had a repeatable I response
  near 5434.644 MHz, but a detector using the first global I/Q principal
  component reported a maximum of only 3.00 noise units. The line occupied
  very few frequency bins, so noise elsewhere selected an almost perpendicular
  projection. All four chronological on-repetition blocks retained the feature.
- Repair: detect peaks with the norm of both smoothed, noise-scaled I/Q
  components, then fit the original signed complex data. The corrected score
  was 9.18 and the fitted vector SNR was 10.21. Retain the previous projection
  score for audit; do not fit magnitude data or weaken line-quality gates.
- Remaining limitation: the provisional FWHM was 0.1380 MHz on a 0.0500 MHz
  grid (2.76 samples per FWHM), so this candidate still fails the sampling gate.
  A finer acquisition is required before accepting its frequency or linewidth.
- Verification: a sparse I-only line in a wide noisy synthetic scan is now
  recovered, while single-line, two-line, reference, readout-pairing and drift
  controls still pass. Reanalysis of earlier task data found no additional
  accepted lines. Exact evidence and source copies are in
  `ac_review/progress_0102` and `ac_v2/.../documentation/analysis_p1`.

### A visible qubit line does not fit a whole wider scan cleanly

- Layer: measurement/analysis.
- Symptom: the September 8 M4 scan contained a resolved main line plus a
  high-edge response; its single-Lorentzian population fit had R2 0.559 even
  though same-pass coherent Rabi had R2 0.997. Centers also shifted from the
  earlier locator measurements. The cause of that additional response and
  frequency change was not established.
- Prevention: retain the complete scan and reject it as final single-line
  evidence. Acquire a fresh, explicitly bounded window around the resolved
  line, then repeat Rabi and SSF at the new fitted frequency. A plotting crop,
  a fitted center alone, or arbitrary population normalization is insufficient.

### A readout-length override is discarded by the selected-row builder

- Layer: configuration/verification.
- Symptom: a diagnostic requests a shorter or longer scalar `res_length`, but
  `build_state.add_readout_cfg` replaces it with the selected `res_length_ge`.
- Prevention: set the selected storage key as well as the scalar key, then
  assert the effective program configuration before acquisition. Persist both
  the requested settings and the actual merged configuration. Existing scripts
  that set only `res_length` need this check before a new length-dependent run;
  do not reinterpret historical measurements from a requested label alone.

### A positive-control fit resolves only one frequency bin

- Layer: verification.
- Symptom: the September 8 M3 control at 1000 averages yielded a nominal center
  near the expected frequency but its joint-I/Q FWHM collapsed to one acquired
  0.01998 MHz bin; the raw trace did not resolve a line.
- Prevention: reject the scalar and establish a stronger independent control
  before interpreting a target's null response. The M1 repeat at drive gain
  0.01/20 us and 2000 averages resolved a 0.17442 MHz line with joint-I/Q
  amplitude/noise 56.94. A control must pass the same shape and sampling checks
  as a new candidate; the fit returning a number is insufficient.

### Unsafe import starts work

- Symptom: importing an orchestration module creates directories or begins a lab
  loop.
- Root cause: substantial module-level execution in round-robin scripts and
  `QICK_experiment` construction.
- Prevention: inspect as text; the offline checker uses `ast` and never imports
  project modules.

### Python starts but NumPy cannot load DLLs

- Layer: execution environment.
- Root cause: direct use of the Conda interpreter without activating its runtime.
- Prevention: query the PyCharm SDK immediately before Python commands and use
  the activated invocation recorded in `ENVIRONMENT_AND_SAFETY.md`.

### A lost terminal process bypasses flux-source cleanup

- Layer: execution environment/safety verification.
- Symptom: a long acquisition stops without a Python traceback or final HDF5,
  while its manifest row remains `running`. The controller's `finally` block
  may not execute; during the September 4 continuous-qspec run, Yoko3 remained
  at +7.5 mA with output enabled after the attached process was terminated.
- Recovery: immediately query every involved source independently, ramp any
  energized source to 0 A at the bounded rate, disable output, and verify both
  current and output state before resuming. Treat the partial row as unusable;
  never stitch it into a replacement acquisition.
- Prevention: launch multi-hour hardware controllers as hidden detached
  processes with stdout/stderr redirected to durable logs, retain a PID and a
  resumable per-row manifest, and independently check source state whenever a
  controller disappears. On Windows, the detached interpreter must inherit
  the activated Conda DLL paths (including `Library\\bin`); merely invoking the
  configured `python.exe` is not an activated environment.

### Qspec shows ringing/oscillations

- Layer: context/configuration, then verification.
- Root cause: coherent sidelobes from a 20 us rectangular drive at excessive
  spectroscopy gain, not a need to replace the QICK program.
- Prevention: tune low-power config and inspect calibrated population/linewidth;
  keep `PulseProbeSpectroscopyProgram` unchanged.

### Qspec fit is outside the sweep or has negative width

- Layer: verification/analysis.
- Root cause: array index used as a frequency seed and unconstrained width.
- Prevention: seed with acquired MHz, bound center to acquired range, require
  positive width, and reject a scalar fit that conflicts with raw data.

### Resonator scan is unexpectedly slow

- Layer: context.
- Root cause: res spec sweeps frequency in Python; putting hundreds of averages
  in QICK `rounds` causes a network round trip at every point. Qspec instead uses
  a hardware frequency loop.
- Prevention: preserve equivalent averaging with res-spec `reps` high and
  `rounds=1`; do not generalize this rule to experiments with different loops.

### Plot exists but raw HDF5 does not

- Layer: verification/environment.
- Root cause: generated Windows path exceeded 260 characters; acquisition and
  plot completed before HDF5 creation failed.
- Prevention: preflight full generated filename, shorten labels, then separately
  assert plot and HDF5 existence before continuing.

### Physical Q4 is confused with displayed Q4

- Layer: task specification/state.
- Root cause: one-based UI names, zero-based list indices, resonator rows, and
  physical device numbers are not equivalent on PUCQ4.
- Prevention: identify all dimensions. Physical Q4 is M5/index 4; M4/index 3 is
  unassigned.

### Rejected M4 candidates reappear as facts

- Layer: state management.
- Root cause: a deck seed or one-point/high-power fluctuation was copied into a
  numeric six-row config and later read as measured truth.
- Prevention: `NaN` in measured-results state, explicit `seed`/`placeholder`
  labels where numeric arrays require a value, and the bounded stop rule in
  `PUCQ4_STATE.md`.

### Blind scanning consumes time without information

- Layer: verification/state.
- Root cause: repeated scans changed power/span but did not target a remaining
  hypothesis or include a positive control.
- Prevention: list the hypothesis each run can falsify, require a control, stop
  when plausible QICK-side explanations are exhausted, and request new external
  evidence.

### Single-shot “optimum” is unstable or poor

- Layer: task specification/verification.
- Root cause: accepting one grid maximum without repeated plateau behavior, or
  failing to re-optimize after upstream qfreq/sigma/readout changes.
- Prevention: alternate length and gain/frequency sweeps, repeat points, choose
  the shortest plateau length, and validate in the normal round-robin chain.

### Shared config and per-run overrides disagree

- Layer: context/state.
- Root cause: `system_config.py`, `pucq4_config.py`, optimization arrays, and
  orchestration overrides can all contain different stages of calibration.
- Prevention: inspect the actual execution path, label starting points versus
  accepted results, and report propagation with old/new/unit/evidence.

### A scalar qfreq seed becomes a QICK sweep object

- Layer: context/configuration.
- Root cause: reading `qubit_freq_ge` from the merged experiment configuration
  after the spectroscopy sweep is constructed can return a `QickParam` rather
  than the scalar device frequency needed by validation and SSF calibration.
- Prevention: capture the scalar qfreq seed from the untouched
  `QICK_experiment.qubit_cfg` before merging sweep settings; use the acquired
  frequency array only for the spectroscopy axis.

### Downstream Rabi uses a stale spectroscopy or readout frequency

- Layer: state management/configuration.
- Root cause: a wide qspec or local resonator measurement succeeds, but Rabi,
  SSF, or coherence inherits an older seed instead of the measured qfreq and
  resonator base. A flat Rabi trace can then be misdiagnosed as a pulse-width
  problem.
- Prevention: explicitly propagate the validated wide/local qfreq and resonator
  base through every downstream override, and log both values in each run before
  adjusting sigma or readout gain.

### Rabi metadata changes but the hardware gain axis does not

- Layer: configuration.
- Root cause: `AmplitudeRabiExperiment` constructs its `QickSweep1D` from
  `expt_cfg`; changing `rabi.config['start']`, `stop`, or `steps` afterward can
  relabel scalar metadata without rebuilding the already-created sweep object.
- Prevention: apply Rabi start/stop/step overrides to `expt_cfg` before creating
  `AmplitudeRabiExperiment`, then retain the post-construction scalar assignments
  only as matching metadata. Verify the saved gain array endpoints, not just the
  requested environment variables.

### A narrow raw-I qspec fit is not a populated qubit line

- Layer: verification/analysis.
- Root cause: at weak drive, noise or one quadrature can fit a narrow Lorentzian
  even though the SSF-projected excited-state population is negligible. This
  occurred at Q4 +5 mA, where a roughly 0.104 MHz raw-I candidate produced only
  about 0.04 population.
- Prevention: validate the center, FWHM, absolute signed SSF population, and
  local-background SNR together. When population requires power broadening,
  report that tradeoff explicitly instead of promoting the narrow raw fit.

### A report loader drops scalar coherence fits

- Layer: analysis.
- Root cause: a numeric decoder required at least two samples for every HDF5
  dataset, so scalar T1/T2 fit values were treated as unavailable even when the
  full delay traces and fits existed.
- Prevention: retain the multi-point minimum for swept traces, but permit one
  finite value when reading a fit-result dataset; verify both the heatmap column
  and summary-table cell after rebuilding the report.

### A qspec-only run discards the optimized readout offset

- Layer: configuration/state management.
- Root cause: `PUCQ4_RES_FREQ_MHZ` is a resonator-base seed. In a normal run,
  the enabled resonator sweep finds that base and then adds the configured
  readout offset. A qspec-only run has no resonator stage; applying the same
  base override after initialization silently replaces the already offset
  readout tone. The Q4 -10 mA common-grid attempt therefore used 9010.49 MHz
  instead of the validated 9011.44 MHz tone.
- Prevention: use the distinct `PUCQ4_READOUT_FREQ_MHZ` override for runs with
  no resonator sweep, print the physical qspec readout frequency, verify it
  against `res_base_MHz + offset_mhz`, and persist that actual tone with every
  common-grid row. Reject and reacquire any row made with the base alone.

### A report loader recursively parses a very large serialized numeric vector

- Layer: analysis/runtime.
- Symptom: the map builder can exit through `conda run` without a useful Python
  traceback; running with `-X faulthandler` reports a Windows fatal stack
  overflow in `ast.literal_eval` while decoding an HDF5 field.
- Root cause: common-grid qspec stores 62,501-value arrays as serialized numeric
  text. Passing that flat vector through Python's recursive AST literal parser
  exhausts the native stack even though the data itself is valid.
- Prevention: recognize flat numeric text and parse it with bounded vectorized
  `numpy.fromstring` before retaining `ast.literal_eval` only as a compatibility
  fallback for genuinely structured legacy values. Verify at least one full
  common-grid row through the report builder after changing sweep density.

### A multi-GHz calibrated qspec heatmap hides its narrow transition

- Layer: analysis/reporting.
- Symptom: each local qspec curve contains a resolved line, but the full-range
  heatmap looks like banded vertical noise or loses the line when rasterized.
- Root cause: independently acquired SSF calibrations shift the signed
  population baseline between bands, while a 0.1-1 MHz transition occupies less
  than one output pixel on a 2.3-3.3 GHz y axis. The stitched calibration seams
  can then dominate the visible map even though each local line exists.
- Prevention: when the hardware supports it, acquire one continuous frequency
  sweep and one SSF calibration per current. Preserve every saved signed
  population in canonical HDF5 and in companion local raw/smoothed panels. A
  full-band diagnostic heatmap may subtract a disclosed slow additive baseline
  from each row so calibration offsets do not consume the color scale, but it
  must never divide by row noise, peak height, range, or another row-dependent
  amplitude scale. For slide rasterization, apply a short fixed-frequency
  smoothing kernel, average ordinary display bins, and retain the local maximum
  only in a disclosed narrow neighborhood of each independently measured line;
  if necessary, repeat that peak bin over a fixed small number of output pixels.
  Use one fixed physical contrast scale for both devices, disclose any nonlinear
  color transform, and keep canonical HDF5 rather than the display-reduced PNG
  as the quantitative source.

### An assumed qspec point-count ceiling forces unnecessary segmentation

- Layer: configuration/runtime/verification.
- Symptom: a requested continuous multi-GHz map is divided into bands because a
  smaller prior program is the largest one known to have run; independently
  calibrated bands then create visible population seams.
- Root cause: treating a previously exercised 62,501-point program size as a
  hardware limit without first testing the requested monolithic grid. PUCQ4 Q6
  subsequently completed 82,501 points from 4000 to 7300 MHz at 500 reps in one
  run, proving that the assumed ceiling was not the active constraint.
- Prevention: test the largest requested grid first at a safe, verified current
  with the final acquisition settings. Segment only after a concrete compiler,
  buffer, or runtime failure. Verify finite point count, monotonic realized DDS
  axis, current-specific physical readout tone, exactly one SSF centroid pair,
  and adequate qfreq-to-edge margin before accepting a continuous row.

### A valid wide QICK sweep is rejected for not reaching the nominal stop exactly

- Layer: verification.
- Symptom: hardware and canonical HDF5 complete, but a post-run validator rejects
  the row because the last realized frequency is below the requested stop.
- Root cause: the requested step is quantized by the QICK DDS. For the validated
  82,501-point Q6 sweep, 0.040 MHz requested became 0.03998337 MHz realized, so
  4000-7300 MHz requested ended at 7298.62785873 MHz with every point present.
- Prevention: validate the start, finite point count, monotonicity, realized
  median spacing, bounded endpoint shortfall, and minimum transition edge margin.
  Persist the exact realized axis. If acquisition completed and only the offline
  validator was too strict, recover the canonical HDF5 from its deterministic
  log rather than rerunning hardware.

### A valid QICK Rabi gain sweep is rejected at its quantized endpoint

- Layer: verification.
- Symptom: the Rabi program, plot, and canonical HDF5 complete, but a strict
  validator rejects the row because a requested 1.000000 gain stop is realized
  as 0.99493377. Repeating the hardware acquisition would produce the same axis.
- Root cause: the QICK gain register quantizes the requested sweep. With 201
  requested points from 0 to 1, PUCQ4 realizes one common monotonic axis from
  0.00000000 to 0.99493377 with 0.00497467 median spacing.
- Prevention: validate point count, finiteness, monotonicity, common cross-row
  axis, realized median spacing, and a bounded endpoint shortfall of no more
  than two requested steps. Persist the exact realized gain axis. If only this
  post-run check failed, recover and revalidate the deterministic canonical HDF5
  and log instead of reacquiring the qubit.

### Ramsey endpoint contrast selects the weak quadrature

- Layer: analysis/verification.
- Symptom: the canonical Ramsey correction reports a plausible frequency but
  the saved quadrature has poor fit SNR, even though the orthogonal quadrature
  visibly contains a strong fringe. This occurred during the first Q6 -10 mA
  correction; validation stopped the campaign before the dependent Rabi run.
- Root cause: the legacy quadrature selector compares endpoint changes. Noise or
  phase can make the weaker quadrature win that local test even when its global
  damped-cosine fit is inferior.
- Prevention: retain the canonical result for compatibility, but independently
  fit both I and Q across the full Ramsey trace. Require bounded correction,
  physically plausible fringe frequency, fit quality/SNR, frequency standard
  error, and agreement between the selected canonical and strongest independent
  solutions. Never continue to Rabi from a rejected correction. A weak saved
  quadrature may be accepted only when the orthogonal fit is strong and all
  uncertainty/agreement limits pass; reacquire ambiguous rows.

### Detached `conda run -p` loses a Windows prefix containing spaces

- Layer: environment/process launch.
- Symptom: a hidden detached controller exits before hardware access because
  the Conda prefix under `C:\Users\Ma Quantum Lab\...` is split at the space,
  or compiled dependencies such as HDF5 DLLs cannot be loaded.
- Root cause: forwarding a quoted `conda run -p` argument list through
  `Start-Process` does not reliably preserve that prefix as one argument, and a
  bare Python launch does not automatically inherit all Conda DLL search paths.
- Prevention: resolve the PyCharm-configured interpreter first. For a detached
  controller, launch that exact Python executable directly and prepend its
  Conda root, `Library\bin`, `Scripts`, and related runtime paths to the child
  `PATH`. Redirect stdout/stderr to explicit files, retain a PID/status record,
  and independently verify both sources are safe when the controller exits.

### Pulse-dependent coherence rows do not share bit-identical delay axes

- Layer: analysis/verification.
- Symptom: a valid current map is rejected as having inconsistent xdata, or all
  rows are plotted against one substituted delay vector and become slightly
  misregistered.
- Root cause: the QICK-reported delay includes pulse timing. A current-indexed
  Gaussian sigma can therefore shift the first/last realized T1, T2R, or T2E
  delay while preserving sample count, monotonicity, and essentially the same
  step size.
- Prevention: validate every row for expected point count, finite increasing
  centers, requested bounds, and stable step size. For a 2D current map, derive
  midpoint edges separately from each row and draw one native-axis quad strip
  per current. Do not demand bit equality, silently reuse one row's axis, or
  interpolate canonical population merely to obtain a rectangular matrix.

### A higher-average coherence retry fits worse than the original

- Layer: measurement/analysis/selection.
- Symptom: increasing reps and rounds produces a lower saved-population R2 at a
  weak-readout current, even though the retry completed and saved normally.
- Root cause: more samples reduce sampling noise but do not remove source drift,
  current-settling variation, or changes in the independently acquired SSF
  projection. A later high-statistics curve is not automatically superior.
- Prevention: preserve the original and retry artifacts, validate each against
  the same saved-population model, and select per experiment by an explicit
  quality criterion rather than acquisition order. Record both R2 values and
  the selection. In the 2026-09-08 PUCQ4 retry, all Q6 -7.5 mA priors beat their
  2,000 x 4 candidates, while all Q6 -10 mA candidates improved and were used.

## Recording a new failure

Use the diagnostic template in `TASK_TEMPLATE.md`. Include the failed command or
run path, evidence, responsible layer, harness change, and re-run outcome. If the
root cause is still uncertain, record bounded hypotheses and the next evidence
that would distinguish them; do not convert the best guess into a rule.

## Reporting optional calibration metrics

- Layer: context / verification. A completed current-row calibration can carry
  an explicit null SSF metric while preserving its readout settings and raw
  calibration shots. A report builder that treats every metric as numeric fails.
- Render a missing optional metric as not recorded. Do not substitute a result
  from another operating point. Check required fit/data fields independently.
- PowerShell on this host also blocks dot-sourcing saved .ps1 helper files.
  Supply reviewed helper definitions inline to the terminal command; do not
  change the machine execution policy. Verify that a failed launch actually
  started no hardware process before retrying.

## Spectroscopy population calibration persistence

- Layer: verification / state. The historical round-robin qspec schema saved
  raw mean I/Q and Population but omitted its four SSF calibration arrays.
  An offline checker must inspect the schema before claiming full population
  reconstruction. Rabi and coherence already preserve these arrays.
- From the September 8 M4 coherence follow-up, round robin also saves the
  existing qspec ss_I_g/ss_Q_g/ss_I_e/ss_Q_e arrays in canonical HDF5.
  No pulse program changes are needed. Require finite equal shot counts and
  reconstruct the affine population from raw IQ and centroids for final data.
  Label older files as lacking HDF5-only calibration reconstruction.

### Coherence SSF override order

- Layer: verification / orchestration. In the M4 pilot,20k/250us coherence
  calibration overrides were applied before Rabi/SSF, then overwritten by
  their50k/100us settings. The acquired50kshots were complete and usable,
  but the request and actual calibration differed. Coherence SSF overrides
  now apply immediately before the coherence phase. Keep preparation-shot
  counts and actual relaxation settings in the acquisition envelope, and
  require final raw calibration counts to match it. Do not alter old data.

### Echo plot title carried a Ramsey label

- Layer: reporting / verification. The T2E plotter formatted its fitted echo
  time as `T2R` inside a title beginning with `T2E`. Identify the experiment
  from its canonical dataset, pulse sequence and configuration, not that
  legacy title string. The two plotting labels now say `T2E`; no pulse code
  changed. Preserve acquired images and explain the typo in report captions.

### PowerPoint slide replacement

- Layer: verification. Replacing slides by adding and then removing slide IDs
  can leave live parts whose names are later reused by python-pptx. The ZIP then
  contains duplicate slide XML entries even though a slide-count check passes.
- Renumber live slide/notes part names before saving, require unique ZIP entries,
  reopen the saved file, and render representative pages with native PowerPoint.
  Preview to a temporary directory before updating an active characterization.

### Population-fit initialization on sub-MHz sweeps

- A default half-width seed of1MHz exceeded the realized0.999069MHz span in
  line_6c, causing x0 infeasible despite a clear raw line. Clamp the initial
  half-width inside the existing bounds; do not alter the pulse program or
  force an output linewidth. Regression: narrow synthetic lines and prior
  accepted raw data. Final acceptance still requires independent fits.
- RR can skip collection after a fit failure yet write placeholder HDF5 files
  and finish without an exception. File existence and script completion are
  insufficient: require finite raw arrays, population reconstruction, accepted
  fits and all requested experiment fields. A plot-only/placeholder run stays
  diagnostic and cannot populate result/configuration tables.
- From the linewidth follow-up, RR collects completed res/qspec arrays before
  checking whether qspec fitting permits continuation. Normal end-of-round
  Data_H5 saving then preserves diagnostic raw data even on a rejected fit.
  A temporary-HDF5 replay verified the failure branch without importing RR.

## Successful temporary plans do not make the default benchmark reproducible

- Symptom: a saved low-gain qspec/coherence curve succeeds, but normal RR omits
  that row or uses a different averaging depth, delay range or population mode.
- Layer: state management / context provision / verification feedback.
- Evidence: September8 all-six reproduction audit found M4 excluded, coherence
  scaling off, and M4/M6 successful short-delay/high-average settings only in
  temporary plan files, despite correct persistent pulse/readout calibrations.
- Correction: persist acquisition profiles alongside the shared pulse/readout
  tables; apply a fresh complete profile for each row/current before experiment
  construction. Save the selected profile and environment overrides with the
  run. Deep-copy Exp Config when collecting a row so later scalar changes cannot
  rewrite prior rows' metadata. Validate normal RR with no acquisition overrides.
- Verification: offline selector coverage and unchanged program/class hashes
  are necessary but do not replace canonical raw-HDF5 and plot inspection.

## Compare like measurements and retain actual sweep timing

- Layer: verification / context. A frequency used for a later Gaussian-pulse
  calibration may differ from the preceding rectangular-pulse qspec center.
  Compare fresh qspec with validated archived qspec; show the pulse-frequency
  difference separately. Do not treat unresolved historical fit candidates as
  ground truth or move a clean new line to reproduce an old table number.
- September8 investigation: archived50-point0--1 us Ramsey corrections used
  nominal4 MHz phase detuning but actual0.01953125 us delay steps. Independent
  phase/time sweep quantization can change phase slope versus actual delay.
  Before interpreting a Ramsey correction, check the actual phase increment
  divided by actual delay increment, not only the nominal detuning parameter.
  Any correction inferred from incomplete archived phase metadata remains a
  diagnostic candidate until a fresh measurement confirms it.
- Older coherence HDF5 may contain only I/Q and Fit, without Population or
  state-calibration shots. Label a normalized quadrature diagnostic as such;
  do not present it as calibrated population. Preserve the original RR figure.

### Identical readout settings can have different SSF contrast

Observed during the September9 PUCQ4 M3 reproduction: identical saved scalar
settings produced.6924 and.52705 contrast. Raw ground clouds were stable while
the prepared-excited cloud moved. Treat the cause as unresolved until an
experiment distinguishes it. Compare actual state shots, repeat candidates,
and select a length plateau using between-run variability. A lone optimization
maximum is insufficient; validate the selected point in a fresh full RR chain.
Source: PUCQ4_round_robin_validation_20260908_artifacts/M3_readout_repeat_variability.json.

### A reused readout offset can lose contrast after a fresh resonator scan

- Class: experiment configuration / readout frequency reference.
- Evidence: M5 atYoko3+2.5 mA initially passed qspec/Rabi/coherence checks but
  SSF fell to.4706 versus prior.615. Its measured dip9014.64 MHz plus the old
  -.48 MHz offset placed readout at9014.16 MHz. Two alternating length and
  gain/frequency grids chose4 us/.95 gain/9014.76 MHz; a fresh dip9014.62
  at that gain/length established offset+.14 MHz. A full normal RR then found
  dip9014.68, applied9014.82 MHz readout and reproduced SSF.5841 with all
  measurements passing. Frequency/readout/pulse values were saved together.
- Correction: inspect the absolute tone, recheck length and gain/frequency,
  then derive the offset from a resonator scan at the selected gain/length.
  Validate the resulting offset through the complete RR sequence. Do not
  attribute a changed resonator minimum to a physical cause without evidence.
- Source: PUCQ4_round_robin_validation_20260908_artifacts/M5_p025_readout_decision.json.

September9 orchestration correction: normal RR now preserves the saved absolute
readout frequency after measuring a new resonator dip, and records the offset
relative to that dip in the experiment configuration. Explicit base/offset
diagnostics retain their meaning; an absolute readout override takes precedence.
M6/Y4+2.5 showed the same reference mismatch (9059.94 intended,9059.58 applied).
Its alternating readout scans selected9059.98MHz/.80/22us; full-chain verification
is required after this correction. Ninety parsed configuration cases checked
all six rows, shifted dip values, normal and explicit-override behavior offline.

## A comparison threshold is different from measurement quality

- Layer: verification and state. Keep raw/calibration/model checks separate
  from comparison to an older value. A clean coherence measurement can lie
  just outside a 30% historical band while an independent, matching-setting
  repeat lies just inside it. If their uncertainties or the predeclared repeat
  tolerance agree, record both values and the band overlap; do not describe
  this as a confirmed persistent change. M2 at Y3+10mA on September9 showed
  T1 12.056/10.621us versus8.767us, while its Ramsey15.607/15.706us pair
  independently confirmed the difference from11.708us. A structured-model
  failure (M1 Ramsey revivals) cannot be resolved by this comparison review.
- A fixed qubit's own current_mA field may be0 even while another Yoko is
  nonzero. Store such reviewed controls under both actual source currents;
  never promote them into zero-current calibration merely from current_mA=0.
- Before claiming a repeat used the same bias conditions, retain explicit
  post-ramp settling settings and measurement order. Q4/M5+10mA centers
  differed between a scan following M1/M2 controls and a prompt scan. A600s
  settling check is in progress; no cause is assigned from those two scans.

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

## Slow Ramsey fringe can confuse automatic fit initialization

- Layer: verification / configuration. A clean, low-frequency Ramsey trace may
  be missed by the existing automatic FFT starting estimate. Compare the saved
  curve with an independent fit of the same full-window model before treating
  a short returned tau as physical loss of coherence.
- At M6/Yoko4 +10 mA, a measured midpoint phase .015398310214337 MHz gave a
  slow ~.04 MHz fringe. One full pass returned 2.655 us automatically, whereas
  the independent fit and the unchanged public t2_fit(guess={'f': .04}) both
  returned 5.768 us with R2 .980. The frequency is an initial value only;
  frequency and tau stay free. It does not waive a two-component model failure.
- RR now routes optional fit_frequency_guess_MHz from its fresh per-row profile
  into the existing fit/plot methods, preserves Fit Unseeded plus its tau/error,
  and records the fitted frequency. None is explicitly restored for other rows.
  No class, QICK program, model or raw data is modified. Validate this path on
  fresh complete RR output; offline refits remain diagnostic evidence only.

### Ramsey starting values versus changing measured decay (2026-09-09)

Verification/state failure: do not assume every disagreement with a prior coherence time is another bad fit. At M6/Y4+10, calibrated population starts repaired low-fringe local minima on prior raw traces, but a fresh trace then gave a stable shorter 2.805 us decay. Thirty free-fit starting combinations per trace confirmed <.001 us near-minimum variation; changing a physical T2 starting value did not recover the old time. Preserve both automatic and configured curves, compare independent full-window fits, and repeat the acquisition with a freshly measured carrier before assigning a physical change. Never tune the reported value toward the reference. The unchanged public T2 guess API has double x-normalization in physical units; offline physical tau starts used tau_us/dot(delays,delays). No persistent T2 guess or class change was introduced.

### Held-bias diagnostics and elapsed-time provenance (2026-09-09)

Verification/state: separate variation within one held current from differences
between independent visits. M5/Yoko3 -2.5 mA had two visits separated by
0.235234 MHz, but three subsequent spectra acquired under one held bias spanned
only 0.014723 MHz and passed consecutive repeat comparisons. This does not
establish a need for extra settling or a constant shift across visits. Keep all
three canonical batches and plots; do not certify a multiple-round diagnostic
as final one-pass calibration. Use timestamps from one acquisition-PC clock for
elapsed intervals: the network-drive bias-file mtime differed by 1.94 s from the
PC log clock. Preserve earlier analysis versions when correcting time labels.

Environment: PowerShell array concatenation inside Start-Process ArgumentList
can insert spaces into quoted paths. Build one quoted argument string with the
format operator for known fixed paths, launch hidden, and check the controller
stderr plus the expected result file before assuming an acquisition started.

### Carrier overrides must agree with spectroscopy bounds (2026-09-09)

Configuration/verification: the RR profile is applied before the optional
PUCQ4_QFREQ_MHZ carrier override. Previously, that override changed the pulse
carrier while leaving qspec bounds centered on the shared seed unless explicit
bounds or a half-span were also supplied. The M6 -2.5 mA full b run therefore
actually scanned6102.736072740--6104.736072740 MHz; its6104.052944660 MHz line
remained inside the acquired window. Preserve those actual bounds and launch
sources. RR now recenters the profile span on an explicit carrier, while
explicit start/stop or half-span retains precedence and ordinary defaults are
unchanged. The actual AST selector passes seven offline cases, including row
isolation and rejection of incomplete bounds. Exact-repeat plans should always
copy recorded start/stop; never reconstruct them from a presumed seed.

### Audit shared calibration settings in acquisition order (2026-09-09)

Verification: qspec, Rabi, SSF and coherence stages share Readout_Optimization.
A qspec calibration override remains active for Rabi unless Rabi replaces it;
the later SSF override then feeds coherence unless a coherence override exists.
An audit that independently falls back to the row profile can falsely flag a
valid Rabi shot count. M6 -2.5 full b requested/acquired20000 shots per state,
but the old audit inferred10000 for Rabi. The audit now follows enabled stages
in normal RR order and records each setting's source, retaining exact count
and population-reconstruction checks. Eleven precedence cases, all five actual
full-b calibration records, and65 records from13 selected full passes agree.
The original audit is preserved; the corrected audit still rejects its noisy
echo and flags lower SSF. No acquisition, class or raw data was changed.

### Slow Ramsey fringes: check identifiability before accepting a short-window scalar

Context/verification: a plausible decay and small local covariance can still come from an insufficient window or a local minimum. M1 archived-pair simulations show that a60us window can return~15us for a34us synthetic decay. A120us window plus existing calibrated-population fit starts resolves this design case; it is not hardware validation. Keep full raw windows, compare independent fits, and validate on fresh hardware. Optional PUCQ4_T2R_FIT_STARTS supplies only existing free-frequency/phase/amplitude/offset starts on one row and never a T2 constraint. The original automatic fit remains saved. Exact repeats copy recorded starts, not current profile guesses.

September 9 repeatability evidence: two predetermined M1 records at the same
carrier, phase, pulse, readout and180us/901-point window give31.038+/-0.774us
and19.474+/-1.153us. Automatic, configured and independent fits agree within
each record; both individual raw/model gates pass. Their11.565us difference
exceeds the unchanged4.167us repeat limit. This is verification feedback about
measurement repeatability, not evidence that another fit initialization resolves
the issue. Retain both records, withhold promotion, and report the unexplained
variation. Do not infer a stable new coherence value or a physical cause from
either record. Evidence: validation artifacts/M1_frame180_repeat_review.json.

### SSF repeat variation: retain the raw timing and inspect shot blocks (2026-09-09)

Verification/state: matching Q6 -2.5 mA pulse/readout settings yielded SSF0.2795 and0.3335 with20kshots/state. The difference0.054 exceeds the predeclared0.03repeat gate; a complete pass with good coherence is not enough to waive that finding. The unchanged histogram is reproduced exactly, and unbinned rotated KS gives0.27955/0.33430, excluding histogram bins as the explanation. Twenty1000-shot blocks show excited-state mean variation beyond within-block counting error. Preserve all records, avoid claiming an unidentified physical cause, and test a predetermined set of longer records before changing persistent averaging. Do not select just the largest contrast or repeatedly test until a favorable pair appears. Ground/excited shot blocks are separate acquisitions. HDF5 Dates may be assigned when the entire round is saved; use the normal SS plot timestamp or RR log for the actual stage time, and label HDF5 timestamps explicitly. No class, raw file, or acceptance threshold changed.

### Detached cmd launch quoting differs from an interactive PowerShell command

Environment: a Start-Process ArgumentList containing doubled quotes around the
Conda paths reached cmd literally and failed before Python started on September9.
Preserve the failed controller log and confirm no acquisition process/result was
created. A reviewed temporary .cmd file with ordinary quoted paths, explicit
activation-error handling, and the configured interpreter launched correctly.
Start it hidden from its own directory with simple launcher/plan filenames;
verify the guard result and child PID before treating acquisition as started.

### Exact coherence repeat returns inside the old band but disagrees with the full pass
M3 final-default T1 on2026-09-09 was31.460196+/-0.331662us; its one exact-setting repeat was37.915038+/-0.512876us. Both raw/model fits passed, but6.454842us exceeded the predeclared5.203143us repeat limit. The repeat returning inside the historical comparison band does not validate the original outlier or authorize selecting the repeat as the complete-chain result. Retain both records, the original comparison finding and prior selected calibration; assign no physical cause from this evidence alone. Artifact: final_default_confirmation_review.json in the round-robin validation artifacts.

### Resolved qspec peaks disagree after returning to the same current
Q4/M5/Yoko3 -5mA on2026-09-09 produced centers6725.225762893 and6725.099256569MHz in exact-setting full/repeat spectroscopy. Both line-quality checks passed, but.126506324MHz exceeded the.015665516MHz repeat limit. A clean peak and a full pass with good coherence do not establish a reproducible static frequency. Preserve both peaks and the reference comparison, keep the point unresolved, and do not promote a favorable line or assign a physical cause without additional evidence. Source: last_bias_pass_evidence.json in validation artifacts.

### Local PowerShell script-file execution is disabled
On this host an unsigned archive_closeout.ps1 was blocked before executing. Ordinary authorized file operations remain available as native PowerShell commands. Use Copy-Item/Get-FileHash directly for the archival task, verify every copied hash, and retain the originals. Do not change execution policy just to run a maintenance wrapper. The September9 validation archive completed1460files this way; see its archive_index.json.

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


## Structured spectra and effective coherence after a warmup

Readout override provenance: during SS-only diagnostic stages, an explicit
PUCQ4_READOUT_FREQ_MHZ sets the actual pulse tone in res_freq_ge, but the
inherited res_freq_offset_ge can still describe the old baseline. Analyze the
actual tone used by the program; derive a physical offset only from a resonator
measurement at the matching readout gain. Final full RR includes res spec and
recomputes the offset. Never relabel the inherited SS-only offset as a newly
measured resonator reference, and do not rewrite raw artifacts to hide it.

October29 follow-up: M6 was subsequently found near6404.64MHz, with resolved
raw-IQ repeats at several readout offsets and a coherent Rabi (R2.992887).
The initial successful tone9057.60MHz lies below the high-power dip9057.72MHz;
the previously focused6447.49-6597.49MHz search did not include the new carrier.
Original VNA seed6401MHz remained available in pucq4_config.py. A historical
post-bias carrier6522.49MHz must not displace all other seed evidence after a
warm cycle. Broad earlier lower-power nulls still do not establish a physical
absence. Search both resonator slopes and relevant seed regions; vary drive
power/length within the unchanged pulse program, verify multi-point I/Q shape,
repeat locally, and require coherent Rabi before accepting the line. Because
readout, search bounds and drive settings differed, do not attribute discovery
solely to one offset change or infer a microscopic cause of the carrier shift.

Layer: verification feedback / state management. October28 M2 two-component
qspec repeated in two independent full passes; two-frequency Ramsey improved
R2 .870->.956. M5 final Ramsey also beats (R2 .863->.970 with two frequencies).
Both separations are about138kHz, so a common instrument/system contribution
is not excluded. Do not identify an intrinsic/device mechanism from these fits.
Retain normal round-robin plots and effective fit values, with supplemental
model comparisons and explicit qualifiers; do not silently substitute another
model's envelope or force a single clean line.

M4 matched controls and M5 separate full passes support relaxation curvature.
A stretched exponential can improve residuals while extrapolating population
above1. Such a fit is phenomenological, not a validated physical initial state
or microscopic relaxation rate. Do not force Echo<=2T1 by changing fit offsets.

M5 bounded gain tests showed a linewidth/population tradeoff and the final
pass still missed preferred qspec targets. Record actual values and the miss;
review-complete does not mean nominal-calibration criteria passed. Preserve
latest qualified results rather than selecting earlier larger lifetimes.

M6 resonator visibility did not establish a qubit line. Broad4300-7300MHz and
focused high-gain-edge6447.49-6597.49MHz scans remained noise-like. Maxima of
3-4 noise units across hundreds/thousands of points are not accepted carriers.
Keep historical qubit/pulse values explicitly marked as seeds, and never run
Rabi/coherence as though they were confirmed after the warmup.

## Windows status-file sharing failures must not abort healthy acquisition

November11 review: November4 Q2 24h run stopped after44m20.5s/2974 saved
rounds because os.replace(ssf_status.json.tmp, ssf_status.json) raised
PermissionError WinError5. The status remained running/2973 despite a failed
controller. Raw HDF5 and PNG counts plus receipt/log are authoritative;
never infer success from stale status or assume missing raw data from status lag.
The precise original lock owner is unknown. Classified execution environment /
state management. Supplemental JSON now uses unique temp names, bounded retries,
nonfatal OSError handling, and a local mirror for the watchdog. This does NOT
make raw-data failure nonfatal or authorize automatic restart on hardware faults.
Actual3s Windows read-handle lock reproduced WinError5 on the restarted run;
rounds173-175 persisted, local heartbeat advanced and network status recovered.
Future controllers must tolerate transient status reads/copies too. Preserve the
original failed run/status/tmp as evidence. Claim24h complete only after timer,
clean exit, receipt and data coverage checks; a restart is not completion.

## Word XML namespace and notebook export constraints

November16 packaging: rewriting word/document.xml with ElementTree can remove
unused namespace declarations while leaving mc:Ignorable="w14 wp14" references.
NativeWord then reports corruption even though XML/ZIP parsing succeeds. Remove
stale Ignorable references or preserve the original namespace declarations, and
open the generatedDOCX in nativeWord before delivery. A validZIP is insufficient.
PyCharm notebook tools may only accept project-relative paths, even when another
workspace path is writable. Create/edit notebooks through IDE withinproject, then
package their completed bytes. A QUEUED result with kerneldown is not execution;
interrupt that task's queuedcell and record runtimeverification as unavailable.
## Short Ramsey correction can select noise instead of Q2's carrier

November16 Zeno preflight measured qspec5438.366367MHz and pi .696454,
but the existing 20rep/2round,1us Ramsey correction fit a noisy high-frequency
component and moved the carrier to5430.904935MHz. The next Rabi returned
pi .562138; stopped the owned acquisition before Zeno, preserving raw data.
Layer: verification feedback/calibration state. A returned fit is not a
validated correction, especially on beating Q2. The requested Zeno script
now keeps this short correction diagnostic-only and retains fresh qspec's
carrier; the second Rabi still runs. It requires final pi .6-.7 before Zeno.
Zeno pulse programs and sweep grids are unchanged. Any later re-enabling of
automatic Ramsey retuning needs adequate SNR and agreement with spectroscopy.

## Zeno maps can lose contrast despite a working baseline

November17 Q2 diagnostic found three differences from baseline: Zeno population
plots used absolute IQ distance rather than signed projection onto the measured
g/e axis; the qspec wrapper hard-coded a 0.5 us reset despite configured reset;
and a shared 5 us timing setting also delayed readout after the last qubit pulse.
Fresh short-delay measurements recovered the qspec line and early coherence
fringes, whereas increased averaging/projection correction alone did not.
Layer: verification feedback and configuration context. Inspect raw signed IQ,
actual reset, and post-pulse timing before declaring a missing qubit or TLS.
The three active Zeno programs now accept optional readout_ringdown_delay,
defaulting to readout_pulse_delay, so ring-up can remain unchanged. Shortening
both together changes the Zeno preparation and is not an equivalent comparison.
Check low/high Zeno gains at the actual Zeno-tone frequency and include the
shifted line in sweep bounds. Noisy g/e references can also distort the scale;
retain signed values and improve reference statistics rather than clipping.
Restored fringes do not establish two-frequency beating or its TLS origin.

November18 full-run review supersedes the short-check restart conclusion:
the first full qspec maps had a line, but later raw IQ lost it. The run kept
the startup carrier and pi amplitude for all subsequent maps. Its saved g/e
mean separation fell from about0.23 to0.013 (0.65 standard errors in round3
T1), making population normalization meaningless. T1 also still had the old
5us final wait/magnitude projection and a30-point100us grid; those omissions
were not covered by the prior three-class diagnostic. The longitudinal script
now refreshes/validates normal qspec and Rabi before each map, retains those
calibrations in canonical HDF5, and rejects unresolved references after saving
the map. T1 gets the same optional final wait/signed projection and local
baseline-like delay/count settings. Do not claim a fresh frequency solely
because a bounded fit returned one: November18 wide/narrow noise candidates
around5436MHz were rejected by raw review and Rabi. As of this entry the
fresh baseline itself has weak contrast; the hardware search remains pending.
This is verification/state failure, not a demonstrated microscopic TLS cause.

November18 afternoon exact-single comparison: expt_config.py and system_config.py
are byte-identical to the working November17 snapshots. Replaying the existing
startup sequence gave qspec R2 .014 and Rabi R2 .024, versus prior .476/.906.
The script accepted noise fits and propagated a repeated pi .179 into coherence
before its late pi-window check. Fixed prior carrier/pi/RO also failed to recover
singles; lower readout power and reapplying the configured DAC Nyquist mode did
not restore contrast. Wide short-pulse spectroscopy found only a weak broad
candidate near5434-5436MHz, and normal qspec/Rabi there did not validate control.
Do not report a new qubit frequency or TLS mechanism from this candidate.

Startup now uses20k g/e reference shots/state and, AFTER each raw QSpec/Rabi
save, checks reference SNR>=5 and population-fit R2>=.3/.75 respectively. The
existing pi .6-.7 window is also checked immediately after each startup Rabi
save. Actual main-script validation stopped at baseline qspec SNR1.217 with raw
data preserved; a diagnostic STOP sentinel also prevents accidental 2D entry.
Saved-data replay accepts prior real calibrations and rejects fresh noise.
This proves the rejection guard, not recovered qubit control. The physical
cause remains unresolved; confirm fridge/RF drive state, then recover singles,
validate actual zero/highgain2D and full50gain maps before another72h start.

November19 cross-row recovery: preserve the distinction between a conservative
screening stop and a physically absent line. M3 coarse/fine spectroscopy repeated
within7kHz; the fine line was sampled at about19points/FWHM with raw R2.674,
matched-filter score19.4 and g/e-mean SNR24.2, but its per-point vector SNR6.41
missed the task's initial8 threshold. The original stop/log/raw were retained.
Recorded scientific review allowed a diagnostic Rabi using the repeated resolved
line, without claiming that the original threshold passed or promoting config.
The independent Rabi then gave R2.937 and pi.65168 versus prior.65666. Do not
weaken the longitudinal startup guards or treat arbitrary low-SNR fits as
calibrations based on this example. Review repetition, actual sampling, raw IQ,
and independent coherent control together. Low-power population traces may be
noisier and their absolute normalization can overshoot; preserve signed values
and qualify conditional fits rather than clipping or substituting old results.

November19 later recovery: Q2 also repeated a resolved5438.7702MHz line and
independent RabiR2.975/pi.67158 with existing sigma.101us. Its initial old-carrier
references had already returned (SNR17), so frequency retuning alone does not
explain the prior loss. Five actual three-gain Zeno maps then saved with healthy
references (SNR18.6-25.3) at the script's offset tone and5us/.5us timing. Dark
T1/Ramsey/Echo controls match the fresh singles within their noisy uncertainties.
Higher-gain spectra shift/broaden and fringes weaken; a weak high-gain sinusoid
fit is not evidence that the whole map acquisition is blank. Review zero-gain
controls, early-time traces, raw IQ and references before changing code. The
fitted/averaged center printed for a gain-dependent map is not automatically
the zero-Zeno carrier. No microscopic TLS cause or reproduced beating follows
from these checks. Three-gain success is not full50-gain/long-duration validation;
the native one-cycle full-grid check must pass before the queued72h acquisition.
