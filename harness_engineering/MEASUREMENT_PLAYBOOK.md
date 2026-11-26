# Persistent qubit measurement playbook

User policy recorded 2026-11-20. Mandatory for measurement work in this
repository, including future chats. Read the safety rules and device state first.
This is a procedure, not authorization to operate hardware. Parameter tuning is
allowed within an explicitly authorized run envelope; ask only when required
bounds or device information are missing or the envelope must change.

## Reuse and provenance

Use existing section classes, QICK programs, plotting definitions, calibrated
population axes, fit conventions, and orchestration. Adjust experiment config
and sweep/x-axis bounds to resolve measurements before changing implementation.
Do not create replacement acquisition or plotting frameworks. If the user asks
for a new measurement, copy the nearest existing experiment and preserve the
class/run interfaces and phase order in EXPERIMENT_CONTRACTS.md.

Keep `<data root>/<run>/<device>/<study>/<sub-study>/<timestamp>/` with
`optimization/`, `study_data/`, and `documentation/`, and raw batches under
`study_data/Data_h5/<dataset label>/`. Descriptive names may change; structure,
HDF5 saving helpers, metadata, and filename provenance must remain. A request
for a new measurement alone does not authorize another storage architecture.
Reuse existing plot locations and link/embed their plots in run documentation.
Never overwrite raw data. Verify both plots and HDF5 after each acquisition.

## Ordered calibration and acceptance gates

1. **Resume and map.** Open docs/MEASUREMENTS.md, the latest handoff, device state,
   and source configs. Resolve physical qubit, resonator, and Python index.
   Record accepted values, their evidence, uncertainties, bias, and applicable
   readout chain. Seeds and stale settings are starting hypotheses only.
2. **Find resonators.** Locate resonances at sufficiently low readout gain to
   avoid frequency shifts/punch-out. Use existing resonator spectroscopy plots.
3. **Measure punch-out.** Acquire a resonator frequency-versus-gain map within
   authorized bounds. Identify the highest measured gain before a reproducible
   resonance shift toward the bare resonator, or other nonlinear behavior.
   Record frequency uncertainty, onset evidence, and a conservative accepted
   ceiling below onset. If onset is unresolved, use a validated lower ceiling;
   do not assert an unmeasured maximum. This ceiling bounds readout optimization
   for the same bias, length, and RF chain; revalidate when those change.
4. **Find and narrow qspec.** Start with an authorized high qubit-drive gain for
   discovery, then reduce to the lowest gain producing a reproducible resolved
   curve without excessive averaging. Require FWHM <= 0.5 MHz; the existing
   PUCQ4 target remains 0.1-0.2 MHz and calibrated peak population 0.3-0.5.
   Zoom the acquired sweep and plotting range and increase frequency resolution
   to put several measured samples across the linewidth. Record linewidth,
   fit uncertainty, repeatability, drive gain, reps/rounds, and elapsed time.
   A narrow fit to an unresolved peak is not acceptance. If no reliable narrow
   line is attainable, record the failure rather than silently relaxing the gate.
5. **Calibrate Rabi.** Use the measured qubit frequency, find pi amplitude, and
   adjust Gaussian sigma in system configuration. For PUCQ4 retain just over
   half an oscillation and pi amplitude 0.6-0.7 DAC gain. Re-run after frequency
   or sigma changes and propagate measured pi amplitude to shared settings.
6. **Optimize SSF.** Require fidelity strictly above 50%; this is a minimum,
   not the optimization target. Alternate length at fixed gain/offset (choose
   the shortest plateau length) and 2D gain/frequency-offset sweeps at that
   length. Never exceed the punch-out ceiling. Repeat to a reproducible plateau
   and validate the chosen point independently to avoid selecting noise.
   At <=50%, diagnose preparation, labels, readout, and calibration and
   reoptimize; do not accept or conceal the result by relabeling it.
7. **Propagate and minimize acquisition time.** Put accepted readout frequency
   offset, gain, length, qubit frequency, sigma, and pi amplitude in applicable
   shared configs and top-of-script variables for every affected active script.
   Inventory overrides explicitly; do not overwrite unrelated devices or
   historical run snapshots. Record file/key, old -> new, units, index, bias,
   evidence, and downstream invalidations. Tune reps and rounds first for
   resonator spectroscopy, then every measurement: compare lower averaging
   against a reliable reference using repeated acquisitions, fit uncertainty,
   visibility and total elapsed time. Define tolerances before selection;
   choose the fastest reproducibly reliable setting and persist it in configs.
8. **Validate coherence.** Measure T1, T2R, and T2E with appropriate time ranges
   and reliable decay/fringe fits. Require at least eight acquired points per
   fringe in T2R/T2E: for fringe frequency f in MHz and actual evolution-time
   spacing dt in us, abs(f)*dt <= 1/8. Account for echo time-axis conventions.
   Use measured fringe frequency, not only nominal detuning; adjust configured
   detuning, sweep spacing, and extent to resolve useful fringes and the decay.
   If no fringe exists, document why the criterion is inapplicable; do not
   manufacture a frequency or count interpolated plot points as samples.
9. **Set relaxation.** Rabi, SSF, T1, T2R, and T2E use a relaxation delay of
   5-10 times measured T1 for that qubit/bias. Before T1 is known, use a
   documented conservative previously validated estimate and revisit it after
   T1 acquisition. Qspec may use shorter relaxation if its curve remains
   reliable. Do not trade reset quality for speed; record the chosen multiple.
10. **Flux characterization.** For documented tunable qubits, repeat this
    sequence at every selected bias within characterized and authorized limits.
    Choose and record steps using observed frequency slope/linewidth and
    resonator tracking; refine near rapid changes or crossings, never guess an
    unsafe universal step. Save bias-indexed calibration, readout ceiling,
    sigma/pi amplitude, delays, reps/rounds, and uncertainties in shared configs.
    Validate interpolation before reuse; no extrapolation. Preserve required
    current-frequency, current-gain, sigma, and coherence maps and device reports.

## Recovery and continuous maintenance

Missing qspec does not imply a missing qubit. Check the resonator mapping,
readout resonance and offset, gain/length, punch-out ceiling, current/bias,
and prior good evidence before increasing drive or averaging. Recover the
readout point, then retry qspec. Broad lines require drive reduction and a
resolved sweep. Poor SSF requires rechecking preparation and the readout loop.
Frequency/sigma/pi changes invalidate downstream preparation/SSF calibration;
readout-chain or bias changes invalidate the applicable readout evidence.
Repeat the affected stages and validate dependent measurements.

After every meaningful accepted or rejected result, update the canonical run's
documentation with plots, raw-data links, exact config, conclusion and next
action; update docs/MEASUREMENTS.md with the current entry point and append
docs/harness-log.md. Record safe hardware state and resume instructions before
pausing. Promote validated settings into device state and configs, keeping
failed candidates explicitly separate. Classify failures with FAILURE_MODES.md;
update this playbook when a durable lesson changes the procedure. Preserve
history and supersede obsolete guidance explicitly. Future agents perform this
maintenance during work; no background process or cross-repository memory is
implied. The eight-point rule supersedes older four-point guidance.
