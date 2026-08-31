# PUCQ4 persistent state

Last consolidated: 2026-08-28. This document separates executable configuration,
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

## Current executable starting values

These are the values present in `system_config.py` plus the readout offsets used
by `round_robin_benchmark.py` at consolidation time. They are not proof that every
row meets the final acceptance target.

| Row | res center (MHz) | RR offset (MHz) | qfreq (MHz) | qspec gain | sigma (us) | pi amp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 8919.850 | -0.52 | 5517.69373 | 0.001 | 0.050 | 0.6786303 |
| M2 | 8951.650 | -0.40 | 5438.71351 | 0.007 | 0.101 | 0.6486907 |
| M3 | 8975.000 | -0.08 | 5557.33447 | 0.002 | 0.124 | 0.6586706 |
| M4 | 8999.950 | +0.40 | 5575.00 seed | 0.010 | 0.100 placeholder | 0.700 placeholder |
| M5 | 9014.950 | +0.44 | 7084.77785 | 0.004 | 0.055 | 0.6486907 |
| M6 | 9059.450 | -0.04 | 6522.54018 | 0.006 | 0.062 | 0.6486907 |

Important override/state notes:

- Accepted readout lengths are `[10, 5, 12, 10 placeholder, 5, 3]` us, gains
  `[0.48, 0.56, 0.41, 0.30 placeholder, 0.33, 0.60]`, and offsets
  `[-0.52, -0.40, -0.08, +0.40 placeholder, +0.44, -0.04]` MHz.
- `pucq4_config.QUBIT_FREQS_MEASURED_0MA` correctly uses `NaN` for M4. Shared
  six-element arrays retain 5575 MHz only because callers require a numeric seed.
- `round_robin_benchmark.py` is restored to one pass over validated rows
  `[0, 1, 2, 4, 5]` with res spec, qspec, Rabi, SSF, T1, T2R, and T2E enabled.

## Accepted zero-current measurements

Both known Yokogawas were verified in current mode at exactly 0 A on the 10 mA
range, with outputs off, for the recorded baseline campaign.

The coherence values below are the latest accepted measurements in the task log,
not simultaneous all-row results. Revalidation is required for a final summary.

| Row / physical identity | accepted res (MHz) | accepted qfreq (MHz) | T1 (us) | T2R (us) | T2E (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 8919.830 | 5517.69373 | 31.45 | 16.53 | 37.13 |
| M2 | 8951.650 | 5438.71351 | 8.77 | 11.71 | 15.88 |
| M3 | 8974.970 | 5557.33447 | 45.79 | 42.85 | 77.22 |
| M4 / unassigned | 8999.980 low-power dip | unknown | unknown | unknown | unknown |
| M5 / physical Q4 | 9014.590 | 7084.77785 | 7.63 | 5.28 | 11.26 |
| M6 / physical Q6 | 9059.310 | 6522.54018 | 3.34 | 1.53 | 4.19 |

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
- Final round-robin SSF values were 58.2%, 34.3%, 37.2%, 17.4%, and 21.5% for
  M1, M2, M3, M5, and M6. Three alternating optimization rounds produced no
  significant further improvement. These are measured/optimized but not high.
- The reusable deck is `pucq4_first_light/PUCQ4_first_light_characterization.pptx`;
  regenerate it with `tools/build_pucq4_first_light_ppt.py`.

## Qspec and Rabi lessons

- The strange oscillation/ringing was coherent sidelobe structure from a 20 us
  rectangular spectroscopy pulse driven at the former 0.03-0.08 gains. Lowering
  config gain removed it; the QICK program was not changed.
- The scaled qspec fitter once used `argmax`'s array index as a MHz initial guess
  and allowed negative linewidth. The wrapper was repaired to seed from acquired
  frequency and bound center/width. A fit outside sweep bounds is invalid.
- Current Rabi acceptance is just over half an oscillation with pi gain 0.6-0.7.
  M2's stored 0.7385 is above that target and should be revisited in a deliberate
  sigma calibration; placeholders must never be presented as measurements.

## M4 bounded search result

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

Stop blind scans. Resume M4 only with new external evidence: original VNA/two-tone
trace and exact conditions, an independent repeat measurement, or a confirmed
device-map/port-routing correction. `_tmp_m4_readout_debug.py` remains only
because the qubit was not found; it is not a reusable calibration script.

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

- Resolve whether M4 has an attached/observable sixth physical qubit using new
  evidence, not another blind scan.
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
  promoted to shared configuration. The evidence now supports an unobservable,
  uncoupled, or nonfunctional sixth device/route rather than a missed frequency.
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
