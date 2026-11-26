# Measurement playbook

The procedure for every measurement, calibration and diagnosis. It is the single home of the gates. User policy of 20 Nov 2026, tightened on 25 Nov 2026 with the stop rule and the label list.

This is a procedure. `RUN_ENVELOPE.md` says what you may run.

## Gates

Only the user changes a gate. Each gate has a number, so there is nothing to interpret.

| Stage | Gate | Notes |
|---|---|---|
| Res spec | Dip resolved and repeatable at a gain below the movement onset | Record gain, length and bias with every dip |
| Punch out | A measured ceiling below the onset of resonator movement | Holds only for that length, bias and RF chain |
| Qspec | FWHM at or below 0.5 MHz, at least 3 acquired points across the FWHM, line repeats | PUCQ4 target is tighter, see `PUCQ4_DEVICE.md`. The 3 points are a tune-up default |
| Rabi | A visibly resolved oscillation, not only a fitted number | PUCQ4 target is pi amplitude 0.6 to 0.7 DAC |
| SSF | Histogram contrast strictly above 50 percent, confirmed by independent repeats | 50 percent is the floor, not the goal. Never convert to assignment accuracy. Never use (1+contrast)/2 |
| T1, T2R, T2E | Decay resolved inside the sweep window, fit bounded and plausible | Sweep to at least 3 times the expected decay time (3 is a tune-up default) |
| T2R, T2E sampling | At least 8 acquired points per fringe. With fringe frequency f in MHz and actual time step dt in us, abs(f) times dt is at most 0.125 | Use the saved time axis and the measured fringe frequency. Interpolated points do not count |
| Relaxation delay | 5 to 10 times the measured T1 of that row and bias, for Rabi, SSF, T1, T2R and T2E | Before T1 is measured use the last T1 in `STATE.md` as a seed, then check the multiple again. Qspec may use less if the line stays reliable. Record the multiple |

## Evidence labels

Use only these six labels. Never invent another label, review type or acceptance category.

| Label | Meaning |
|---|---|
| `validated-measurement` | Passed every gate that applies, repeated, raw data saved |
| `measured-below-gate` | A real, repeatable measurement that misses a gate or target. Record the miss |
| `rejected-candidate` | Failed repeat, bounds or control. Not a measurement of the qubit |
| `executable-config` | What the code will use now. Not automatically validated |
| `deck-seed` | A value from a deck, an old log or an archive. A starting guess |
| `acceptance-target` | A wished value, such as a linewidth target |

Only `validated-measurement` values go into shared config or summary tables.

## Stop rule

1. A missed gate is recorded as a miss with its number. Never relax, rename or reinterpret a gate to pass.
2. Two attempts per gate and row. An attempt is one full pass of the stage as written under Order of work. The second attempt must change one named thing for a stated reason. More averaging alone is not a reason. (Two is a tune-up default. The user may change it.)
3. After the second miss, label the result `measured-below-gate`, update `STATE.md`, and move on to work that does not depend on it. Tell the user in one short block. Give the number, the gate, what was tried, and the one decision you need.
4. When a value that used to pass now fails, first rerun the last validated setting unchanged. That one cheap run comes before any optimisation. If a limit in `RUN_ENVELOPE.md` forbids that setting, ask the user at once.
5. Never start a dependent stage on a failed gate. Rabi needs a passing qspec. SSF needs a resolved Rabi. T1, T2R and T2E need a resolved Rabi and a readout point at or below the ceiling.
   Below the SSF gate they run only when the user asks for them by name. Their results are then `measured-below-gate` and stay out of config and deck tables.
   Promotion, the final benchmark and the Zeno long run need the SSF gate. Another long run below the SSF gate needs the user's ask by name. (The by name exception is a tune-up default.)
6. Say before launch how long a stage should take. Stop and report when it takes more than twice that.
7. Every run names a hypothesis and what result would reject it. A null result counts only with a passing positive control in the same run.

## Order of work

Do the stages in order. Skip a stage only when `STATE.md` holds a `validated-measurement` for it that nothing has invalidated since.

1. Resume. Read `STATE.md`. Name row, index and physical qubit. Treat every stored number as a seed.
2. Find the resonator at low readout gain with the existing res spec. For speed use high reps and one round.
3. Measure punch out. Take frequency against gain inside the allowed range. The ceiling is the highest gain before a repeatable shift. If the onset is not resolved, use a lower validated ceiling. Never state a maximum you did not measure.
4. Find the qubit. Start with a higher drive gain to locate the line, then lower it to the smallest gain that gives a repeatable resolved line. Zoom the sweep so several points sit on the line. Record FWHM, fit error, drive gain, reps, rounds and time taken. If no line is found, check the list under Recovery before raising drive or averaging.
5. Calibrate Rabi at the measured frequency. Tune sigma, find the pi amplitude. Run it again after any change of frequency or sigma.
6. Optimise SSF. First run `python tools/estimate_ssf.py --ssf <percent> --gain <DAC> --length <us>` with the last measured contrast. If it says the gate is far out of reach inside the ceiling, do not start. Tell the user. It is a rough estimate, so a close call is still worth one attempt.
   Otherwise alternate a length sweep (shortest length on the plateau) with a 2D sweep of gain against frequency offset at that length. Stay at or below the punch out ceiling. Confirm the chosen point with independent repeats so that noise is not selected.
7. Propagate. Write validated readout offset, gain, length, qubit frequency, sigma and pi amplitude into the shared config and into the top-of-script values of every active script that overrides them. Report file and key, old to new, unit, row, bias, evidence path, and what must be measured again.
8. Cut acquisition time. First for res spec, then for each stage, compare lower reps and rounds with a reliable reference by repeats, fit error and time. Fix the tolerance before choosing. Keep the fastest setting that stays reliable.
9. Measure T1, then set the relaxation delay from it. Then T2R and T2E with the sampling gate. Before the run, `python tools/check_config_rules.py --t1 "M1=<us>,M2=<us>"` shows offline whether the sweep settings meet the sampling and relaxation gates. If there is no fringe, write why the gate does not apply. Never make up a frequency.
10. Flux. For the tunable qubits repeat stages 2 to 9 at each bias the user allowed. Choose the step from the measured slope and refine near fast changes and crossings. Save calibration by bias in the shared config. Check interpolation before use. Never extrapolate.

## What a change invalidates

| What changed | Measure again |
|---|---|
| Qubit frequency | Qspec fit, Rabi, then SSF |
| Sigma or pi amplitude | Rabi, then SSF, T1 and T2 as far as they are used |
| Readout frequency, gain or length | SSF, and check every population plot |
| Readout length or RF chain | Punch out ceiling |
| Yoko current | Everything from stage 2. Never reuse zero bias values at another bias |
| Warm up, cooldown or a long pause | Res spec and qspec before any pulse work |

## Recovery

- No qspec line does not mean no qubit. Check in this order. Row mapping, the readout dip and offset, readout gain and length against the ceiling, the bias, the last good evidence. Recover the readout point first, then retry qspec with a wide saved locator.
- A broad line needs less drive and a finer sweep.
- Low SSF with a good Rabi is a readout question. Report both numbers as they are. Then apply rule 4 of the stop rule.
- A fit that returns a number is not a result. Look at the raw trace.

## After every result, accepted or failed

1. Run `python tools/check_run_folder.py check` on the new run folder, in the Conda form, because it needs `h5py`. Both the plot and the HDF5 must exist and hold data. `CHECK INCOMPLETE` means the data was not opened and the check does not count.
2. Compare what you asked for with what was saved. Sweeps, reps and delays are in the saved `Exp Config`. Readout gain, length and tone, qubit frequency, sigma and pi amplitude are in the saved `Syst Config`. Overrides have been silently dropped before.
3. Overwrite `STATE.md` (values, hardware state, next step).
4. Append one entry of at most 12 lines to `docs/log/` (format in `TASK_TEMPLATE.md`).
5. When a campaign closes, add one line to `docs/MEASUREMENT_INDEX.md`.
6. Put plots and notes in the run's own `documentation/` folder. Link them. Do not copy raw data.
7. A new durable lesson gets one line in `LESSONS.md`.

Before you pause or the chat ends, bring the hardware to the parked state and write that state into `STATE.md`.
