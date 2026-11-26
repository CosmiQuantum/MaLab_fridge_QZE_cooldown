# Current state

This file is the single place for the current state. Overwrite it after every result. Never append dated blocks. History goes to `docs/log/`. Keep it under 12 KB.

Last update. 2026-11-25, from the 24 Nov closeout record. No measurement was made. Two config changes were made offline and have not run on hardware yet (readout gain caps, T2 sweep steps).

## Resume here

| Item | State |
|---|---|
| Hardware | Nothing running. Both Yokogawas in current mode, 0 A, output off. QICK tProc stopped, clocks locked. Last verified at the 24 Nov closeout. Attenuators were not touched and cannot be read back. Verify again before any run. |
| Envelope | Closed. No open request. The 24 Nov zero bias campaign is finished. |
| Next step | The user chose `check` on 25 Nov. Run the SSF check on M1 when the user asks for it in the chat. All steps are in `SSF_CHECK.md`. |
| Long runs | None. The Zeno run on M2 (index 1, plot label Q2, study `qze_005`) was stopped by the user on 20 Nov. |

When you start a long run, write its process id into the Long runs row. Take it out when the run has ended.

## Open decision for the user

No row reaches SSF above 50 percent. Best is M1 at 31 percent.

- The last values above 50 percent are from 29 Oct, at readout gain 0.90 to 0.95 DAC (M1 77, M3 69, M5 67, M6 79 percent. M2 49 and M4 43 were below the gate even then). Since 30 Oct readout stays below the onset of resonator movement, at 0.10 to 0.40 DAC.
- Estimate of 25 Nov, a simple model and not a measurement (`tools/estimate_ssf.py`). Scaling the 29 Oct contrast by the gain ratio and the root of the length ratio gives 30, 13, 21, 24, 9 and 9 percent for M1 to M6. Measured 31, 13, 25, 18, 12 and 7. M1 and M2 agree within about 1 point, M3 to M6 within 2 to 6.
- So the size of the drop fits the lower readout gain. The estimate gives no sign of a fault in the readout chain. It does not clear the chain. The model is rough (the old points were far above the onset, and longer readout did not raise SSF on M3 in October), and the chain was never measured.
- From the 24 Nov values, 50 percent would need about 0.42 DAC on M1, 0.53 on M3, 0.46 on M5 and 0.98 on M2. All are above the ceilings. More optimisation below the ceilings is not expected to reach the gate.
- Two rules of the user conflict here. Readout below the onset (30 Oct). SSF above 50 percent (20 Nov).

Default until the user says otherwise. Set on 25 Nov, when the user left the open points to the tune-up.

1. Both rules stay. Readout stays at or below the caps.
2. No further SSF optimisation loop at these powers.
3. T1, T2R and T2E run only when the user asks for them by name, with the res spec, qspec and Rabi they need. They are averaged measurements. Their results carry the label `measured-below-gate` and stay out of config and deck tables.
4. Promotion into shared config, the final one-pass benchmark, the Zeno long run and any other run over 2 hours stay blocked until SSF passes.

The words the user can choose from. One word, or a word and a number.

| Word | Meaning |
|---|---|
| check | One pass on M1 at the 29 Oct point, above the cap once. It tests the estimate and the chain. CHOSEN by the user on 25 Nov 2026, not run yet. Steps in `SSF_CHECK.md` |
| power | Allow readout above the onset for SSF, up to the 29 Oct points. If the chain is unchanged the 29 Oct contrast should come back. Then raise the caps in `system_config.py` to those gains |
| gate | Keep the low power. Give the new SSF gate as a number, or say drop |
| hardware | Keep both rules and improve the readout chain first |

## Zero bias values, 24 Nov 2026

Evidence label for every row is `measured-below-gate`. Qspec passes the 0.5 MHz gate on all rows. SSF fails on all rows. Nothing was promoted into shared config.

| Row | Index | Physical | Qubit MHz | FWHM MHz | Peak pop | Sigma us | Pi DAC | Readout MHz | Gain DAC | Length us | SSF opt percent | SSF fresh percent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 0 | not assigned | 5515.347298 | 0.1090 | 0.484 | 0.050 | 0.6815 | 8918.242916 | 0.25 | 8 | 31.34 | 31.57 |
| M2 | 1 | not assigned | 5438.508261 | 0.1945 | 0.249 | 0.097 | 0.6865 | 8949.955019 | 0.24 | 7 | 13.08 | 12.00 |
| M3 | 2 | not assigned | 5551.353725 | 0.0993 | 0.471 | 0.124 | 0.6567 | 8973.858219 | 0.25 | 13 | 24.78 | 23.39 |
| M4 | 3 | not assigned | 5594.634433 | 0.1739 | 0.422 | 0.0695 (29 Oct) | 0.6417 | 8999.438145 | 0.40 | 8 | 18.15 | 16.00 |
| M5 | 4 | Q4 | 7065.025317 | 0.1659 | 0.476 | 0.0569 | 0.6169 | 9013.695634 | 0.10 | 8 | 11.63 | 12.08 |
| M6 | 5 | Q6 | 6405.160106 | 0.2875 | 0.349 | 0.0556 | 0.6368 | 9058.849687 | 0.20 | 8 | 7.05 | 5.28 |

- SSF opt is the mean histogram contrast of three independent 20000 shot checks at the listed point. SSF fresh is the later check in the closeout table.
- Sigma comes from the notes per row. M2 peak population and M6 width miss the PUCQ4 targets.
- These are seeds for the next session. Measure again before use.

## Readout gain ceilings, 24 Nov 2026

Measured with a 30 kHz resonance shift tolerance. Working limits, not exact punch out points. A ceiling holds only for its length, bias and RF chain.

| Row | Ceiling as recorded |
|---|---|
| M1 | 0.25 DAC at 8 us and 10 us. On 20 Nov the same 8 us gave only 0.10. None measured at 6 us |
| M2 | 0.30 DAC measured at 7 us. Working cap 0.24 DAC |
| M3 | 0.20 DAC at 13 us. The 0.25 DAC point in the table above has no written qualification. Use 0.20 DAC, 13 us, 8974.058219 MHz (SSF 19.4 percent) until 0.25 is requalified |
| M4 | 0.20 DAC at 5 us. 0.40 DAC at 4 us and 8 us. 0.45 failed at 8 us |
| M5 | The 0.10 DAC reference qualified at 8 us. No higher gain is recorded as qualified. The 6 us reference drifted and is rejected |
| M6 | 0.10 DAC at 45 us. 0.25 failed. 2 us and 4 us did not qualify. The 0.20 DAC point at 8 us in the table above has no written qualification. Requalify it before use |

## Shared config and the caps

`system_config.py` still holds the 29 Oct readout point for five rows. Gains are 0.9, 0.95 and 0.95 for M1, M3 and M4 (`res_gain_ge`) and 0.95 for M5 and M6 (0 mA rows of `PUCQ4_FLUX_PULSE_CALIBRATION`). Tone, length, qubit frequency, sigma and pi amplitude there are old too. M2 holds 0.24 from 20 Nov.

Since 25 Nov `PUCQ4_ZERO_CURRENT_READOUT_GAIN_MAX` holds a cap for every row, 0.25, 0.24, 0.20, 0.20, 0.10 and 0.10 DAC for M1 to M6. A zero bias round robin run above a cap stops with an error before it touches hardware. So a plain run on any row but M2 stops until the gain is set.

- The cap does not know the readout length. The config lengths are 6, 7, 13, 5, 6 and 45 us. M1 has no ceiling at 6 us and M5 at 6 us is rejected, so set 8 us for both.
- The M4 cap is the 5 us ceiling. To use 0.40 DAC on M4 set 8 us (or 4 us) and raise its cap to 0.40. Report that as a config change.
- For every run set `PUCQ4_READOUT_GAIN`, `PUCQ4_READOUT_GAIN_MAX`, `PUCQ4_READOUT_LENGTH_US` and `PUCQ4_READOUT_FREQ_MHZ` for the row, one row per run. Then compare them with the saved `Syst Config`.
- Only `round_robin_benchmark.py` checks the caps, and only at zero bias. At a nonzero bias set `PUCQ4_READOUT_GAIN_MAX` yourself.
- `Combined_Optimization_Scriptsv2.py`, the `longitudinal_qze` scripts and `pucq4_12_m4_search.py` check no cap. The optimiser sweeps around the old config gain by default. Set its gain with `PUCQ4_OPT_READOUT_GAIN` and `PUCQ4_OPT_MAX_GAIN`.
- The flux controllers `pucq4_07` to `pucq4_11` pass the stored 0.95 DAC at 0 mA. Their 0 mA stages now stop at the cap for M5 and M6.

## Sweep settings and the gates

Checked offline with `python tools/check_config_rules.py`.

- Fixed on 25 Nov. T2R and T2E on M1 and M3 had 6.7 and 5.0 acquired points per fringe. The steps in `pucq4_round_robin_profile` (`expt_config.py`) were raised. Every row now has 10 or more. Not run on hardware yet. At 1200 reps the M3 echo takes about 5 minutes. Time grows with reps.
- Still open. With the last T1 values, relaxation delays are outside 5 to 10 times T1 on most rows (M2, M4 and M6 use 14 to 37 times, M3 uses 4.8 times), and the M3 T1 window of 150 us is under 3 times its T1. Set them from a fresh T1, then run the tool again.

## Last coherence on record

No T1, T2R or T2E has been validated since 29 Oct. The 29 Oct values were taken at the high readout gain and are history. A coherence run the user asks for by name runs at or below the caps. M3 then uses 0.20 DAC, 13 us, 8974.058219 MHz. M6 uses 0.10 DAC at 45 us, where this file holds no SSF.

| Row | T1 us | T2R us | T2E us | Date |
|---|---|---|---|---|
| M1 | 23.5 | 20.2 | 30.5 | 29 Oct |
| M2 | 5.35 | 6.60 | 12.8 | 29 Oct, fits depend on time (doublet). A T1 check on 20 Nov gave 4.07 |
| M3 | 51.8 | 39.5 | 65.6 | 29 Oct |
| M4 | 7.07 | 8.96 | 12.7 | 29 Oct |
| M5 | 13.6 | 5.11 | 13.4 | 29 Oct |
| M6 | 4.01 | 1.66 | 4.40 | 29 Oct |

Use these only to choose a first relaxation delay (5 to 10 times T1). Measure T1 again before trusting them.

## Open problems

1. SSF below gate on all rows. See the open decision above.
2. Attenuation cannot be read back. The values in code are not measurements.
3. M5 reference dip drifts at 6 us readout length (41 kHz spread).
4. M2 (index 1) shows a doublet of about 130 kHz. No cause is confirmed.
5. Physical numbers of M1 to M4 are not assigned.

## Where the data is

| What | Path |
|---|---|
| Data root | `M:/_Data/20250822 - Olivia` |
| Run and device | `pucq4_run_started_Oct_7/PUCQ4` below the data root. Called the device folder below |
| 24 Nov zero bias campaign | `<device folder>/find_resonators_sandbox/` sub-studies `zb1124a`, `zbopt24a`, `zbfine24a`, `zb5r24a`, `zb5opt24a`, `zb6opt24a`, `zb6fine24a` |
| 24 Nov evidence and report | `<device folder>/find_resonators_sandbox/zb1124a/2026-11-24_13-17-41/documentation/zero_bias_review` |
| Zeno on M2 | `<device folder>/qze_005/` |
| Newest deck | `M:/_Data/20250822 - Olivia/pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light/PUCQ4_zero_bias_20261124.pptx` |
| Older campaigns | `docs/MEASUREMENT_INDEX.md` |
