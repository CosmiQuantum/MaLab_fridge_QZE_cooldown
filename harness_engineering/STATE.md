# Current state

This file is the single place for the current state. Overwrite it after every result. Never append dated blocks. History goes to `docs/log/`. Keep it under 10 KB.

Last update. 2026-11-25. Written during the harness tune-up from the 24 Nov closeout record. No measurement was made for this update.

## Resume here

| Item | State |
|---|---|
| Hardware | Nothing running. Both Yokogawas in current mode, 0 A, output off. QICK tProc stopped, clocks locked. Last verified at the 24 Nov closeout. Attenuators were not touched and cannot be read back. Verify again before any run. |
| Envelope | Closed. No open request. The 24 Nov zero bias campaign is finished. |
| Next step | Needs the user. See the blocker below. |
| Long runs | None. The Zeno run on M2 (index 1, plot label Q2, study `qze_005`) was stopped by the user on 20 Nov. It must not restart before SSF passes its gate. |

When you start a long run, write its process id into the Long runs row. Take it out when the run has ended.

## Blocker for the user

No row reaches SSF above 50 percent. Best is M1 at 31 percent.

- The last SSF values above 50 percent are from 29 Oct (M1 77, M3 69, M5 67, M6 79 percent). They were taken at readout gain 0.90 to 0.95 DAC. M2 (49 percent) and M4 (43 percent) were below the gate even then.
- Since 30 Oct the standing rule is that readout stays below the onset of resonator movement. That keeps the gain at 0.10 to 0.40 DAC.
- The cause of the drop is not known. No SSF was taken at 0.90 to 0.95 DAC since 30 Oct, so the gain rule is a suspect, not a finding. On 21 Nov the user said the RF chain was unchanged. Nobody measured it.
- The 24 Nov closeout named one next step. Check the physical readout chain against its last documented working setup, then repeat the best M1 point.
- The stop rule applies. Three SSF campaigns since 20 Nov did not pass the gate, so do not start another optimisation loop.
- Decision needed from the user. May one SSF run on M1 repeat the 29 Oct setting once (8918.375 MHz, 0.90 DAC, 6 us), which is above today's ceiling? Stop rule 4 asks for that run first.

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

- SSF opt is the mean histogram contrast of three independent 20000 shot checks at the listed readout point. SSF fresh is the later check in the closeout table.
- Sigma comes from the notes per row. The closeout table has no sigma.
- M2 peak population and M6 width miss the PUCQ4 targets.
- These are seeds for the next session. Measure again before use.

## Readout gain ceilings, 24 Nov 2026

Measured with a 30 kHz resonance shift tolerance. They are working limits, not exact punch out points. A ceiling holds only for its length, bias and RF chain.

| Row | Ceiling as recorded |
|---|---|
| M1 | 0.25 DAC at 8 us and 10 us |
| M2 | 0.30 DAC measured at 7 us. Working cap 0.24 DAC, also set in `system_config.py` |
| M3 | 0.20 DAC at 13 us. The 0.25 DAC point in the table above has no written qualification. Use 0.20 DAC, 13 us, 8974.058219 MHz (SSF 19.4 percent) until 0.25 is requalified |
| M4 | 0.40 DAC at 4 us and 8 us. 0.45 failed at 8 us |
| M5 | The 0.10 DAC reference qualified at 8 us. No higher gain is recorded as qualified. The 6 us reference drifted and is rejected |
| M6 | 0.10 DAC at 45 us. 0.25 failed. 2 us and 4 us did not qualify. The 0.20 DAC point at 8 us in the table above has no written qualification. Requalify it before use |

## Shared config warning

`system_config.py` still holds the 29 Oct readout point for five rows. Readout gains are 0.9, 0.95 and 0.95 for M1, M3 and M4 (`res_gain_ge`) and 0.95 for M5 and M6 (the 0 mA rows of `PUCQ4_FLUX_PULSE_CALIBRATION`, at 6 us and 45 us). Tone, length, qubit frequency, sigma and pi amplitude there are old too.
M2 holds 0.24 from 20 Nov and is the only row with a cap (`PUCQ4_ZERO_CURRENT_READOUT_GAIN_MAX`).

A plain round robin run on any row but M2 would read out far above the ceilings. For every run set `PUCQ4_READOUT_GAIN`, `PUCQ4_READOUT_GAIN_MAX`, `PUCQ4_READOUT_LENGTH_US` and `PUCQ4_READOUT_FREQ_MHZ` for the row, one row per run. Then compare them with the saved `Syst Config`.

## Sweep settings that miss the gates

Found offline on 25 Nov with `python tools/check_config_rules.py`. Nothing was changed. Fix these in `pucq4_round_robin_profile` (`expt_config.py`) before the next coherence run, then run the tool again.

- T2R on M1 and M3 has 6.7 acquired points per fringe. T2E on M1 and M3 has 5.0. The gate is 8.
- With the last T1 values, relaxation delays are outside 5 to 10 times T1 on most rows. M2, M4 and M6 use 14 to 37 times. M3 uses 4.8 times.

## Last coherence on record

No T1, T2R or T2E has been validated since 29 Oct. The 29 Oct values were taken at the high readout gain and are history. The final one-pass benchmark is blocked until SSF is settled. A coherence run the user asks for by name may still run at the readout points above. Its results are `measured-below-gate`.

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

1. SSF below gate on all rows, cause unknown. See the blocker above.
2. Attenuation cannot be read back. The values in code are not measurements.
3. M5 reference dip drifts at 6 us readout length (41 kHz spread).
4. M2 (index 1) shows a doublet of about 130 kHz. No cause is confirmed.
5. Physical numbers of M1 to M4 are not assigned.
6. The all-six benchmark and deck update planned on 4 Nov were never done.

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
