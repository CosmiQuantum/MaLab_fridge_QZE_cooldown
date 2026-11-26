# Measurement index

One line per campaign, newest first. It tells you where old data and old write-ups are, so nobody has to read an archive to find them. The current state is in `harness_engineering/STATE.md`, not here.

When a campaign closes, add one line at the top. Keep each cell short. Details stay in the run's own `documentation/` folder.

Paths are relative to the device folder `M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/` unless a row says otherwise. They are copied from the archives as written there. The September write-ups and scripts still print `pucq4_run_started_Oct_7`. Read `pucq4_run_started_Aug_3` there.

| When (2026) | What | Rows | Study and sub-studies | Outcome | More |
|---|---|---|---|---|---|
| 25 to 26 Nov | Overnight Ramsey drift at a fixed carrier, 10 batches | M1 to M6 | `find_resonators_sandbox/` with `ramsey_drift`, run `2026-11-25_21-48-10` | Peak to peak offsets 2.8, 5.2, 2.6, 4.2, 83.5 and 105.8 kHz for M1 to M6. Nothing promoted | `docs/log/2026-11.md`, entry of 26 Nov |
| 24 Nov | Zero bias reoptimisation. Res spec, power check, qspec, Rabi, SSF search | M1 to M6 | `find_resonators_sandbox/` with `zb1124a`, `zbopt24a`, `zbfine24a`, `zb5r24a`, `zb5opt24a`, `zb6opt24a`, `zb6fine24a` (the record names the study for `zb1124a` only) | SSF below gate on all rows, best M1 at 31 percent. Nothing promoted | `docs/archive/MEASUREMENTS_through_2026-11-25.md` lines 118 to 173 |
| 20 to 21 Nov | SSF checks on all rows, then optimisation at higher readout gain | M1 to M6 | `find_resonators_sandbox/` with `ssall1120`, `ssup1120`, `ssopt1120`, `ssfine1120`, `rt1121a` | Below gate. User stopped everything on 21 Nov | same file, lines 51 to 116 |
| 16 to 20 Nov | Zeno long run and SSF recovery on M2 (plot label Q2) | M2 | `qze_005/` | Stopped by the user on 20 Nov. SSF above 50 percent not recovered | `harness_engineering/archive/PUCQ4_STATE_through_2026-11-25.md` lines 1 to 151 |
| 4 to 16 Nov | 24 hour SSF tracking on M2 (plot label Q2) | M2 | `find_resonators_sandbox/` with `ssf_24h_q2`, run `2026-11-11_12-52-52` | Run ended 12 Nov, data package delivered 16 Nov | same file, lines 152 to 464 |
| 27 to 29 Oct | Fresh cooldown benchmark of all six rows at zero bias | M1 to M6 | `find_resonators_sandbox/` with `full1_a_1029` to `full6_a_1029` | Complete, at readout gain 0.90 to 0.95. Last coherence values on record | same file, lines 465 to 547 |
| 9 to 16 Sept | Search for the Q5 and Q6 avoided crossing | M6 with M2 | `pucq4_first_light/` with `ac_*` sub-studies, under run `pucq4_run_started_Aug_3` | Paired lines at two biases only. No coupling accepted | `docs/Q5_Q6_NEXT_CHAT_HANDOFF.md` lines 1 to 53, `docs/pucq4-avoided-crossing.md` |
| 8 to 9 Sept | Round robin reproduction, six rows plus flux points | M1 to M6 | `pucq4_first_light/` under run `pucq4_run_started_Aug_3` | Closed with gaps, 3 of 6 rows passed | `docs/pucq4-round-robin-closeout.md` |
| 28 Aug to 8 Sept | Flux campaign for Q4 and Q6. Qspec, Rabi, sigma and coherence maps against current | M5, M6 | `pucq4_first_light/` with `yoko_flux_campaign`, under run `pucq4_run_started_Aug_3` | Calibration per bias, now in `system_config.PUCQ4_FLUX_PULSE_CALIBRATION` | state archive, lines 2070 to 2838 |
