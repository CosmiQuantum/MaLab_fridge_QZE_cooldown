# PUCQ4 device card

Facts about the chip and its wiring that change rarely. Daily values live in `STATE.md`. Live numbers live in the config files.

## Identity map

Always name a qubit with all three of row, index and physical number.

| Resonator row | Python index | Label in plots and file names | Physical qubit | Flux line |
|---|---|---|---|---|
| M1 | 0 | Q1 | not assigned (design intent Q3, never confirmed) | none |
| M2 | 1 | Q2 | not assigned (design intent Q5, never confirmed) | none |
| M3 | 2 | Q3 | not assigned (design intent Q1, never confirmed) | none |
| M4 | 3 | Q4 | not assigned (design intent Q2, never confirmed). This is NOT physical Q4 | none |
| M5 | 4 | Q5 | physical Q4, DC line D4 | Yoko 3 at 192.168.1.73 |
| M6 | 5 | Q6 | physical Q6, DC line A5 | Yoko 4 at 192.168.1.77 |

- The label in plots and file names is index plus one. It is not the physical number.
- When the user says Q4 or Q6 and it is not clear whether they mean the plot label or the physical qubit, ask once before any run.
- Do not assign a physical number to M1 to M4. The design record found on 15 Sept gives design intent only. A Q5 experiment still needs a confirmed identity and a wired Q5 drive, so ask the user first.

## Flux bias

- Only physical Q4 (M5) and physical Q6 (M6) are flux tunable. Treat the other rows as fixed frequency.
- Characterised range is -10 to +10 mA. That is the hard limit. Never extrapolate. The driver's own 15 mA guard does not widen it.
- Allowed grid in the round robin is -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10 mA.
- Parked state is current mode, 0.01 A range, 0 A, output off, 30 V limit.
- Ramp at 0.5 mA per second. Ramp to 0 A before switching output off. Never jump the current.
- Settle 600 s at Yoko 3 at -10 mA and at +10 mA.
- Qspec must run on every pass on M5 at -2.5, 0 and +10 mA and on M6 at 0 mA. The round robin refuses Rabi, SSF, T1, T2R and T2E there without `q_spec` or a fresh `PUCQ4_QFREQ_MHZ`.
- The level readback is the source setting. It is not an ammeter.
- Both Yokogawas away from zero at the same time has never been verified.
- Yoko serial numbers are 90Z114063 (Yoko 3) and 90Z114062 (Yoko 4). Check them before a run.
- Bias is chosen with `YOKO3_CURRENT_MA` and `YOKO4_CURRENT_MA` near the top of `round_robin_benchmark.py`, or with the environment variables `PUCQ4_YOKO3_CURRENT_MA` and `PUCQ4_YOKO4_CURRENT_MA`. `set_yoko_bias` sets, verifies and returns both sources to zero and off.
- Calibration per bias lives in `system_config.PUCQ4_FLUX_PULSE_CALIBRATION` (9 currents times 2 qubits). Key new entries by physical line and current with units. Never overwrite the zero current lists with a biased value.

## RF chain

- Ports. DAC0 drives the resonators, DAC1 is the shared qubit drive, ADC0 is readout. Never invent a port.
- Attenuator values in code are resonator DAC 10 and 15 dB, qubit DAC 5 and 4 dB, ADC 17 dB. These are constructor values. The setter lines are commented out and there is no readback. Do not change them and do not report them as measured.
- The qubit DAC Nyquist boundary is 4915.2 MHz. A sweep must not cross it. The 9 GHz resonators sit in DAC zone 2. M6 at -7.5 and -10 mA sits below the boundary (near 4839 and 4376 MHz). Ask before a run there.
- A pulse can be at most about 106 us (614.4 cycles per us, 16 bit length). Keep every pulse at or under 100 us.
- Gaussian pulse length is 4 times sigma.

## Device targets

The gates in `MEASUREMENT_PLAYBOOK.md` always apply. These are the tighter PUCQ4 targets. A value that passes the gate but misses the target is recorded with the miss.

| Stage | PUCQ4 target |
|---|---|
| Qspec | FWHM near 0.1 to 0.2 MHz, calibrated peak population 0.3 to 0.5, several acquired points across the line. Reach it with drive gain and length in config. Never edit the qspec QICK program. |
| Rabi | Just over half an oscillation in the default plot and a fitted pi amplitude of 0.6 to 0.7 DAC gain, reached by tuning sigma. A sigma update that worked is `sigma *= sqrt(pi_amp / 0.65)`. |
| SSF | Alternate (1) a length sweep at fixed gain and offset, taking the shortest length on the plateau, and (2) a 2D gain against frequency offset sweep at that length. Repeat until the improvement is no longer significant. 50 percent is the floor, not the goal. Confirm the chosen point with independent repeats. |
| Final evidence | One pass of `round_robin_benchmark.py` with its normal calibrated population plots and HDF5. Res spec, qspec, Rabi, SSF, T1, T2R and T2E for every row. |

## Crossing candidates (Q6 tuned with Yoko 4)

- At -5.135 mA single lines were seen on M6 (5557.87 MHz) and on M3 (5557.32 MHz), 0.55 MHz apart. At -5.305 mA M1 was seen at 5517.69 MHz and M6 showed no accepted line. No crossing was resolved at either bias.
- Near -5.66 mA one line near 5431.7 MHz shows on both M6 and M2.
- Paired lines were accepted only at -5.8 mA and -5.74 mA. No minimum splitting, no coupling J and no physical Q5 were accepted.
- Details are in `docs/pucq4-avoided-crossing.md` and `docs/Q5_Q6_NEXT_CHAT_HANDOFF.md` lines 1 to 53.

## Reports and decks

- Keep one reusable PowerPoint in `pucq4_run_started_Aug_3/PUCQ4/pucq4_first_light` below the data root. That is the old run folder. The newest deck there is `PUCQ4_zero_bias_20261124.pptx`. Backups go only into `_PowerPoint_archive`.
- First slide is the zero current device map table. One line per row with physical qubit, row and index, resonator frequency, qubit frequency, T1, T2R and T2E, with units.
- After a Yoko characterisation add the matching current sweep table for the two flux tunable qubits. Show FWHM with actual current, qspec gain and pulse length.
- Fill tables only from validated results. A field that is unknown or below a gate stays visibly unknown or marked.
- Then one group of slides per experiment (res spec, qspec, Rabi, SSF, T1, T2R, T2E) with all rows, then the readout length and the gain against frequency optimisation slides.
- Use the plots the round robin made. Write the acquisition settings on the slide.
- For a Yoko characterisation add current against frequency maps for qspec and res spec, current against gain Rabi maps, the chosen sigma against current, and T1, T2R and T2E maps. Current on x, actual swept frequency on y, no scaling per row.
- Never set both width and height when placing a plot. Keep the aspect ratio and center the plot in its cell.
- Render the deck natively and look at it before calling it done.
- SSF in tables is histogram contrast in percent. It is never converted to assignment accuracy.
- Writing a deck to the `M:` drive needs the user's word. See the ask list in `RUN_ENVELOPE.md`.
