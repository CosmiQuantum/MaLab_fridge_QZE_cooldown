# SSF check on M1

One diagnostic run. The user chose it on 25 Nov 2026 with the words "Check first". The permission is in `RUN_ENVELOPE.md`. Run it only when the user asks for it in the chat.

What it is. One round robin pass on M1 (index 0, physical qubit not assigned) at zero bias at the 29 Oct readout point, tone 8918.375 MHz, 0.90 DAC, 6 us. The pass runs qspec, Rabi and SSF in that order, and all three read out at 0.90 DAC. That is above today's cap of 0.25 DAC. It tests whether the old SSF comes back.

This run is the one exception to these rules. Do not ask again for any of them.

- The readout gain limit, the ceilings and the caps.
- The rule to set 8 us and readout overrides for M1 (`STATE.md`).
- The order of work in the playbook. No res spec and no punch out first.

## Steps

1. Write into the Envelope row of `STATE.md` that this request is open, and that the M1 cap in `system_config.py` is raised to 0.90 for this run and must go back to 0.25.
2. In `system_config.py` change the first value of `PUCQ4_ZERO_CURRENT_READOUT_GAIN_MAX` from 0.25 to 0.90.
3. Do the Before a run list in `ENVIRONMENT_AND_SAFETY.md`. For its step 4 the gain is 0.90 on purpose.
4. Run `round_robin_benchmark.py` once with these four environment values and no others.

| Name | Value |
|---|---|
| `PUCQ4_QS` | `0` |
| `PUCQ4_RUN_FLAGS` | `q_spec,rabi,ss` |
| `PUCQ4_SUB_STUDY` | `ssf_check_m1_0mA_<MMDD>` |
| `PUCQ4_PHASE_CHECKPOINTS` | `1` |

5. Set no readout override. Keep the stored delays and shot counts (250 us relaxation and 50000 shots per state, as on 29 Oct). The stored config for M1 is the 29 Oct point. Acquisition takes about 4 minutes. Launch it detached as under Long runs in `ENVIRONMENT_AND_SAFETY.md` and stop it at 15 minutes.
6. Right after the run, whatever the result, put the cap back to 0.25. `git diff -- system_config.py` must then show no change. Then set the Envelope row back to closed.
7. Check what ran. `acquisition_profile.json` lists only the four names above. The saved `ss_ge` `Syst Config` holds readout gain 0.9, readout length 6.0 us and readout frequency 8918.375 MHz (keys `res_gain_ge`, `res_length`, `res_freq_ge`). Run `python tools/check_run_folder.py check` on the folder with `--expect "qspec_ge,rabi_ge,ss_ge"`.
8. The script checks no gate. It runs SSF whenever both fits return a number. Look at the qspec and Rabi plots. The check counts only when both pass their gates. If they do not, do steps 1 to 7 once more with one named change to qspec or Rabi, never to the readout point. (This one repeat is a tune-up default.)

## Reading the result

Compare the SSF contrast with 29 Oct at this point. That was 77 percent in the full pass and 74 to 75 percent in two independent checks.

| Contrast | Reading | Then |
|---|---|---|
| 65 percent or more | The old contrast is back. The drop is the readout power | Ask the user to choose between power, gate and hardware |
| 50 up to 65 percent | Power explains most of the drop. Part is open | Give the number, name no cause, ask the user |
| below 50 percent | Not back. Something other than power changed | Stop and tell the user |

65 and 50 are tune-up defaults. Never repeat the SSF on your own.

## Afterwards

- Label the result `measured-below-gate`. The missed gate is the readout gain ceiling. If qspec or Rabi missed twice, the SSF number is a `rejected-candidate`.
- It is never promoted. It does not count as SSF passing. Default 4 in `STATE.md` stays.
- Set the permission in `RUN_ENVELOPE.md` to used, with the date. A run that saved no SSF data does not use it up.
- Write date, result, label and data set folder into `STATE.md`, and one log entry.
- Replace everything below the title of this file with one line that says when the check was done.
