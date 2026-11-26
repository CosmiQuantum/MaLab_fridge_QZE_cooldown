# Environment and hardware safety

## What counts as hardware

Each of these is a lab operation and needs an open envelope (`RUN_ENVELOPE.md`).

- Building `QICK_experiment` or connecting to the Pyro proxy. The constructor connects and creates a folder.
- Running or importing a round robin, a numbered `pucq4_NN` script, a `longitudinal_qze` script or any other acquisition script.
- Building the Yokogawa driver object, or changing current, output or range.
- Changing attenuators or filters.
- Starting or checking Visdom as part of an experiment. Leave live plotting off unless the user asks.
- Creating run folders or writing data on `M:`.

Reading source, parsing with `ast`, git commands and analysing saved files are offline. An offline check never proves how the instrument behaves.

## Python on the lab PC

- Target Python 3.9 syntax. The interpreter on record is `C:\Users\Ma Quantum Lab\anaconda3\python.exe` (3.9.13). Ask the IDE for the configured interpreter if a command fails. In the harness files `python` stands for this interpreter. Do not use another `python` that PATH happens to find.
- Calling that `python.exe` directly gave a NumPy DLL error. Activate Conda first. The form that worked is

```powershell
cmd.exe /d /c "call ""C:\Users\Ma Quantum Lab\anaconda3\Scripts\activate.bat"" ""C:\Users\Ma Quantum Lab\anaconda3"" && ""C:\Users\Ma Quantum Lab\anaconda3\python.exe"" tools\harness_check.py"
```

- `tools/harness_check.py`, `tools/check_config_rules.py` and `tools/check_run_folder.py plan` use only the standard library, so they also run without activation.
- `tools/check_run_folder.py check` opens HDF5 files and needs `h5py`. Run it in the Conda form above. If it ends with `CHECK INCOMPLETE`, the data was not opened and the check does not count.
- PowerShell splits a comma list into separate arguments. Put every comma list in double quotes, like `--expect "res_ge,qspec_ge"`.
- Do not install packages to make a check pass unless the user agrees.
- PowerShell facts that cost time before. `.ps1` files are blocked, so use inline commands and never change the execution policy. Put Python code in a small `.py` file, because `python -c` loses double quotes. Set `PYTHONIOENCODING=utf-8`. Never rewrite a whole text file through `Get-Content` and `Set-Content`. Write JSON as UTF-8 without BOM.
- Connection points for context only. Pyro name server `192.168.1.139:8888`, proxy name `myqick`, fridge `BOB`. They are not permission to connect.

## Before a run

One list. Do it once per run and write the answers into the run's `sub_study_notes.txt` or the log.

1. `git status`. Changes you did not make belong to the user. Leave them.
2. Name row, index and physical qubit. Name the bias.
3. Read the run settings block of the script as text. Check `run_flags`, the rows, `n`, and every override that will be active.
4. Check the readout gain that will really be used against the ceiling in `STATE.md`. The shared config still holds higher gains.
5. Run `python tools/check_run_folder.py plan --study <study> --sub-study <sub_study>`. Check that the planned folder holds no raw file the run would overwrite.
6. Both Yokogawas read current mode, 0 A, output off (unless the user allowed a bias). No other acquisition is running.
7. Work out the run time from points, reps, rounds and delays. Say it. Set a hard bound and a no-progress watchdog.
8. Name the stop condition and the parked state the run must end in.

## Stop a run when

- QICK, Pyro, timing or buffer errors appear.
- A Yoko readback is unexpected.
- A sweep would leave the envelope.
- A trace is clearly saturated or flat, so that going on makes no sense.
- Raw data failed to save. Never start the next stage after a failed save.

## Long runs

Anything over a few minutes must survive the chat tool. A command the tool started can die when the tool call ends.

- Launch hidden and detached through a small `.cmd` file that activates Conda, stops if activation fails, sets one override per line, and calls the interpreter with normal quoting.
- Name it `launch_<sub_study>.cmd` in the repo root (git ignores that name) and send its output to `launch_<sub_study>.log` next to it. The time stamp folder does not exist before the start, so move both files into the run's `documentation/` folder when the run has ended.
- The round robin writes its own log to `documentation/RR_script.log`.
- When Python starts the child, reuse `activated_subprocess_env` from `pucq4_07_flux_campaign.py`. It adds the Conda DLL folders.
- Write the process id of the parent you started into the launch log and into the Long runs row of `STATE.md`.
- Confirm the start from the log and from the first saved file. A queue status alone is not proof.
- Use `PUCQ4_PHASE_CHECKPOINTS=1` so that each finished stage is saved. It works only with one row, one pass, `save_r` 1 and the seven standard stages, and not with timed SSF. Without it the HDF5 is saved only at the end of the pass. A time limit must never cost acquired data.
- Give the parent a wall clock bound, a 600 s no-progress watchdog and no automatic hardware retry.
- Status file writes go through `tools/ssf_progress.py`. A locked status file on `M:` must never end a run. Raw files are the truth.
- One acquisition at a time. A queued stage waits for the saved result of the one before it.
- Stop with the `STOP` marker file at a cycle boundary, or stop only the parent you started so that the child can finish and save. The marker sits in a different place per script. Round robin timed SSF reads `documentation/STOP`. `longitudinal_qze_005_ge_ramsey.py` reads `STOP` in the sub-study folder. `pucq4_13_avoided_crossing.py` reads `STOP` in its run folder.
- For a few runs, start single row runs one after another. For a real queue see section 5 of `EXPERIMENT_CONTRACTS.md`.
- A running process keeps the code it imported. Change shared config only between children.
- Other old Python processes exist on this PC. Never end a process you did not start.
- After any crash, kill or lost terminal, read back both Yokogawas, ramp to 0 A, switch output off and verify. Throw away the partial row.

## Yokogawa procedure

Limits and addresses are in `PUCQ4_DEVICE.md`.

1. Query identity, mode, output, level, range and voltage limit. Require current mode and the 0.01 A range. `source_readbacks` in `pucq4_12_m4_search.py` is a read only example.
2. Ramp to the target at 0.0005 A per second. Verify the level within a stated tolerance.
3. At the end ramp to 0 A, switch output off and verify.

Never jump the current, never switch output on silently, never leave a source in an unknown state after an error.

## Data and git safety

- Keep the user's code changes and all raw data. Deleting measurements is never cleanup.
- Measurement data, deck extraction files, credentials and lab secrets do not go into git.
- If a save failed after acquisition, write down exactly what was lost. Do not call that run reproducible.
- Commit or push only when the user asked for it in this chat.
- Write harness files in plain ASCII. PowerShell 5 has damaged other encodings here before.
