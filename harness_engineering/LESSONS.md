# Lessons index

One line per lesson. Search this file by symptom. Do not read it top to bottom unless you are diagnosing a failure.

Each line reads symptom, then `->`, then the rule. The tag at the end points to the full story.
`F` is a line in `archive/FAILURE_MODES_through_2026-11-25.md`.
`S` is a line in `archive/PUCQ4_STATE_through_2026-11-25.md`.
`G` is a line in `docs/archive/harness-log_through_2026-11-25.md`.
Line numbers are as of 25 Nov 2026. If one is off, search the archive for the words.

To add a lesson, add one line under the right heading. Keep it under 200 characters. Put the story in the month log, not here.

## Top ten

1. A returned fit, a saved plot or a finished script is not acceptance. Check raw data, sampling and a repeat. F170 F313 F700
2. Never promote a seed, a qspec-only frequency or a single grid maximum into shared config. F389 F407 F811
3. A missed gate is recorded as a miss. Never relax or rename a gate, raise power or add reps just to finish. F3 F102
4. After any crash or lost process, read back both Yokogawas and bring them to 0 A with output off. F331
5. After every run check that the plot and the HDF5 both exist and hold finite data, with paths under 260 characters. F373 F700
6. Carry the fresh qubit frequency, readout tone and pi amplitude into every override, then compare with the saved Exp Config. F292 F433 F444 F476
7. When repeats disagree, keep both, promote nothing and name no physical cause. F744 F899 F913
8. A null result needs a passing positive control in the same run. More averaging is not a fix. F205 F232 F398
9. Long runs need a time budget from point, rep and delay counts, a detached launch with a log, and status writes that cannot kill the run. F29 F590 F1021
10. Readout power stays below the onset of resonator movement. State gain and metric with every contrast number. F14 F1156

## Readout and resonator

- Res scan slow -> for res spec use high reps and rounds=1. F364
- Resonator visible but SSF weak -> visibility floor, movement onset and best SSF power are three separate things. Requalify a ceiling with low, trial, low, trial spectra. F14 G4637
- Low power dip fails its repeat -> repeat at matched averaging and record the averaging with each SNR number. F48
- `res_length` override ignored -> also set `res_length_ge`, then check the effective config. F292 F1163
- Qspec-only run drops the readout offset -> use `PUCQ4_READOUT_FREQ_MHZ`, not `PUCQ4_RES_FREQ_MHZ`, and save the actual tone. F476
- Old offset reused after a new dip moved the readout tone and SSF dropped -> keep the absolute readout tone, derive the offset from a res scan at the chosen gain and length. F754 F770 F975 S3120
- Ceiling blocks a res-only requalification -> use `PUCQ4_RES_POWER_CHARACTERIZATION_MAX`, never edit the ceiling, compare fitted dips not raw minima. F1178
- M5 at -10 mA shows a peak near 9012.87 MHz, not a dip -> do not force a dip fit there. S2252

## Qubit spectroscopy

- Ringing and low Lorentzian R2 -> too much gain on the 20 us rectangular drive. Lower gain or use 100 us. Keep the QICK program. F152 F349 S1952
- Fit center outside the sweep or negative width -> invalid. Seed in MHz from the acquired axis, bound the center, keep the width positive. F357 F693 S1955
- Broad fit picks the weak quadrature or noise -> review I and Q, use the joint IQ fit, repeat locally. A coarse null is not absence. F163 F258
- Line sits in one bin or under about 3 samples per FWHM -> reject and rescan on a finer grid. Controls must pass the same sampling gate. F270 F303 F1189 S1580
- Narrow fit with peak population near 0.04 -> check center, FWHM, signed population and SNR together. F455 S2321
- Narrow window around a stale seed found aliases (5984 MHz against the true 6037.97 MHz) -> when Rabi fails, run a wide saved locator first. S2312 G1126
- Wide high power candidate collapses at low power -> wide scans are locators only. S2119
- Null while the positive control is also gone -> not interpretable. Recover sensitivity first. F232
- 3 to 4 noise-unit maxima or a bounded fit of noise -> not a carrier. Needs raw review, a repeat and a coherent Rabi. F1015 F1090
- Qubit lost after a warm cycle -> search both resonator slopes and every seed region. F983
- Carrier override left the sweep bounds on the old seed -> read the actual requested window in the log, and copy recorded start and stop for exact repeats. F857 S3436
- Clean peaks disagree between visits -> label `measured-below-gate`, rerun qspec each pass, never promote a qspec-only frequency. F798 F916
- Gaussian pulse carrier against rectangular qspec center -> compare like with like. Never move a line to match an old table. F726
- Sweep crossed 4915.2 MHz -> split the sweep by Nyquist zone of the qubit DAC. S2193 G984
- Center drifted 5594.22 to 5594.49 MHz within one session -> rerun qspec before pulse work. S1995
- Failed qspec wrote HDF5 that holds the string None -> open the file and check the arrays. S2132 G931

## Rabi and pulse calibration

- Flat Rabi from a stale qubit frequency or readout -> propagate the validated values into every override and log both. F433
- Rabi sweep override only relabelled the axis -> set `expt_cfg` before the Rabi object is built, then check the saved gain end points. F444
- Fitted pi lands in 0.6 to 0.7 but the curve is scatter (repeated 0.8482879 alias) -> require a visibly resolved oscillation. S2109 G896
- Pulse longer than the 16 bit length field (150 us is 92160 cycles) -> keep pulses at or under 100 us. F119 G3860
- Good Rabi but low SSF -> report both. Never restate contrast as (1+contrast)/2. F1167
- Rabi and SSF depend on each other at a new bias -> run raw IQ Rabi first with the bias-specific readout seed. S2083 S2109

## Single shot fidelity

- One grid maximum taken as the optimum -> alternate length and gain or frequency sweeps, repeat the chosen point, pick the shortest plateau length. F407 F744
- SSF moves faster than a serial grid -> interleave a same-grid reference, use balanced block order. F81
- Cycle budget reached while still improving -> more bounded cycles inside the qualified limits, never a higher ceiling. F3
- Repeat exceeds the gate -> keep both, look at shot blocks, no cherry picking. F899
- Calibration overrides leak between stages -> apply the coherence override right before coherence and audit in acquisition order. F666 F871
- Long SSF series slows down -> check the exact last saved path, do not glob a growing folder. F60
- SSF collapsed against an older run -> first rerun the last validated setting unchanged. If a standing limit forbids that setting, ask the user at once. G4539 G4774 (rule added 25 Nov)

## T1 and T2

- High R2 but the decay is not resolved in the window -> extend delays. Reject extrapolated values. F142 F885
- Delay quantisation looks like a detuning error -> use the saved time axis (0.2 us requested became 0.19856771 us). F130 F733 G2366 G2706
- Ramsey fit picks the weak quadrature or noise -> fit both I and Q. A short Ramsey correction is diagnostic only. F572 F1049 S2652
- Short noisy Ramsey moved the carrier 5438.366 to 5430.905 MHz -> keep the qspec carrier. No automatic frequency correction without a gate. S3496 G4250
- Two frequency fit with cancelling amplitudes -> reject. F69
- Slow fringe gives a short tau -> independent full window fit. Guesses are starts, never constraints. F819
- Result disagrees with an old value -> a comparison band is not a quality gate. Repeat with a fresh carrier, never tune toward the old value. F779 F836
- Higher average retry is worse -> choose by a criterion fixed beforehand, not by order. F621
- Old HDF5 without Population -> label normalised quadrature as diagnostic. F740

## Flux bias and Yokogawa

- Controller died and Yoko 3 stayed at +7.5 mA with output on -> launch detached under a guard, read back the sources after any crash, discard the partial row. F331 S2505 G1328
- Requested against recorded current -> analyse the recorded current. `SOURCE:LEVEL?` is the source setting, not an ammeter. F222
- `current_mA=0` while the other Yoko is not at zero -> store under both actual currents. F790
- Q4 center at Yoko 3 +10 mA moved 0.126 MHz between visits and 0.028 MHz with matched 600 s settling, cause not assigned -> settle 600 s and run qspec every pass. S3278 F793
- Pilots pass but the map fails -> no retries, pooling or extra averaging. Lines found at different currents are not a splitting. F175 F254 F922
- Search for physical Q5 ran 23 runs before the design record was found -> settle the qubit map from design records or the user before a search run. G2799 G3040

## Zeno and Stark experiments

- Map loses contrast -> check raw signed IQ, the actual reset, and timing after the pulse (`readout_ringdown_delay`). F1062
- Long run kept its startup carrier and pi for about 23 h while reference SNR fell 11.43 to 1.65 -> refresh and gate qspec and Rabi before each map. F1080 F1106 S3517
- Guard stops a real line -> judge repetition, sampling, raw IQ and an independent Rabi together. Keep the guards. F1116
- Fit exception threw away an acquired map -> save raw first. An optional fit never blocks saving. F1144 G4537
- Long run started at SSF 8.6 percent -> no long run below the SSF gate. F1175 G4263
- TLS or beating claims -> none without a distinguishing experiment. F1078 F1094

## Data saving and paths

- Plot saved but no HDF5 (268 character path) -> compute the longest path before launch and check both files after. F373 S2026 G737
- Placeholder HDF5 after a fit failure -> require finite raw arrays and all fields. F700 F705
- Time limit hit before the final batch save -> use `PUCQ4_PHASE_CHECKPOINTS` (one row, one pass only). F40 G145
- HDF5 `Dates` can be the save time -> use one PC clock for timing claims. F901 F848
- Temporary plan settings missing from the default round robin -> persist acquisition profiles and deep copy Exp Config. F710
- Config layers disagree -> inspect the actual execution path. Report old, new, unit and evidence. F415
- Seed became a fact -> label seeds, use NaN for unknown, apply the stop rule. F389 F1018
- A run folder rename rewrote 46 old paths in 25 files -> change only the new acquisition root. Never edit historical paths. G3446 G3492
- A hidden second dictionary silently replaced the main resonator settings -> one owner per setting, no hidden override. G3398
- Opening an HDF5 while it is being saved gives a bad object header -> wait for the stage to finish. G3598

## Long runs and launching

- Run tied to a terminal was killed mid-row -> start long runs hidden and detached from an activated Conda runtime. F324 F590 G1328
- Seven-experiment child hit a 3600 s limit -> budget from counts. Later bound 10800 s with a 600 s no-progress watchdog. Kill only the child tree you own. F29 G154
- Status file lock on `M:` (WinError 5) ended a 24 h series -> status writes are nonfatal with retries (`tools/ssf_progress.py`). Raw files are the truth. F1021 G4160
- Import started a lab loop -> read acquisition scripts as text or with `ast` only. F315
- A running process keeps the code it imported -> edit shared config only between children. S3308 G2456
- Stopping a queue -> stop only the parent you own so the running child finishes and saves. Never overlap two queues. G119

## Fits and acceptance

- Blind scanning in circles -> every run names a hypothesis and a control. Otherwise stop and ask. F398 G749
- Physical cause claimed -> none without an experiment that tells the causes apart. F748 F896
- Fit half-width seed larger than the sweep span wrote placeholders -> keep fit seeds inside the bounds and test the fit on a synthetic line first. S2978 G2184
- Stretched exponential or echo above 2 T1 -> report as phenomenological. Do not force offsets. F1005

## Reports and decks

- Loader drops a scalar fit or a null metric -> allow one finite value and render "not recorded". F466 F642
- Heatmap hides a narrow line -> one continuous sweep, no scaling per row, HDF5 stays the source. F504
- Slide replacement or reorder went wrong -> unique ZIP entries, compare in presentation order, render natively. F684 F956
- Plot stretched on a slide -> keep the aspect ratio, fit inside the cell. G816
- T2E title said T2R -> identify by dataset. F676
- 29 backup decks piled up -> backups only in `_PowerPoint_archive`. G2087

## Environment and tooling

- PowerShell 5 corrupts UTF-8 on whole-file rewrites -> never rewrite a file through `Get-Content` and `Set-Content`. Use the edit tool, or `IO.File.ReadAllText` with an explicit UTF-8 writer. Keep harness files ASCII. F193 G2986
- `.ps1` files are blocked by execution policy -> use inline commands. Never change the policy. F649 F919
- `python -c` loses embedded double quotes in PowerShell 5.1 -> put the Python in a small `.py` file. G2458
- cp1252 console errors -> set `PYTHONIOENCODING=utf-8`. G1629
- PowerShell splits an unquoted comma list into separate arguments -> put comma lists in double quotes.
- `Set-Content -Encoding UTF8` adds a BOM -> write JSON as UTF-8 without BOM. G409
- Forcing `core.autocrlf=false` shows whole-file whitespace errors -> use the repo's normal git config for `git diff --check`. G2409
- Notebook shows QUEUED with the kernel down -> that is not execution. F1045

## Agent process

- Failure -> classify it (task, context, environment, verification, state), fix the narrowest layer, no extra power, span or reps. F102 F107
- Denied helper or approval -> read the source, do not work around the denial. F951
- Design evidence looks missing -> search the design archives before a search run. F943
- Status text said more than was measured -> raw configs and raw data are the authority. Say what was measured, nothing more. G4307
