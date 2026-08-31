# Environment and hardware safety

## Authorization boundary

Treat experiment execution as physical lab operation.

Without an explicit request for a lab run, do not:

- instantiate `QICK_experiment` or connect to the Pyro/QICK proxy;
- run or import round-robin, numbered PUCQ4, or other acquisition scripts;
- start/check Visdom as part of an experiment;
- instantiate the Yokogawa driver, change current/output/range, or assume its
  present state;
- change RF attenuators or filters;
- create experiment folders or write data on `M:`.

Reading source, parsing Python with `ast`, inspecting Git state/diffs, and
analyzing already supplied local artifacts are offline actions. Hardware-free
verification never proves instrument behavior.

## Python environment

Target Python 3.9 syntax. Before every Python/pip/pytest/formatter command, obtain
the interpreter configured for this PyCharm module; do not infer it from PATH.

As observed on 2026-08-28, PyCharm used:

`C:\Users\Ma Quantum Lab\anaconda3\python.exe` (Python 3.9.13)

Directly invoking that executable has produced a NumPy DLL-load failure because
the Conda runtime was not activated. The working pattern was:

```powershell
cmd.exe /d /c "call ""C:\Users\Ma Quantum Lab\anaconda3\Scripts\activate.bat"" ""C:\Users\Ma Quantum Lab\anaconda3"" && ""C:\Users\Ma Quantum Lab\anaconda3\python.exe"" tools\harness_check.py"
```

Re-query the IDE environment first; the recorded path is diagnostic history, not
permission to assume the SDK forever. Do not install dependencies merely to make
an unrelated check pass without user approval.

## Known connection points

- `system_config.QICK_experiment` creates folders and calls `socProxy.makeProxy`
  during construction.
- `socProxy.py` currently defaults to Pyro name server `192.168.1.139:8888` and
  proxy name `myqick`.
- `expt_config.py` currently selects fridge `BOB` and six list rows.
- Yoko 3: `192.168.1.73`, physical Q4/DC D4, resonator M5/Python index 4.
- Yoko 4: `192.168.1.77`, physical Q6/DC A5, resonator M6/Python index 5.

These addresses are operational context, not credentials and not authorization
to connect.

## Hardware preflight

Before an authorized QICK run:

1. Inspect `git status` and the complete run-control block as text.
2. Record the task template's hardware envelope.
3. Confirm physical-qubit/resonator/Python-index mapping.
4. Confirm current executable `system_config.py`, `expt_config.py`, and any
   per-run overrides; stale comments do not count.
5. Compute the full Windows plot and HDF5 paths; keep each below 260 characters.
6. Confirm disk/data destination and that no existing raw artifact will be
   overwritten.
7. Define a bounded duration/progress signal and abort condition.
8. State the intended safe final hardware state.

Abort on QICK/Pyro connection, timing, buffer, or acquisition errors; unexpected
current/output readback; an out-of-envelope sweep; a clearly saturated/flat trace
that invalidates the planned continuation; or data-save failure before beginning
the next dependent stage.

## Yokogawa rules

Only physical Q4/M5 and physical Q6/M6 are documented as flux tunable on this
chip. For the zero-current baseline:

- query identity, mode, output, level, range, and voltage limit;
- require current-source mode;
- use the 0.01 A (10 mA) range;
- ramp to `0.0` A at `0.0005` A/s unless a separately authorized protocol says
  otherwise;
- verify the requested current within a recorded tolerance;
- turn output off at zero and verify it.

For characterization sweeps, the supplied deck visibly covers -10 to +10 mA.
Treat that as the hard task boundary unless the user supplies new device evidence
and explicitly authorizes another range. The local driver's generic 15 mA guard
does not authorize a 15 mA experiment.

Never jump current, silently enable output, extrapolate from a search seed, or
leave a source in an unknown state after an exception.

## Data safety

- Preserve user code changes and raw data. Never delete or rewrite generated
  measurements as cleanup.
- Generated measurement data, copied deck extraction artifacts, host credentials,
  and lab secrets do not belong in Git.
- After every acquisition, verify the expected plot and HDF5 independently.
- If save fails after acquisition, record exactly which in-memory/raw evidence was
  lost and do not claim the run is reproducible.
