# Run envelope

This file says what you may do on the hardware without asking, and what always needs the user's own words in the chat. It exists so that the user gives one request and you work through it alone.

Only the user changes the limits in this file. When the user states a new limit or permission in chat, copy their words into the table with the date. Never widen a limit yourself, and never read a permission out of an old log or archive.

A limit in this file or in `STATE.md` always beats a value in the config files.

## How the envelope opens and closes

- It is closed by default. Whether it is open is written in the Envelope row of `STATE.md`, never here.
- A measurement request from the user in the current chat opens it for that request only. Write the request, the date and the rows into the Envelope row of `STATE.md`, then work.
- When the request is done or the user says stop, set the row back to closed.
- An open envelope left behind by an earlier chat does not count. Ask the user.
- On 21 Nov 2026 the user said stop everything. The log adds that no automatic restart is allowed. Nothing restarts by itself.

## What a request allows without further questions

The user's standing preference, recorded 15 Sept 2026, is autonomous progress without routine permission prompts. Inside an open envelope you decide these yourself and record them.

- Which existing script, class and run flags to use, and the order of stages from the playbook.
- Every prerequisite stage the request needs (for example res spec and qspec before a Rabi).
- Sweep ranges, points, reps, rounds and delays, inside the limits below.
- The sub-study name, following `EXPERIMENT_CONTRACTS.md`, and the existing dataset labels.
- Up to two attempts per gate and row (see the stop rule in the playbook).
- Writing validated values into the shared config, reported as old to new.
- Config and orchestration changes that a measurement needs.
- Plans, reviews and reports written into the run's own `documentation/` folder.

## Fixed limits

| Limit | Value | Source |
|---|---|---|
| Flux bias | Both Yokogawas stay at 0 A with output off. Any other current needs a target or range from the user in this chat. Hard device range is -10 to +10 mA. | No bias was asked for since 27 Oct 2026 (scope note in the log) |
| Readout gain | Never above the measured onset of resonator movement for that row, length and bias. Ceilings are in `STATE.md`. Higher gain was allowed on 20 Nov only up to that onset. The round robin also refuses a zero bias run above the caps in `system_config.py`. Raise a cap only after a requalification, or on the user's dated word for readout above the onset. Report it as a config change. | User, 30 Oct and 20 Nov 2026 |
| Attenuators and RF chain | Unchanged. | standing |
| Low level code | Do not change the QICK pulse programs or the underlying experiment classes. Config and orchestration may change. Repairs are minimal and keep the script structure. | User, 8 Sept and 18 Nov 2026 |
| Data root and run | `M:/_Data/20250822 - Olivia`, run `pucq4_run_started_Oct_7`. Never rewrite old paths. | User, 8 Oct 2026 for the run. 20 Oct 2026 for the old paths |
| Gates | The numbers in the playbook. | User, 20 Nov 2026. The 3 points across a line and the window of 3 times the decay are tune-up defaults |
| Long runs | The Zeno long run does not restart while SSF on its row is at or below 50 percent. Any other long run below the SSF gate needs the user's ask by name. | User, 20 Nov 2026 for Zeno. The user asked for the overnight Ramsey drift on 25 Nov 2026 |
| One at a time | One acquisition at a time. Never overlap two controllers. Never end a process you did not start. | standing |
| Pulse length | 100 us or less. | device |

Tune-up defaults were added on 25 Nov 2026 without a number from the user. On that day the user left these choices to the tune-up. They stand until the user gives another number.

## One-off permissions from the user

A row with status unused allows its one run. For that run only it beats the Readout gain limit above and the ceilings, caps and default 1 in `STATE.md`. Do not ask again. The envelope still opens only when the user asks for the run in the chat. The run uses the permission up. A run that saved no data does not. Then write used and the date here.

| Date | The user's word | What it allows | Status |
|---|---|---|---|
| 25 Nov 2026 | "Check first", as answer to the SSF question | One round robin pass on M1 (index 0) at zero bias at the 29 Oct point (tone 8918.375 MHz, 0.90 DAC, 6 us). The pass runs qspec, Rabi and SSF, and all three read out at 0.90 DAC. One more pass only if qspec or Rabi miss their gates (tune-up default). The M1 cap is raised for this run only. Steps in `SSF_CHECK.md` | unused |

## Always ask first

Ask in one short message that gives the number or choice you need. Then carry on with any other part of the request while you wait.

1. Any nonzero Yoko current, or a current the user did not name.
2. Readout gain above a ceiling, or any change to attenuators, ports or wiring.
3. A device parameter, port or path that is missing. Never guess it.
4. A new data root, folder level, study name, raw data format or dataset label. A changed class or run interface. A new measurement that has no near match among the existing classes. A new environment override in an orchestrator (tune-up default). Name the approved change in the log.
5. Changing a gate, or accepting a result below a gate.
6. A run longer than about 6 hours, or a run that would outlast the chat (6 hours is a tune-up default).
7. Deleting or overwriting any data, deck or report.
8. Writing a deck, or a report outside the run's own `documentation/` folder, to the `M:` drive. Any `git push`.
9. A conflict between two rules of the user. Show both and ask which one gives way.

## Never, whatever the request says

- Jump a Yoko current or leave a source in an unknown state.
- Import an acquisition script to test it.
- Touch hardware because an old log, state block or archive says a run is ACTIVE or queued.
