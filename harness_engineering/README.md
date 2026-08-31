# Harness engineering entry point

This folder is the repository's durable execution harness. It turns the five
failure layers from the Walking Labs tutorial into repo-specific instructions,
safe tools, verification feedback, and persistent experiment state. It does not
authorize a hardware run.

## Mandatory start sequence

1. Inspect `git status`; assume existing modifications and data belong to the
   user.
2. Read [ENVIRONMENT_AND_SAFETY.md](ENVIRONMENT_AND_SAFETY.md), even for an
   offline task.
3. Write or mentally resolve the brief in [TASK_TEMPLATE.md](TASK_TEMPLATE.md):
   goal, non-goals, scope, evidence, and command-verifiable completion criteria.
4. Use the routing table below for only the context the task needs.
5. Check the live source files before relying on a copied number in a document.
6. Verify at the highest safe rung in [VERIFICATION.md](VERIFICATION.md).
7. Add durable discoveries and failed approaches to `docs/harness-log.md`, then
   update the relevant harness document if the lesson changes future behavior.

## Five-layer implementation

| Tutorial defense layer | Repository implementation |
| --- | --- |
| Task specification | [TASK_TEMPLATE.md](TASK_TEMPLATE.md) makes behavior, non-goals, authority, and Definition of Done explicit. |
| Context provision | [REPOSITORY_MAP.md](REPOSITORY_MAP.md) and [EXPERIMENT_CONTRACTS.md](EXPERIMENT_CONTRACTS.md) expose architecture and local conventions. |
| Execution environment | [ENVIRONMENT_AND_SAFETY.md](ENVIRONMENT_AND_SAFETY.md) records the PyCharm/Conda setup and separates offline work from lab operation. |
| Verification feedback | [VERIFICATION.md](VERIFICATION.md) and `tools/harness_check.py` provide an executable, hardware-safe first rung and an evidence ladder. |
| State management | [PUCQ4_STATE.md](PUCQ4_STATE.md), [FAILURE_MODES.md](FAILURE_MODES.md), and `docs/harness-log.md` preserve validated state and diagnostic history. |

## Task routing

| Task | Read before acting |
| --- | --- |
| Explain or review code | Repository map, then the directly relevant source files |
| Add or change `section_*.py` | Experiment contracts and verification |
| Add or change orchestration | Repository map, experiment contracts, safety, verification |
| Change PUCQ4 config or analyze a run | PUCQ4 state, sources, verification |
| Run QICK/Pyro/Visdom, an experiment script, or a Yoko | Safety, PUCQ4 state, task template hardware envelope, verification |
| Diagnose a failure | Failure modes, chronological log, then the relevant layer document |
| Prepare the PUCQ4 PowerPoint or Yoko maps | PUCQ4 state and sources; use only validated artifacts |

## Source-of-truth order

Use this order when facts disagree:

1. A newly verified instrument readback or raw HDF5 result with its exact run
   path and configuration.
2. Current executable values in `system_config.py`, `expt_config.py`, and the
   explicitly selected orchestration script.
3. Accepted results in [PUCQ4_STATE.md](PUCQ4_STATE.md) and the chronological
   log.
4. The characterization PowerPoint and upstream driver as reference evidence.
5. Comments, search seeds, rejected fit candidates, and chat recollection.

Executable config is not automatically validated calibration. Conversely, a
validated measurement does not change executable config until the relevant
source files are deliberately updated and reviewed.

## Harness maintenance rule

Treat every repeat failure as a harness defect. Classify it as specification,
context, environment, verification, or state; fix the narrowest responsible
layer; re-run the safe check; and record whether the failure reproduced. Do not
hide critical safety rules only in a long log, and do not turn transient guesses
into permanent instructions.

For a large or long-running objective, work in bounded
inspect/design/change/review/verify blocks. After each meaningful accepted or
rejected result, persist the evidence, current safe hardware state, and exact next
step; do not wait for the chat session to end. This makes a new session a resume,
not a fresh rediscovery pass, and prevents a shrinking context window from
becoming permission to skip verification.

The machine-readable contract is `manifest.json`. The offline verifier checks
that its required documents exist, that `AGENTS.md` points here, and that the
hardware-authorization guard remains enabled.
