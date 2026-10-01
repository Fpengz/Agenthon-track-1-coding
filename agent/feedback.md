# Execution feedback

The entrant can run local probes and repair scripts inside one `solve` invocation. The final
deliverables are graded once after the agent exits. Internal House calls and script executions
consume the same request and runtime budgets; the official checker supplies no feedback to this
loop.

```mermaid
flowchart TD
    G[House: generate or repair] --> Q{Numerical probe requested?}
    Q -->|yes| P[Execute local probe in scratch output copy]
    P --> O[Record stdout, errors and output version]
    O --> G
    Q -->|complete solution| E[Execute solution]
    E -->|failure| F[Previous code, exception and retained observations]
    F --> G
    E -->|clean outputs| S[Snapshot outputs]
    S --> R[House: review specification, code and outputs]
    R -->|probe| V[Probe on fresh snapshot copy]
    V --> R
    R -->|corrected code or edits| E
    R -->|defects without code| G
    R -->|complete PASS| D[Final deliverables]
```

Probe observations are required task context in all prompt variants. A probe requested during
repair or review returns to that pending operation, retaining its previous code, error, verdict
state and edit base. A response containing both a probe and a proposed decision runs the probe
first while affordable; the next model response makes the decision with the observations.

Each probe gets a fresh copy through `OUTPUT_DIR`. Its results name the phase and solution turn
they inspected. Earlier observations remain historical evidence after edits; they are not a
validation of newly generated outputs. The prompt asks the model to check the task's definitions,
compare independently calculated and observed numbers, and print the discrepancy. A probe's
claim of success is not an official verdict.

Truncated probes, including responses with an early closed block, do not execute. Probes cannot
become solution continuations. Tool exhaustion asks the model to finish its pending operation;
deadline or request exhaustion retains the last clean snapshot when one exists. Its correctness
still depends on the offline checker.

## Reproduced defects

Four failures were reproduced on synthetic tasks with a recorded model client and real local
Python execution, then fixed:

- Observations were appended only to the immediate generation prompt. Repair and review rebuilt
  the original task prompt and lost them. Observations now travel in every task-prompt variant.
- The dispatcher accepted probes only during generation. A review probe was treated as a missing
  verdict and a repair probe as missing code. Both now execute and resume their original phase.
- Probe extraction preceded the truncation check. A closed early probe in a token-limited reply
  executed despite incomplete generation. Such responses now return to the pending operation.
- Repeated failures were compared using the final variable-dump line. Different locals hid the
  same exception, and identical locals falsely equated different exceptions. The signature now
  uses the exception tail before the structured error context.

The regression command exercises numerical evidence retention, context reduction, copied-output
inspection, assertion feedback, repair/review resumption, truncation, continuation, exhaustion,
and repeated exceptions:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_evidence.py
```

## Experiment

Keep `AGENT_EXPLORE` off by default until measured. Compare `AGENT_EXPLORE=2` with `0`, using the
same committed image, units, budgets and simultaneous load, with at least two runs per arm.
Inspect whether probes perform numerical computations and read outputs; a pass-rate change
without those observations does not demonstrate that execution feedback helped. Record fixed
denominator pass@1, request/runtime costs, and the phases in which probes executed. Launch long
runs with `setsid nohup`; wait for prior cohorts to finish before adding model load.
