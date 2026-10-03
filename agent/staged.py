"""Staged generation: a plan of short step scripts instead of one script that overruns the cap.

Every House reply is capped at 4,000 tokens, and low-effort reasoning still uses about half of
a long reply. Across 999 consensus unit runs, units whose replies were cut off 3+ times passed
11% of the time (498 runs), against 46% with no cut-off (293 runs); 49 of 86 units hit the cap
in at least half their runs, and the never-passed units with missing deliverables used 22-25
requests on continuations and rewrites. Here one request plans 2-8 steps (their files), each
step is generated and debugged as its own short script with the earlier steps' intermediate
files previewed, and the concatenation becomes the first script of the ordinary loop (review,
edits, guardrails and consensus are unchanged).

Prompts and plan handling live here; the request/execute loop is ``AgentSolver._staged_generate``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

MIN_STEPS = 2
MAX_STEPS = 8
MAX_DELIVERABLES_PER_STEP = 2  # the model ignores this limit in its plans, so it is enforced
MAX_SPLIT_STEPS = 12
STAGE_ENV = "STAGE_DIR"
_JSON_BLOCK = re.compile(r"```json\s*\n(.*?)```", re.S)

COMBINED_HEADER = f'''# Staged solution: the steps below run in order in one process. Intermediate files go to
# {STAGE_ENV} (a fresh scratch directory unless the environment already provides one).
import os as _agent_os
import tempfile as _agent_tempfile

_agent_os.environ.setdefault("{STAGE_ENV}", _agent_tempfile.mkdtemp(prefix="stage-"))
'''


@dataclass
class Step:
    name: str
    goal: str
    writes: list[str] = field(default_factory=list)

    def describe(self, number: int) -> str:
        writes = ", ".join(self.writes) or "(nothing listed)"
        return f"{number}. {self.name}: {self.goal}\n   writes: {writes}"


def build_plan_prompt(task_prompt: str, deliverables: list[str]) -> str:
    required = ", ".join(f"OUTPUT_DIR/{name}" for name in deliverables) or "(see the task)"
    return f"""{task_prompt}

---
### PLAN THE SOLUTION AS STEPS (do not write code yet)
A single script for this task does not fit in one reply, so the solution is built as
{MIN_STEPS}-{MAX_STEPS} SHORT Python scripts that run in order. Each step reads inputs from
TASK_DIR and/or files written by earlier steps, and writes its results as files: intermediate
data to STAGE_DIR (`os.environ["{STAGE_ENV}"]`), deliverables to OUTPUT_DIR. Group the work so
that every step is under ~150 lines and writes AT MOST TWO deliverables (split larger groups
into more steps). Typical order: load and clean inputs -> core computation(s)
-> derived results -> write every deliverable in the exact requested format.

Every required deliverable must be written by some step: {required}.
Prefer CSV/JSON/Parquet intermediates with explicit column names.

Reply with ONLY a JSON block:
```json
{{"steps": [{{"name": "<short_id>", "goal": "<what it computes, with the specification's conventions>",
  "writes": ["STAGE_DIR/<file>", "OUTPUT_DIR/<deliverable>"]}}]}}
```
"""


def parse_plan(text: str, deliverables: list[str]) -> list[Step] | None:
    """The plan's steps, with every expected deliverable assigned; None if unusable."""
    blocks = _JSON_BLOCK.findall(text)
    candidates = blocks[::-1] or [text[text.find("{") : text.rfind("}") + 1]]
    for raw in candidates:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        items = data.get("steps") if isinstance(data, dict) else data
        if not isinstance(items, list):
            continue
        steps = [
            Step(
                str(item.get("name") or f"step_{i + 1}"),
                str(item.get("goal") or ""),
                [str(w) for w in item.get("writes") or [] if isinstance(w, str)],
            )
            for i, item in enumerate(items)
            if isinstance(item, dict)
        ]
        if not MIN_STEPS <= len(steps) <= MAX_STEPS:
            continue
        written = {w.rsplit("/", 1)[-1] for s in steps for w in s.writes}
        steps[-1].writes += [
            f"OUTPUT_DIR/{d}" for d in deliverables if d.rsplit("/", 1)[-1] not in written
        ]
        steps = split_large_steps(steps)
        return steps if len(steps) <= MAX_SPLIT_STEPS else None
    return None


def split_large_steps(steps: list[Step]) -> list[Step]:
    """Split steps that write more than two deliverables (one such 5-deliverable step overran
    the 4,000-token cap on every reply, even without thinking). Intermediate files stay with
    the first part; each later part reads them like any later step."""
    result: list[Step] = []
    for step in steps:
        outputs = [w for w in step.writes if w.upper().startswith("OUTPUT_DIR/")]
        if len(outputs) <= MAX_DELIVERABLES_PER_STEP:
            result.append(step)
            continue
        others = [w for w in step.writes if w not in outputs]
        chunks = [
            outputs[i : i + MAX_DELIVERABLES_PER_STEP]
            for i in range(0, len(outputs), MAX_DELIVERABLES_PER_STEP)
        ]
        for part, chunk in enumerate(chunks, 1):
            names = ", ".join(c.split("/", 1)[1] for c in chunk)
            result.append(
                Step(
                    f"{step.name}_{part}",
                    f"{step.goal} -- PART {part} of {len(chunks)}: write only {names}",
                    (others if part == 1 else []) + chunk,
                )
            )
    return result


def build_step_prompt(
    task_prompt: str,
    steps: list[Step],
    index: int,
    done_code: list[str],
    stage_previews: str,
    previous_code: str = "",
    feedback: str = "",
) -> str:
    step = steps[index]
    plan = "\n".join(s.describe(i + 1) for i, s in enumerate(steps))
    earlier = "\n\n".join(
        f"# ---- step {i + 1}: {steps[i].name} (ran successfully) ----\n{code}"
        for i, code in enumerate(done_code)
    )
    repair = (
        f"""
### YOUR PREVIOUS VERSION OF THIS STEP FAILED
```python
{previous_code}
```
{feedback}
Fix it. Return the complete corrected step script.
"""
        if previous_code or feedback
        else ""
    )
    return f"""{task_prompt}

---
### STAGED SOLUTION
The solution is built as these steps, each a separate script run in order with the same
TASK_DIR and OUTPUT_DIR; intermediate files live in `os.environ["{STAGE_ENV}"]`:
{plan}
{f"{chr(10)}### EARLIER STEPS (already ran; do not repeat their work, read their files){chr(10)}```python{chr(10)}{earlier}{chr(10)}```" if earlier else ""}
{f"{chr(10)}### FILES IN STAGE_DIR / OUTPUT_DIR SO FAR{chr(10)}{stage_previews}" if stage_previews else ""}
### WRITE STEP {index + 1} ONLY: {step.name}
Goal: {step.goal}
It must write: {", ".join(step.writes) or "the files its goal requires"}.
A standalone script (its own imports and helpers), under ~150 lines, following every
specification detail that concerns this step. Read earlier results from files, never recompute
them. Never call sys.exit() or exit(). One ```python ... ``` block.
{repair}"""


def combine(step_code: list[str], steps: list[Step]) -> str:
    """One script running every step in order (the first script of the ordinary loop)."""
    parts = [COMBINED_HEADER]
    for i, code in enumerate(step_code):
        parts.append(f"\n# ===== step {i + 1}: {steps[i].name} =====\n{code.strip()}\n")
    return "\n".join(parts)


def missing_writes(step: Step, stage_dir, out_dir) -> list[str]:
    """Declared files the step did not produce."""
    missing = []
    for target in step.writes:
        root, _, name = target.partition("/")
        if not name:
            continue
        base = (
            stage_dir
            if root.upper() == STAGE_ENV
            else out_dir
            if root.upper() == "OUTPUT_DIR"
            else None
        )
        if base is not None and not (base / name).exists():
            missing.append(target)
    return missing
