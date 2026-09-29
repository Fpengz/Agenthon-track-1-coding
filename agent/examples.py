"""Few-shot reference examples from the team's own verified solutions to OTHER units.

Track 1 rule 8 allows an image to carry the team's own solutions to public Development units and
use them as in-context examples when solving OTHER units -- never the unit a solution answers,
however the unit is recognised (by id or by content). So an example is excluded when it shares
the task's unit id or instruction hash, or when its instruction is a near-duplicate of the task's
(some public units are re-issues of each other). Canary-like GUIDs are stripped from scripts.

The library (``agent/examples/library.jsonl``) is built by ``agent.main build-examples`` from
batch runs whose outputs passed the unit's checker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pathlib
import re
from collections import Counter
from typing import Any

from agent.knowledge import read_card

logger = logging.getLogger(__name__)

LIBRARY = pathlib.Path(__file__).resolve().parent / "examples" / "library.jsonl"
NEAR_DUPLICATE_JACCARD = 0.5
MIN_RELEVANCE = 0.12
MAX_SOLUTION_CHARS = 9000
MAX_SUMMARY_CHARS = 1200
_GUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_TOKEN = re.compile(r"[a-z][a-z0-9_]+")


def instruction_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def load_library(path: pathlib.Path = LIBRARY) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _cosine_ranker(docs: list[str]):
    counts = [Counter(_tokens(d)) for d in docs]
    df = Counter(t for c in counts for t in c)
    idf = {t: math.log((1 + len(docs)) / (1 + n)) + 1 for t, n in df.items()}

    def vec(c: Counter[str]) -> dict[str, float]:
        v = {t: (1 + math.log(n)) * idf.get(t, 1.0) for t, n in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    vectors = [vec(c) for c in counts]

    def score(query: str) -> list[float]:
        q = vec(Counter(_tokens(query)))
        return [sum(q.get(t, 0.0) * w for t, w in v.items()) for v in vectors]

    return score


def select_example(
    task_dir: pathlib.Path, instruction: str, library: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """The most relevant permitted example for this task, or None."""
    library = load_library() if library is None else library
    if not library:
        return None
    card = read_card(task_dir)
    unit_id = card.get("task", {}).get("id")
    meta = card.get("metadata", {})
    own_hash = instruction_hash(instruction)
    permitted = [
        ex
        for ex in library
        if ex.get("unit_id") != unit_id
        and ex.get("instruction_sha256") != own_hash
        and _jaccard(ex.get("instruction", ""), instruction) < NEAR_DUPLICATE_JACCARD
        and len(ex.get("solution", "")) <= MAX_SOLUTION_CHARS
    ]
    if not permitted:
        return None
    docs = [f"{ex.get('instruction', '')} {' '.join(ex.get('tags', []))}" for ex in permitted]
    query = f"{instruction} {meta.get('category', '')} {' '.join(meta.get('tags', []))}"
    scores = _cosine_ranker(docs)(query)
    best = max(range(len(permitted)), key=scores.__getitem__)
    if scores[best] < MIN_RELEVANCE:
        return None
    logger.info("Reference example: %s (relevance %.2f)", permitted[best]["unit_id"], scores[best])
    return permitted[best]


def format_example(example: dict[str, Any] | None) -> str:
    if not example:
        return ""
    summary = example.get("instruction", "")[:MAX_SUMMARY_CHARS].rstrip()
    return f"""
### REFERENCE EXAMPLE (a verified solution to a DIFFERENT task; use it only as a guide to
### structure and conventions -- do NOT copy its logic, parameters, file names or values):
Task excerpt:
{summary}
…

```python
{example.get("solution", "").strip()}
```
"""


def harvest(run_dirs: list[pathlib.Path], units_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Examples from checker-passed units that saved their accepted script."""
    found: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        for row in summary.get("results", []):
            script = run_dir / row["unit"] / "meta" / "transcripts" / "solution.py"
            if row.get("status") != "passed" or not script.is_file():
                continue
            unit_dir = units_dir / row["unit"]
            instruction = (unit_dir / "instruction.md").read_text(encoding="utf-8")
            meta = read_card(unit_dir).get("metadata", {})
            solution = _GUID.sub("<guid-removed>", script.read_text(encoding="utf-8"))
            candidate = {
                "unit_id": read_card(unit_dir).get("task", {}).get("id", row["unit"]),
                "instruction_sha256": instruction_hash(instruction),
                "category": meta.get("category", ""),
                "tags": meta.get("tags", []),
                "instruction": _GUID.sub("<guid-removed>", instruction),
                "solution": solution,
                "source_run": run_dir.name,
            }
            previous = found.get(candidate["unit_id"])
            if previous is None or len(solution) < len(previous["solution"]):
                found[candidate["unit_id"]] = candidate  # keep the most compact solution
    return sorted(found.values(), key=lambda ex: ex["unit_id"])
