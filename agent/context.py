"""Safe task context: what the model may see of the task's metadata, and canary hygiene.

59 of the 86 public instructions carry the unit's canary GUID -- in an HTML comment, or in a
``# ...-canary GUID <id>`` heading line the model reads as part of the task. Any canary in a
deliverable is a g2 ``CONTAMINATION_CANARY`` disqualification, so the model never sees it:
instructions are sanitised before prompting, and deliverables are scanned before acceptance.

Card facts useful for choosing a method (category, time limit, compute, data cutoff) are passed
as an explicit, small allow-list; everything else on the card (canary, author, manifest,
provenance) stays out of the prompt.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

from agent.knowledge import read_card

_GUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_CANARY_LINE = re.compile(r"^.*\bcanary\b.*$\n?", re.I | re.M)
_SCAN_BYTES = 50_000_000


def task_canaries(task_dir: pathlib.Path, instruction: str) -> set[str]:
    """This task's canary GUIDs: the card's, plus any GUID on an instruction canary line."""
    found = set()
    guid = read_card(task_dir).get("contamination", {}).get("canary_guid")
    if guid:
        found.add(str(guid).lower())
    for line in _CANARY_LINE.findall(instruction):
        found.update(g.lower() for g in _GUID.findall(line))
    return found


def sanitize_instruction(instruction: str, canaries: set[str]) -> str:
    """The instruction without HTML comments, canary lines or canary GUIDs."""
    text = _HTML_COMMENT.sub("", instruction)
    text = _CANARY_LINE.sub("", text)
    for guid in canaries:
        text = re.sub(re.escape(guid), "<removed>", text, flags=re.I)
    return text.lstrip("\n")


def canary_leaks(out_dir: pathlib.Path, canaries: set[str]) -> list[str]:
    """Deliverables that contain a canary GUID (each one would disqualify the attempt)."""
    if not canaries or not out_dir.is_dir():
        return []
    leaks = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        if path.name in {"reward.json", "pytest_report.json"}:
            continue
        with path.open("rb") as fh:
            text = fh.read(_SCAN_BYTES).decode("utf-8", errors="ignore").lower()
        if any(guid in text for guid in canaries):
            leaks.append(path.name)
    return leaks


def card_facts(task_dir: pathlib.Path) -> str:
    """An explicit allow-list of card facts for the prompt ("" if there is no card)."""
    card: dict[str, Any] = read_card(task_dir)
    if not card:
        return ""
    meta, agent, env = card.get("metadata", {}), card.get("agent", {}), card.get("environment", {})
    cutoff = card.get("provenance", {}).get("data_cutoff")
    facts = [
        f"- Category: {meta['category']}" if meta.get("category") else "",
        f"- Difficulty: {meta['difficulty']}" if meta.get("difficulty") else "",
        f"- Data cutoff: {cutoff} (use no data after it)" if cutoff else "",
        f"- Agent time limit: {int(agent['timeout_sec'])} s for the whole task; keep the script's "
        "runtime well under a few minutes"
        if agent.get("timeout_sec")
        else "",
        "- Compute: "
        + ", ".join(f"{k}={env[k]}" for k in ("cpus", "memory", "gpu") if k in env)
        + "; no network"
        if env
        else "",
    ]
    return "\n".join(f for f in facts if f)
