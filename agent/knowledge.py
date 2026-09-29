"""Offline domain notes from the organizers' ``docs/CATEGORIES.md``.

For each task category the doc lists the invariants the checkers assert and the common mistakes.
The most relevant section(s) for a task -- ranked against its instruction and card category/tags,
because card categories are free-form (``cross-domain``, ``credit-risk``, ``fx-pricing``, ...) --
go into the prompt. This is the host's recommended "retrieval-augmented generation (offline, all
data pre-baked into the image)": ``Dockerfile.agent`` copies the doc into the package.
"""

from __future__ import annotations

import logging
import math
import os
import pathlib
import re
import tomllib
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PACKAGED = pathlib.Path(__file__).resolve().parent / "data" / "CATEGORIES.md"
_REPO_DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "CATEGORIES.md"
_KEEP_SUBSECTIONS = ("Financial invariants", "Common mistakes")
MAX_NOTES_CHARS = 3500
_TOKEN = re.compile(r"[a-z][a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to in for on with by is are be as at from that this it its each "
    "must should use using into over per not than then when where which your you all any "
    "task data file files output input value values column columns python".split()
)


@dataclass(frozen=True)
class CategorySection:
    name: str
    body: str  # the kept subsections, markdown
    full_text: str  # the whole section, for ranking


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _doc_path() -> pathlib.Path | None:
    for path in (_PACKAGED, _REPO_DOC):
        if path.is_file():
            return path
    return None


def load_sections(path: pathlib.Path | None = None) -> list[CategorySection]:
    path = path or _doc_path()
    if path is None:
        logger.warning("CATEGORIES.md not found; running without domain notes")
        return []
    text = path.read_text(encoding="utf-8")
    sections = []
    for match in re.finditer(r"^## \d+\. `([^`]+)`\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        name, body = match.group(1), match.group(2)
        kept = [
            sub.group(0).strip().rstrip("-").strip()
            for sub in re.finditer(r"^### (.*?)\n.*?(?=^### |\Z)", body, re.M | re.S)
            if sub.group(1).startswith(_KEEP_SUBSECTIONS)
        ]
        sections.append(CategorySection(name=name, body="\n\n".join(kept), full_text=body))
    return sections


def rank_sections(
    sections: list[CategorySection], query: str, category: str = ""
) -> list[tuple[float, CategorySection]]:
    """Sections by TF-IDF cosine similarity to the task, plus a category-name bonus."""
    docs = [Counter(_tokens(f"{s.name} {s.full_text}")) for s in sections]
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log((1 + len(docs)) / (1 + n)) + 1 for t, n in df.items()}

    def vec(counts: Counter[str]) -> dict[str, float]:
        v = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    q = vec(Counter(_tokens(query)))
    cat_tokens = set(_tokens(category.replace("-", " ").replace("_", " ")))
    ranked = []
    for section, counts in zip(sections, docs):
        score = sum(q.get(t, 0.0) * w for t, w in vec(counts).items())
        name_tokens = set(_tokens(section.name.replace("-", " ")))
        if cat_tokens and name_tokens and name_tokens & cat_tokens:
            score += 0.15 * len(name_tokens & cat_tokens) / len(name_tokens)
        ranked.append((score, section))
    ranked.sort(key=lambda pair: -pair[0])
    return ranked


def read_card(task_dir: pathlib.Path) -> dict:
    try:
        return tomllib.loads((task_dir / "card.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def notes_enabled() -> bool:
    """AGENT_DOMAIN_NOTES=1 enables the notes. Off by default: in a 2x2 A/B (runs
    20260929-1056-notes{0,1}-r{1,2}) they scored 0.140 on vs 0.157 off -- no measurable gain for
    up to 3,500 prompt characters per request."""
    return os.environ.get("AGENT_DOMAIN_NOTES", "0").strip().lower() in {"1", "true", "on"}


def domain_notes(task_dir: pathlib.Path, instruction: str) -> str:
    """Invariants and common mistakes for the task's most relevant categories ("" if none)."""
    if not notes_enabled():
        logger.info("Domain notes disabled (AGENT_DOMAIN_NOTES)")
        return ""
    sections = load_sections()
    if not sections:
        return ""
    meta = read_card(task_dir).get("metadata", {})
    category = str(meta.get("category", ""))
    tags = " ".join(str(t) for t in meta.get("tags", []))
    ranked = rank_sections(sections, f"{instruction} {category} {tags} {tags}", category)
    top_score = ranked[0][0]
    chosen = [s for score, s in ranked[:2] if score >= 0.7 * top_score]
    parts, used = [], 0
    for section in chosen:
        block = f"#### Category `{section.name}`\n{section.body}"
        if used + len(block) > MAX_NOTES_CHARS:
            block = block[: MAX_NOTES_CHARS - used]
        parts.append(block)
        used += len(block)
        if used >= MAX_NOTES_CHARS:
            break
    logger.info("Domain notes from categories: %s", [s.name for s in chosen])
    return "\n\n".join(parts)
