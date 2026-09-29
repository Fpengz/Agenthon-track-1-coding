"""Mechanical output checks derived from the task instruction (``AGENT_SPEC_CHECKS``).

Instructions state the required columns of each table deliverable and the keys of each JSON
deliverable, in varied formats: ``### /output/x.json`` or ``### File 2: x.csv`` headings, markdown
column tables, ``Columns: `a, b` `` lines, and JSON examples with ``<float>`` placeholders. This
module extracts what it can per deliverable and reports only MISSING columns / key paths, which
is cheap, precise repair feedback for the ``t1.mislabeling`` failures. Extra columns or keys are
never flagged, and anything that cannot be parsed is simply not checked.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".pqt"}
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
_BACKTICKED = re.compile(r"`([^`]+)`")
_PLACEHOLDER = re.compile(r"<[^<>\n]*>")
_SECTION_CHARS = 6000


@dataclass
class DeliverableSpec:
    name: str
    columns: list[str] = field(default_factory=list)
    json_keys: list[str] = field(default_factory=list)  # dotted paths of required dict keys
    numeric_columns: list[str] = field(default_factory=list)  # Type column says float/int
    numeric_keys: list[str] = field(default_factory=list)  # JSON example value <float>/<int>/number
    row_count: int | None = None  # an exact "N rows" statement


def _section(instruction: str, name: str) -> str:
    """Text describing ``name``: from its first heading (or bold) mention to the next heading."""
    for heading in _HEADING.finditer(instruction):
        if name in heading.group(2):
            level = len(heading.group(1))
            rest = instruction[heading.end() :]
            nxt = re.search(rf"^#{{1,{level}}}\s", rest, re.M)
            return rest[: nxt.start() if nxt else _SECTION_CHARS][:_SECTION_CHARS]
    bold = re.search(rf"\*\*[^*\n]*{re.escape(name)}[^*\n]*\*\*", instruction)
    if bold:
        # A bold mention (often a bullet) describes the file in its own paragraph only: stop
        # at a blank line, the next bullet, or the next heading.
        rest = instruction[bold.end() :]
        nxt = re.search(r"\n\s*\n|\n\s*[-*]\s|\n#", rest)
        return rest[: nxt.start() if nxt else _SECTION_CHARS][:_SECTION_CHARS]
    return ""


_NUMERIC_TYPE = re.compile(r"\b(float|int|double|numeric|decimal|number)", re.I)
_ROW_COUNT = re.compile(r"(?<![\w.])(?:exactly\s+)?(\d{1,6})\s+rows?\b", re.I)
_NOT_EXACT = re.compile(
    r"(at most|up to|at least|max(imum)?|min(imum)?|about|approx\w*|~|<|>|≤|≥)\s*$", re.I
)
_NUMERIC_PLACEHOLDER = re.compile(r"<\s*(float|int|integer|number|double|decimal)[^<>\n]*>", re.I)


def _numeric_columns(section: str) -> list[str]:
    """Columns whose markdown-table Type cell names a numeric type."""
    lines = section.splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip().lower() for c in line.strip().strip("|").split("|")]
        if not re.match(r"^\|\s*(column|field|name|col)\b", line.strip(), re.I):
            continue
        type_idx = next(
            (j for j, c in enumerate(cells) if c in {"type", "dtype", "data type"}), None
        )
        if type_idx is None:
            return []
        numeric = []
        for row in lines[i + 2 :]:
            if not row.strip().startswith("|"):
                break
            parts = row.strip().strip("|").split("|")
            if len(parts) <= type_idx:
                continue
            name = (_BACKTICKED.findall(parts[0]) or [parts[0].strip()])[0].strip()
            if _NUMERIC_TYPE.search(parts[type_idx]) and not re.search(
                r"str|object|date|bool", parts[type_idx], re.I
            ):
                numeric.append(name)
        return numeric
    return []


def _row_count(section: str) -> int | None:
    for match in _ROW_COUNT.finditer(section):
        if not _NOT_EXACT.search(section[max(0, match.start() - 14) : match.start()]):
            return int(match.group(1))
    return None


def _numeric_json_keys(section: str) -> list[str]:
    block = re.search(r"```json\s*\n(.*?)```", section, re.S)
    if not block:
        return []
    text = _NUMERIC_PLACEHOLDER.sub("0", _mark_dynamic_keys(block.group(1)))
    text = _PLACEHOLDER.sub("null", text)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",\s*\.\.\.\s*", "", text).replace("...", "")
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    paths: list[str] = []

    def walk(obj: object, prefix: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == _DYNAMIC_KEY:
                    continue
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    paths.append(path)
                walk(value, path)

    walk(data, "")
    return paths


def _table_columns(section: str) -> list[str]:
    # Markdown table whose header starts with Column/Field/Name: first-cell names.
    lines = section.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\|\s*(column|field|name|col)\b", line.strip(), re.I):
            names = []
            for row in lines[i + 2 :]:
                if not row.strip().startswith("|"):
                    break
                first = row.strip().strip("|").split("|")[0]
                found = _BACKTICKED.findall(first) or [first.strip()]
                names.extend(n.strip() for n in found if n.strip())
            if names:
                return names
    # "Columns: `a, b, c`" or "Columns: `a`, `b`"
    match = re.search(r"columns?\s*(?:\([^)]*\))?\s*:\s*(.+)", section, re.I)
    if match:
        # The list ends with its sentence ("... `bic`. One row per pair ..."): later backticked
        # words on the line are values or other fields, not columns.
        listed = re.split(r"(?<=[`\w)])\.\s", match.group(1), maxsplit=1)[0]
        ticked = _BACKTICKED.findall(listed)
        parts = ticked[0].split(",") if len(ticked) == 1 else ticked
        names = [p.strip() for p in parts if re.fullmatch(r"[\w.\- ]+", p.strip() or "#")]
        if len(names) >= 2:
            return names
    return []


_DYNAMIC_KEY = "__dynamic__"


def _mark_dynamic_keys(text: str) -> str:
    """``"<sector>": ...`` is a placeholder KEY: the real keys vary, so none is required."""
    return re.sub(r'"<[^"<>\n]*>"\s*:', f'"{_DYNAMIC_KEY}":', text)


def _key_paths(obj: object, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == _DYNAMIC_KEY:
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(path)
            paths.extend(_key_paths(value, path))
    return paths


def _json_keys(section: str) -> list[str]:
    block = re.search(r"```json\s*\n(.*?)```", section, re.S)
    if not block:
        return []
    text = _PLACEHOLDER.sub("null", _mark_dynamic_keys(block.group(1)))
    text = re.sub(r"//[^\n]*", "", text)  # comments
    text = re.sub(r",\s*\.\.\.\s*", "", text).replace("...", "")  # elisions
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # trailing commas
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return _key_paths(data)


def parse_spec(instruction: str, deliverables: list[str]) -> list[DeliverableSpec]:
    specs = []
    for name in deliverables:
        section = _section(instruction, name)
        if not section:
            continue
        suffix = pathlib.Path(name).suffix.lower()
        spec = DeliverableSpec(name=name)
        if suffix in _TABLE_SUFFIXES:
            spec.columns = _table_columns(section)
            spec.numeric_columns = _numeric_columns(section)
            spec.row_count = _row_count(section)
        elif suffix == ".json":
            spec.json_keys = _json_keys(section)
            spec.numeric_keys = _numeric_json_keys(section)
        if spec.columns or spec.json_keys or spec.numeric_columns or spec.row_count:
            specs.append(spec)
    logger.info(
        "Spec checks: %s",
        {s.name: len(s.columns) or len(s.json_keys) for s in specs} or "nothing parseable",
    )
    return specs


def _read_columns(path: pathlib.Path) -> list[str]:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return list(pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=0).columns)
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def _present_paths(obj: object, prefix: str = "") -> set[str]:
    """Key paths present in an output; list elements share their parent's path."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.add(path)
            found |= _present_paths(value, path)
    elif isinstance(obj, list):
        for value in obj[:50]:
            found |= _present_paths(value, prefix)
    return found


def _read_table(path: pathlib.Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    return pd.read_parquet(path)


def _table_value_problems(spec: DeliverableSpec, path: pathlib.Path) -> list[str]:
    import pandas as pd

    if path.stat().st_size > 200_000_000:
        return []
    df = _read_table(path)
    problems = []
    if spec.row_count is not None and len(df) != spec.row_count:
        problems.append(
            f"{spec.name}: has {len(df)} rows; the specification says {spec.row_count} rows"
        )
    for col in spec.numeric_columns:
        if col not in df.columns or pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].dropna()
        converted = pd.to_numeric(values, errors="coerce")
        bad = values[converted.isna()]
        if len(bad):
            problems.append(
                f"{spec.name}: column {col!r} must be numeric but has {len(bad)} non-numeric "
                f"value(s), e.g. {bad.iloc[0]!r}"
            )
    return problems


def _json_value_problems(spec: DeliverableSpec, path: pathlib.Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for key in spec.numeric_keys:
        node: object = data
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, str):
            problems.append(f"{spec.name}: key {key!r} must be a number, not the string {node!r}")
    return problems


def spec_problems(specs: list[DeliverableSpec], out_dir: pathlib.Path) -> list[str]:
    """Missing columns / JSON keys in the outputs (never raises)."""
    problems: list[str] = []
    for spec in specs:
        path = out_dir / spec.name
        if not path.is_file():
            continue  # missing files are the deliverable check's job
        try:
            if spec.numeric_columns or spec.row_count:
                problems.extend(_table_value_problems(spec, path))
            if spec.numeric_keys:
                problems.extend(_json_value_problems(spec, path))
            if spec.columns:
                have = set(_read_columns(path))
                missing = [c for c in spec.columns if c not in have]
                if missing:
                    problems.append(
                        f"{spec.name}: missing required column(s) {missing}; the specification "
                        f"lists {spec.columns}"
                    )
            elif spec.json_keys:
                have = _present_paths(json.loads(path.read_text(encoding="utf-8")))
                missing = [k for k in spec.json_keys if k not in have]
                if missing:
                    problems.append(
                        f"{spec.name}: missing required key(s) {missing[:12]} (the "
                        "specification's example shows them)"
                    )
        except Exception as exc:
            problems.append(f"{spec.name}: could not be read ({exc.__class__.__name__}: {exc})")
    return problems
