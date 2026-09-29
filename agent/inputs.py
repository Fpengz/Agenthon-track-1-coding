"""Compact previews of task input files for the prompt.

Without them the model sees only file names and guesses column names and JSON keys, then
repeats the same KeyError on every repair. Each preview is bounded, and the total is capped
so the prompt stays well inside the House model's context.
"""

from __future__ import annotations

import json
import logging
import pathlib
import warnings
import zipfile
from typing import Any

logger = logging.getLogger(__name__)

PER_FILE_CHARS = 1200
TOTAL_CHARS = 9000
_JSON_LOAD_LIMIT = 5_000_000
_SKIP_NAMES = frozenset({"manifest.json"})  # organizer checksums, not task data


def _clip(text: str, limit: int = PER_FILE_CHARS) -> str:
    return text if len(text) <= limit else f"{text[:limit]}\n… [preview truncated]"


def _json_skeleton(obj: Any, depth: int = 0) -> Any:
    """Structure of a JSON value: keys and short scalars kept, long lists summarised."""
    if depth > 4:
        return "…"
    if isinstance(obj, dict):
        items = list(obj.items())
        out = {k: _json_skeleton(v, depth + 1) for k, v in items[:25]}
        if len(items) > 25:
            out["…"] = f"{len(items) - 25} more keys"
        return out
    if isinstance(obj, list):
        if not obj:
            return []
        head = [_json_skeleton(v, depth + 1) for v in obj[:2]]
        return head + ([f"… {len(obj)} items total"] if len(obj) > 2 else [])
    if isinstance(obj, str) and len(obj) > 80:
        return obj[:80] + "…"
    return obj


def _head_lines(path: pathlib.Path, n: int) -> str:
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in range(n):
            line = fh.readline()
            if not line:
                break
            lines.append(line.rstrip("\n")[:300])
    return "\n".join(lines)


def _count_lines(path: pathlib.Path) -> int | None:
    if path.stat().st_size > 50_000_000:
        return None
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


_FACT_ROWS = 200_000
_FACT_MAX_BYTES = 200_000_000


def table_facts(df, include_dtypes: bool, sampled: bool) -> str:
    """Bounded facts about a table: dtypes, nulls, date ranges, key uniqueness, duplicates.

    Placed before the sample rows so they survive clipping; this is what a read-only data
    inspector would answer, without spending House requests on round-trips.
    """
    import pandas as pd

    facts = []
    if include_dtypes:
        facts.append("dtypes: " + ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items()))
    nulls = {c: int(n) for c, n in df.isna().sum().items() if n}
    if nulls:
        shown = list(nulls.items())[:8]
        facts.append("nulls: " + ", ".join(f"{c}={n}" for c, n in shown))
    ranges = []
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_datetime64_any_dtype(series):
            textual = pd.api.types.is_string_dtype(series) or series.dtype == object
            if not textual or not any(w in str(col).lower() for w in ("date", "time")):
                continue
            series = pd.to_datetime(series, errors="coerce")
            if series.notna().mean() < 0.9:
                continue
        if series.notna().any():
            ranges.append(f"{col}={series.min().date()}..{series.max().date()}")
    if ranges:
        facts.append("date ranges: " + ", ".join(ranges[:4]))
    if len(df.columns):
        first = df.columns[0]
        distinct = int(df[first].nunique(dropna=True))
        facts.append(
            f"first column {first!r}: {distinct} distinct"
            + (", unique" if distinct == len(df) else "")
        )
    dupes = int(df.duplicated().sum())
    if dupes:
        facts.append(f"{dupes} fully duplicated rows")
    note = f" (first {_FACT_ROWS} rows)" if sampled else ""
    return f"facts{note}: " + "; ".join(facts) if facts else ""


def _facts_or_empty(read) -> str:
    try:
        with warnings.catch_warnings():  # dtype/date-inference chatter must not flood the log
            warnings.simplefilter("ignore")
            df, include_dtypes, sampled = read()
            return table_facts(df, include_dtypes, sampled)
    except Exception as exc:  # facts are a bonus; never block the preview
        logger.debug("Table facts failed: %r", exc)
        return ""


def preview_file(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        import pandas as pd

        rows = _count_lines(path)
        count = f"{rows - 1} data rows" if rows else "row count unknown"
        facts = ""
        if path.stat().st_size <= _FACT_MAX_BYTES:
            sep = "\t" if suffix == ".tsv" else ","
            facts = _facts_or_empty(
                lambda: (
                    pd.read_csv(path, sep=sep, nrows=_FACT_ROWS),
                    True,
                    bool(rows and rows - 1 > _FACT_ROWS),
                )
            )
        facts_str = f"\n{facts}" if facts else ""
        return f"{count}{facts_str}\nfirst lines:\n{_head_lines(path, 6)}"
    if suffix in {".parquet", ".pqt"}:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(path)
        schema = ", ".join(f"{f.name}: {f.type}" for f in meta.schema_arrow)
        head = meta.read_row_group(0).slice(0, 3).to_pandas() if meta.num_row_groups else None
        facts = ""
        if meta.num_row_groups and path.stat().st_size <= _FACT_MAX_BYTES:
            total = meta.metadata.num_rows
            facts = _facts_or_empty(
                lambda: (
                    meta.read().slice(0, _FACT_ROWS).to_pandas(),
                    False,  # the schema line already shows the types
                    total > _FACT_ROWS,
                )
            )
        facts_str = f"\n{facts}" if facts else ""
        shown = "" if head is None else f"\nfirst rows:\n{head.to_string(max_colwidth=40)}"
        return f"{meta.metadata.num_rows} rows; columns: {schema}{facts_str}{shown}"
    if suffix == ".json":
        if path.stat().st_size > _JSON_LOAD_LIMIT:
            return f"large JSON; starts with:\n{_head_lines(path, 5)}"
        data = json.loads(path.read_text(encoding="utf-8"))
        return "structure:\n" + json.dumps(_json_skeleton(data), indent=1, default=str)
    if suffix == ".jsonl":
        return f"{_count_lines(path)} lines; first lines:\n{_head_lines(path, 3)}"
    if suffix == ".xlsx":
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None, nrows=3)
        return "\n".join(f"sheet {name!r}:\n{df.to_string()}" for name, df in sheets.items())
    if suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        return f"zip with {len(names)} entries, e.g.: {', '.join(names[:15])}"
    if suffix in {".txt", ".md", ".xml", ".py", ".toml", ".yaml", ".yml"}:
        return f"starts with:\n{_head_lines(path, 20)}"
    return f"{path.stat().st_size} bytes (binary; no preview)"


def build_input_previews(
    task_dir: pathlib.Path, files: list[pathlib.Path], label: str = "TASK_DIR"
) -> str:
    """Bounded previews for the prompt, one section per file (inputs, or outputs for review)."""
    files = [p for p in files if p.name not in _SKIP_NAMES]
    # Share the budget: with many inputs each preview gets shorter instead of later files
    # losing their preview entirely.
    per_file = max(300, min(PER_FILE_CHARS, TOTAL_CHARS // max(1, len(files))))
    sections: list[str] = []
    used = 0
    for path in files:
        rel = path.relative_to(task_dir).as_posix()
        try:
            body = _clip(preview_file(path), per_file)
        except Exception as exc:  # a preview must never stop the solve
            logger.debug("Preview of %s failed: %r", rel, exc)
            body = f"(preview unavailable: {exc.__class__.__name__})"
        section = f"#### {label}/{rel}\n{body}"
        if used + len(section) > TOTAL_CHARS:
            sections.append(
                f"… previews of the remaining files omitted (budget {TOTAL_CHARS} chars)"
            )
            break
        sections.append(section)
        used += len(section)
    return "\n\n".join(sections)
