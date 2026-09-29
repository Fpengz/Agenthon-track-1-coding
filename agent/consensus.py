"""Compare the deliverables of independently generated candidate solutions.

Best-of-N consensus (``AGENT_CANDIDATES``): identical code flips units between runs, so one clean
run says little about correctness. Two independent solutions that agree on their outputs are far
more likely right; a disagreement is evidence one of them is wrong and triggers a tie-breaker.
Tables are compared cell by cell (numbers within tolerance, other values exactly) and JSON leaf
by leaf; charts and other files only need to exist in both.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

from agent.review import output_files

logger = logging.getLogger(__name__)

RTOL = 5e-3  # relative tolerance for numbers (seeded Monte Carlo still matches; methods differ)
ATOL = 1e-8
AGREE_THRESHOLD = 0.95  # share of compared values that must match for two candidates to agree
_MAX_TABLE_BYTES = 100_000_000
_MAX_DIFFS = 15
_TABLES = {".csv", ".tsv", ".parquet", ".pqt"}


@dataclass
class Agreement:
    score: float  # mean per-file share of matching values, in [0, 1]
    diffs: list[str] = field(default_factory=list)  # human-readable disagreements

    @property
    def agree(self) -> bool:
        return self.score >= AGREE_THRESHOLD


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, value in enumerate(obj):
            out.update(_flatten(value, f"{prefix}[{i}]"))
        return out
    return {prefix: obj}


def _compare_json(name: str, a: pathlib.Path, b: pathlib.Path, diffs: list[str]) -> float:
    fa = _flatten(json.loads(a.read_text(encoding="utf-8")))
    fb = _flatten(json.loads(b.read_text(encoding="utf-8")))
    keys = sorted(set(fa) | set(fb))
    if not keys:
        return 1.0
    matched = 0
    for key in keys:
        if key in fa and key in fb and _close(fa[key], fb[key]):
            matched += 1
        elif len(diffs) < _MAX_DIFFS:
            diffs.append(
                f"{name}:{key}  A={fa.get(key, '<missing>')!r}  B={fb.get(key, '<missing>')!r}"
            )
    return matched / len(keys)


def _read_table(path: pathlib.Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    return pd.read_parquet(path)


def _compare_table(name: str, a: pathlib.Path, b: pathlib.Path, diffs: list[str]) -> float:
    import numpy as np
    import pandas as pd

    if max(a.stat().st_size, b.stat().st_size) > _MAX_TABLE_BYTES:
        return 1.0  # too large to compare cheaply; treated as agreeing
    ta, tb = _read_table(a), _read_table(b)
    columns = [c for c in ta.columns if c in tb.columns]
    only = sorted(set(ta.columns) ^ set(tb.columns), key=str)
    if only and len(diffs) < _MAX_DIFFS:
        diffs.append(f"{name}: columns only in one candidate: {only[:8]}")
    if len(ta) != len(tb) and len(diffs) < _MAX_DIFFS:
        diffs.append(f"{name}: row count A={len(ta)} B={len(tb)}")
    rows = min(len(ta), len(tb))
    total = max(len(ta), len(tb)) * len(set(ta.columns) | set(tb.columns))
    if total == 0:
        return 1.0
    matched = 0
    for col in columns:
        ca, cb = ta[col].iloc[:rows], tb[col].iloc[:rows]
        if pd.api.types.is_numeric_dtype(ca) and pd.api.types.is_numeric_dtype(cb):
            va, vb = ca.to_numpy(dtype=float), cb.to_numpy(dtype=float)
            same = np.isclose(va, vb, rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            same = ca.astype(str).to_numpy() == cb.astype(str).to_numpy()
        matched += int(same.sum())
        if not same.all() and len(diffs) < _MAX_DIFFS:
            i = int(np.argmin(same))
            diffs.append(
                f"{name}:{col} differs in {int((~same).sum())}/{rows} rows, "
                f"e.g. row {i}: A={ca.iloc[i]!r} B={cb.iloc[i]!r}"
            )
    return matched / total


def compare_outputs(dir_a: pathlib.Path, dir_b: pathlib.Path) -> Agreement:
    """How closely two candidates' deliverables agree (never raises)."""
    files_a = {p.relative_to(dir_a).as_posix(): p for p in output_files(dir_a)}
    files_b = {p.relative_to(dir_b).as_posix(): p for p in output_files(dir_b)}
    diffs: list[str] = []
    scores: list[float] = []
    for name in sorted(set(files_a) | set(files_b)):
        a, b = files_a.get(name), files_b.get(name)
        if a is None or b is None:
            scores.append(0.0)
            diffs.append(f"{name}: only produced by candidate {'A' if b is None else 'B'}")
            continue
        suffix = a.suffix.lower()
        try:
            if suffix == ".json":
                scores.append(_compare_json(name, a, b, diffs))
            elif suffix in _TABLES:
                scores.append(_compare_table(name, a, b, diffs))
            # charts, reports and other files: existence in both is all that is compared
        except Exception as exc:  # unreadable output: count as disagreement
            scores.append(0.0)
            diffs.append(f"{name}: could not be compared ({exc.__class__.__name__})")
    score = sum(scores) / len(scores) if scores else 1.0
    return Agreement(score=score, diffs=diffs)
