"""Self-verification after a clean run.

A script that exits 0 and writes every expected file can still be wrong, and the unit's
checker is sealed from the agent. Before accepting a result the loop asks the model to review
it against the specification, with previews of the outputs and cheap automatic findings
computed here (NaN/inf values, empty tables, nulls in JSON). A review either approves the
outputs or returns a corrected script, which then goes through the normal run/repair path.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import re
import shutil
from dataclasses import dataclass
from typing import Any

from agent.executor import ExecutionResult, strip_reasoning

logger = logging.getLogger(__name__)

HARNESS_FILES = frozenset({"reward.json", "pytest_report.json"})
_VERDICT = re.compile(r"VERDICT:\s*\**\s*(PASS|FAIL)", re.IGNORECASE)
_CHECK_LINE = re.compile(r"^[ \t]*(PASS|FAIL)[ \t]*:[ \t]*(.*)$", re.IGNORECASE | re.M)
_ASSERTION = re.compile(r"^AssertionError(?:\s*:.*)?$", re.M)
_MAX_FINDINGS = 25
# Diagnostics must stay cheap: very large tables are sampled, and huge files skipped.
_TABLE_MAX_BYTES = 200_000_000
_CSV_MAX_ROWS = 200_000


def output_files(out_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name not in HARNESS_FILES)


def _json_findings(obj: Any, path: str, found: list[str]) -> None:
    if len(found) >= _MAX_FINDINGS:
        return
    if obj is None:
        found.append(f"{path} is null")
    elif isinstance(obj, float) and not math.isfinite(obj):
        found.append(f"{path} is {obj}")
    elif isinstance(obj, dict):
        if not obj:
            found.append(f"{path} is an empty object")
        for key, value in obj.items():
            _json_findings(value, f"{path}.{key}", found)
    elif isinstance(obj, list):
        if not obj:
            found.append(f"{path} is an empty list")
        for i, value in enumerate(obj[:200]):
            _json_findings(value, f"{path}[{i}]", found)


def _table_findings(path: pathlib.Path) -> list[str]:
    import numpy as np
    import pandas as pd

    if path.stat().st_size > _TABLE_MAX_BYTES:
        return []
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, nrows=_CSV_MAX_ROWS)
    else:
        df = pd.read_parquet(path)
    if df.empty:
        return [f"{path.name}: table has no rows (columns: {list(df.columns)})"]
    found = []
    for col in df.columns:
        nans = int(df[col].isna().sum())
        if nans:
            found.append(f"{path.name}: column {col!r} has {nans}/{len(df)} missing values")
        if pd.api.types.is_numeric_dtype(df[col]):
            infs = int(np.isinf(df[col].to_numpy(dtype=float, na_value=np.nan)).sum())
            if infs:
                found.append(f"{path.name}: column {col!r} has {infs} infinite values")
    return found


def output_diagnostics(out_dir: pathlib.Path) -> list[str]:
    """Mechanical red flags in the deliverables; never raises."""
    findings: list[str] = []
    for path in output_files(out_dir):
        suffix = path.suffix.lower()
        try:
            if path.stat().st_size == 0:
                findings.append(f"{path.name}: file is empty")
            elif suffix in {".csv", ".tsv", ".parquet", ".pqt"}:
                findings.extend(_table_findings(path))
            elif suffix == ".json":
                text = path.read_text(encoding="utf-8")
                if re.search(r"\b(NaN|-?Infinity)\b", text):
                    findings.append(f"{path.name}: contains NaN/Infinity (invalid strict JSON)")
                json_found: list[str] = []
                _json_findings(json.loads(text), path.name, json_found)
                findings.extend(json_found)
        except Exception as exc:  # unreadable output is itself a finding
            findings.append(
                f"{path.name}: could not be read back ({exc.__class__.__name__}: {exc})"
            )
    return findings[:_MAX_FINDINGS]


def parse_verdict(response: str) -> str | None:
    """``"PASS"``/``"FAIL"`` from the review response, or None if absent."""
    matches = _VERDICT.findall(strip_reasoning(response))
    return matches[-1].upper() if matches else None


@dataclass(frozen=True)
class VerifierResult:
    passes: int
    failures: list[str]
    incomplete: str = ""


def verifier_result(run: ExecutionResult) -> VerifierResult:
    """Separate executed check failures from a verifier that could not finish its checks.

    Assertions are executed evidence even when the model omitted the requested FAIL print.
    Other crashes need a repaired verifier; they do not establish a defect in the solution.
    """
    checks = _CHECK_LINE.findall(run.stdout)
    passes = sum(kind.upper() == "PASS" for kind, _ in checks)
    failures = [f"FAIL: {text}" for kind, text in checks if kind.upper() == "FAIL"]
    if not run.success and not run.timed_out and _ASSERTION.search(run.stderr):
        failures.append("FAIL: verifier assertion failed\n" + run.feedback)
    if failures:
        return VerifierResult(passes, failures)
    if not run.success:
        reason = "Verifier did not complete its checks: " + run.failure_kind
        return VerifierResult(passes, [], reason + "\n" + run.feedback)
    if not passes:
        return VerifierResult(0, [], "Verifier ran but reported no PASS: or FAIL: checks.")
    return VerifierResult(passes, [])


def snapshot_outputs(out_dir: pathlib.Path, dest: pathlib.Path) -> None:
    """Keep a copy of a clean run's deliverables in case a review 'fix' makes things worse."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for path in output_files(out_dir):
        target = dest / path.relative_to(out_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def restore_outputs(snapshot: pathlib.Path, out_dir: pathlib.Path) -> None:
    for path in output_files(out_dir):
        path.unlink()
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        target = out_dir / path.relative_to(snapshot)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
