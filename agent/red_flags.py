"""Red flags in a clean run's deliverables that the outputs themselves expose.

Wrong numbers are the main failure (45-53 of 86 units per run end ``t1.wrong_numeric``), and
the model's prose review accepts most of them. Two kinds of wrong output are visible without
knowing the answer, and both are near-certain failures in past runs:

- **failed self-checks**: the task asked for validation flags (``mc_validates``,
  ``parity_holds``, ``all_checks_passed`` ...) and the script reported one as ``false``.
  188 checker-failed vs 3 checker-passed unit runs (12 units) in the Sept 30 - Oct 1 runs.
- **impossible values**: a negative price, probability, volatility or variance, or a
  probability above 1. 103 checker-failed vs 0 checker-passed unit runs (14 units).

Findings drive bounded repair turns (``AGENT_RED_FLAGS``); they are never answer keys and
never name a unit. Every rule must stay (near-)silent on checker-passed outputs.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Any

from agent.review import HARNESS_FILES

_MAX_BYTES = 20_000_000
_MAX_FINDINGS = 12
# Validation-flag names. Data properties that may honestly be false (e.g. "feller_satisfied",
# "is_stationary", "reject_null") are deliberately not matched.
_CHECK_KEY = re.compile(
    r"valid|pass|holds|(^|_)ok($|_)|check|consisten|within|agree|monoton|converge|match|"
    r"reprice|bounded|positive",
    re.I,
)
_LABEL_KEYS = ("name", "check", "test", "description", "label")
# Quantities that can never be negative, as whole name tokens (call_price, vol, variance...).
_NONNEGATIVE = re.compile(
    r"(^|_)(price|prob|probability|vol|volatility|variance|premium|std|stdev|sigma)(_|$)", re.I
)
_PROBABILITY = re.compile(r"(^|_)(prob|probability)(_|$)", re.I)
# Names that look like those but hold signed quantities (differences, Greeks, returns...).
_SIGNED = re.compile(
    r"diff|change|pnl|err|resid|delta|gamma|vega|theta|rho|return|spread|log|skew|corr|beta|"
    r"alpha|basis|bias|vs_|excess|drift|slope|sens|ratio",
    re.I,
)

SELF_CHECK_HINT = (
    "Your own output reports a validation check as FAILED. A failed self-check almost always "
    "means a bug in the computation it validates (formula, units, discounting, sign, "
    "annualisation, input alignment, too few simulation paths or a mismatched random draw). "
    "Find and fix that cause so the check passes honestly. Never force the flag, and never "
    "loosen or redefine a check or tolerance the specification gives."
)
IMPOSSIBLE_HINT = (
    "A price, probability, volatility or variance can never be negative, and a probability "
    "cannot exceed 1: the computation that produced it is wrong (formula, sign convention, "
    "discounting or numerical method). Fix the computation; do not clip the value."
)

RED_FLAGS_GUIDANCE = (
    "### SELF-CHECKS:\n"
    "- When the specification asks for validation checks (parity, Monte Carlo agreement, "
    "monotonicity, repricing, positivity...), compute them before writing outputs. A failing "
    "check means a bug: fix the computation, never the flag or the tolerance.\n"
    "- Prices, probabilities, volatilities and variances must be non-negative (probabilities "
    "at most 1); a value outside that range is a computation error, not a result."
)


def _name(key: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(key)).strip("_")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _impossible(key: str, value: float) -> str | None:
    name = _name(key)
    if not _NONNEGATIVE.search(name) or _SIGNED.search(name):
        return None
    if value < -1e-12:
        return f"negative value {value!r}"
    if _PROBABILITY.search(name) and value > 1 + 1e-9:
        return f"probability above 1 ({value!r})"
    return None


def _json_flags(value: Any, path: str, checks: list[str], impossible: list[str]) -> None:
    if len(checks) + len(impossible) >= _MAX_FINDINGS:
        return
    if isinstance(value, dict):
        label = next((str(value[k]) for k in _LABEL_KEYS if isinstance(value.get(k), str)), "")
        for key, child in list(value.items())[:500]:
            where = f"{path}.{key}" + (f" ({label})" if label else "")
            if isinstance(child, bool):
                if child is False and _CHECK_KEY.search(_name(key)):
                    checks.append(f"{where} = false")
            elif isinstance(child, str):
                if child.strip().upper() in {"FAIL", "FAILED"} and _CHECK_KEY.search(_name(key)):
                    checks.append(f"{where} = {child!r}")
            elif (number := _number(child)) is not None:
                problem = _impossible(str(key), number)
                if problem:
                    impossible.append(f"{where}: {problem}")
            else:
                _json_flags(child, f"{path}.{key}", checks, impossible)
    elif isinstance(value, list):
        for i, child in enumerate(value[:500]):
            _json_flags(child, f"{path}[{i}]", checks, impossible)


def _table_flags(path: pathlib.Path, name: str, checks: list[str], impossible: list[str]) -> None:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pqt"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=200_000)
    for column in frame.columns:
        series = frame[column]
        col = _name(column)
        if series.dtype == bool or set(series.dropna().astype(str).str.lower()) <= {
            "true",
            "false",
        }:
            failed = series.astype(str).str.lower().eq("false")
            if failed.any() and _CHECK_KEY.search(col):
                checks.append(
                    f"{name} column {column!r}: false in {int(failed.sum())} of {len(series)} "
                    f"rows (first at row {int(failed.to_numpy().argmax()) + 1})"
                )
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue
        values = series.dropna()
        if values.empty:
            continue
        lowest, highest = float(values.min()), float(values.max())
        problem = _impossible(str(column), lowest) or _impossible(str(column), highest)
        if problem:
            bad = (values < -1e-12) | ((values > 1 + 1e-9) if _PROBABILITY.search(col) else False)
            impossible.append(
                f"{name} column {column!r}: {problem} in {int(bad.sum())} of {len(values)} rows"
            )


def red_flags(out_dir: pathlib.Path) -> list[str]:
    """Findings for a clean run's deliverables (empty when nothing is visibly wrong)."""
    checks: list[str] = []
    impossible: list[str] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name in HARNESS_FILES:
            continue
        name = path.relative_to(out_dir).as_posix()
        suffix = path.suffix.lower()
        try:
            if path.stat().st_size > _MAX_BYTES:
                continue
            if suffix == ".json":
                _json_flags(json.loads(path.read_text(encoding="utf-8")), name, checks, impossible)
            elif suffix in {".csv", ".tsv", ".parquet", ".pqt"}:
                _table_flags(path, name, checks, impossible)
        except Exception:  # unreadable outputs are reported by the other output checks
            continue
    findings = []
    if checks:
        findings.append(SELF_CHECK_HINT + "\n" + "\n".join(f"- {c}" for c in checks[:8]))
    if impossible:
        findings.append(IMPOSSIBLE_HINT + "\n" + "\n".join(f"- {c}" for c in impossible[:8]))
    return findings
