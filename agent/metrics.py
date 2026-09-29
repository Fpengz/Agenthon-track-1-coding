"""Competition-style metrics for a local batch run (host tooling for ``solve-units``).

Official Track 1 metric: mean pass@1 -- one execution per unit, over a FIXED denominator (every
selected unit; wrong, crashed, timed-out and missing results count as 0), no confidence interval.
A pass must clear all four admissibility gates. Locally they are approximated as:

    g0_integrity         the unit's manifest.json checksums match the files
    g1_schema            the output directory holds at least one deliverable
    g2_cutoff_resource   the agent finished within the card's timeout and no canary GUID from
                         any public card appears in its outputs (the scorer checks the registry)
    g3_domain_semantics  the unit's checks pass (checker reward == 1.0)

Failure labels use the official ``FailureLabel`` names; the g3 label comes from the organizer's
own classifier when ``qfbench2_track_coding`` is importable. Everything else in the report (test
pass fractions, request use, timing, category breakdowns) is diagnostic, not a leaderboard score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import tomllib
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

GATES = ("g0_integrity", "g1_schema", "g2_cutoff_resource", "g3_domain_semantics")
HOUSE_REQUEST_LIMIT = 25
# Development runs share ONE wall-clock allowance for the whole roster (README rule 3).
ROSTER_WALL_CLOCK_SEC = 12 * 3600
_HARNESS_FILES = frozenset({"reward.json", "pytest_report.json"})
_CANARY_SCAN_BYTES = 50_000_000

LABEL_TIMEOUT = "shared.resource.timeout"
LABEL_SCHEMA = "shared.schema.invalid_output"
LABEL_CANARY = "shared.contamination.canary_emitted"
LABEL_INTEGRITY = "shared.integrity.bad_manifest"
LABEL_WRONG_NUMERIC = "t1.wrong_numeric"


def card_metadata(unit_dir: pathlib.Path) -> dict[str, Any]:
    try:
        return tomllib.loads((unit_dir / "card.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def canary_registry(units_dir: pathlib.Path) -> set[str]:
    """Every canary GUID declared by a public card (g2 flags any of them, not only the unit's)."""
    found = set()
    for card in units_dir.glob("*/card.toml"):
        guid = card_metadata(card.parent).get("contamination", {}).get("canary_guid")
        if guid:
            found.add(str(guid).lower())
    return found


def manifest_ok(unit_dir: pathlib.Path) -> bool | None:
    try:
        manifest = json.loads((unit_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in manifest.get("files", []):
        path = unit_dir / entry.get("path", "")
        expected = entry.get("sha256")
        if not expected or not path.is_file():
            return False
        with path.open("rb") as fh:  # streamed: some unit inputs are tens of MB
            if hashlib.file_digest(fh, "sha256").hexdigest() != expected:
                return False
    return True


def canary_hits(out_dir: pathlib.Path, registry: set[str]) -> list[str]:
    hits = []
    for path in out_dir.rglob("*"):
        if not path.is_file() or path.name in _HARNESS_FILES:
            continue
        with path.open("rb") as fh:
            text = fh.read(_CANARY_SCAN_BYTES).decode("utf-8", errors="ignore").lower()
        hits.extend(f"{path.name}:{guid}" for guid in registry if guid in text)
    return hits


def pytest_counts(out_dir: pathlib.Path) -> tuple[int, int, list[str]]:
    """(passed, total, failed node ids) from the checker's pytest report."""
    try:
        report = json.loads((out_dir / "pytest_report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, 0, []
    tests = report.get("tests", [])
    failed = [t.get("nodeid", "?") for t in tests if t.get("outcome") in {"failed", "error"}]
    passed = sum(1 for t in tests if t.get("outcome") == "passed")
    return passed, len(tests), failed


def classify_failed_checks(failed_nodeids: list[str]) -> str:
    """Official g3 label for failed checks, via the organizer classifier when available."""
    try:
        from qfbench2_track_coding.scoring import _classify_trusted_failures
    except Exception:  # organizer package absent (e.g. inside the agent image)
        return LABEL_WRONG_NUMERIC
    cases = []
    for nodeid in failed_nodeids:
        case = ET.Element("testcase", name=nodeid.split("::")[-1])
        ET.SubElement(case, "failure")
        cases.append(case)
    label = _classify_trusted_failures(cases)
    return label.value if label is not None else LABEL_WRONG_NUMERIC


def assess_unit(
    unit_dir: pathlib.Path,
    out_dir: pathlib.Path,
    timed_out: bool,
    reward: float | None,
    registry: set[str],
) -> dict[str, Any]:
    """Gate outcomes, the first failing gate's label, and check diagnostics for one unit."""
    deliverables = (
        [p for p in out_dir.rglob("*") if p.is_file() and p.name not in _HARNESS_FILES]
        if out_dir.is_dir()
        else []
    )
    hits = canary_hits(out_dir, registry) if deliverables else []
    passed, total, failed = pytest_counts(out_dir)
    gates: dict[str, bool | None] = {
        "g0_integrity": manifest_ok(unit_dir),
        "g1_schema": bool(deliverables),
        "g2_cutoff_resource": not timed_out and not hits,
        "g3_domain_semantics": None if reward is None else reward == 1.0,
    }
    label = None
    if gates["g0_integrity"] is False:
        label = LABEL_INTEGRITY
    elif timed_out:
        label = LABEL_TIMEOUT
    elif not deliverables:
        label = LABEL_SCHEMA
    elif hits:
        label = LABEL_CANARY
    elif gates["g3_domain_semantics"] is False:
        label = classify_failed_checks(failed)
    meta = card_metadata(unit_dir).get("metadata", {})
    return {
        "gates": gates,
        "failure_label": label,
        "canary_hits": hits,
        "tests_passed": passed,
        "tests_total": total,
        "category": meta.get("category", "unknown"),
        "difficulty": meta.get("difficulty", "unknown"),
    }


def _rate(passed: int, total: int) -> dict[str, Any]:
    return {
        "passed": passed,
        "units": total,
        "pass_at_1": round(passed / total, 4) if total else 0.0,
    }


def build_report(results: list[dict[str, Any]], checked: bool) -> dict[str, Any]:
    """Aggregate per-unit results (``UnitResult`` dicts) into the run's metric report."""
    n = len(results)
    passed = sum(1 for r in results if r["status"] == "passed")
    report: dict[str, Any] = {
        "official": {
            "metric": "mean pass@1 (one execution per unit, fixed denominator)",
            "units": n,
            "passed": passed,
            "pass_at_1": round(passed / n, 4) if n and checked else None,
            "note": None if checked else "checker did not run: pass@1 unavailable",
        }
    }

    gate_counts = {}
    for gate in GATES:
        values = [r["gates"].get(gate) for r in results]
        gate_counts[gate] = {
            "cleared": sum(1 for v in values if v is True),
            "failed": sum(1 for v in values if v is False),
            "not_evaluated": sum(1 for v in values if v is None),
        }
    report["gates"] = gate_counts

    labels: dict[str, int] = defaultdict(int)
    for r in results:
        if r["status"] != "passed":
            labels[r["failure_label"] or r["status"]] += 1
    report["failure_labels"] = dict(sorted(labels.items(), key=lambda kv: -kv[1]))

    for key in ("category", "difficulty"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in results:
            groups[r[key]].append(r)
        report[f"by_{key}"] = {
            name: _rate(sum(1 for r in rs if r["status"] == "passed"), len(rs))
            for name, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        }

    checked_units = [r for r in results if r["tests_total"]]
    requests = [r["model_requests"] for r in results if r["model_requests"] is not None]
    seconds = [r["agent_seconds"] for r in results]
    report["diagnostics"] = {
        "note": "not leaderboard metrics",
        "mean_check_pass_fraction": round(
            sum(r["tests_passed"] / r["tests_total"] for r in checked_units) / len(checked_units), 4
        )
        if checked_units
        else None,
        "house_requests": {
            "total": sum(requests),
            "mean": round(sum(requests) / len(requests), 2) if requests else None,
            "max": max(requests, default=None),
            "units_without_house_request": [
                r["unit"] for r in results if not r["model_requests"]
            ],  # rule 9: a pass without run-time House use earns no credit
            "units_over_limit": [
                r["unit"] for r in results if (r["model_requests"] or 0) > HOUSE_REQUEST_LIMIT
            ],
        },
        "agent_seconds": {
            "total": round(sum(seconds)),
            "mean": round(sum(seconds) / n, 1) if n else None,
            "max": round(max(seconds, default=0)),
            "sequential_share_of_12h_allowance": round(sum(seconds) / ROSTER_WALL_CLOCK_SEC, 3),
        },
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    off = report["official"]
    lines = ["", "=" * 72, "COMPETITION METRICS (local estimate)", "=" * 72]
    if off["pass_at_1"] is None:
        lines.append(f"pass@1: n/a -- {off['note']}")
    else:
        lines.append(
            f"pass@1 (official metric): {off['pass_at_1']:.3f}  "
            f"= {off['passed']}/{off['units']} units, fixed denominator"
        )
    lines.append("\nAdmissibility gates (cleared / failed / not evaluated):")
    for gate, c in report["gates"].items():
        lines.append(f"  {gate:<21} {c['cleared']:>4} / {c['failed']:>4} / {c['not_evaluated']:>4}")
    if report["failure_labels"]:
        lines.append("\nFailure labels:")
        lines.extend(
            f"  {label:<36} {count:>4}" for label, count in report["failure_labels"].items()
        )
    for key in ("difficulty", "category"):
        lines.append(f"\npass@1 by {key}:")
        lines.extend(
            f"  {name:<24} {g['passed']:>3}/{g['units']:<3} {g['pass_at_1']:.3f}"
            for name, g in report[f"by_{key}"].items()
        )
    d = report["diagnostics"]
    req, secs = d["house_requests"], d["agent_seconds"]
    lines.append("\nDiagnostics (not leaderboard metrics):")
    if d["mean_check_pass_fraction"] is not None:
        lines.append(f"  mean fraction of checks passed   {d['mean_check_pass_fraction']:.3f}")
    lines.append(
        f"  House requests                   total {req['total']}, mean {req['mean']}, "
        f"max {req['max']} (limit {HOUSE_REQUEST_LIMIT}/unit)"
    )
    if req["units_without_house_request"]:
        lines.append(f"  units with no House request      {req['units_without_house_request']}")
    if req["units_over_limit"]:
        lines.append(f"  units over the request limit     {req['units_over_limit']}")
    lines.append(
        f"  agent time                       total {secs['total']}s, mean {secs['mean']}s, "
        f"max {secs['max']}s; sequential = {secs['sequential_share_of_12h_allowance']:.0%} "
        "of the 12 h roster allowance"
    )
    return "\n".join(lines)
