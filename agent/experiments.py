"""Experiment records for local evaluation runs (see experiments/README.md).

Every ``solve-units`` run gets a run id (``YYYYMMDD-HHMM-<name>``), a ``manifest.json`` written
at start (git commit and dirty flag, image IDs, model, settings, units, note), and -- once it
finishes -- one line in the tracked, append-only ``experiments/registry.jsonl`` holding the
manifest, headline metrics and every unit's status. Raw artifacts under ``experiments/runs/``
are gitignored and prunable; comparisons only need the registry.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pathlib
import re
import shutil
import socket
import subprocess
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
REGISTRY = EXPERIMENTS_DIR / "registry.jsonl"
# Settings that change agent behaviour; recorded so a run can be reproduced.
SETTING_VARS = (
    "HOUSE_REASONING",
    "AGENT_MAX_REVIEWS",
    "AGENT_TIME_BUDGET_SEC",
    "HOUSE_CONTEXT_TOKENS",
    "HOUSE_REQUEST_TIMEOUT",
    "AGENT_DOMAIN_NOTES",
    "AGENT_CANDIDATES",
)
_SLUG = re.compile(r"[^a-z0-9]+")


def now_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def make_run_id(name: str) -> str:
    slug = _SLUG.sub("-", name.lower()).strip("-") or "run"
    return f"{dt.datetime.now().strftime('%Y%m%d-%H%M')}-{slug}"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def git_state() -> dict[str, Any]:
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]) or None,
        "branch": _run(["git", "-C", str(REPO_ROOT), "branch", "--show-current"]) or None,
        # Uncommitted changes to tracked files mean the commit alone does not reproduce the run.
        "dirty": bool(status),
    }


def image_id(image: str | None) -> str | None:
    """Immutable image ID: tags get rebuilt, IDs do not."""
    if not image:
        return None
    return _run(["docker", "image", "inspect", "--format", "{{.Id}}", image]) or None


def served_model(endpoint: str | None) -> str | None:
    """The model root reported by an OpenAI-compatible server (e.g. the vLLM checkpoint)."""
    if not endpoint:
        return None
    url = endpoint.rstrip("/") + ("" if endpoint.rstrip("/").endswith("/v1") else "/v1") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 (local endpoint)
            data = json.load(resp).get("data", [])
        return data[0].get("root") or data[0].get("id") if data else None
    except Exception:
        return None


def write_manifest(
    run_dir: pathlib.Path,
    *,
    name: str,
    note: str,
    baseline: str | None,
    units: list[str],
    jobs: int,
    agent_image: str | None,
    checker_image: str | None,
    timeout_override: float | None,
) -> dict[str, Any]:
    endpoint = os.environ.get("MODEL_ENDPOINT")
    manifest = {
        "run_id": run_dir.name,
        "name": name,
        "note": note,
        "baseline": baseline,
        "started_at": now_utc(),
        "host": socket.gethostname(),
        "git": git_state(),
        "agent_image": agent_image,
        "agent_image_id": image_id(agent_image),
        "checker_image": checker_image,
        "checker_image_id": image_id(checker_image),
        "model": {
            "endpoint": endpoint,
            "name": os.environ.get("MODEL_NAME"),
            "served": served_model(endpoint),
        },
        "settings": {var: os.environ.get(var) for var in SETTING_VARS},
        "jobs": jobs,
        "timeout_override": timeout_override,
        "units": units,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest["git"]["dirty"]:
        logger.warning("Working tree has uncommitted changes: this run is not reproducible")
    return manifest


def registry_entry(manifest: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    diagnostics = metrics.get("diagnostics", {})
    results = summary.get("results", [])
    planned = summary.get("units_planned") or summary.get("units") or len(results)
    passed = sum(r.get("status") == "passed" for r in results)
    checked = any(r.get("status") in {"passed", "failed"} for r in results)
    return {
        **manifest,
        "finished_at": now_utc(),
        "units_planned": planned,
        "units_finished": len(results),
        # Fixed denominator: unreached units of an interrupted run count as failures.
        "pass_at_1": round(passed / planned, 4) if planned and checked else None,
        "passed": passed,
        "gates": metrics.get("gates"),
        "failure_labels": metrics.get("failure_labels"),
        "mean_check_pass_fraction": diagnostics.get("mean_check_pass_fraction"),
        "house_requests_mean": diagnostics.get("house_requests", {}).get("mean"),
        "agent_seconds": diagnostics.get("agent_seconds"),
        # Compact per-unit outcome: enough to compare runs after raw artifacts are pruned.
        "unit_status": {r["unit"]: r["status"] for r in summary.get("results", [])},
    }


def append_registry(entry: dict[str, Any], registry: pathlib.Path = REGISTRY) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def load_registry(registry: pathlib.Path = REGISTRY) -> list[dict[str, Any]]:
    if not registry.is_file():
        return []
    return [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line]


def find_run(run_id: str, registry: pathlib.Path = REGISTRY) -> dict[str, Any]:
    matches = [e for e in load_registry(registry) if e["run_id"] == run_id or e["name"] == run_id]
    if not matches:
        raise KeyError(f"run {run_id!r} is not in {registry}")
    return matches[-1]


def format_list(entries: list[dict[str, Any]]) -> str:
    lines = [f"{'run_id':<34} {'pass@1':>7} {'passed':>9} {'reqs':>5} {'commit':<9} note"]
    for e in entries:
        pass_at_1 = "-" if e.get("pass_at_1") is None else f"{e['pass_at_1']:.3f}"
        passed = f"{e.get('passed') or 0}/{e.get('units_planned') or 0}"
        commit = (e.get("git", {}).get("commit") or "unrec")[:7]
        commit += "*" if e.get("git", {}).get("dirty") else ""
        reqs = e.get("house_requests_mean")
        lines.append(
            f"{e['run_id']:<34} {pass_at_1:>7} {passed:>9} {reqs if reqs is not None else '-':>5} "
            f"{commit:<9} {e.get('note', '')[:70]}"
        )
    return "\n".join(lines)


def compare(entries: list[dict[str, Any]]) -> str:
    """pass@1 on the units every run shares, plus per-unit flips against the first run."""
    shared = sorted(set.intersection(*(set(e["unit_status"]) for e in entries)))
    lines = [f"{len(shared)} units shared by all {len(entries)} runs"]
    for e in entries:
        passed = sum(e["unit_status"][u] == "passed" for u in shared)
        lines.append(
            f"  {e['run_id']:<34} {passed:>3}/{len(shared)} = {passed / max(1, len(shared)):.3f}"
        )
    base = entries[0]
    for other in entries[1:]:
        gained = [
            u for u in shared if other["unit_status"][u] == "passed" != base["unit_status"][u]
        ]
        lost = [u for u in shared if base["unit_status"][u] == "passed" != other["unit_status"][u]]
        lines.append(f"\n{other['run_id']} vs {base['run_id']}: +{len(gained)} / -{len(lost)}")
        lines.extend(f"  + {u}" for u in gained)
        lines.extend(f"  - {u}" for u in lost)
    ever = {u for e in entries for u in shared if e["unit_status"][u] == "passed"}
    always = [u for u in shared if all(e["unit_status"][u] == "passed" for e in entries)]
    lines.append(
        f"\npassed in any run: {len(ever)}; in every run: {len(always)} "
        "(the gap is run-to-run noise: compare with >= 2 runs per variant)"
    )
    return "\n".join(lines)


def prune_run(run_dir: pathlib.Path) -> int:
    """Delete a run's raw artifacts (its registry entry stays). Returns bytes freed."""
    if not run_dir.is_dir():
        return 0
    freed = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    shutil.rmtree(run_dir)
    return freed
