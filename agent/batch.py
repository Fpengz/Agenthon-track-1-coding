"""Run the agent over a folder of units, then (optionally) check each output.

Each unit is staged the way evaluation mounts it -- task spec and ``environment/`` only, with
the answer-bearing ``checks/`` and ``reference/`` directories removed -- and solved in its own
``solve`` subprocess with a fresh output directory and the unit card's agent timeout. The
checker step reproduces README step 6: the unit's own ``checks/test.sh`` in the sandbox image,
with the unit at ``/input`` and the output at both ``/app/output`` and ``/output``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import json
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.loop import ANSWER_DIRS, container_path_map
from agent.metrics import assess_unit, build_report, canary_registry, format_report

logger = logging.getLogger(__name__)

DEFAULT_AGENT_TIMEOUT_SEC = 1800.0
CHECKER_TIMEOUT_SEC = 900
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class UnitResult:
    unit: str
    status: str  # passed | failed | unchecked | agent_failed | timeout | error
    agent_succeeded: bool = False
    model_requests: int | None = None
    reward: float | None = None
    agent_seconds: float = 0.0
    timeout_sec: float = 0.0
    out_dir: str = ""
    detail: str = ""
    # Competition-style assessment (agent/metrics.py)
    gates: dict[str, bool | None] = dataclasses.field(default_factory=dict)
    failure_label: str | None = None
    canary_hits: list[str] = dataclasses.field(default_factory=list)
    tests_passed: int = 0
    tests_total: int = 0
    category: str = "unknown"
    difficulty: str = "unknown"


def discover_units(units_dir: pathlib.Path, patterns: list[str]) -> list[pathlib.Path]:
    """Unit directories (those with instruction.md), filtered by name globs if given."""
    units = sorted(p.parent for p in units_dir.glob("*/instruction.md"))
    if patterns:
        units = [u for u in units if any(fnmatch.fnmatch(u.name, pat) for pat in patterns)]
    return units


def agent_timeout(unit_dir: pathlib.Path) -> float:
    """The card's ``[agent].timeout_sec`` (authoritative per unit)."""
    card = unit_dir / "card.toml"
    try:
        data = tomllib.loads(card.read_text(encoding="utf-8"))
        return float(data["agent"]["timeout_sec"])
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError):
        logger.warning("No [agent].timeout_sec in %s; using %.0fs", card, DEFAULT_AGENT_TIMEOUT_SEC)
        return DEFAULT_AGENT_TIMEOUT_SEC


def stage_unit(unit_dir: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """Copy the unit without its answer directories, mirroring the evaluation mount."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        unit_dir,
        dest,
        ignore=lambda src, names: (
            ([n for n in names if n in ANSWER_DIRS] if pathlib.Path(src) == unit_dir else [])
            + [n for n in names if n == "__pycache__"]
        ),
    )
    return dest


def checker_available(image: str) -> bool:
    if shutil.which("docker") is None:
        logger.warning("docker not found; outputs will not be checked")
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        logger.warning(
            "Checker image %s not found; outputs will not be checked. Build it with: "
            "docker build -t %s -f docker/sandbox.Dockerfile .",
            image,
            image,
        )
        return False
    return True


def checker_input_mounts(unit_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Where the official scorer presents unit inputs to the checks (container path -> source).

    Mirrors ``qfbench2_track_coding.scoring._input_presentations``: checks read inputs at the
    unit image's COPY targets (e.g. ``/app/data/x.parquet``) and reference data at
    ``/tests/reference_data``. The plain README step 6 command mounts neither, so checks that
    read their inputs fail there regardless of the agent's output.
    """
    mounts: dict[str, pathlib.Path] = {}
    reference = unit_dir / "checks" / "reference_data"
    if reference.is_dir():
        mounts["/tests/reference_data"] = reference
    if (unit_dir / "environment" / "Dockerfile").is_file():
        for target, rel in container_path_map(unit_dir):
            if target.startswith("/app/") and not target.startswith("/app/output"):
                mounts[target] = unit_dir / rel
    elif (unit_dir / "environment" / "data").is_dir():
        mounts["/app/data"] = unit_dir / "environment" / "data"
    return mounts


def run_checker(
    unit_dir: pathlib.Path, out_dir: pathlib.Path, image: str, log_path: pathlib.Path
) -> tuple[float | None, str]:
    """README step 6 checker. Returns (reward, detail).

    Runs as the invoking user so the reward files it writes stay deletable; ``test.sh`` also
    writes ``/logs/verifier/reward.txt``, so a user-owned directory is mounted at ``/logs``.
    """
    logs_dir = log_path.parent / "checker_logs"
    logs_dir.mkdir(exist_ok=True)
    input_mounts = [
        arg
        for target, src in checker_input_mounts(unit_dir).items()
        for arg in ("-v", f"{src}:{target}:ro")
    ]
    cmd = [
        "docker", "run", "--rm", "--network=none",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "OUTPUT_DIR=/app/output", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
        "-v", f"{unit_dir}:/input:ro",
        "-v", f"{out_dir}:/app/output", "-v", f"{out_dir}:/output",
        "-v", f"{logs_dir}:/logs",
        *input_mounts,
        image, "bash", "/input/checks/test.sh",
    ]  # fmt: skip
    with log_path.open("w", encoding="utf-8") as log:
        try:
            subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=CHECKER_TIMEOUT_SEC, check=False
            )
        except subprocess.TimeoutExpired:
            return None, f"checker timed out after {CHECKER_TIMEOUT_SEC}s"
    reward_path = out_dir / "reward.json"
    try:
        reward = json.loads(reward_path.read_text(encoding="utf-8")).get("reward")
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"no readable reward.json ({exc.__class__.__name__})"
    return (float(reward) if reward is not None else None), _pytest_failures(out_dir)


def _pytest_failures(out_dir: pathlib.Path) -> str:
    try:
        report = json.loads((out_dir / "pytest_report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    failed = [
        t.get("nodeid", "?").split("::")[-1]
        for t in report.get("tests", [])
        if t.get("outcome") in {"failed", "error"}
    ]
    return f"failed: {', '.join(failed[:5])}{' …' if len(failed) > 5 else ''}" if failed else ""


# Variables forwarded into an agent container (values come from the environment / .env).
AGENT_ENV_VARS = (
    "MODEL_ENDPOINT",
    "MODEL_NAME",
    "MODEL_TOKEN",
    "HOUSE_REASONING",
    "AGENT_MAX_REVIEWS",
    "HOUSE_CONTEXT_TOKENS",
    "HOUSE_REQUEST_TIMEOUT",
    "AGENT_TIME_BUDGET_SEC",
    "AGENT_DOMAIN_NOTES",
    "AGENT_CANDIDATES",
    "AGENT_CONSENSUS_EXTRA",
    "AGENT_STAGED",
    "AGENT_SPEC_CHECKS",
    "AGENT_GUARDRAILS",
    "AGENT_RED_FLAGS",
    "AGENT_REPAIR_V2",
    "AGENT_EXAMPLES",
    "AGENT_STRUCTURED",
    "AGENT_VERIFY",
    "AGENT_TESTS",
    "AGENT_PLAN",
    "AGENT_EXPLORE",
    "AGENT_SKILLS",
    "AGENT_ADAPTIVE",
    "HOUSE_LATENCY_PRIOR_SEC",
)


def agent_command(
    task_dir: pathlib.Path,
    out_dir: pathlib.Path,
    unit_run: pathlib.Path,
    agent_image: str | None,
    container_name: str,
) -> tuple[list[str], dict[str, str]]:
    """The solve invocation: host interpreter, or the agent image as evaluation runs it.

    The image mode matters because generated scripts execute with the image's libraries
    (the sandbox stack: statsmodels, arch, matplotlib, plotly, openpyxl, ...), which a local
    venv may lack. ``--network=host`` lets the container reach a House model served on the host.
    """
    env = os.environ.copy()
    status_name = "agent_status.json"
    if agent_image is None:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
        env.setdefault("AGENT_TRANSCRIPT_DIR", str(unit_run / "meta" / "transcripts"))
        cmd = [
            sys.executable, "-m", "agent.main", "solve",
            "--task-dir", str(task_dir), "--out", str(out_dir),
            "--status-file", str(unit_run / status_name),
        ]  # fmt: skip
        return cmd, env
    forwarded = [arg for var in AGENT_ENV_VARS if var in env for arg in ("-e", var)]
    cmd = [
        "docker", "run", "--rm", "--name", container_name, "--network=host",
        "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", *forwarded,
        "-v", f"{task_dir}:/input:ro",
        "-v", f"{out_dir}:/app/output", "-v", f"{out_dir}:/output",
        "-v", f"{unit_run / 'meta'}:/agent-meta",
        "-e", "AGENT_TRANSCRIPT_DIR=/agent-meta/transcripts",
        agent_image, "solve", "--task-dir", "/input", "--out", "/app/output",
        "--status-file", f"/agent-meta/{status_name}",
    ]  # fmt: skip
    return cmd, env


def _container_name(run_dir: pathlib.Path, unit_dir: pathlib.Path) -> str:
    return f"agent-{run_dir.name}-{unit_dir.name}".lower()[:120]


def _kill_run_containers(run_dir: pathlib.Path) -> None:
    """Stop this run's agent containers (they would keep using House requests)."""
    listed = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"name=agent-{run_dir.name.lower()}-"],
        capture_output=True,
        text=True,
        check=False,
    )
    ids = listed.stdout.split()
    if ids:
        logger.warning("Killing %d running agent container(s)", len(ids))
        subprocess.run(["docker", "kill", *ids], capture_output=True, check=False)


def run_unit(
    unit_dir: pathlib.Path,
    run_dir: pathlib.Path,
    check_image: str | None,
    timeout_override: float | None,
    agent_image: str | None = None,
    registry: set[str] | None = None,
    keep_inputs: bool = False,
) -> UnitResult:
    """Solve, check, then assess one unit against the admissibility gates."""
    result = _execute_unit(
        unit_dir, run_dir, check_image, timeout_override, agent_image, keep_inputs
    )
    assessment = assess_unit(
        unit_dir,
        pathlib.Path(result.out_dir),
        timed_out=result.status == "timeout",
        reward=result.reward,
        registry=registry or set(),
    )
    for key, value in assessment.items():
        setattr(result, key, value)
    if result.canary_hits and result.status == "passed":
        result.status = "failed"  # g2: an emitted canary disqualifies the attempt
        result.detail = f"canary emitted: {result.canary_hits[:3]}"
    return result


def _execute_unit(
    unit_dir: pathlib.Path,
    run_dir: pathlib.Path,
    check_image: str | None,
    timeout_override: float | None,
    agent_image: str | None,
    keep_inputs: bool = False,
) -> UnitResult:
    unit_run = run_dir / unit_dir.name
    if unit_run.exists():
        # A fresh directory per attempt: a unit interrupted mid-run (then resumed) must not
        # leave half-written outputs for the checker.
        shutil.rmtree(unit_run)
    task_dir = stage_unit(unit_dir, unit_run / "input")
    out_dir = unit_run / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (unit_run / "meta").mkdir(exist_ok=True)
    status_file = unit_run / ("meta" if agent_image else ".") / "agent_status.json"
    timeout = timeout_override or agent_timeout(unit_dir)
    result = UnitResult(
        unit=unit_dir.name, status="error", timeout_sec=timeout, out_dir=str(out_dir)
    )

    container_name = _container_name(run_dir, unit_dir)
    cmd, env = agent_command(task_dir, out_dir, unit_run, agent_image, container_name)
    started = time.perf_counter()
    with (unit_run / "agent.log").open("w", encoding="utf-8") as log:
        # Own process group: on timeout the agent AND the scripts it spawned are killed.
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            if agent_image:  # killing the docker client does not stop the container
                subprocess.run(["docker", "kill", container_name], capture_output=True, check=False)
            result.agent_seconds = time.perf_counter() - started
            if not keep_inputs:
                shutil.rmtree(task_dir, ignore_errors=True)
            result.status = "timeout"
            result.detail = f"agent exceeded card timeout {timeout:.0f}s"
            return result
    result.agent_seconds = time.perf_counter() - started
    if not keep_inputs:
        # The staged copy is reproducible from units/ and dominates a run's disk use.
        shutil.rmtree(task_dir, ignore_errors=True)

    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
        result.agent_succeeded = bool(status.get("succeeded"))
        result.model_requests = status.get("model_requests")
    except (OSError, json.JSONDecodeError):
        result.detail = "agent wrote no status (crashed?); see agent.log"

    if check_image is None:
        result.status = "unchecked" if result.agent_succeeded else "agent_failed"
        return result

    # The checker needs the real unit (with checks/), exactly as README step 6 mounts it.
    reward, detail = run_checker(unit_dir, out_dir, check_image, unit_run / "checker.log")
    result.reward = reward
    result.status = "passed" if reward == 1.0 else "failed"
    result.detail = detail or result.detail
    return result


def run_units(
    units_dir: pathlib.Path,
    run_dir: pathlib.Path,
    patterns: list[str],
    jobs: int,
    check: bool,
    checker_image: str,
    timeout_override: float | None,
    limit: int | None,
    agent_image: str | None = None,
    keep_inputs: bool = False,
    on_start: Callable[[list[pathlib.Path], str | None], None] | None = None,
    prior_results: list[UnitResult] | None = None,
) -> list[UnitResult]:
    """Run the agent over units. ``prior_results`` resumes an interrupted run: those units are
    kept as they are, only the remaining ones run, and the summary covers all of them."""
    units = discover_units(units_dir, patterns)[: limit or None]
    if not units:
        raise ValueError(f"No units matched in {units_dir} (patterns={patterns or 'all'})")
    planned = len(units)
    prior = prior_results or []
    finished = {r.unit for r in prior}
    units = [u for u in units if u.name not in finished]
    if prior:
        logger.info("Resuming: %d units already finished, %d to run", len(prior), len(units))
    run_dir.mkdir(parents=True, exist_ok=True)
    check_image = checker_image if check and checker_available(checker_image) else None
    if on_start is not None:
        on_start(units, check_image)  # e.g. write the experiment manifest
    registry = canary_registry(units_dir)
    logger.info(
        "Running %d units with %d parallel job(s) into %s (agent=%s, checker=%s)",
        len(units),
        jobs,
        run_dir,
        agent_image or "host interpreter",
        check_image or "off",
    )

    results: list[UnitResult] = []
    results_path = run_dir / "results.jsonl"
    pool = ThreadPoolExecutor(max_workers=jobs)
    try:
        _collect(
            pool,
            units,
            run_dir,
            (check_image, timeout_override, agent_image, registry, keep_inputs),
            results,
            results_path,
            append=bool(prior),
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted: cancelling pending units; writing a partial summary")
        pool.shutdown(wait=False, cancel_futures=True)
        if agent_image:
            _kill_run_containers(run_dir)
        if not results:
            raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    results = sorted(prior + results, key=lambda r: r.unit)
    checked = check_image is not None or any(r.reward is not None for r in prior)
    write_summary(run_dir, results, checked=checked, planned=planned)
    return results


def load_results(run_dir: pathlib.Path) -> list[UnitResult]:
    """Per-unit results an interrupted run already recorded (one per unit, last line wins).

    A line cut off by the interruption is dropped, and the file is rewritten with the valid
    lines so resumed results append cleanly.
    """
    path = run_dir / "results.jsonl"
    if not path.is_file():
        return []
    fields = {f.name for f in dataclasses.fields(UnitResult)}
    by_unit: dict[str, UnitResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("unit"):
            by_unit[row["unit"]] = UnitResult(**{k: v for k, v in row.items() if k in fields})
    results = list(by_unit.values())
    path.write_text(
        "".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in results), encoding="utf-8"
    )
    return results


def _collect(
    pool: ThreadPoolExecutor,
    units: list[pathlib.Path],
    run_dir: pathlib.Path,
    unit_args: tuple[str | None, float | None, str | None, set[str], bool],
    results: list[UnitResult],
    results_path: pathlib.Path,
    append: bool = False,
) -> None:
    """Run units on ``pool``, appending each result (and a results.jsonl line) as it lands."""
    with results_path.open("a" if append else "w") as sink:
        futures = {pool.submit(run_unit, u, run_dir, *unit_args): u for u in units}
        for done, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            try:
                res = future.result()
            except Exception as exc:  # staging/IO failure; keep the batch going
                logger.exception("Unit %s crashed the runner", unit.name)
                res = UnitResult(
                    unit=unit.name,
                    status="error",
                    detail=repr(exc),
                    gates={gate: None for gate in ("g0_integrity", "g1_schema")},
                )
            results.append(res)
            sink.write(json.dumps(dataclasses.asdict(res)) + "\n")
            sink.flush()
            logger.info(
                "[%d/%d] %s: %s (reward=%s, requests=%s, %.0fs) %s",
                done,
                len(units),
                res.unit,
                res.status,
                res.reward,
                res.model_requests,
                res.agent_seconds,
                res.detail,
            )


def write_summary(
    run_dir: pathlib.Path, results: list[UnitResult], checked: bool, planned: int | None = None
) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    passed = counts.get("passed", 0)
    rows = [dataclasses.asdict(r) for r in results]
    report = build_report(rows, checked=checked)
    planned = planned or len(results)
    if len(results) < planned:
        # Official pass@1 uses a fixed denominator: unreached units count as failures.
        logger.warning("Partial run: %d of %d units finished", len(results), planned)
        report["official"]["partial"] = f"{len(results)} of {planned} units finished"
        if report["official"]["pass_at_1"] is not None:
            report["official"]["pass_at_1_over_planned"] = round(
                report["official"]["passed"] / planned, 4
            )
    summary = {
        "units": len(results),
        "units_planned": planned,
        "status_counts": counts,
        "pass_at_1": report["official"]["pass_at_1"],
        "metrics": report,
        "results": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    width = max(len(r.unit) for r in results)
    lines = [f"{'unit':<{width}}  {'status':<12}  reward  checks  reqs  agent_s  label / detail"]
    for r in results:
        reward = "-" if r.reward is None else f"{r.reward:.2f}"
        reqs = "-" if r.model_requests is None else str(r.model_requests)
        checks = f"{r.tests_passed}/{r.tests_total}" if r.tests_total else "-"
        note = " ".join(filter(None, [r.failure_label, r.detail]))
        lines.append(
            f"{r.unit:<{width}}  {r.status:<12}  {reward:>6}  {checks:>6}  {reqs:>4}  "
            f"{r.agent_seconds:>7.0f}  {note}"
        )
    lines.append(
        f"\n{len(results)} units: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + (
            f" | pass@1={passed / len(results):.3f}"
            if "passed" in counts or "failed" in counts
            else ""
        )
    )
    lines.append(format_report(report))
    lines.append(f"\nSummary: {run_dir / 'summary.json'}")
    print("\n".join(lines))
