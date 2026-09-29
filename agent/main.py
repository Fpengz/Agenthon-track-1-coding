"""Typer command-line entrypoint for the Track 1 submission agent.

The public interface remains ``solve --task-dir <path> --out <path>``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from agent.batch import run_units
from agent.examples import LIBRARY, harvest, load_library
from agent.loop import AgentSolver

logger = logging.getLogger(__name__)
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

app = typer.Typer(
    add_completion=False,
    help="Agenthon Track 1 financial coding agent.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def cli() -> None:
    """Run the Agenthon Track 1 coding agent."""


def setup_logging() -> None:
    """Configure concise, timestamped logs for container runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.captureWarnings(True)


@app.command()
def solve(
    task_dir: Annotated[
        Path,
        typer.Option(
            "--task-dir",
            help="Task directory containing instruction.md and task data.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Destination directory for task deliverables (normally /app/output).",
        ),
    ],
    status_file: Annotated[
        Path | None,
        typer.Option(
            "--status-file",
            hidden=True,
            help="Write a JSON run status here (used by solve-units; keep it outside --out).",
        ),
    ] = None,
) -> None:
    """Solve one task and write deliverables to the output directory."""
    task_path = task_dir.expanduser().resolve()
    out_path = out.expanduser().resolve()
    logger.info("Starting solve command (task_dir=%s, out=%s)", task_path, out_path)

    solver: AgentSolver | None = None
    succeeded = False
    try:
        solver = AgentSolver(task_dir=task_path, out_dir=out_path)
        succeeded = solver.run()
    except Exception:
        logger.exception("Fatal error while solving task %s", task_path)
        # Keep the competition contract: the checker must be allowed to record this attempt.
    finally:
        if status_file is not None:
            status = {
                "succeeded": succeeded,
                "model_requests": getattr(solver.client, "request_count", None) if solver else 0,
            }
            try:
                status_file.write_text(json.dumps(status), encoding="utf-8")
            except OSError:
                # Never let a debug side channel mask the solve outcome or its exception.
                logger.warning("Could not write status file %s", status_file, exc_info=True)

    if succeeded:
        logger.info("Solve command completed successfully")
    else:
        logger.warning("Solve command finished without verified deliverables")


@app.command("solve-units")
def solve_units(
    patterns: Annotated[
        list[str] | None,
        typer.Argument(help="Unit names or globs to run (e.g. 't1-EXAMPLE-*'). Default: all."),
    ] = None,
    units_dir: Annotated[
        Path, typer.Option("--units-dir", help="Folder containing unit directories.")
    ] = Path("units"),
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Where to write per-unit runs. Default: tmp/runs/<time>."),
    ] = None,
    jobs: Annotated[int, typer.Option("--jobs", "-j", min=1, help="Units to run in parallel.")] = 1,
    check: Annotated[
        bool,
        typer.Option(
            "--check/--no-check",
            help="Run each unit's checker in the sandbox image (README step 6).",
        ),
    ] = True,
    checker_image: Annotated[
        str, typer.Option("--checker-image", help="Sandbox image that runs checks/test.sh.")
    ] = "finance-bench-sandbox:latest",
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="Override every card's agent timeout (seconds)."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Run at most this many units.")
    ] = None,
    agent_image: Annotated[
        str | None,
        typer.Option(
            "--agent-image",
            help="Run solve inside this agent image (as evaluation does) instead of the host "
            "interpreter, so generated scripts get the image's libraries.",
        ),
    ] = None,
) -> None:
    """Run the agent over every unit in a folder and summarise the results."""
    run_path = (run_dir or Path("tmp/runs") / time.strftime("%Y%m%d-%H%M%S")).resolve()
    try:
        run_units(
            units_dir=units_dir.expanduser().resolve(),
            run_dir=run_path,
            patterns=patterns or [],
            jobs=jobs,
            check=check,
            checker_image=checker_image,
            timeout_override=timeout,
            limit=limit,
            agent_image=agent_image,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=2) from exc


@app.command("build-examples")
def build_examples(
    run_dirs: Annotated[
        list[Path], typer.Argument(help="solve-units run directories to harvest passed units from.")
    ],
    units_dir: Annotated[Path, typer.Option("--units-dir")] = Path("units"),
    out: Annotated[Path, typer.Option("--out", help="Library file (merged, not replaced).")] = (
        LIBRARY
    ),
) -> None:
    """Collect checker-passed solutions into the few-shot library (used for OTHER units only)."""
    existing = {ex["unit_id"]: ex for ex in load_library(out)}
    added = 0
    for ex in harvest([d.resolve() for d in run_dirs], units_dir.resolve()):
        previous = existing.get(ex["unit_id"])
        if previous is None or len(ex["solution"]) < len(previous["solution"]):
            existing[ex["unit_id"]] = ex
            added += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(ex) for ex in sorted(existing.values(), key=lambda e: e["unit_id"])]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Library %s: %d examples (%d added or replaced)", out, len(existing), added)


def main(argv: list[str] | None = None) -> int:
    """Run the Typer app while preserving an integer-returning module entrypoint."""
    setup_logging()
    if load_dotenv(dotenv_path=ENV_FILE, override=False):
        logger.info("Loaded environment variables from %s", ENV_FILE)
    try:
        app(args=argv, prog_name="agent")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
