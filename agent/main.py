"""Command-line entrypoint for the Track 1 Submission Agent.

Conforms to the competition submission CLI contract:
  solve --task-dir <path> --out <path>
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from agent.loop import AgentSolver


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Agenthon Track 1 Coding Agent")
    parser.add_argument("verb", choices=["solve"], help="Competition verb (must be 'solve')")
    parser.add_argument(
        "--task-dir",
        required=True,
        help="Path to the task directory containing instruction.md and data",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the destination directory for agent deliverables (/app/output)",
    )

    args = parser.parse_args(argv)

    task_dir = pathlib.Path(args.task_dir).resolve()
    out_dir = pathlib.Path(args.out).resolve()

    solver = AgentSolver(task_dir=task_dir, out_dir=out_dir)
    try:
        success = solver.run()
        if not success:
            logging.warning("Agent completed with partial or unverified deliverables.")
    except Exception as e:
        logging.exception("Fatal error during agent execution: %s", e)
        # Even if solving fails, exit 0 to allow the checker (test.sh) to record the attempt and output logs
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
