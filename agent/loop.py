"""Core Agent solver loop with multi-turn self-repair.
"""

from __future__ import annotations

import logging
import pathlib
import re
from typing import Any

from agent.client import HouseModelClient
from agent.executor import extract_python_code, run_code
from agent.prompts import (
    SYSTEM_PROMPT,
    build_initial_prompt,
    build_repair_prompt,
)

logger = logging.getLogger(__name__)


class AgentSolver:
    """Manages the iterative solving, execution, and self-repair loop for a task."""

    def __init__(
        self,
        task_dir: pathlib.Path,
        out_dir: pathlib.Path,
        max_retries: int = 4,
        client: HouseModelClient | None = None,
    ) -> None:
        self.task_dir = pathlib.Path(task_dir).resolve()
        self.out_dir = pathlib.Path(out_dir).resolve()
        self.max_retries = max_retries
        self.client = client or HouseModelClient()

    def discover_inputs(self) -> list[pathlib.Path]:
        """Discover data files available in the task directory."""
        data_paths: list[pathlib.Path] = []
        search_dirs = [
            self.task_dir / "environment" / "data",
            self.task_dir / "data",
            self.task_dir,
        ]
        for sdir in search_dirs:
            if sdir.is_dir():
                for p in sdir.rglob("*"):
                    if p.is_file() and p.suffix.lower() in {
                        ".parquet",
                        ".csv",
                        ".json",
                        ".h5",
                        ".pkl",
                        ".txt",
                    }:
                        if p not in data_paths:
                            data_paths.append(p)
        return data_paths

    def find_instruction(self) -> str:
        """Read instruction.md from the task directory."""
        candidates = [
            self.task_dir / "instruction.md",
            self.task_dir / "input" / "instruction.md",
        ]
        for c in candidates:
            if c.is_file():
                return c.read_text(encoding="utf-8")
        raise FileNotFoundError(f"instruction.md not found in {self.task_dir}")

    def detect_expected_deliverables(
        self, instruction_text: str, input_files: list[pathlib.Path] | None = None
    ) -> list[str]:
        """Heuristically extract mentioned deliverable filenames from instruction."""
        input_names = {p.name for p in input_files} if input_files else set()
        # Find references to .parquet, .csv, .json filenames
        pattern = r"[\w\-]+\.(?:parquet|csv|json)"
        matches = re.findall(pattern, instruction_text, re.IGNORECASE)
        # Exclude common metadata/benchmark files and input files
        ignore = {"card.toml", "manifest.json", "reward.json", "pytest_report.json"} | input_names
        deliverables = sorted({m for m in matches if m not in ignore})
        return deliverables

    def check_deliverables(self, expected_names: list[str]) -> tuple[bool, list[str]]:
        """Verify that deliverables exist and are non-empty in out_dir."""
        missing = []
        if expected_names:
            for name in expected_names:
                p = self.out_dir / name
                if not p.exists() or p.stat().st_size == 0:
                    missing.append(name)
            return len(missing) == 0, missing

        # Fallback: check if any valid deliverable file exists
        valid_files = [
            p
            for p in self.out_dir.iterdir()
            if p.is_file()
            and p.name not in {"_solution.py", "reward.json", "pytest_report.json"}
            and p.suffix.lower() in {".parquet", ".csv", ".json"}
            and p.stat().st_size > 0
        ]
        if not valid_files:
            return False, ["(any .parquet, .csv, or .json deliverable)"]
        return True, []

    def run(self) -> bool:
        """Run the end-to-end agent loop."""
        logger.info("Starting AgentSolver for task: %s", self.task_dir.name)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        instruction_text = self.find_instruction()
        input_files = self.discover_inputs()
        expected_deliverables = self.detect_expected_deliverables(
            instruction_text=instruction_text, input_files=input_files
        )

        logger.info(
            "Discovered %d input files, expecting deliverables: %s",
            len(input_files),
            expected_deliverables or "auto-detect",
        )

        relative_inputs = [str(p.relative_to(self.task_dir)) for p in input_files]
        initial_prompt = build_initial_prompt(
            instruction_text=instruction_text,
            task_dir=str(self.task_dir),
            out_dir=str(self.out_dir),
            discovered_files=relative_inputs,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_prompt},
        ]

        # Iterative execution & self-repair loop
        for attempt in range(1, self.max_retries + 2):
            logger.info("Attempt %d/%d: Calling House Model...", attempt, self.max_retries + 1)
            try:
                response = self.client.chat(messages=messages)
            except Exception as e:
                logger.error("Failed to query model: %s", e)
                break

            code = extract_python_code(response)
            if not code:
                logger.warning("No executable Python code block extracted from response.")
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": "No executable Python code block was found. Please provide the complete solution inside a ```python ... ``` code block.",
                    }
                )
                continue

            # Execute code locally
            exec_result = run_code(
                code=code,
                work_dir=self.out_dir,
                target_out_dir=self.out_dir,
            )

            # Check output files
            deliverables_ok, missing = self.check_deliverables(expected_deliverables)

            if exec_result.success and deliverables_ok:
                logger.info(
                    "Execution succeeded on attempt %d. Deliverables verified in %s.",
                    attempt,
                    self.out_dir,
                )
                return True

            # If failed, prepare feedback for the self-repair loop
            logger.warning(
                "Attempt %d failed (returncode=%d, deliverables_ok=%s, missing=%s). Initiating repair.",
                attempt,
                exec_result.returncode,
                deliverables_ok,
                missing,
            )

            err_msg = exec_result.stderr if exec_result.stderr else exec_result.stdout
            if not err_msg and missing:
                err_msg = f"Script completed with exit code 0 but required deliverables were not found in {self.out_dir}."

            repair_prompt = build_repair_prompt(
                previous_code=code,
                error_message=err_msg,
                out_dir=str(self.out_dir),
                missing_deliverables=missing if not deliverables_ok else None,
            )

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": repair_prompt})

        logger.error("Exhausted retries without producing all verified deliverables.")
        return False
