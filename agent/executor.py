"""Local code execution engine for the coding agent.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def extract_python_code(text: str) -> str:
    """Extract Python code block from LLM markdown response."""
    # Look for ```python ... ```
    match = re.search(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Look for generic ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return raw text if no code blocks found
    return text.strip()


class ExecutionResult:
    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        script_path: pathlib.Path,
        timed_out: bool = False,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.script_path = script_path
        self.timed_out = timed_out

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_code(
    code: str,
    work_dir: pathlib.Path,
    target_out_dir: pathlib.Path,
    timeout_sec: int = 300,
) -> ExecutionResult:
    """Write and execute the Python code locally inside the container."""
    work_dir.mkdir(parents=True, exist_ok=True)
    target_out_dir.mkdir(parents=True, exist_ok=True)

    script_path = work_dir / "_solution.py"
    script_path.write_text(code, encoding="utf-8")

    env = os.environ.copy()
    # Provide hints to script via environment
    env["OUTPUT_DIR"] = str(target_out_dir)
    env["PYTHONUNBUFFERED"] = "1"

    logger.info("Executing generated solution in %s (timeout=%ds)...", work_dir, timeout_sec)

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        result = ExecutionResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            script_path=script_path,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("Code execution timed out after %d seconds", timeout_sec)
        result = ExecutionResult(
            returncode=-1,
            stdout=exc.stdout or "",
            stderr=f"Execution timed out after {timeout_sec} seconds.\n" + (exc.stderr or ""),
            script_path=script_path,
            timed_out=True,
        )

    # Safety net: If deliverables were saved in work_dir instead of target_out_dir,
    # copy relevant data files (.parquet, .csv, .json) over to target_out_dir.
    if work_dir != target_out_dir:
        for file in work_dir.iterdir():
            if file.is_file() and file.name not in {"_solution.py", "reward.json", "pytest_report.json"}:
                if file.suffix.lower() in {".parquet", ".csv", ".json", ".h5", ".pkl"}:
                    dest = target_out_dir / file.name
                    if not dest.exists():
                        logger.info("Copying misplaced deliverable %s -> %s", file.name, dest)
                        shutil.copy2(file, dest)

    return result
