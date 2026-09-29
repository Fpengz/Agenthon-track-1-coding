"""Local code execution engine for the coding agent."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
import tokenize

logger = logging.getLogger(__name__)
_LOG_PREVIEW_LIMIT = 1200
_FEEDBACK_STDOUT_LIMIT = 1500
_FEEDBACK_STDERR_LIMIT = 3000


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _preview(value: str) -> str:
    if len(value) <= _LOG_PREVIEW_LIMIT:
        return value
    return f"{value[:_LOG_PREVIEW_LIMIT]}… [truncated]"


_THINK_END = "</think>"
_FENCED_BLOCK = re.compile(r"^```[ \t]*(\w*)[^\n]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)


def strip_reasoning(text: str) -> str:
    """Drop inline chain-of-thought emitted before the final answer.

    The House model (served without a reasoning parser) writes its thinking into
    ``content`` and closes it with ``</think>``, usually without an opening tag.
    """
    if _THINK_END in text:
        return text.rsplit(_THINK_END, 1)[1]
    return text


def _compiles(code: str) -> bool:
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError:
        return False
    return True


def extract_python_code(text: str) -> str:
    """Extract the solution script from an LLM markdown response.

    Returns an empty string when no closed code block is present (for example when
    the response was cut off at the output-token limit). Prose is never returned as
    code.
    """
    answer = strip_reasoning(text)
    blocks = [
        # Trim blank lines only: a continuation block may start inside an indented body.
        (lang.lower(), re.sub(r"\A(?:[ \t]*\n)+", "", body).rstrip())
        for lang, body in _FENCED_BLOCK.findall(answer)
        if lang.lower() in {"python", "py", "python3", ""}
        and body.strip()
        and EDIT_MARKER not in body
    ]
    if not blocks and _THINK_END in text:
        # Reasoning ended without an answer block; fall back to blocks inside the reasoning.
        return extract_python_code(text.replace(_THINK_END, ""))
    if not blocks:
        return ""

    # Prefer the last block that is valid Python (later blocks are usually the final version).
    for _, body in reversed(blocks):
        if _compiles(body):
            return body
    return blocks[-1][1]


_FENCE_LINE = re.compile(r"^```[^\n]*$", re.MULTILINE)


def extract_partial_code(text: str) -> str:
    """The code of a final ``` block that was cut off before its closing fence.

    Only complete lines are kept (a token-limit cut usually lands mid-line), so a continuation
    can resume from the next line. Returns ``""`` if the answer has no unclosed code block.
    """
    answer = strip_reasoning(text)
    inside, start = False, 0
    for fence in _FENCE_LINE.finditer(answer):
        inside = not inside
        start = fence.end() + 1
    if not inside:
        return ""
    tail = answer[start:]
    return _complete_statements(tail[: tail.rfind("\n")]) if "\n" in tail else ""


def _complete_statements(code: str) -> str:
    """Longest prefix of ``code`` that ends at a complete logical line.

    A token-limit cut often lands inside a multi-line statement (an open bracket or string);
    resuming after it makes the continuation restart that statement, leaving the bracket
    unclosed. Trimming back to the last logical-line end gives the continuation a clean seam.
    """
    lines = code.splitlines(keepends=True)
    last_end = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.NEWLINE:
                last_end = tok.end[0]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # EOF inside a bracket/string: keep what was complete before it
    return "".join(lines[:last_end]).rstrip()


def join_continuation(partial: str, continuation: str) -> str:
    """Append a continuation to a cut-off script, tolerating a model that restarted it."""
    first_line = next((ln for ln in partial.splitlines() if ln.strip()), "")
    if first_line and first_line in continuation.splitlines()[:5] and _compiles(continuation):
        return continuation  # the model rewrote the whole script instead of continuing
    return f"{partial}\n{continuation}"


_EDIT_BLOCK = re.compile(
    r"^<{5,9} ?SEARCH[ \t]*\n(.*?)^={5,9}[ \t]*\n(.*?)^>{5,9} ?REPLACE[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
EDIT_MARKER = "<<<<<<< SEARCH"


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _find_line_runs(lines: list[str], target: list[str], key) -> list[int]:
    wanted = [key(t) for t in target]
    return [
        i
        for i in range(len(lines) - len(target) + 1)
        if [key(ln) for ln in lines[i : i + len(target)]] == wanted
    ]


def _replace_lines_tolerant(code: str, search: str, replace: str) -> tuple[str | None, str]:
    """Replace the unique run of lines matching ``search`` loosely; returns (code, problem).

    First ignores trailing whitespace, then indentation too -- the model often copies the right
    lines at the wrong nesting level -- re-indenting the replacement by the same offset.
    """
    lines = code.splitlines()
    target = search.splitlines()
    if not any(t.strip() for t in target):
        return None, "empty SEARCH"
    for key, reindent in ((str.rstrip, False), (str.strip, True)):
        hits = _find_line_runs(lines, target, key)
        if len(hits) > 1:
            return None, f"matches {len(hits)} places; include more surrounding lines"
        if len(hits) == 1:
            i = hits[0]
            new_lines = replace.splitlines()
            if reindent:
                first = next(n for n, t in enumerate(target) if t.strip())
                have, want = _indent(target[first]), _indent(lines[i + first])
                new_lines = [
                    want + ln[len(have) :] if ln.startswith(have) else want + ln.lstrip()
                    for ln in new_lines
                ]
            return "\n".join(lines[:i] + new_lines + lines[i + len(target) :]), ""
    stripped = {ln.strip() for ln in lines}
    missing = next((t.strip() for t in target if t.strip() and t.strip() not in stripped), "")
    return None, f"line not in the script: {missing[:100]!r}" if missing else "lines not contiguous"


def apply_edits(code: str, response: str) -> tuple[str | None, list[str]]:
    """Apply SEARCH/REPLACE edit blocks from ``response`` to ``code``.

    Returns ``(new_code, [])`` when every block applied, ``(None, problems)`` when some block
    could not be placed unambiguously, and ``(None, [])`` when the response has no edit blocks.
    Edits keep long-script repairs small: a one-line fix no longer needs a 300-line rewrite
    that overruns the output cap and invites the model to copy the old script back.
    """
    blocks = _EDIT_BLOCK.findall(strip_reasoning(response))
    if not blocks:
        return None, []
    new, problems = code, []
    for search, replace in blocks:
        search, replace = search.rstrip("\n"), replace.rstrip("\n")
        if search and new.count(search) == 1:
            new = new.replace(search, replace, 1)
            continue
        placed, problem = _replace_lines_tolerant(new, search, replace)
        if placed is None:
            first = search.strip().splitlines()[0][:80] if search.strip() else ""
            problems.append(f"SEARCH starting {first!r}: {problem}")
        else:
            new = placed
    return (None, problems) if problems else (new, [])


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
        self.duration = 0.0

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def error_signature(self) -> str:
        """Last non-empty stderr line (e.g. ``KeyError: ...``), to spot a repeated failure."""
        lines = [ln.strip() for ln in self.stderr.splitlines() if ln.strip()]
        return lines[-1][:300] if lines else ""

    @property
    def feedback(self) -> str:
        """Condensed stdout/stderr for the repair prompt, keeping the traceback tail."""
        parts = []
        if self.stdout.strip():
            parts.append(f"[stdout]\n{_tail(self.stdout, _FEEDBACK_STDOUT_LIMIT)}")
        if self.stderr.strip():
            parts.append(f"[stderr]\n{_tail(self.stderr, _FEEDBACK_STDERR_LIMIT)}")
        return "\n".join(parts)


def _tail(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"… [{len(value) - limit} characters truncated]\n{value[-limit:]}"


def run_process(cmd: list[str], timeout_sec: float, **kwargs) -> tuple[int, str, str, bool]:
    """Run ``cmd`` in its own process group; on timeout kill the WHOLE group.

    ``subprocess.run(timeout=...)`` kills only the direct child: workers a generated script
    spawns (multiprocessing, joblib) keep the output pipes open, and the call then blocks far
    past its timeout. Returns ``(returncode, stdout, stderr, timed_out)``.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, **kwargs
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return proc.returncode, _as_text(stdout), _as_text(stderr), False
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        return -1, _as_text(stdout), _as_text(stderr), True


_MISPLACED_SUFFIXES = frozenset({".parquet", ".csv", ".json", ".h5", ".pkl", ".html", ".png"})


def run_code(
    code: str,
    work_dir: pathlib.Path,
    target_out_dir: pathlib.Path,
    task_dir: pathlib.Path,
    timeout_sec: int = 300,
) -> ExecutionResult:
    """Write and execute the Python code locally inside the container.

    Each attempt runs in a fresh ``work_dir/attempt`` scratch directory (never the deliverable
    directory, and never shared with earlier attempts), so only real deliverables end up in
    ``target_out_dir``.
    """
    attempt_dir = work_dir / "attempt"
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True)
    target_out_dir.mkdir(parents=True, exist_ok=True)

    script_path = attempt_dir / "_solution.py"
    script_path.write_text(code, encoding="utf-8")
    logger.info(
        "Saved generated Python solution to %s (%d characters)",
        script_path,
        len(code),
    )

    env = os.environ.copy()
    # The script resolves its input/output locations from these variables.
    env["TASK_DIR"] = str(task_dir)
    env["OUTPUT_DIR"] = str(target_out_dir)
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"  # headless: chart deliverables must not need a display

    logger.info(
        "Executing generated solution (cwd=%s, output_dir=%s, timeout=%ds)",
        attempt_dir,
        target_out_dir,
        timeout_sec,
    )
    started_at = time.perf_counter()
    returncode, stdout, stderr, timed_out = run_process(
        [sys.executable, str(script_path)], timeout_sec, cwd=str(attempt_dir), env=env
    )
    if timed_out:
        logger.warning("Code execution timed out after %d seconds", timeout_sec)
        stderr = f"Execution timed out after {timeout_sec} seconds.\n{stderr}"
    result = ExecutionResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        script_path=script_path,
        timed_out=timed_out,
    )
    result.duration = time.perf_counter() - started_at
    logger.info(
        "Generated solution finished (returncode=%d, timed_out=%s, duration=%.2fs, "
        "stdout_chars=%d, stderr_chars=%d)",
        result.returncode,
        result.timed_out,
        result.duration,
        len(result.stdout),
        len(result.stderr),
    )
    if result.success:
        logger.debug("Generated solution stdout:\n%s", _preview(result.stdout))
    else:
        if result.stdout:
            logger.warning("Generated solution stdout preview:\n%s", _preview(result.stdout))
        if result.stderr:
            logger.warning("Generated solution stderr preview:\n%s", _preview(result.stderr))

    # Safety net: deliverables written to the working directory instead of OUTPUT_DIR.
    for file in attempt_dir.iterdir():
        if (
            file.is_file()
            and file.name not in {"_solution.py", "reward.json", "pytest_report.json"}
            and file.suffix.lower() in _MISPLACED_SUFFIXES
            and not (target_out_dir / file.name).exists()
        ):
            logger.info("Copying misplaced deliverable %s -> %s", file.name, target_out_dir)
            shutil.copy2(file, target_out_dir / file.name)

    return result
