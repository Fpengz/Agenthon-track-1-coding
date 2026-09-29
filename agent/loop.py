"""Core Agent solver loop with multi-turn self-repair."""

from __future__ import annotations

import logging
import os
import pathlib
import re
import tempfile
import time

from openai.types.chat import ChatCompletionMessageParam

from agent.client import MAX_OUTPUT_TOKENS, HouseModelClient
from agent.examples import format_example, select_example
from agent.executor import (
    apply_edits,
    extract_partial_code,
    extract_python_code,
    join_continuation,
    run_code,
)
from agent.inputs import build_input_previews
from agent.knowledge import domain_notes, read_card
from agent.prompts import (
    SYSTEM_PROMPT,
    build_continuation_prompt,
    build_initial_prompt,
    build_no_code_prompt,
    build_repair_prompt,
    build_review_prompt,
)
from agent.review import (
    output_diagnostics,
    output_files,
    parse_verdict,
    restore_outputs,
    snapshot_outputs,
)

logger = logging.getLogger(__name__)
NO_CODE_MARKER = "\n\n### NOTE:\n"
# Follow-up requests allowed for one script that outgrows the 4,000-token output cap.
MAX_CONTINUATIONS = 2
# Self-reviews of a clean run before it is accepted (each costs one House request).
MAX_REVIEWS = int(os.environ.get("AGENT_MAX_REVIEWS", "2"))  # 0 disables review (A/B runs)
# No new review starts after this many seconds: the whole roster shares one wall-clock
# allowance (12 h for 86 units in Development, ~8 min per unit on average).
REVIEW_DEADLINE_SEC = 480.0
EXEC_TIMEOUT_SEC = 300
# Per-unit time budget. The card's [agent].timeout_sec is a hard limit (a timeout scores 0 even
# with good outputs), and the whole roster shares one wall-clock allowance (12 h / 86 units in
# Development), so the agent stops starting attempts at the smaller of 85% of the card timeout
# and AGENT_TIME_BUDGET_SEC, and clips script timeouts to the time left.
TIME_BUDGET_SEC = float(os.environ.get("AGENT_TIME_BUDGET_SEC", "600"))
CARD_TIMEOUT_SHARE = 0.85
DEFAULT_CARD_TIMEOUT_SEC = 1800.0
MIN_ATTEMPT_SEC = 60.0  # do not start an attempt (request + run) with less time than this
MIN_EXEC_TIMEOUT_SEC = 30
# Repairs/reviews of scripts at least this long ask for SEARCH/REPLACE edits, not a rewrite.
EDIT_MODE_MIN_LINES = 80
# Context guard. The House route documents no serving window, and an over-long request is
# rejected upstream after admission, so it still costs one of the 25 requests. The local vLLM
# serves 32,768 tokens. 2.5 chars/token sits just below the lowest ratio measured on real
# prompts (2.53; median 3.24), so estimates err toward "too big".
CONTEXT_TOKENS = int(os.environ.get("HOUSE_CONTEXT_TOKENS", "32768"))
CHARS_PER_TOKEN = 2.5
CONTEXT_MARGIN_TOKENS = 256
MIN_OUTPUT_TOKENS = 1024
# Optional task-prompt parts, dropped cumulatively in this order when a request would not fit.
_DROP_ORDER: list[frozenset[str]] = [
    frozenset(),
    frozenset({"reference_example"}),
    frozenset({"reference_example", "domain_notes"}),
    frozenset({"reference_example", "domain_notes", "input_previews"}),
]


def _estimated_tokens(user_prompt: str) -> int:
    return int((len(SYSTEM_PROMPT) + len(user_prompt)) / CHARS_PER_TOKEN)


def fit_to_context(
    user_prompt: str, task_prompt: str, variants: list[str]
) -> tuple[str, str, int] | None:
    """Make a request fit the context window: ``(user_prompt, task_prompt, max_tokens)``.

    Every turn's prompt starts with the task prompt, so a leaner task-prompt variant can be
    swapped in (and kept for later turns). If even the leanest does not fit with the full output
    cap, the output cap shrinks; ``None`` if not even ``MIN_OUTPUT_TOKENS`` would remain.
    """
    budget = CONTEXT_TOKENS - MAX_OUTPUT_TOKENS - CONTEXT_MARGIN_TOKENS
    if _estimated_tokens(user_prompt) <= budget:
        return user_prompt, task_prompt, MAX_OUTPUT_TOKENS
    suffix = user_prompt[len(task_prompt) :] if user_prompt.startswith(task_prompt) else None
    if suffix is not None:
        start = variants.index(task_prompt) + 1 if task_prompt in variants else len(variants)
        for variant in variants[start:]:
            task_prompt, user_prompt = variant, variant + suffix
            if _estimated_tokens(user_prompt) <= budget:
                logger.warning(
                    "Prompt too large for the context window; using a leaner task prompt "
                    "(~%d tokens)",
                    _estimated_tokens(user_prompt),
                )
                return user_prompt, task_prompt, MAX_OUTPUT_TOKENS
    room = CONTEXT_TOKENS - CONTEXT_MARGIN_TOKENS - _estimated_tokens(user_prompt)
    if room < MIN_OUTPUT_TOKENS:
        return None
    logger.warning("Prompt near the context limit; reducing the output cap to %d tokens", room)
    return user_prompt, task_prompt, min(room, MAX_OUTPUT_TOKENS)


def _edit_mode(code: str) -> bool:
    return len(code.splitlines()) >= EDIT_MODE_MIN_LINES


def _review_findings(response: str) -> str:
    """The problem list from a review reply that was cut off before its corrected script."""
    answer = response.split("</think>")[-1]
    tail = answer.split("VERDICT", 1)[-1] if "VERDICT" in answer else answer
    return f"Review findings:\n{tail.strip()[:4000]}"


# Graded answers in a local unit checkout; never mounted at evaluation (scripts/build_dev_dataset.py).
ANSWER_DIRS = frozenset({"checks", "reference"})
DATA_SUFFIXES = frozenset(
    {".parquet", ".pqt", ".csv", ".tsv", ".json", ".jsonl", ".h5", ".pkl", ".txt", ".xlsx", ".xml"}
)
# Deliverables named as `/output/<name>` in an instruction (charts and reports included).
OUTPUT_REF = re.compile(
    r"/output/([\w\-]+\.(?:parquet|csv|tsv|jsonl?|html|png|svg|pdf|txt|md|xlsx|npy))",
    re.IGNORECASE,
)


def container_path_map(task_dir: pathlib.Path) -> list[tuple[str, str]]:
    """Map image paths an instruction may cite (e.g. ``/app/data/x.parquet``) to task files.

    Unit instructions describe the unit image, whose ``environment/Dockerfile`` COPYs data
    (sometimes renamed) into ``/app``. A submission never runs in that image: it gets the raw
    unit at ``/input``. Resolving the COPY lines tells the model which task file each cited
    path is. Returns ``(container_path, path relative to task_dir)`` pairs.
    """
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return []
    context = task_dir / "environment"
    mapping: list[tuple[str, str]] = []
    for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0].upper() not in {"COPY", "ADD"}:
            continue
        args = [a for a in parts[1:] if not a.startswith("--")]
        *sources, dest = args
        for src in sources:
            src_path = context / src
            if src_path.is_dir():
                for f in sorted(p for p in src_path.rglob("*") if p.is_file()):
                    target = f"{dest.rstrip('/')}/{f.relative_to(src_path).as_posix()}"
                    mapping.append((target, f.relative_to(task_dir).as_posix()))
            elif src_path.is_file():
                into_dir = dest.endswith("/") or len(sources) > 1
                target = f"{dest.rstrip('/')}/{src_path.name}" if into_dir else dest
                mapping.append((target, src_path.relative_to(task_dir).as_posix()))
    return mapping


class AgentSolver:
    """Manages the iterative solving, execution, and self-repair loop for a task."""

    def __init__(
        self,
        task_dir: pathlib.Path,
        out_dir: pathlib.Path,
        max_retries: int = 15,
        client: HouseModelClient | None = None,
    ) -> None:
        self.task_dir = pathlib.Path(task_dir).resolve()
        self.out_dir = pathlib.Path(out_dir).resolve()
        self.max_retries = max_retries
        self.client = client or HouseModelClient()
        self.deadline = time.monotonic() + self._time_budget()
        logger.debug(
            "Configured AgentSolver (task_dir=%s, out_dir=%s, max_retries=%d)",
            self.task_dir,
            self.out_dir,
            self.max_retries,
        )

    def _time_budget(self) -> float:
        card = read_card(self.task_dir).get("agent", {})
        try:
            card_timeout = float(card.get("timeout_sec", DEFAULT_CARD_TIMEOUT_SEC))
        except (TypeError, ValueError):
            card_timeout = DEFAULT_CARD_TIMEOUT_SEC
        return min(CARD_TIMEOUT_SHARE * card_timeout, TIME_BUDGET_SEC)

    def _time_left(self) -> float:
        return self.deadline - time.monotonic()

    def discover_inputs(self) -> list[pathlib.Path]:
        """Discover the input files supplied with the task.

        Everything under ``environment/`` (except its Dockerfile) is task data, whatever its
        format (.pqt, .tsv, .xlsx, .xml, .zip, ...); elsewhere only data-like files count.
        """
        data_paths: list[pathlib.Path] = []
        for p in sorted(self.task_dir.rglob("*")):
            rel = p.relative_to(self.task_dir)
            # A local unit checkout also carries graded answers; they are never
            # mounted at evaluation and must not reach the prompt.
            if not p.is_file() or ANSWER_DIRS.intersection(rel.parts[:1]):
                continue
            if "__pycache__" in rel.parts or p.name in {"card.toml", "instruction.md"}:
                continue
            if rel.parts[0] == "environment":
                if p.name != "Dockerfile":
                    data_paths.append(p)
            elif p.suffix.lower() in DATA_SUFFIXES:
                data_paths.append(p)
        logger.info(
            "Input discovery found %d supported files: %s",
            len(data_paths),
            [
                f"{path.relative_to(self.task_dir)} ({path.stat().st_size} bytes)"
                for path in data_paths
            ],
        )
        return data_paths

    def find_instruction(self) -> str:
        """Read instruction.md from the task directory."""
        candidates = [
            self.task_dir / "instruction.md",
            self.task_dir / "input" / "instruction.md",
        ]
        for c in candidates:
            if c.is_file():
                instruction = c.read_text(encoding="utf-8")
                logger.info(
                    "Loaded task instruction from %s (%d characters)",
                    c,
                    len(instruction),
                )
                return instruction
        raise FileNotFoundError(f"instruction.md not found in {self.task_dir}")

    def detect_expected_deliverables(
        self, instruction_text: str, input_files: list[pathlib.Path] | None = None
    ) -> list[str]:
        """Heuristically extract mentioned deliverable filenames from instruction."""
        input_names = {p.name for p in input_files} if input_files else set()
        # Exclude common metadata/benchmark files and input files
        ignore = {"card.toml", "manifest.json", "reward.json", "pytest_report.json"} | input_names
        # Prefer explicit `/output/<name>` references; fall back to any data filename.
        explicit = OUTPUT_REF.findall(instruction_text)
        matches = explicit or re.findall(
            r"[\w\-]+\.(?:parquet|csv|json)", instruction_text, re.IGNORECASE
        )
        deliverables = sorted({m for m in matches if m not in ignore})
        return deliverables

    def clear_stale_deliverables(self, expected_names: list[str]) -> None:
        """Remove expected deliverables left over from earlier attempts or runs."""
        for name in expected_names:
            path = self.out_dir / name
            if path.is_file():
                logger.info("Removing stale deliverable before execution: %s", path)
                path.unlink()

    def check_deliverables(
        self, expected_names: list[str], produced_after: float = 0.0
    ) -> tuple[bool, list[str]]:
        """Verify that deliverables exist, are non-empty, and were written by this attempt."""
        missing = []
        if expected_names:
            for name in expected_names:
                p = self.out_dir / name
                if not p.exists() or p.stat().st_size == 0 or p.stat().st_mtime < produced_after:
                    missing.append(name)
            logger.info(
                "Deliverable check: expected=%s, missing=%s",
                expected_names,
                missing,
            )
            return len(missing) == 0, missing

        # Fallback: check if any valid deliverable file exists
        valid_files = [
            p
            for p in self.out_dir.iterdir()
            if p.is_file()
            and p.name not in {"_solution.py", "reward.json", "pytest_report.json"}
            and p.suffix.lower() in DATA_SUFFIXES | {".html", ".png", ".svg", ".pdf", ".md"}
            and p.stat().st_size > 0
            and p.stat().st_mtime >= produced_after
        ]
        if not valid_files:
            logger.warning("Deliverable check found no non-empty output files in %s", self.out_dir)
            return False, ["(any deliverable file)"]
        logger.info("Deliverable check found output files: %s", [path.name for path in valid_files])
        return True, []

    def run(self) -> bool:
        """Run the end-to-end agent loop."""
        logger.info("Starting AgentSolver for task: %s", self.task_dir.name)
        logger.info("Task directory: %s", self.task_dir)
        logger.info("Deliverable directory: %s", self.out_dir)
        logger.info("Repair attempts available: %d", self.max_retries)
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
        path_map = container_path_map(self.task_dir)
        if path_map:
            logger.info("Resolved %d container path(s) from environment/Dockerfile", len(path_map))
        optional = {
            "input_previews": build_input_previews(self.task_dir, input_files),
            "domain_notes": domain_notes(self.task_dir, instruction_text),
            "reference_example": format_example(select_example(self.task_dir, instruction_text)),
        }
        # Task prompts from richest to leanest; the context guard falls back along this list.
        prompt_variants = [
            build_initial_prompt(
                instruction_text=instruction_text,
                task_dir=str(self.task_dir),
                out_dir=str(self.out_dir),
                discovered_files=relative_inputs,
                path_map=path_map,
                **{k: v for k, v in optional.items() if k not in dropped},
            )
            for dropped in _DROP_ORDER
        ]
        initial_prompt = prompt_variants[0]
        logger.info(
            "Prepared initial prompt (system_chars=%d, user_chars=%d)",
            len(SYSTEM_PROMPT),
            len(initial_prompt),
        )

        with tempfile.TemporaryDirectory(prefix="agent-work-") as work_dir:
            return self._solve_loop(
                prompt_variants=prompt_variants,
                expected_deliverables=expected_deliverables,
                work_dir=pathlib.Path(work_dir),
            )

    def _solve_loop(
        self,
        prompt_variants: list[str],
        expected_deliverables: list[str],
        work_dir: pathlib.Path,
    ) -> bool:
        initial_prompt = prompt_variants[0]
        # The conversation is rebuilt each turn instead of accumulated: the latest repair
        # prompt already carries the previous code and error, and replaying the model's
        # inline reasoning would overflow the House model's 32k context within a few turns.
        user_prompt = initial_prompt
        temperature = 0.0
        partial_code = ""  # prefix of a script cut off at the output cap, while continuing it
        continuations = 0
        reviews = 0
        reviewing = False  # the pending request is a review of a clean run
        accepted_dir = work_dir / "accepted"  # snapshot of the latest clean run's outputs
        have_accepted = False
        accepted_seconds = 0.0  # runtime of the accepted script
        reviewed_code = ""
        last_error = ""
        last_failed_code = ""
        last_feedback = ""
        edit_base = ""  # the script shown in the pending repair/review prompt (edits apply to it)
        loop_started = time.monotonic()

        # Iterative execution & self-repair loop
        try:
            for attempt in range(1, self.max_retries + 2):
                if self._time_left() < MIN_ATTEMPT_SEC:
                    logger.warning("Time budget exhausted before attempt %d; stopping", attempt)
                    break
                logger.info("Attempt %d/%d: Calling House Model...", attempt, self.max_retries + 1)
                messages: list[ChatCompletionMessageParam] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    fitted = fit_to_context(user_prompt, initial_prompt, prompt_variants)
                    if fitted is None:
                        logger.error("Request cannot fit the model context window; stopping")
                        break
                    user_prompt, initial_prompt, max_tokens = fitted
                    messages[1]["content"] = user_prompt
                    result = self.client.chat(
                        messages=messages, temperature=temperature, max_tokens=max_tokens
                    )
                except Exception:
                    logger.exception("Failed to query the model on attempt %d", attempt)
                    break

                code = extract_python_code(result.content)
                if edit_base:
                    edited, unmatched = apply_edits(edit_base, result.content)
                    if edited is not None:
                        logger.info(
                            "Applied edit blocks to the %d-line script", len(edit_base.splitlines())
                        )
                        code = edited
                    elif unmatched and not code:
                        logger.warning("Edit blocks did not match the script: %s", unmatched[:3])
                        user_prompt = (
                            f"{user_prompt.split(NO_CODE_MARKER)[0]}{NO_CODE_MARKER}Your edit "
                            "blocks could not be applied (no edit was made). Resend ALL edits, "
                            "fixing these: " + "; ".join(unmatched[:5])
                        )
                        continue
                if reviewing:
                    reviewing = False
                    verdict = parse_verdict(result.content)
                    cut_off_fix = result.truncated and bool(extract_partial_code(result.content))
                    logger.info("Review verdict: %s (corrected script: %s)", verdict, bool(code))
                    if verdict == "PASS" or (verdict is None and not (code or cut_off_fix)):
                        return self._accept(attempt, "review passed", reviewed_code)
                    if not (code or cut_off_fix):
                        # FAIL, but the reply ran out of tokens before the corrected script: turn the
                        # review's findings into an ordinary repair turn instead of accepting.
                        logger.warning("Review failed without a corrected script; requesting a fix")
                        edit_base = reviewed_code
                        user_prompt = build_repair_prompt(
                            edit_mode=_edit_mode(reviewed_code),
                            task_prompt=initial_prompt,
                            previous_code=reviewed_code,
                            error_message=_review_findings(result.content),
                            out_dir=str(self.out_dir),
                            headline="Your previous script ran, but a review of its outputs against "
                            "the specification found the problems below. Fix them.",
                        )
                        continue
                    logger.warning("Review found problems; running the corrected script")
                if partial_code:
                    # This request continued a cut-off script: stitch the pieces together.
                    piece = code or (
                        extract_partial_code(result.content) if result.truncated else ""
                    )
                    if code:
                        code = join_continuation(partial_code, code)
                        partial_code = ""
                    elif piece and continuations < MAX_CONTINUATIONS:
                        partial_code = join_continuation(partial_code, piece)
                        continuations += 1
                        logger.warning(
                            "Continuation was cut off again; continuing (%d lines so far)",
                            len(partial_code.splitlines()),
                        )
                        user_prompt = build_continuation_prompt(initial_prompt, partial_code)
                        continue
                    else:
                        partial_code = ""
                if not code:
                    logger.warning(
                        "No complete Python code block in response (%d characters, finish_reason=%s)",
                        len(result.content),
                        result.finish_reason,
                    )
                    partial = extract_partial_code(result.content) if result.truncated else ""
                    if partial and continuations < MAX_CONTINUATIONS:
                        # The script itself outgrew the output cap: ask for the rest of it rather
                        # than a shorter rewrite that would be cut off at the same place.
                        partial_code = partial
                        continuations += 1
                        logger.info(
                            "Script cut off after %d lines; requesting a continuation",
                            len(partial.splitlines()),
                        )
                        user_prompt = build_continuation_prompt(initial_prompt, partial_code)
                        temperature = 0.0
                        continue
                    # Re-ask from the same base prompt. A response cut off mid-reasoning is still
                    # useful analysis: hand it back so the model writes code instead of re-deriving
                    # it (re-asking from scratch just truncates again). Sample so the retry differs.
                    user_prompt = (
                        f"{user_prompt.split(NO_CODE_MARKER)[0]}{NO_CODE_MARKER}"
                        f"{build_no_code_prompt(truncated=result.truncated, notes=result.content)}"
                    )
                    temperature = 0.6
                    continue
                temperature = 0.0
                if code == last_failed_code:
                    # The model copied the failing script back verbatim (copying stays near-
                    # deterministic even when sampling): it cannot see the bug. Running it again
                    # only repeats the error, so ask for a fresh script WITHOUT showing the old one.
                    logger.warning(
                        "Model returned the previous failing script unchanged; re-asking"
                    )
                    edit_base = ""
                    user_prompt = (
                        f"{initial_prompt}{NO_CODE_MARKER}An earlier script for this task failed "
                        f"repeatedly with this error:\n```\n{last_feedback}\n```\nWrite a new "
                        "complete script from scratch that avoids the construct that caused it."
                    )
                    temperature = 0.7
                    continue

                logger.info(
                    "Extracted generated solution on attempt %d (%d characters, %d lines)",
                    attempt,
                    len(code),
                    len(code.splitlines()),
                )

                # Execute code in a scratch directory; deliverables go to out_dir.
                self.clear_stale_deliverables(expected_deliverables)
                started_at = time.time()
                exec_result = run_code(
                    code=code,
                    work_dir=work_dir,
                    target_out_dir=self.out_dir,
                    task_dir=self.task_dir,
                    # Once a clean run exists, a rewrite that runs far longer is a regression, not
                    # a fix; cap it so it cannot burn the roster's shared wall-clock allowance.
                    timeout_sec=max(
                        MIN_EXEC_TIMEOUT_SEC,
                        round(
                            min(
                                EXEC_TIMEOUT_SEC,
                                max(60.0, 10 * accepted_seconds)
                                if have_accepted
                                else EXEC_TIMEOUT_SEC,
                                self._time_left() - 10,
                            )
                        ),
                    ),
                )

                # Check output files
                deliverables_ok, missing = self.check_deliverables(
                    expected_deliverables, produced_after=started_at
                )

                if exec_result.success and deliverables_ok:
                    snapshot_outputs(self.out_dir, accepted_dir)
                    have_accepted = True
                    accepted_seconds = exec_result.duration
                    reviewed_code = code
                    last_error = ""
                    elapsed = time.monotonic() - loop_started
                    requests_left = getattr(self.client, "max_requests", 0) - getattr(
                        self.client, "request_count", 0
                    )
                    if reviews >= MAX_REVIEWS or elapsed > REVIEW_DEADLINE_SEC or requests_left < 2:
                        return self._accept(attempt, "no review budget left", reviewed_code)
                    # Exit code 0 and present files say nothing about correctness: ask the model
                    # to verify the outputs against the specification before accepting them.
                    findings = output_diagnostics(self.out_dir)
                    reviews += 1
                    reviewing = True
                    continuations = 0
                    edit_base = code
                    user_prompt = build_review_prompt(
                        edit_mode=_edit_mode(code),
                        task_prompt=initial_prompt,
                        code=code,
                        stdout=exec_result.stdout,
                        output_previews=build_input_previews(
                            self.out_dir, output_files(self.out_dir), label="OUTPUT_DIR"
                        ),
                        findings=findings,
                    )
                    logger.info(
                        "Clean run on attempt %d; requesting review %d/%d (%d automatic findings)",
                        attempt,
                        reviews,
                        MAX_REVIEWS,
                        len(findings),
                    )
                    continue

                if have_accepted:
                    # A failed rewrite deleted the accepted deliverables before running: put them
                    # back now, so an external kill (card timeout) never leaves the output empty.
                    restore_outputs(accepted_dir, self.out_dir)

                # If failed, prepare feedback for the self-repair loop
                logger.warning(
                    "Attempt %d failed (returncode=%d, deliverables_ok=%s, missing=%s). Initiating repair.",
                    attempt,
                    exec_result.returncode,
                    deliverables_ok,
                    missing,
                )

                err_msg = exec_result.feedback
                if exec_result.success and missing:
                    err_msg = (
                        f"Script exited with code 0 but required deliverables were not written to "
                        f"OUTPUT_DIR ({self.out_dir}).\n{err_msg}"
                    )

                continuations = 0
                last_failed_code = code
                last_feedback = exec_result.feedback[-2000:]
                signature = exec_result.error_signature
                repeated = bool(signature) and signature == last_error
                last_error = signature
                if repeated:
                    logger.warning("Same error as the previous attempt: %s", signature[:120])
                edit_base = code
                user_prompt = build_repair_prompt(
                    edit_mode=_edit_mode(code),
                    task_prompt=initial_prompt,
                    repeated_error=repeated,
                    previous_code=code,
                    error_message=err_msg,
                    out_dir=str(self.out_dir),
                    missing_deliverables=missing if not deliverables_ok else None,
                )
                logger.info(
                    "Prepared repair prompt for attempt %d (error_chars=%d, prompt_chars=%d)",
                    attempt + 1,
                    len(err_msg),
                    len(user_prompt),
                )
        except Exception:
            # Never let an unexpected error lose accepted outputs or skip the fallback below.
            logger.exception("Unexpected error in the solve loop")

        if have_accepted:
            # A review "fix" failed to produce a clean run: fall back to the last clean outputs.
            restore_outputs(accepted_dir, self.out_dir)
            return self._accept(self.max_retries + 1, "restored the last clean run", reviewed_code)
        logger.error("Agent run stopped without producing all verified deliverables.")
        logger.info(
            "Agent run summary: status=failed, model_requests=%s, output_dir=%s",
            getattr(self.client, "request_count", "unknown"),
            self.out_dir,
        )
        return False

    def _accept(self, attempt: int, reason: str, code: str) -> bool:
        # With transcripts on (local batch runs), keep the accepted script: checker-passed ones
        # become few-shot examples for OTHER units (agent.main build-examples).
        transcript_dir = os.environ.get("AGENT_TRANSCRIPT_DIR")
        if transcript_dir:
            try:
                path = pathlib.Path(transcript_dir)
                path.mkdir(parents=True, exist_ok=True)
                (path / "solution.py").write_text(code, encoding="utf-8")
            except OSError:
                logger.debug("Could not save the accepted script", exc_info=True)
        logger.info(
            "Task completed on attempt %d (%s; verified_outputs=%s, model_requests=%s, "
            "output_dir=%s)",
            attempt,
            reason,
            [p.name for p in output_files(self.out_dir)],
            getattr(self.client, "request_count", "unknown"),
            self.out_dir,
        )
        return True
